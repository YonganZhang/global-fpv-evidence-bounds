"""Rejoin raw historical LM scores without turning missing evidence into bad scores.

Outputs are diagnostic inputs, NOT verified policy parameters or future forecasts.
No historical mixed climate score is used. Current country ownership takes
precedence over historical waterbody IDs; stale province labels are rejected.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALIASES = {
    "México": "Mexico", "Republic of the Congo": "Republic of Congo",
    "eSwatini": "Eswatini", "Hong Kong S.A.R.": "Hong Kong",
    "United States": "United States of America", "USA": "United States of America",
    "Russian Federation": "Russia", "Türkiye": "Turkey", "Czechia": "Czech Republic",
    "Viet Nam": "Vietnam", "Republic of Serbia": "Serbia",
    "Macedonia": "North Macedonia", "Guinea Bissau": "Guinea-Bissau",
    "Côte d'Ivoire": "Ivory Coast", "DR Congo": "Democratic Republic of the Congo",
    "Dem. Rep. Congo": "Democratic Republic of the Congo", "Congo": "Republic of Congo",
    "Republic of Korea": "South Korea", "Korea, Republic of": "South Korea",
    "United Republic of Tanzania": "Tanzania", "Swaziland": "Eswatini",
    "Czech Rep.": "Czech Republic", "Dominican Rep.": "Dominican Republic",
    "Central African Rep.": "Central African Republic", "S. Sudan": "South Sudan",
    "Bosnia and Herz.": "Bosnia and Herzegovina", "Eq. Guinea": "Equatorial Guinea",
}


def country_key(value: str | None) -> str:
    value = " ".join(str(value or "").split())
    return ALIASES.get(value, value)


def entity_key(value: str, province: bool = False) -> str:
    if province:
        parts = str(value).split("|", 1)
        return country_key(parts[0]) + "|" + (parts[1].strip() if len(parts) == 2 else "")
    return country_key(value)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_lookup(rows: list[dict], score_column: str, province: bool = False,
                 minimum: float = 0., maximum: float = 1.) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[entity_key(row.get("entity_key", ""), province)].append(row)
    result = {}
    for key, group in grouped.items():
        values, invalid, urls = [], 0, set()
        for row in group:
            try:
                value = float(row.get(score_column, ""))
                if not math.isfinite(value) or not minimum <= value <= maximum:
                    raise ValueError("invalid range")
                values.append(value)
            except (TypeError, ValueError):
                invalid += 1
            try:
                evidence = json.loads(row.get("evidence_urls") or "[]")
            except (TypeError, json.JSONDecodeError):
                evidence = []
            if isinstance(evidence, list):
                urls.update(u for u in evidence if isinstance(u, str) and u.startswith(("https://", "http://")))
        conflict = len(set(values)) > 1
        state = "conflicting_scores" if conflict else "invalid_score" if invalid else "available_unverified"
        result[key] = dict(score=values[0] if values and not conflict and not invalid else None,
                           state=state, raw_rows=len(group), invalid_rows=invalid,
                           minimum=min(values) if values else None,
                           maximum=max(values) if values else None,
                           source_url_count=len(urls), source_verified=False)
    return result


def missing(state: str = "missing") -> dict:
    return dict(score=None, state=state, raw_rows=0, invalid_rows=0, minimum=None,
                maximum=None, source_url_count=0, source_verified=False)


def diagnostic_multiplier(score: float | None, alpha: float = .5) -> float:
    """Only for labelled historical-score sensitivity; unknown means no adjustment."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be within [0, 1]")
    if score is None:
        return 1.0
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("score must be finite and within [0, 1]")
    return 1 - alpha * (1 - score)


def align_one(waterbody: dict, old_tag: dict | None, country_lookup: dict,
              province_lookup: dict) -> dict:
    current = country_key(waterbody.get("country_v117"))
    old = country_key((old_tag or {}).get("country"))
    c = country_lookup.get(current, missing())
    province = (old_tag or {}).get("province", "").strip()
    pkey = current + "|" + province
    if not old_tag or not province or province.lower() in {"nan", "none"}:
        p = missing("missing_province")
    elif current != old:
        p = missing("stale_province_country")
    else:
        p = province_lookup.get(pkey, missing())
    return {
        "wb_id": waterbody["wb_id"], "country_iso3": waterbody.get("country_iso3_v117"),
        "current_country_key": current, "historical_country_key": old,
        "country_changed": bool(old and old != current),
        "province_key": pkey if old == current and province else None,
        **{"country_" + k: v for k, v in c.items()},
        **{"province_" + k: v for k, v in p.items()},
        "country_diagnostic_multiplier": diagnostic_multiplier(c["score"]),
        # No provenance-verified country/province scores have been approved here.
        "formal_lm_multiplier": 1.0,
        "mixed_climate_score_used": False,
        "future_policy_prediction": False,
    }


def main() -> None:
    import pyarrow.parquet as pq

    score_dir = ROOT / "Dataset/data_processed/llm_scores"
    country_path = score_dir / "dim_1a_country_policy.csv"
    province_path = score_dir / "dim_1b_province_modifier.csv"
    tag_path = ROOT / "Dataset/data_processed/merged/waterbody_tags_enriched.csv"
    inventory_path = ROOT / "_outputs/v117/data/fpv_reference_inventory_v117.parquet"
    countries = build_lookup(read_csv(country_path), "s_policy_overall")
    provinces = build_lookup(read_csv(province_path), "s_province_modifier", True, .5, 1.5)
    tags = {}
    for tag in read_csv(tag_path):
        key = int(tag["wb_id"])
        if key in tags:
            raise ValueError(f"duplicate tag wb_id: {key}")
        tags[key] = tag
    inventory = pq.read_table(inventory_path, columns=["wb_id", "country_v117", "country_iso3_v117"]).to_pylist()
    assert len({r["wb_id"] for r in inventory}) == len(inventory) == 199976
    output = ROOT / "_outputs/v122"
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "reports").mkdir(parents=True, exist_ok=True)
    aligned = [align_one(row, tags.get(row["wb_id"]), countries, provinces) for row in inventory]
    path = output / "data/llm_country_province_alignment.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aligned[0]))
        writer.writeheader()
        writer.writerows(aligned)
    report = {
        "status": "diagnostic_alignment_completed_not_future_results",
        "waterbodies": len(aligned), "country_entities": len(countries),
        "country_entity_states": dict(Counter(x["state"] for x in countries.values())),
        "waterbody_country_states": dict(Counter(x["country_state"] for x in aligned)),
        "waterbody_province_states": dict(Counter(x["province_state"] for x in aligned)),
        "country_changed_after_aliases": sum(x["country_changed"] for x in aligned),
        "formal_lm_adjustments": sum(x["formal_lm_multiplier"] != 1 for x in aligned),
        "unknown_country_scores_discounted": sum(x["country_score"] is None and x["country_diagnostic_multiplier"] != 1 for x in aligned),
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (country_path, province_path, tag_path)},
        "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "limitations": ["Raw LM scores are not provenance verified or future policy predictions.",
                        "Conflicting scores are unresolved, not averaged or silently selected.",
                        "Climate/ecology/named-waterbody aggregate scores are not consumed by this module.",
                        "Province tags are historical; matching country does not prove current province accuracy."],
    }
    (output / "reports/llm_alignment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
