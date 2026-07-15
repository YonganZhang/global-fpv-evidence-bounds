#!/usr/bin/env python3
"""Consolidated anti-fake-completion gate for the v113--v116 FPV rebuild."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ARCHIVES = {
    Path("_outputs/v113/raw/glev/1_surfacewater_area.zip"): (
        1_179_230_974,
        "08e965c3dddeaf2375de54cc83b9e338c589d9393c2dd485b9a25a608e3ed559",
    ),
    Path("_outputs/v115/raw/roads/GRIP4_density_total.zip"): (
        3_641_668,
        "782fe4b609befbb294528d2b7abce212df4d8f883c07775134fe9a6372dae7d9",
    ),
    Path("_outputs/v115/raw/transmission/osm_power_tmm.zip"): (
        104_810_570,
        "50ce1f4879d45b3a1709370ba2386b5dcbb9c443780658797ecfb80a77d5a537",
    ),
    Path("_outputs/v116/raw/globathy/GLOBathy_basic_parameters.zip"): (
        115_568_963,
        "52056668b1a3060fd860d7780be1cbe4ff7f21d7057fcee5d942858a243ed61b",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_close(
    errors: list[str], actual: float, expected: float, label: str, atol: float = 1e-6
) -> None:
    if not np.isclose(actual, expected, atol=atol):
        errors.append(f"{label} changed: {actual} != {expected}")


def validate(root: Path, output_path: Path) -> None:
    errors: list[str] = []

    for version in ("v113", "v114", "v115", "v116"):
        gate = root / f"_outputs/{version}/reports/validation_gate.json"
        if not gate.exists():
            errors.append(f"missing stage gate: {gate.relative_to(root)}")
            continue
        result = json.loads(gate.read_text(encoding="utf-8"))
        if result.get("status") != "pass" or result.get("errors") != []:
            errors.append(f"stage gate is not clean: {version}")

        meta = root / f"_outputs/{version}/_run_meta.yml"
        if not meta.exists():
            errors.append(f"missing run metadata: {meta.relative_to(root)}")
        else:
            text = meta.read_text(encoding="utf-8")
            required_markers = [
                f"run_id: {version}",
                "pipeline: _pipelines/fpv-rebuild-v116.yml",
                "outcome: partial",
                "paper_used: false",
            ]
            for marker in required_markers:
                if marker not in text:
                    errors.append(f"{version} metadata lacks marker: {marker}")

    for relative, (expected_size, expected_digest) in EXPECTED_ARCHIVES.items():
        path = root / relative
        if not path.exists():
            errors.append(f"missing source archive: {relative}")
            continue
        if path.stat().st_size != expected_size:
            errors.append(f"source archive size changed: {relative}")
        elif sha256(path) != expected_digest:
            errors.append(f"source archive checksum changed: {relative}")

    l2_path = root / "_outputs/v114/reports/paper_scope_level_summary.csv"
    l3_path = root / "_outputs/v115/reports/l3_accessibility_summary.csv"
    if not l2_path.exists() or not l3_path.exists():
        errors.append("missing L2 or L3 summary")
    else:
        l2 = pd.read_csv(l2_path).set_index("tier")
        l3 = pd.read_csv(l3_path).set_index("tier")
        expected_l2 = {
            "L0-recommended-type-deduplicated": 16_579.544592689523,
            "L1-recommended-type-hybrid-ice-deduplicated": 10_870.93473613666,
            "L2-core-historical-known-pass-lower-bound-deduplicated": 4_128.897245065607,
            "L2-core-historical-unknown-pass-upper-envelope-deduplicated": 4_743.373752951582,
            "L2-conservative-historical-known-pass-lower-bound-deduplicated": 2_869.260325038811,
            "L2-conservative-historical-unknown-pass-upper-envelope-deduplicated": 3_359.3810500711556,
        }
        expected_l3 = {
            "L3-access-core-evidence-lower-bound": 1_569.653772223607,
            "L3-access-core-mapping-upper-envelope": 3_458.3529273272243,
            "L3-access-conservative-evidence-lower-bound": 1_083.105567097121,
            "L3-access-conservative-mapping-upper-envelope": 2_457.246386620049,
        }
        for tier, expected in expected_l2.items():
            if tier not in l2.index:
                errors.append(f"missing consolidated L2 tier: {tier}")
            else:
                require_close(errors, float(l2.at[tier, "generation_twh"]), expected, tier)
        for tier, expected in expected_l3.items():
            if tier not in l3.index:
                errors.append(f"missing consolidated L3 tier: {tier}")
            else:
                require_close(errors, float(l3.at[tier, "generation_twh"]), expected, tier)

        if not errors:
            l0 = expected_l2["L0-recommended-type-deduplicated"]
            l1 = expected_l2["L1-recommended-type-hybrid-ice-deduplicated"]
            core_l2_lower = expected_l2[
                "L2-core-historical-known-pass-lower-bound-deduplicated"
            ]
            core_l2_upper = expected_l2[
                "L2-core-historical-unknown-pass-upper-envelope-deduplicated"
            ]
            conservative_l2_lower = expected_l2[
                "L2-conservative-historical-known-pass-lower-bound-deduplicated"
            ]
            conservative_l2_upper = expected_l2[
                "L2-conservative-historical-unknown-pass-upper-envelope-deduplicated"
            ]
            core_l3_lower = expected_l3["L3-access-core-evidence-lower-bound"]
            core_l3_upper = expected_l3["L3-access-core-mapping-upper-envelope"]
            conservative_l3_lower = expected_l3[
                "L3-access-conservative-evidence-lower-bound"
            ]
            conservative_l3_upper = expected_l3[
                "L3-access-conservative-mapping-upper-envelope"
            ]
            if not (
                l0
                > l1
                > core_l2_upper
                >= core_l2_lower
                > core_l3_lower
                and core_l2_upper > core_l3_upper >= core_l3_lower
                and conservative_l2_upper >= conservative_l2_lower
                > conservative_l3_lower
                and conservative_l2_upper > conservative_l3_upper
                >= conservative_l3_lower
            ):
                errors.append("cross-stage L0/L1/L2/L3 monotonicity is broken")

        for tier in ("L2-complete-paper-equivalent", "L3"):
            if tier not in l2.index:
                errors.append(f"missing incomplete-tier marker: {tier}")
            elif bool(l2.at[tier, "scientific_tier_complete"]) or pd.notna(
                l2.at[tier, "generation_twh"]
            ):
                errors.append(f"{tier} must remain incomplete and NA")
        complete_l3 = "L3-complete-deployable"
        if complete_l3 not in l3.index:
            errors.append("missing complete-L3 marker")
        elif bool(l3.at[complete_l3, "scientific_tier_complete"]) or pd.notna(
            l3.at[complete_l3, "generation_twh"]
        ):
            errors.append("complete deployable L3 must remain incomplete and NA")

    issue_path = root / "_outputs/v116/reports/issue_status_20260715.csv"
    if not issue_path.exists():
        errors.append("missing v116 issue ledger")
    else:
        issues = pd.read_csv(issue_path)
        counts = issues["status"].value_counts().to_dict()
        expected_counts = {
            "resolved_or_replaced": 12,
            "partial": 17,
            "isolated_not_fixed": 6,
            "open": 13,
        }
        if len(issues) != 48 or counts != expected_counts:
            errors.append(f"issue ledger status counts changed: rows={len(issues)}, {counts}")

    inventory_path = root / "_outputs/v116/reports/data_candidate_inventory_20260715.csv"
    search_path = root / "_outputs/v116/reports/data_search_log_20260715.csv"
    if not inventory_path.exists() or not search_path.exists():
        errors.append("missing data-discovery inventory or search log")
    else:
        inventory = pd.read_csv(inventory_path)
        search = pd.read_csv(search_path)
        if len(inventory) != 52 or inventory["candidate_id"].nunique() != 52:
            errors.append("data candidate inventory must contain 52 unique candidates")
        if len(search) != 20 or search["search_id"].nunique() != 20:
            errors.append("data search log must contain 20 unique searches")

    report_path = root / "_outputs/v116/reports/fpv_strict_rebuild_report_20260715.md"
    if not report_path.exists():
        errors.append("missing v116 decision report")
    else:
        report = report_path.read_text(encoding="utf-8")
        for marker in (
            "16,579.545",
            "4,128.897",
            "4,743.374",
            "1,569.654",
            "3,458.353",
            "完整论文等价 L2",
            "**NA**",
            "36 / 48",
        ):
            if marker not in report:
                errors.append(f"decision report lacks marker: {marker}")

    result = {
        "status": "pass" if not errors else "fail",
        "scope": "v113-v116 consolidated partial rebuild",
        "paper_used": False,
        "complete_l2": False,
        "complete_l3": False,
        "errors": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if errors:
        raise SystemExit("consolidated validation failed: " + "; ".join(errors))
    print("v113-v116 consolidated anti-fake-completion validation: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("_outputs/v116/reports/consolidated_validation_gate.json"),
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.root / args.output
    validate(args.root.resolve(), output)


if __name__ == "__main__":
    main()
