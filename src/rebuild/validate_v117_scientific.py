#!/usr/bin/env python3
"""Independent integrity checks for the accepted v117 scientific products."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "_outputs/v117/data/fpv_reference_inventory_v117.parquet"
REPORTS = ROOT / "_outputs/v117/reports"


def close(value: float, expected: float, tolerance: float = 1e-6) -> bool:
    return abs(float(value) - expected) <= tolerance


def main() -> None:
    frame = pd.read_parquet(DATA)
    levels = pd.read_csv(REPORTS / "level_summary_v117.csv").set_index("tier")
    adjudication = pd.read_csv(REPORTS / "inventory_adjudication.csv")
    consolidated = json.loads(
        (REPORTS / "consolidated_validation_gate_v117.json").read_text(encoding="utf-8")
    )

    expected_levels = {
        "L0-reference": 16712.988951423074,
        "L1-reference": 10994.671646587763,
        "L2-core-public-evidence-lower": 4190.139560430422,
        "L2-core-public-evidence-upper": 4804.616068316397,
        "L2-conservative-public-evidence-lower": 2914.319715492447,
        "L2-conservative-public-evidence-upper": 3404.4404405247924,
        "L3-core-partial-access-lower": 1594.5943379021421,
        "L3-core-partial-access-upper": 3508.2823401709124,
        "L3-conservative-partial-access-lower": 1100.5209820384507,
        "L3-conservative-partial-access-upper": 2493.593205004449,
    }
    action_counts = adjudication["adjudication_v117"].value_counts().to_dict()
    checks = {
        "inventory_rows_199976": len(frame) == 199_976,
        "wb_id_unique": frame["wb_id"].is_unique,
        "country_complete": frame["country_v117"].notna().all(),
        "iso3_complete": frame["country_iso3_v117"].notna().all(),
        "continent_complete": frame["continent_v117"].notna().all(),
        "identity_actions_match_audit": action_counts == {
            "restore_no_duplicate_evidence": 1_245,
            "remove_spatial_high_confidence_reference": 645,
            "remove_authoritative_duplicate": 108,
        },
        "level_values_match_trace": all(
            tier in levels.index
            and close(levels.loc[tier, "generation_twh"], expected)
            for tier, expected in expected_levels.items()
        ),
        "level_chain_monotonic": (
            levels.loc["L1-reference", "generation_twh"]
            <= levels.loc["L0-reference", "generation_twh"]
            and levels.loc["L2-core-public-evidence-upper", "generation_twh"]
            <= levels.loc["L1-reference", "generation_twh"]
            and levels.loc["L3-core-partial-access-upper", "generation_twh"]
            <= levels.loc["L2-core-public-evidence-upper", "generation_twh"]
        ),
        "complete_l2_and_l3_are_na": (
            pd.isna(
                levels.loc[
                    "L2-complete-published-1991-2020-criterion", "generation_twh"
                ]
            )
            and pd.isna(levels.loc["L3-complete-deployable", "generation_twh"])
        ),
        "consolidated_gate_passes": (
            consolidated.get("status") == "pass"
            and consolidated.get("passed") == 13
            and consolidated.get("total") == 13
            and consolidated.get("complete_l2") is False
            and consolidated.get("complete_l3") is False
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "passed": sum(checks.values()),
        "total": len(checks),
        "checks": checks,
    }
    (REPORTS / "independent_validation_gate_v117.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
