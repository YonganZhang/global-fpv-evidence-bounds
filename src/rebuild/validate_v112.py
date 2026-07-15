#!/usr/bin/env python3
"""Validation gate for the declared-scope v112 hybrid-ice rebuild."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_WATERBODIES = 198_737
EXPECTED_ICE_SOURCE_COUNTS = {
    "li_ccr_modis_phenology_2002_2020": 31_203,
    "woolway_air_temperature_lag_model_1991_2020": 111_614,
    "nasa_power_monthly_temp_proxy_2019_2023": 55_920,
}


def validate(run_dir: Path) -> None:
    errors: list[str] = []
    data_path = run_dir / "data/paper_scope_hybrid_generation.parquet"
    summary_path = run_dir / "reports/paper_scope_level_summary.csv"
    qc_path = run_dir / "reports/paper_scope_hybrid_qc.json"

    if not data_path.exists():
        errors.append("missing paper-scope hybrid table")
    else:
        columns = [
            "wb_id",
            "ice_source_v112",
            "ice_pass_v112",
            "l1_area_pass_v112",
            "l1_pass_v112",
            "l0_generation_gwh",
            "l0_type_specific_generation_gwh",
            "l1_uniform_generation_gwh_v112",
            "l1_type_generation_gwh_v112",
            "l2_core_type_generation_gwh_v112",
            "l2_conservative_type_generation_gwh_v112",
            "l2_author_visible_type_generation_gwh_v112",
        ]
        frame = pd.read_parquet(data_path, columns=columns)
        if len(frame) != EXPECTED_WATERBODIES or frame["wb_id"].nunique() != EXPECTED_WATERBODIES:
            errors.append("paper-scope row/key coverage mismatch")
        if frame[columns].isna().any().any():
            errors.append("paper-scope required columns contain missing values")
        counts = frame["ice_source_v112"].value_counts().to_dict()
        if counts != EXPECTED_ICE_SOURCE_COUNTS:
            errors.append(f"ice source hierarchy mismatch: {counts}")
        if not frame["l1_area_pass_v112"].all():
            errors.append("declared 0.01 km2 L1 area gate unexpectedly excludes rows")
        pairs = [
            ("l0_generation_gwh", "l1_uniform_generation_gwh_v112"),
            ("l0_type_specific_generation_gwh", "l1_type_generation_gwh_v112"),
            ("l1_type_generation_gwh_v112", "l2_core_type_generation_gwh_v112"),
            ("l2_core_type_generation_gwh_v112", "l2_conservative_type_generation_gwh_v112"),
        ]
        for upper, lower in pairs:
            if (frame[lower] > frame[upper] + 1e-5).any():
                errors.append(f"non-monotonic rows: {lower} > {upper}")

    if not summary_path.exists():
        errors.append("missing paper-scope level summary")
    else:
        summary = pd.read_csv(summary_path).set_index("tier")
        required = {
            "L0-paper-uniform",
            "L0-recommended-type",
            "L1-paper-uniform-hybrid-ice",
            "L1-recommended-type-hybrid-ice",
            "L2-core-screened-no-dry-type",
            "L2-conservative-screened-no-dry-type",
            "L2-author-visible-10km-no-dry-type",
            "L2-complete",
            "L3",
        }
        if not required.issubset(summary.index):
            errors.append("paper-scope summary tiers missing")
        else:
            order = [
                "L0-recommended-type",
                "L1-recommended-type-hybrid-ice",
                "L2-core-screened-no-dry-type",
                "L2-conservative-screened-no-dry-type",
            ]
            if not summary.loc[order, "generation_twh"].is_monotonic_decreasing:
                errors.append("recommended tier totals are non-monotonic")
            for tier in [
                "L2-core-screened-no-dry-type",
                "L2-conservative-screened-no-dry-type",
                "L2-author-visible-10km-no-dry-type",
                "L2-complete",
                "L3",
            ]:
                if bool(summary.at[tier, "scientific_tier_complete"]):
                    errors.append(f"{tier} is incorrectly marked complete")
            for tier in ["L2-complete", "L3"]:
                if pd.notna(summary.at[tier, "generation_twh"]):
                    errors.append(f"{tier} must remain NA")

    if not qc_path.exists():
        errors.append("missing paper-scope hybrid QC")
    else:
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if qc.get("waterbodies") != EXPECTED_WATERBODIES:
            errors.append("paper-scope QC count mismatch")
        if qc.get("area_failures") != 0:
            errors.append("paper-scope QC area failures")
        if qc.get("dry_up_gate_available") is not False:
            errors.append("dry-up gate must remain explicitly unavailable")
        if qc.get("l2_complete") is not False or qc.get("l3_complete") is not False:
            errors.append("L2/L3 completeness flags must remain false")

    result = {"status": "pass" if not errors else "fail", "errors": errors}
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "validation_gate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if errors:
        raise SystemExit("validation failed: " + "; ".join(errors))
    print("v112 declared-scope hybrid validation: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("_outputs/v112"))
    args = parser.parse_args()
    validate(args.run_dir)


if __name__ == "__main__":
    main()
