#!/usr/bin/env python3
"""Validation gate for the v113 historical dry-up evidence rebuild."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_ROWS = 198_737
EXPECTED_KNOWN = 190_586
EXPECTED_UNKNOWN = 8_151
EXPECTED_DRY = 14_639
EXPECTED_KEY_SOURCES = {
    "hydrolakes_direct": 177_795,
    "grand_id_authoritative": 6_796,
    "spatial_nearest_high_confidence": 5_996,
    "unmatched": 8_150,
}


def validate(run_dir: Path) -> None:
    errors: list[str] = []
    data_path = run_dir / "data/paper_scope_historical_dryup_generation.parquet"
    flags_path = run_dir / "data/waterbody_historical_dryup_flags.parquet"
    summary_path = run_dir / "reports/paper_scope_level_summary.csv"
    qc_path = run_dir / "reports/historical_dryup_qc.json"
    cache_path = run_dir / "raw/glev/glev_scope_1991_2018.parquet"

    for path in [data_path, flags_path, summary_path, qc_path, cache_path]:
        if not path.exists():
            errors.append(f"missing required artifact: {path}")
    if errors:
        finish(run_dir, errors)
        return

    columns = [
        "wb_id",
        "wb_type",
        "dryup_hylak_id_v113",
        "dryup_key_source_v113",
        "glev_valid_months_1991_2018",
        "glev_zero_months_1991_2018",
        "glev_negative_months_1991_2018",
        "historical_dryup_evidence_known_v113",
        "historical_dryup_observed_1991_2018_v113",
        "historical_dryup_pass_1991_2018_v113",
        "l0_type_specific_generation_gwh",
        "l1_type_generation_gwh_v112",
        "l2_core_no_dry_pass_v112",
        "l2_conservative_no_dry_pass_v112",
    ]
    frame = pd.read_parquet(data_path, columns=columns)
    if len(frame) != EXPECTED_ROWS or frame["wb_id"].nunique() != EXPECTED_ROWS:
        errors.append("declared-scope row/key coverage mismatch")
    sources = frame["dryup_key_source_v113"].value_counts().to_dict()
    if sources != EXPECTED_KEY_SOURCES:
        errors.append(f"dry-up key-source counts changed: {sources}")
    known = frame["historical_dryup_evidence_known_v113"].fillna(False)
    observed = frame["historical_dryup_observed_1991_2018_v113"].fillna(False)
    dry_pass = frame["historical_dryup_pass_1991_2018_v113"]
    if int(known.sum()) != EXPECTED_KNOWN:
        errors.append(f"known dry-up coverage changed: {int(known.sum())}")
    if int(dry_pass.isna().sum()) != EXPECTED_UNKNOWN:
        errors.append(f"unknown dry-up coverage changed: {int(dry_pass.isna().sum())}")
    if int(observed.sum()) != EXPECTED_DRY:
        errors.append(f"observed dry-up count changed: {int(observed.sum())}")
    if dry_pass.loc[known].isna().any() or dry_pass.loc[~known].notna().any():
        errors.append("known/unknown nullable dry-up pass semantics are inconsistent")
    if not frame.loc[known, "glev_valid_months_1991_2018"].eq(336).all():
        errors.append("known rows do not all contain 336 valid months")
    if not frame.loc[observed, "glev_zero_months_1991_2018"].gt(0).all():
        errors.append("observed dry-up rows lack a zero-area month")
    if frame["glev_negative_months_1991_2018"].fillna(0).sum() != 0:
        errors.append("negative GLEV surface-area values found")
    duplicate = frame.loc[
        frame["dryup_hylak_id_v113"].notna()
        & frame["dryup_hylak_id_v113"].duplicated(keep=False)
    ]
    if len(duplicate) != 12 or duplicate["dryup_hylak_id_v113"].nunique() != 6:
        errors.append("known source-scope duplicate groups changed")
    if not duplicate.loc[duplicate["wb_type"].eq("controlled_lake")].shape[0] == 6:
        errors.append("source-scope duplicate diagnosis no longer isolates six controlled lakes")

    cache = pd.read_parquet(cache_path, columns=["Hylak_id"])
    if len(cache) != 190_580 or not cache["Hylak_id"].is_unique:
        errors.append("cached GLEV subset coverage mismatch")

    summary = pd.read_csv(summary_path).set_index("tier")
    required = [
        "L0-recommended-type",
        "L1-recommended-type-hybrid-ice",
        "L2-core-historical-known-pass-lower-bound-type",
        "L2-core-historical-unknown-pass-upper-envelope-type",
        "L2-conservative-historical-known-pass-lower-bound-type",
        "L2-conservative-historical-unknown-pass-upper-envelope-type",
        "L2-complete-paper-equivalent",
        "L3",
    ]
    if not set(required).issubset(summary.index):
        errors.append("required v113 summary tiers missing")
    else:
        values = summary["generation_twh"]
        pairs = [
            (
                "L2-core-historical-known-pass-lower-bound-type",
                "L2-core-historical-unknown-pass-upper-envelope-type",
            ),
            (
                "L2-conservative-historical-known-pass-lower-bound-type",
                "L2-conservative-historical-unknown-pass-upper-envelope-type",
            ),
        ]
        for lower, upper in pairs:
            if values[lower] > values[upper]:
                errors.append(f"historical evidence interval inverted: {lower} > {upper}")
        if values["L2-core-historical-unknown-pass-upper-envelope-type"] > values[
            "L1-recommended-type-hybrid-ice"
        ]:
            errors.append("L2 core upper envelope exceeds L1")
        if values["L2-conservative-historical-unknown-pass-upper-envelope-type"] > values[
            "L2-core-historical-unknown-pass-upper-envelope-type"
        ]:
            errors.append("conservative L2 upper envelope exceeds core L2")
        expected_retention = {
            "L2-core-historical-known-pass-lower-bound-type": 24.893198,
            "L2-core-historical-unknown-pass-upper-envelope-type": 28.597635,
            "L2-conservative-historical-known-pass-lower-bound-type": 17.297641,
            "L2-conservative-historical-unknown-pass-upper-envelope-type": 20.252386,
        }
        for tier, expected in expected_retention.items():
            actual = float(summary.at[tier, "retention_vs_recommended_l0_pct"])
            if not np.isclose(actual, expected, atol=1e-5):
                errors.append(f"unexpected retention for {tier}: {actual}")
        for tier in ["L2-complete-paper-equivalent", "L3"]:
            if bool(summary.at[tier, "scientific_tier_complete"]):
                errors.append(f"{tier} incorrectly marked complete")
            if pd.notna(summary.at[tier, "generation_twh"]):
                errors.append(f"{tier} must remain NA")

    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    if qc.get("known_complete_rows") != EXPECTED_KNOWN:
        errors.append("QC known coverage mismatch")
    crosswalk = qc.get("reservoir_crosswalk", {})
    if crosswalk.get("grand_id_authoritative_matches") != 6_796:
        errors.append("QC authoritative crosswalk count mismatch")
    if crosswalk.get("spatial_high_confidence_matches") != 5_996:
        errors.append("QC spatial crosswalk count mismatch")
    if float(crosswalk.get("spatial_validation_precision", 0)) < 0.95:
        errors.append("spatial crosswalk validation precision below 95%")
    area_validation = qc.get("area_cross_validation", {})
    if float(area_validation.get("pearson_log10_area", 0)) < 0.95:
        errors.append("GLEV/Woolway area log-correlation below 0.95")
    if not 0.95 <= float(area_validation.get("ratio_median", 0)) <= 1.05:
        errors.append("GLEV/Woolway median area ratio outside 0.95--1.05")
    if qc.get("l2_complete") is not False or qc.get("l3_complete") is not False:
        errors.append("incomplete tiers are incorrectly marked complete in QC")

    finish(run_dir, errors)


def finish(run_dir: Path, errors: list[str]) -> None:
    result = {"status": "pass" if not errors else "fail", "errors": errors}
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "validation_gate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if errors:
        raise SystemExit("validation failed: " + "; ".join(errors))
    print("v113 historical dry-up evidence validation: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("_outputs/v113"))
    args = parser.parse_args()
    validate(args.run_dir)


if __name__ == "__main__":
    main()
