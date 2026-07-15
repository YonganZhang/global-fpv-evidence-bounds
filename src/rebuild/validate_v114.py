#!/usr/bin/env python3
"""Validation gate for the v114 authoritative scope deduplication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_ROWS = 198_731


def validate(run_dir: Path) -> None:
    errors: list[str] = []
    data_path = run_dir / "data/paper_scope_deduplicated_historical_generation.parquet"
    removal_path = run_dir / "reports/authoritative_scope_duplicate_removals.csv"
    summary_path = run_dir / "reports/paper_scope_level_summary.csv"
    qc_path = run_dir / "reports/scope_dedup_qc.json"
    for path in [data_path, removal_path, summary_path, qc_path]:
        if not path.exists():
            errors.append(f"missing artifact: {path}")
    if not errors:
        frame = pd.read_parquet(
            data_path,
            columns=[
                "wb_id",
                "wb_type",
                "dryup_hylak_id_v113",
                "scope_deduplicated_v114",
                "historical_dryup_evidence_known_v113",
                "historical_dryup_pass_1991_2018_v113",
            ],
        )
        if len(frame) != EXPECTED_ROWS or frame["wb_id"].nunique() != EXPECTED_ROWS:
            errors.append("deduplicated row/key coverage mismatch")
        if not frame["scope_deduplicated_v114"].all():
            errors.append("scope deduplication marker is incomplete")
        if frame.loc[
            frame["dryup_hylak_id_v113"].notna(), "dryup_hylak_id_v113"
        ].duplicated().any():
            errors.append("duplicate dry-up HydroLAKES keys remain")
        removals = pd.read_csv(removal_path)
        if len(removals) != 6 or not removals["wb_type"].eq("controlled_lake").all():
            errors.append("authoritative duplicate-removal set changed")
        if set(removals["dryup_hylak_id_v113"]) != {4, 1079, 1104, 8278, 14587, 158874}:
            errors.append("authoritative duplicate HydroLAKES IDs changed")
        if set(removals["wb_id"]) & set(frame["wb_id"]):
            errors.append("removed duplicate rows remain in output")
        known = frame["historical_dryup_evidence_known_v113"].fillna(False)
        if int(known.sum()) != 190_580:
            errors.append("deduplicated known dry-up evidence count mismatch")
        if int(frame["historical_dryup_pass_1991_2018_v113"].isna().sum()) != 8_151:
            errors.append("deduplicated unknown dry-up evidence count mismatch")

        summary = pd.read_csv(summary_path).set_index("tier")
        expected = {
            "L0-recommended-type-deduplicated": 100.0,
            "L1-recommended-type-hybrid-ice-deduplicated": 65.568355,
            "L2-core-historical-known-pass-lower-bound-deduplicated": 24.903562,
            "L2-core-historical-unknown-pass-upper-envelope-deduplicated": 28.609795,
            "L2-conservative-historical-known-pass-lower-bound-deduplicated": 17.306026,
            "L2-conservative-historical-unknown-pass-upper-envelope-deduplicated": 20.262203,
        }
        if not set(expected).issubset(summary.index):
            errors.append("deduplicated summary tiers missing")
        else:
            for tier, value in expected.items():
                actual = float(summary.at[tier, "retention_vs_recommended_l0_pct"])
                if not np.isclose(actual, value, atol=1e-5):
                    errors.append(f"unexpected retention for {tier}: {actual}")
            if summary.at[
                "L2-core-historical-known-pass-lower-bound-deduplicated",
                "generation_twh",
            ] > summary.at[
                "L2-core-historical-unknown-pass-upper-envelope-deduplicated",
                "generation_twh",
            ]:
                errors.append("core L2 interval is inverted")
            if summary.at[
                "L2-conservative-historical-unknown-pass-upper-envelope-deduplicated",
                "generation_twh",
            ] > summary.at[
                "L2-core-historical-unknown-pass-upper-envelope-deduplicated",
                "generation_twh",
            ]:
                errors.append("conservative L2 exceeds core L2")
            for tier in ["L2-complete-paper-equivalent", "L3"]:
                if bool(summary.at[tier, "scientific_tier_complete"]):
                    errors.append(f"{tier} incorrectly marked complete")
                if pd.notna(summary.at[tier, "generation_twh"]):
                    errors.append(f"{tier} must remain NA")

        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if qc.get("removed_rows") != 6 or qc.get("output_rows") != EXPECTED_ROWS:
            errors.append("scope dedup QC counts mismatch")
        if qc.get("remaining_duplicate_dryup_hylak_groups") != 0:
            errors.append("scope dedup QC still reports duplicates")
        if qc.get("l2_complete") is not False or qc.get("l3_complete") is not False:
            errors.append("incomplete tiers incorrectly marked complete")

    result = {"status": "pass" if not errors else "fail", "errors": errors}
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "validation_gate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if errors:
        raise SystemExit("validation failed: " + "; ".join(errors))
    print("v114 authoritative scope deduplication validation: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("_outputs/v114"))
    args = parser.parse_args()
    validate(args.run_dir)


if __name__ == "__main__":
    main()
