#!/usr/bin/env python3
"""Remove physical waterbody duplicates proven by authoritative GRanD links."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_INPUT_ROWS = 198_737
EXPECTED_REMOVALS = 6


def add_row(
    rows: list[dict[str, object]],
    tier: str,
    definition: str,
    mask: np.ndarray | None,
    values: np.ndarray | None,
    recommended_l0_twh: float,
    complete: bool,
) -> None:
    generation_twh = float(values[mask].sum() / 1000.0) if mask is not None else np.nan
    rows.append(
        {
            "tier": tier,
            "definition": definition,
            "scientific_tier_complete": complete,
            "eligible_waterbodies": int(mask.sum()) if mask is not None else pd.NA,
            "generation_twh": generation_twh,
            "retention_vs_recommended_l0_pct": generation_twh / recommended_l0_twh * 100.0
            if mask is not None
            else np.nan,
        }
    )


def make_summary(frame: pd.DataFrame) -> pd.DataFrame:
    all_rows = np.ones(len(frame), dtype=bool)
    l1 = frame["l1_pass_v112"].to_numpy(bool)
    known = frame["historical_dryup_evidence_known_v113"].fillna(False).to_numpy(bool)
    no_dry = frame["historical_no_dryup_observed_1991_2018_v113"].fillna(False).to_numpy(bool)
    core = frame["l2_core_no_dry_pass_v112"].to_numpy(bool)
    conservative = frame["l2_conservative_no_dry_pass_v112"].to_numpy(bool)
    uniform = frame["l0_generation_gwh"].to_numpy(float)
    recommended = frame["l0_type_specific_generation_gwh"].to_numpy(float)
    recommended_l0_twh = float(recommended.sum() / 1000.0)
    rows: list[dict[str, object]] = []
    add_row(
        rows,
        "L0-paper-uniform-deduplicated",
        "Authoritative physical-waterbody deduplication; 30% footprint for all types with 30 km2 cap",
        all_rows,
        uniform,
        recommended_l0_twh,
        True,
    )
    add_row(
        rows,
        "L0-recommended-type-deduplicated",
        "Authoritative physical-waterbody deduplication; reservoirs 30%, lakes/controlled lakes 10%, 30 km2 cap",
        all_rows,
        recommended,
        recommended_l0_twh,
        True,
    )
    add_row(
        rows,
        "L1-paper-uniform-hybrid-ice-deduplicated",
        "Deduplicated scope plus declared area and hybrid ice gates; uniform 30% footprint",
        l1,
        uniform,
        recommended_l0_twh,
        True,
    )
    add_row(
        rows,
        "L1-recommended-type-hybrid-ice-deduplicated",
        "Deduplicated scope plus declared area and hybrid ice gates; type-specific footprint",
        l1,
        recommended,
        recommended_l0_twh,
        True,
    )
    add_row(
        rows,
        "L2-core-historical-known-pass-lower-bound-deduplicated",
        "Deduplicated L1 + WDPA I-IV/population + complete 1991-2018 GLEV evidence with no zero-area month; unknowns fail",
        core & known & no_dry,
        recommended,
        recommended_l0_twh,
        False,
    )
    add_row(
        rows,
        "L2-core-historical-unknown-pass-upper-envelope-deduplicated",
        "Same core gates; observed dry-ups fail while unmatched waterbodies and 2019-2020 gap provisionally pass",
        core & (~known | no_dry),
        recommended,
        recommended_l0_twh,
        False,
    )
    add_row(
        rows,
        "L2-conservative-historical-known-pass-lower-bound-deduplicated",
        "Deduplicated L1 + any WDPA/Ramsar/population + complete 1991-2018 GLEV evidence with no zero-area month; unknowns fail",
        conservative & known & no_dry,
        recommended,
        recommended_l0_twh,
        False,
    )
    add_row(
        rows,
        "L2-conservative-historical-unknown-pass-upper-envelope-deduplicated",
        "Same conservative gates; observed dry-ups fail while unmatched waterbodies and 2019-2020 gap provisionally pass",
        conservative & (~known | no_dry),
        recommended,
        recommended_l0_twh,
        False,
    )
    add_row(
        rows,
        "L2-complete-paper-equivalent",
        "Unavailable: public surface-area evidence ends in 2018 and 8,151 source rows remain without complete dry-up evidence",
        None,
        None,
        recommended_l0_twh,
        False,
    )
    add_row(
        rows,
        "L3",
        "Unavailable: road/grid/engineering/economic/permitting gates remain incomplete",
        None,
        None,
        recommended_l0_twh,
        False,
    )
    return pd.DataFrame(rows)


def run(source_path: Path, output_dir: Path) -> None:
    frame = pd.read_parquet(source_path)
    if len(frame) != EXPECTED_INPUT_ROWS or not frame["wb_id"].is_unique:
        raise ValueError("v113 input scope mismatch")
    duplicate = frame["dryup_hylak_id_v113"].notna() & frame[
        "dryup_hylak_id_v113"
    ].duplicated(keep=False)
    groups = frame.loc[duplicate].copy()
    removal = groups.loc[
        groups["wb_type"].eq("controlled_lake")
        & groups["dryup_key_source_v113"].eq("hydrolakes_direct")
    ].copy()
    if len(removal) != EXPECTED_REMOVALS:
        raise ValueError(f"expected six authoritative duplicates to remove, found {len(removal)}")
    group_checks = groups.groupby("dryup_hylak_id_v113").agg(
        rows=("wb_id", "size"),
        reservoirs=("wb_type", lambda values: int(values.eq("reservoir").sum())),
        controlled_lakes=("wb_type", lambda values: int(values.eq("controlled_lake").sum())),
        authoritative=(
            "dryup_key_source_v113",
            lambda values: int(values.eq("grand_id_authoritative").sum()),
        ),
    )
    if not (
        group_checks["rows"].eq(2).all()
        and group_checks["reservoirs"].eq(1).all()
        and group_checks["controlled_lakes"].eq(1).all()
        and group_checks["authoritative"].eq(1).all()
    ):
        raise ValueError("duplicate groups do not all have authoritative 1:1 reservoir/control pairs")

    result = frame.loc[~frame["wb_id"].isin(removal["wb_id"])].copy()
    result["scope_deduplicated_v114"] = True
    result["scope_dedup_rule_v114"] = "retain_external_reservoir_drop_hydrolakes_controlled_same_grand_id"
    summary = make_summary(result)

    data_dir = output_dir / "data"
    reports_dir = output_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(
        data_dir / "paper_scope_deduplicated_historical_generation.parquet", index=False
    )
    removal_columns = [
        "wb_id",
        "wb_type",
        "hylak_id",
        "dryup_hylak_id_v113",
        "area_km2",
        "l0_generation_gwh",
        "l0_type_specific_generation_gwh",
        "l1_pass_v112",
        "l2_core_no_dry_pass_v112",
        "l2_conservative_no_dry_pass_v112",
    ]
    removal[removal_columns].to_csv(
        reports_dir / "authoritative_scope_duplicate_removals.csv", index=False
    )
    summary.to_csv(reports_dir / "paper_scope_level_summary.csv", index=False)
    qc = {
        "input_rows": int(len(frame)),
        "removed_rows": int(len(removal)),
        "removed_duplicate_hylak_groups": int(removal["dryup_hylak_id_v113"].nunique()),
        "output_rows": int(len(result)),
        "dedup_rule": "retain external GeoDAR/GRanD reservoir and drop HydroLAKES controlled-lake row sharing the same authoritative GRanD-linked Hylak_id",
        "removed_l0_recommended_twh": float(
            removal["l0_type_specific_generation_gwh"].sum() / 1000.0
        ),
        "removed_l1_recommended_twh": float(
            removal["l1_type_generation_gwh_v112"].sum() / 1000.0
        ),
        "remaining_duplicate_dryup_hylak_groups": int(
            result.loc[
                result["dryup_hylak_id_v113"].notna()
                & result["dryup_hylak_id_v113"].duplicated(keep=False),
                "dryup_hylak_id_v113",
            ].nunique()
        ),
        "l2_complete": False,
        "l3_complete": False,
    }
    (reports_dir / "scope_dedup_qc.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(qc, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-path",
        type=Path,
        default=Path(
            "_outputs/v113/data/paper_scope_historical_dryup_generation.parquet"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_outputs/v114"))
    args = parser.parse_args()
    run(args.source_path, args.output_dir)


if __name__ == "__main__":
    main()
