#!/usr/bin/env python3
"""Recompute the declared 198,737-waterbody scope with the best ice hierarchy.

This run is deliberately separate from the v111 HydroLAKES scope sensitivity.
It keeps the manuscript inventory, fixes the L1 area threshold to the declared
0.01 km2 inventory minimum, uses LI-CCR ice duration where usable, then the
Woolway et al. public ice fraction, and only then the monthly air-temperature
proxy.  Numeric L2 screened envelopes remain scientifically incomplete until
the dry-up gate exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_WATERBODIES = 198_737
MIN_DECLARED_AREA_KM2 = 0.01


def add_hybrid_ice(frame: pd.DataFrame, benchmark_path: Path) -> pd.DataFrame:
    benchmark = pd.read_parquet(
        benchmark_path,
        columns=[
            "hylak_id",
            "woolway_ice_cover_fraction",
            "woolway_population_distance_km",
            "woolway_protected_area",
        ],
    )
    frame = frame.merge(benchmark, on="hylak_id", how="left", validate="many_to_one")
    frame["woolway_table_match_v112"] = frame[
        [
            "woolway_ice_cover_fraction",
            "woolway_population_distance_km",
            "woolway_protected_area",
        ]
    ].notna().any(axis=1)
    proxy_pass = frame["ice_months_temp_proxy"].le(6)
    author_available = frame["woolway_ice_cover_fraction"].notna()
    li_available = frame["ice_direct_usable"].fillna(False).astype(bool)

    frame["ice_pass_v112"] = proxy_pass
    frame["ice_source_v112"] = "nasa_power_monthly_temp_proxy_2019_2023"
    frame.loc[author_available, "ice_pass_v112"] = frame.loc[
        author_available, "woolway_ice_cover_fraction"
    ].le(0.5)
    frame.loc[author_available, "ice_source_v112"] = (
        "woolway_air_temperature_lag_model_1991_2020"
    )
    frame.loc[li_available, "ice_pass_v112"] = frame.loc[
        li_available, "ice_pass_li_ccr"
    ].astype(bool)
    frame.loc[li_available, "ice_source_v112"] = "li_ccr_modis_phenology_2002_2020"
    frame["l1_area_pass_v112"] = frame["area_km2"].ge(MIN_DECLARED_AREA_KM2)
    frame["l1_pass_v112"] = frame["l1_area_pass_v112"] & frame["ice_pass_v112"]
    return frame


def add_levels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    energy = frame["annual_footprint_energy_kwh_m2"].to_numpy(float)
    uniform_l0 = frame["l0_generation_gwh"].to_numpy(float)
    type_l0 = frame["l0_type_specific_generation_gwh"].to_numpy(float)
    l1 = frame["l1_pass_v112"].to_numpy(bool)
    strict_core = (
        l1
        & ~frame["in_wdpa_polygon_strict_i_iv"].to_numpy(bool)
        & frame["within_10km_population_center"].to_numpy(bool)
    )
    conservative = (
        l1
        & ~frame["in_wdpa_polygon_any"].to_numpy(bool)
        & ~frame["in_ramsar_polygon"].to_numpy(bool)
        & frame["within_10km_population_center"].to_numpy(bool)
    )
    author_visible = (
        frame["woolway_ice_cover_fraction"].le(0.5)
        & ~frame["woolway_protected_area"].eq(1.0)
        & frame["woolway_population_distance_km"].le(10.0)
    ).fillna(False).to_numpy(bool)

    frame["l1_uniform_generation_gwh_v112"] = np.where(l1, uniform_l0, 0.0)
    frame["l1_type_generation_gwh_v112"] = np.where(l1, type_l0, 0.0)
    frame["l2_core_no_dry_pass_v112"] = strict_core
    frame["l2_conservative_no_dry_pass_v112"] = conservative
    frame["l2_author_visible_pass_v112"] = author_visible
    frame["l2_core_uniform_generation_gwh_v112"] = np.where(strict_core, uniform_l0, 0.0)
    frame["l2_core_type_generation_gwh_v112"] = np.where(strict_core, type_l0, 0.0)
    frame["l2_conservative_uniform_generation_gwh_v112"] = np.where(
        conservative, uniform_l0, 0.0
    )
    frame["l2_conservative_type_generation_gwh_v112"] = np.where(
        conservative, type_l0, 0.0
    )
    frame["l2_author_visible_type_generation_gwh_v112"] = np.where(
        author_visible, type_l0, 0.0
    )

    scenarios = [
        (
            "L0-paper-uniform",
            np.ones(len(frame), dtype=bool),
            uniform_l0,
            True,
            "Declared paper footprint: 30% for all types with 30 km2 cap",
        ),
        (
            "L0-recommended-type",
            np.ones(len(frame), dtype=bool),
            type_l0,
            True,
            "30% reservoirs and 10% natural/controlled lakes with 30 km2 cap",
        ),
        (
            "L1-paper-uniform-hybrid-ice",
            l1,
            frame["l1_uniform_generation_gwh_v112"].to_numpy(float),
            True,
            "Declared area minimum plus LI-CCR, Woolway ice, then labelled temperature fallback",
        ),
        (
            "L1-recommended-type-hybrid-ice",
            l1,
            frame["l1_type_generation_gwh_v112"].to_numpy(float),
            True,
            "Recommended footprint plus the same hybrid ice hierarchy",
        ),
        (
            "L2-core-screened-no-dry-type",
            strict_core,
            frame["l2_core_type_generation_gwh_v112"].to_numpy(float),
            False,
            "L1 + outside WDPA I-IV + within 10 km population centre; dry-up missing",
        ),
        (
            "L2-conservative-screened-no-dry-type",
            conservative,
            frame["l2_conservative_type_generation_gwh_v112"].to_numpy(float),
            False,
            "L1 + outside any WDPA/Ramsar polygon + within 10 km population centre; dry-up missing",
        ),
        (
            "L2-author-visible-10km-no-dry-type",
            author_visible,
            frame["l2_author_visible_type_generation_gwh_v112"].to_numpy(float),
            False,
            "Public Woolway table's visible ice/protected/population columns; dry-up field absent",
        ),
        (
            "L2-complete",
            None,
            None,
            False,
            "Unavailable: explicit dry-up flag is missing",
        ),
        (
            "L3",
            None,
            None,
            False,
            "Unavailable: road/grid/engineering/economic/permitting gates are missing",
        ),
    ]
    rows = []
    uniform_l0_twh = float(uniform_l0.sum() / 1000.0)
    recommended_l0_twh = float(type_l0.sum() / 1000.0)
    for tier, mask, values, complete, definition in scenarios:
        generation_twh = float(np.sum(values) / 1000.0) if values is not None else np.nan
        corresponding_l0_twh = (
            uniform_l0_twh if "paper-uniform" in tier else recommended_l0_twh
        )
        rows.append(
            {
                "tier": tier,
                "definition": definition,
                "scientific_tier_complete": complete,
                "eligible_waterbodies": int(mask.sum()) if mask is not None else pd.NA,
                "generation_twh": generation_twh,
                "retention_vs_corresponding_l0_pct": (
                    generation_twh / corresponding_l0_twh * 100.0
                    if values is not None
                    else np.nan
                ),
                "retention_vs_recommended_l0_pct": (
                    generation_twh / recommended_l0_twh * 100.0
                    if values is not None
                    else np.nan
                ),
            }
        )
    return frame, pd.DataFrame(rows)


def run(source_dir: Path, benchmark_path: Path, output_dir: Path) -> None:
    source = pd.read_parquet(source_dir / "data/waterbody_generation_rebuild.parquet")
    assert len(source) == EXPECTED_WATERBODIES and source["wb_id"].is_unique
    result = add_hybrid_ice(source, benchmark_path)
    result, summary = add_levels(result)

    data_dir = output_dir / "data"
    reports = output_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "wb_id",
        "hylak_id",
        "wb_type",
        "area_km2",
        "country",
        "continent",
        "annual_footprint_energy_kwh_m2",
        "l0_generation_gwh",
        "l0_type_specific_generation_gwh",
        "ice_months_temp_proxy",
        "ice_direct_usable",
        "ice_pass_li_ccr",
        "woolway_table_match_v112",
        "woolway_ice_cover_fraction",
        "ice_source_v112",
        "ice_pass_v112",
        "l1_area_pass_v112",
        "l1_pass_v112",
        "in_wdpa_polygon_any",
        "in_wdpa_polygon_strict_i_iv",
        "in_ramsar_polygon",
        "within_10km_population_center",
        "woolway_population_distance_km",
        "woolway_protected_area",
        "l1_uniform_generation_gwh_v112",
        "l1_type_generation_gwh_v112",
        "l2_core_no_dry_pass_v112",
        "l2_conservative_no_dry_pass_v112",
        "l2_author_visible_pass_v112",
        "l2_core_type_generation_gwh_v112",
        "l2_conservative_type_generation_gwh_v112",
        "l2_author_visible_type_generation_gwh_v112",
    ]
    result[output_columns].to_parquet(
        data_dir / "paper_scope_hybrid_generation.parquet", index=False
    )
    summary.to_csv(reports / "paper_scope_level_summary.csv", index=False)

    source_counts = result["ice_source_v112"].value_counts().to_dict()
    qc = {
        "waterbodies": len(result),
        "scope": "declared manuscript inventory: external reservoirs plus HydroLAKES natural/controlled lakes >=1 km2",
        "declared_l1_area_minimum_km2": MIN_DECLARED_AREA_KM2,
        "area_failures": int((~result["l1_area_pass_v112"]).sum()),
        "ice_source_counts": {str(key): int(value) for key, value in source_counts.items()},
        "woolway_public_table_matches": int(result["woolway_table_match_v112"].sum()),
        "woolway_ice_values_available": int(result["woolway_ice_cover_fraction"].notna().sum()),
        "dry_up_gate_available": False,
        "l2_complete": False,
        "l3_complete": False,
    }
    (reports / "paper_scope_hybrid_qc.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(qc, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("_outputs/v110"))
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("_outputs/v110/data/woolway_public_lake_info.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_outputs/v112"))
    args = parser.parse_args()
    run(args.source_dir, args.benchmark_path, args.output_dir)


if __name__ == "__main__":
    main()
