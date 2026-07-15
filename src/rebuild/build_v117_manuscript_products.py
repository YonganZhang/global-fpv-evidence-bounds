#!/usr/bin/env python3
"""Create manuscript-facing trace tables from the validated v117 inventory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rebuild_v117_scientific_revision import ROOT, masks


DATA = ROOT / "_outputs/v117/data/fpv_reference_inventory_v117.parquet"
REPORTS = ROOT / "_outputs/v117/reports"
WOOLWAY_PUBLIC = ROOT / "_outputs/v110/data/woolway_public_lake_info.parquet"


def summarize(mask: np.ndarray, frame: pd.DataFrame) -> tuple[int, float]:
    energy = frame["l0_type_specific_generation_gwh"].to_numpy(float)
    return int(mask.sum()), float(energy[mask].sum() / 1000.0)


def type_level_summary(frame: pd.DataFrame) -> pd.DataFrame:
    selected = masks(frame)
    rows: list[dict[str, object]] = []
    for water_type, type_mask in {
        "lake_or_controlled_lake": frame["wb_type"].ne("reservoir").to_numpy(bool),
        "external_reservoir": frame["wb_type"].eq("reservoir").to_numpy(bool),
        "all": np.ones(len(frame), dtype=bool),
    }.items():
        for level, level_mask in selected.items():
            count, generation = summarize(type_mask & level_mask, frame)
            rows.append(
                {
                    "water_type": water_type,
                    "level": level,
                    "waterbodies": count,
                    "generation_twh": generation,
                }
            )
    return pd.DataFrame(rows)


def evidence_state_summary(frame: pd.DataFrame) -> pd.DataFrame:
    energy = frame["l0_type_specific_generation_gwh"].to_numpy(float)
    known = frame["historical_dryup_evidence_known_v113"].fillna(False).to_numpy(bool)
    dry = frame["historical_dryup_observed_1991_2018_v113"].fillna(False).to_numpy(bool)
    rows: list[dict[str, object]] = []
    for definition, column in [
        ("core", "l2_core_no_dry_pass_v112"),
        ("conservative", "l2_conservative_no_dry_pass_v112"),
    ]:
        base = frame[column].fillna(False).to_numpy(bool)
        for state, state_mask in [
            ("known_no_observed_dryup", known & ~dry),
            ("known_observed_dryup", known & dry),
            ("unknown_public_record", ~known),
        ]:
            mask = base & state_mask
            rows.append(
                {
                    "conservation_definition": definition,
                    "evidence_state": state,
                    "waterbodies": int(mask.sum()),
                    "generation_twh": float(energy[mask].sum() / 1000.0),
                }
            )
    return pd.DataFrame(rows)


def sequential_summary(frame: pd.DataFrame) -> pd.DataFrame:
    energy = frame["l0_type_specific_generation_gwh"].to_numpy(float)
    all_rows = np.ones(len(frame), dtype=bool)
    l1 = frame["l1_pass_v112"].fillna(False).to_numpy(bool)
    known = frame["historical_dryup_evidence_known_v113"].fillna(False).to_numpy(bool)
    no_dry = frame["historical_no_dryup_observed_1991_2018_v113"].fillna(False).to_numpy(bool)
    stages = [
        ("L0_reference", all_rows),
        ("L1_ice", l1),
        ("core_conservation", l1 & ~frame["in_wdpa_polygon_strict_i_iv"].fillna(False).to_numpy(bool)),
        ("core_population", frame["l2_core_no_dry_pass_v112"].fillna(False).to_numpy(bool)),
        ("core_L2_lower", frame["l2_core_no_dry_pass_v112"].fillna(False).to_numpy(bool) & known & no_dry),
        ("conservative_conservation", l1 & ~frame["in_wdpa_polygon_any"].fillna(False).to_numpy(bool) & ~frame["in_ramsar_polygon"].fillna(False).to_numpy(bool)),
        ("conservative_population", frame["l2_conservative_no_dry_pass_v112"].fillna(False).to_numpy(bool)),
        ("conservative_L2_lower", frame["l2_conservative_no_dry_pass_v112"].fillna(False).to_numpy(bool) & known & no_dry),
    ]
    return pd.DataFrame(
        [
            {
                "stage": name,
                "waterbodies": int(mask.sum()),
                "generation_twh": float(energy[mask].sum() / 1000.0),
            }
            for name, mask in stages
        ]
    )


def ice_source_summary(frame: pd.DataFrame) -> pd.DataFrame:
    energy = frame["l0_type_specific_generation_gwh"].to_numpy(float)
    rows = []
    for source, part in frame.groupby("ice_source_v112", dropna=False):
        idx = part.index.to_numpy(int)
        passed = part["ice_pass_v112"].fillna(False).to_numpy(bool)
        rows.append(
            {
                "ice_source": source,
                "waterbodies": int(len(part)),
                "passed": int(passed.sum()),
                "pass_rate_pct": float(passed.mean() * 100.0),
                "l0_generation_twh": float(energy[idx].sum() / 1000.0),
                "l1_generation_twh": float(energy[idx][passed].sum() / 1000.0),
            }
        )
    return pd.DataFrame(rows).sort_values("waterbodies", ascending=False)


def engineering_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    selected = masks(frame)
    energy = frame["l0_type_specific_generation_gwh"].to_numpy(float)
    depth = frame["globathy_dmax_use_m_v117"].to_numpy(float)
    lcoe = frame["lcoe_fpv_reference_usd_mwh_v117"].to_numpy(float)
    rows = []
    for population in ["core_l3_lower", "core_l3_upper"]:
        base = selected[population]
        for depth_limit in [None, 20.0, 50.0, 100.0]:
            depth_mask = np.ones(len(frame), dtype=bool) if depth_limit is None else np.isfinite(depth) & (depth <= depth_limit)
            for lcoe_limit in [None, 75.0, 100.0, 125.0]:
                lcoe_mask = np.ones(len(frame), dtype=bool) if lcoe_limit is None else np.isfinite(lcoe) & (lcoe <= lcoe_limit)
                mask = base & depth_mask & lcoe_mask
                rows.append(
                    {
                        "population": population,
                        "maximum_depth_limit_m": "none" if depth_limit is None else depth_limit,
                        "lcoe_limit_usd_mwh": "none" if lcoe_limit is None else lcoe_limit,
                        "waterbodies": int(mask.sum()),
                        "generation_twh": float(energy[mask].sum() / 1000.0),
                    }
                )
    return pd.DataFrame(rows)


def yield_benchmark_density(frame: pd.DataFrame, bins: int = 65) -> pd.DataFrame:
    """Aggregate the Woolway comparison for licence-safe figure rebuilding.

    The active figure needs the two-dimensional density, not row identities.
    Publishing fixed bins keeps the panel reproducible without redistributing
    the upstream row-level Woolway columns under an unverified blanket licence.
    """
    published = pd.read_parquet(
        WOOLWAY_PUBLIC,
        columns=["hylak_id", "woolway_fpv_output_kwh"],
    )
    present = frame.loc[
        frame["hylak_id"].notna(),
        ["hylak_id", "annual_specific_yield_kwh_kwp_v117"],
    ].copy()
    present["hylak_id"] = present["hylak_id"].astype(int)
    joined = present.merge(published, on="hylak_id", validate="one_to_one").dropna()
    joined = joined.loc[joined["woolway_fpv_output_kwh"].gt(0)]
    x = joined["woolway_fpv_output_kwh"].to_numpy(float)
    y = joined["annual_specific_yield_kwh_kwp_v117"].to_numpy(float)
    lower = float(min(x.min(), y.min()))
    upper = float(max(x.max(), y.max()))
    counts, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[lower, upper], [lower, upper]],
    )
    rows = []
    for x_index in range(bins):
        for y_index in range(bins):
            rows.append(
                {
                    "x_left_kwh_kwp": float(x_edges[x_index]),
                    "x_right_kwh_kwp": float(x_edges[x_index + 1]),
                    "y_bottom_kwh_kwp": float(y_edges[y_index]),
                    "y_top_kwh_kwp": float(y_edges[y_index + 1]),
                    "count": int(counts[x_index, y_index]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    gate_path = REPORTS / "consolidated_validation_gate_v117.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "pass":
        raise RuntimeError("v117 validation gate must pass before manuscript products")
    frame = pd.read_parquet(DATA)
    products = {
        "type_level_summary_v117.csv": type_level_summary(frame),
        "evidence_state_summary_v117.csv": evidence_state_summary(frame),
        "sequential_constraint_summary_v117.csv": sequential_summary(frame),
        "ice_source_summary_v117.csv": ice_source_summary(frame),
        "engineering_sensitivity_v117.csv": engineering_sensitivity(frame),
        "yield_benchmark_density_v117.csv": yield_benchmark_density(frame),
    }
    for name, table in products.items():
        table.to_csv(REPORTS / name, index=False)
        print(f"{name}: {len(table)} rows")


if __name__ == "__main__":
    main()
