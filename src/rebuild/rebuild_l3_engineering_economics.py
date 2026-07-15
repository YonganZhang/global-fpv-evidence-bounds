#!/usr/bin/env python3
"""Add bathymetry coverage and parameterized FPV LCOE sensitivity to L3.

GLOBathy Dmax_use is estimated maximum waterbody depth, not installation-point
depth, so it is never used here as a definitive feasibility exclusion.  LCOE
scenarios are transparent parametric sensitivities based on the simulated
specific yield.  They omit grid reinforcement, country finance, taxes, land or
water rights, permitting and decommissioning, and therefore cannot close L3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SCOPE_ROWS = 198_731
EXPECTED_GLOBATHY_MD5 = "0d4af9b1a394e37ebe29058bfd3c8d69"
LIFETIME_YEARS = 25
ANNUAL_DEGRADATION = 0.005
ECONOMIC_SCENARIOS = {
    "optimistic": {"capex_usd_kw": 800.0, "opex_fraction": 0.015, "wacc": 0.06},
    "central": {"capex_usd_kw": 1000.0, "opex_fraction": 0.020, "wacc": 0.08},
    "conservative": {
        "capex_usd_kw": 1200.0,
        "opex_fraction": 0.025,
        "wacc": 0.10,
    },
}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_globathy(zip_path: Path) -> pd.DataFrame:
    if md5(zip_path) != EXPECTED_GLOBATHY_MD5:
        raise ValueError("GLOBathy archive checksum differs from Figshare metadata")
    member = (
        "GLOBathy_basic_parameters/"
        "GLOBathy_basic_parameters(ALL_LAKES).csv"
    )
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as stream:
        frame = pd.read_csv(
            stream,
            usecols=[
                "Hylak_id",
                "Dmax_box_m",
                "Dmax_est_PA_m",
                "Dmax_est_PAVEW_m",
                "Dmax_use_m",
            ],
            dtype={"Hylak_id": "int32"},
        )
    if len(frame) != 1_427_688 or not frame["Hylak_id"].is_unique:
        raise ValueError("GLOBathy row/key coverage mismatch")
    frame = frame.rename(
        columns={
            "Hylak_id": "dryup_hylak_id_v113",
            "Dmax_box_m": "globathy_dmax_box_m_v116",
            "Dmax_est_PA_m": "globathy_dmax_pa_m_v116",
            "Dmax_est_PAVEW_m": "globathy_dmax_pavew_m_v116",
            "Dmax_use_m": "globathy_dmax_use_m_v116",
        }
    )
    return frame


def present_value_lcoe(
    annual_yield_kwh_kw: np.ndarray,
    capex_usd_kw: float,
    opex_fraction: float,
    wacc: float,
) -> np.ndarray:
    years = np.arange(1, LIFETIME_YEARS + 1, dtype=float)
    discount = (1.0 + wacc) ** years
    discounted_opex = np.sum((capex_usd_kw * opex_fraction) / discount)
    discounted_energy_factor = np.sum(
        (1.0 - ANNUAL_DEGRADATION) ** (years - 1.0) / discount
    )
    cost_pv = capex_usd_kw + discounted_opex
    # USD/kWh multiplied by 1000 gives USD/MWh.
    return cost_pv / (annual_yield_kwh_kw * discounted_energy_factor) * 1000.0


def add_economics(frame: pd.DataFrame) -> pd.DataFrame:
    frame["annual_specific_yield_kwh_kwp_v116"] = (
        frame["annual_footprint_energy_kwh_m2"] * 10.0
    )
    frame["fpv_capacity_mw_v116"] = (
        frame["l0_type_specific_generation_gwh"]
        / frame["annual_specific_yield_kwh_kwp_v116"]
        * 1_000.0
    )
    annual_yield = frame["annual_specific_yield_kwh_kwp_v116"].to_numpy(float)
    for name, assumptions in ECONOMIC_SCENARIOS.items():
        frame[f"lcoe_{name}_usd_mwh_v116"] = present_value_lcoe(
            annual_yield, **assumptions
        ).astype(np.float32)
    return frame


def coverage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    depth_known = frame["globathy_dmax_use_m_v116"].notna()
    masks = {
        "all_deduplicated_scope": np.ones(len(frame), dtype=bool),
        "core_l2_dryup_known_lower": (
            frame["l2_core_no_dry_pass_v112"]
            & frame["historical_dryup_evidence_known_v113"].fillna(False)
            & frame["historical_no_dryup_observed_1991_2018_v113"].fillna(False)
        ).to_numpy(bool),
        "core_l2_unknown_pass_upper": (
            frame["l2_core_no_dry_pass_v112"]
            & (
                ~frame["historical_dryup_evidence_known_v113"].fillna(False)
                | frame["historical_no_dryup_observed_1991_2018_v113"].fillna(False)
            )
        ).to_numpy(bool),
        "core_l3_access_lower": (
            frame["l2_core_no_dry_pass_v112"]
            & frame["historical_dryup_evidence_known_v113"].fillna(False)
            & frame["historical_no_dryup_observed_1991_2018_v113"].fillna(False)
            & frame["road_within_10km_guaranteed_v115"]
            & frame["grid_within_25km_observed_or_hydropower_v115"]
        ).to_numpy(bool),
        "core_l3_access_upper": (
            frame["l2_core_no_dry_pass_v112"]
            & (
                ~frame["historical_dryup_evidence_known_v113"].fillna(False)
                | frame["historical_no_dryup_observed_1991_2018_v113"].fillna(False)
            )
            & frame["road_within_10km_possible_v115"]
            & frame["grid_within_25km_possible_or_hydropower_v115"]
        ).to_numpy(bool),
    }
    rows: list[dict[str, object]] = []
    for population, mask in masks.items():
        selected = frame.loc[mask]
        known_selected = selected.loc[depth_known[mask]]
        rows.append(
            {
                "population": population,
                "waterbodies": int(mask.sum()),
                "globathy_depth_known": int(depth_known[mask].sum()),
                "globathy_depth_coverage_pct": float(depth_known[mask].mean() * 100.0),
                "dmax_use_m_median_known": float(
                    known_selected["globathy_dmax_use_m_v116"].median()
                )
                if len(known_selected)
                else np.nan,
                "dmax_use_m_p95_known": float(
                    known_selected["globathy_dmax_use_m_v116"].quantile(0.95)
                )
                if len(known_selected)
                else np.nan,
                "central_lcoe_usd_mwh_median": float(
                    selected["lcoe_central_usd_mwh_v116"].median()
                ),
                "central_lcoe_usd_mwh_p05": float(
                    selected["lcoe_central_usd_mwh_v116"].quantile(0.05)
                ),
                "central_lcoe_usd_mwh_p95": float(
                    selected["lcoe_central_usd_mwh_v116"].quantile(0.95)
                ),
            }
        )
    return pd.DataFrame(rows)


def depth_lcoe_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    known_dry = frame["historical_dryup_evidence_known_v113"].fillna(False).to_numpy(bool)
    no_dry = frame["historical_no_dryup_observed_1991_2018_v113"].fillna(False).to_numpy(bool)
    core = frame["l2_core_no_dry_pass_v112"].to_numpy(bool)
    road_guaranteed = frame["road_within_10km_guaranteed_v115"].to_numpy(bool)
    road_possible = frame["road_within_10km_possible_v115"].to_numpy(bool)
    grid_observed = frame["grid_within_25km_observed_or_hydropower_v115"].to_numpy(bool)
    grid_possible = frame["grid_within_25km_possible_or_hydropower_v115"].to_numpy(bool)
    depth = frame["globathy_dmax_use_m_v116"].to_numpy(float)
    lcoe = frame["lcoe_central_usd_mwh_v116"].to_numpy(float)
    energy = frame["l0_type_specific_generation_gwh"].to_numpy(float)
    policies = {
        "access_evidence_lower": core
        & known_dry
        & no_dry
        & road_guaranteed
        & grid_observed,
        "access_mapping_upper": core
        & (~known_dry | no_dry)
        & road_possible
        & grid_possible,
    }
    rows: list[dict[str, object]] = []
    for policy, base in policies.items():
        for depth_threshold in [20.0, 50.0, 100.0, np.inf]:
            # Unknown depth is retained only in the no-depth-limit row. This is
            # a sensitivity, not a recommended hard exclusion.
            depth_pass = np.isfinite(depth) & (depth <= depth_threshold)
            if np.isinf(depth_threshold):
                depth_pass = np.ones(len(frame), dtype=bool)
            for lcoe_threshold in [50.0, 75.0, 100.0, 125.0, np.inf]:
                mask = base & depth_pass & (lcoe <= lcoe_threshold)
                rows.append(
                    {
                        "access_policy": policy,
                        "globathy_max_depth_sensitivity_m": "none"
                        if np.isinf(depth_threshold)
                        else depth_threshold,
                        "central_lcoe_sensitivity_usd_mwh": "none"
                        if np.isinf(lcoe_threshold)
                        else lcoe_threshold,
                        "eligible_waterbodies": int(mask.sum()),
                        "generation_twh": float(energy[mask].sum() / 1000.0),
                    }
                )
    return pd.DataFrame(rows)


def run(source_path: Path, globathy_zip: Path, output_dir: Path) -> None:
    frame = pd.read_parquet(source_path)
    if len(frame) != EXPECTED_SCOPE_ROWS or not frame["wb_id"].is_unique:
        raise ValueError("v115 accessibility scope mismatch")
    globathy = load_globathy(globathy_zip)
    frame = frame.merge(
        globathy, on="dryup_hylak_id_v113", how="left", validate="many_to_one"
    )
    frame = add_economics(frame)
    coverage = coverage_summary(frame)
    sensitivity = depth_lcoe_sensitivity(frame)

    data_dir = output_dir / "data"
    reports_dir = output_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(
        data_dir / "paper_scope_l3_engineering_economics.parquet", index=False
    )
    flag_columns = [
        "wb_id",
        "wb_type",
        "dryup_hylak_id_v113",
        "globathy_dmax_box_m_v116",
        "globathy_dmax_pa_m_v116",
        "globathy_dmax_pavew_m_v116",
        "globathy_dmax_use_m_v116",
        "annual_specific_yield_kwh_kwp_v116",
        "fpv_capacity_mw_v116",
        "lcoe_optimistic_usd_mwh_v116",
        "lcoe_central_usd_mwh_v116",
        "lcoe_conservative_usd_mwh_v116",
    ]
    frame[flag_columns].to_parquet(
        data_dir / "waterbody_engineering_economic_sensitivity.parquet", index=False
    )
    coverage.to_csv(reports_dir / "engineering_economic_coverage.csv", index=False)
    sensitivity.to_csv(
        reports_dir / "depth_lcoe_sensitivity.csv", index=False
    )
    qc = {
        "scope_rows": int(len(frame)),
        "globathy_source_rows": int(len(globathy)),
        "globathy_scope_matches": int(frame["globathy_dmax_use_m_v116"].notna().sum()),
        "globathy_scope_unknown": int(frame["globathy_dmax_use_m_v116"].isna().sum()),
        "globathy_dmax_nonpositive": int(
            frame["globathy_dmax_use_m_v116"].le(0).fillna(False).sum()
        ),
        "economic_assumptions": {
            "lifetime_years": LIFETIME_YEARS,
            "annual_degradation_fraction": ANNUAL_DEGRADATION,
            "scenarios": ECONOMIC_SCENARIOS,
        },
        "lcoe_crosscheck": "World Bank 2019 handbook reports illustrative FPV LCOE of roughly 49-92.6 USD/MWh across climate and WACC cases; this model is a parameter sensitivity, not a country project appraisal",
        "limitations": [
            "GLOBathy Dmax_use is modelled maximum lake depth, not depth at the proposed FPV footprint or anchor location",
            "the simple LCOE excludes grid connection/reinforcement, taxes, country risk, water rights, permitting, insurance variation and decommissioning",
            "no global comparable layers close wind gust, wave fetch, water-level range, sediment/geotechnical anchoring, competing use or social acceptance",
        ],
        "l3_complete": False,
    }
    (reports_dir / "engineering_economic_qc.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(coverage.to_string(index=False))
    print(json.dumps(qc, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-path",
        type=Path,
        default=Path("_outputs/v115/data/paper_scope_l3_accessibility.parquet"),
    )
    parser.add_argument(
        "--globathy-zip",
        type=Path,
        default=Path("_outputs/v116/raw/globathy/GLOBathy_basic_parameters.zip"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_outputs/v116"))
    args = parser.parse_args()
    run(args.source_path, args.globathy_zip, args.output_dir)


if __name__ == "__main__":
    main()
