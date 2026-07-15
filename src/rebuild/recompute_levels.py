#!/usr/bin/env python3
"""Recompute transparent FPV potential tiers from versioned rebuild data.

Key corrections relative to the legacy model:

* one definition is used for every waterbody type;
* gross full-water energy is separated from the designed FPV footprint;
* the 30 km2 cap applies to FPV footprint, not only to reservoirs;
* footprint power density is explicit (1 kWp per 10 m2, 0.1 kWp/m2);
* ice uses LI-CCR satellite-derived duration where available and a labelled
  monthly-temperature fallback elsewhere;
* protected-area status can be supplied for every waterbody polygon;
* uncomputed population, dry-up and practical/economic gates remain NA rather
  than being silently treated as passing.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
REPRESENTATIVE_DAY = np.array([17, 47, 75, 105, 135, 162, 198, 228, 258, 288, 318, 344])
SOLAR_CONSTANT = 1367.0
ALBEDO_WATER = 0.06
TEMP_COEFF_POWER = -0.0037
T_REF_C = 25.0
FAIMAN_U0 = 25.0
FAIMAN_U1 = 6.84
BASE_PERFORMANCE_RATIO = 0.80
PACKING_DENSITY_KWP_M2 = 0.10  # Woolway et al.: 10 m2 footprint per kW
MIN_WATERBODY_AREA_KM2 = 0.10
MAX_ICE_MONTHS = 6
FPV_COVERAGE_PRIMARY = 0.30
FPV_COVERAGE_LITERATURE = 0.10
MAX_FPV_FOOTPRINT_KM2 = 30.0


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_waterbodies(root: Path) -> pd.DataFrame:
    gpkg = root / "Dataset/data_processed/merged/all_waterbodies.gpkg"
    with sqlite3.connect(gpkg) as con:
        wb = pd.read_sql_query(
            "SELECT wb_id, wb_type, country, continent, area_km2, centroid_lon, centroid_lat "
            "FROM waterbodies ORDER BY wb_id",
            con,
        )
    assert len(wb) == 198_737 and wb["wb_id"].is_unique
    return wb


def solar_declination(day: int) -> float:
    return 23.45 * np.sin(np.radians(360 * (284 + day) / 365))


def eccentricity_correction(day: int) -> float:
    b = 2 * np.pi * (day - 1) / 365
    return (
        1.000110
        + 0.034221 * np.cos(b)
        + 0.001280 * np.sin(b)
        + 0.000719 * np.cos(2 * b)
        + 0.000077 * np.sin(2 * b)
    )


def sunset_hour_angle(lat_rad: np.ndarray, dec_rad: float) -> np.ndarray:
    return np.arccos(np.clip(-np.tan(lat_rad) * np.tan(dec_rad), -1.0, 1.0))


def extraterrestrial_daily(lat_deg: np.ndarray, month: int) -> np.ndarray:
    lat = np.radians(lat_deg)
    day = int(REPRESENTATIVE_DAY[month - 1])
    dec = np.radians(solar_declination(day))
    ws = sunset_hour_angle(lat, dec)
    h0 = (24 * 3600 / np.pi) * SOLAR_CONSTANT * eccentricity_correction(day) * (
        ws * np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.sin(ws)
    )
    return np.maximum(h0 / 3.6e6, 0.001)


def erbs_diffuse_fraction(kt: np.ndarray) -> np.ndarray:
    kt = np.clip(kt, 0.0, 1.0)
    value = np.where(
        kt <= 0.22,
        1.0 - 0.09 * kt,
        np.where(
            kt <= 0.80,
            0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4,
            0.165,
        ),
    )
    return np.clip(value, 0.0, 1.0)


def beam_tilt_factor(lat_deg: np.ndarray, tilt_deg: np.ndarray, month: int) -> np.ndarray:
    lat = np.radians(lat_deg)
    tilt = np.radians(tilt_deg)
    day = int(REPRESENTATIVE_DAY[month - 1])
    dec = np.radians(solar_declination(day))
    ws = sunset_hour_angle(lat, dec)
    sign = np.where(lat_deg >= 0, 1.0, -1.0)
    effective_lat = lat - sign * tilt
    ws_tilt = sunset_hour_angle(effective_lat, dec)
    ws_prime = np.minimum(ws, ws_tilt)
    numerator = (
        ws_prime * np.sin(effective_lat) * np.sin(dec)
        + np.cos(effective_lat) * np.cos(dec) * np.sin(ws_prime)
    )
    denominator = ws * np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.sin(ws)
    ratio = np.ones_like(denominator, dtype=float)
    np.divide(numerator, denominator, out=ratio, where=denominator > 0.001)
    return np.clip(ratio, 0.0, 5.0)


def monthly_poa(
    ghi_wm2: np.ndarray,
    lat_deg: np.ndarray,
    tilt_deg: np.ndarray,
    month: int,
) -> np.ndarray:
    days = DAYS_IN_MONTH[month - 1]
    horizontal = ghi_wm2 * 24 * days / 1000.0
    extraterrestrial = extraterrestrial_daily(lat_deg, month) * days
    diffuse_fraction = erbs_diffuse_fraction(np.clip(horizontal / extraterrestrial, 0.0, 1.0))
    diffuse = diffuse_fraction * horizontal
    beam = horizontal - diffuse
    rb = beam_tilt_factor(lat_deg, tilt_deg, month)
    tilt = np.radians(tilt_deg)
    poa = (
        beam * rb
        + diffuse * (1 + np.cos(tilt)) / 2
        + horizontal * ALBEDO_WATER * (1 - np.cos(tilt)) / 2
    )
    return np.maximum(poa, 0.0)


def simulate_specific_energy(wb: pd.DataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = weather.merge(
        wb[["wb_id", "centroid_lat"]], on="wb_id", how="left", validate="many_to_one"
    )
    merged["tilt_deg"] = np.minimum(np.abs(merged["centroid_lat"]), 20.0)
    monthly_frames = []
    for month in range(1, 13):
        frame = merged[merged["month"] == month].copy()
        poa = monthly_poa(
            frame["ghi_wm2"].to_numpy(float),
            frame["centroid_lat"].to_numpy(float),
            frame["tilt_deg"].to_numpy(float),
            month,
        )
        poa_wm2 = poa * 1000.0 / (24 * DAYS_IN_MONTH[month - 1])
        wind = np.maximum(frame["wind_speed_ms"].to_numpy(float), 0.0)
        cell_temp = frame["temp_c"].to_numpy(float) + poa_wm2 / (FAIMAN_U0 + FAIMAN_U1 * wind)
        temperature_factor = np.clip(1.0 + TEMP_COEFF_POWER * (cell_temp - T_REF_C), 0.75, 1.15)
        # kWh per kWp for this month.  Footprint energy is applied later using
        # the explicit 0.1 kWp/m2 packing density.
        specific_yield = poa * BASE_PERFORMANCE_RATIO * temperature_factor
        frame["poa_kwh_m2"] = poa
        frame["cell_temp_c"] = cell_temp
        frame["temperature_factor"] = temperature_factor
        frame["specific_yield_kwh_kwp"] = specific_yield
        frame["footprint_energy_kwh_m2"] = specific_yield * PACKING_DENSITY_KWP_M2
        monthly_frames.append(frame)
    monthly = pd.concat(monthly_frames, ignore_index=True).sort_values(["wb_id", "month"])
    annual = monthly.groupby("wb_id", as_index=False).agg(
        annual_poa_kwh_m2=("poa_kwh_m2", "sum"),
        annual_specific_yield_kwh_kwp=("specific_yield_kwh_kwp", "sum"),
        annual_footprint_energy_kwh_m2=("footprint_energy_kwh_m2", "sum"),
        avg_temp_c=("temp_c", "mean"),
        avg_wind_ms=("wind_speed_ms", "mean"),
        ice_months_temp_proxy=("temp_c", lambda x: int((x < 0).sum())),
    )
    annual["capacity_factor"] = annual["annual_specific_yield_kwh_kwp"] / 8760.0
    assert len(annual) == 198_737
    return annual, monthly


def add_tiers(run_dir: Path, wb: pd.DataFrame, annual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = wb.merge(annual, on="wb_id", how="left", validate="one_to_one")
    assert df["annual_footprint_energy_kwh_m2"].notna().all()
    energy = df["annual_footprint_energy_kwh_m2"]
    df["gross_full_water_area_km2"] = df["area_km2"]
    df["l0_fpv_area_km2"] = np.minimum(df["area_km2"] * FPV_COVERAGE_PRIMARY, MAX_FPV_FOOTPRINT_KM2)
    df["l0_10pct_fpv_area_km2"] = np.minimum(
        df["area_km2"] * FPV_COVERAGE_LITERATURE, MAX_FPV_FOOTPRINT_KM2
    )
    type_specific_coverage = np.where(
        df["wb_type"].eq("reservoir"), FPV_COVERAGE_PRIMARY, FPV_COVERAGE_LITERATURE
    )
    df["l0_type_specific_fpv_area_km2"] = np.minimum(
        df["area_km2"] * type_specific_coverage, MAX_FPV_FOOTPRINT_KM2
    )
    df["l1_area_pass"] = df["area_km2"] >= MIN_WATERBODY_AREA_KM2
    df["l1_ice_pass_temp_proxy"] = df["ice_months_temp_proxy"] <= MAX_ICE_MONTHS
    df["l1_pass_temp_proxy"] = df["l1_area_pass"] & df["l1_ice_pass_temp_proxy"]
    ice_path = run_dir / "data/waterbody_ice_flags.parquet"
    if ice_path.exists():
        ice = pd.read_parquet(ice_path).drop(
            columns=["ice_months_temp_proxy", "ice_pass_temp_proxy"], errors="ignore"
        )
        df = df.merge(ice, on="wb_id", how="left", validate="one_to_one")
        assert df["ice_pass_best_available"].notna().all()
        df["l1_ice_pass"] = df["ice_pass_best_available"].astype(bool)
        ice_best_available = True
    else:
        df["l1_ice_pass"] = df["l1_ice_pass_temp_proxy"]
        df["ice_source"] = "nasa_power_monthly_temp_proxy_2019_2023"
        df["ice_direct_usable"] = False
        ice_best_available = False
    df["l1_pass"] = df["l1_area_pass"] & df["l1_ice_pass"]

    wdpa_path = run_dir / "data/wdpa_all_waterbody_polygon_flags.parquet"
    if wdpa_path.exists():
        wdpa = pd.read_parquet(wdpa_path)
        df = df.merge(wdpa, on="wb_id", how="left", validate="one_to_one")
        assert df["in_wdpa_polygon_any"].notna().all()
        df["l2_partial_pass"] = df["l1_pass"] & ~df["in_wdpa_polygon_any"].astype(bool)
        wdpa_complete = True
    else:
        df["in_wdpa_polygon_any"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
        df["in_wdpa_polygon_strict_i_iv"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
        df["l2_partial_pass"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
        wdpa_complete = False

    ramsar_path = run_dir / "data/ramsar_waterbody_polygon_flags.parquet"
    if ramsar_path.exists():
        ramsar = pd.read_parquet(ramsar_path)
        df = df.merge(ramsar, on="wb_id", how="left", validate="one_to_one")
        assert df["in_ramsar_polygon"].notna().all()
        ramsar_complete = True
    else:
        df["in_ramsar_polygon"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
        df["ramsar_site_count"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        ramsar_complete = False
    if wdpa_complete and ramsar_complete:
        df["l2_ecological_pass"] = (
            df["l2_partial_pass"] & ~df["in_ramsar_polygon"].astype(bool)
        )
    else:
        df["l2_ecological_pass"] = pd.Series(pd.NA, index=df.index, dtype="boolean")

    pop_path = run_dir / "data/population_center_10km_flags.parquet"
    dry_path = run_dir / "data/waterbody_dryup_flags.parquet"
    population_complete = pop_path.exists()
    if population_complete:
        population = pd.read_parquet(pop_path)
        df = df.merge(population, on="wb_id", how="left", validate="one_to_one")
        assert df["within_10km_population_center"].notna().all()
    if wdpa_complete and population_complete:
        df["l2_screened_no_dry_pass"] = (
            df["l2_partial_pass"] & df["within_10km_population_center"].astype(bool)
        )
        screened_l2 = True
    else:
        df["l2_screened_no_dry_pass"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
        screened_l2 = False
    if wdpa_complete and ramsar_complete and population_complete:
        df["l2_screened_ramsar_no_dry_pass"] = (
            df["l2_ecological_pass"]
            & df["within_10km_population_center"].astype(bool)
        )
        screened_ramsar_l2 = True
    else:
        df["l2_screened_ramsar_no_dry_pass"] = pd.Series(
            pd.NA, index=df.index, dtype="boolean"
        )
        screened_ramsar_l2 = False

    complete_l2 = screened_l2 and dry_path.exists()
    if complete_l2:
        dry = pd.read_parquet(dry_path)
        df = df.merge(dry, on="wb_id", how="left", validate="one_to_one")
        df["l2_complete_pass"] = (
            df["l2_screened_no_dry_pass"]
            & ~df["dried_during_study_period"].astype(bool)
        )
        if screened_ramsar_l2:
            df["l2_complete_ramsar_pass"] = (
                df["l2_screened_ramsar_no_dry_pass"]
                & ~df["dried_during_study_period"].astype(bool)
            )
        else:
            df["l2_complete_ramsar_pass"] = pd.Series(
                pd.NA, index=df.index, dtype="boolean"
            )
    else:
        df["l2_complete_pass"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
        df["l2_complete_ramsar_pass"] = pd.Series(pd.NA, index=df.index, dtype="boolean")

    df["gross_generation_gwh"] = energy * df["gross_full_water_area_km2"]
    df["l0_generation_gwh"] = energy * df["l0_fpv_area_km2"]
    df["l0_10pct_generation_gwh"] = energy * df["l0_10pct_fpv_area_km2"]
    df["l0_type_specific_generation_gwh"] = energy * df["l0_type_specific_fpv_area_km2"]
    df["l1_generation_gwh"] = np.where(df["l1_pass"], energy * df["l0_fpv_area_km2"], 0.0)
    df["l1_temp_proxy_generation_gwh"] = np.where(
        df["l1_pass_temp_proxy"], energy * df["l0_fpv_area_km2"], 0.0
    )
    df["l1_type_specific_generation_gwh"] = np.where(
        df["l1_pass"], energy * df["l0_type_specific_fpv_area_km2"], 0.0
    )
    if wdpa_complete:
        df["l2_partial_generation_gwh"] = np.where(
            df["l2_partial_pass"], energy * df["l0_fpv_area_km2"], 0.0
        )
        df["l2_partial_type_generation_gwh"] = np.where(
            df["l2_partial_pass"], energy * df["l0_type_specific_fpv_area_km2"], 0.0
        )
    else:
        df["l2_partial_generation_gwh"] = np.nan
        df["l2_partial_type_generation_gwh"] = np.nan
    if screened_l2:
        df["l2_screened_no_dry_generation_gwh"] = np.where(
            df["l2_screened_no_dry_pass"], energy * df["l0_fpv_area_km2"], 0.0
        )
        df["l2_screened_no_dry_type_generation_gwh"] = np.where(
            df["l2_screened_no_dry_pass"], energy * df["l0_type_specific_fpv_area_km2"], 0.0
        )
    else:
        df["l2_screened_no_dry_generation_gwh"] = np.nan
        df["l2_screened_no_dry_type_generation_gwh"] = np.nan
    if screened_ramsar_l2:
        df["l2_screened_ramsar_no_dry_generation_gwh"] = np.where(
            df["l2_screened_ramsar_no_dry_pass"],
            energy * df["l0_fpv_area_km2"],
            0.0,
        )
        df["l2_screened_ramsar_no_dry_type_generation_gwh"] = np.where(
            df["l2_screened_ramsar_no_dry_pass"],
            energy * df["l0_type_specific_fpv_area_km2"],
            0.0,
        )
    else:
        df["l2_screened_ramsar_no_dry_generation_gwh"] = np.nan
        df["l2_screened_ramsar_no_dry_type_generation_gwh"] = np.nan
    if complete_l2:
        df["l2_complete_generation_gwh"] = np.where(
            df["l2_complete_pass"], energy * df["l0_fpv_area_km2"], 0.0
        )
        df["l2_complete_type_generation_gwh"] = np.where(
            df["l2_complete_pass"],
            energy * df["l0_type_specific_fpv_area_km2"],
            0.0,
        )
        if screened_ramsar_l2:
            df["l2_complete_ramsar_generation_gwh"] = np.where(
                df["l2_complete_ramsar_pass"],
                energy * df["l0_fpv_area_km2"],
                0.0,
            )
            df["l2_complete_ramsar_type_generation_gwh"] = np.where(
                df["l2_complete_ramsar_pass"],
                energy * df["l0_type_specific_fpv_area_km2"],
                0.0,
            )
        else:
            df["l2_complete_ramsar_generation_gwh"] = np.nan
            df["l2_complete_ramsar_type_generation_gwh"] = np.nan
    else:
        df["l2_complete_generation_gwh"] = np.nan
        df["l2_complete_type_generation_gwh"] = np.nan
        df["l2_complete_ramsar_generation_gwh"] = np.nan
        df["l2_complete_ramsar_type_generation_gwh"] = np.nan
    # A deployable/economic L3 is deliberately not inferred from an LLM score.
    df["l3_generation_gwh"] = np.nan

    tiers = [
        ("G0", "gross full-water upper bound (context only)", "gross_generation_gwh", np.ones(len(df), bool), True),
        ("L0", "30% coverage, 30 km2 footprint cap", "l0_generation_gwh", np.ones(len(df), bool), True),
        ("L0-10", "10% coverage sensitivity, 30 km2 footprint cap", "l0_10pct_generation_gwh", np.ones(len(df), bool), True),
        ("L0-type", "30% reservoirs; 10% natural/controlled lakes; 30 km2 cap", "l0_type_specific_generation_gwh", np.ones(len(df), bool), True),
        ("L1-proxy", "L0 + area >=0.1 km2 + <=6 sub-zero monthly means", "l1_temp_proxy_generation_gwh", df["l1_pass_temp_proxy"], True),
        ("L1", "L0 + area >=0.1 km2 + LI-CCR ice duration where available, labelled temperature fallback elsewhere", "l1_generation_gwh", df["l1_pass"], True),
        ("L1-type", "L0-type + area and sub-zero-month screens", "l1_type_specific_generation_gwh", df["l1_pass"], True),
        ("L2-partial", "L1 + no intersection with WDPA polygon", "l2_partial_generation_gwh", df["l2_partial_pass"] if wdpa_complete else None, wdpa_complete),
        ("L2-partial-type", "L1-type + no intersection with WDPA polygon", "l2_partial_type_generation_gwh", df["l2_partial_pass"] if wdpa_complete else None, wdpa_complete),
        ("L2-screened", "L2-partial + within 10 km of a population centre; dry-up still missing", "l2_screened_no_dry_generation_gwh", df["l2_screened_no_dry_pass"] if screened_l2 else None, screened_l2),
        ("L2-screened-type", "L2-screened with 30% reservoir and 10% lake coverage", "l2_screened_no_dry_type_generation_gwh", df["l2_screened_no_dry_pass"] if screened_l2 else None, screened_l2),
        ("L2-screened-Ramsar", "L2-screened + no intersection with published Ramsar polygons", "l2_screened_ramsar_no_dry_generation_gwh", df["l2_screened_ramsar_no_dry_pass"] if screened_ramsar_l2 else None, screened_ramsar_l2),
        ("L2-screened-Ramsar-type", "L2-screened-Ramsar with 30% reservoir and 10% lake coverage", "l2_screened_ramsar_no_dry_type_generation_gwh", df["l2_screened_ramsar_no_dry_pass"] if screened_ramsar_l2 else None, screened_ramsar_l2),
        ("L2-complete", "L2-partial + population-centre proximity + no dry-up", "l2_complete_generation_gwh", df["l2_complete_pass"] if complete_l2 else None, complete_l2),
        ("L2-complete-type", "L2-complete with 30% reservoir and 10% lake coverage", "l2_complete_type_generation_gwh", df["l2_complete_pass"] if complete_l2 else None, complete_l2),
        ("L2-complete-Ramsar", "L2-complete + no intersection with published Ramsar polygons", "l2_complete_ramsar_generation_gwh", df["l2_complete_ramsar_pass"] if complete_l2 and screened_ramsar_l2 else None, complete_l2 and screened_ramsar_l2),
        ("L2-complete-Ramsar-type", "L2-complete-Ramsar with 30% reservoir and 10% lake coverage", "l2_complete_ramsar_type_generation_gwh", df["l2_complete_ramsar_pass"] if complete_l2 and screened_ramsar_l2 else None, complete_l2 and screened_ramsar_l2),
        ("L3", "practical/economic/grid/road/site-design gates", "l3_generation_gwh", None, False),
    ]
    rows = []
    gross = float(df["gross_generation_gwh"].sum())
    l0 = float(df["l0_generation_gwh"].sum())
    for tier, definition, generation_col, gate, complete in tiers:
        total = float(df[generation_col].sum()) if complete else np.nan
        rows.append(
            {
                "tier": tier,
                "definition": definition,
                "complete": complete,
                "eligible_waterbodies": int(np.asarray(gate, dtype=bool).sum()) if gate is not None else pd.NA,
                "generation_twh": total / 1000.0 if complete else np.nan,
                "retention_vs_gross_pct": total / gross * 100 if complete else np.nan,
                "retention_vs_l0_pct": total / l0 * 100 if complete else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    return df, summary


def legacy_comparison(root: Path, new_summary: pd.DataFrame, reports: Path) -> None:
    legacy = pd.read_csv(
        root / "Dataset/data_processed/synthesis/f_score_synthesis.csv",
        low_memory=False,
    )
    rows = []
    for level in [0, 1, 2]:
        col = f"generation_L{level}_gwh"
        rows.append({"series": f"legacy_L{level}", "generation_twh": legacy[col].sum() / 1000.0})
    rows.append(
        {
            "series": "legacy_F_weighted",
            "generation_twh": (legacy["generation_L2_gwh"].fillna(0) * legacy["f_score_L2"].fillna(0)).sum() / 1000.0,
        }
    )
    for row in new_summary.to_dict("records"):
        rows.append({"series": f"rebuild_{row['tier']}", "generation_twh": row["generation_twh"]})
    pd.DataFrame(rows).to_csv(reports / "legacy_vs_rebuild_levels.csv", index=False)


def coverage_sensitivity(result: pd.DataFrame, reports: Path) -> None:
    rows = []
    energy = result["annual_footprint_energy_kwh_m2"]
    l2_partial_available = result["l2_partial_pass"].notna().all()
    l2_screened_available = result["l2_screened_no_dry_pass"].notna().all()
    for coverage in [0.05, 0.10, 0.20, 0.30]:
        for capped in [False, True]:
            area = result["area_km2"] * coverage
            if capped:
                area = np.minimum(area, MAX_FPV_FOOTPRINT_KM2)
            gross_generation = energy * area
            l1_generation = np.where(result["l1_pass"], gross_generation, 0.0)
            l2_partial_generation = (
                np.where(result["l2_partial_pass"].astype(bool), gross_generation, 0.0)
                if l2_partial_available
                else np.full(len(result), np.nan)
            )
            l2_screened_generation = (
                np.where(result["l2_screened_no_dry_pass"].astype(bool), gross_generation, 0.0)
                if l2_screened_available
                else np.full(len(result), np.nan)
            )
            rows.append(
                {
                    "coverage_pct": coverage * 100,
                    "max_footprint_cap_km2": MAX_FPV_FOOTPRINT_KM2 if capped else np.nan,
                    "l0_generation_twh": float(gross_generation.sum() / 1000.0),
                    "l1_generation_twh": float(l1_generation.sum() / 1000.0),
                    "l1_retention_pct": float(l1_generation.sum() / gross_generation.sum() * 100.0),
                    "l2_partial_generation_twh": (
                        float(np.sum(l2_partial_generation) / 1000.0)
                        if l2_partial_available
                        else np.nan
                    ),
                    "l2_screened_no_dry_generation_twh": (
                        float(np.sum(l2_screened_generation) / 1000.0)
                        if l2_screened_available
                        else np.nan
                    ),
                    "l2_screened_retention_pct": (
                        float(np.sum(l2_screened_generation) / gross_generation.sum() * 100.0)
                        if l2_screened_available
                        else np.nan
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(reports / "coverage_sensitivity.csv", index=False)


def gate_loss_decomposition(result: pd.DataFrame, reports: Path) -> None:
    """Write sequential, auditable losses instead of only endpoint totals."""
    uniform = result["annual_footprint_energy_kwh_m2"] * result["l0_fpv_area_km2"]
    type_specific = (
        result["annual_footprint_energy_kwh_m2"]
        * result["l0_type_specific_fpv_area_km2"]
    )
    masks: list[tuple[str, str, pd.Series]] = [
        ("L0", "30% coverage and 30 km2 cap", pd.Series(True, index=result.index)),
        ("L1-area", "waterbody area >=0.1 km2", result["l1_area_pass"].astype(bool)),
        (
            "L1-ice",
            "plus <=6 months of ice using LI-CCR where available and labelled temperature fallback elsewhere",
            result["l1_pass"].astype(bool),
        ),
    ]
    if result["l2_partial_pass"].notna().all():
        masks.append(
            ("L2-WDPA", "plus no intersection with any WDPA polygon", result["l2_partial_pass"].astype(bool))
        )
    if "l2_ecological_pass" in result and result["l2_ecological_pass"].notna().all():
        masks.append(
            (
                "L2-Ramsar",
                "plus no intersection with published Ramsar polygons",
                result["l2_ecological_pass"].astype(bool),
            )
        )
    if "l2_screened_no_dry_pass" in result and result["l2_screened_no_dry_pass"].notna().all():
        population_mask = (
            result["l2_screened_ramsar_no_dry_pass"].astype(bool)
            if "l2_screened_ramsar_no_dry_pass" in result
            and result["l2_screened_ramsar_no_dry_pass"].notna().all()
            else result["l2_screened_no_dry_pass"].astype(bool)
        )
        masks.append(
            (
                "L2-population",
                "plus within 10 km of a qualifying population centre",
                population_mask,
            )
        )
    if "l2_complete_ramsar_pass" in result and result["l2_complete_ramsar_pass"].notna().all():
        masks.append(
            (
                "L2-dry-up",
                "plus no observed complete dry-up during the study period",
                result["l2_complete_ramsar_pass"].astype(bool),
            )
        )

    rows = []
    previous_uniform = None
    previous_type = None
    baseline_uniform = float(uniform.sum())
    baseline_type = float(type_specific.sum())
    for step, definition, mask in masks:
        uniform_total = float(uniform[mask].sum())
        type_total = float(type_specific[mask].sum())
        rows.append(
            {
                "step": step,
                "definition": definition,
                "eligible_waterbodies": int(mask.sum()),
                "excluded_incremental_waterbodies": (
                    0 if not rows else rows[-1]["eligible_waterbodies"] - int(mask.sum())
                ),
                "uniform_30pct_generation_twh": uniform_total / 1000.0,
                "uniform_incremental_loss_twh": (
                    0.0 if previous_uniform is None else (previous_uniform - uniform_total) / 1000.0
                ),
                "uniform_retention_vs_l0_pct": uniform_total / baseline_uniform * 100.0,
                "type_specific_generation_twh": type_total / 1000.0,
                "type_specific_incremental_loss_twh": (
                    0.0 if previous_type is None else (previous_type - type_total) / 1000.0
                ),
                "type_specific_retention_vs_l0_type_pct": type_total / baseline_type * 100.0,
            }
        )
        previous_uniform = uniform_total
        previous_type = type_total
    pd.DataFrame(rows).to_csv(reports / "gate_loss_decomposition.csv", index=False)


def generation_by_waterbody_type(result: pd.DataFrame, reports: Path) -> None:
    columns = {
        "G0": "gross_generation_gwh",
        "L0": "l0_generation_gwh",
        "L0-10": "l0_10pct_generation_gwh",
        "L0-type": "l0_type_specific_generation_gwh",
        "L1": "l1_generation_gwh",
        "L1-proxy": "l1_temp_proxy_generation_gwh",
        "L1-type": "l1_type_specific_generation_gwh",
        "L2-partial": "l2_partial_generation_gwh",
        "L2-partial-type": "l2_partial_type_generation_gwh",
        "L2-screened": "l2_screened_no_dry_generation_gwh",
        "L2-screened-type": "l2_screened_no_dry_type_generation_gwh",
        "L2-screened-Ramsar": "l2_screened_ramsar_no_dry_generation_gwh",
        "L2-screened-Ramsar-type": "l2_screened_ramsar_no_dry_type_generation_gwh",
        "L2-complete": "l2_complete_generation_gwh",
        "L2-complete-type": "l2_complete_type_generation_gwh",
        "L2-complete-Ramsar": "l2_complete_ramsar_generation_gwh",
        "L2-complete-Ramsar-type": "l2_complete_ramsar_type_generation_gwh",
    }
    rows = []
    for wb_type, group in result.groupby("wb_type", dropna=False):
        for tier, column in columns.items():
            rows.append(
                {
                    "wb_type": wb_type,
                    "tier": tier,
                    "waterbodies": len(group),
                    "generation_twh": (
                        float(group[column].sum() / 1000.0)
                        if group[column].notna().all()
                        else np.nan
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(reports / "generation_by_waterbody_type.csv", index=False)


def run(run_dir: Path, root: Path) -> None:
    data_dir = run_dir / "data"
    reports = run_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    weather_path = data_dir / "weather_power_monthly_2019_2023.parquet"
    assert weather_path.exists(), f"missing rebuilt weather: {weather_path}"
    wb = load_waterbodies(root)
    weather = pd.read_parquet(weather_path)
    assert len(weather) == len(wb) * 12
    annual, monthly = simulate_specific_energy(wb, weather)
    result, summary = add_tiers(run_dir, wb, annual)
    result.to_parquet(data_dir / "waterbody_generation_rebuild.parquet", index=False)
    monthly[
        [
            "wb_id",
            "month",
            "ghi_wm2",
            "temp_c",
            "wind_speed_ms",
            "poa_kwh_m2",
            "cell_temp_c",
            "temperature_factor",
            "specific_yield_kwh_kwp",
            "footprint_energy_kwh_m2",
        ]
    ].to_parquet(data_dir / "waterbody_monthly_simulation_rebuild.parquet", index=False)
    summary.to_csv(reports / "level_summary.csv", index=False)
    legacy_comparison(root, summary, reports)
    coverage_sensitivity(result, reports)
    gate_loss_decomposition(result, reports)
    generation_by_waterbody_type(result, reports)
    completeness = {
        "weather": True,
        "waterbody_count": len(result),
        "wdpa_polygon_gate": bool((run_dir / "data/wdpa_all_waterbody_polygon_flags.parquet").exists()),
        "ramsar_polygon_gate": bool((run_dir / "data/ramsar_waterbody_polygon_flags.parquet").exists()),
        "population_center_gate": bool((run_dir / "data/population_center_10km_flags.parquet").exists()),
        "li_ccr_ice_gate": bool((run_dir / "data/waterbody_ice_flags.parquet").exists()),
        "dry_up_gate": bool((run_dir / "data/waterbody_dryup_flags.parquet").exists()),
        "practical_economic_l3": False,
        "model_notes": [
            "NASA POWER is an independent 2019-2023 cross-check, not an ERA5 replacement claimed as identical.",
            "LI-CCR gap-filled MODIS ice duration is used where at least five 2002-2020 years are valid; remaining waterbodies retain a labelled monthly-temperature fallback.",
            "Published Ramsar polygons form a conservative sensitivity beyond the paper's WDPA gate; sites without published boundaries remain a known gap.",
            "Monthly radiation transposition is lower fidelity than hourly simulation and remains a model limitation.",
            "L3 is NA; F-score is not multiplied into energy as a deployment fraction.",
        ],
    }
    (reports / "rebuild_completeness.json").write_text(
        json.dumps(completeness, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))


def self_test() -> None:
    assert np.isclose(PACKING_DENSITY_KWP_M2, 1 / 10)
    assert np.isclose(np.minimum(400 * 0.30, 30), 30)
    assert MAX_ICE_MONTHS == 6
    lat = np.array([0.0, 30.0, -30.0])
    poa = monthly_poa(np.array([200.0, 200.0, 200.0]), lat, np.array([0.0, 20.0, 20.0]), 6)
    assert np.isfinite(poa).all() and (poa >= 0).all()
    print("self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "self-test"])
    parser.add_argument("--run-dir", type=Path, default=Path("_outputs/v109"))
    args = parser.parse_args()
    if args.stage == "self-test":
        self_test()
        return
    root = project_root()
    run_dir = (root / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir.resolve()
    run(run_dir, root)


if __name__ == "__main__":
    main()
