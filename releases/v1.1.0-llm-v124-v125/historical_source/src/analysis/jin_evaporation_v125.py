"""Jin et al. (2023)-like reservoir evaporation adaptation.

This module intentionally implements the minimum climate-data adaptation needed by
v125.  It is not a strict reproduction: the local 2019--2023 ERA5 climatology has
Tmean/GHI/wind/pressure but no Tmax/Tmin or vapour pressure, so Tmax=Tmin=Tmean and
Köppen-based RH from the legacy Penman implementation are explicit approximations.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

KOPPEN_RH = {"A": 0.75, "B": 0.35, "C": 0.65, "D": 0.70, "E": 0.75}
MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
ALBEDO = 0.08
SIGMA = 4.903e-9
LAMBDA_INTERCEPT = 2.501
LAMBDA_SLOPE = 0.002361
GAMMA = 0.000665 * 101.325  # Jin MATLAB code: fixed 101.325 kPa
A_MAX_KM2 = 30.0


def saturation_vapour_pressure(t_c: np.ndarray) -> np.ndarray:
    t = np.asarray(t_c, dtype=float)
    # T=-237.3 is outside physically useful input; fail loudly rather than emit inf.
    if np.any(t <= -237.2):
        raise ValueError("Temperature outside Magnus formula domain")
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def estimate_koppen_rh(t_c: np.ndarray, koppen: Iterable[str | float]) -> np.ndarray:
    """Legacy calculate_penman.py seasonal RH approximation, vectorized.

    The legacy function uses one global mean temperature for its seasonal anomaly;
    this deliberately preserves that behavior.  Missing Köppen classes use RH=.60.
    """
    t = np.asarray(t_c, dtype=float)
    k = pd.Series(koppen, dtype="string")
    base = k.map(KOPPEN_RH).fillna(0.60).to_numpy(float)
    t_mean = float(np.nanmean(t))
    rh = base - 0.05 * ((t - t_mean) / 10.0)
    rh = np.where(k.fillna("").to_numpy(dtype=object) == "B", rh - 0.05, rh)
    return np.clip(rh, 0.10, 0.95)


def penman_from_net_radiation(
    t_c: np.ndarray,
    u2_ms: np.ndarray,
    pressure_radiation_unused: np.ndarray,
    rn_mj_m2_day: np.ndarray,
    ea_kpa: np.ndarray,
    *,
    nonnegative: bool = True,
) -> np.ndarray:
    """Jin Eq. (1) using net radiation already expressed in MJ m-2 d-1.

    ``pressure_radiation_unused`` is retained in the signature only to make the
    climate adapter's call explicit; Jin's MATLAB WaterSavings.m uses fixed gamma
    and does not use local pressure.
    """
    del pressure_radiation_unused
    t = np.asarray(t_c, dtype=float)
    u2 = np.asarray(u2_ms, dtype=float)
    rn = np.asarray(rn_mj_m2_day, dtype=float)
    ea = np.asarray(ea_kpa, dtype=float)
    es = saturation_vapour_pressure(t)
    delta = 4098.0 * es / (t + 237.3) ** 2
    lam = LAMBDA_INTERCEPT - LAMBDA_SLOPE * t
    d = np.maximum(es - ea, 0.0)
    e = (delta / (delta + GAMMA)) * rn / lam
    e += (GAMMA / (delta + GAMMA)) * (6.43 * (1.0 + 0.536 * u2) * d / lam)
    e = np.where(t < 0.0, 0.0, e)  # Jin MATLAB behavior
    return np.maximum(e, 0.0) if nonnegative else e


def solar_radiation_clear_sky(lat_deg: np.ndarray, month: np.ndarray) -> np.ndarray:
    """FAO/Jin monthly extraterrestrial radiation with polar-safe geometry."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    m = np.asarray(month, dtype=float)
    j = m * 31.0 - 15.0  # Jin MATLAB code's monthly representative day
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi / 365.0 * j)
    decl = 0.409 * np.sin(2.0 * np.pi / 365.0 * j - 1.39)
    arg = -np.tan(lat) * np.tan(decl)
    omega = np.arccos(np.clip(arg, -1.0, 1.0))
    # Polar night has arg >= 1; midnight sun has arg <= -1.
    omega = np.where(arg >= 1.0, 0.0, omega)
    omega = np.where(arg <= -1.0, np.pi, omega)
    ra = (24.0 * 60.0 / np.pi) * 0.082 * dr * (
        omega * np.sin(lat) * np.sin(decl)
        + np.cos(lat) * np.cos(decl) * np.sin(omega)
    )
    return np.maximum(ra, 0.0)


def jin_monthly_rates(climate: pd.DataFrame, *, rh_shift: float = 0.0) -> pd.DataFrame:
    """Calculate baseline and Jin-like after-covering monthly rates.

    Required columns: wb_id, month, temp_c, wind_speed_ms, pressure_pa, ghi_wm2,
    analysis_lat, koppen_main, c_L1, c_L2, c_L3.
    """
    required = {
        "wb_id", "month", "temp_c", "wind_speed_ms", "pressure_pa", "ghi_wm2",
        "analysis_lat", "koppen_main", "c_L1", "c_L2", "c_L3",
    }
    missing = required - set(climate.columns)
    if missing:
        raise ValueError("Missing climate columns: " + ",".join(sorted(missing)))
    if climate.duplicated(["wb_id", "month"]).any():
        raise ValueError("Duplicate wb_id/month climate rows")
    if not climate.month.between(1, 12).all():
        raise ValueError("Month must be 1..12")
    x = climate.copy()
    t = x.temp_c.to_numpy(float)
    ghi = x.ghi_wm2.to_numpy(float)
    wind = np.maximum(x.wind_speed_ms.to_numpy(float), 0.0)
    pressure = x.pressure_pa.to_numpy(float)
    if not np.isfinite(x[["temp_c", "ghi_wm2", "wind_speed_ms", "pressure_pa", "analysis_lat"]].to_numpy(float)).all():
        raise ValueError("Climate data contain non-finite values")
    if (ghi < 0).any() or (pressure <= 0).any():
        raise ValueError("Invalid radiation or pressure")

    rh = estimate_koppen_rh(t, x.koppen_main) + float(rh_shift)
    rh = np.clip(rh, 0.10, 0.95)
    es = saturation_vapour_pressure(t)
    ea = es * rh
    # Exact Jin MATLAB/FAO conversion from 10 m to 2 m (WaterSavings.m:71).
    u2 = wind * 4.87 / np.log(67.8 * 10.0 - 5.42)
    rs = ghi * 0.0864  # W m-2 daily mean -> MJ m-2 d-1
    ra = solar_radiation_clear_sky(x.analysis_lat.to_numpy(float), x.month.to_numpy(int))
    rso = 0.75 * ra
    ratio = np.divide(rs, rso, out=np.zeros_like(rs), where=rso > 1e-12)
    f_free = np.where(rso > 1e-12, 1.35 * np.minimum(ratio, 1.0) - 0.35, 0.1)
    rnl_factor = (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.0))) * SIGMA * (t + 273.2) ** 4
    rn_free = (1.0 - ALBEDO) * rs - f_free * rnl_factor
    rn_cover = -0.1 * rnl_factor  # Rns_cover=0, n/N=0 -> f_cover=.1
    baseline = penman_from_net_radiation(t, u2, pressure, rn_free, ea)
    result = x[["wb_id", "month"]].copy()
    result["baseline_mm_day"] = baseline
    result["rn_free_mj_m2_day"] = rn_free
    result["rn_cover_mj_m2_day"] = rn_cover
    for level in ("L1", "L2", "L3"):
        c = x[f"c_{level}"].to_numpy(float)
        if not np.isfinite(c).all() or (c < 0).any() or (c > 1).any():
            raise ValueError(f"Invalid {level} retained/full area fraction")
        rn_star = (1.0 - c) * rn_free + c * rn_cover
        covered_free_rate = penman_from_net_radiation(t, u2, pressure, rn_star, ea)
        result[f"after_{level}_mm_day"] = (1.0 - c) * covered_free_rate
        result[f"no_feedback_after_{level}_mm_day"] = (1.0 - c) * baseline
    result["days"] = MONTH_DAYS[result.month.to_numpy(int) - 1]
    return result


def annualize_rates(monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rates to mm/year, retaining only exactly 12 valid months."""
    if monthly.duplicated(["wb_id", "month"]).any():
        raise ValueError("Duplicate monthly rate rows")
    counts = monthly.groupby("wb_id").month.nunique()
    if not counts.eq(12).all():
        raise ValueError("Each wb_id must have 12 valid months")
    m = monthly.copy()
    m["baseline_mm_month"] = m.baseline_mm_day * m.days
    result = m.groupby("wb_id", sort=False)[["baseline_mm_month"]].sum().rename(
        columns={"baseline_mm_month": "annual_evap_mm"}
    ).reset_index()
    for level in ("L1", "L2", "L3"):
        m[f"water_saved_mm_{level}"] = (m.baseline_mm_day - m[f"after_{level}_mm_day"]) * m.days
        m[f"water_saved_no_feedback_mm_{level}"] = (
            m.baseline_mm_day - m[f"no_feedback_after_{level}_mm_day"]
        ) * m.days
        saved = m.groupby("wb_id", sort=False)[
            [f"water_saved_mm_{level}", f"water_saved_no_feedback_mm_{level}"]
        ].sum()
        result = result.merge(saved, left_on="wb_id", right_index=True, how="left", validate="one_to_one")
    return result


def build_climate_frame(
    inventory: pd.DataFrame,
    old_evaporation: pd.DataFrame,
    reservoir_climate: pd.DataFrame,
    lake_climate: pd.DataFrame,
    parent_waterbody: pd.DataFrame,
) -> tuple[pd.DataFrame, set[int]]:
    """Join climate, Köppen, physical area, and fixed parent retained areas."""
    inv = inventory[["wb_id", "area_km2", "analysis_lat"]].copy()
    inv["area_km2"] = pd.to_numeric(inv.area_km2, errors="raise")
    if inv.wb_id.duplicated().any() or (inv.area_km2 <= 0).any():
        raise ValueError("Invalid inventory IDs or full-water area")
    old = old_evaporation[["wb_id", "koppen_main"]].drop_duplicates("wb_id")
    known = set(old.wb_id) & set(inv.wb_id)
    if len(known) != 176478:
        raise ValueError(f"Known evaporation cohort changed: {len(known)}")
    climate = pd.concat([reservoir_climate, lake_climate], ignore_index=True)
    climate = climate.rename(columns={"merge_id": "wb_id"})
    climate = climate[climate.wb_id.isin(known)].copy()
    if climate.duplicated(["wb_id", "month"]).any() or not climate.groupby("wb_id").month.nunique().eq(12).all():
        raise ValueError("Known cohort climate must have 12 unique months")
    fixed = parent_waterbody[["wb_id", "L1_covered_area_km2", "L2_covered_area_km2", "L3_covered_area_km2"]].drop_duplicates("wb_id")
    x = climate.merge(inv, on="wb_id", how="inner", validate="many_to_one").merge(old, on="wb_id", how="left", validate="many_to_one")
    x = x.merge(fixed, on="wb_id", how="inner", validate="many_to_one")
    for level in ("L1", "L2", "L3"):
        c = x[f"{level}_covered_area_km2"] / x.area_km2
        if not np.isfinite(c).all() or (c < 0).any() or (c > 1).any():
            raise ValueError(f"Retained {level} area is outside full-water area")
        x[f"c_{level}"] = c
    if len(x) != len(known) * 12:
        raise ValueError("Climate join does not cover exactly the known cohort")
    return x, known
