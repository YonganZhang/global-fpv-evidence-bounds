"""Run the bounded v125 Jin-like water-saving adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Permit both ``python -m src.analysis.run_jin_water_v125`` and direct execution.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.jin_evaporation_v125 import (
    A_MAX_KM2,
    annualize_rates,
    build_climate_frame,
    jin_monthly_rates,
)

PARENT = ROOT / "_outputs/v124/global/lean/weighted_levels/primaryrecovery01/present_benefits"
INVENTORY = ROOT / "_outputs/v117/data/fpv_reference_inventory_v117.parquet"
RES_CLIMATE = ROOT / "Dataset/data_processed/ERA5/era5_monthly_climatology.csv"
LAKE_CLIMATE = ROOT / "Dataset/data_processed/ERA5/era5_lakes_monthly_climatology.csv"
OLD_EVAP = ROOT / "Dataset/data_processed/evaporation/evaporation_savings.csv"
OUT = ROOT / "_outputs/v125/jin_water"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_climate(path: Path, key: str) -> pd.DataFrame:
    usecols = [key, "month", "ghi_wm2", "temp_c", "wind_speed_ms", "pressure_pa"]
    return pd.read_csv(path, usecols=usecols).rename(columns={key: "wb_id"})


def _replace_country_water(parent: pd.DataFrame, waterbody: pd.DataFrame) -> pd.DataFrame:
    out = parent.copy()
    keys = ["protection", "endpoint", "missing_case", "ISO"]
    grouped = waterbody.groupby(keys, dropna=False)[
        ["L1_water_saving_km3", "L2_water_saving_km3", "L3_water_saving_km3"]
    ].sum(min_count=1).reset_index()
    out = out.drop(columns=["L1_water_saving_km3", "L2_water_saving_km3", "L3_water_saving_km3"])
    out = out.merge(grouped, on=keys, how="left", validate="one_to_one", sort=False)
    return out[parent.columns]


def _replace_global_water(parent: pd.DataFrame, waterbody: pd.DataFrame) -> pd.DataFrame:
    out = parent.copy()
    keys = ["protection", "endpoint", "missing_case"]
    grouped = waterbody.groupby(keys, dropna=False)[
        ["L1_water_saving_km3", "L2_water_saving_km3", "L3_water_saving_km3"]
    ].sum(min_count=1).reset_index()
    grouped = grouped.rename(columns={f"{level}_water_saving_km3": f"{level}_water_saving_known_subset_km3" for level in ("L1", "L2", "L3")})
    out = out.drop(columns=[f"{level}_water_saving_known_subset_km3" for level in ("L1", "L2", "L3")])
    out = out.merge(grouped, on=keys, how="left", validate="one_to_one", sort=False)
    return out[parent.columns]


def _sensitivity_rows(
    parent_waterbody: pd.DataFrame,
    scenario_water: dict[str, pd.DataFrame],
    no_feedback: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["protection", "endpoint", "missing_case"]
    configs = list(parent_waterbody[keys].drop_duplicates().itertuples(index=False, name=None))
    rows = []
    for label, water in {"legacy_fixed_46": parent_waterbody, **scenario_water}.items():
        for config in configs:
            mask = np.ones(len(water), dtype=bool)
            for key, value in zip(keys, config):
                mask &= water[key].eq(value).to_numpy()
            for level in ("L1", "L2", "L3"):
                rows.append({"scenario": label, **dict(zip(keys, config)), "level": level,
                             "water_saved_km3": float(water.loc[mask, f"{level}_water_saving_km3"].sum(min_count=1))})
    for config in configs:
        mask = np.ones(len(no_feedback), dtype=bool)
        for key, value in zip(keys, config):
            mask &= no_feedback[key].eq(value).to_numpy()
        for level in ("L1", "L2", "L3"):
            rows.append({"scenario": "no_feedback", **dict(zip(keys, config)), "level": level,
                         "water_saved_km3": float(no_feedback.loc[mask, f"{level}_water_saving_km3"].sum(min_count=1))})
    result = pd.DataFrame(rows)
    legacy = result.loc[result.scenario.eq("legacy_fixed_46"), keys + ["level", "water_saved_km3"]].rename(
        columns={"water_saved_km3": "legacy_water_saved_km3"}
    )
    result = result.merge(legacy, on=keys + ["level"], how="left", validate="many_to_one")
    result["delta_vs_legacy_km3"] = result.water_saved_km3 - result.legacy_water_saved_km3
    result["pct_delta_vs_legacy"] = result.delta_vs_legacy_km3 / result.legacy_water_saved_km3 * 100.0
    return result


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    parent_water_path = PARENT / "waterbody_benefits.parquet"
    parent_country_path = PARENT / "country_benefits.csv"
    parent_global_path = PARENT / "global_benefits.csv"
    parent_water = pd.read_parquet(parent_water_path)
    parent_country = pd.read_csv(parent_country_path)
    parent_global = pd.read_csv(parent_global_path)
    inventory = pd.read_parquet(INVENTORY, columns=["wb_id", "area_km2", "centroid_lat"])
    inventory = inventory.rename(columns={"centroid_lat": "analysis_lat"})
    old = pd.read_csv(OLD_EVAP, usecols=["wb_id", "koppen_main"])
    reservoir = _read_climate(RES_CLIMATE, "merge_id")
    lakes = _read_climate(LAKE_CLIMATE, "wb_id")
    climate_frame, known = build_climate_frame(inventory, old, reservoir, lakes, parent_water)
    if len(parent_water) != 199976 * 8 or parent_water.wb_id.nunique() != 199976:
        raise ValueError("Unexpected v124 parent population")
    if parent_water.duplicated(["wb_id", "protection", "endpoint", "missing_case"]).any():
        raise ValueError("Duplicate parent scenario rows")

    inventory_indexed = inventory.set_index("wb_id")
    config_keys = ["protection", "endpoint", "missing_case"]
    configs = list(parent_water[config_keys].drop_duplicates().itertuples(index=False, name=None))
    scenario_water = {label: parent_water.copy() for label in ("jin_like", "rh_minus_0.10", "rh_plus_0.10")}
    no_feedback = parent_water.copy()
    for config in configs:
        mask_parent = np.ones(len(parent_water), dtype=bool)
        for key, value in zip(config_keys, config):
            mask_parent &= parent_water[key].eq(value).to_numpy()
        fixed = parent_water.loc[mask_parent, ["wb_id", "L1_covered_area_km2", "L2_covered_area_km2", "L3_covered_area_km2"]]
        frame = climate_frame.drop(columns=["c_L1", "c_L2", "c_L3", "L1_covered_area_km2", "L2_covered_area_km2", "L3_covered_area_km2"]).merge(fixed, on="wb_id", how="inner", validate="many_to_one")
        for level in ("L1", "L2", "L3"):
            c = frame[f"{level}_covered_area_km2"] / frame.area_km2
            if not np.isfinite(c).all() or (c < 0).any() or (c > 1).any():
                raise ValueError(f"{config}: retained {level} area outside full-water area")
            frame[f"c_{level}"] = c
        for label, shift in [("jin_like", 0.0), ("rh_minus_0.10", -0.10), ("rh_plus_0.10", 0.10)]:
            annual = annualize_rates(jin_monthly_rates(frame, rh_shift=shift)).set_index("wb_id")
            for level in ("L1", "L2", "L3"):
                vals = annual[f"water_saved_mm_{level}"] * inventory_indexed.area_km2.reindex(annual.index).to_numpy(float) * 1e-6
                mapped = parent_water.wb_id.map(pd.Series(vals.to_numpy(), index=annual.index))
                scenario_water[label].loc[mask_parent, f"{level}_water_saving_km3"] = mapped.loc[mask_parent].to_numpy()
                if label == "jin_like":
                    vals_nf = annual[f"water_saved_no_feedback_mm_{level}"] * inventory_indexed.area_km2.reindex(annual.index).to_numpy(float) * 1e-6
                    mapped_nf = parent_water.wb_id.map(pd.Series(vals_nf.to_numpy(), index=annual.index))
                    no_feedback.loc[mask_parent, f"{level}_water_saving_km3"] = mapped_nf.loc[mask_parent].to_numpy()
            if set(annual.index) != known:
                raise ValueError("Jin annual cohort changed")

    result_water = scenario_water["jin_like"]
    result_country = _replace_country_water(parent_country, result_water)
    result_global = _replace_global_water(parent_global, result_water)
    sensitivity = _sensitivity_rows(parent_water, scenario_water, no_feedback)

    fixed_water_columns = [c for c in parent_water.columns if c not in {"L1_water_saving_km3", "L2_water_saving_km3", "L3_water_saving_km3"}]
    for c in fixed_water_columns:
        if not result_water[c].equals(parent_water[c]):
            raise ValueError("Non-waterbody column changed: " + c)
    for c in parent_country.columns:
        if c not in {"L1_water_saving_km3", "L2_water_saving_km3", "L3_water_saving_km3"} and not result_country[c].equals(parent_country[c]):
            raise ValueError("Non-country-water column changed: " + c)
    for c in parent_global.columns:
        if c not in {"L1_water_saving_known_subset_km3", "L2_water_saving_known_subset_km3", "L3_water_saving_known_subset_km3"} and not result_global[c].equals(parent_global[c]):
            raise ValueError("Non-global-water column changed: " + c)

    result_water.to_parquet(OUT / "waterbody_benefits.parquet", index=False)
    result_country.to_csv(OUT / "country_benefits.csv", index=False)
    result_global.to_csv(OUT / "global_benefits.csv", index=False)
    sensitivity.to_csv(OUT / "global_sensitivity.csv", index=False)

    sources = [parent_water_path, parent_country_path, parent_global_path, INVENTORY, RES_CLIMATE, LAKE_CLIMATE, OLD_EVAP]
    code = [Path(__file__), Path(__file__).with_name("jin_evaporation_v125.py")]
    contract = {
        "status": "conditional_jin_like_adaptation_not_strict_reproduction",
        "method": "Eafter_total=(1-c)*Penman(Rn_star), covered_evaporation=0; Rn_star=(1-c)*Rn_free+c*Rn_cover",
        "coverage": "c=retained_area_km2/full_water_surface_area_km2; retained areas and all generation/demand fields copied from v124",
        "constants": {"albedo": 0.08, "gamma_kpa_per_c": 0.000665 * 101.325, "lambda": "2.501-0.002361*T", "max_area_km2": A_MAX_KM2},
        "approximations": ["Tmax=Tmin=Tmean", "Koppen-based RH from legacy calculate_penman.py", "ERA5 GHI replaces original CERES Rs", "fixed gamma as in Jin MATLAB"],
        "known_evaporation_cohort": len(known),
        "jin_reference": {
            "repository": "https://github.com/YubinJin98/Floating-solar-power",
            "commit": "df38ec94e87ca896a4c86c49fe8e659c8ceabb37",
            "WaterSavings.m_sha256": sha256(ROOT / "_refs/_literature/2026-09-10_jin_evap/Floating-solar-power/WaterSavings.m"),
        },
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in code},
    }
    (OUT / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")

    checks = {
        "waterbody_rows": len(result_water) == 199976 * 8,
        "country_rows": len(result_country) == len(parent_country),
        "global_rows": len(result_global) == len(parent_global) == 8,
        "known_cohort": len(known) == 176478,
        "twelve_unique_months": bool(climate_frame.groupby("wb_id").month.nunique().eq(12).all()),
        "finite_new_water": bool(np.isfinite(result_water.loc[result_water.wb_id.isin(known), ["L1_water_saving_km3", "L2_water_saving_km3", "L3_water_saving_km3"]].to_numpy(float)).all()),
        "generation_unchanged": bool(np.array_equal(result_water[["L1_generation_twh", "L2_generation_twh", "L3_generation_twh"]].to_numpy(), parent_water[["L1_generation_twh", "L2_generation_twh", "L3_generation_twh"]].to_numpy())),
        "demand_unchanged": bool(np.allclose(result_country[["L1_demand_ratio_pct", "L2_demand_ratio_pct", "L3_demand_ratio_pct"]].to_numpy(), parent_country[["L1_demand_ratio_pct", "L2_demand_ratio_pct", "L3_demand_ratio_pct"]].to_numpy(), equal_nan=True)),
    }
    validation = {
        "status": "conditional_jin_like_adaptation_not_strict_reproduction",
        "all_checks_passed": bool(all(checks.values())),
        "checks": checks,
        "known_evaporation_waterbodies": len(known),
        "scenario_rows": len(result_global),
        "output_sha256": {p.name: sha256(p) for p in [OUT / "waterbody_benefits.parquet", OUT / "country_benefits.csv", OUT / "global_benefits.csv", OUT / "global_sensitivity.csv"]},
        "global_sensitivity": sensitivity.to_dict(orient="records"),
        "note": "Jin-like mathematical adaptation, not strict reproduction of Jin et al. (2023); no 46% multiplier is applied.",
    }
    if not validation["all_checks_passed"]:
        raise RuntimeError("v125 validation failed: " + json.dumps(checks, ensure_ascii=False))
    (OUT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return validation


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
