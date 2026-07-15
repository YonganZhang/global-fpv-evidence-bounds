#!/usr/bin/env python3
"""Rebuild the >=0.1 km2 HydroLAKES source universe as a scope sensitivity.

The manuscript currently declares a >=1.0 km2 natural/controlled-lake scope,
so this run must not be described as a like-for-like correction of that paper
inventory.  It quantifies the consequence of expanding to all native
HydroLAKES records (including its reservoir and controlled-lake types) while
avoiding a second source-dependent deduplication during this audit.  The
current NASA POWER climatology is reused from v109 without new downloads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rebuild_weather_power import complete_grid_matrix, nearest_grid
from recompute_levels import (
    BASE_PERFORMANCE_RATIO,
    DAYS_IN_MONTH,
    FAIMAN_U0,
    FAIMAN_U1,
    MAX_FPV_FOOTPRINT_KM2,
    PACKING_DENSITY_KWP_M2,
    TEMP_COEFF_POWER,
    T_REF_C,
    monthly_poa,
)


MIN_AREA_KM2 = 0.1
PRIMARY_COVERAGE = 0.10
RESERVOIR_SENSITIVITY_COVERAGE = 0.30
MODEL_MAX_SUBZERO_MONTHS = 6


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_dbf_inventory(root: Path) -> pd.DataFrame:
    from dbfread import DBF

    dbf = (
        root
        / "Dataset/data_processed/HydroLAKES/HydroLAKES_polys_v10_shp/"
        "HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.dbf"
    )
    columns: dict[str, list[object]] = {
        "hylak_id": [],
        "lake_type_code": [],
        "lake_area_km2": [],
        "country": [],
        "continent": [],
        "grand_id": [],
        "depth_avg_m": [],
        "elevation_m": [],
        "pour_lon": [],
        "pour_lat": [],
    }
    for index, row in enumerate(DBF(dbf, load=False), start=1):
        columns["hylak_id"].append(int(row["Hylak_id"]))
        columns["lake_type_code"].append(int(row["Lake_type"]))
        columns["lake_area_km2"].append(float(row["Lake_area"]))
        columns["country"].append(str(row["Country"]))
        columns["continent"].append(str(row["Continent"]))
        columns["grand_id"].append(int(row["Grand_id"]))
        columns["depth_avg_m"].append(float(row["Depth_avg"]))
        columns["elevation_m"].append(float(row["Elevation"]))
        columns["pour_lon"].append(float(row["Pour_long"]))
        columns["pour_lat"].append(float(row["Pour_lat"]))
        if index % 250_000 == 0:
            print(f"  read {index:,} HydroLAKES attributes", flush=True)
    frame = pd.DataFrame(columns)
    assert len(frame) == 1_427_688 and frame["hylak_id"].is_unique
    assert frame["lake_area_km2"].min() >= MIN_AREA_KM2
    frame["wb_type"] = frame["lake_type_code"].map(
        {1: "lake", 2: "reservoir", 3: "controlled_lake"}
    )
    frame["hylak_id"] = frame["hylak_id"].astype(np.int32)
    frame["lake_type_code"] = frame["lake_type_code"].astype(np.int8)
    frame["grand_id"] = frame["grand_id"].astype(np.int32)
    for column in [
        "lake_area_km2",
        "depth_avg_m",
        "elevation_m",
        "pour_lon",
        "pour_lat",
    ]:
        frame[column] = frame[column].astype(np.float32)
    return frame


def attach_public_benchmark(inventory: pd.DataFrame, benchmark_path: Path) -> pd.DataFrame:
    columns = [
        "hylak_id",
        "longitude",
        "latitude",
        "median_surface_area_m2",
        "woolway_ice_cover_fraction",
        "woolway_population_distance_km",
        "woolway_protected_area",
        "woolway_total_fpv_output_gwh",
    ]
    benchmark = pd.read_parquet(benchmark_path, columns=columns)
    result = inventory.merge(benchmark, on="hylak_id", how="left", validate="one_to_one")
    result["in_woolway_public_table"] = result["longitude"].notna()
    result["analysis_lon"] = result["longitude"].fillna(result["pour_lon"]).astype(np.float32)
    result["analysis_lat"] = result["latitude"].fillna(result["pour_lat"]).astype(np.float32)
    return result


def load_power_grids(source_run: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    coordinates: dict[str, np.ndarray] = {}
    matrices: dict[str, np.ndarray] = {}
    for short in ["ghi", "dni", "temp", "wind", "pressure"]:
        grid = pd.read_parquet(source_run / f"power_grid_climatology_{short}.parquet")
        coordinates[short], matrices[short] = complete_grid_matrix(grid)
    return coordinates, matrices


def simulate_current_energy(frame: pd.DataFrame, source_run: Path) -> pd.DataFrame:
    coordinates, matrices = load_power_grids(source_run)
    lon = frame["analysis_lon"].to_numpy(float)
    lat = frame["analysis_lat"].to_numpy(float)
    mappings: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for short in matrices:
        print(f"  mapping {len(frame):,} waterbodies to {short} grid", flush=True)
        mappings[short] = nearest_grid(lon, lat, coordinates[short])
    assert np.array_equal(mappings["ghi"][0], mappings["dni"][0])
    assert np.array_equal(mappings["temp"][0], mappings["wind"][0])
    assert np.array_equal(mappings["temp"][0], mappings["pressure"][0])

    annual_specific = np.zeros(len(frame), dtype=np.float64)
    annual_footprint = np.zeros(len(frame), dtype=np.float64)
    subzero_months = np.zeros(len(frame), dtype=np.int8)
    tilt = np.minimum(np.abs(lat), 20.0)
    for month in range(1, 13):
        ghi_wm2 = matrices["ghi"][mappings["ghi"][0], month - 1] * 1000.0 / 24.0
        temp_c = matrices["temp"][mappings["temp"][0], month - 1]
        wind = np.maximum(matrices["wind"][mappings["wind"][0], month - 1], 0.0)
        poa = monthly_poa(ghi_wm2, lat, tilt, month)
        poa_wm2 = poa * 1000.0 / (24 * DAYS_IN_MONTH[month - 1])
        cell_temp = temp_c + poa_wm2 / (FAIMAN_U0 + FAIMAN_U1 * wind)
        temperature_factor = np.clip(
            1.0 + TEMP_COEFF_POWER * (cell_temp - T_REF_C), 0.75, 1.15
        )
        specific = poa * BASE_PERFORMANCE_RATIO * temperature_factor
        annual_specific += specific
        annual_footprint += specific * PACKING_DENSITY_KWP_M2
        subzero_months += temp_c < 0.0
        print(f"  simulated month {month}/12", flush=True)

    frame["annual_specific_yield_kwh_kwp"] = annual_specific.astype(np.float32)
    frame["annual_footprint_energy_kwh_m2"] = annual_footprint.astype(np.float32)
    frame["ice_months_temp_proxy"] = subzero_months
    frame["solar_distance_deg"] = mappings["ghi"][1].astype(np.float32)
    frame["met_distance_deg"] = mappings["temp"][1].astype(np.float32)
    return frame


def add_scenarios(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    energy = frame["annual_footprint_energy_kwh_m2"].to_numpy(float)
    static_area = frame["lake_area_km2"].to_numpy(float)
    footprint_10 = np.minimum(static_area * PRIMARY_COVERAGE, MAX_FPV_FOOTPRINT_KM2)
    footprint_type = np.minimum(
        static_area
        * np.where(
            frame["lake_type_code"].to_numpy() == 2,
            RESERVOIR_SENSITIVITY_COVERAGE,
            PRIMARY_COVERAGE,
        ),
        MAX_FPV_FOOTPRINT_KM2,
    )
    frame["l0_10pct_generation_gwh"] = (energy * footprint_10).astype(np.float32)
    frame["l0_type_generation_gwh"] = (energy * footprint_type).astype(np.float32)
    paper_ice_available = frame["woolway_ice_cover_fraction"].notna()
    frame["ice_pass_method_matched"] = frame["ice_months_temp_proxy"].le(
        MODEL_MAX_SUBZERO_MONTHS
    )
    frame.loc[paper_ice_available, "ice_pass_method_matched"] = frame.loc[
        paper_ice_available, "woolway_ice_cover_fraction"
    ].le(0.5)
    frame["ice_source"] = "nasa_power_monthly_temp_proxy_2019_2023"
    frame.loc[paper_ice_available, "ice_source"] = "woolway_air_temperature_lag_model_1991_2020"
    frame["l1_10pct_generation_gwh"] = np.where(
        frame["ice_pass_method_matched"], frame["l0_10pct_generation_gwh"], 0.0
    ).astype(np.float32)
    frame["l1_type_generation_gwh"] = np.where(
        frame["ice_pass_method_matched"], frame["l0_type_generation_gwh"], 0.0
    ).astype(np.float32)
    known_public_gates = (
        frame["in_woolway_public_table"]
        & frame["woolway_ice_cover_fraction"].le(0.5)
        & ~frame["woolway_protected_area"].eq(1.0)
        & frame["woolway_population_distance_km"].le(10.0)
    )
    frame["published_known_gates_10km_pass"] = known_public_gates
    frame["published_known_gates_10km_generation_gwh"] = np.where(
        known_public_gates, frame["l0_10pct_generation_gwh"], 0.0
    ).astype(np.float32)

    totals = {
        "L0-10-current-full": (
            pd.Series(True, index=frame.index),
            "l0_10pct_generation_gwh",
            "All 1.427688 million HydroLAKES; static area; 10% coverage and 30 km2 cap",
        ),
        "L0-type-current-full": (
            pd.Series(True, index=frame.index),
            "l0_type_generation_gwh",
            "HydroLAKES reservoirs 30%; lakes 10%; 30 km2 cap",
        ),
        "L1-10-current-full": (
            frame["ice_pass_method_matched"],
            "l1_10pct_generation_gwh",
            "L0-10 + Woolway ice output where published and labelled temperature fallback elsewhere",
        ),
        "L1-type-current-full": (
            frame["ice_pass_method_matched"],
            "l1_type_generation_gwh",
            "L0-type + same ice gate",
        ),
        "L2-published-known-gates-10km-current-energy": (
            known_public_gates,
            "published_known_gates_10km_generation_gwh",
            "Public-table ice/protected/population columns only; dry-up not published; not complete L2",
        ),
    }
    rows = []
    for tier, (mask, column, definition) in totals.items():
        rows.append(
            {
                "tier": tier,
                "definition": definition,
                "waterbodies": int(mask.sum()),
                "generation_twh": float(frame[column].sum() / 1000.0),
                "complete": not tier.startswith("L2-"),
            }
        )
    return frame, pd.DataFrame(rows)


def run(run_dir: Path, root: Path, source_run: Path, benchmark_path: Path) -> None:
    data_dir = run_dir / "data"
    reports = run_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    frame = load_dbf_inventory(root)
    frame = attach_public_benchmark(frame, benchmark_path)
    inventory_path = data_dir / "hydrolakes_full_inventory.parquet"
    frame.to_parquet(inventory_path, index=False)
    frame = simulate_current_energy(frame, source_run)
    frame, summary = add_scenarios(frame)
    output_columns = [
        "hylak_id",
        "lake_type_code",
        "wb_type",
        "lake_area_km2",
        "country",
        "continent",
        "analysis_lon",
        "analysis_lat",
        "in_woolway_public_table",
        "woolway_ice_cover_fraction",
        "woolway_population_distance_km",
        "woolway_protected_area",
        "woolway_total_fpv_output_gwh",
        "annual_specific_yield_kwh_kwp",
        "annual_footprint_energy_kwh_m2",
        "ice_months_temp_proxy",
        "ice_pass_method_matched",
        "ice_source",
        "solar_distance_deg",
        "met_distance_deg",
        "l0_10pct_generation_gwh",
        "l0_type_generation_gwh",
        "l1_10pct_generation_gwh",
        "l1_type_generation_gwh",
        "published_known_gates_10km_pass",
        "published_known_gates_10km_generation_gwh",
    ]
    frame[output_columns].to_parquet(
        data_dir / "hydrolakes_full_generation.parquet", index=False
    )
    summary.to_csv(reports / "hydrolakes_full_level_summary.csv", index=False)
    qc = {
        "hydrolakes_waterbodies": len(frame),
        "minimum_area_km2": float(frame["lake_area_km2"].min()),
        "type_counts": {
            str(key): int(value) for key, value in frame["wb_type"].value_counts().items()
        },
        "woolway_public_table_matches": int(frame["in_woolway_public_table"].sum()),
        "woolway_public_table_missing": int((~frame["in_woolway_public_table"]).sum()),
        "student_legacy_hydrolakes_count": 177_795,
        "omitted_by_student_ge1km_filter": len(frame) - 177_795,
        "source_geometry": "Dataset/data_processed/HydroLAKES/.../HydroLAKES_polys_v10.shp",
        "base_universe_policy": "scope sensitivity using HydroLAKES native types; no external GeoDAR/GRanD merge and not like-for-like with the manuscript inventory",
        "l2_status": "incomplete because the public table has no explicit dry-up flag and independent full-universe WDPA/population intersections are not yet rebuilt",
    }
    (reports / "hydrolakes_full_qc.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(qc, indent=2, ensure_ascii=False))


def self_test() -> None:
    assert MIN_AREA_KM2 == 0.1
    assert PRIMARY_COVERAGE == 0.10
    assert RESERVOIR_SENSITIVITY_COVERAGE == 0.30
    print("self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "self-test"])
    parser.add_argument("--run-dir", type=Path, default=Path("_outputs/v111"))
    parser.add_argument(
        "--source-grid-dir",
        type=Path,
        default=Path("_outputs/v109/data"),
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("_outputs/v110/data/woolway_public_lake_info.parquet"),
    )
    args = parser.parse_args()
    if args.stage == "self-test":
        self_test()
        return
    root = project_root()
    run_dir = (root / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir
    source_grid_dir = (
        (root / args.source_grid_dir).resolve()
        if not args.source_grid_dir.is_absolute()
        else args.source_grid_dir
    )
    benchmark_path = (
        (root / args.benchmark_path).resolve()
        if not args.benchmark_path.is_absolute()
        else args.benchmark_path
    )
    run(run_dir, root, source_grid_dir, benchmark_path)


if __name__ == "__main__":
    main()
