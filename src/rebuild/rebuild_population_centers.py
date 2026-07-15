#!/usr/bin/env python3
"""Derive the 10 km population-centre gate from LandScan 2024.

The population-centre definition follows the explicit thresholds reported by
Woolway et al. (Nature Water, 2024): a contiguous region with population at
least 1,000 and density at least 400 people per km2.  This project has LandScan
2024 population counts at 30 arc-second resolution, so density is calculated
using latitude-dependent cell area.

Proximity is evaluated against densified waterbody polygon boundaries on a
sphere, not against waterbody centroids.  The 10.95 km numerical threshold is
10 km plus the maximum half-cell and boundary-sampling allowances, because
population is stored as a cell total rather than an exact within-cell point.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree


EARTH_RADIUS_KM = 6371.0088
DENSITY_THRESHOLD = 400.0
POPULATION_THRESHOLD = 1_000.0
DISTANCE_THRESHOLD_KM = 10.95
MAX_BOUNDARY_SEGMENT_DEG = 0.005


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def spherical_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    cos_lat = np.cos(lat)
    return np.column_stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)])


def chord_to_km(chord: np.ndarray) -> np.ndarray:
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


def derive_population_cells(root: Path, run_dir: Path) -> Path:
    import rasterio

    output = run_dir / "data/population_centers_landscan2024_cells.parquet"
    reports = run_dir / "reports"
    output.parent.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"Using cached population-centre cells: {output}")
        return output

    source = root / "Dataset/data_processed/LandSCAN/landscan-global-2024.tif"
    print("Reading LandScan 2024 global raster (this step is memory intensive)")
    with rasterio.open(source) as ds:
        population = ds.read(1)
        transform = ds.transform
        nodata = ds.nodata
        height, width = ds.shape
    if nodata is not None:
        population[population == nodata] = 0
    population[population < 0] = 0

    dlat = abs(float(transform.e))
    dlon = abs(float(transform.a))
    row = np.arange(height)
    lat_top = float(transform.f) - row * dlat
    lat_bottom = lat_top - dlat
    cell_area_km2 = (
        EARTH_RADIUS_KM**2
        * np.radians(dlon)
        * np.abs(np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bottom)))
    )
    threshold_count = DENSITY_THRESHOLD * cell_area_km2
    dense = population >= threshold_count[:, None]
    print(f"High-density LandScan cells: {int(dense.sum()):,}")

    labels, component_count = ndimage.label(dense, structure=np.ones((3, 3), dtype=np.uint8))
    del dense
    print(f"Connected high-density regions: {component_count:,}")
    component_population = np.bincount(
        labels.ravel(), weights=population.ravel(), minlength=component_count + 1
    )
    keep = component_population >= POPULATION_THRESHOLD
    keep[0] = False
    rows, cols = np.nonzero(keep[labels])
    retained_component_count = int(keep.sum())
    del labels, keep, population
    gc.collect()

    lon = float(transform.c) + (cols + 0.5) * dlon
    lat = float(transform.f) - (rows + 0.5) * dlat
    cells = pd.DataFrame(
        {
            "lon": lon.astype(np.float32),
            "lat": lat.astype(np.float32),
        }
    )
    cells.to_parquet(output, index=False)
    qc = {
        "source": str(source.relative_to(root)),
        "raster_shape": [height, width],
        "density_threshold_people_km2": DENSITY_THRESHOLD,
        "component_population_threshold": POPULATION_THRESHOLD,
        "connectivity": 8,
        "high_density_components": component_count,
        "retained_population_centres": retained_component_count,
        "retained_population_centre_cells": len(cells),
    }
    (reports / "population_center_derivation_qc.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Saved {len(cells):,} cells in {retained_component_count:,} population centres: {output}"
    )
    return output


def proximity_flags(root: Path, run_dir: Path, cell_path: Path) -> Path:
    import geopandas as gpd
    import shapely

    cells = pd.read_parquet(cell_path)
    tree = cKDTree(spherical_xyz(cells["lon"].to_numpy(), cells["lat"].to_numpy()))
    del cells
    gpkg = root / "Dataset/data_processed/merged/all_waterbodies.gpkg"
    print("Loading and densifying waterbody boundaries")
    wb = gpd.read_file(gpkg, layer="waterbodies", columns=["wb_id"])
    assert len(wb) == 198_737 and wb["wb_id"].is_unique
    if wb.crs is not None and wb.crs.to_epsg() != 4326:
        wb = wb.to_crs(4326)
    invalid = ~wb.geometry.is_valid
    if invalid.any():
        wb.loc[invalid, "geometry"] = wb.loc[invalid, "geometry"].make_valid()

    minimum_distance = np.full(len(wb), np.inf, dtype=np.float32)
    batch_size = 1_000
    for start in range(0, len(wb), batch_size):
        stop = min(start + batch_size, len(wb))
        geoms = np.asarray(wb.geometry.iloc[start:stop].values, dtype=object)
        segmented = shapely.segmentize(geoms, max_segment_length=MAX_BOUNDARY_SEGMENT_DEG)
        coords, geometry_index = shapely.get_coordinates(segmented, return_index=True)
        # Representative points protect against unusual polygons for which the
        # closest centre lies inside rather than near the exterior boundary.
        reps = shapely.get_coordinates(shapely.point_on_surface(geoms))
        if len(reps):
            coords = np.vstack([coords, reps])
            geometry_index = np.concatenate(
                [geometry_index, np.arange(len(geoms), dtype=geometry_index.dtype)]
            )
        chord, _ = tree.query(spherical_xyz(coords[:, 0], coords[:, 1]), workers=-1)
        distance = chord_to_km(chord)
        batch_min = np.full(len(geoms), np.inf)
        np.minimum.at(batch_min, geometry_index, distance)
        minimum_distance[start:stop] = batch_min.astype(np.float32)
        if stop % 10_000 == 0 or stop == len(wb):
            print(
                f"  {stop:,}/{len(wb):,} waterbodies | within 10 km so far: "
                f"{int((minimum_distance[:stop] <= DISTANCE_THRESHOLD_KM).sum()):,}",
                flush=True,
            )

    flags = pd.DataFrame(
        {
            "wb_id": wb["wb_id"].to_numpy(np.int32),
            "population_center_distance_km": minimum_distance,
            "within_10km_population_center": minimum_distance <= DISTANCE_THRESHOLD_KM,
        }
    ).sort_values("wb_id")
    output = run_dir / "data/population_center_10km_flags.parquet"
    flags.to_parquet(output, index=False)
    qc = {
        "waterbodies": len(flags),
        "within_10km": int(flags["within_10km_population_center"].sum()),
        "outside_10km": int((~flags["within_10km_population_center"]).sum()),
        "distance_threshold_km_including_half_cell_allowance": DISTANCE_THRESHOLD_KM,
        "boundary_max_segment_degrees": MAX_BOUNDARY_SEGMENT_DEG,
        "distance_quantiles_km": {
            str(q): float(flags["population_center_distance_km"].quantile(q))
            for q in [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
        },
    }
    (run_dir / "reports/population_proximity_qc.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved population-centre proximity flags: {output}")
    return output


def run(root: Path, run_dir: Path) -> None:
    cells = derive_population_cells(root, run_dir)
    proximity_flags(root, run_dir, cells)


def self_test() -> None:
    xyz = spherical_xyz(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
    chord = np.linalg.norm(xyz[0] - xyz[1])
    assert abs(chord_to_km(np.array([chord]))[0] - 111.2) < 0.5
    assert DISTANCE_THRESHOLD_KM > 10
    print("self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "cells", "proximity", "self-test"])
    parser.add_argument("--run-dir", type=Path, default=Path("_outputs/v109"))
    args = parser.parse_args()
    if args.stage == "self-test":
        self_test()
        return
    root = project_root()
    run_dir = (root / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir.resolve()
    if args.stage in {"run", "cells"}:
        cells = derive_population_cells(root, run_dir)
    else:
        cells = run_dir / "data/population_centers_landscan2024_cells.parquet"
        assert cells.exists()
    if args.stage in {"run", "proximity"}:
        proximity_flags(root, run_dir, cells)


if __name__ == "__main__":
    main()
