#!/usr/bin/env python3
"""Build an independent, all-waterbody monthly weather dataset from NASA POWER.

This is a versioned audit/rebuild path.  It never overwrites the manuscript's
legacy ERA5 files.  POWER is used as an independent source because the legacy
lake ERA5 chain copied climate from distant nearest neighbours for 147,885
lakes.  The output covers every waterbody centroid with actual source-grid
values and records source-cell distances for audit.

Run with the project Python plus temporary scientific dependencies, for example:

    uv run --with xarray --with netcdf4 --with pyarrow \
      python src/rebuild/rebuild_weather_power.py all --run-dir _outputs/v109
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree


POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/regional"
YEARS = (2019, 2023)
TILE_DEGREES = 10
# NASA POWER rate-limits sustained regional requests.  Keep the retry pass
# deliberately conservative; successful files are cached between runs.
MAX_WORKERS = int(os.environ.get("POWER_MAX_WORKERS", "2"))
PARAMETERS = {
    "ghi": {"api": "ALLSKY_SFC_SW_DWN", "units": "kW-hr/m^2/day", "grid": "solar"},
    "dni": {"api": "ALLSKY_SFC_SW_DNI", "units": "kW-hr/m^2/day", "grid": "solar"},
    "temp": {"api": "T2M", "units": "C", "grid": "met"},
    "wind": {"api": "WS10M", "units": "m/s", "grid": "met"},
    "pressure": {"api": "PS", "units": "kPa", "grid": "met"},
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_waterbodies(root: Path) -> pd.DataFrame:
    gpkg = root / "Dataset/data_processed/merged/all_waterbodies.gpkg"
    with sqlite3.connect(gpkg) as con:
        df = pd.read_sql_query(
            "SELECT wb_id, wb_type, centroid_lon, centroid_lat, area_km2 "
            "FROM waterbodies ORDER BY wb_id",
            con,
        )
    assert len(df) == 198_737, f"unexpected waterbody count: {len(df)}"
    assert df["wb_id"].is_unique
    assert df[["centroid_lon", "centroid_lat"]].notna().all().all()
    return df


def base_tiles(wb: pd.DataFrame) -> set[tuple[int, int]]:
    lat = wb["centroid_lat"].clip(-89.999999, 89.999999)
    lon = wb["centroid_lon"].clip(-179.999999, 179.999999)
    lat_i = np.floor((lat + 90) / TILE_DEGREES).astype(int).clip(0, 17)
    lon_i = np.floor((lon + 180) / TILE_DEGREES).astype(int).clip(0, 35)
    return set(zip(lat_i.tolist(), lon_i.tolist()))


def expanded_tiles(wb: pd.DataFrame) -> list[tuple[int, int]]:
    """Add a one-tile halo so the true nearest POWER cell is always present."""
    expanded: set[tuple[int, int]] = set()
    for lat_i, lon_i in base_tiles(wb):
        for dlat in (-1, 0, 1):
            candidate_lat = lat_i + dlat
            if not 0 <= candidate_lat < 18:
                continue
            for dlon in (-1, 0, 1):
                expanded.add((candidate_lat, (lon_i + dlon) % 36))
    return sorted(expanded)


def tile_bounds(tile: tuple[int, int]) -> tuple[int, int, int, int]:
    lat_i, lon_i = tile
    lat_min = -90 + lat_i * TILE_DEGREES
    lon_min = -180 + lon_i * TILE_DEGREES
    return lat_min, lat_min + TILE_DEGREES, lon_min, lon_min + TILE_DEGREES


def is_netcdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1_000:
        return False
    with path.open("rb") as fh:
        signature = fh.read(8)
    return signature.startswith(b"\x89HDF") or signature.startswith(b"CDF")


def download_one(task: tuple[str, dict, tuple[int, int], Path]) -> dict:
    short, spec, tile, path = task
    if is_netcdf(path):
        return {"status": "cached", "parameter": short, "tile": tile, "bytes": path.stat().st_size}

    lat_min, lat_max, lon_min, lon_max = tile_bounds(tile)
    params = {
        "parameters": spec["api"],
        "community": "RE",
        "latitude-min": lat_min,
        "latitude-max": lat_max,
        "longitude-min": lon_min,
        "longitude-max": lon_max,
        "start": YEARS[0],
        "end": YEARS[1],
        "format": "NETCDF",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    last_error = None
    for attempt in range(1, 9):
        try:
            response = requests.get(
                POWER_URL,
                params=params,
                timeout=(15, 120),
                headers={"User-Agent": "FPV-global-rebuild/2026-07 academic audit"},
            )
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "90"))
                raise RuntimeError(f"POWER rate limit; retry_after={retry_after}")
            response.raise_for_status()
            partial.write_bytes(response.content)
            if not is_netcdf(partial):
                preview = response.content[:240].decode("utf-8", errors="replace")
                raise RuntimeError(f"non-NetCDF response: {preview}")
            os.replace(partial, path)
            return {"status": "downloaded", "parameter": short, "tile": tile, "bytes": path.stat().st_size}
        except Exception as exc:  # network retry is intentional
            last_error = f"{type(exc).__name__}: {exc}"
            partial.unlink(missing_ok=True)
            if attempt < 8:
                if "rate limit" in last_error.lower():
                    time.sleep(90)
                else:
                    time.sleep(min(60, 2**attempt))
    return {"status": "failed", "parameter": short, "tile": tile, "error": last_error}


def download(run_dir: Path, wb: pd.DataFrame) -> None:
    raw_dir = run_dir / "raw/power_monthly_2019_2023"
    tiles = expanded_tiles(wb)
    tasks = []
    for short, spec in PARAMETERS.items():
        for lat_i, lon_i in tiles:
            path = raw_dir / short / f"tile_lat{lat_i:02d}_lon{lon_i:02d}.nc"
            tasks.append((short, spec, (lat_i, lon_i), path))

    print(f"POWER download plan: {len(tiles)} tiles x {len(PARAMETERS)} parameters = {len(tasks)} files")
    counts = {"cached": 0, "downloaded": 0, "failed": 0}
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(download_one, task) for task in tasks]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            counts[result["status"]] += 1
            if result["status"] == "failed":
                failures.append(result)
            if index % 50 == 0 or index == len(tasks):
                print(f"  {index}/{len(tasks)} files | {counts}", flush=True)

    manifest = {
        "source": "NASA POWER Monthly and Annual API, POWER v10",
        "endpoint": POWER_URL,
        "years": list(YEARS),
        "tile_degrees": TILE_DEGREES,
        "tile_count": len(tiles),
        "parameter_specs": PARAMETERS,
        "file_counts": counts,
        "failures": failures,
    }
    (raw_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    assert not failures, f"POWER downloads failed: {len(failures)}"


def read_parameter_climatology(files: list[Path], api_name: str) -> pd.DataFrame:
    import xarray as xr

    frames = []
    for index, path in enumerate(files, start=1):
        with xr.open_dataset(path) as ds:
            assert api_name in ds.data_vars, f"{path}: missing {api_name}"
            values = ds[api_name]
            months = (values["time"].values.astype(np.int64) % 100).astype(np.int16)
            valid = (months >= 1) & (months <= 12)
            subset = values.isel(time=np.where(valid)[0]).load()
            subset = subset.assign_coords(month=("time", months[valid]))
            climatology = subset.groupby("month").mean("time", skipna=True)
            frame = climatology.to_dataframe(name="value").reset_index()
            frame = frame[["lat", "lon", "month", "value"]]
            frame["lat"] = frame["lat"].round(6)
            frame["lon"] = frame["lon"].round(6)
            frames.append(frame)
        if index % 100 == 0 or index == len(files):
            print(f"  read {api_name}: {index}/{len(files)} tiles", flush=True)

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.groupby(["lat", "lon", "month"], as_index=False, sort=True)["value"]
        .mean()
        .sort_values(["lat", "lon", "month"])
        .reset_index(drop=True)
    )
    return combined


def complete_grid_matrix(grid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pivot = grid.pivot(index=["lat", "lon"], columns="month", values="value")
    pivot = pivot.reindex(columns=range(1, 13)).dropna(how="any")
    assert len(pivot) > 0 and pivot.shape[1] == 12
    return pivot.index.to_frame(index=False).to_numpy(float), pivot.to_numpy(float)


def nearest_grid(
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    grid_coords_lat_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest neighbour on a regular lon/lat grid with antimeridian wrapping."""
    lat = grid_coords_lat_lon[:, 0]
    lon = grid_coords_lat_lon[:, 1]
    base_index = np.arange(len(grid_coords_lat_lon))
    augmented = np.column_stack(
        [
            np.concatenate([lon - 360, lon, lon + 360]),
            np.concatenate([lat, lat, lat]),
        ]
    )
    augmented_index = np.concatenate([base_index, base_index, base_index])
    tree = cKDTree(augmented)
    distance, found = tree.query(np.column_stack([target_lon, target_lat]), workers=-1)
    source_index = augmented_index[found]
    return source_index, distance, grid_coords_lat_lon[source_index]


def assemble(run_dir: Path, root: Path, wb: pd.DataFrame) -> Path:
    raw_dir = run_dir / "raw/power_monthly_2019_2023"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    grids = {}
    matrices = {}
    coordinates = {}

    for short, spec in PARAMETERS.items():
        files = sorted((raw_dir / short).glob("*.nc"))
        expected = len(expanded_tiles(wb))
        assert len(files) == expected, f"{short}: expected {expected} files, found {len(files)}"
        print(f"Assembling {short} from {len(files)} tiles")
        grid = read_parameter_climatology(files, spec["api"])
        grid_path = data_dir / f"power_grid_climatology_{short}.parquet"
        grid.to_parquet(grid_path, index=False)
        coords, matrix = complete_grid_matrix(grid)
        grids[short] = grid
        coordinates[short] = coords
        matrices[short] = matrix

    target_lon = wb["centroid_lon"].to_numpy(float)
    target_lat = wb["centroid_lat"].to_numpy(float)
    mapping = {}
    for short in PARAMETERS:
        source_index, distance, source_coords = nearest_grid(
            target_lon, target_lat, coordinates[short]
        )
        mapping[short] = (source_index, distance, source_coords)

    # Solar parameters share a 1-degree source grid; meteorological parameters
    # share the POWER/MERRA-2 0.5 x 0.625-degree grid.  Assert this rather than
    # silently combining mismatched cells.
    assert np.array_equal(mapping["ghi"][0], mapping["dni"][0])
    assert np.array_equal(mapping["temp"][0], mapping["wind"][0])
    assert np.array_equal(mapping["temp"][0], mapping["pressure"][0])

    n = len(wb)
    months = np.tile(np.arange(1, 13, dtype=np.int8), n)
    wb_id = np.repeat(wb["wb_id"].to_numpy(np.int32), 12)
    ghi = matrices["ghi"][mapping["ghi"][0], :].reshape(-1) * 1000.0 / 24.0
    dni = matrices["dni"][mapping["dni"][0], :].reshape(-1) * 1000.0 / 24.0
    temp = matrices["temp"][mapping["temp"][0], :].reshape(-1)
    wind = matrices["wind"][mapping["wind"][0], :].reshape(-1)
    pressure = matrices["pressure"][mapping["pressure"][0], :].reshape(-1) * 1000.0

    solar_src = mapping["ghi"][2]
    met_src = mapping["temp"][2]
    weather = pd.DataFrame(
        {
            "wb_id": wb_id,
            "month": months,
            "ghi_wm2": ghi.astype(np.float32),
            "dni_wm2": dni.astype(np.float32),
            "temp_c": temp.astype(np.float32),
            "wind_speed_ms": wind.astype(np.float32),
            "pressure_pa": pressure.astype(np.float32),
            "solar_src_lat": np.repeat(solar_src[:, 0], 12).astype(np.float32),
            "solar_src_lon": np.repeat(solar_src[:, 1], 12).astype(np.float32),
            "solar_distance_deg": np.repeat(mapping["ghi"][1], 12).astype(np.float32),
            "met_src_lat": np.repeat(met_src[:, 0], 12).astype(np.float32),
            "met_src_lon": np.repeat(met_src[:, 1], 12).astype(np.float32),
            "met_distance_deg": np.repeat(mapping["temp"][1], 12).astype(np.float32),
        }
    )
    assert len(weather) == 198_737 * 12
    assert weather[["ghi_wm2", "temp_c", "wind_speed_ms", "pressure_pa"]].notna().all().all()
    assert weather["wb_id"].nunique() == 198_737
    assert weather["month"].nunique() == 12

    weather_path = data_dir / "weather_power_monthly_2019_2023.parquet"
    weather.to_parquet(weather_path, index=False)
    qc = {
        "waterbodies": int(weather["wb_id"].nunique()),
        "rows": int(len(weather)),
        "missing_required_values": int(
            weather[["ghi_wm2", "temp_c", "wind_speed_ms", "pressure_pa"]].isna().sum().sum()
        ),
        "solar_distance_deg": weather.groupby("wb_id")["solar_distance_deg"].first().describe().to_dict(),
        "met_distance_deg": weather.groupby("wb_id")["met_distance_deg"].first().describe().to_dict(),
        "monthly_value_ranges": {
            col: [float(weather[col].min()), float(weather[col].max())]
            for col in ["ghi_wm2", "dni_wm2", "temp_c", "wind_speed_ms", "pressure_pa"]
        },
    }
    (data_dir / "weather_qc.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    compare_to_exact_gee(root, run_dir, weather)
    print(f"Saved all-waterbody POWER climatology: {weather_path}")
    return weather_path


def compare_to_exact_gee(root: Path, run_dir: Path, power: pd.DataFrame) -> None:
    era5_dir = root / "Dataset/data_processed/ERA5"
    reservoir = pd.read_csv(era5_dir / "era5_monthly_climatology.csv").rename(
        columns={"merge_id": "wb_id"}
    )
    old_lakes = pd.read_csv(era5_dir / "era5_lakes_monthly_climatology_old_22253.csv.bak")
    exact = pd.concat([reservoir, old_lakes], ignore_index=True)
    exact = exact.drop_duplicates(["wb_id", "month"])
    columns = ["ghi_wm2", "temp_c", "wind_speed_ms", "pressure_pa"]
    merged = exact[["wb_id", "month"] + columns].merge(
        power[["wb_id", "month"] + columns],
        on=["wb_id", "month"],
        suffixes=("_gee", "_power"),
        validate="one_to_one",
    )
    records = []
    for column in columns:
        x = merged[f"{column}_gee"].to_numpy(float)
        y = merged[f"{column}_power"].to_numpy(float)
        records.append(
            {
                "variable": column,
                "n_monthly_pairs": len(x),
                "mean_gee": float(np.mean(x)),
                "mean_power": float(np.mean(y)),
                "bias_power_minus_gee": float(np.mean(y - x)),
                "mae": float(np.mean(np.abs(y - x))),
                "rmse": float(np.sqrt(np.mean((y - x) ** 2))),
                "pearson_r": float(np.corrcoef(x, y)[0, 1]),
            }
        )
    pd.DataFrame(records).to_csv(run_dir / "reports/weather_vs_exact_gee.csv", index=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(run_dir: Path) -> None:
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted((run_dir / "data").glob("*")):
        if path.is_file():
            files.append({"path": str(path.relative_to(run_dir)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    (reports / "weather_artifact_manifest.json").write_text(
        json.dumps({"files": files}, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def self_test() -> None:
    sample = pd.DataFrame(
        {
            "centroid_lat": [0.1, 89.0, -89.0, 20.0],
            "centroid_lon": [179.9, -179.9, 0.0, 30.0],
        }
    )
    tiles = expanded_tiles(sample)
    assert len(tiles) > 0
    assert all(0 <= lat < 18 and 0 <= lon < 36 for lat, lon in tiles)
    assert tile_bounds((0, 0)) == (-90, -80, -180, -170)
    print("self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["download", "assemble", "all", "self-test"])
    parser.add_argument("--run-dir", type=Path, default=Path("_outputs/v109"))
    args = parser.parse_args()
    if args.stage == "self-test":
        self_test()
        return

    root = project_root()
    run_dir = (root / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    wb = load_waterbodies(root)
    if args.stage in {"download", "all"}:
        download(run_dir, wb)
    if args.stage in {"assemble", "all"}:
        assemble(run_dir, root, wb)
        write_provenance(run_dir)


if __name__ == "__main__":
    main()
