#!/usr/bin/env python3
"""Rebuild protected-area flags for every waterbody polygon.

Source: UNEP-WCMC June 2026 WDPA ArcGIS Feature Service.  The legacy project
checked only reservoir centroids.  This rebuild downloads polygon features and
uses polygon-polygon intersection for all 198,737 waterbodies.

The WDPA licence permits academic/non-commercial analysis with attribution but
restricts redistribution.  Raw WDPA pages therefore remain run-local and are
not intended for publication or repository commits.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


SERVICE = (
    "https://data-gis.unep-wcmc.org/server/rest/services/ProtectedSites/"
    "The_World_Database_of_Protected_Areas/FeatureServer/1"
)
STRICT_CATEGORIES = {"Ia", "Ib", "II", "III", "IV"}
BATCH_SIZE = 1_000
MAX_WORKERS = 2


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def fetch_object_ids(raw_dir: Path) -> list[int]:
    path = raw_dir / "object_ids.json"
    if path.exists():
        return [int(x) for x in json.loads(path.read_text(encoding="utf-8"))["objectIds"]]
    response = requests.get(
        f"{SERVICE}/query",
        params={"where": "1=1", "returnIdsOnly": "true", "f": "json"},
        timeout=(15, 120),
    )
    response.raise_for_status()
    payload = response.json()
    ids = sorted(int(x) for x in payload["objectIds"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"objectIds": ids}), encoding="utf-8")
    return ids


def valid_geojson(path: Path, expected_minimum: int = 1) -> bool:
    if not path.exists() or path.stat().st_size < 200:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("type") == "FeatureCollection" and len(payload.get("features", [])) >= expected_minimum
    except Exception:
        return False


def download_batch(task: tuple[int, list[int], Path]) -> dict:
    batch_number, object_ids, path = task
    if valid_geojson(path):
        return {"status": "cached", "batch": batch_number, "features": len(object_ids), "bytes": path.stat().st_size}
    data = {
        "objectIds": ",".join(str(x) for x in object_ids),
        "outFields": "objectid,site_id,iucn_cat,realm,status,inlnd_wtrs",
        "returnGeometry": "true",
        "outSR": "4326",
        # About 100 m at the equator.  This is explicitly recorded as a
        # global-screening simplification, not a site permitting boundary.
        "maxAllowableOffset": "0.001",
        "geometryPrecision": "5",
        "f": "geojson",
    }
    partial = path.with_suffix(".geojson.part")
    last_error = None
    for attempt in range(1, 7):
        try:
            response = requests.post(SERVICE + "/query", data=data, timeout=(20, 180))
            response.raise_for_status()
            partial.write_bytes(response.content)
            if not valid_geojson(partial):
                preview = response.content[:300].decode("utf-8", errors="replace")
                raise RuntimeError(f"invalid FeatureServer response: {preview}")
            os.replace(partial, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            return {
                "status": "downloaded",
                "batch": batch_number,
                "features": len(payload["features"]),
                "bytes": path.stat().st_size,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            partial.unlink(missing_ok=True)
            if attempt < 6:
                time.sleep(min(60, 2**attempt))
    return {"status": "failed", "batch": batch_number, "error": last_error}


def download(run_dir: Path) -> None:
    raw_dir = run_dir / "raw/wdpa_polygons_2026_06"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ids = fetch_object_ids(raw_dir)
    tasks = []
    for batch_number, start in enumerate(range(0, len(ids), BATCH_SIZE)):
        batch_ids = ids[start : start + BATCH_SIZE]
        path = raw_dir / f"page_{batch_number:04d}.geojson"
        tasks.append((batch_number, batch_ids, path))
    print(f"WDPA polygon download plan: {len(ids):,} features in {len(tasks)} pages")
    counts = {"cached": 0, "downloaded": 0, "failed": 0}
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(download_batch, task) for task in tasks]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            counts[result["status"]] += 1
            if result["status"] == "failed":
                failures.append(result)
            if index % 10 == 0 or index == len(tasks):
                print(f"  {index}/{len(tasks)} pages | {counts}", flush=True)
    manifest = {
        "source": "UNEP-WCMC and IUCN, WDPA, June 2026",
        "service": SERVICE,
        "license": "https://www.unep-wcmc.org/wdpa-data-license",
        "polygon_features": len(ids),
        "batch_size": BATCH_SIZE,
        "geometry_simplification_degrees": 0.001,
        "file_counts": counts,
        "failures": failures,
        "known_gap": "7,747 point-only WDPA records are not buffered in this polygon intersection pass",
    }
    (raw_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    assert not failures, f"WDPA downloads failed: {len(failures)}"


def intersect(run_dir: Path, root: Path) -> Path:
    import geopandas as gpd

    raw_dir = run_dir / "raw/wdpa_polygons_2026_06"
    pages = sorted(raw_dir.glob("page_*.geojson"))
    ids = fetch_object_ids(raw_dir)
    assert len(pages) == math.ceil(len(ids) / BATCH_SIZE)

    gpkg = root / "Dataset/data_processed/merged/all_waterbodies.gpkg"
    print("Loading 198,737 waterbody polygons and building spatial index")
    wb = gpd.read_file(gpkg, layer="waterbodies", columns=["wb_id"])
    assert len(wb) == 198_737 and wb["wb_id"].is_unique
    if wb.crs is None:
        wb = wb.set_crs(4326)
    elif wb.crs.to_epsg() != 4326:
        wb = wb.to_crs(4326)
    invalid_count = int((~wb.geometry.is_valid).sum())
    if invalid_count:
        print(f"Repairing {invalid_count:,} invalid waterbody geometries")
        wb.geometry = wb.geometry.make_valid()
    spatial_index = wb.sindex

    any_hit = np.zeros(len(wb), dtype=bool)
    strict_hit = np.zeros(len(wb), dtype=bool)
    feature_count = 0
    strict_feature_count = 0
    for index, page in enumerate(pages, start=1):
        pa = gpd.read_file(page)
        feature_count += len(pa)
        if pa.crs is None:
            pa = pa.set_crs(4326)
        elif pa.crs.to_epsg() != 4326:
            pa = pa.to_crs(4326)
        invalid_pa = ~pa.geometry.is_valid
        if invalid_pa.any():
            pa.loc[invalid_pa, "geometry"] = pa.loc[invalid_pa, "geometry"].make_valid()
        pa = pa[pa.geometry.notna() & ~pa.geometry.is_empty]
        if len(pa):
            pairs = spatial_index.query(pa.geometry, predicate="intersects")
            if pairs.size:
                any_hit[np.unique(pairs[1])] = True
        strict = pa[pa["iucn_cat"].isin(STRICT_CATEGORIES)]
        strict_feature_count += len(strict)
        if len(strict):
            pairs = spatial_index.query(strict.geometry, predicate="intersects")
            if pairs.size:
                strict_hit[np.unique(pairs[1])] = True
        if index % 10 == 0 or index == len(pages):
            print(
                f"  {index}/{len(pages)} pages | any={any_hit.sum():,}, strict={strict_hit.sum():,}",
                flush=True,
            )

    assert feature_count == len(ids), (feature_count, len(ids))
    flags = pd.DataFrame(
        {
            "wb_id": wb["wb_id"].to_numpy(np.int32),
            "in_wdpa_polygon_any": any_hit,
            "in_wdpa_polygon_strict_i_iv": strict_hit,
        }
    ).sort_values("wb_id")
    data_dir = run_dir / "data"
    reports = run_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    output = data_dir / "wdpa_all_waterbody_polygon_flags.parquet"
    flags.to_parquet(output, index=False)

    legacy = pd.read_csv(root / "Dataset/data_processed/exclusion/wdpa_reservoir_flags.csv")
    legacy = legacy[["merge_id", "in_wdpa", "in_wdpa_strict"]].rename(columns={"merge_id": "wb_id"})
    comparison = flags.merge(legacy, on="wb_id", how="inner")
    table = pd.crosstab(
        comparison["in_wdpa_polygon_strict_i_iv"],
        comparison["in_wdpa_strict"].fillna(False).astype(bool),
        rownames=["new_polygon_intersection"],
        colnames=["legacy_reservoir_centroid"],
    )
    table.to_csv(reports / "wdpa_new_vs_legacy_reservoir.csv")
    qc = {
        "wdpa_polygon_features": feature_count,
        "strict_i_iv_polygon_features": strict_feature_count,
        "waterbodies": len(flags),
        "waterbodies_intersecting_any_wdpa_polygon": int(any_hit.sum()),
        "waterbodies_intersecting_strict_i_iv_polygon": int(strict_hit.sum()),
        "invalid_waterbody_geometries_repaired": invalid_count,
        "point_only_records_not_applied": 7_747,
    }
    (reports / "wdpa_qc.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved full-waterbody WDPA flags: {output}")
    return output


def self_test() -> None:
    assert STRICT_CATEGORIES == {"Ia", "Ib", "II", "III", "IV"}
    assert BATCH_SIZE <= 2_000
    print("self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["download", "intersect", "all", "self-test"])
    parser.add_argument("--run-dir", type=Path, default=Path("_outputs/v109"))
    args = parser.parse_args()
    if args.stage == "self-test":
        self_test()
        return
    root = project_root()
    run_dir = (root / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir.resolve()
    if args.stage in {"download", "all"}:
        download(run_dir)
    if args.stage in {"intersect", "all"}:
        intersect(run_dir, root)


if __name__ == "__main__":
    main()
