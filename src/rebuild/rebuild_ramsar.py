#!/usr/bin/env python3
"""Intersect all project waterbodies with official Ramsar RSIS boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def intersect(run_dir: Path, root: Path) -> Path:
    import geopandas as gpd

    source = run_dir / "raw/ramsar/unpacked/features_publishedPolygon.shp"
    assert source.exists(), source
    waterbody_path = root / "Dataset/data_processed/merged/all_waterbodies.gpkg"
    ramsar = gpd.read_file(source)
    waterbodies = gpd.read_file(waterbody_path, layer="waterbodies", columns=["wb_id"])
    assert len(waterbodies) == 198_737 and waterbodies["wb_id"].is_unique
    if ramsar.crs is None:
        ramsar = ramsar.set_crs(4326)
    elif ramsar.crs.to_epsg() != 4326:
        ramsar = ramsar.to_crs(4326)
    if waterbodies.crs is None:
        waterbodies = waterbodies.set_crs(4326)
    elif waterbodies.crs.to_epsg() != 4326:
        waterbodies = waterbodies.to_crs(4326)

    invalid_ramsar = int((~ramsar.geometry.is_valid).sum())
    invalid_waterbodies = int((~waterbodies.geometry.is_valid).sum())
    if invalid_ramsar:
        ramsar.geometry = ramsar.geometry.make_valid()
    if invalid_waterbodies:
        waterbodies.geometry = waterbodies.geometry.make_valid()
    ramsar = ramsar[ramsar.geometry.notna() & ~ramsar.geometry.is_empty].reset_index(drop=True)
    waterbodies = waterbodies[
        waterbodies.geometry.notna() & ~waterbodies.geometry.is_empty
    ].reset_index(drop=True)

    pairs = waterbodies.sindex.query(ramsar.geometry, predicate="intersects")
    hit_count = np.zeros(len(waterbodies), dtype=np.int32)
    in_ramsar = np.zeros(len(waterbodies), dtype=bool)
    unique_pairs = pd.DataFrame(
        {
            "ramsar_row": pairs[0],
            "waterbody_row": pairs[1],
            "ramsarid": ramsar.iloc[pairs[0]]["ramsarid"].to_numpy(),
        }
    ).drop_duplicates(["waterbody_row", "ramsarid"])
    if len(unique_pairs):
        counts = unique_pairs.groupby("waterbody_row")["ramsarid"].nunique()
        hit_count[counts.index.to_numpy(int)] = counts.to_numpy(np.int32)
        in_ramsar[counts.index.to_numpy(int)] = True

    flags = pd.DataFrame(
        {
            "wb_id": waterbodies["wb_id"].to_numpy(np.int32),
            "in_ramsar_polygon": in_ramsar,
            "ramsar_site_count": hit_count,
        }
    ).sort_values("wb_id")
    assert len(flags) == 198_737 and flags["wb_id"].is_unique
    data_dir = run_dir / "data"
    reports = run_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    output = data_dir / "ramsar_waterbody_polygon_flags.parquet"
    flags.to_parquet(output, index=False)

    wdpa_path = data_dir / "wdpa_all_waterbody_polygon_flags.parquet"
    outside_wdpa = None
    if wdpa_path.exists():
        wdpa = pd.read_parquet(wdpa_path, columns=["wb_id", "in_wdpa_polygon_any"])
        joined = flags.merge(wdpa, on="wb_id", validate="one_to_one")
        outside_wdpa = int(
            (joined["in_ramsar_polygon"] & ~joined["in_wdpa_polygon_any"].astype(bool)).sum()
        )
        pd.crosstab(
            joined["in_ramsar_polygon"],
            joined["in_wdpa_polygon_any"].astype(bool),
            rownames=["ramsar"],
            colnames=["wdpa_any"],
        ).to_csv(reports / "ramsar_vs_wdpa_confusion.csv")

    qc = {
        "source": "Ramsar Sites Information Service official WFS features_published",
        "source_url": "https://rsis.ramsar.org/geoserver/wfs",
        "license": "CC BY 4.0 unless otherwise indicated",
        "polygon_parts": len(ramsar),
        "unique_ramsar_ids_with_boundaries": int(ramsar["ramsarid"].nunique()),
        "waterbodies": len(flags),
        "waterbodies_intersecting_ramsar": int(in_ramsar.sum()),
        "ramsar_waterbodies_outside_current_wdpa_polygons": outside_wdpa,
        "invalid_ramsar_geometries_repaired": invalid_ramsar,
        "invalid_waterbody_geometries_repaired": invalid_waterbodies,
        "known_gap": "RSIS boundary export contains only sites with published polygon geometry; centroid-only sites are not buffered.",
    }
    (reports / "ramsar_qc.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(qc, indent=2, ensure_ascii=False))
    return output


def self_test() -> None:
    assert project_root().name
    print("self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["intersect", "self-test"])
    parser.add_argument("--run-dir", type=Path, default=Path("_outputs/v110"))
    args = parser.parse_args()
    if args.stage == "self-test":
        self_test()
        return
    root = project_root()
    run_dir = (
        (root / args.run_dir).resolve()
        if not args.run_dir.is_absolute()
        else args.run_dir.resolve()
    )
    intersect(run_dir, root)


if __name__ == "__main__":
    main()
