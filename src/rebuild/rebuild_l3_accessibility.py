#!/usr/bin/env python3
"""Build partial L3 road and mapped-grid accessibility evidence.

This is an infrastructure-accessibility screen, not a complete deployability
tier.  GRIP is a 5-arcminute road-density raster, so true road distance is
bracketed by each occupied cell's half diagonal.  The OSM transmission layer
has no voltage field; its lines are sampled at no more than 5 km spacing and
supplemented with authoritative GRanD hydropower-use evidence.  Engineering,
economic, interconnection-capacity and permitting gates remain unavailable.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
from scipy.spatial import cKDTree


EXPECTED_SCOPE_ROWS = 198_731
EARTH_RADIUS_KM = 6371.0088
ROAD_PRIMARY_THRESHOLD_KM = 10.0
GRID_PRIMARY_THRESHOLD_KM = 25.0
MAX_GRID_SAMPLE_SPACING_KM = 5.0


def xyz(longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(longitude)
    lat = np.deg2rad(latitude)
    cosine = np.cos(lat)
    return np.column_stack(
        [cosine * np.cos(lon), cosine * np.sin(lon), np.sin(lat)]
    ).astype(np.float32)


def chord_to_km(chord: np.ndarray) -> np.ndarray:
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.minimum(1.0, chord / 2.0))


def segment_distance_km(points: np.ndarray) -> np.ndarray:
    lon1 = np.deg2rad(points[:-1, 0])
    lon2 = np.deg2rad(points[1:, 0])
    lat1 = np.deg2rad(points[:-1, 1])
    lat2 = np.deg2rad(points[1:, 1])
    delta_lon = (lon2 - lon1 + np.pi) % (2.0 * np.pi) - np.pi
    delta_lat = lat2 - lat1
    haversine = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
        delta_lon / 2.0
    ) ** 2
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.minimum(1.0, haversine)))


def densify_part(points: np.ndarray) -> list[np.ndarray]:
    pieces = [points.astype(np.float32, copy=False)]
    if len(points) < 2:
        return pieces
    distances = segment_distance_km(points)
    long_segments = np.flatnonzero(distances > MAX_GRID_SAMPLE_SPACING_KM)
    for index in long_segments:
        divisions = int(np.ceil(distances[index] / MAX_GRID_SAMPLE_SPACING_KM))
        fractions = np.arange(1, divisions, dtype=np.float32) / divisions
        start = points[index]
        end = points[index + 1]
        delta_lon = (end[0] - start[0] + 180.0) % 360.0 - 180.0
        longitude = (start[0] + fractions * delta_lon + 180.0) % 360.0 - 180.0
        latitude = start[1] + fractions * (end[1] - start[1])
        pieces.append(np.column_stack([longitude, latitude]).astype(np.float32))
    return pieces


def mapped_powerline_distance(
    shp_path: Path, longitude: np.ndarray, latitude: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    reader = shapefile.Reader(shp_path, encoding="latin1")
    point_parts: list[np.ndarray] = []
    original_vertices = 0
    source_segments = 0
    for shape_index, shape in enumerate(reader.iterShapes(), start=1):
        points = np.asarray(shape.points, dtype=np.float64)
        original_vertices += len(points)
        endings = list(shape.parts[1:]) + [len(points)]
        for start, end in zip(shape.parts, endings):
            part = points[start:end]
            source_segments += max(len(part) - 1, 0)
            point_parts.extend(densify_part(part))
        if shape_index % 50_000 == 0:
            print(f"  read {shape_index:,} mapped power lines", flush=True)
    samples = np.concatenate(point_parts, axis=0)
    sample_tree = cKDTree(xyz(samples[:, 0], samples[:, 1]))
    chord, _ = sample_tree.query(xyz(longitude, latitude), k=1, workers=-1)
    distances = chord_to_km(chord)
    qc = {
        "line_features": int(len(reader)),
        "source_segments": int(source_segments),
        "original_vertices": int(original_vertices),
        "densified_sample_points": int(len(samples)),
        "maximum_sample_spacing_km": MAX_GRID_SAMPLE_SPACING_KM,
        "distance_method": "nearest sampled OSM power-line vertex on a sphere; true mapped-line distance bracketed by 2.5 km sampling half-gap",
    }
    return distances, qc


def read_ascii_grid(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    header: dict[str, float] = {}
    with path.open(encoding="ascii") as stream:
        for _ in range(6):
            key, value = stream.readline().split()
            header[key.lower()] = float(value)
    values = np.loadtxt(path, skiprows=6, dtype=np.float32)
    if values.shape != (int(header["nrows"]), int(header["ncols"])):
        raise ValueError("GRIP ASCII grid dimensions do not match its header")
    return values, header


def road_distance_bracket(
    raster_path: Path, longitude: np.ndarray, latitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    values, header = read_ascii_grid(raster_path)
    rows, columns = np.nonzero(values > 0.0)
    cellsize = float(header["cellsize"])
    road_lon = float(header["xllcorner"]) + (columns + 0.5) * cellsize
    # ESRI ASCII rows run north to south.
    northern_edge = float(header["yllcorner"]) + int(header["nrows"]) * cellsize
    road_lat = northern_edge - (rows + 0.5) * cellsize
    tree = cKDTree(xyz(road_lon, road_lat))
    chord, nearest_index = tree.query(xyz(longitude, latitude), k=1, workers=-1)
    centre_distance = chord_to_km(chord)
    half_latitude_km = EARTH_RADIUS_KM * np.deg2rad(cellsize / 2.0)
    half_longitude_km = half_latitude_km * np.cos(
        np.deg2rad(road_lat[nearest_index])
    )
    cell_radius = np.sqrt(half_latitude_km**2 + half_longitude_km**2)
    lower = np.maximum(0.0, centre_distance - cell_radius)
    upper = centre_distance + cell_radius
    qc = {
        "grid_rows": int(values.shape[0]),
        "grid_columns": int(values.shape[1]),
        "cellsize_degrees": cellsize,
        "positive_road_cells": int(len(rows)),
        "distance_method": "spherical distance to nearest positive road-density cell centre bracketed by that cell's half diagonal",
        "maximum_cell_half_diagonal_km": float(np.max(cell_radius)),
    }
    return centre_distance, lower, upper, qc


def hydropower_evidence(
    reservoir_gpkg: Path, grand_dams_path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    with sqlite3.connect(reservoir_gpkg) as connection:
        links = pd.read_sql_query(
            "select merge_id as wb_id, cast(grand_id_link as integer) as grand_id "
            "from reservoirs where grand_id_link is not null",
            connection,
        )
    reader = shapefile.Reader(grand_dams_path, encoding="latin1")
    records: list[tuple[int, str, str]] = []
    for record in reader.iterRecords(fields=["GRAND_ID", "USE_ELEC", "MAIN_USE"]):
        records.append(
            (
                int(record["GRAND_ID"]),
                str(record["USE_ELEC"]).strip(),
                str(record["MAIN_USE"]).strip(),
            )
        )
    dams = pd.DataFrame(records, columns=["grand_id", "use_elec", "main_use"])
    dams["grand_hydropower_evidence"] = dams["use_elec"].ne("") | dams[
        "main_use"
    ].str.contains("hydro|electric", case=False, regex=True)
    result = links.merge(dams, on="grand_id", how="left", validate="many_to_one")
    result["grand_hydropower_evidence"] = (
        result["grand_hydropower_evidence"].astype("boolean").fillna(False)
    )
    qc = {
        "scope_reservoirs_with_grand_link": int(len(links)),
        "grand_links_matched_to_local_v1_1": int(result["use_elec"].notna().sum()),
        "hydropower_evidence_rows": int(result["grand_hydropower_evidence"].sum()),
    }
    return result[["wb_id", "grand_id", "grand_hydropower_evidence"]], qc


def scenario(
    name: str,
    definition: str,
    mask: np.ndarray,
    energy: np.ndarray,
    l0_twh: float,
) -> dict[str, object]:
    generation_twh = float(energy[mask].sum() / 1000.0)
    return {
        "tier": name,
        "definition": definition,
        "scientific_tier_complete": False,
        "eligible_waterbodies": int(mask.sum()),
        "generation_twh": generation_twh,
        "retention_vs_recommended_l0_pct": generation_twh / l0_twh * 100.0,
    }


def make_summary(frame: pd.DataFrame) -> pd.DataFrame:
    known = frame["historical_dryup_evidence_known_v113"].fillna(False).to_numpy(bool)
    no_dry = frame["historical_no_dryup_observed_1991_2018_v113"].fillna(False).to_numpy(bool)
    core = frame["l2_core_no_dry_pass_v112"].to_numpy(bool)
    conservative = frame["l2_conservative_no_dry_pass_v112"].to_numpy(bool)
    road_guaranteed = frame["road_within_10km_guaranteed_v115"].to_numpy(bool)
    road_possible = frame["road_within_10km_possible_v115"].to_numpy(bool)
    grid_observed = frame["grid_within_25km_observed_or_hydropower_v115"].to_numpy(bool)
    grid_possible = frame["grid_within_25km_possible_or_hydropower_v115"].to_numpy(bool)
    energy = frame["l0_type_specific_generation_gwh"].to_numpy(float)
    l0_twh = float(energy.sum() / 1000.0)
    rows = [
        scenario(
            "L3-access-core-evidence-lower-bound",
            "Deduplicated core L2 evidence lower bound + GRIP road guaranteed within 10 km + mapped power line observed within 25 km or GRanD hydropower evidence",
            core & known & no_dry & road_guaranteed & grid_observed,
            energy,
            l0_twh,
        ),
        scenario(
            "L3-access-core-mapping-upper-envelope",
            "Core L2 unknown-pass envelope + GRIP road possibly within 10 km + mapped power line possibly within 25 km or GRanD hydropower evidence",
            core & (~known | no_dry) & road_possible & grid_possible,
            energy,
            l0_twh,
        ),
        scenario(
            "L3-access-conservative-evidence-lower-bound",
            "Deduplicated conservative L2 evidence lower bound + guaranteed 10 km road + observed 25 km mapped-grid/hydropower evidence",
            conservative & known & no_dry & road_guaranteed & grid_observed,
            energy,
            l0_twh,
        ),
        scenario(
            "L3-access-conservative-mapping-upper-envelope",
            "Conservative L2 unknown-pass envelope + possible 10 km road + possible 25 km mapped-grid/hydropower evidence",
            conservative & (~known | no_dry) & road_possible & grid_possible,
            energy,
            l0_twh,
        ),
        {
            "tier": "L3-complete-deployable",
            "definition": "Unavailable: engineering, interconnection capacity, economics, permitting, water-level/depth, wind/wave and social acceptance are not globally resolved",
            "scientific_tier_complete": False,
            "eligible_waterbodies": pd.NA,
            "generation_twh": np.nan,
            "retention_vs_recommended_l0_pct": np.nan,
        },
    ]
    return pd.DataFrame(rows)


def sensitivity_table(frame: pd.DataFrame) -> pd.DataFrame:
    known = frame["historical_dryup_evidence_known_v113"].fillna(False).to_numpy(bool)
    no_dry = frame["historical_no_dryup_observed_1991_2018_v113"].fillna(False).to_numpy(bool)
    core = frame["l2_core_no_dry_pass_v112"].to_numpy(bool)
    energy = frame["l0_type_specific_generation_gwh"].to_numpy(float)
    rows: list[dict[str, object]] = []
    road_centre = frame["grip_road_cell_center_distance_km_v115"].to_numpy(float)
    grid_upper = frame["osm_powerline_sample_distance_km_v115"].to_numpy(float)
    hydro = frame["grand_hydropower_evidence"].to_numpy(bool)
    for road_km in [5.0, 10.0, 25.0]:
        for grid_km in [10.0, 25.0, 50.0, 80.0]:
            infrastructure = (road_centre <= road_km) & ((grid_upper <= grid_km) | hydro)
            for evidence, base in [
                ("dryup_known_only", core & known & no_dry),
                ("dryup_unknown_pass", core & (~known | no_dry)),
            ]:
                mask = base & infrastructure
                rows.append(
                    {
                        "road_cell_center_threshold_km": road_km,
                        "mapped_grid_threshold_km": grid_km,
                        "dryup_evidence_policy": evidence,
                        "eligible_waterbodies": int(mask.sum()),
                        "generation_twh": float(energy[mask].sum() / 1000.0),
                    }
                )
    return pd.DataFrame(rows)


def run(
    source_path: Path,
    coordinate_path: Path,
    roads_path: Path,
    transmission_path: Path,
    reservoir_gpkg: Path,
    grand_dams_path: Path,
    output_dir: Path,
) -> None:
    frame = pd.read_parquet(source_path)
    if len(frame) != EXPECTED_SCOPE_ROWS or not frame["wb_id"].is_unique:
        raise ValueError("v114 deduplicated scope mismatch")
    coordinates = pd.read_parquet(
        coordinate_path, columns=["wb_id", "centroid_lon", "centroid_lat"]
    )
    frame = frame.merge(coordinates, on="wb_id", how="left", validate="one_to_one")
    if frame[["centroid_lon", "centroid_lat"]].isna().any().any():
        raise ValueError("missing waterbody coordinates")
    longitude = frame["centroid_lon"].to_numpy(float)
    latitude = frame["centroid_lat"].to_numpy(float)

    road_centre, road_lower, road_upper, road_qc = road_distance_bracket(
        roads_path, longitude, latitude
    )
    frame["grip_road_cell_center_distance_km_v115"] = road_centre.astype(np.float32)
    frame["grip_road_distance_lower_bound_km_v115"] = road_lower.astype(np.float32)
    frame["grip_road_distance_upper_bound_km_v115"] = road_upper.astype(np.float32)
    frame["road_within_10km_guaranteed_v115"] = road_upper <= ROAD_PRIMARY_THRESHOLD_KM
    frame["road_within_10km_possible_v115"] = road_lower <= ROAD_PRIMARY_THRESHOLD_KM

    grid_distance, grid_qc = mapped_powerline_distance(
        transmission_path, longitude, latitude
    )
    frame["osm_powerline_sample_distance_km_v115"] = grid_distance.astype(np.float32)
    frame["osm_powerline_distance_lower_bound_km_v115"] = np.maximum(
        0.0, grid_distance - MAX_GRID_SAMPLE_SPACING_KM / 2.0
    ).astype(np.float32)
    hydro, hydro_qc = hydropower_evidence(reservoir_gpkg, grand_dams_path)
    frame = frame.merge(
        hydro[["wb_id", "grand_hydropower_evidence"]],
        on="wb_id",
        how="left",
        validate="one_to_one",
    )
    frame["grand_hydropower_evidence"] = (
        frame["grand_hydropower_evidence"].astype("boolean").fillna(False)
    )
    frame["grid_within_25km_observed_or_hydropower_v115"] = (
        frame["osm_powerline_sample_distance_km_v115"].le(GRID_PRIMARY_THRESHOLD_KM)
        | frame["grand_hydropower_evidence"]
    )
    frame["grid_within_25km_possible_or_hydropower_v115"] = (
        frame["osm_powerline_distance_lower_bound_km_v115"].le(
            GRID_PRIMARY_THRESHOLD_KM
        )
        | frame["grand_hydropower_evidence"]
    )

    summary = make_summary(frame)
    sensitivity = sensitivity_table(frame)
    data_dir = output_dir / "data"
    reports_dir = output_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(data_dir / "paper_scope_l3_accessibility.parquet", index=False)
    accessibility_columns = [
        "wb_id",
        "wb_type",
        "centroid_lon",
        "centroid_lat",
        "grip_road_cell_center_distance_km_v115",
        "grip_road_distance_lower_bound_km_v115",
        "grip_road_distance_upper_bound_km_v115",
        "road_within_10km_guaranteed_v115",
        "road_within_10km_possible_v115",
        "osm_powerline_sample_distance_km_v115",
        "osm_powerline_distance_lower_bound_km_v115",
        "grand_id",
        "grand_hydropower_evidence",
        "grid_within_25km_observed_or_hydropower_v115",
        "grid_within_25km_possible_or_hydropower_v115",
    ]
    frame[accessibility_columns].to_parquet(
        data_dir / "waterbody_l3_accessibility_flags.parquet", index=False
    )
    summary.to_csv(reports_dir / "l3_accessibility_summary.csv", index=False)
    sensitivity.to_csv(reports_dir / "l3_accessibility_threshold_sensitivity.csv", index=False)
    qc = {
        "scope_rows": int(len(frame)),
        "road": road_qc,
        "mapped_power_lines": grid_qc,
        "hydropower": hydro_qc,
        "primary_thresholds_km": {
            "road": ROAD_PRIMARY_THRESHOLD_KM,
            "mapped_power_line": GRID_PRIMARY_THRESHOLD_KM,
        },
        "road_guaranteed_pass_rows": int(frame["road_within_10km_guaranteed_v115"].sum()),
        "road_possible_pass_rows": int(frame["road_within_10km_possible_v115"].sum()),
        "grid_observed_or_hydro_pass_rows": int(
            frame["grid_within_25km_observed_or_hydropower_v115"].sum()
        ),
        "grid_possible_or_hydro_pass_rows": int(
            frame["grid_within_25km_possible_or_hydropower_v115"].sum()
        ),
        "limitations": [
            "GRIP road proximity is based on waterbody centroids and 5-arcminute occupied cells, not exact shoreline-to-road distance",
            "the OSM power-line layer has no voltage or substation/interconnection-capacity field and is spatially incomplete",
            "a GRanD hydropower-use flag is connection evidence, not proof of spare grid capacity",
            "engineering, economics, permitting and social acceptance are not included",
        ],
        "l3_complete": False,
    }
    (reports_dir / "l3_accessibility_qc.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(qc, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-path",
        type=Path,
        default=Path(
            "_outputs/v114/data/paper_scope_deduplicated_historical_generation.parquet"
        ),
    )
    parser.add_argument(
        "--coordinate-path",
        type=Path,
        default=Path("_outputs/v110/data/waterbody_generation_rebuild.parquet"),
    )
    parser.add_argument(
        "--roads-path",
        type=Path,
        default=Path(
            "_outputs/v115/raw/roads/extracted/grip4_total_dens_m_km2.asc"
        ),
    )
    parser.add_argument(
        "--transmission-path",
        type=Path,
        default=Path("_outputs/v115/raw/transmission/extracted/osm_power_tmm.shp"),
    )
    parser.add_argument(
        "--reservoir-gpkg",
        type=Path,
        default=Path("Dataset/data_processed/merged/merged_reservoirs.gpkg"),
    )
    parser.add_argument(
        "--grand-dams-path",
        type=Path,
        default=Path(
            "Dataset/data_processed/GRanD/reserviors-dams-rev01-global-shp/"
            "GRanD_dams_v1_1.shp"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_outputs/v115"))
    args = parser.parse_args()
    run(
        args.source_path,
        args.coordinate_path,
        args.roads_path,
        args.transmission_path,
        args.reservoir_gpkg,
        args.grand_dams_path,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
