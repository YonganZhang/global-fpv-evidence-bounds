#!/usr/bin/env python3
"""Build the post-v116 FPV scientific revision.

The v116 gate validated the calculations that started from the legacy merged
inventory.  A later source-level audit found that the legacy merge had removed
every HydroLAKES natural or controlled lake whose centroid was within 1 km of
an external reservoir.  This script replaces that proximity-only deletion
with an evidence-tier adjudication, restores unsupported deletions, completes
country attribution for external reservoirs, and generates the sensitivity
and validation tables required by the revised manuscript.

The reference inventory removes authoritative and calibrated high-confidence
reservoir matches.  Spatial-only matches are also reported as an identity
sensitivity because their calibration precision is high but not perfect.
Complete paper-equivalent L2 and complete deployable L3 remain unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapefile
import shapely
import xlrd
from scipy.spatial import cKDTree

from rebuild_historical_dryup_glev import (
    EXPECTED_MONTHS,
    read_glev_subset,
    summarize_glev,
)
from rebuild_l3_accessibility import (
    GRID_PRIMARY_THRESHOLD_KM,
    MAX_GRID_SAMPLE_SPACING_KM,
    ROAD_PRIMARY_THRESHOLD_KM,
    mapped_powerline_distance,
    road_distance_bracket,
)
from rebuild_l3_engineering_economics import (
    ANNUAL_DEGRADATION,
    LIFETIME_YEARS,
    load_globathy,
    present_value_lcoe,
)
from rebuild_population_centers import (
    DISTANCE_THRESHOLD_KM,
    MAX_BOUNDARY_SEGMENT_DEG,
    chord_to_km,
    spherical_xyz,
)
from recompute_levels import DAYS_IN_MONTH, FAIMAN_U0, FAIMAN_U1, TEMP_COEFF_POWER, T_REF_C


ROOT = Path(__file__).resolve().parents[2]
V116_ROWS = 198_731
RAW_HYDROLAKES_SCOPE_ROWS = 179_787
REFERENCE_ADDITIONS = 1_245
REFERENCE_ROWS = 199_976
AUTHORITATIVE_REMOVALS = 108
SPATIAL_HIGH_CONFIDENCE_REMOVALS = 645
REFERENCE_RESERVOIR_COVERAGE = 0.30
REFERENCE_LAKE_COVERAGE = 0.10
MAX_FPV_FOOTPRINT_KM2 = 30.0
PACKING_DENSITY_KWP_M2 = 0.10
ICE_DURATION_THRESHOLD_DAYS = 182.5
DIRECT_MIN_VALID_YEARS = 5
MODEL_MAX_SUBZERO_MONTHS = 6
STRICT_CATEGORIES = {"Ia", "Ib", "II", "III", "IV"}

HYDRO_DBf = ROOT / (
    "Dataset/data_processed/HydroLAKES/HydroLAKES_polys_v10_shp/"
    "HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.dbf"
)
HYDRO_SHP = HYDRO_DBf.with_suffix(".shp")
V116_PATH = ROOT / "_outputs/v116/data/paper_scope_l3_engineering_economics.parquet"
V111_PATH = ROOT / "_outputs/v111/data/hydrolakes_full_generation.parquet"
WOOLWAY_PATH = ROOT / "_outputs/v110/data/woolway_public_lake_info.parquet"
LI_CCR_ARCHIVE = ROOT / "_outputs/v110/raw/li_ccr/Lake ice phenology.zip"
LI_CCR_PROBABILITY = ROOT / "_outputs/v110/raw/li_ccr/Probability of complete ice-cover occurrence.xls"
GLEV_ARCHIVE = ROOT / "_outputs/v113/raw/glev/1_surfacewater_area.zip"
NATURAL_EARTH_SHP = ROOT / (
    "_outputs/v117/raw/natural_earth_admin0/unpacked/ne_10m_admin_0_countries.shp"
)
ROADS_PATH = ROOT / "_outputs/v115/raw/roads/extracted/grip4_total_dens_m_km2.asc"
POWER_LINES_PATH = ROOT / "_outputs/v115/raw/transmission/extracted/osm_power_tmm.shp"
GLOBATHY_ZIP = ROOT / "_outputs/v116/raw/globathy/GLOBathy_basic_parameters.zip"
MONTHLY_SIM_PATH = ROOT / "_outputs/v109/data/waterbody_monthly_simulation_rebuild.parquet"
POPULATION_CELLS_PATH = ROOT / "_outputs/v109/data/population_centers_landscan2024_cells.parquet"


def json_dump(path: Path, payload: object) -> None:
    def convert(value: object) -> object:
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if value is pd.NA:
            return None
        raise TypeError(f"unsupported JSON value: {type(value).__name__}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=convert),
        encoding="utf-8",
    )


def load_hydrolakes_scope() -> pd.DataFrame:
    reader = shapefile.Reader(dbf=HYDRO_DBf, encoding="latin1")
    rows: list[tuple[object, ...]] = []
    fields = [
        "Hylak_id",
        "Lake_type",
        "Lake_area",
        "Grand_id",
        "Lake_name",
        "Country",
        "Continent",
        "Pour_long",
        "Pour_lat",
    ]
    for record in reader.iterRecords(fields=fields):
        lake_type = int(record["Lake_type"])
        area = float(record["Lake_area"])
        if lake_type not in {1, 3} or area < 1.0:
            continue
        rows.append(
            (
                int(record["Hylak_id"]),
                lake_type,
                area,
                int(record["Grand_id"]),
                str(record["Lake_name"]).strip(),
                str(record["Country"]).strip(),
                str(record["Continent"]).strip(),
                float(record["Pour_long"]),
                float(record["Pour_lat"]),
            )
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "hylak_id",
            "lake_type_code",
            "area_km2",
            "hydrolakes_grand_id",
            "name",
            "hydrolakes_country",
            "hydrolakes_continent",
            "pour_lon",
            "pour_lat",
        ],
    )
    if len(frame) != RAW_HYDROLAKES_SCOPE_ROWS or not frame["hylak_id"].is_unique:
        raise ValueError("raw HydroLAKES >=1 km2 natural/controlled scope mismatch")
    return frame


def adjudicate_inventory(
    v116: pd.DataFrame, raw_scope: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    retained_direct = set(
        v116.loc[v116["wb_type"].ne("reservoir"), "hylak_id"].dropna().astype(int)
    )
    removed = raw_scope.loc[~raw_scope["hylak_id"].isin(retained_direct)].copy()
    reservoir_map = v116.loc[
        v116["wb_type"].eq("reservoir") & v116["mapped_hylak_id"].notna(),
        ["wb_id", "mapped_hylak_id", "dryup_key_source_v113", "nearest_distance_km", "nearest_area_ratio"],
    ].copy()
    reservoir_map["hylak_id"] = reservoir_map["mapped_hylak_id"].astype(int)
    removed = removed.merge(
        reservoir_map.drop(columns="mapped_hylak_id"),
        on="hylak_id",
        how="left",
        validate="one_to_one",
    )
    removed["adjudication_v117"] = np.select(
        [
            removed["dryup_key_source_v113"].eq("grand_id_authoritative"),
            removed["dryup_key_source_v113"].eq("spatial_nearest_high_confidence"),
        ],
        ["remove_authoritative_duplicate", "remove_spatial_high_confidence_reference"],
        default="restore_no_duplicate_evidence",
    )
    counts = removed["adjudication_v117"].value_counts().to_dict()
    expected = {
        "remove_authoritative_duplicate": AUTHORITATIVE_REMOVALS,
        "remove_spatial_high_confidence_reference": SPATIAL_HIGH_CONFIDENCE_REMOVALS,
        "restore_no_duplicate_evidence": REFERENCE_ADDITIONS,
    }
    if counts != expected:
        raise ValueError(f"inventory adjudication mismatch: {counts}")
    additions = removed.loc[
        removed["adjudication_v117"].eq("restore_no_duplicate_evidence")
    ].copy()
    spatial_sensitivity = removed.loc[
        removed["adjudication_v117"].eq("remove_spatial_high_confidence_reference")
    ].copy()
    return removed, additions, spatial_sensitivity


def load_target_geometries(target_ids: pd.Series) -> gpd.GeoDataFrame:
    ids = sorted(set(target_ids.astype(int)))
    where = "Hylak_id IN (" + ",".join(str(value) for value in ids) + ")"
    columns = ["Hylak_id", "Lake_type", "Lake_area", "Country", "Continent"]
    frame = gpd.read_file(
        HYDRO_SHP,
        where=where,
        columns=columns,
        engine="pyogrio",
    ).rename(
        columns={
            "Hylak_id": "hylak_id",
            "Lake_type": "lake_type_code",
            "Lake_area": "area_km2",
            "Country": "hydrolakes_country",
            "Continent": "hydrolakes_continent",
        }
    )
    frame["hylak_id"] = frame["hylak_id"].astype(int)
    if len(frame) != len(ids) or not frame["hylak_id"].is_unique:
        raise ValueError("failed to recover every restored HydroLAKES geometry")
    if frame.crs is None:
        frame = frame.set_crs(4326)
    elif frame.crs.to_epsg() != 4326:
        frame = frame.to_crs(4326)
    invalid = ~frame.geometry.is_valid
    if invalid.any():
        frame.loc[invalid, "geometry"] = frame.loc[invalid, "geometry"].make_valid()
    return frame


def parse_li_ccr_targets(target_ids: pd.Series) -> pd.DataFrame:
    target = set(target_ids.astype(int))
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(LI_CCR_ARCHIVE) as bundle:
        members = {
            int(Path(item.filename).stem): item
            for item in bundle.infolist()
            if not item.is_dir() and item.filename.lower().endswith(".xls")
        }
        for hylak_id in sorted(target & set(members)):
            workbook = xlrd.open_workbook(file_contents=bundle.read(members[hylak_id]))
            sheet = workbook.sheet_by_index(0)
            header = {str(value).strip(): pos for pos, value in enumerate(sheet.row_values(0))}
            baseline: list[float] = []
            for row_number in range(1, sheet.nrows):
                try:
                    year = int(float(sheet.cell_value(row_number, header["Year"])))
                    ice_days = float(sheet.cell_value(row_number, header["ICD"]))
                except (TypeError, ValueError):
                    continue
                if 2002 <= year <= 2020 and np.isfinite(ice_days):
                    baseline.append(ice_days)
            rows.append(
                {
                    "hylak_id": hylak_id,
                    "li_ccr_valid_years_2002_2020_v117": len(baseline),
                    "li_ccr_mean_icd_days_2002_2020_v117": (
                        float(np.mean(baseline)) if baseline else np.nan
                    ),
                }
            )
    direct = pd.DataFrame(rows)
    probability = pd.read_excel(LI_CCR_PROBABILITY).rename(
        columns={"Hydro_id": "hylak_id", "PCIO": "li_ccr_pcio_v117"}
    )
    probability["hylak_id"] = probability["hylak_id"].astype(int)
    if len(direct):
        direct = direct.merge(probability, on="hylak_id", how="left", validate="one_to_one")
    else:
        direct = pd.DataFrame(
            columns=[
                "hylak_id",
                "li_ccr_valid_years_2002_2020_v117",
                "li_ccr_mean_icd_days_2002_2020_v117",
                "li_ccr_pcio_v117",
            ]
        )
    return direct


def protected_area_flags(additions: gpd.GeoDataFrame) -> pd.DataFrame:
    pages = sorted((ROOT / "_outputs/v109/raw/wdpa_polygons_2026_06").glob("page_*.geojson"))
    any_hit = np.zeros(len(additions), dtype=bool)
    strict_hit = np.zeros(len(additions), dtype=bool)
    index = additions.sindex
    for number, page in enumerate(pages, start=1):
        protected = gpd.read_file(page, engine="pyogrio")
        if protected.crs is None:
            protected = protected.set_crs(4326)
        elif protected.crs.to_epsg() != 4326:
            protected = protected.to_crs(4326)
        protected = protected[protected.geometry.notna() & ~protected.geometry.is_empty]
        invalid = ~protected.geometry.is_valid
        if invalid.any():
            protected.loc[invalid, "geometry"] = protected.loc[invalid, "geometry"].make_valid()
        pairs = index.query(protected.geometry, predicate="intersects")
        if pairs.size:
            any_hit[np.unique(pairs[1])] = True
        strict = protected.loc[protected["iucn_cat"].isin(STRICT_CATEGORIES)]
        pairs = index.query(strict.geometry, predicate="intersects")
        if pairs.size:
            strict_hit[np.unique(pairs[1])] = True
        if number % 50 == 0 or number == len(pages):
            print(f"  WDPA pages {number}/{len(pages)}", flush=True)
    return pd.DataFrame(
        {
            "hylak_id": additions["hylak_id"].to_numpy(int),
            "in_wdpa_polygon_any": any_hit,
            "in_wdpa_polygon_strict_i_iv": strict_hit,
        }
    )


def ramsar_flags(additions: gpd.GeoDataFrame) -> pd.DataFrame:
    path = ROOT / "_outputs/v110/raw/ramsar/unpacked/features_publishedPolygon.shp"
    ramsar = gpd.read_file(path, engine="pyogrio")
    if ramsar.crs is None:
        ramsar = ramsar.set_crs(4326)
    elif ramsar.crs.to_epsg() != 4326:
        ramsar = ramsar.to_crs(4326)
    invalid = ~ramsar.geometry.is_valid
    if invalid.any():
        ramsar.loc[invalid, "geometry"] = ramsar.loc[invalid, "geometry"].make_valid()
    ramsar = ramsar[ramsar.geometry.notna() & ~ramsar.geometry.is_empty]
    pairs = additions.sindex.query(ramsar.geometry, predicate="intersects")
    count = np.zeros(len(additions), dtype=np.int16)
    if pairs.size:
        unique = pd.DataFrame(
            {
                "ramsar_row": pairs[0],
                "waterbody_row": pairs[1],
                "ramsarid": ramsar.iloc[pairs[0]]["ramsarid"].to_numpy(),
            }
        ).drop_duplicates(["waterbody_row", "ramsarid"])
        grouped = unique.groupby("waterbody_row")["ramsarid"].nunique()
        count[grouped.index.to_numpy(int)] = grouped.to_numpy(np.int16)
    return pd.DataFrame(
        {
            "hylak_id": additions["hylak_id"].to_numpy(int),
            "in_ramsar_polygon": count > 0,
            "ramsar_site_count_v117": count,
        }
    )


def population_flags(additions: gpd.GeoDataFrame) -> pd.DataFrame:
    cells = pd.read_parquet(POPULATION_CELLS_PATH)
    tree = cKDTree(spherical_xyz(cells["lon"].to_numpy(), cells["lat"].to_numpy()))
    minimum = np.full(len(additions), np.inf, dtype=np.float32)
    for start in range(0, len(additions), 200):
        stop = min(start + 200, len(additions))
        geoms = np.asarray(additions.geometry.iloc[start:stop].values, dtype=object)
        segmented = shapely.segmentize(geoms, max_segment_length=MAX_BOUNDARY_SEGMENT_DEG)
        coords, geometry_index = shapely.get_coordinates(segmented, return_index=True)
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
        minimum[start:stop] = batch_min.astype(np.float32)
    return pd.DataFrame(
        {
            "hylak_id": additions["hylak_id"].to_numpy(int),
            "population_center_distance_km": minimum,
            "within_10km_population_center": minimum <= DISTANCE_THRESHOLD_KM,
        }
    )


def build_added_rows(
    additions: pd.DataFrame,
    geometries: gpd.GeoDataFrame,
    direct_ice: pd.DataFrame,
    run_dir: Path,
) -> pd.DataFrame:
    full = pd.read_parquet(V111_PATH)
    added = additions.merge(
        full,
        on="hylak_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_v111"),
    )
    if added["annual_footprint_energy_kwh_m2"].isna().any():
        raise ValueError("restored rows missing v111 generation")
    added = added.merge(direct_ice, on="hylak_id", how="left", validate="one_to_one")
    wdpa = protected_area_flags(geometries)
    ramsar = ramsar_flags(geometries)
    population = population_flags(geometries)
    for table in [wdpa, ramsar, population]:
        added = added.merge(table, on="hylak_id", how="left", validate="one_to_one")

    glev_raw, months = read_glev_subset(
        GLEV_ARCHIVE,
        added["hylak_id"],
        run_dir / "raw/glev/glev_restored_1991_2018.parquet",
    )
    glev = summarize_glev(glev_raw, months)
    added = added.merge(glev, on="hylak_id", how="left", validate="one_to_one")

    geom = geometries[["hylak_id", "geometry"]].copy()
    centroid = geom.to_crs(6933).geometry.centroid
    centroid = gpd.GeoSeries(centroid, crs=6933).to_crs(4326)
    geom["centroid_lon"] = centroid.x.to_numpy(float)
    geom["centroid_lat"] = centroid.y.to_numpy(float)
    added = added.merge(
        pd.DataFrame(geom.drop(columns="geometry")),
        on="hylak_id",
        how="left",
        validate="one_to_one",
    )

    added["wb_id"] = np.arange(198_738, 198_738 + len(added), dtype=np.int32)
    added["wb_type"] = added["lake_type_code"].map({1: "lake", 3: "controlled_lake"})
    added["country"] = added["hydrolakes_country"]
    added["continent"] = added["hydrolakes_continent"]
    energy = added["annual_footprint_energy_kwh_m2"].to_numpy(float)
    area = added["area_km2"].to_numpy(float)
    added["l0_generation_gwh"] = energy * np.minimum(0.30 * area, MAX_FPV_FOOTPRINT_KM2)
    added["l0_type_specific_generation_gwh"] = energy * np.minimum(
        REFERENCE_LAKE_COVERAGE * area, MAX_FPV_FOOTPRINT_KM2
    )
    proxy_pass = added["ice_months_temp_proxy"].le(MODEL_MAX_SUBZERO_MONTHS)
    woolway_available = added["woolway_ice_cover_fraction"].notna()
    direct_usable = added["li_ccr_valid_years_2002_2020_v117"].fillna(0).ge(
        DIRECT_MIN_VALID_YEARS
    )
    added["ice_direct_usable"] = direct_usable
    added["ice_pass_li_ccr"] = pd.Series(pd.NA, index=added.index, dtype="boolean")
    added.loc[direct_usable, "ice_pass_li_ccr"] = added.loc[
        direct_usable, "li_ccr_mean_icd_days_2002_2020_v117"
    ].le(ICE_DURATION_THRESHOLD_DAYS)
    added["ice_pass_v112"] = proxy_pass
    added["ice_source_v112"] = "nasa_power_monthly_temp_proxy_2019_2023"
    added.loc[woolway_available, "ice_pass_v112"] = added.loc[
        woolway_available, "woolway_ice_cover_fraction"
    ].le(0.5)
    added.loc[woolway_available, "ice_source_v112"] = (
        "woolway_air_temperature_lag_model_1991_2020"
    )
    added.loc[direct_usable, "ice_pass_v112"] = added.loc[
        direct_usable, "ice_pass_li_ccr"
    ].astype(bool)
    added.loc[direct_usable, "ice_source_v112"] = "li_ccr_modis_phenology_2002_2020"
    added["l1_area_pass_v112"] = added["area_km2"].ge(0.01)
    added["l1_pass_v112"] = added["l1_area_pass_v112"] & added["ice_pass_v112"]
    added["l1_uniform_generation_gwh_v112"] = np.where(
        added["l1_pass_v112"], added["l0_generation_gwh"], 0.0
    )
    added["l1_type_generation_gwh_v112"] = np.where(
        added["l1_pass_v112"], added["l0_type_specific_generation_gwh"], 0.0
    )
    added["l2_core_no_dry_pass_v112"] = (
        added["l1_pass_v112"]
        & ~added["in_wdpa_polygon_strict_i_iv"]
        & added["within_10km_population_center"]
    )
    added["l2_conservative_no_dry_pass_v112"] = (
        added["l1_pass_v112"]
        & ~added["in_wdpa_polygon_any"]
        & ~added["in_ramsar_polygon"]
        & added["within_10km_population_center"]
    )
    for prefix, mask in [
        ("core", added["l2_core_no_dry_pass_v112"]),
        ("conservative", added["l2_conservative_no_dry_pass_v112"]),
    ]:
        added[f"l2_{prefix}_type_generation_gwh_v112"] = np.where(
            mask, added["l0_type_specific_generation_gwh"], 0.0
        )
    added["dryup_hylak_id_v113"] = added["hylak_id"].astype("Int64")
    added["dryup_key_source_v113"] = "hydrolakes_direct_restored_v117"
    added["historical_dryup_evidence_known_v113"] = added[
        "historical_dryup_evidence_known_v113"
    ].fillna(False)
    added["historical_dryup_observed_1991_2018_v113"] = added[
        "historical_dryup_observed_1991_2018_v113"
    ].fillna(False)
    added["historical_no_dryup_observed_1991_2018_v113"] = added[
        "historical_no_dryup_observed_1991_2018_v113"
    ].fillna(False)
    added["scope_reference_v117"] = True
    added["inventory_action_v117"] = "restored_no_duplicate_evidence"
    return added


def attach_current_ice_details(frame: pd.DataFrame, added: pd.DataFrame) -> pd.DataFrame:
    ice = pd.read_parquet(
        ROOT / "_outputs/v110/data/waterbody_ice_flags.parquet",
        columns=[
            "wb_id",
            "li_ccr_valid_years_2002_2020",
            "li_ccr_mean_icd_days_2002_2020",
            "li_ccr_pcio",
        ],
    ).rename(
        columns={
            "li_ccr_valid_years_2002_2020": "li_ccr_valid_years_2002_2020_v117",
            "li_ccr_mean_icd_days_2002_2020": "li_ccr_mean_icd_days_2002_2020_v117",
            "li_ccr_pcio": "li_ccr_pcio_v117",
        }
    )
    frame = frame.merge(ice, on="wb_id", how="left", validate="one_to_one")
    added_columns = set(frame.columns) - set(added.columns)
    for column in added_columns:
        added[column] = pd.NA
    frame_columns = set(added.columns) - set(frame.columns)
    for column in frame_columns:
        frame[column] = pd.NA
    return pd.concat([frame[added.columns], added], ignore_index=True)


def add_accessibility(frame: pd.DataFrame, additions_mask: np.ndarray) -> pd.DataFrame:
    longitude = frame.loc[additions_mask, "centroid_lon"].to_numpy(float)
    latitude = frame.loc[additions_mask, "centroid_lat"].to_numpy(float)
    road_centre, road_lower, road_upper, _ = road_distance_bracket(
        ROADS_PATH, longitude, latitude
    )
    grid_distance, _ = mapped_powerline_distance(POWER_LINES_PATH, longitude, latitude)
    index = frame.index[additions_mask]
    frame.loc[index, "grip_road_cell_center_distance_km_v115"] = road_centre
    frame.loc[index, "grip_road_distance_lower_bound_km_v115"] = road_lower
    frame.loc[index, "grip_road_distance_upper_bound_km_v115"] = road_upper
    frame.loc[index, "road_within_10km_guaranteed_v115"] = (
        road_upper <= ROAD_PRIMARY_THRESHOLD_KM
    )
    frame.loc[index, "road_within_10km_possible_v115"] = (
        road_lower <= ROAD_PRIMARY_THRESHOLD_KM
    )
    frame.loc[index, "osm_powerline_sample_distance_km_v115"] = grid_distance
    frame.loc[index, "osm_powerline_distance_lower_bound_km_v115"] = np.maximum(
        0.0, grid_distance - MAX_GRID_SAMPLE_SPACING_KM / 2.0
    )
    frame.loc[index, "grand_hydropower_evidence"] = False
    frame.loc[index, "grid_within_25km_observed_or_hydropower_v115"] = (
        grid_distance <= GRID_PRIMARY_THRESHOLD_KM
    )
    frame.loc[index, "grid_within_25km_possible_or_hydropower_v115"] = (
        np.maximum(0.0, grid_distance - MAX_GRID_SAMPLE_SPACING_KM / 2.0)
        <= GRID_PRIMARY_THRESHOLD_KM
    )
    boolean_columns = [
        "road_within_10km_guaranteed_v115",
        "road_within_10km_possible_v115",
        "grand_hydropower_evidence",
        "grid_within_25km_observed_or_hydropower_v115",
        "grid_within_25km_possible_or_hydropower_v115",
    ]
    for column in boolean_columns:
        frame[column] = frame[column].fillna(False).astype(bool)
    return frame


def add_country_attribution(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    countries = gpd.read_file(NATURAL_EARTH_SHP, engine="pyogrio")
    countries = countries[["ADMIN", "CONTINENT", "ADM0_A3", "geometry"]].copy()
    countries = countries.reset_index(drop=True)
    lookup = countries.drop_duplicates("ADMIN").set_index("ADMIN")

    full_hydro = pd.read_parquet(
        V111_PATH, columns=["hylak_id", "country", "continent"]
    ).rename(
        columns={
            "hylak_id": "mapped_hylak_id",
            "country": "mapped_country_v117",
            "continent": "mapped_continent_v117",
        }
    )
    full_hydro["mapped_hylak_id"] = full_hydro["mapped_hylak_id"].astype("Int64")
    country_continent = (
        full_hydro.dropna(subset=["mapped_country_v117", "mapped_continent_v117"])
        .groupby("mapped_country_v117")["mapped_continent_v117"]
        .agg(lambda values: values.mode().iat[0])
    )
    frame["mapped_hylak_id"] = pd.to_numeric(
        frame["mapped_hylak_id"], errors="coerce"
    ).astype("Int64")
    frame = frame.merge(full_hydro, on="mapped_hylak_id", how="left", validate="many_to_one")
    frame["country_v117"] = frame["country"]
    frame["continent_v117"] = frame["continent"]
    frame["country_source_v117"] = np.where(
        frame["wb_type"].eq("reservoir"), pd.NA, "hydrolakes_attribute"
    )
    mapped_reservoir = frame["wb_type"].eq("reservoir") & frame[
        "mapped_country_v117"
    ].notna()
    frame.loc[mapped_reservoir, "country_v117"] = frame.loc[
        mapped_reservoir, "mapped_country_v117"
    ]
    frame.loc[mapped_reservoir, "continent_v117"] = frame.loc[
        mapped_reservoir, "mapped_continent_v117"
    ]
    frame.loc[mapped_reservoir, "country_source_v117"] = "reservoir_hylak_crosswalk"

    # Unmapped external reservoirs can carry generic ocean-region labels in the
    # source inventory (for example, island states).  Where a country occurs in
    # HydroLAKES, use its modal HydroLAKES continent before falling back to the
    # Natural Earth centroid.  Mapped reservoirs retain their lake-specific
    # HydroLAKES continent, which avoids flattening transcontinental countries.
    unmapped_reservoir = frame["wb_type"].eq("reservoir") & ~mapped_reservoir
    reservoir_continent = frame.loc[unmapped_reservoir, "country_v117"].map(
        country_continent
    )
    replace_continent = reservoir_continent.notna()
    replace_index = reservoir_continent.index[replace_continent]
    frame.loc[replace_index, "continent_v117"] = reservoir_continent.loc[
        replace_index
    ]

    frame["country_iso3_v117"] = frame["country_v117"].map(lookup["ADM0_A3"])
    need_spatial = frame["country_v117"].isna() | frame["country_iso3_v117"].isna()
    points = gpd.GeoDataFrame(
        frame.loc[need_spatial, ["wb_id"]].copy(),
        geometry=gpd.points_from_xy(
            frame.loc[need_spatial, "centroid_lon"],
            frame.loc[need_spatial, "centroid_lat"],
        ),
        crs=4326,
    )
    joined = gpd.sjoin(
        points,
        countries,
        how="left",
        predicate="within",
    ).sort_values("wb_id").drop_duplicates("wb_id")
    missing = joined["ADMIN"].isna()
    if missing.any():
        nearest = gpd.sjoin_nearest(
            joined.loc[missing, ["wb_id", "geometry"]].to_crs(6933),
            countries.to_crs(6933),
            how="left",
            max_distance=250_000,
            distance_col="country_distance_m",
        ).sort_values("wb_id").drop_duplicates("wb_id")
        joined = joined.set_index("wb_id")
        nearest = nearest.set_index("wb_id")
        for column in ["ADMIN", "CONTINENT", "ADM0_A3"]:
            joined.loc[nearest.index, column] = nearest[column]
        joined = joined.reset_index()
    spatial = joined.set_index("wb_id")
    index = frame.loc[need_spatial].index
    keys = frame.loc[index, "wb_id"]
    spatial_country = keys.map(spatial["ADMIN"])
    spatial_continent = keys.map(spatial["CONTINENT"])
    spatial_iso = keys.map(spatial["ADM0_A3"])
    frame.loc[index, "country_v117"] = spatial_country.combine_first(
        frame.loc[index, "country_v117"]
    ).to_numpy()
    frame.loc[index, "continent_v117"] = spatial_continent.combine_first(
        frame.loc[index, "continent_v117"]
    ).to_numpy()
    frame.loc[index, "country_iso3_v117"] = spatial_iso.combine_first(
        frame.loc[index, "country_iso3_v117"]
    ).to_numpy()
    reservoir_spatial = frame.index.isin(index) & frame["wb_type"].eq("reservoir")
    frame.loc[reservoir_spatial, "country_source_v117"] = "natural_earth_centroid"
    other_spatial = frame.index.isin(index) & ~frame["wb_type"].eq("reservoir")
    frame.loc[other_spatial, "country_source_v117"] = "natural_earth_name_or_centroid"

    # Repeat the country-to-continent fill after spatial country assignment.
    # This catches external reservoirs whose source country was null and was
    # only recovered by the Natural Earth point join (for example Mauritius).
    post_spatial_reservoir = frame["wb_type"].eq("reservoir") & ~mapped_reservoir
    post_spatial_continent = frame.loc[
        post_spatial_reservoir, "country_v117"
    ].map(country_continent)
    post_spatial_index = post_spatial_continent.index[
        post_spatial_continent.notna()
    ]
    frame.loc[post_spatial_index, "continent_v117"] = post_spatial_continent.loc[
        post_spatial_index
    ]

    qc = {
        "waterbodies": int(len(frame)),
        "country_complete": int(frame["country_v117"].notna().sum()),
        "continent_complete": int(frame["continent_v117"].notna().sum()),
        "standard_continent_complete": int(
            frame["continent_v117"].isin(
                ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
            ).sum()
        ),
        "iso3_complete": int(frame["country_iso3_v117"].notna().sum()),
        "source_counts": {
            str(k): int(v)
            for k, v in frame["country_source_v117"].value_counts(dropna=False).items()
        },
        "boundary_source": "Natural Earth 1:10m Admin-0 Countries version 5.1.1",
        "assignment_rule": "HydroLAKES country attributes; mapped HydroLAKES attributes for external reservoirs; Natural Earth centroid assignment for remaining records and ISO codes",
    }
    return frame, qc


def add_depth_and_cost(frame: pd.DataFrame) -> pd.DataFrame:
    globathy = load_globathy(GLOBATHY_ZIP).rename(
        columns={
            "globathy_dmax_box_m_v116": "globathy_dmax_box_m_v117",
            "globathy_dmax_pa_m_v116": "globathy_dmax_pa_m_v117",
            "globathy_dmax_pavew_m_v116": "globathy_dmax_pavew_m_v117",
            "globathy_dmax_use_m_v116": "globathy_dmax_use_m_v117",
        }
    )
    frame = frame.merge(globathy, on="dryup_hylak_id_v113", how="left", validate="many_to_one")
    frame["annual_specific_yield_kwh_kwp_v117"] = (
        frame["annual_footprint_energy_kwh_m2"].astype(float) / PACKING_DENSITY_KWP_M2
    )
    assumptions = {
        "utility_pv_floor": {"capex_usd_kw": 691.0, "opex_fraction": 0.015, "wacc": 0.06},
        "fpv_reference": {"capex_usd_kw": 900.0, "opex_fraction": 0.020, "wacc": 0.08},
        "fpv_high_cost": {"capex_usd_kw": 1200.0, "opex_fraction": 0.025, "wacc": 0.10},
    }
    annual = frame["annual_specific_yield_kwh_kwp_v117"].to_numpy(float)
    for name, values in assumptions.items():
        frame[f"lcoe_{name}_usd_mwh_v117"] = present_value_lcoe(
            annual, **values
        ).astype(np.float32)
    return frame


def masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    known = frame["historical_dryup_evidence_known_v113"].fillna(False).to_numpy(bool)
    no_dry = frame["historical_no_dryup_observed_1991_2018_v113"].fillna(False).to_numpy(bool)
    core = frame["l2_core_no_dry_pass_v112"].fillna(False).to_numpy(bool)
    conservative = frame["l2_conservative_no_dry_pass_v112"].fillna(False).to_numpy(bool)
    road_lower = frame["road_within_10km_guaranteed_v115"].fillna(False).to_numpy(bool)
    road_upper = frame["road_within_10km_possible_v115"].fillna(False).to_numpy(bool)
    grid_lower = frame["grid_within_25km_observed_or_hydropower_v115"].fillna(False).to_numpy(bool)
    grid_upper = frame["grid_within_25km_possible_or_hydropower_v115"].fillna(False).to_numpy(bool)
    return {
        "all": np.ones(len(frame), dtype=bool),
        "l1": frame["l1_pass_v112"].fillna(False).to_numpy(bool),
        "core_l2_lower": core & known & no_dry,
        "core_l2_upper": core & (~known | no_dry),
        "conservative_l2_lower": conservative & known & no_dry,
        "conservative_l2_upper": conservative & (~known | no_dry),
        "core_l3_lower": core & known & no_dry & road_lower & grid_lower,
        "core_l3_upper": core & (~known | no_dry) & road_upper & grid_upper,
        "conservative_l3_lower": conservative & known & no_dry & road_lower & grid_lower,
        "conservative_l3_upper": conservative & (~known | no_dry) & road_upper & grid_upper,
    }


def level_summary(frame: pd.DataFrame) -> pd.DataFrame:
    selected = masks(frame)
    energy = frame["l0_type_specific_generation_gwh"].to_numpy(float)
    l0 = energy.sum() / 1000.0
    labels = [
        ("L0-reference", "all", True),
        ("L1-reference", "l1", True),
        ("L2-core-public-evidence-lower", "core_l2_lower", False),
        ("L2-core-public-evidence-upper", "core_l2_upper", False),
        ("L2-conservative-public-evidence-lower", "conservative_l2_lower", False),
        ("L2-conservative-public-evidence-upper", "conservative_l2_upper", False),
        ("L3-core-partial-access-lower", "core_l3_lower", False),
        ("L3-core-partial-access-upper", "core_l3_upper", False),
        ("L3-conservative-partial-access-lower", "conservative_l3_lower", False),
        ("L3-conservative-partial-access-upper", "conservative_l3_upper", False),
    ]
    rows = []
    for tier, key, complete in labels:
        mask = selected[key]
        total = float(energy[mask].sum() / 1000.0)
        rows.append(
            {
                "tier": tier,
                "scientific_tier_complete": complete,
                "eligible_waterbodies": int(mask.sum()),
                "generation_twh": total,
                "retention_vs_reference_l0_pct": total / l0 * 100.0,
            }
        )
    rows.extend(
        [
            {
                "tier": "L2-complete-published-1991-2020-criterion",
                "scientific_tier_complete": False,
                "eligible_waterbodies": pd.NA,
                "generation_twh": np.nan,
                "retention_vs_reference_l0_pct": np.nan,
            },
            {
                "tier": "L3-complete-deployable",
                "scientific_tier_complete": False,
                "eligible_waterbodies": pd.NA,
                "generation_twh": np.nan,
                "retention_vs_reference_l0_pct": np.nan,
            },
        ]
    )
    return pd.DataFrame(rows)


def coverage_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    energy_density = frame["annual_footprint_energy_kwh_m2"].to_numpy(float)
    area = frame["area_km2"].to_numpy(float)
    is_reservoir = frame["wb_type"].eq("reservoir").to_numpy(bool)
    l1 = frame["l1_pass_v112"].fillna(False).to_numpy(bool)
    scenarios = [
        ("uniform_5pct", 0.05, 0.05),
        ("uniform_10pct", 0.10, 0.10),
        ("uniform_20pct", 0.20, 0.20),
        ("uniform_30pct", 0.30, 0.30),
        ("reservoir20_lake10", 0.20, 0.10),
        ("reservoir30_lake10_reference", 0.30, 0.10),
        ("reservoir30_lake20", 0.30, 0.20),
    ]
    rows = []
    for name, reservoir_fraction, lake_fraction in scenarios:
        fraction = np.where(is_reservoir, reservoir_fraction, lake_fraction)
        generation = energy_density * np.minimum(area * fraction, MAX_FPV_FOOTPRINT_KM2)
        rows.append(
            {
                "scenario": name,
                "reservoir_coverage_fraction": reservoir_fraction,
                "lake_coverage_fraction": lake_fraction,
                "footprint_cap_km2": MAX_FPV_FOOTPRINT_KM2,
                "l0_generation_twh": float(generation.sum() / 1000.0),
                "l1_generation_twh": float(generation[l1].sum() / 1000.0),
                "l1_retention_pct": float(generation[l1].sum() / generation.sum() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def ice_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    energy = frame["l0_type_specific_generation_gwh"].to_numpy(float)
    proxy = frame["ice_months_temp_proxy"].le(MODEL_MAX_SUBZERO_MONTHS).to_numpy(bool)
    woolway_available = frame["woolway_ice_cover_fraction"].notna().to_numpy(bool)
    woolway = frame["woolway_ice_cover_fraction"].le(0.5).fillna(False).to_numpy(bool)
    direct_available = frame["li_ccr_valid_years_2002_2020_v117"].fillna(0).ge(
        DIRECT_MIN_VALID_YEARS
    ).to_numpy(bool)
    direct = frame["li_ccr_mean_icd_days_2002_2020_v117"].le(
        ICE_DURATION_THRESHOLD_DAYS
    ).fillna(False).to_numpy(bool)
    policies = {
        "reference_li_ccr_then_woolway_then_proxy": np.where(
            direct_available, direct, np.where(woolway_available, woolway, proxy)
        ),
        "woolway_then_proxy": np.where(woolway_available, woolway, proxy),
        "temperature_proxy_all": proxy,
    }
    rows = []
    for policy, passed in policies.items():
        rows.append(
            {
                "policy": policy,
                "eligible_waterbodies": int(passed.sum()),
                "generation_twh": float(energy[passed].sum() / 1000.0),
            }
        )
    overlap = direct_available & woolway_available
    rows.append(
        {
            "policy": "li_ccr_vs_woolway_overlap_disagreement",
            "eligible_waterbodies": int((direct[overlap] != woolway[overlap]).sum()),
            "generation_twh": float(
                energy[overlap & (direct != woolway)].sum() / 1000.0
            ),
        }
    )
    return pd.DataFrame(rows)


def yield_benchmark(frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    benchmark = pd.read_parquet(
        WOOLWAY_PATH, columns=["hylak_id", "woolway_fpv_output_kwh"]
    )
    comparison = frame.loc[
        frame["hylak_id"].notna(),
        ["hylak_id", "continent_v117", "annual_specific_yield_kwh_kwp_v117"],
    ].copy()
    comparison["hylak_id"] = comparison["hylak_id"].astype(int)
    comparison = comparison.merge(benchmark, on="hylak_id", how="inner", validate="one_to_one")
    comparison = comparison.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["annual_specific_yield_kwh_kwp_v117", "woolway_fpv_output_kwh"]
    )
    comparison = comparison.loc[comparison["woolway_fpv_output_kwh"].gt(0)].copy()
    comparison["difference_kwh_kwp"] = (
        comparison["annual_specific_yield_kwh_kwp_v117"]
        - comparison["woolway_fpv_output_kwh"]
    )

    def metrics(part: pd.DataFrame) -> dict[str, object]:
        diff = part["difference_kwh_kwp"]
        return {
            "n": int(len(part)),
            "pearson_r": float(
                part["annual_specific_yield_kwh_kwp_v117"].corr(
                    part["woolway_fpv_output_kwh"]
                )
            ),
            "bias_kwh_kwp": float(diff.mean()),
            "mae_kwh_kwp": float(diff.abs().mean()),
            "rmse_kwh_kwp": float(np.sqrt(np.mean(diff**2))),
            "median_relative_difference_pct": float(
                (
                    diff / part["woolway_fpv_output_kwh"] * 100.0
                ).median()
            ),
        }

    summary = {
        "comparison": "v117 monthly NASA POWER screen versus Woolway et al. published per-kW model output",
        "interpretation": "cross-model and climate-period benchmark, not operating-plant validation",
        **metrics(comparison),
    }
    regional = []
    for continent, part in comparison.groupby("continent_v117"):
        regional.append({"continent": continent, **metrics(part)})
    return summary, pd.DataFrame(regional)


def wind_sensitivity() -> pd.DataFrame:
    monthly = pd.read_parquet(MONTHLY_SIM_PATH)
    base = pd.read_parquet(
        ROOT / "_outputs/v110/data/waterbody_generation_rebuild.parquet",
        columns=["wb_id", "wb_type", "area_km2"],
    )
    poa_wm2 = (
        monthly["poa_kwh_m2"].to_numpy(float)
        * 1000.0
        / (24.0 * np.take(DAYS_IN_MONTH, monthly["month"].to_numpy(int) - 1))
    )
    temp = monthly["temp_c"].to_numpy(float)
    wind = monthly["wind_speed_ms"].to_numpy(float)
    rows = []
    for name, scale in [
        ("POWER_wind_reference", 1.0),
        ("wind_scaled_to_GEE_mean", 1.333 / 4.073),
        ("zero_wind_stress_test", 0.0),
    ]:
        cell = temp + poa_wm2 / (FAIMAN_U0 + FAIMAN_U1 * wind * scale)
        factor = np.clip(1.0 + TEMP_COEFF_POWER * (cell - T_REF_C), 0.75, 1.15)
        monthly_energy = monthly["poa_kwh_m2"].to_numpy(float) * 0.80 * factor * 0.10
        annual = pd.DataFrame(
            {"wb_id": monthly["wb_id"].to_numpy(int), "energy": monthly_energy}
        ).groupby("wb_id", as_index=False)["energy"].sum()
        joined = base.merge(annual, on="wb_id", validate="one_to_one")
        coverage = np.where(joined["wb_type"].eq("reservoir"), 0.30, 0.10)
        footprint = np.minimum(joined["area_km2"].to_numpy(float) * coverage, 30.0)
        total = float(np.sum(joined["energy"].to_numpy(float) * footprint) / 1000.0)
        rows.append({"wind_case": name, "legacy_scope_l0_twh": total})
    result = pd.DataFrame(rows)
    reference = result.loc[result["wind_case"].eq("POWER_wind_reference"), "legacy_scope_l0_twh"].iat[0]
    result["difference_vs_reference_pct"] = (
        result["legacy_scope_l0_twh"] / reference - 1.0
    ) * 100.0
    return result


def glev_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    energy = frame["l0_type_specific_generation_gwh"].to_numpy(float)
    known = frame["historical_dryup_evidence_known_v113"].fillna(False).to_numpy(bool)
    zeros = frame["glev_zero_months_1991_2018"].fillna(0).to_numpy(float)
    minimum = frame["glev_min_area_km2_1991_2018"].to_numpy(float)
    median = frame["glev_median_area_km2_1991_2018"].to_numpy(float)
    minimum_to_median = np.divide(
        minimum,
        median,
        out=np.full_like(minimum, np.nan, dtype=float),
        where=np.isfinite(median) & (median > 0),
    )
    criteria = {
        "any_exact_zero_month": zeros >= 1,
        "at_least_two_exact_zero_months": zeros >= 2,
        "minimum_area_at_most_1pct_of_median": np.isfinite(minimum)
        & np.isfinite(median)
        & (median > 0)
        & (minimum_to_median <= 0.01),
    }
    rows = []
    for conservation, base_column in [
        ("core", "l2_core_no_dry_pass_v112"),
        ("conservative", "l2_conservative_no_dry_pass_v112"),
    ]:
        base = frame[base_column].fillna(False).to_numpy(bool)
        for criterion, dried in criteria.items():
            for policy, evidence in [
                ("known_only_lower", known & ~dried),
                ("unknown_pass_upper", ~known | ~dried),
            ]:
                mask = base & evidence
                rows.append(
                    {
                        "conservation_definition": conservation,
                        "dryup_criterion": criterion,
                        "evidence_policy": policy,
                        "eligible_waterbodies": int(mask.sum()),
                        "generation_twh": float(energy[mask].sum() / 1000.0),
                        "unquantified_temporal_gap": "2019-2020 for all records",
                    }
                )
    return pd.DataFrame(rows)


def l3_factorial(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    energy = frame["l0_type_specific_generation_gwh"].to_numpy(float)
    known = frame["historical_dryup_evidence_known_v113"].fillna(False).to_numpy(bool)
    no_dry = frame["historical_no_dryup_observed_1991_2018_v113"].fillna(False).to_numpy(bool)
    road_lower = frame["road_within_10km_guaranteed_v115"].to_numpy(bool)
    road_upper = frame["road_within_10km_possible_v115"].to_numpy(bool)
    line_observed = frame["osm_powerline_sample_distance_km_v115"].le(25).to_numpy(bool)
    line_possible = frame["osm_powerline_distance_lower_bound_km_v115"].le(25).to_numpy(bool)
    hydro = frame["grand_hydropower_evidence"].to_numpy(bool)
    rows = []
    hydro_rows = []
    for conservation, column in [
        ("core", "l2_core_no_dry_pass_v112"),
        ("conservative", "l2_conservative_no_dry_pass_v112"),
    ]:
        base = frame[column].fillna(False).to_numpy(bool)
        for evidence_name, evidence in [
            ("known_only", known & no_dry),
            ("unknown_pass", ~known | no_dry),
        ]:
            for mapping_name, infrastructure in [
                ("lower_mapping", road_lower & (line_observed | hydro)),
                ("upper_mapping", road_upper & (line_possible | hydro)),
            ]:
                mask = base & evidence & infrastructure
                rows.append(
                    {
                        "conservation_definition": conservation,
                        "dryup_evidence_policy": evidence_name,
                        "infrastructure_mapping_policy": mapping_name,
                        "eligible_waterbodies": int(mask.sum()),
                        "generation_twh": float(energy[mask].sum() / 1000.0),
                    }
                )
            for mapping_name, road, line in [
                ("lower_mapping", road_lower, line_observed),
                ("upper_mapping", road_upper, line_possible),
            ]:
                with_hydro = base & evidence & road & (line | hydro)
                without_hydro = base & evidence & road & line
                hydro_rows.append(
                    {
                        "conservation_definition": conservation,
                        "dryup_evidence_policy": evidence_name,
                        "infrastructure_mapping_policy": mapping_name,
                        "with_hydropower_proxy_twh": float(energy[with_hydro].sum() / 1000.0),
                        "mapped_lines_only_twh": float(energy[without_hydro].sum() / 1000.0),
                        "hydropower_proxy_increment_twh": float(
                            energy[with_hydro & ~without_hydro].sum() / 1000.0
                        ),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(hydro_rows)


def regional_summary(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    scenario_masks = masks(frame)
    energy = frame["l0_type_specific_generation_gwh"].to_numpy(float)
    table = pd.DataFrame({group: frame[group]})
    for name, mask in scenario_masks.items():
        table[f"{name}_waterbodies"] = mask.astype(int)
        table[f"{name}_twh"] = np.where(mask, energy / 1000.0, 0.0)
    return table.groupby(group, dropna=False).sum(numeric_only=True).reset_index()


def engineering_summary(frame: pd.DataFrame) -> pd.DataFrame:
    depth = frame["globathy_dmax_use_m_v117"].notna()
    rows = []
    for population, mask in masks(frame).items():
        selected = frame.loc[mask]
        rows.append(
            {
                "population": population,
                "waterbodies": int(mask.sum()),
                "globathy_depth_known": int(depth[mask].sum()),
                "globathy_depth_coverage_pct": float(depth[mask].mean() * 100.0),
                "dmax_use_m_median_known": float(
                    selected["globathy_dmax_use_m_v117"].median()
                ),
                "fpv_reference_lcoe_usd_mwh_median": float(
                    selected["lcoe_fpv_reference_usd_mwh_v117"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def identity_sensitivity(
    spatial: pd.DataFrame, direct_ice: pd.DataFrame
) -> dict[str, object]:
    full = pd.read_parquet(V111_PATH)
    selected = spatial[["hylak_id"]].merge(full, on="hylak_id", validate="one_to_one")
    selected = selected.merge(direct_ice, on="hylak_id", how="left", validate="one_to_one")
    proxy = selected["ice_months_temp_proxy"].le(6)
    woolway_available = selected["woolway_ice_cover_fraction"].notna()
    passed = proxy.copy()
    passed.loc[woolway_available] = selected.loc[
        woolway_available, "woolway_ice_cover_fraction"
    ].le(0.5)
    direct_available = selected["li_ccr_valid_years_2002_2020_v117"].fillna(0).ge(5)
    passed.loc[direct_available] = selected.loc[
        direct_available, "li_ccr_mean_icd_days_2002_2020_v117"
    ].le(182.5)
    return {
        "spatial_only_matches": int(len(selected)),
        "calibrated_spatial_match_precision_pct": 98.5625,
        "reference_policy": "excluded as high-confidence duplicates",
        "authoritative_only_dedup_sensitivity_rows": int(REFERENCE_ROWS + len(selected)),
        "authoritative_only_dedup_added_l0_twh": float(
            selected["l0_type_generation_gwh"].sum() / 1000.0
        ),
        "authoritative_only_dedup_added_l1_twh": float(
            selected.loc[passed, "l0_type_generation_gwh"].sum() / 1000.0
        ),
    }


def validation_gate(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    country_qc: dict[str, object],
) -> dict[str, object]:
    by_tier = summary.set_index("tier")
    checks = {
        "reference_rows_199976": len(frame) == REFERENCE_ROWS,
        "unique_wb_id": bool(frame["wb_id"].is_unique),
        "reference_additions_1245": int(
            frame["inventory_action_v117"].eq("restored_no_duplicate_evidence").sum()
        )
        == REFERENCE_ADDITIONS,
        "country_complete": country_qc["country_complete"] == REFERENCE_ROWS,
        "continent_complete": country_qc["continent_complete"] == REFERENCE_ROWS,
        "standard_continent_complete": country_qc["standard_continent_complete"]
        == REFERENCE_ROWS,
        "l1_not_above_l0": by_tier.loc["L1-reference", "generation_twh"]
        <= by_tier.loc["L0-reference", "generation_twh"],
        "core_l2_monotonic": by_tier.loc[
            "L2-core-public-evidence-lower", "generation_twh"
        ]
        <= by_tier.loc["L2-core-public-evidence-upper", "generation_twh"]
        <= by_tier.loc["L1-reference", "generation_twh"],
        "conservative_l2_monotonic": by_tier.loc[
            "L2-conservative-public-evidence-lower", "generation_twh"
        ]
        <= by_tier.loc["L2-conservative-public-evidence-upper", "generation_twh"]
        <= by_tier.loc["L1-reference", "generation_twh"],
        "core_l3_monotonic": by_tier.loc[
            "L3-core-partial-access-lower", "generation_twh"
        ]
        <= by_tier.loc["L3-core-partial-access-upper", "generation_twh"]
        <= by_tier.loc["L2-core-public-evidence-upper", "generation_twh"],
        "complete_l2_is_na": bool(
            pd.isna(
                by_tier.loc[
                    "L2-complete-published-1991-2020-criterion", "generation_twh"
                ]
            )
        ),
        "complete_l3_is_na": bool(
            pd.isna(by_tier.loc["L3-complete-deployable", "generation_twh"])
        ),
        "glev_expected_months": bool(
            frame.loc[
                frame["historical_dryup_evidence_known_v113"].fillna(False),
                "glev_valid_months_1991_2018",
            ].eq(EXPECTED_MONTHS).all()
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "passed": int(sum(checks.values())),
        "total": int(len(checks)),
        "complete_l2": False,
        "complete_l3": False,
    }


def run(run_dir: Path) -> None:
    data_dir = run_dir / "data"
    reports = run_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    print("Loading v116 and raw HydroLAKES scope", flush=True)
    v116 = pd.read_parquet(V116_PATH)
    if len(v116) != V116_ROWS or not v116["wb_id"].is_unique:
        raise ValueError("v116 source mismatch")
    raw_scope = load_hydrolakes_scope()
    adjudication, additions, spatial = adjudicate_inventory(v116, raw_scope)
    adjudication.to_csv(reports / "inventory_adjudication.csv", index=False)

    target_ice_ids = pd.concat([additions["hylak_id"], spatial["hylak_id"]])
    direct_ice = parse_li_ccr_targets(target_ice_ids)
    identity_qc = identity_sensitivity(spatial, direct_ice)
    identity_qc.update(
        {
            "raw_hydrolakes_scope_rows": RAW_HYDROLAKES_SCOPE_ROWS,
            "v116_reference_rows": V116_ROWS,
            "removed_from_raw_direct_branch": int(len(adjudication)),
            "authoritative_duplicate_removals": AUTHORITATIVE_REMOVALS,
            "spatial_high_confidence_reference_removals": SPATIAL_HIGH_CONFIDENCE_REMOVALS,
            "unsupported_deletions_restored": REFERENCE_ADDITIONS,
            "restored_area_km2": float(additions["area_km2"].sum()),
            "v117_reference_rows": REFERENCE_ROWS,
        }
    )
    json_dump(reports / "inventory_adjudication_qc.json", identity_qc)

    print("Recovering restored lake geometries and constraints", flush=True)
    geometries = load_target_geometries(additions["hylak_id"])
    added = build_added_rows(additions, geometries, direct_ice, run_dir)

    v116["scope_reference_v117"] = True
    v116["inventory_action_v117"] = "retained_v116_reference"
    frame = attach_current_ice_details(v116, added)
    additions_mask = frame["inventory_action_v117"].eq(
        "restored_no_duplicate_evidence"
    ).to_numpy(bool)
    print("Computing restored-row infrastructure proximity", flush=True)
    frame = add_accessibility(frame, additions_mask)
    print("Assigning country and continent fields", flush=True)
    frame, country_qc = add_country_attribution(frame)
    json_dump(reports / "country_attribution_qc.json", country_qc)
    print("Adding depth and cost sensitivities", flush=True)
    frame = add_depth_and_cost(frame)
    frame.to_parquet(data_dir / "fpv_reference_inventory_v117.parquet", index=False)

    summary = level_summary(frame)
    summary.to_csv(reports / "level_summary_v117.csv", index=False)
    coverage_sensitivity(frame).to_csv(reports / "coverage_sensitivity_v117.csv", index=False)
    ice_sensitivity(frame).to_csv(reports / "ice_sensitivity_v117.csv", index=False)
    benchmark_summary, benchmark_regional = yield_benchmark(frame)
    json_dump(reports / "yield_benchmark_summary_v117.json", benchmark_summary)
    benchmark_regional.to_csv(reports / "yield_benchmark_by_continent_v117.csv", index=False)
    wind_sensitivity().to_csv(reports / "wind_sensitivity_v117.csv", index=False)
    glev_sensitivity(frame).to_csv(reports / "glev_criterion_sensitivity_v117.csv", index=False)
    factorial, hydro = l3_factorial(frame)
    factorial.to_csv(reports / "l3_factorial_v117.csv", index=False)
    hydro.to_csv(reports / "l3_hydropower_proxy_sensitivity_v117.csv", index=False)
    regional_summary(frame, "country_v117").to_csv(
        reports / "country_level_summary_v117.csv", index=False
    )
    regional_summary(frame, "continent_v117").to_csv(
        reports / "continent_level_summary_v117.csv", index=False
    )
    engineering_summary(frame).to_csv(
        reports / "engineering_economic_coverage_v117.csv", index=False
    )

    gate = validation_gate(frame, summary, country_qc)
    gate["inventory_identity_sensitivity"] = identity_qc
    gate["yield_benchmark"] = benchmark_summary
    gate["cost_basis"] = {
        "utility_pv_floor_capex_usd_kw": 691,
        "source": "IRENA global weighted-average utility-scale solar PV installed cost for 2024",
        "fpv_reference_capex_usd_kw": 900,
        "fpv_high_cost_capex_usd_kw": 1200,
        "interpretation": "parametric sensitivity only; not a project quote or bankability assessment",
    }
    json_dump(reports / "consolidated_validation_gate_v117.json", gate)
    print(summary.to_string(index=False))
    print((reports / "consolidated_validation_gate_v117.json").read_text(encoding="utf-8"))
    if gate["status"] != "pass":
        raise RuntimeError("v117 validation gate failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("_outputs/v117"))
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run(run_dir.resolve())


if __name__ == "__main__":
    main()
