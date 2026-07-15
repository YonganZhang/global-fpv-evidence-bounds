#!/usr/bin/env python3
"""Add auditable historical dry-up evidence from the public GLEV area series.

GLEV supplies monthly reconstructed surface area for HydroLAKES water bodies
through 2018.  Woolway et al. screen any lake that dried during 1991--2020,
but their public per-lake table omits that flag and the exact monthly series.
This rebuild therefore treats an explicit zero-area month in 1991--2018 as
positive dry-up evidence, and preserves 2019--2020 plus unmatched reservoirs
as unknown.  It reports an evidence-only lower bound and an unknown-pass upper
envelope; neither is mislabeled as the complete paper-equivalent L2 result.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
from scipy.spatial import cKDTree


EXPECTED_SCOPE_ROWS = 198_737
EXPECTED_MONTHS = 28 * 12
START_DATE = "1991-01-01"
END_DATE = "2018-12-01"
SPATIAL_MAX_DISTANCE_KM = 1.0
SPATIAL_MIN_AREA_RATIO = 0.25
SPATIAL_MAX_AREA_RATIO = 4.0


def spherical_xyz(longitude: pd.Series, latitude: pd.Series) -> np.ndarray:
    lon = np.deg2rad(longitude.to_numpy(float))
    lat = np.deg2rad(latitude.to_numpy(float))
    cosine = np.cos(lat)
    return np.column_stack(
        [cosine * np.cos(lon), cosine * np.sin(lon), np.sin(lat)]
    )


def build_reservoir_hylak_crosswalk(
    frame: pd.DataFrame,
    coordinate_path: Path,
    reservoir_gpkg: Path,
    hydrolakes_dbf: Path,
    benchmark_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Recover external-reservoir HydroLAKES keys in two confidence tiers."""
    coordinates = pd.read_parquet(
        coordinate_path,
        columns=["wb_id", "wb_type", "centroid_lon", "centroid_lat"],
    )
    reservoirs = frame.loc[frame["wb_type"].eq("reservoir"), ["wb_id", "area_km2"]].merge(
        coordinates.drop(columns="wb_type"), on="wb_id", how="left", validate="one_to_one"
    )

    with sqlite3.connect(reservoir_gpkg) as connection:
        grand_links = pd.read_sql_query(
            "select merge_id as wb_id, cast(grand_id_link as integer) as grand_id "
            "from reservoirs where grand_id_link is not null",
            connection,
        )
    reader = shapefile.Reader(dbf=hydrolakes_dbf, encoding="latin1")
    hydrolakes_grand: list[tuple[int, int]] = []
    for record in reader.iterRecords(fields=["Hylak_id", "Grand_id"]):
        grand_id = int(record["Grand_id"])
        if grand_id > 0:
            hydrolakes_grand.append((int(record["Hylak_id"]), grand_id))
    grand_lookup = pd.DataFrame(hydrolakes_grand, columns=["mapped_hylak_id", "grand_id"])
    if grand_lookup["grand_id"].duplicated().any():
        raise ValueError("HydroLAKES Grand_id crosswalk is not one-to-one")
    authoritative = grand_links.merge(
        grand_lookup, on="grand_id", how="inner", validate="many_to_one"
    )[["wb_id", "mapped_hylak_id", "grand_id"]]
    authoritative["dryup_key_source_v113"] = "grand_id_authoritative"

    benchmark = pd.read_parquet(
        benchmark_path,
        columns=["hylak_id", "longitude", "latitude", "median_surface_area_m2"],
    ).dropna(subset=["longitude", "latitude"])
    tree = cKDTree(spherical_xyz(benchmark["longitude"], benchmark["latitude"]))
    chord, nearest_index = tree.query(
        spherical_xyz(reservoirs["centroid_lon"], reservoirs["centroid_lat"]), k=1
    )
    nearest = benchmark.iloc[nearest_index].reset_index(drop=True)
    reservoirs["nearest_hylak_id"] = nearest["hylak_id"].to_numpy(np.int64)
    reservoirs["nearest_distance_km"] = 6371.0088 * 2.0 * np.arcsin(
        np.minimum(1.0, chord / 2.0)
    )
    reservoirs["nearest_median_area_km2"] = (
        nearest["median_surface_area_m2"].to_numpy(float) / 1_000_000.0
    )
    reservoirs["nearest_area_ratio"] = (
        reservoirs["area_km2"] / reservoirs["nearest_median_area_km2"]
    )
    calibrated = reservoirs.merge(
        authoritative[["wb_id", "mapped_hylak_id"]],
        on="wb_id",
        how="left",
        validate="one_to_one",
    )
    threshold = (
        calibrated["nearest_distance_km"].le(SPATIAL_MAX_DISTANCE_KM)
        & calibrated["nearest_area_ratio"].between(
            SPATIAL_MIN_AREA_RATIO, SPATIAL_MAX_AREA_RATIO
        )
    )
    validation = calibrated["mapped_hylak_id"].notna() & threshold
    validation_precision = float(
        calibrated.loc[validation, "nearest_hylak_id"]
        .eq(calibrated.loc[validation, "mapped_hylak_id"])
        .mean()
    )

    candidate = calibrated.loc[
        calibrated["mapped_hylak_id"].isna() & threshold,
        [
            "wb_id",
            "nearest_hylak_id",
            "nearest_distance_km",
            "nearest_area_ratio",
        ],
    ].copy()
    occupied = set(frame["hylak_id"].dropna().astype(int)) | set(
        authoritative["mapped_hylak_id"].astype(int)
    )
    occupied_exclusions = int(candidate["nearest_hylak_id"].isin(occupied).sum())
    candidate = candidate.loc[~candidate["nearest_hylak_id"].isin(occupied)].copy()
    candidate["mapping_score"] = candidate["nearest_distance_km"] + np.abs(
        np.log(candidate["nearest_area_ratio"])
    )
    duplicate_target_rows = int(
        candidate["nearest_hylak_id"].duplicated(keep=False).sum()
    )
    candidate = candidate.sort_values("mapping_score").drop_duplicates(
        "nearest_hylak_id", keep="first"
    )
    spatial = candidate.rename(columns={"nearest_hylak_id": "mapped_hylak_id"})
    spatial["grand_id"] = pd.NA
    spatial["dryup_key_source_v113"] = "spatial_nearest_high_confidence"
    spatial = spatial[
        [
            "wb_id",
            "mapped_hylak_id",
            "grand_id",
            "dryup_key_source_v113",
            "nearest_distance_km",
            "nearest_area_ratio",
        ]
    ]
    authoritative["nearest_distance_km"] = np.nan
    authoritative["nearest_area_ratio"] = np.nan
    crosswalk = pd.concat([authoritative, spatial], ignore_index=True)
    if crosswalk["wb_id"].duplicated().any() or crosswalk["mapped_hylak_id"].duplicated().any():
        raise ValueError("reservoir-to-HydroLAKES crosswalk is not one-to-one")
    crosswalk["mapped_hylak_id"] = crosswalk["mapped_hylak_id"].astype("int64")
    qc = {
        "reservoir_rows": int(len(reservoirs)),
        "reservoirs_with_grand_id": int(len(grand_links)),
        "grand_id_authoritative_matches": int(len(authoritative)),
        "spatial_thresholds": {
            "maximum_distance_km": SPATIAL_MAX_DISTANCE_KM,
            "area_ratio_range": [SPATIAL_MIN_AREA_RATIO, SPATIAL_MAX_AREA_RATIO],
        },
        "spatial_validation_rows_with_authoritative_truth": int(validation.sum()),
        "spatial_validation_precision": validation_precision,
        "spatial_candidates_excluded_occupied_hylak_id": occupied_exclusions,
        "spatial_duplicate_target_rows_before_resolution": duplicate_target_rows,
        "spatial_high_confidence_matches": int(len(spatial)),
        "total_crosswalk_matches": int(len(crosswalk)),
    }
    return crosswalk, qc


def read_glev_subset(
    zip_path: Path, target_ids: pd.Series, cache_path: Path
) -> tuple[pd.DataFrame, list[str]]:
    """Stream GLEV in chunks and retain only paper-scope crosswalk keys."""
    ids = target_ids.dropna().astype("int64")
    target = set(ids.tolist())
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one CSV in GLEV archive, found {members}")
        member = members[0]
        with archive.open(member) as stream:
            header = stream.readline().decode("utf-8").rstrip("\r\n").split(",")

    month_columns = [name for name in header[1:] if START_DATE <= name <= END_DATE]
    if len(month_columns) != EXPECTED_MONTHS:
        raise ValueError(
            f"expected {EXPECTED_MONTHS} monthly columns, found {len(month_columns)}"
        )
    if cache_path.exists():
        raw = pd.read_parquet(cache_path)
        missing = target - set(raw["Hylak_id"].astype(int))
        # Hylak_id 1 (Caspian Sea in HydroLAKES) is absent from public GLEV.
        if missing.issubset({1}) and set(raw["Hylak_id"].astype(int)).issubset(target):
            print(f"  reused cached GLEV scope subset: {len(raw):,} rows", flush=True)
            return raw, month_columns
    dtype = {name: "float32" for name in month_columns}
    retained: list[pd.DataFrame] = []
    rows_seen = 0
    for chunk in pd.read_csv(
        zip_path,
        usecols=["Hylak_id", *month_columns],
        dtype=dtype,
        chunksize=250_000,
    ):
        rows_seen += len(chunk)
        selected = chunk.loc[chunk["Hylak_id"].isin(target)]
        if len(selected):
            retained.append(selected)
        print(
            f"  streamed {rows_seen:,} GLEV rows; retained "
            f"{sum(len(part) for part in retained):,}",
            flush=True,
        )
    raw = pd.concat(retained, ignore_index=True)
    if not raw["Hylak_id"].is_monotonic_increasing:
        raise ValueError("GLEV Hylak_id is no longer sorted")
    raw["Hylak_id"] = raw["Hylak_id"].astype("int64")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cache_path, index=False)
    return raw, month_columns


def summarize_glev(raw: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    values = raw[months].to_numpy(dtype=np.float32, copy=False)
    finite = np.isfinite(values)
    valid_months = finite.sum(axis=1)
    safe = np.where(finite, values, np.nan)
    minimum = np.nanmin(safe, axis=1)
    median = np.nanmedian(safe, axis=1)
    zero_months = np.sum(values == 0.0, axis=1)
    negative_months = np.sum(values < 0.0, axis=1)
    known = valid_months == EXPECTED_MONTHS
    result = pd.DataFrame(
        {
            "hylak_id": raw["Hylak_id"].to_numpy(dtype=np.int64),
            "glev_valid_months_1991_2018": valid_months.astype(np.int16),
            "glev_min_area_km2_1991_2018": minimum.astype(np.float32),
            "glev_median_area_km2_1991_2018": median.astype(np.float32),
            "glev_zero_months_1991_2018": zero_months.astype(np.int16),
            "glev_negative_months_1991_2018": negative_months.astype(np.int16),
            "historical_dryup_evidence_known_v113": known,
            "historical_dryup_observed_1991_2018_v113": known & (zero_months > 0),
            "historical_no_dryup_observed_1991_2018_v113": known & (zero_months == 0),
        }
    )
    return result


def add_nullable_pass(frame: pd.DataFrame) -> pd.DataFrame:
    for column in [
        "historical_dryup_evidence_known_v113",
        "historical_dryup_observed_1991_2018_v113",
        "historical_no_dryup_observed_1991_2018_v113",
    ]:
        frame[column] = frame[column].astype("boolean")
    known = frame["historical_dryup_evidence_known_v113"].fillna(False).astype(bool)
    observed = frame["historical_dryup_observed_1991_2018_v113"].fillna(False).astype(bool)
    dry_pass = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    dry_pass.loc[known] = ~observed.loc[known]
    frame["historical_dryup_pass_1991_2018_v113"] = dry_pass
    frame["historical_dryup_coverage_v113"] = np.where(
        known,
        "known_1991_2018",
        np.where(
            frame["dryup_hylak_id_v113"].isna(),
            "unknown_no_reliable_hylak_crosswalk",
            "unknown_no_glev_match_or_incomplete",
        ),
    )
    return frame


def scenario_row(
    tier: str,
    definition: str,
    mask: np.ndarray | None,
    energy: np.ndarray,
    l0_twh: float,
    complete: bool = False,
) -> dict[str, object]:
    generation_twh = float(energy[mask].sum() / 1000.0) if mask is not None else np.nan
    return {
        "tier": tier,
        "definition": definition,
        "scientific_tier_complete": complete,
        "eligible_waterbodies": int(mask.sum()) if mask is not None else pd.NA,
        "generation_twh": generation_twh,
        "retention_vs_recommended_l0_pct": generation_twh / l0_twh * 100.0
        if mask is not None
        else np.nan,
    }


def make_summary(frame: pd.DataFrame, baseline_path: Path) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_path)
    keep = baseline[baseline["tier"].isin([
        "L0-paper-uniform",
        "L0-recommended-type",
        "L1-paper-uniform-hybrid-ice",
        "L1-recommended-type-hybrid-ice",
    ])].copy()
    keep = keep.drop(columns=["retention_vs_corresponding_l0_pct"], errors="ignore")
    energy = frame["l0_type_specific_generation_gwh"].to_numpy(float)
    l0_twh = float(energy.sum() / 1000.0)
    known = frame["historical_dryup_evidence_known_v113"].fillna(False).to_numpy(bool)
    no_dry = frame["historical_no_dryup_observed_1991_2018_v113"].fillna(False).to_numpy(bool)
    core = frame["l2_core_no_dry_pass_v112"].to_numpy(bool)
    conservative = frame["l2_conservative_no_dry_pass_v112"].to_numpy(bool)
    rows = [
        scenario_row(
            "L2-core-historical-known-pass-lower-bound-type",
            "L1 + WDPA I-IV/population gates + complete public GLEV evidence and no zero-area month in 1991-2018; unknowns fail",
            core & known & no_dry,
            energy,
            l0_twh,
        ),
        scenario_row(
            "L2-core-historical-unknown-pass-upper-envelope-type",
            "Same core gates; explicit 1991-2018 dry-ups fail while unavailable 2019-2020 and unmatched water bodies provisionally pass",
            core & (~known | no_dry),
            energy,
            l0_twh,
        ),
        scenario_row(
            "L2-conservative-historical-known-pass-lower-bound-type",
            "L1 + any WDPA/Ramsar/population gates + complete public GLEV evidence and no zero-area month in 1991-2018; unknowns fail",
            conservative & known & no_dry,
            energy,
            l0_twh,
        ),
        scenario_row(
            "L2-conservative-historical-unknown-pass-upper-envelope-type",
            "Same conservative gates; explicit 1991-2018 dry-ups fail while unavailable 2019-2020 and unmatched water bodies provisionally pass",
            conservative & (~known | no_dry),
            energy,
            l0_twh,
        ),
        scenario_row(
            "L2-complete-paper-equivalent",
            "Unavailable: public evidence ends in 2018, some external reservoirs still lack reliable HydroLAKES keys, and the authors did not publish their dry-up flag",
            None,
            energy,
            l0_twh,
        ),
        scenario_row(
            "L3",
            "Unavailable: road/grid/engineering/economic/permitting gates remain incomplete",
            None,
            energy,
            l0_twh,
        ),
    ]
    return pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)


def validation_eda(
    frame: pd.DataFrame, benchmark_path: Path
) -> tuple[dict[str, object], pd.DataFrame]:
    benchmark = pd.read_parquet(
        benchmark_path, columns=["hylak_id", "median_surface_area_m2"]
    )
    comparison = frame.loc[
        frame["glev_median_area_km2_1991_2018"].notna(),
        ["dryup_hylak_id_v113", "glev_median_area_km2_1991_2018"],
    ].drop_duplicates("dryup_hylak_id_v113").rename(
        columns={"dryup_hylak_id_v113": "hylak_id"}
    ).merge(
        benchmark, on="hylak_id", how="inner", validate="one_to_one"
    )
    comparison = comparison.loc[
        comparison["median_surface_area_m2"].gt(0)
        & comparison["glev_median_area_km2_1991_2018"].gt(0)
    ].copy()
    comparison["woolway_median_area_km2_1991_2020"] = (
        comparison["median_surface_area_m2"] / 1_000_000.0
    )
    comparison["glev_to_woolway_median_ratio"] = (
        comparison["glev_median_area_km2_1991_2018"]
        / comparison["woolway_median_area_km2_1991_2020"]
    )
    log_a = np.log10(comparison["glev_median_area_km2_1991_2018"])
    log_b = np.log10(comparison["woolway_median_area_km2_1991_2020"])
    ratio = comparison["glev_to_woolway_median_ratio"]
    eda = {
        "comparison_rows_positive": int(len(comparison)),
        "pearson_log10_area": float(log_a.corr(log_b)),
        "ratio_median": float(ratio.median()),
        "ratio_p05": float(ratio.quantile(0.05)),
        "ratio_p95": float(ratio.quantile(0.95)),
    }
    return eda, comparison


def run(
    source_path: Path,
    coordinate_path: Path,
    zip_path: Path,
    benchmark_path: Path,
    reservoir_gpkg: Path,
    hydrolakes_dbf: Path,
    output_dir: Path,
) -> None:
    frame = pd.read_parquet(source_path)
    if len(frame) != EXPECTED_SCOPE_ROWS or not frame["wb_id"].is_unique:
        raise ValueError("declared paper-scope source row/key mismatch")

    crosswalk, crosswalk_qc = build_reservoir_hylak_crosswalk(
        frame,
        coordinate_path,
        reservoir_gpkg,
        hydrolakes_dbf,
        benchmark_path,
    )
    frame = frame.merge(crosswalk, on="wb_id", how="left", validate="one_to_one")
    frame["dryup_hylak_id_v113"] = frame["hylak_id"].astype("Int64")
    mapped = frame["mapped_hylak_id"].notna()
    frame.loc[mapped, "dryup_hylak_id_v113"] = frame.loc[
        mapped, "mapped_hylak_id"
    ].astype("int64")
    missing_source = frame["dryup_key_source_v113"].isna()
    frame.loc[missing_source, "dryup_key_source_v113"] = np.where(
        frame.loc[missing_source, "hylak_id"].notna(),
        "hydrolakes_direct",
        "unmatched",
    )

    assigned = frame["dryup_hylak_id_v113"].notna()
    duplicated = assigned & frame["dryup_hylak_id_v113"].duplicated(keep=False)
    crosswalk_qc["scope_rows_in_duplicate_hylak_groups"] = int(duplicated.sum())
    crosswalk_qc["scope_duplicate_hylak_groups"] = int(
        frame.loc[duplicated, "dryup_hylak_id_v113"].nunique()
    )

    raw, months = read_glev_subset(
        zip_path,
        frame["dryup_hylak_id_v113"],
        output_dir / "raw/glev/glev_scope_1991_2018.parquet",
    )
    flags = summarize_glev(raw, months)
    flags = flags.rename(columns={"hylak_id": "dryup_hylak_id_v113"})
    frame = frame.merge(
        flags, on="dryup_hylak_id_v113", how="left", validate="many_to_one"
    )
    frame = add_nullable_pass(frame)
    summary = make_summary(
        frame, source_path.parent.parent / "reports/paper_scope_level_summary.csv"
    )
    eda, comparison = validation_eda(frame, benchmark_path)

    data_dir = output_dir / "data"
    reports_dir = output_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(data_dir / "paper_scope_historical_dryup_generation.parquet", index=False)
    flag_columns = [
        "wb_id",
        "hylak_id",
        "wb_type",
        "dryup_hylak_id_v113",
        "dryup_key_source_v113",
        "grand_id",
        "nearest_distance_km",
        "nearest_area_ratio",
        "historical_dryup_coverage_v113",
        "glev_valid_months_1991_2018",
        "glev_min_area_km2_1991_2018",
        "glev_median_area_km2_1991_2018",
        "glev_zero_months_1991_2018",
        "historical_dryup_evidence_known_v113",
        "historical_dryup_observed_1991_2018_v113",
        "historical_dryup_pass_1991_2018_v113",
    ]
    frame[flag_columns].to_parquet(
        data_dir / "waterbody_historical_dryup_flags.parquet", index=False
    )
    summary.to_csv(reports_dir / "paper_scope_level_summary.csv", index=False)
    comparison.to_parquet(reports_dir / "glev_woolway_area_comparison.parquet", index=False)

    coverage = (
        frame.groupby(["wb_type", "historical_dryup_coverage_v113"], dropna=False)
        .agg(
            waterbodies=("wb_id", "size"),
            core_candidates=("l2_core_no_dry_pass_v112", "sum"),
            conservative_candidates=("l2_conservative_no_dry_pass_v112", "sum"),
        )
        .reset_index()
    )
    coverage.to_csv(reports_dir / "historical_dryup_coverage_by_type.csv", index=False)
    qc = {
        "scope_rows": int(len(frame)),
        "glev_source_period": [START_DATE, END_DATE],
        "glev_expected_months": EXPECTED_MONTHS,
        "reservoir_crosswalk": crosswalk_qc,
        "glev_matched_rows": int(frame["glev_valid_months_1991_2018"].notna().sum()),
        "known_complete_rows": int(
            frame["historical_dryup_evidence_known_v113"].fillna(False).sum()
        ),
        "observed_dryup_rows_1991_2018": int(
            frame["historical_dryup_observed_1991_2018_v113"].fillna(False).sum()
        ),
        "unknown_rows": int(
            frame["historical_dryup_pass_1991_2018_v113"].isna().sum()
        ),
        "negative_area_months": int(
            frame["glev_negative_months_1991_2018"].fillna(0).sum()
        ),
        "definition": "dry-up observed when any reconstructed monthly surface area equals exactly 0 km2 during 1991-2018",
        "limitations": [
            "GLEV public area series ends in 2018, two years before the paper study period",
            "some external reservoirs still lack an authoritative or high-confidence HydroLAKES crosswalk",
            "the paper's author-supplied per-lake table does not expose the dry-up flag",
        ],
        "area_cross_validation": eda,
        "l2_complete": False,
        "l3_complete": False,
    }
    (reports_dir / "historical_dryup_qc.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(qc, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-path",
        type=Path,
        default=Path("_outputs/v112/data/paper_scope_hybrid_generation.parquet"),
    )
    parser.add_argument(
        "--glev-zip",
        type=Path,
        default=Path("_outputs/v113/raw/glev/1_surfacewater_area.zip"),
    )
    parser.add_argument(
        "--coordinate-path",
        type=Path,
        default=Path("_outputs/v110/data/waterbody_generation_rebuild.parquet"),
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("_outputs/v110/data/woolway_public_lake_info.parquet"),
    )
    parser.add_argument(
        "--reservoir-gpkg",
        type=Path,
        default=Path("Dataset/data_processed/merged/merged_reservoirs.gpkg"),
    )
    parser.add_argument(
        "--hydrolakes-dbf",
        type=Path,
        default=Path(
            "Dataset/data_processed/HydroLAKES/HydroLAKES_polys_v10_shp/"
            "HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.dbf"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_outputs/v113"))
    args = parser.parse_args()
    run(
        args.source_path,
        args.coordinate_path,
        args.glev_zip,
        args.benchmark_path,
        args.reservoir_gpkg,
        args.hydrolakes_dbf,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
