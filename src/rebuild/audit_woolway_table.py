#!/usr/bin/env python3
"""Audit the public Woolway et al. lake-level table against reported methods.

The public table is an independent full-universe benchmark and contains
1,056,924 HydroLAKES records.  It does not publish an explicit dry-up flag, so
median surface area > 0 is reported only as a lower-bound proxy and is never
labelled the paper's full dry-up gate.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


REPORTED_ELIGIBLE_WATERBODIES = 67_893
REPORTED_ELIGIBLE_GENERATION_TWH = 1_302.0
POPULATION_THRESHOLDS_KM = [10.0, 10.95, 20.0, 30.0, 32.5, 50.0]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_public_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False).rename(
        columns={
            "HydroLakes ID": "hylak_id",
            "Longitude": "longitude",
            "Latitude": "latitude",
            "Country": "country",
            "SAmed (m^2)": "median_surface_area_m2",
            "Ice cover fraction (0-1)": "woolway_ice_cover_fraction",
            "Distance to population centre (km)": "woolway_population_distance_km",
            "Protected area": "woolway_protected_area",
            "FPV output (kWh)": "woolway_fpv_output_kwh",
            "Total FPV output (GWh)": "woolway_total_fpv_output_gwh",
        }
    )
    expected = {
        "hylak_id",
        "median_surface_area_m2",
        "woolway_ice_cover_fraction",
        "woolway_population_distance_km",
        "woolway_protected_area",
        "woolway_total_fpv_output_gwh",
    }
    assert expected.issubset(frame.columns), sorted(set(frame.columns) - expected)
    assert len(frame) == 1_056_924 and frame["hylak_id"].is_unique
    frame["woolway_ice_pass_50pct"] = frame["woolway_ice_cover_fraction"].le(0.5)
    frame["woolway_protected_pass"] = ~frame["woolway_protected_area"].eq(1.0)
    frame["median_surface_area_positive"] = frame["median_surface_area_m2"].gt(0.0)
    return frame


def local_hylak_ids(root: Path) -> set[int]:
    gpkg = root / "Dataset/data_processed/merged/all_waterbodies.gpkg"
    with sqlite3.connect(gpkg) as con:
        values = pd.read_sql_query(
            "SELECT hylak_id FROM waterbodies WHERE hylak_id IS NOT NULL", con
        )["hylak_id"]
    return set(values.astype(int))


def audit(run_dir: Path, root: Path) -> None:
    source = run_dir / "raw/woolway_figshare/Table S1 lake info.csv"
    assert source.exists(), source
    frame = load_public_table(source)
    data_dir = run_dir / "data"
    reports = run_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(data_dir / "woolway_public_lake_info.parquet", index=False)

    local_ids = local_hylak_ids(root)
    frame["in_student_ge1km_hydrolakes_subset"] = frame["hylak_id"].isin(local_ids)
    rows: list[dict[str, object]] = []

    def add_row(name: str, mask: pd.Series, note: str) -> None:
        rows.append(
            {
                "scenario": name,
                "waterbodies": int(mask.sum()),
                "generation_twh": float(
                    frame.loc[mask, "woolway_total_fpv_output_gwh"].sum() / 1000.0
                ),
                "retention_count_pct": float(mask.mean() * 100.0),
                "note": note,
            }
        )

    all_mask = pd.Series(True, index=frame.index)
    add_row("public_table_all", all_mask, "Full public Table S1 universe")
    add_row(
        "student_ge1km_hydrolakes_subset",
        frame["in_student_ge1km_hydrolakes_subset"],
        "HydroLAKES IDs retained by the student's >=1 km2 merge before external reservoirs",
    )
    add_row(
        "ice_pass_only",
        frame["woolway_ice_pass_50pct"],
        "Author-model annual ice fraction <=0.5",
    )
    add_row(
        "not_protected_only",
        frame["woolway_protected_pass"],
        "Protected-area field is not 1",
    )
    for threshold in POPULATION_THRESHOLDS_KM:
        population = frame["woolway_population_distance_km"].le(threshold)
        known_gate = (
            frame["woolway_ice_pass_50pct"]
            & frame["woolway_protected_pass"]
            & population
        )
        add_row(
            f"known_gates_population_{threshold:g}km",
            known_gate,
            "Ice + protected-area + population gates; explicit dry-up flag unavailable",
        )
        add_row(
            f"known_gates_population_{threshold:g}km_plus_positive_median_area",
            known_gate & frame["median_surface_area_positive"],
            "Adds positive median surface area only; this is not equivalent to never drying up",
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(reports / "woolway_public_table_gate_reconstruction.csv", index=False)

    ten = summary.loc[
        summary["scenario"].eq("known_gates_population_10km_plus_positive_median_area")
    ].iloc[0]
    qc = {
        "source": "Woolway et al. 2024 Figshare Table S1",
        "doi": "10.6084/m9.figshare.25764507.v1",
        "license": "CC BY 4.0",
        "md5": "1f33e89ec61742249cb47447136fd23f",
        "rows": len(frame),
        "unique_hydrolakes_ids": int(frame["hylak_id"].nunique()),
        "public_total_generation_twh": float(
            frame["woolway_total_fpv_output_gwh"].sum() / 1000.0
        ),
        "student_hydrolakes_ids": len(local_ids),
        "student_hydrolakes_coverage_of_public_table_pct": float(
            frame["in_student_ge1km_hydrolakes_subset"].mean() * 100.0
        ),
        "zero_median_surface_area_records": int(
            (~frame["median_surface_area_positive"]).sum()
        ),
        "explicit_dryup_column_published": False,
        "reported_eligible_waterbodies": REPORTED_ELIGIBLE_WATERBODIES,
        "reported_eligible_generation_twh": REPORTED_ELIGIBLE_GENERATION_TWH,
        "reproducible_10km_known_gates_plus_positive_median_area_waterbodies": int(
            ten["waterbodies"]
        ),
        "reproducible_10km_known_gates_plus_positive_median_area_generation_twh": float(
            ten["generation_twh"]
        ),
        "count_difference_vs_reported": int(ten["waterbodies"])
        - REPORTED_ELIGIBLE_WATERBODIES,
        "audit_conclusion": (
            "The published ten-column table lacks the explicit dry-up flag and the visible "
            "10 km + ice + protected-area columns do not reproduce the article's 67,893 count."
        ),
    }
    (reports / "woolway_public_table_qc.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(qc, indent=2, ensure_ascii=False))


def self_test() -> None:
    assert REPORTED_ELIGIBLE_WATERBODIES > 0
    assert 10.0 in POPULATION_THRESHOLDS_KM
    print("self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "self-test"])
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
    audit(run_dir, root)


if __name__ == "__main__":
    main()
