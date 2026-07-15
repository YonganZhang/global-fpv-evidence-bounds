#!/usr/bin/env python3
"""Validation gate for the v115 partial L3 accessibility screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_ROWS = 198_731


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(run_dir: Path) -> None:
    errors: list[str] = []
    data_path = run_dir / "data/paper_scope_l3_accessibility.parquet"
    flags_path = run_dir / "data/waterbody_l3_accessibility_flags.parquet"
    summary_path = run_dir / "reports/l3_accessibility_summary.csv"
    sensitivity_path = run_dir / "reports/l3_accessibility_threshold_sensitivity.csv"
    qc_path = run_dir / "reports/l3_accessibility_qc.json"
    road_zip = run_dir / "raw/roads/GRIP4_density_total.zip"
    grid_zip = run_dir / "raw/transmission/osm_power_tmm.zip"
    for path in [
        data_path,
        flags_path,
        summary_path,
        sensitivity_path,
        qc_path,
        road_zip,
        grid_zip,
    ]:
        if not path.exists():
            errors.append(f"missing artifact: {path}")

    if not errors:
        if md5(road_zip) != "24d3ed5c079f7a898fbc4a5d68be337a":
            errors.append("GRIP archive checksum differs from the official Zenodo checksum")
        if grid_zip.stat().st_size != 104_810_570:
            errors.append("OSM power-line archive size differs from portal metadata")

        columns = [
            "wb_id",
            "grip_road_cell_center_distance_km_v115",
            "grip_road_distance_lower_bound_km_v115",
            "grip_road_distance_upper_bound_km_v115",
            "road_within_10km_guaranteed_v115",
            "road_within_10km_possible_v115",
            "osm_powerline_sample_distance_km_v115",
            "osm_powerline_distance_lower_bound_km_v115",
            "grand_hydropower_evidence",
            "grid_within_25km_observed_or_hydropower_v115",
            "grid_within_25km_possible_or_hydropower_v115",
        ]
        frame = pd.read_parquet(data_path, columns=columns)
        if len(frame) != EXPECTED_ROWS or frame["wb_id"].nunique() != EXPECTED_ROWS:
            errors.append("L3 accessibility row/key coverage mismatch")
        if frame[columns].isna().any().any():
            errors.append("L3 accessibility required columns contain missing values")
        road_lower = frame["grip_road_distance_lower_bound_km_v115"]
        road_centre = frame["grip_road_cell_center_distance_km_v115"]
        road_upper = frame["grip_road_distance_upper_bound_km_v115"]
        if not ((road_lower <= road_centre) & (road_centre <= road_upper)).all():
            errors.append("GRIP road-distance brackets are inconsistent")
        if not frame["road_within_10km_guaranteed_v115"].eq(road_upper.le(10.0)).all():
            errors.append("guaranteed 10 km road-pass semantics changed")
        if not frame["road_within_10km_possible_v115"].eq(road_lower.le(10.0)).all():
            errors.append("possible 10 km road-pass semantics changed")
        if not (
            frame["road_within_10km_guaranteed_v115"]
            <= frame["road_within_10km_possible_v115"]
        ).all():
            errors.append("guaranteed road pass is not a subset of possible road pass")
        grid_upper = frame["osm_powerline_sample_distance_km_v115"]
        grid_lower = frame["osm_powerline_distance_lower_bound_km_v115"]
        hydro = frame["grand_hydropower_evidence"]
        if not (grid_lower <= grid_upper).all():
            errors.append("mapped-grid distance brackets are inconsistent")
        if not frame["grid_within_25km_observed_or_hydropower_v115"].eq(
            grid_upper.le(25.0) | hydro
        ).all():
            errors.append("observed mapped-grid/hydropower semantics changed")
        if not frame["grid_within_25km_possible_or_hydropower_v115"].eq(
            grid_lower.le(25.0) | hydro
        ).all():
            errors.append("possible mapped-grid/hydropower semantics changed")
        if not (
            frame["grid_within_25km_observed_or_hydropower_v115"]
            <= frame["grid_within_25km_possible_or_hydropower_v115"]
        ).all():
            errors.append("observed grid pass is not a subset of possible grid pass")
        expected_counts = {
            "road_within_10km_guaranteed_v115": 47_858,
            "road_within_10km_possible_v115": 87_278,
            "grid_within_25km_observed_or_hydropower_v115": 54_917,
            "grid_within_25km_possible_or_hydropower_v115": 57_332,
            "grand_hydropower_evidence": 2_130,
        }
        for column, expected in expected_counts.items():
            actual = int(frame[column].sum())
            if actual != expected:
                errors.append(f"unexpected pass count for {column}: {actual}")

        summary = pd.read_csv(summary_path).set_index("tier")
        expected_retention = {
            "L3-access-core-evidence-lower-bound": 9.467412,
            "L3-access-core-mapping-upper-envelope": 20.859155,
            "L3-access-conservative-evidence-lower-bound": 6.532782,
            "L3-access-conservative-mapping-upper-envelope": 14.820952,
        }
        if not set(expected_retention).issubset(summary.index):
            errors.append("L3 accessibility tiers missing")
        else:
            for tier, expected in expected_retention.items():
                actual = float(summary.at[tier, "retention_vs_recommended_l0_pct"])
                if not np.isclose(actual, expected, atol=1e-5):
                    errors.append(f"unexpected L3 retention for {tier}: {actual}")
            if summary.at[
                "L3-access-core-evidence-lower-bound", "generation_twh"
            ] > summary.at[
                "L3-access-core-mapping-upper-envelope", "generation_twh"
            ]:
                errors.append("core L3 accessibility interval inverted")
            if summary.at[
                "L3-access-conservative-mapping-upper-envelope", "generation_twh"
            ] > summary.at[
                "L3-access-core-mapping-upper-envelope", "generation_twh"
            ]:
                errors.append("conservative L3 upper envelope exceeds core")
            complete = "L3-complete-deployable"
            if bool(summary.at[complete, "scientific_tier_complete"]):
                errors.append("complete L3 is incorrectly marked complete")
            if pd.notna(summary.at[complete, "generation_twh"]):
                errors.append("complete L3 must remain NA")

        sensitivity = pd.read_csv(sensitivity_path)
        if len(sensitivity) != 24:
            errors.append("L3 threshold-sensitivity grid must have 24 rows")
        for policy, group in sensitivity.groupby("dryup_evidence_policy"):
            pivot = group.pivot(
                index="road_cell_center_threshold_km",
                columns="mapped_grid_threshold_km",
                values="generation_twh",
            ).sort_index().sort_index(axis=1)
            if (np.diff(pivot.to_numpy(), axis=0) < -1e-9).any() or (
                np.diff(pivot.to_numpy(), axis=1) < -1e-9
            ).any():
                errors.append(f"non-monotonic threshold sensitivity for {policy}")

        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if qc.get("scope_rows") != EXPECTED_ROWS:
            errors.append("L3 QC scope count mismatch")
        if qc.get("mapped_power_lines", {}).get("line_features") != 254_625:
            errors.append("mapped power-line feature count mismatch")
        if qc.get("road", {}).get("positive_road_cells") != 949_361:
            errors.append("positive GRIP road-cell count mismatch")
        if qc.get("l3_complete") is not False:
            errors.append("L3 QC incorrectly marks the tier complete")

    result = {"status": "pass" if not errors else "fail", "errors": errors}
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "validation_gate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if errors:
        raise SystemExit("validation failed: " + "; ".join(errors))
    print("v115 partial L3 accessibility validation: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("_outputs/v115"))
    args = parser.parse_args()
    validate(args.run_dir)


if __name__ == "__main__":
    main()
