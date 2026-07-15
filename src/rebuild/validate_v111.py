#!/usr/bin/env python3
"""Hard completion gate for the HydroLAKES full-source v111 sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_WATERBODIES = 1_427_688
EXPECTED_TYPE_COUNTS = {"lake": 1_420_891, "reservoir": 6_687, "controlled_lake": 110}
EXPECTED_PUBLIC_ROWS = 1_056_924
EXPECTED_PUBLIC_OUTPUT_TWH = 14_905.667643


def validate(run_dir: Path, benchmark_path: Path) -> None:
    errors: list[str] = []
    data_dir = run_dir / "data"
    report_dir = run_dir / "reports"
    inventory_path = data_dir / "hydrolakes_full_inventory.parquet"
    generation_path = data_dir / "hydrolakes_full_generation.parquet"
    summary_path = report_dir / "hydrolakes_full_level_summary.csv"
    qc_path = report_dir / "hydrolakes_full_qc.json"

    if not inventory_path.exists():
        errors.append("missing full inventory")
    else:
        inventory = pd.read_parquet(
            inventory_path,
            columns=["hylak_id", "wb_type", "lake_area_km2", "analysis_lon", "analysis_lat"],
        )
        if len(inventory) != EXPECTED_WATERBODIES:
            errors.append(f"inventory rows={len(inventory):,}")
        if inventory["hylak_id"].nunique() != EXPECTED_WATERBODIES:
            errors.append("inventory key coverage mismatch")
        counts = inventory["wb_type"].value_counts().to_dict()
        if counts != EXPECTED_TYPE_COUNTS:
            errors.append(f"type counts mismatch: {counts}")
        if inventory["lake_area_km2"].min() < 0.1:
            errors.append("inventory contains area below 0.1 km2")
        if inventory[["analysis_lon", "analysis_lat", "lake_area_km2"]].isna().any().any():
            errors.append("inventory contains missing required coordinates/area")

    generation = None
    if not generation_path.exists():
        errors.append("missing full generation table")
    else:
        required = [
            "hylak_id",
            "annual_specific_yield_kwh_kwp",
            "annual_footprint_energy_kwh_m2",
            "ice_pass_method_matched",
            "solar_distance_deg",
            "met_distance_deg",
            "l0_10pct_generation_gwh",
            "l0_type_generation_gwh",
            "l1_10pct_generation_gwh",
            "l1_type_generation_gwh",
            "published_known_gates_10km_generation_gwh",
        ]
        generation = pd.read_parquet(generation_path, columns=required)
        if len(generation) != EXPECTED_WATERBODIES:
            errors.append(f"generation rows={len(generation):,}")
        if generation["hylak_id"].nunique() != EXPECTED_WATERBODIES:
            errors.append("generation key coverage mismatch")
        numeric = [column for column in required if column not in {"hylak_id", "ice_pass_method_matched"}]
        if generation[numeric].isna().any().any():
            errors.append("generation contains missing required values")
        if (generation[numeric] < 0).any().any():
            errors.append("generation contains negative values")
        for upper, lower in [
            ("l0_10pct_generation_gwh", "l1_10pct_generation_gwh"),
            ("l0_type_generation_gwh", "l1_type_generation_gwh"),
            ("l1_10pct_generation_gwh", "published_known_gates_10km_generation_gwh"),
        ]:
            if (generation[lower] > generation[upper] + 1e-5).any():
                errors.append(f"non-monotonic rows: {lower} > {upper}")

    if not summary_path.exists():
        errors.append("missing full level summary")
    else:
        summary = pd.read_csv(summary_path).set_index("tier")
        required_tiers = {
            "L0-10-current-full",
            "L0-type-current-full",
            "L1-10-current-full",
            "L1-type-current-full",
            "L2-published-known-gates-10km-current-energy",
        }
        if not required_tiers.issubset(summary.index):
            errors.append("level summary tiers missing")
        else:
            pairs = [
                ("L0-10-current-full", "L1-10-current-full"),
                ("L0-type-current-full", "L1-type-current-full"),
                ("L1-10-current-full", "L2-published-known-gates-10km-current-energy"),
            ]
            for upper, lower in pairs:
                if summary.at[lower, "generation_twh"] > summary.at[upper, "generation_twh"] + 1e-6:
                    errors.append(f"summary non-monotonic: {lower} > {upper}")
            l2 = summary.loc["L2-published-known-gates-10km-current-energy"]
            if bool(l2["complete"]):
                errors.append("partial public-gate L2 is incorrectly marked complete")

    if not qc_path.exists():
        errors.append("missing full-universe QC")
    else:
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if qc.get("hydrolakes_waterbodies") != EXPECTED_WATERBODIES:
            errors.append("QC universe count mismatch")
        if qc.get("omitted_by_student_ge1km_filter") != 1_249_893:
            errors.append("QC legacy omission count mismatch")
        if not str(qc.get("l2_status", "")).startswith("incomplete"):
            errors.append("QC must explicitly label L2 incomplete")

    if not benchmark_path.exists():
        errors.append("missing public author benchmark")
    else:
        benchmark = pd.read_parquet(
            benchmark_path, columns=["hylak_id", "woolway_total_fpv_output_gwh"]
        )
        if len(benchmark) != EXPECTED_PUBLIC_ROWS or benchmark["hylak_id"].nunique() != EXPECTED_PUBLIC_ROWS:
            errors.append("public benchmark row/key count mismatch")
        benchmark_twh = benchmark["woolway_total_fpv_output_gwh"].sum() / 1000.0
        if abs(benchmark_twh - EXPECTED_PUBLIC_OUTPUT_TWH) > 0.001:
            errors.append(f"public benchmark total mismatch: {benchmark_twh:.6f} TWh")

    result = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "expected_waterbodies": EXPECTED_WATERBODIES,
        "l2_complete": False,
        "l3_complete": False,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "validation_gate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if errors:
        raise SystemExit("validation failed: " + "; ".join(errors))
    print("v111 HydroLAKES full-source sensitivity validation: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("_outputs/v111"))
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("_outputs/v110/data/woolway_public_lake_info.parquet"),
    )
    args = parser.parse_args()
    validate(args.run_dir, args.benchmark_path)


if __name__ == "__main__":
    main()
