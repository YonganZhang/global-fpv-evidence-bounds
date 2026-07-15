#!/usr/bin/env python3
"""Machine-readable completion gate for the versioned FPV rebuild."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_WATERBODIES = 198_737
EXPECTED_MONTHLY_ROWS = EXPECTED_WATERBODIES * 12


def validate(run_dir: Path, allow_partial: bool) -> None:
    errors = []
    data = run_dir / "data"
    reports = run_dir / "reports"
    weather_path = data / "weather_power_monthly_2019_2023.parquet"
    if not weather_path.exists():
        errors.append("missing rebuilt weather")
    else:
        weather = pd.read_parquet(weather_path)
        if len(weather) != EXPECTED_MONTHLY_ROWS:
            errors.append(f"weather rows={len(weather)}")
        if weather["wb_id"].nunique() != EXPECTED_WATERBODIES:
            errors.append("weather unique wb_id mismatch")
        if weather[["ghi_wm2", "temp_c", "wind_speed_ms", "pressure_pa"]].isna().any().any():
            errors.append("weather contains missing required values")

    for name, columns in {
        "wdpa_all_waterbody_polygon_flags.parquet": ["in_wdpa_polygon_any"],
        "population_center_10km_flags.parquet": ["within_10km_population_center"],
        "waterbody_generation_rebuild.parquet": ["annual_footprint_energy_kwh_m2"],
    }.items():
        path = data / name
        if not path.exists():
            if not allow_partial:
                errors.append(f"missing {name}")
            continue
        frame = pd.read_parquet(path, columns=["wb_id"] + columns)
        if len(frame) != EXPECTED_WATERBODIES or frame["wb_id"].nunique() != EXPECTED_WATERBODIES:
            errors.append(f"{name}: key coverage mismatch")
        if frame[columns].isna().any().any():
            errors.append(f"{name}: missing values")

    summary_path = reports / "level_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if not {"G0", "L0", "L0-10", "L1", "L2-partial", "L2-complete", "L3"}.issubset(
            set(summary["tier"])
        ):
            errors.append("level summary tiers missing")
        indexed = summary.set_index("tier")
        uniform_order = ["L0", "L1", "L2-partial", "L2-screened"]
        type_order = ["L0-type", "L1-type", "L2-partial-type", "L2-screened-type"]
        for label, order in [("uniform", uniform_order), ("type-specific", type_order)]:
            if set(order).issubset(indexed.index):
                values = indexed.loc[order, "generation_twh"]
                if values.isna().any() or not values.is_monotonic_decreasing:
                    errors.append(f"{label} tier energy is missing or non-monotonic")
        for unavailable in ["L2-complete", "L3"]:
            if unavailable in indexed.index:
                row = indexed.loc[unavailable]
                if bool(row["complete"]) or pd.notna(row["generation_twh"]):
                    errors.append(f"{unavailable} must remain incomplete/NA")
    elif not allow_partial:
        errors.append("missing level_summary.csv")

    completeness_path = reports / "rebuild_completeness.json"
    if completeness_path.exists():
        completeness = json.loads(completeness_path.read_text(encoding="utf-8"))
        if completeness.get("dry_up_gate") is not False:
            errors.append("dry-up completeness must be explicitly false")
        if completeness.get("practical_economic_l3") is not False:
            errors.append("L3 completeness must be explicitly false")
    elif not allow_partial:
        errors.append("missing rebuild_completeness.json")

    result = {
        "status": "pass" if not errors else "fail",
        "allow_partial": allow_partial,
        "errors": errors,
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "validation_gate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if errors:
        raise SystemExit("validation failed: " + "; ".join(errors))
    print("v109 validation: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("_outputs/v109"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    validate(args.run_dir, args.allow_partial)


if __name__ == "__main__":
    main()
