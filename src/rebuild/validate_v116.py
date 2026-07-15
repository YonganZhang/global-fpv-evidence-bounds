#!/usr/bin/env python3
"""Validation gate for v116 engineering coverage and LCOE sensitivities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_ROWS = 198_731


def validate(run_dir: Path) -> None:
    errors: list[str] = []
    data_path = run_dir / "data/paper_scope_l3_engineering_economics.parquet"
    flags_path = run_dir / "data/waterbody_engineering_economic_sensitivity.parquet"
    coverage_path = run_dir / "reports/engineering_economic_coverage.csv"
    sensitivity_path = run_dir / "reports/depth_lcoe_sensitivity.csv"
    qc_path = run_dir / "reports/engineering_economic_qc.json"
    for path in [data_path, flags_path, coverage_path, sensitivity_path, qc_path]:
        if not path.exists():
            errors.append(f"missing artifact: {path}")

    if not errors:
        columns = [
            "wb_id",
            "dryup_hylak_id_v113",
            "globathy_dmax_use_m_v116",
            "annual_specific_yield_kwh_kwp_v116",
            "fpv_capacity_mw_v116",
            "l0_type_specific_generation_gwh",
            "lcoe_optimistic_usd_mwh_v116",
            "lcoe_central_usd_mwh_v116",
            "lcoe_conservative_usd_mwh_v116",
        ]
        frame = pd.read_parquet(data_path, columns=columns)
        if len(frame) != EXPECTED_ROWS or frame["wb_id"].nunique() != EXPECTED_ROWS:
            errors.append("v116 row/key coverage mismatch")
        depth_known = frame["globathy_dmax_use_m_v116"].notna()
        if int(depth_known.sum()) != 190_581 or int((~depth_known).sum()) != 8_150:
            errors.append("GLOBathy scope coverage changed")
        if not frame.loc[depth_known, "globathy_dmax_use_m_v116"].gt(0).all():
            errors.append("known GLOBathy maximum depths must be positive")
        if frame.loc[~depth_known, "dryup_hylak_id_v113"].notna().any():
            errors.append("GLOBathy missing rows unexpectedly have HydroLAKES keys")
        if not frame["annual_specific_yield_kwh_kwp_v116"].gt(0).all():
            errors.append("annual specific yield must be positive")
        expected_capacity = (
            frame["l0_type_specific_generation_gwh"]
            * 1000.0
            / frame["annual_specific_yield_kwh_kwp_v116"]
        )
        if not np.allclose(
            frame["fpv_capacity_mw_v116"], expected_capacity, rtol=1e-10, atol=1e-8
        ):
            errors.append("FPV capacity conversion is inconsistent")
        optimistic = frame["lcoe_optimistic_usd_mwh_v116"]
        central = frame["lcoe_central_usd_mwh_v116"]
        conservative = frame["lcoe_conservative_usd_mwh_v116"]
        if not ((optimistic < central) & (central < conservative)).all():
            errors.append("economic scenario LCOE ordering is inconsistent")

        coverage = pd.read_csv(coverage_path).set_index("population")
        expected_coverage = {
            "all_deduplicated_scope": (198_731, 190_581),
            "core_l2_dryup_known_lower": (21_687, 21_687),
            "core_l2_unknown_pass_upper": (27_752, 21_687),
            "core_l3_access_lower": (10_734, 10_734),
            "core_l3_access_upper": (19_827, 15_677),
        }
        if not set(expected_coverage).issubset(coverage.index):
            errors.append("engineering/economic coverage populations missing")
        else:
            for population, (rows, known) in expected_coverage.items():
                if int(coverage.at[population, "waterbodies"]) != rows:
                    errors.append(f"unexpected population size for {population}")
                if int(coverage.at[population, "globathy_depth_known"]) != known:
                    errors.append(f"unexpected depth coverage for {population}")
            if not np.isclose(
                coverage.at["core_l2_dryup_known_lower", "central_lcoe_usd_mwh_median"],
                91.488678,
                atol=1e-5,
            ):
                errors.append("central LCOE median changed unexpectedly")

        sensitivity = pd.read_csv(sensitivity_path, dtype=str)
        if len(sensitivity) != 40:
            errors.append("depth/LCOE sensitivity grid must contain 40 rows")
        else:
            sensitivity["depth"] = sensitivity[
                "globathy_max_depth_sensitivity_m"
            ].replace("none", np.inf).astype(float)
            sensitivity["lcoe"] = sensitivity[
                "central_lcoe_sensitivity_usd_mwh"
            ].replace("none", np.inf).astype(float)
            sensitivity["generation"] = sensitivity["generation_twh"].astype(float)
            for policy, group in sensitivity.groupby("access_policy"):
                pivot = group.pivot(
                    index="depth", columns="lcoe", values="generation"
                ).sort_index().sort_index(axis=1)
                matrix = pivot.to_numpy()
                if (np.diff(matrix, axis=0) < -1e-9).any() or (
                    np.diff(matrix, axis=1) < -1e-9
                ).any():
                    errors.append(f"non-monotonic depth/LCOE sensitivity for {policy}")

        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if qc.get("scope_rows") != EXPECTED_ROWS:
            errors.append("v116 QC scope mismatch")
        if qc.get("globathy_source_rows") != 1_427_688:
            errors.append("GLOBathy source row count mismatch")
        if qc.get("globathy_dmax_nonpositive") != 0:
            errors.append("v116 QC reports nonpositive maximum depths")
        if qc.get("economic_assumptions", {}).get("lifetime_years") != 25:
            errors.append("LCOE lifetime assumption changed")
        if qc.get("l3_complete") is not False:
            errors.append("v116 incorrectly marks L3 complete")

    result = {"status": "pass" if not errors else "fail", "errors": errors}
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "validation_gate.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if errors:
        raise SystemExit("validation failed: " + "; ".join(errors))
    print("v116 engineering/economic sensitivity validation: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("_outputs/v116"))
    args = parser.parse_args()
    validate(args.run_dir)


if __name__ == "__main__":
    main()
