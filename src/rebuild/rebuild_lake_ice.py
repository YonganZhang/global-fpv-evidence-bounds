#!/usr/bin/env python3
"""Build lake-ice duration flags from LI-CCR and an explicit gap model.

LI-CCR contains gap-filled MODIS lake-ice phenology for 32,800 cold-region
HydroLAKES.  It is joined by the native HydroLAKES identifier.  Waterbodies
without a qualifying LI-CCR record retain the existing monthly-air-temperature
model as a clearly labelled fallback; they are never described as directly
observed.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


ICE_DURATION_THRESHOLD_DAYS = 365.0 * 0.5
DIRECT_BASELINE_START = 2002
DIRECT_BASELINE_END = 2020
DIRECT_MIN_VALID_YEARS = 5
MODEL_MAX_SUBZERO_MONTHS = 6


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_waterbody_keys(root: Path) -> pd.DataFrame:
    gpkg = root / "Dataset/data_processed/merged/all_waterbodies.gpkg"
    with sqlite3.connect(gpkg) as con:
        frame = pd.read_sql_query(
            "SELECT wb_id, wb_type, hylak_id FROM waterbodies ORDER BY wb_id", con
        )
    assert len(frame) == 198_737 and frame["wb_id"].is_unique
    frame["hylak_id"] = frame["hylak_id"].astype("Int64")
    return frame


def numeric(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def parse_li_ccr_archive(archive: Path) -> pd.DataFrame:
    import xlrd

    rows: list[dict[str, object]] = []
    with ZipFile(archive) as bundle:
        members = [
            item
            for item in bundle.infolist()
            if not item.is_dir() and item.filename.lower().endswith(".xls")
        ]
        assert len(members) == 32_800, len(members)
        for index, member in enumerate(members, start=1):
            hylak_id = int(Path(member.filename).stem)
            workbook = xlrd.open_workbook(file_contents=bundle.read(member))
            sheet = workbook.sheet_by_index(0)
            header = {str(value).strip(): pos for pos, value in enumerate(sheet.row_values(0))}
            assert {"Year", "ICD", "CICD"}.issubset(header), (member.filename, header)
            baseline: list[float] = []
            overlap: list[float] = []
            for row_number in range(1, sheet.nrows):
                year = numeric(sheet.cell_value(row_number, header["Year"]))
                icd = numeric(sheet.cell_value(row_number, header["ICD"]))
                if not np.isfinite(year) or not np.isfinite(icd):
                    continue
                year_int = int(year)
                if DIRECT_BASELINE_START <= year_int <= DIRECT_BASELINE_END:
                    baseline.append(icd)
                if 2019 <= year_int <= 2021:
                    overlap.append(icd)
            mean_baseline = float(np.mean(baseline)) if baseline else np.nan
            rows.append(
                {
                    "hylak_id": hylak_id,
                    "li_ccr_valid_years_2002_2020": len(baseline),
                    "li_ccr_mean_icd_days_2002_2020": mean_baseline,
                    "li_ccr_median_icd_days_2002_2020": (
                        float(np.median(baseline)) if baseline else np.nan
                    ),
                    "li_ccr_max_icd_days_2002_2020": (
                        float(np.max(baseline)) if baseline else np.nan
                    ),
                    "li_ccr_valid_years_2019_2021": len(overlap),
                    "li_ccr_mean_icd_days_2019_2021": (
                        float(np.mean(overlap)) if overlap else np.nan
                    ),
                }
            )
            if index % 5_000 == 0:
                print(f"  parsed {index:,}/{len(members):,} LI-CCR workbooks", flush=True)
    result = pd.DataFrame(rows).sort_values("hylak_id")
    assert len(result) == 32_800 and result["hylak_id"].is_unique
    return result


def build_flags(run_dir: Path, root: Path) -> Path:
    archive = run_dir / "raw/li_ccr/Lake ice phenology.zip"
    probability_path = run_dir / "raw/li_ccr/Probability of complete ice-cover occurrence.xls"
    assert archive.exists() and probability_path.exists()

    direct = parse_li_ccr_archive(archive)
    probability = pd.read_excel(probability_path).rename(
        columns={"Hydro_id": "hylak_id", "PCIO": "li_ccr_pcio"}
    )
    probability["hylak_id"] = probability["hylak_id"].astype(int)
    assert len(probability) == 32_800 and probability["hylak_id"].is_unique
    direct = direct.merge(probability, on="hylak_id", how="left", validate="one_to_one")

    keys = load_waterbody_keys(root)
    flags = keys.merge(direct, on="hylak_id", how="left", validate="many_to_one")
    weather = pd.read_parquet(
        run_dir / "data/weather_power_monthly_2019_2023.parquet",
        columns=["wb_id", "month", "temp_c"],
    )
    assert len(weather) == len(keys) * 12
    model = (
        weather.assign(subzero=weather["temp_c"].lt(0.0))
        .groupby("wb_id", as_index=False)["subzero"]
        .sum()
        .rename(columns={"subzero": "ice_months_temp_proxy"})
    )
    flags = flags.merge(model, on="wb_id", how="left", validate="one_to_one")
    flags["ice_pass_temp_proxy"] = (
        flags["ice_months_temp_proxy"] <= MODEL_MAX_SUBZERO_MONTHS
    )
    flags["ice_direct_usable"] = (
        flags["li_ccr_valid_years_2002_2020"].fillna(0) >= DIRECT_MIN_VALID_YEARS
    )
    flags["ice_pass_li_ccr"] = pd.Series(pd.NA, index=flags.index, dtype="boolean")
    usable = flags["ice_direct_usable"]
    flags.loc[usable, "ice_pass_li_ccr"] = (
        flags.loc[usable, "li_ccr_mean_icd_days_2002_2020"]
        <= ICE_DURATION_THRESHOLD_DAYS
    )
    flags["ice_pass_best_available"] = flags["ice_pass_temp_proxy"].astype(bool)
    flags.loc[usable, "ice_pass_best_available"] = flags.loc[
        usable, "ice_pass_li_ccr"
    ].astype(bool)
    flags["ice_source"] = "nasa_power_monthly_temp_proxy_2019_2023"
    flags.loc[usable, "ice_source"] = "li_ccr_gap_filled_modis_2002_2020"
    flags["ice_threshold_borderline_15d"] = usable & (
        (flags["li_ccr_mean_icd_days_2002_2020"] - ICE_DURATION_THRESHOLD_DAYS).abs()
        <= 15.0
    )

    comparable = flags[usable].copy()
    mismatch = (
        comparable["ice_pass_li_ccr"].astype(bool)
        != comparable["ice_pass_temp_proxy"].astype(bool)
    )
    comparison = pd.crosstab(
        comparable["ice_pass_li_ccr"].astype(bool),
        comparable["ice_pass_temp_proxy"].astype(bool),
        rownames=["li_ccr_pass"],
        colnames=["temp_proxy_pass"],
    )

    output_columns = [
        "wb_id",
        "hylak_id",
        "li_ccr_valid_years_2002_2020",
        "li_ccr_mean_icd_days_2002_2020",
        "li_ccr_median_icd_days_2002_2020",
        "li_ccr_max_icd_days_2002_2020",
        "li_ccr_valid_years_2019_2021",
        "li_ccr_mean_icd_days_2019_2021",
        "li_ccr_pcio",
        "ice_months_temp_proxy",
        "ice_pass_temp_proxy",
        "ice_direct_usable",
        "ice_pass_li_ccr",
        "ice_pass_best_available",
        "ice_source",
        "ice_threshold_borderline_15d",
    ]
    data_dir = run_dir / "data"
    reports = run_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    output = data_dir / "waterbody_ice_flags.parquet"
    flags[output_columns].to_parquet(output, index=False)
    comparison.to_csv(reports / "ice_direct_vs_temp_proxy_confusion.csv")
    qc = {
        "source": "LI-CCR DOI 10.5281/zenodo.17687699",
        "waterbodies": len(flags),
        "li_ccr_workbooks": len(direct),
        "direct_joined_waterbodies": int(flags["li_ccr_mean_icd_days_2002_2020"].notna().sum()),
        "direct_usable_waterbodies": int(usable.sum()),
        "direct_usable_pass": int(flags.loc[usable, "ice_pass_li_ccr"].astype(bool).sum()),
        "direct_usable_fail": int((~flags.loc[usable, "ice_pass_li_ccr"].astype(bool)).sum()),
        "temp_proxy_pass_all": int(flags["ice_pass_temp_proxy"].sum()),
        "best_available_pass_all": int(flags["ice_pass_best_available"].sum()),
        "direct_vs_proxy_mismatches": int(mismatch.sum()),
        "direct_vs_proxy_mismatch_pct": float(mismatch.mean() * 100.0),
        "borderline_within_15_days": int(flags["ice_threshold_borderline_15d"].sum()),
        "direct_threshold_days": ICE_DURATION_THRESHOLD_DAYS,
        "direct_baseline": f"{DIRECT_BASELINE_START}-{DIRECT_BASELINE_END}",
        "minimum_direct_valid_years": DIRECT_MIN_VALID_YEARS,
        "gap_policy": "LI-CCR where >=5 valid baseline years; otherwise labelled monthly-air-temperature proxy",
        "known_limitations": [
            "LI-CCR covers cold-region HydroLAKES and not the project's reservoir-only records.",
            "The direct and fallback periods differ; the source field preserves that distinction.",
            "LI-CCR is gap-filled satellite-derived data, not an in-situ observation for every lake.",
        ],
    }
    (reports / "ice_qc.json").write_text(
        json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(qc, indent=2, ensure_ascii=False))
    return output


def self_test() -> None:
    assert ICE_DURATION_THRESHOLD_DAYS == 182.5
    assert DIRECT_BASELINE_START <= DIRECT_BASELINE_END
    assert numeric("") != numeric("")
    assert numeric(3) == 3.0
    print("self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["build", "self-test"])
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
    build_flags(run_dir, root)


if __name__ == "__main__":
    main()
