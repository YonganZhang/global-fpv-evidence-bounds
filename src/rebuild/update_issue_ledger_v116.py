#!/usr/bin/env python3
"""Consolidate the v109--v116 audit into a current issue-status ledger."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


UPDATES: dict[int, tuple[str, str]] = {
    2: (
        "resolved_or_replaced",
        "all 198731 deduplicated paper-scope waterbodies retain coordinate-based POWER assignments; no lake-to-lake climate copying remains",
    ),
    3: (
        "resolved_or_replaced",
        "solar and meteorological source-grid distances remain traceable through the v109 rebuild inherited by v116",
    ),
    4: (
        "resolved_or_replaced",
        "v114 removes six authoritative physical duplicates and v115-v116 require all 198731 unique paper-scope wb_id rows",
    ),
    6: (
        "resolved_or_replaced",
        "v114 L0/L1/L2 summaries and v115 L3-access summaries are generated from unique-key versioned result tables",
    ),
    9: (
        "partial",
        "hybrid ice uses LI-CCR then Woolway then a labelled temperature proxy; direct/model mismatch remains and 55920 rows still use the proxy",
    ),
    10: (
        "resolved_or_replaced",
        "WDPA polygon intersection covers the complete deduplicated 198731-row paper scope; the 1.42M HydroLAKES run remains only a scope sensitivity",
    ),
    12: (
        "partial",
        "Ramsar polygon intersections cover the paper scope and add exclusions beyond WDPA; published centroid-only sites and full-universe sensitivity remain",
    ),
    13: (
        "partial",
        "v113-v116 add historical dry-up evidence, road/grid accessibility, GLOBathy depth coverage and LCOE sensitivities; complete water-use, engineering, permitting and social gates remain unavailable",
    ),
    32: (
        "partial",
        "v114 removes all six GRanD-authoritative duplicate controlled lakes; 33 high-confidence occupied-key candidates, country assignment and legacy version drift still require source-level adjudication",
    ),
    38: (
        "partial",
        "v116 is the latest validated numeric evidence chain, but the manuscript and figures have not been switched to it and complete paper-equivalent L2/L3 remain unavailable",
    ),
    41: (
        "partial",
        "v114 enforces and deduplicates the declared external-reservoir plus >=1 km2 natural/controlled scope; manuscript wording and any >=0.1 km2 main claim still require correction",
    ),
    42: (
        "partial",
        "public GLEV reconstructs 1991-2018 dry-up evidence for 190586 pre-dedup rows and narrows corrected core L2 to 24.90-28.61 percent of L0; author flag/code, 2019-2020 and 8151 source rows remain unavailable",
    ),
}


NEW_ISSUES = [
    {
        "issue_id": 43,
        "status": "resolved_or_replaced",
        "evidence_or_remaining": "six HydroLAKES controlled-lake rows sharing authoritative GRanD-linked identities with external reservoirs were removed; scope is 198731 and no assigned dry-up Hylak_id duplicates remain",
    },
    {
        "issue_id": 44,
        "status": "partial",
        "evidence_or_remaining": "GRIP road accessibility is integrated with 5-arcminute cell-distance brackets, but centroid-to-cell distance can misrepresent shoreline access for large waterbodies",
    },
    {
        "issue_id": 45,
        "status": "partial",
        "evidence_or_remaining": "254625 mapped OSM power-line features plus 2130 GRanD hydropower links provide grid evidence, but voltage, substation, spare capacity, age and mapping completeness are absent",
    },
    {
        "issue_id": 46,
        "status": "partial",
        "evidence_or_remaining": "GLOBathy supplies modelled maximum depth for 190581 deduplicated rows, but installation-footprint depth, bathymetry uncertainty, sediment and anchor design are unresolved",
    },
    {
        "issue_id": 47,
        "status": "partial",
        "evidence_or_remaining": "transparent CAPEX/OPEX/WACC LCOE scenarios are calculated, but connection reinforcement, country finance, tax, insurance, rights, permit and decommissioning costs are excluded",
    },
    {
        "issue_id": 48,
        "status": "open",
        "evidence_or_remaining": "no comparable global production layers close wind gust, wave/fetch, water-level range, geotechnical anchoring, competing use, navigation, permitting or social acceptance, so complete L3 remains NA",
    },
]


def run(source_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(source_path)
    if len(frame) != 42 or set(frame["issue_id"]) != set(range(1, 43)):
        raise ValueError("expected the 42-row v111 issue ledger")
    for issue_id, (status, evidence) in UPDATES.items():
        mask = frame["issue_id"].eq(issue_id)
        frame.loc[mask, "status"] = status
        frame.loc[mask, "evidence_or_remaining"] = evidence
    frame = pd.concat([frame, pd.DataFrame(NEW_ISSUES)], ignore_index=True)
    frame = frame.sort_values("issue_id")
    if len(frame) != 48 or not frame["issue_id"].is_unique:
        raise ValueError("v116 issue ledger row/key mismatch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(frame["status"].value_counts().to_string())
    print(f"incomplete={(frame['status'] != 'resolved_or_replaced').sum()} of {len(frame)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-path",
        type=Path,
        default=Path("_outputs/v111/reports/issue_status_20260714.csv"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("_outputs/v116/reports/issue_status_20260715.csv"),
    )
    args = parser.parse_args()
    run(args.source_path, args.output_path)


if __name__ == "__main__":
    main()
