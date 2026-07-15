#!/usr/bin/env python3
"""Append v113--v116 discoveries and decisions to the auditable data registry."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NEW_CANDIDATES = [
    {
        "candidate_id": "C049",
        "tier": "Tier0",
        "platform": "Zenodo and public Google Drive",
        "dataset": "GLEV reconstructed monthly surface-water area",
        "constraint": "dry_up",
        "coverage": "1427687 HydroLAKES waterbodies",
        "resolution": "monthly waterbody table",
        "time_span": "1985-2018",
        "license": "CC BY 4.0",
        "access": "anonymous direct download",
        "validation_status": "downloaded_integrated_v113",
        "decision": "adopt",
        "reason": "Same Zhao reconstruction family cited by Woolway; 336 months in 1991-2018 and explicit zero-area months support a public dry-up evidence gate",
        "url": "https://zenodo.org/records/4646621",
    },
    {
        "candidate_id": "C050",
        "tier": "Tier0",
        "platform": "Zenodo",
        "dataset": "Lake drought monthly area dataset",
        "constraint": "dry_up",
        "coverage": "162413 lakes with mean area above 1 km2",
        "resolution": "monthly HydroLAKES-linked table",
        "time_span": "1985-2018",
        "license": "repository metadata",
        "access": "anonymous direct download",
        "validation_status": "metadata_verified",
        "decision": "backup",
        "reason": "Useful drought cross-check but smaller coverage and no extension beyond the adopted GLEV endpoint",
        "url": "https://zenodo.org/records/14997634",
    },
    {
        "candidate_id": "C051",
        "tier": "Tier0",
        "platform": "Zenodo",
        "dataset": "Monthly global lake area 2001-2023",
        "constraint": "dry_up_recent_validation",
        "coverage": "lakes at least 1 km2 excluding near-coastal inventory",
        "resolution": "monthly waterbody table",
        "time_span": "2001-2023",
        "license": "repository metadata",
        "access": "anonymous direct download",
        "validation_status": "metadata_verified",
        "decision": "shortlist",
        "reason": "Could validate the 2019-2020 gap for a restricted inventory, but identifiers and coastal exclusions prevent a direct full-scope replacement",
        "url": "https://zenodo.org/records/18454553",
    },
    {
        "candidate_id": "C052",
        "tier": "Tier4",
        "platform": "Nature Communications",
        "dataset": "Zhao et al. global lake evaporation and area reconstruction",
        "constraint": "dry_up_method",
        "coverage": "1427687 waterbodies",
        "resolution": "monthly reconstruction",
        "time_span": "1985-2018",
        "license": "open-access article",
        "access": "article code and Zenodo data",
        "validation_status": "methods_verified",
        "decision": "adopt",
        "reason": "Primary method provenance reports validation against 6715 reservoirs and documents the public GLEV reconstruction used in v113",
        "url": "https://doi.org/10.1038/s41467-022-31125-6",
    },
]


NEW_SEARCHES = [
    {
        "search_id": "S017",
        "timestamp_local": "2026-07-15T00:05:00+08:00",
        "tier": "Tier0",
        "platform": "Zenodo Figshare and author repositories",
        "query": "HydroLAKES monthly surface area dry-up 1991 2020",
        "result_summary": "GLEV full monthly surface-area archive plus two drought/recent-area alternatives identified",
        "candidates_retained": 4,
        "status": "complete",
        "notes": "Downloaded 1.18 GB GLEV archive; ZIP and 336-month schema validated; 190586 pre-dedup scope rows joined",
    },
    {
        "search_id": "S018",
        "timestamp_local": "2026-07-15T00:55:00+08:00",
        "tier": "Tier1",
        "platform": "World Bank EnergyData and PBL/Zenodo",
        "query": "global transmission lines road network FPV accessibility",
        "result_summary": "OSM mapped power-line shapefile and GRIP global road-density raster downloaded and integrated",
        "candidates_retained": 2,
        "status": "complete",
        "notes": "254625 mapped line features and 949361 positive road cells; distance uncertainty explicitly bracketed",
    },
    {
        "search_id": "S019",
        "timestamp_local": "2026-07-15T01:10:00+08:00",
        "tier": "Tier0",
        "platform": "Springer Nature Figshare",
        "query": "GLOBathy maximum depth HydroLAKES basic parameters",
        "result_summary": "Complete 1427688-row Dmax parameter archive downloaded and integrated",
        "candidates_retained": 1,
        "status": "complete",
        "notes": "Official MD5 verified; 190581 deduplicated scope matches; retained as sensitivity rather than hard site-depth gate",
    },
    {
        "search_id": "S020",
        "timestamp_local": "2026-07-15T01:20:00+08:00",
        "tier": "Tier4",
        "platform": "World Bank NREL and peer-reviewed literature",
        "query": "FPV road grid distance depth CAPEX OPEX WACC LCOE thresholds",
        "result_summary": "Published thresholds vary widely and the systematic review confirms no siting consensus",
        "candidates_retained": 5,
        "status": "complete",
        "notes": "Used 10 km road and 25 km mapped-grid primary sensitivities; depth and LCOE remain parameterized and do not close L3",
    },
]


def run(
    inventory_source: Path,
    search_source: Path,
    inventory_output: Path,
    search_output: Path,
) -> None:
    inventory = pd.read_csv(inventory_source)
    searches = pd.read_csv(search_source)
    if len(inventory) != 48 or len(searches) != 16:
        raise ValueError("expected v110 discovery registry with 48 candidates and 16 searches")
    changes = {
        "C002": ("downloaded_integrated_v116", "adopt", "Official 1427688-row Dmax archive downloaded, checksum-verified and integrated as a non-exclusion engineering sensitivity"),
        "C024": ("downloaded_integrated_v115", "adopt", "Official 104810570-byte line archive integrated with spherical distance sampling; voltage/capacity limitations preserved"),
        "C026": ("downloaded_integrated_v115", "adopt", "Official GRIP total-density raster checksum verified and integrated with 5-arcminute road-distance brackets"),
    }
    for candidate_id, (status, decision, reason) in changes.items():
        mask = inventory["candidate_id"].eq(candidate_id)
        inventory.loc[mask, "validation_status"] = status
        inventory.loc[mask, "decision"] = decision
        inventory.loc[mask, "reason"] = reason
    inventory = pd.concat([inventory, pd.DataFrame(NEW_CANDIDATES)], ignore_index=True)
    searches = pd.concat([searches, pd.DataFrame(NEW_SEARCHES)], ignore_index=True)
    if len(inventory) != 52 or not inventory["candidate_id"].is_unique:
        raise ValueError("updated candidate registry mismatch")
    if len(searches) != 20 or not searches["search_id"].is_unique:
        raise ValueError("updated search registry mismatch")
    inventory_output.parent.mkdir(parents=True, exist_ok=True)
    search_output.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(inventory_output, index=False)
    searches.to_csv(search_output, index=False)
    print(f"candidates={len(inventory)} searches={len(searches)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory-source",
        type=Path,
        default=Path("_outputs/v110/reports/data_candidate_inventory_20260714.csv"),
    )
    parser.add_argument(
        "--search-source",
        type=Path,
        default=Path("_outputs/v110/reports/data_search_log_20260714.csv"),
    )
    parser.add_argument(
        "--inventory-output",
        type=Path,
        default=Path("_outputs/v116/reports/data_candidate_inventory_20260715.csv"),
    )
    parser.add_argument(
        "--search-output",
        type=Path,
        default=Path("_outputs/v116/reports/data_search_log_20260715.csv"),
    )
    args = parser.parse_args()
    run(
        args.inventory_source,
        args.search_source,
        args.inventory_output,
        args.search_output,
    )


if __name__ == "__main__":
    main()
