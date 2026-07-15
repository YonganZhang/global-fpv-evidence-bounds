#!/usr/bin/env python3
"""Independent gate for the licence-aware v118 public repository."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    # The builder copies this gate from ``src/release/`` to ``scripts/`` in
    # the public repository.  Resolve the repository root correctly from
    # either location so both the canonical source and exported copy are
    # independently runnable.
    source = Path(__file__).resolve()
    repo = source.parents[1]
    if not (repo / "data").is_dir():
        repo = source.parents[2]
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    required = [
        "README.md", "LICENSE", "LICENSE_DATA.md", "CITATION.cff", "codemeta.json",
        ".zenodo.json", "pyproject.toml", "Makefile", "REPRODUCIBILITY.md",
        "data/SOURCES.json", "data/data_dictionary.csv", "data/excluded_row_level_fields.csv",
        "data/open/fpv_reference_inventory_open_v117.parquet",
        "data/odbl/fpv_osm_powerline_derivatives_v117.parquet",
        "results/level_summary_v117.csv", "results/consolidated_validation_gate_v117.json",
        "paper/draft/frontmatter.tex", "paper/draft/main_中文版_v118_审阅稿.md",
        "src/rebuild/rebuild_v117_scientific_revision.py",
        "src/figures/draw_v117_manuscript_figures.py", "checksums.sha256",
    ]
    missing = [name for name in required if not (repo / name).is_file()]
    details["missing_files"] = missing
    checks["required_release_files"] = not missing

    open_frame = pd.read_parquet(repo / "data/open/fpv_reference_inventory_open_v117.parquet")
    odbl = pd.read_parquet(repo / "data/odbl/fpv_osm_powerline_derivatives_v117.parquet")
    checks["open_table_199976_unique"] = len(open_frame) == 199_976 and open_frame["wb_id"].is_unique
    checks["odbl_table_199976_unique"] = len(odbl) == 199_976 and odbl["wb_id"].is_unique
    forbidden = {
        "in_wdpa_polygon_any", "in_wdpa_polygon_strict_i_iv",
        "population_center_distance_km", "within_10km_population_center",
        "l2_core_no_dry_pass_v112", "l2_conservative_no_dry_pass_v112",
        "osm_powerline_sample_distance_km_v115", "osm_powerline_distance_lower_bound_km_v115",
        "grand_id", "grand_hydropower_evidence",
    }
    leaks = sorted(forbidden & set(open_frame.columns))
    details["forbidden_open_table_columns"] = leaks
    checks["restricted_or_sharealike_columns_separated"] = not leaks
    checks["odbl_table_minimal"] = list(odbl.columns) == [
        "wb_id", "osm_powerline_sample_distance_km_v115", "osm_powerline_distance_lower_bound_km_v115"
    ]

    level = pd.read_csv(repo / "results/level_summary_v117.csv").set_index("tier")
    expected = {
        "L0-reference": 16712.988951423074,
        "L1-reference": 10994.671646587763,
        "L2-core-public-evidence-lower": 4190.139560430422,
        "L2-core-public-evidence-upper": 4804.616068316397,
        "L2-conservative-public-evidence-lower": 2914.319715492447,
        "L2-conservative-public-evidence-upper": 3404.4404405247924,
        "L3-core-partial-access-lower": 1594.5943379021421,
        "L3-core-partial-access-upper": 3508.2823401709124,
        "L3-conservative-partial-access-lower": 1100.5209820384507,
        "L3-conservative-partial-access-upper": 2493.593205004449,
    }
    checks["exact_v117_endpoints"] = all(abs(float(level.loc[key, "generation_twh"]) - value) < 1e-9 for key, value in expected.items())
    checks["complete_l2_l3_are_na"] = pd.isna(level.loc["L2-complete-published-1991-2020-criterion", "generation_twh"]) and pd.isna(level.loc["L3-complete-deployable", "generation_twh"])

    front = (repo / "paper/draft/frontmatter.tex").read_text(encoding="utf-8")
    order = ["Yongan Zhang", "Yujie Jiang", "Xiaoyuan Li", "Zhiling Guo", "Yuntian Chen", "Jinyue Yan"]
    positions = [front.find(name) for name in order]
    details["author_positions"] = dict(zip(order, positions))
    checks["six_author_order"] = all(position >= 0 for position in positions) and positions == sorted(positions)

    runtime_paths = [repo / "src/rebuild", repo / "src/figures", repo / "src/data_prep"]
    source_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for directory in runtime_paths
        for path in directory.rglob("*.py")
    )
    checks["no_machine_specific_source_paths"] = not re.search(r"/mnt/data|/home/|\.codex", source_text)
    checks["no_embedded_credentials"] = not re.search(r"ghp_[A-Za-z0-9]+|api[_-]?key\s*=|password\s*=|secret\s*=", source_text, re.I)

    sources = json.loads((repo / "data/SOURCES.json").read_text(encoding="utf-8"))
    checks["source_licence_manifest"] = len(sources) >= 14 and all({"dataset", "url", "licence", "redistribution"} <= set(item) for item in sources)

    checksum_failures = []
    for line in (repo / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected_hash, relative = line.split("  ", 1)
        path = repo / relative
        if not path.is_file() or sha256(path) != expected_hash:
            checksum_failures.append(relative)
    details["checksum_failures"] = checksum_failures
    checks["release_checksums"] = not checksum_failures

    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "passed": sum(checks.values()),
        "total": len(checks),
        "checks": checks,
        "details": details,
    }
    report = repo / "OPEN_RELEASE_VALIDATION.json"
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
