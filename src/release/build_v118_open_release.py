#!/usr/bin/env python3
"""Build the licence-aware v118 public repository export.

The canonical v117 inventory combines open, share-alike and restricted
third-party evidence.  This builder deliberately avoids applying one blanket
licence to that mixed table.  It publishes all code, all aggregate manuscript
results, a CC-BY-compatible row-level table, and a separate ODbL derivative;
WDPA/LandScan-derived row-level fields are listed but not redistributed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "_outputs/v118/open_repository"
SOURCE_INVENTORY = ROOT / "_outputs/v117/data/fpv_reference_inventory_v117.parquet"

RESTRICTED_COLUMNS = {
    "in_wdpa_polygon_any": ("Protected Planet WDPA", "redistribution and sublicensing are prohibited without permission"),
    "in_wdpa_polygon_strict_i_iv": ("Protected Planet WDPA", "redistribution and sublicensing are prohibited without permission"),
    "population_center_distance_km": ("LandScan Global 2024", "row-level redistribution permission was not established"),
    "within_10km_population_center": ("LandScan Global 2024", "row-level redistribution permission was not established"),
    "published_known_gates_10km_pass": ("Woolway/WDPA/LandScan compound derivative", "contains restricted gate results"),
    "published_known_gates_10km_generation_gwh": ("Woolway/WDPA/LandScan compound derivative", "contains restricted gate results"),
    "l2_core_no_dry_pass_v112": ("WDPA/LandScan compound derivative", "downloadable row-level derivative is not redistributed"),
    "l2_conservative_no_dry_pass_v112": ("WDPA/LandScan compound derivative", "downloadable row-level derivative is not redistributed"),
    "l2_core_type_generation_gwh_v112": ("WDPA/LandScan compound derivative", "downloadable row-level derivative is not redistributed"),
    "l2_conservative_type_generation_gwh_v112": ("WDPA/LandScan compound derivative", "downloadable row-level derivative is not redistributed"),
    "l2_author_visible_type_generation_gwh_v112": ("WDPA/LandScan compound derivative", "downloadable row-level derivative is not redistributed"),
    "l2_author_visible_pass_v112": ("WDPA/LandScan compound derivative", "downloadable row-level derivative is not redistributed"),
}

WOOLWAY_COLUMNS = {
    "in_woolway_public_table",
    "woolway_ice_cover_fraction",
    "woolway_population_distance_km",
    "woolway_protected_area",
    "woolway_total_fpv_output_gwh",
    "woolway_table_match_v112",
}

ODBL_COLUMNS = [
    "wb_id",
    "osm_powerline_sample_distance_km_v115",
    "osm_powerline_distance_lower_bound_km_v115",
]

GRAND_COLUMNS = {
    "grand_id",
    "grand_hydropower_evidence",
    "grid_within_25km_possible_or_hydropower_v115",
    "grid_within_25km_observed_or_hydropower_v115",
}


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path, pattern: str = "*") -> None:
    for path in sorted(source.rglob(pattern)):
        if path.is_file() and "__pycache__" not in path.parts:
            copy_file(path, destination / path.relative_to(source))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authors() -> list[dict[str, object]]:
    return [
        {"given": "Yongan", "family": "Zhang", "affiliations": ["PolyU", "International Centre of Urban Energy Nexus", "Eastern Institute of Technology"]},
        {"given": "Yujie", "family": "Jiang", "affiliations": ["PolyU", "International Centre of Urban Energy Nexus"]},
        {"given": "Xiaoyuan", "family": "Li", "affiliations": ["PolyU", "International Centre of Urban Energy Nexus"]},
        {"given": "Zhiling", "family": "Guo", "affiliations": ["PolyU", "International Centre of Urban Energy Nexus"]},
        {"given": "Yuntian", "family": "Chen", "affiliations": ["Eastern Institute of Technology", "Ningbo Institute of Digital Twin"]},
        {"given": "Jinyue", "family": "Yan", "affiliations": ["PolyU", "International Centre of Urban Energy Nexus"]},
    ]


def build_data(out: Path) -> None:
    frame = pd.read_parquet(SOURCE_INVENTORY)
    if len(frame) != 199_976 or not frame["wb_id"].is_unique:
        raise ValueError("canonical v117 inventory failed its row/key invariant")

    excluded = set(RESTRICTED_COLUMNS) | WOOLWAY_COLUMNS | GRAND_COLUMNS | set(ODBL_COLUMNS[1:])
    missing = sorted(excluded - set(frame.columns))
    if missing:
        raise KeyError(f"expected protected columns are missing: {missing}")
    open_columns = [column for column in frame.columns if column not in excluded]
    open_frame = frame.loc[:, open_columns]
    open_path = out / "data/open/fpv_reference_inventory_open_v117.parquet"
    open_path.parent.mkdir(parents=True, exist_ok=True)
    open_frame.to_parquet(open_path, index=False, compression="zstd")

    odbl = frame.loc[:, ODBL_COLUMNS]
    odbl_path = out / "data/odbl/fpv_osm_powerline_derivatives_v117.parquet"
    odbl_path.parent.mkdir(parents=True, exist_ok=True)
    odbl.to_parquet(odbl_path, index=False, compression="zstd")

    exclusions: list[dict[str, str]] = []
    for column, (source, reason) in RESTRICTED_COLUMNS.items():
        exclusions.append({"column": column, "source": source, "reason": reason, "public_route": "recompute locally from provider-authorized input"})
    for column in sorted(WOOLWAY_COLUMNS):
        exclusions.append({"column": column, "source": "Woolway et al. public table", "reason": "kept out of the blanket data licence pending item-level licence confirmation", "public_route": "download from the article data source and rebuild"})
    for column in sorted(GRAND_COLUMNS):
        exclusions.append({"column": column, "source": "GRanD v1.1", "reason": "raw identifier/attribute redistribution licence was not verified for the local version", "public_route": "download GRanD from Global Dam Watch/USGS and rebuild"})
    with (out / "data/excluded_row_level_fields.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["column", "source", "reason", "public_route"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(exclusions)

    dictionary_rows = []
    for column in open_frame.columns:
        dictionary_rows.append({"column": column, "dtype": str(open_frame[column].dtype), "licence_group": "CC-BY-4.0-compatible research output", "notes": "See data/SOURCES.json and manuscript Methods for provenance."})
    for column in ODBL_COLUMNS:
        dictionary_rows.append({"column": column, "dtype": str(odbl[column].dtype), "licence_group": "ODbL-1.0 derivative", "notes": "Derived from the World Bank 2016 OpenStreetMap transmission-line snapshot; © OpenStreetMap contributors."})
    with (out / "data/data_dictionary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["column", "dtype", "licence_group", "notes"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(dictionary_rows)


def build_metadata(out: Path) -> None:
    repository = "https://github.com/YonganZhang/global-fpv-evidence-bounds"
    write(out / "LICENSE", """MIT License

Copyright (c) 2026 Yongan Zhang and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \"Software\"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.""")
    write(out / "LICENSE_DATA.md", """# Data licensing

The root MIT licence applies to code only.

- `data/open/` contains an author-assembled research output released under CC BY 4.0, subject to attribution of the upstream open sources listed in `data/SOURCES.json`.
- `data/odbl/` is a separate OpenStreetMap-derived database released under ODbL 1.0. Attribute © OpenStreetMap contributors and retain share-alike terms.
- `results/` and `figures/` are author-created aggregate results and publication graphics released under CC BY 4.0.
- `_outputs/v117/raw/natural_earth_admin0/` contains the public-domain Natural Earth admin-0 basemap required to reproduce the map figure.
- WDPA- and LandScan-derived row-level fields are deliberately absent. Protected Planet prohibits redistribution/sublicensing without permission; the LandScan permission applicable to row-level derivatives was not established. See `data/excluded_row_level_fields.csv`.

No file in this repository changes or overrides an upstream provider's terms.""")
    write(out / ".gitignore", """__pycache__/
*.py[cod]
.venv/
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
_outputs/v117/data/fpv_reference_inventory_v117.parquet
/raw/
""")
    write(out / "CITATION.cff", f"""cff-version: 1.2.0
message: \"If you use this repository, cite the software/data release and the associated paper.\"
title: \"Global FPV evidence-bounded resource assessment\"
type: software
version: 1.0.2-v118
date-released: 2026-07-15
repository-code: \"{repository}\"
license: MIT
authors:
  - family-names: Zhang
    given-names: Yongan
  - family-names: Jiang
    given-names: Yujie
  - family-names: Li
    given-names: Xiaoyuan
  - family-names: Guo
    given-names: Zhiling
  - family-names: Chen
    given-names: Yuntian
  - family-names: Yan
    given-names: Jinyue
""")
    zenodo = {
        "title": "Global FPV evidence-bounded resource assessment: code and licence-aware processed data",
        "upload_type": "software",
        "description": "Reproducible code, aggregate results and licence-aware processed data for the global floating-photovoltaic evidence-bounded assessment.",
        "creators": [{"name": f"{item['family']}, {item['given']}"} for item in authors()],
        "access_right": "open",
        "license": "other-open",
        "keywords": ["floating photovoltaics", "geospatial screening", "renewable energy", "open science"],
    }
    write(out / ".zenodo.json", json.dumps(zenodo, indent=2))
    codemeta = {
        "@context": "https://doi.org/10.5063/schema/codemeta-2.0",
        "@type": "SoftwareSourceCode",
        "name": "Global FPV evidence-bounded resource assessment",
        "version": "1.0.2-v118",
        "codeRepository": repository,
        "license": "https://spdx.org/licenses/MIT",
        "datePublished": "2026-07-15",
        "programmingLanguage": "Python",
        "author": [{"@type": "Person", "givenName": item["given"], "familyName": item["family"]} for item in authors()],
    }
    write(out / "codemeta.json", json.dumps(codemeta, indent=2))

    sources = [
        {"dataset": "HydroLAKES", "url": "https://www.hydrosheds.org/products/hydrolakes", "licence": "CC-BY-4.0", "redistribution": "allowed with attribution"},
        {"dataset": "GeoDAR v1.1", "url": "https://zenodo.org/records/6163413", "licence": "CC-BY-4.0", "redistribution": "allowed with attribution"},
        {"dataset": "GRanD", "url": "https://www.globaldamwatch.org/grand/", "licence": "verify the downloaded version", "redistribution": "not bundled"},
        {"dataset": "NASA POWER Release 10", "url": "https://power.larc.nasa.gov/", "licence": "NASA open-data terms", "redistribution": "derived values bundled"},
        {"dataset": "LI-CCR", "url": "https://zenodo.org/records/17687699", "licence": "CC-BY-4.0", "redistribution": "derived values bundled"},
        {"dataset": "Woolway et al. public lake table", "url": "https://doi.org/10.1038/s44221-024-00251-4", "licence": "verify item-level source-data terms", "redistribution": "source columns not bundled"},
        {"dataset": "Protected Planet WDPA", "url": "https://www.protectedplanet.net/en/legal", "licence": "Protected Planet terms", "redistribution": "prohibited without permission; not bundled"},
        {"dataset": "Ramsar Sites Information Service", "url": "https://rsis.ramsar.org/", "licence": "CC-BY-4.0", "redistribution": "derived flags bundled"},
        {"dataset": "LandScan Global 2024", "url": "https://landscan.ornl.gov/", "licence": "provider terms", "redistribution": "row-level derivatives not bundled pending permission"},
        {"dataset": "GLEV", "url": "https://zenodo.org/records/4646621", "licence": "CC-BY-4.0", "redistribution": "derived values bundled"},
        {"dataset": "GRIP4", "url": "https://www.globio.info/download-grip-dataset", "licence": "CC0", "redistribution": "derived distances bundled"},
        {"dataset": "OpenStreetMap transmission lines via World Bank", "url": "https://energydata.info/dataset/global-transmission-network", "licence": "ODbL-1.0", "redistribution": "separate ODbL derivative"},
        {"dataset": "GLOBathy basic parameters", "url": "https://doi.org/10.6084/m9.figshare.13402070.v1", "licence": "CC0", "redistribution": "derived values bundled"},
        {"dataset": "Natural Earth admin-0", "url": "https://www.naturalearthdata.com/about/terms-of-use/", "licence": "public domain", "redistribution": "country assignments bundled"},
    ]
    write(out / "data/SOURCES.json", json.dumps(sources, indent=2))

    write(out / "AUTHORS.md", """# Authors and submission metadata

Author order for the v118 Applied Energy manuscript:

1. Yongan Zhang — PolyU BEEE; International Centre of Urban Energy Nexus; Eastern Institute of Technology. Email: yongan.zhang@connect.polyu.hk.
2. Yujie Jiang — PolyU BEEE; International Centre of Urban Energy Nexus. Email and ORCID: pending author confirmation.
3. Xiaoyuan Li — PolyU BEEE; International Centre of Urban Energy Nexus. Email and ORCID: pending author confirmation.
4. Zhiling Guo — PolyU BEEE; International Centre of Urban Energy Nexus. Corresponding email: zhiling.guo@polyu.edu.hk.
5. Yuntian Chen — Eastern Institute of Technology; Ningbo Institute of Digital Twin. Corresponding email: ychen@eitech.edu.cn.
6. Jinyue Yan — PolyU BEEE; International Centre of Urban Energy Nexus. Corresponding email: jinyue.yan@polyu.edu.hk.

The current CRediT entry for Yujie Jiang and Xiaoyuan Li is the conservative project draft “Investigation, Validation” and requires all-author approval before submission. No ORCID has been inferred.""")
    write(out / "FUNDING.md", """# Funding control note

The user-designated Window 17 reference manuscript contains only a generic funding placeholder and no verified funder or grant number. The unrelated default grants in the general collaborator roster were not inserted into this FPV paper. Funding remains an author-controlled submission item and must be supplied or explicitly confirmed as “no specific funding” before submission.""")

    write(out / "README.md", f"""# Global FPV evidence-bounded resource assessment

This repository contains the six-author v118 reproducibility release for **Evidence-bounded global floating photovoltaic resource for large lakes and mapped reservoirs**.

## Validated result

The audited inventory contains 199,976 unique large lakes and mapped reservoirs. The reference physical resource is 16,712.989 TWh yr⁻¹ (L0); climate screening retains 10,994.672 TWh yr⁻¹ (65.79%, L1). Public-evidence L2 is 4,190.140–4,804.616 TWh yr⁻¹ under the core definition and 2,914.320–3,404.440 TWh yr⁻¹ under the conservative definition. Road/mapped-power-line partial L3 is 1,594.594–3,508.282 and 1,100.521–2,493.593 TWh yr⁻¹, respectively. Complete published-criterion L2 and complete deployable L3 remain unavailable.

## What is open

- all Python workflows, validation gates, figure code and active LaTeX sources;
- all aggregate result tables and validation JSON files;
- a 199,976-row licence-compatible processed table in `data/open/`;
- an explicitly separated ODbL power-line-distance derivative in `data/odbl/`;
- all publication figures and figure-ready outputs.
- the public-domain Natural Earth admin-0 basemap required by the map builder.

WDPA/LandScan-derived row-level fields are not downloadable here because upstream terms do not permit a blanket open-data relicence. They are named in `data/excluded_row_level_fields.csv`, and the complete reconstruction code and source-acquisition manifest are included. This is a legal boundary, not a hidden analytical omission.

## Quick validation

```bash
uv sync
uv run python scripts/validate_release.py
```

To compile the manuscript (TeX Live/latexmk required):

```bash
make paper
```

To rerun figures, first reconstruct or provide the full licensed local inventory at `_outputs/v117/data/fpv_reference_inventory_v117.parquet`; then run `make figures`. The command fails loudly if restricted inputs are absent.

## Licences and citation

Code is MIT. Data are split by licence; see `LICENSE_DATA.md` and `data/SOURCES.json`. Repository: {repository}.
""")
    write(out / "data/README.md", """# Data layout

- `open/fpv_reference_inventory_open_v117.parquet`: 199,976-row processed table excluding restricted and incompatible-licence fields.
- `odbl/fpv_osm_powerline_derivatives_v117.parquet`: ODbL derivative, separated to preserve share-alike attribution.
- `data_dictionary.csv`: columns, dtypes and licence groups.
- `excluded_row_level_fields.csv`: fields withheld from download, source terms and reconstruction route.
- `SOURCES.json`: authoritative source URLs and redistribution decisions.

Aggregate L2/L3 tables are in `results/`. Complete L2 and L3 are explicitly NA.""")
    write(out / "REPRODUCIBILITY.md", """# Reproducibility routes

## Route A: validate the released evidence

Run `uv run python scripts/validate_release.py`. This checks row counts, unique keys, licence separation, exact manuscript endpoints, NA complete tiers, author order, source manifests and file checksums.

## Route B: rebuild all figures from a licensed local inventory

Acquire the upstream inputs in `data/SOURCES.json` under each provider's terms. Run the versioned scripts in `src/rebuild/` in the order documented by `_pipelines/fpv-rebuild-v117.yml`. Place the resulting full table at `_outputs/v117/data/fpv_reference_inventory_v117.parquet`, then run `make figures`.

## Route C: rebuild raw-to-results

The repository includes the v109–v117 pipeline scripts because the final inventory depends on weather/power, protected-area, population-centre, ice, GLEV, identity, accessibility and engineering-economics stages. The pipeline intentionally does not auto-download WDPA or LandScan, whose provider access terms must be accepted by the researcher.
""")
    write(out / "pyproject.toml", """[project]
name = "global-fpv-evidence-bounds"
version = "1.0.2"
description = "Evidence-bounded global floating photovoltaic resource assessment"
requires-python = ">=3.10"
dependencies = [
  "geopandas>=1.0,<2",
  "matplotlib>=3.8,<4",
  "numpy>=1.26,<3",
  "pandas>=2.2,<3",
  "pyarrow>=15,<24",
  "pyshp>=2.3,<3",
  "requests>=2.31,<3",
  "scipy>=1.12,<2",
  "shapely>=2,<3",
  "xlrd>=2,<3",
]

[tool.uv]
package = false
""")
    write(out / "Makefile", """.PHONY: validate figures paper

validate:
	uv run python scripts/validate_release.py

figures:
	test -f _outputs/v117/data/fpv_reference_inventory_v117.parquet || (echo "Full licensed inventory missing; see REPRODUCIBILITY.md" && exit 2)
	uv run python src/figures/draw_v117_manuscript_figures.py

paper:
	cd paper/draft && latexmk -g -pdf -interaction=nonstopmode -halt-on-error main_submission.tex
	cd paper/draft && latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
""")
    write(out / ".github/workflows/validate.yml", """name: validate-open-release
on:
  push:
  pull_request:
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --locked
      - run: uv run python scripts/validate_release.py
""")


def copy_assets(out: Path) -> None:
    copy_tree(ROOT / "src/rebuild", out / "src/rebuild", "*.py")
    copy_tree(ROOT / "src/release", out / "src/release", "*.py")
    for name in ["draw_v117_manuscript_figures.py", "figure_standards.py"]:
        copy_file(ROOT / "src/figures" / name, out / "src/figures" / name)
    copy_file(ROOT / "src/data_prep/merge_waterbodies.py", out / "src/data_prep/merge_waterbodies.py")
    copy_file(ROOT / "_pipelines/fpv-rebuild-v117.yml", out / "_pipelines/fpv-rebuild-v117.yml")
    copy_tree(ROOT / "_outputs/v117/reports", out / "results")
    copy_tree(ROOT / "_outputs/v117/reports", out / "_outputs/v117/reports")
    copy_tree(
        ROOT / "_outputs/v117/raw/natural_earth_admin0/unpacked",
        out / "_outputs/v117/raw/natural_earth_admin0/unpacked",
    )
    copy_tree(ROOT / "paper/figures/v2026-07-15_v117_rebuild/main", out / "paper/figures/v2026-07-15_v117_rebuild/main")
    copy_tree(ROOT / "paper/figures/v2026-07-15_v117_rebuild/main", out / "figures")

    paper_files = [
        "abstract.tex", "backmatter.tex", "conclusion.tex",
        "frontmatter.tex", "highlights.tex", "highlights.txt", "introduction.tex",
        "main.tex", "main_figures.tex", "main_submission.tex", "methods.tex",
        "references.bib", "results_discussion.tex", "supplementary.tex",
        "supplementary_prompts.tex", "elsarticle.cls", "elsarticle-num.bst",
        "elsarticle-num-names.bst", "main_中文版_v118_审阅稿.md",
    ]
    for name in paper_files:
        copy_file(ROOT / "paper/draft" / name, out / "paper/draft" / name)
    copy_file(Path(__file__).with_name("validate_v118_open_release.py"), out / "scripts/validate_release.py")


def write_checksums(out: Path) -> None:
    files = [path for path in out.rglob("*") if path.is_file()]
    checksum_path = out / "checksums.sha256"
    with checksum_path.open("w", encoding="utf-8") as stream:
        for path in sorted(files):
            if path == checksum_path or ".git" in path.parts:
                continue
            stream.write(f"{sha256(path)}  {path.relative_to(out).as_posix()}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        if not args.replace:
            raise FileExistsError(f"output exists: {out}; pass --replace for generated output")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    build_metadata(out)
    build_data(out)
    copy_assets(out)
    write_checksums(out)
    print(out)


if __name__ == "__main__":
    main()
