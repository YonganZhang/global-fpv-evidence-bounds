# Global FPV evidence-bounded resource assessment

This repository contains the six-author v118 reproducibility release for **Evidence-bounded global floating photovoltaic resource for large lakes and mapped reservoirs**.

## Validated result

The audited inventory contains 199,976 unique large lakes and mapped reservoirs. The reference physical resource is 16,712.989 TWh yr⁻¹ (L0); climate screening retains 10,994.672 TWh yr⁻¹ (65.79%, L1). Public-evidence L2 is 4,190.140–4,804.616 TWh yr⁻¹ under the core definition and 2,914.320–3,404.440 TWh yr⁻¹ under the conservative definition. Road/mapped-power-line partial L3 is 1,594.594–3,508.282 and 1,100.521–2,493.593 TWh yr⁻¹, respectively. Complete published-criterion L2 and complete deployable L3 remain unavailable.

## What is open

- all Python workflows, validation gates, figure code and active LaTeX sources;
- all aggregate result tables and validation JSON files;
- a 199,976-row licence-compatible processed table in `data/open/`;
- an explicitly separated ODbL power-line-distance derivative in `data/odbl/`;
- all publication figures and figure-ready outputs.

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

Code is MIT. Data are split by licence; see `LICENSE_DATA.md` and `data/SOURCES.json`. Repository: https://github.com/YonganZhang/global-fpv-evidence-bounds.
