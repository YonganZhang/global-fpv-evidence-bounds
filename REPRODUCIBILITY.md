# Reproducibility routes

## Route A: validate the released evidence

Run `uv run python scripts/validate_release.py`. This checks row counts, unique keys, licence separation, exact manuscript endpoints, NA complete tiers, author order, source manifests and file checksums.

## Route B: rebuild all figures from a licensed local inventory

Acquire the upstream inputs in `data/SOURCES.json` under each provider's terms. Run the versioned scripts in `src/rebuild/` in the order documented by `_pipelines/fpv-rebuild-v117.yml`. Place the resulting full table at `_outputs/v117/data/fpv_reference_inventory_v117.parquet`, then run `make figures`.

## Route C: rebuild raw-to-results

The repository includes the v109–v117 pipeline scripts because the final inventory depends on weather/power, protected-area, population-centre, ice, GLEV, identity, accessibility and engineering-economics stages. The pipeline intentionally does not auto-download WDPA or LandScan, whose provider access terms must be accepted by the researcher.
