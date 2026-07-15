# Data layout

- `open/fpv_reference_inventory_open_v117.parquet`: 199,976-row processed table excluding restricted and incompatible-licence fields.
- `odbl/fpv_osm_powerline_derivatives_v117.parquet`: ODbL derivative, separated to preserve share-alike attribution.
- `data_dictionary.csv`: columns, dtypes and licence groups.
- `excluded_row_level_fields.csv`: fields withheld from download, source terms and reconstruction route.
- `SOURCES.json`: authoritative source URLs and redistribution decisions.

Aggregate L2/L3 tables are in `results/`. Complete L2 and L3 are explicitly NA.
