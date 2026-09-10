# Publication scope and upstream rights

The existing repository MIT licence covers author code. Author-produced documentation,
LLM experiment exports, score mappings and aggregate research outputs in this release
are offered under CC BY 4.0, to the extent the authors hold rights. This does not
override upstream terms or assert that generated judgments are reliable facts.

Only the following data categories are bundled:

- Researcher-authored prompts, recorded final model answers and necessary run metadata.
- Internal waterbody identifiers and researcher-created entity assignments. These
  contain no coordinates, protected-area flags, population distances, source polygons
  or proprietary dam attributes. Grouping does not select on current potential gates.
- Coarse textual context actually used in the requests and derived LLM scores/weights.
  Names and categories are not redistributed polygon/raster products and must not be
  treated as validated ecological surveys or administrative boundary data.
- Author-created global aggregate results and selected uncertainty configurations.
  Underlying IEA country observations and protected/population site records are excluded.

## Upstream attribution and excluded material

HydroLAKES: Messager et al. (2016), https://doi.org/10.1038/ncomms13603.
The provider identifies CC BY 4.0 at https://www.hydrosheds.org/products/hydrolakes.
GeoDAR: Wang et al. (2022), https://doi.org/10.5194/essd-14-1869-2022,
versioned data https://zenodo.org/records/6163413, whose data/code availability section
identifies CC BY 4.0. Their raw datasets are not included in this LLM-focused archive.
The paper's internal IDs are not advertised as provider IDs; use the original inventory
construction and source crosswalk when reconnecting to provider geography.

Protected Planet WDPA/WDPCA: UNEP-WCMC and IUCN, provider terms at
https://www.protectedplanet.net/en/legal restrict redistribution and sublicensing
without permission. No source files, site membership flags, combined eligibility
flags or row-level retained-potential outputs are included. Obtain permission and
download the historically appropriate version directly for licensed reconstruction.

LandScan, GRanD proprietary attributes, Woolway source-table fields and IEA raw
electricity observations are not included because this release does not establish
the applicable redistribution permission for those particular items. This is a
conservative exclusion, not a claim that every product is legally unavailable.
OSM geometries/distances are also excluded; any later OSM derivative release must
retain the applicable ODbL attribution and share-alike conditions.

Other physical inputs (POWER, ERA5, CMIP6, TerraClimate, ice, GLEV and depth) are
referenced in the source contracts and manuscript Methods, not bundled here.
The v125 implementation is a documented adaptation, not an exact replication of
Jin et al. Source reference: https://github.com/YubinJin98/Floating-solar-power,
commit df38ec94e87ca896a4c86c49fe8e659c8ceabb37. No third-party MATLAB source is bundled.

Provider-page checks were performed on 11 September 2026. Unresolved permissions
remain unresolved; hashes demonstrate provenance/integrity, not redistribution rights.
No credentials, account stores, request authentication headers or internal reasoning
events are intentionally exported. The exporter never reads a credential store.
