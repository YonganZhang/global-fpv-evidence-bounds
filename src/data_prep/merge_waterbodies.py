"""
Merge reservoirs (GeoDAR/GRanD) + HydroLAKES natural lakes/controlled lakes
into a unified water body database for FPV assessment.

Output: all_waterbodies.gpkg (layer='waterbodies')

RETIRED: this legacy script removes every HydroLAKES centroid within 1 km of
an external-reservoir centroid.  The v117 audit showed that proximity alone is
not duplicate evidence and that this rule wrongly removed 1,245 lakes.  The
validated replacement is ``src/rebuild/rebuild_v117_scientific_revision.py``,
which uses authoritative and calibrated evidence tiers and preserves an
explicit identity sensitivity.  The guard below prevents accidental reuse of
the superseded merge rule.
"""

raise RuntimeError(
    "Retired proximity-only merge: run "
    "src/rebuild/rebuild_v117_scientific_revision.py instead."
)

import time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ── Paths ──────────────────────────────────────────────────────────────
RESERVOIRS_PATH = "E:/Phone/自己-水上光伏事件/Dataset/data_processed/merged/merged_reservoirs.gpkg"
HYDROLAKES_PATH = "zip://E:/Phone/自己-水上光伏事件/Dataset/Rawdata/HydroLAKES/HydroLAKES_polys_v10_shp.zip!HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp"
OUTPUT_PATH = "E:/Phone/自己-水上光伏事件/Dataset/data_processed/merged/all_waterbodies.gpkg"

# ── Step 1: Load reservoirs ───────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading existing reservoirs...")
t0 = time.time()
res_gdf = gpd.read_file(RESERVOIRS_PATH, layer="reservoirs")
print(f"  Loaded {len(res_gdf):,} reservoirs in {time.time()-t0:.1f}s")
print(f"  Sources: {res_gdf['source'].value_counts().to_dict()}")

# Add wb_type
res_gdf["wb_type"] = "reservoir"

# ── Step 2: Load HydroLAKES ──────────────────────────────────────────
print("\nStep 2: Loading HydroLAKES (this may take a few minutes)...")
t0 = time.time()
hl_gdf = gpd.read_file(HYDROLAKES_PATH)
print(f"  Loaded {len(hl_gdf):,} water bodies in {time.time()-t0:.1f}s")
print(f"  Lake_type distribution:")
for lt, cnt in hl_gdf["Lake_type"].value_counts().sort_index().items():
    label = {1: "Lake", 2: "Reservoir", 3: "Controlled lake"}.get(lt, "Unknown")
    print(f"    Type {lt} ({label}): {cnt:,}")

# ── Step 3: Filter HydroLAKES ────────────────────────────────────────
print("\nStep 3: Filtering HydroLAKES...")
# Keep type 1 (lakes) and type 3 (controlled lakes) with area >= 1.0 km²
# Exclude type 2 (reservoirs) entirely
mask = (
    (hl_gdf["Lake_type"].isin([1, 3])) &
    (hl_gdf["Lake_area"] >= 1.0)
)
hl_filtered = hl_gdf[mask].copy()
print(f"  After filtering: {len(hl_filtered):,} water bodies")
print(f"    Type 1 (Lake, >=1 km²): {(hl_filtered['Lake_type']==1).sum():,}")
print(f"    Type 3 (Controlled, >=1 km²): {(hl_filtered['Lake_type']==3).sum():,}")

# Free memory
del hl_gdf

# ── Step 4: Deduplication ────────────────────────────────────────────
print("\nStep 4: Spatial deduplication (centroid distance < 1 km)...")
t0 = time.time()

# Build reservoir centroids as a GeoDataFrame in projected CRS for distance calc
res_centroids = gpd.GeoDataFrame(
    {"idx": range(len(res_gdf))},
    geometry=[Point(lon, lat) for lon, lat in zip(res_gdf["centroid_lon"], res_gdf["centroid_lat"])],
    crs="EPSG:4326",
)

# Compute HydroLAKES centroids
hl_filtered["_centroid"] = hl_filtered.geometry.centroid
hl_centroids = gpd.GeoDataFrame(
    {"hl_idx": hl_filtered.index},
    geometry=hl_filtered["_centroid"].values,
    crs="EPSG:4326",
)

# Use sjoin_nearest with max_distance=1km (need to project to metric CRS)
# For efficiency, use a simple buffer approach: convert to EPSG:3857 for approximate distances
res_centroids_m = res_centroids.to_crs("EPSG:3857")
hl_centroids_m = hl_centroids.to_crs("EPSG:3857")

# sjoin_nearest with 1000m threshold
joined = gpd.sjoin_nearest(
    hl_centroids_m, res_centroids_m,
    max_distance=1000,  # meters in EPSG:3857
    how="inner",
)
# Indices of HydroLAKES entries that are too close to existing reservoirs
dup_hl_indices = set(joined["hl_idx"].values)
print(f"  Found {len(dup_hl_indices)} HydroLAKES entries within 1km of existing reservoirs")

# Remove duplicates
hl_filtered = hl_filtered[~hl_filtered.index.isin(dup_hl_indices)].copy()
hl_filtered.drop(columns=["_centroid"], inplace=True)
print(f"  After dedup: {len(hl_filtered):,} HydroLAKES entries remain")
print(f"  Elapsed: {time.time()-t0:.1f}s")

# ── Step 5: Build unified schema ─────────────────────────────────────
print("\nStep 5: Building unified schema...")

# -- Reservoirs --
res_unified = pd.DataFrame({
    "wb_type": "reservoir",
    "source": res_gdf["source"],
    "hylak_id": np.nan,
    "grand_id": res_gdf["grand_id_link"],
    "geodar_id": res_gdf["geodar_id"],
    "name": np.nan,
    "country": np.nan,
    "continent": np.nan,
    "area_km2": res_gdf["area_km2"],
    "depth_avg_m": np.nan,
    "elevation_m": np.nan,
    "shore_dev": np.nan,
    "vol_total_mcm": np.nan,
    "dis_avg_m3s": np.nan,
    "res_time_days": np.nan,
    "centroid_lon": res_gdf["centroid_lon"],
    "centroid_lat": res_gdf["centroid_lat"],
})
res_unified = gpd.GeoDataFrame(res_unified, geometry=res_gdf.geometry.values, crs="EPSG:4326")

# -- HydroLAKES --
wb_type_map = {1: "lake", 3: "controlled_lake"}
hl_centroids_wgs = hl_filtered.geometry.centroid

hl_unified = pd.DataFrame({
    "wb_type": hl_filtered["Lake_type"].map(wb_type_map),
    "source": "HydroLAKES",
    "hylak_id": hl_filtered["Hylak_id"].astype(float),
    "grand_id": hl_filtered["Grand_id"].replace(0, np.nan).astype(float),
    "geodar_id": np.nan,
    "name": hl_filtered["Lake_name"].replace("", np.nan),
    "country": hl_filtered["Country"],
    "continent": hl_filtered["Continent"],
    "area_km2": hl_filtered["Lake_area"],
    "depth_avg_m": hl_filtered["Depth_avg"],
    "elevation_m": hl_filtered["Elevation"].astype(float),
    "shore_dev": hl_filtered["Shore_dev"],
    "vol_total_mcm": hl_filtered["Vol_total"] / 1e6,  # m³ -> million m³
    "dis_avg_m3s": hl_filtered["Dis_avg"],
    "res_time_days": hl_filtered["Res_time"],
    "centroid_lon": hl_centroids_wgs.x.values,
    "centroid_lat": hl_centroids_wgs.y.values,
})
hl_unified = gpd.GeoDataFrame(hl_unified, geometry=hl_filtered.geometry.values, crs="EPSG:4326")

# ── Step 6: Concatenate ─────────────────────────────────────────────
print("\nStep 6: Concatenating...")
all_wb = pd.concat([res_unified, hl_unified], ignore_index=True)
all_wb = gpd.GeoDataFrame(all_wb, crs="EPSG:4326")

# Assign sequential wb_id
all_wb.insert(0, "wb_id", range(1, len(all_wb) + 1))

print(f"  Total water bodies: {len(all_wb):,}")

# ── Step 7: Export ───────────────────────────────────────────────────
print(f"\nStep 7: Exporting to {OUTPUT_PATH}...")
t0 = time.time()
all_wb.to_file(OUTPUT_PATH, layer="waterbodies", driver="GPKG")
print(f"  Done in {time.time()-t0:.1f}s")

# ── Step 8: Summary statistics ───────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print(f"\nTotal water bodies: {len(all_wb):,}")
print(f"\nBy type:")
for wt, cnt in all_wb["wb_type"].value_counts().items():
    area = all_wb.loc[all_wb["wb_type"] == wt, "area_km2"].sum()
    print(f"  {wt:20s}: {cnt:>7,} entries, total area = {area:>12,.1f} km²")

print(f"\nBy source:")
for src, cnt in all_wb["source"].value_counts().items():
    print(f"  {src:20s}: {cnt:>7,}")

print(f"\nBy continent (HydroLAKES only):")
hl_only = all_wb[all_wb["source"] == "HydroLAKES"]
for cont, cnt in hl_only["continent"].value_counts().sort_values(ascending=False).items():
    print(f"  {cont:20s}: {cnt:>7,}")

print(f"\nArea statistics (km²):")
print(f"  Min:    {all_wb['area_km2'].min():.4f}")
print(f"  Median: {all_wb['area_km2'].median():.2f}")
print(f"  Mean:   {all_wb['area_km2'].mean():.2f}")
print(f"  Max:    {all_wb['area_km2'].max():.1f}")
print(f"  Total:  {all_wb['area_km2'].sum():,.1f}")

print(f"\nOutput file: {OUTPUT_PATH}")
print("=" * 60)
