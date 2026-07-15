#!/usr/bin/env python3
"""Draw the validated v117 Applied Energy figure set from trace tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "_outputs/v117/reports"
INVENTORY = ROOT / "_outputs/v117/data/fpv_reference_inventory_v117.parquet"
WORLD = ROOT / "_outputs/v117/raw/natural_earth_admin0/unpacked/ne_10m_admin_0_countries.shp"
OUT = ROOT / "paper/figures/v2026-07-15_v117_rebuild/main"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_standards import normalize_fonts  # noqa: E402

BLUE = "#376795"
SKY = "#72BCD5"
YELLOW = "#FFD06F"
RED = "#E76254"
NAVY = "#1E466E"
MID = "#528FAD"
PALE = "#AADCE0"
GRAY = "#777777"
LIGHT = "#E8E8E8"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def read_csv(name: str) -> pd.DataFrame:
    path = REPORTS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load() -> dict[str, object]:
    gate = json.loads((REPORTS / "consolidated_validation_gate_v117.json").read_text())
    if gate.get("status") != "pass" or gate.get("passed") != gate.get("total"):
        raise RuntimeError("v117 consolidated gate is not PASS")
    frame = pd.read_parquet(INVENTORY)
    if len(frame) != 199_976 or not frame["wb_id"].is_unique:
        raise ValueError("v117 inventory scope mismatch")
    data: dict[str, object] = {
        "gate": gate,
        "frame": frame,
        "level": read_csv("level_summary_v117.csv"),
        "types": read_csv("type_level_summary_v117.csv"),
        "continents": read_csv("continent_level_summary_v117.csv"),
        "countries": read_csv("country_level_summary_v117.csv"),
        "coverage": read_csv("coverage_sensitivity_v117.csv"),
        "ice": read_csv("ice_sensitivity_v117.csv"),
        "ice_sources": read_csv("ice_source_summary_v117.csv"),
        "wind": read_csv("wind_sensitivity_v117.csv"),
        "yield_regions": read_csv("yield_benchmark_by_continent_v117.csv"),
        "yield_density": read_csv("yield_benchmark_density_v117.csv"),
        "sequential": read_csv("sequential_constraint_summary_v117.csv"),
        "evidence": read_csv("evidence_state_summary_v117.csv"),
        "glev": read_csv("glev_criterion_sensitivity_v117.csv"),
        "factorial": read_csv("l3_factorial_v117.csv"),
        "hydro": read_csv("l3_hydropower_proxy_sensitivity_v117.csv"),
        "engineering": read_csv("engineering_economic_coverage_v117.csv"),
        "engineering_sensitivity": read_csv("engineering_sensitivity_v117.csv"),
    }
    return data


def panel(ax: mpl.axes.Axes, letter: str) -> None:
    ax.text(-0.10, 1.04, letter, transform=ax.transAxes, va="bottom", ha="left", fontweight="bold")


def clean(ax: mpl.axes.Axes, grid: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#DDDDDD", linewidth=0.5, zorder=0)
    ax.tick_params(direction="out", length=3, width=0.5)


def save(fig: mpl.figure.Figure, name: str, sizes: dict[str, float] | None = None) -> None:
    normalize_fonts(
        fig,
        sizes
        or {"axis_label": 8, "tick": 7, "legend": 7, "annotation": 7, "panel_label": 11},
        scale_by_subplot=True,
    )
    for ax in fig.axes:
        for text in [*ax.texts, *ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.label, ax.yaxis.label]:
            text.set_fontfamily("serif")
    fig.savefig(OUT / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def interval_rows(level: pd.DataFrame) -> dict[str, tuple[float, float]]:
    values = level.set_index("tier")["generation_twh"]
    return {
        "Core L2": (values["L2-core-public-evidence-lower"], values["L2-core-public-evidence-upper"]),
        "Conservative L2": (values["L2-conservative-public-evidence-lower"], values["L2-conservative-public-evidence-upper"]),
        "Core partial L3": (values["L3-core-partial-access-lower"], values["L3-core-partial-access-upper"]),
        "Conservative partial L3": (values["L3-conservative-partial-access-lower"], values["L3-conservative-partial-access-upper"]),
    }


def draw_fig1(data: dict[str, object]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [0.9, 1.3]}, constrained_layout=True)
    outcomes = pd.Series(
        [108, 645, 1_245],
        index=["Authoritative\nduplicates", "Spatial high-confidence\nmatches", "Restored: no duplicate\nevidence"],
    )
    ax1.barh(outcomes.index, outcomes.values, color=[RED, YELLOW, BLUE], zorder=3)
    ax1.set_xlim(0, 1_350)
    ax1.set_xlabel("Adjudicated omitted lakes")
    clean(ax1, "x")
    panel(ax1, "a")

    level = data["level"]
    values = level.set_index("tier")["generation_twh"]
    point_names = ["L0 reference", "L1"]
    point_values = [values["L0-reference"], values["L1-reference"]]
    ax2.scatter(point_values, [5, 4], s=42, color=[NAVY, BLUE], zorder=4)
    intervals = interval_rows(level)
    y_positions = [3, 2, 1, 0]
    for (name, (low, high)), y, color in zip(intervals.items(), y_positions, [MID, SKY, YELLOW, RED]):
        ax2.hlines(y, low, high, color=color, linewidth=7, zorder=3)
        ax2.scatter([low, high], [y, y], s=22, color=color, edgecolor="white", linewidth=0.4, zorder=4)
    ax2.set_yticks([5, 4, 3, 2, 1, 0], point_names + list(intervals))
    ax2.set_xlabel("Annual generation (TWh yr$^{-1}$)")
    ax2.set_xlim(0, 18_000)
    clean(ax2, "x")
    panel(ax2, "b")
    save(fig, "Fig1")


def draw_fig2(data: dict[str, object]) -> None:
    fig = plt.figure(figsize=(7.2, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    world = gpd.read_file(WORLD, engine="pyogrio")
    world.plot(ax=ax1, color="#F3F3F3", edgecolor="#AAAAAA", linewidth=0.25)
    frame = data["frame"]
    energy = frame["l0_type_specific_generation_gwh"].to_numpy(float) / 1000.0
    hb = ax1.hexbin(
        frame["centroid_lon"], frame["centroid_lat"], C=energy, reduce_C_function=np.sum,
        gridsize=(120, 50), mincnt=1, norm=LogNorm(vmin=0.002, vmax=max(10.0, np.nanpercentile(energy, 99.9))),
        cmap=mpl.colors.LinearSegmentedColormap.from_list("ukiyoe", [PALE, SKY, MID, NAVY]), linewidths=0,
    )
    ax1.set_xlim(-180, 180); ax1.set_ylim(-60, 85)
    ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")
    ax1.text(0.015, 0.90, "N", transform=ax1.transAxes, ha="center", va="bottom", fontweight="bold")
    ax1.annotate("", xy=(0.015, 0.89), xytext=(0.015, 0.79), xycoords="axes fraction", arrowprops={"arrowstyle": "-|>", "color": NAVY, "lw": 0.8})
    cb = fig.colorbar(hb, ax=ax1, orientation="horizontal", fraction=0.045, pad=0.08)
    cb.set_label("Aggregated L0 generation (TWh yr$^{-1}$; log scale)")
    panel(ax1, "a")

    cont = data["continents"].sort_values("all_twh")
    y = np.arange(len(cont))
    ax2.barh(y - 0.17, cont["all_twh"], height=0.32, color=SKY, label="L0")
    ax2.barh(y + 0.17, cont["l1_twh"], height=0.32, color=BLUE, label="L1")
    ax2.set_yticks(y, cont["continent_v117"].str.replace("North America", "N. America").str.replace("South America", "S. America"))
    ax2.set_xlabel("Generation (TWh yr$^{-1}$)")
    ax2.legend(frameon=False, ncol=2, loc="lower right")
    clean(ax2, "x"); panel(ax2, "b")

    countries = data["countries"].nlargest(10, "all_twh").sort_values("all_twh")
    y = np.arange(len(countries))
    ax3.barh(y, countries["all_twh"], color=PALE, label="L0")
    ax3.barh(y, countries["l1_twh"], color=SKY, label="L1")
    ax3.barh(y, countries["core_l2_lower_twh"], color=BLUE, label="Core L2 lower")
    names = countries["country_v117"].replace({"United States of America": "United States", "Russian Federation": "Russia"})
    ax3.set_yticks(y, names)
    ax3.set_xlabel("Generation (TWh yr$^{-1}$)")
    ax3.legend(frameon=False, loc="lower right")
    clean(ax3, "x"); panel(ax3, "c")
    save(fig, "Fig2")


def draw_fig3(data: dict[str, object]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.flat
    density = data["yield_density"]
    x_edges = np.r_[np.sort(density["x_left_kwh_kwp"].unique()), density["x_right_kwh_kwp"].max()]
    y_edges = np.r_[np.sort(density["y_bottom_kwh_kwp"].unique()), density["y_top_kwh_kwp"].max()]
    counts = density.pivot(
        index="y_bottom_kwh_kwp",
        columns="x_left_kwh_kwp",
        values="count",
    ).sort_index().sort_index(axis=1).to_numpy(dtype=float, copy=True)
    counts[counts == 0] = np.nan
    ax1.pcolormesh(
        x_edges,
        y_edges,
        counts,
        norm=mpl.colors.LogNorm(vmin=1, vmax=np.nanmax(counts)),
        cmap=mpl.colors.LinearSegmentedColormap.from_list("density", [PALE, SKY, BLUE, NAVY]),
        shading="flat",
    )
    lo = min(x_edges.min(), y_edges.min())
    hi = max(x_edges.max(), y_edges.max())
    ax1.plot([lo, hi], [lo, hi], color=RED, linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Published yield (kWh kWp$^{-1}$ yr$^{-1}$)")
    ax1.set_ylabel("Present screen (kWh kWp$^{-1}$ yr$^{-1}$)")
    clean(ax1, None); panel(ax1, "a")

    cov = data["coverage"]
    x = np.arange(len(cov))
    ax2.bar(x - 0.18, cov["l0_generation_twh"], 0.36, color=SKY, label="L0")
    ax2.bar(x + 0.18, cov["l1_generation_twh"], 0.36, color=BLUE, label="L1")
    labels = ["5/5", "10/10", "20/20", "30/30", "20/10", "30/10", "30/20"]
    ax2.set_xticks(x, labels, rotation=35, ha="right")
    ax2.set_xlabel("Reservoir/lake coverage (%)")
    ax2.set_ylabel("Generation (TWh yr$^{-1}$)")
    ax2.legend(frameon=False, ncol=2)
    clean(ax2); panel(ax2, "b")

    ice = data["ice"].iloc[:3].copy()
    ax3.bar(["Reference\nhierarchy", "Woolway +\nproxy", "Temperature\nproxy"], ice["generation_twh"], color=[BLUE, SKY, YELLOW])
    ax3.set_ylabel("L1 generation (TWh yr$^{-1}$)")
    ax3.set_ylim(0, 14_000)
    clean(ax3); panel(ax3, "c")

    wind = data["wind"]
    ax4.bar(["POWER", "Scaled wind", "Zero wind"], wind["legacy_scope_l0_twh"], color=[BLUE, SKY, YELLOW])
    ax4.set_ylabel("Common-scope L0 (TWh yr$^{-1}$)")
    ax4.set_ylim(15_500, 16_800)
    clean(ax4); panel(ax4, "d")
    save(fig, "Fig3")


def draw_fig4(data: dict[str, object]) -> None:
    fig = plt.figure(figsize=(7.2, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, :]); ax2 = fig.add_subplot(gs[1, 0]); ax3 = fig.add_subplot(gs[1, 1])
    seq = data["sequential"].set_index("stage")
    stages = ["L1_ice", "core_conservation", "core_population", "core_L2_lower"]
    stages_c = ["L1_ice", "conservative_conservation", "conservative_population", "conservative_L2_lower"]
    labels = ["L1", "Conservation", "Population", "L2 lower"]
    x = np.arange(4)
    ax1.plot(x, seq.loc[stages, "generation_twh"], marker="o", color=BLUE, label="Core")
    ax1.plot(x, seq.loc[stages_c, "generation_twh"], marker="s", color=RED, label="Conservative")
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Generation (TWh yr$^{-1}$)")
    ax1.legend(frameon=False)
    clean(ax1); panel(ax1, "a")

    evidence = data["evidence"]
    piv = evidence.pivot(index="conservation_definition", columns="evidence_state", values="generation_twh").loc[["core", "conservative"]]
    bottom = np.zeros(2)
    for col, label, color in [
        ("known_no_observed_dryup", "Known, no dry-up", BLUE),
        ("known_observed_dryup", "Observed dry-up", RED),
        ("unknown_public_record", "Unknown", YELLOW),
    ]:
        ax2.bar(["Core", "Conservative"], piv[col], bottom=bottom, color=color, label=label)
        bottom += piv[col].to_numpy(float)
    ax2.set_ylabel("Generation after population screen (TWh yr$^{-1}$)")
    ax2.legend(frameon=False)
    clean(ax2); panel(ax2, "b")

    glev = data["glev"]
    criteria = ["any_exact_zero_month", "at_least_two_exact_zero_months", "minimum_area_at_most_1pct_of_median"]
    pretty = ["Any zero month", "At least two zero months", r"Minimum $\leq$1% median"]
    for offset, definition, color in [(-0.08, "core", BLUE), (0.08, "conservative", RED)]:
        sub = glev[glev["conservation_definition"].eq(definition)].set_index(["dryup_criterion", "evidence_policy"])
        low = np.array([sub.loc[(c, "known_only_lower"), "generation_twh"] for c in criteria])
        high = np.array([sub.loc[(c, "unknown_pass_upper"), "generation_twh"] for c in criteria])
        y = np.arange(3) + offset
        ax3.hlines(y, low, high, color=color, linewidth=5, label=definition.capitalize())
        ax3.scatter(low, y, color=color, s=18, zorder=3); ax3.scatter(high, y, color=color, s=18, zorder=3)
    ax3.set_yticks(np.arange(3), pretty)
    ax3.set_xlabel("L2 generation (TWh yr$^{-1}$)")
    ax3.legend(frameon=False)
    clean(ax3, "x"); panel(ax3, "c")
    save(fig, "Fig4")


def draw_fig5(data: dict[str, object]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.5), constrained_layout=True)
    ax1, ax2, ax3 = axes
    intervals = interval_rows(data["level"])
    names = list(intervals)
    y = np.arange(4)[::-1]
    for pos, name, color in zip(y, names, [MID, SKY, YELLOW, RED]):
        low, high = intervals[name]
        ax1.hlines(pos, low, high, color=color, linewidth=6)
        ax1.scatter([low, high], [pos, pos], color=color, s=18, zorder=3)
    ax1.set_yticks(y, ["Core L2", "Conservative L2", "Core partial L3", "Conservative partial L3"])
    ax1.set_xlabel("Generation (TWh yr$^{-1}$)")
    clean(ax1, "x"); panel(ax1, "a")

    fact = data["factorial"]
    core = fact[fact["conservation_definition"].eq("core")]
    matrix = core.pivot(index="dryup_evidence_policy", columns="infrastructure_mapping_policy", values="generation_twh").loc[["known_only", "unknown_pass"], ["lower_mapping", "upper_mapping"]].to_numpy()
    im = ax2.imshow(matrix, cmap=mpl.colors.LinearSegmentedColormap.from_list("factor", [PALE, SKY, BLUE, NAVY]), aspect="auto")
    ax2.set_xticks([0, 1], ["Lower", "Upper"]); ax2.set_yticks([0, 1], ["Known only", "Unknown pass"])
    ax2.set_xlabel("Mapping policy"); ax2.set_ylabel("Dry-up evidence policy")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color="black")
    cb = fig.colorbar(im, ax=ax2, fraction=0.05, pad=0.04); cb.set_label("TWh yr$^{-1}$")
    panel(ax2, "b")

    hydro = data["hydro"]
    sub = hydro[(hydro["conservation_definition"].eq("core")) & (hydro["dryup_evidence_policy"].eq("known_only"))]
    x = np.arange(2)
    ax3.bar(x - 0.18, sub["with_hydropower_proxy_twh"], 0.36, color=YELLOW, label="With hydropower proxy")
    ax3.bar(x + 0.18, sub["mapped_lines_only_twh"], 0.36, color=BLUE, label="Mapped lines only")
    ax3.set_xticks(x, ["Lower", "Upper"]); ax3.set_xlabel("Mapping policy")
    ax3.set_ylabel("Generation (TWh yr$^{-1}$)")
    ax3.legend(frameon=False)
    clean(ax3); panel(ax3, "c")
    save(fig, "Fig5")


def draw_supplementary(data: dict[str, object]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.5), constrained_layout=True)
    regions = data["yield_regions"].sort_values("pearson_r")
    axes[0].barh(regions["continent"], regions["pearson_r"], color=BLUE)
    axes[0].set_xlabel("Published-yield correlation"); axes[0].set_xlim(0.5, 1.0); clean(axes[0], "x"); panel(axes[0], "a")
    src = data["ice_sources"]
    y = np.arange(len(src)); axes[1].barh(y, src["passed"], color=BLUE, label="Pass"); axes[1].barh(y, src["waterbodies"] - src["passed"], left=src["passed"], color=LIGHT, label="Fail")
    axes[1].set_yticks(y, ["Woolway", "Temperature proxy", "LI-CCR"]); axes[1].set_xlabel("Water bodies"); axes[1].legend(frameon=False); clean(axes[1], "x"); panel(axes[1], "b")
    ice = data["ice"].iloc[:3]; axes[2].bar(["Reference", "Woolway +\nproxy", "Temperature\nproxy"], ice["generation_twh"], color=[BLUE, SKY, YELLOW]); axes[2].set_ylabel("L1 generation (TWh yr$^{-1}$)"); clean(axes[2]); panel(axes[2], "c")
    save(fig, "FigS1")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2), constrained_layout=True)
    eng = data["engineering"].set_index("population").loc[["all", "l1", "core_l2_lower", "core_l3_lower"]]
    labels = ["All", "L1", "Core L2 lower", "Core L3 lower"]
    axes[0, 0].bar(labels, eng["globathy_depth_coverage_pct"], color=[PALE, SKY, MID, BLUE]); axes[0, 0].set_ylabel("GLOBathy coverage (%)"); axes[0, 0].tick_params(axis="x", rotation=30); clean(axes[0, 0]); panel(axes[0, 0], "a")
    axes[0, 1].bar(labels, eng["fpv_reference_lcoe_usd_mwh_median"], color=[PALE, SKY, MID, BLUE]); axes[0, 1].set_ylabel("Median parametric LCOE (USD MWh$^{-1}$)"); axes[0, 1].tick_params(axis="x", rotation=30); clean(axes[0, 1]); panel(axes[0, 1], "b")
    sens = data["engineering_sensitivity"]
    for ax, pop, letter in [(axes[1, 0], "core_l3_lower", "c"), (axes[1, 1], "core_l3_upper", "d")]:
        sub = sens[sens["population"].eq(pop)].copy()
        sub["maximum_depth_limit_m"] = sub["maximum_depth_limit_m"].astype(str)
        sub["lcoe_limit_usd_mwh"] = sub["lcoe_limit_usd_mwh"].astype(str)
        depths = ["none", "20.0", "50.0", "100.0"]; costs = ["none", "75.0", "100.0", "125.0"]
        mat = sub.pivot(index="maximum_depth_limit_m", columns="lcoe_limit_usd_mwh", values="generation_twh").reindex(index=depths, columns=costs).to_numpy()
        im = ax.imshow(mat, cmap=mpl.colors.LinearSegmentedColormap.from_list("eng", [PALE, SKY, BLUE, NAVY]), aspect="auto")
        ax.set_xticks(range(4), ["None", "75", "100", "125"]); ax.set_yticks(range(4), ["None", "20", "50", "100"])
        ax.set_xlabel("LCOE limit (USD MWh$^{-1}$)"); ax.set_ylabel("Maximum-depth limit (m)")
        for i in range(4):
            for j in range(4): ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="TWh yr$^{-1}$"); panel(ax, letter)
    save(fig, "FigS2")


def draw_graphical_abstract(data: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    x = [0.9, 2.9, 4.9, 6.9, 9.0]
    labels = ["Scope", "L0", "L1", "L2\npublic", "L3\npartial"]
    values = [
        "199,976 water bodies",
        "16,713 TWh yr$^{-1}$",
        "10,995 TWh yr$^{-1}$",
        "Core 4,190-4,805\nConservative 2,914-3,404\nTWh yr$^{-1}$",
        "Core 1,595-3,508\nConservative 1,101-2,494\nTWh yr$^{-1}$",
    ]
    colors = [NAVY, BLUE, SKY, YELLOW, RED]
    sizes = [1050, 950, 850, 1500, 1500]
    for i, (xp, label, value, color, size) in enumerate(zip(x, labels, values, colors, sizes)):
        ax.scatter([xp], [2.35], s=size, color=color, edgecolor="white", linewidth=1.0, zorder=3)
        ax.text(xp, 2.35, label, ha="center", va="center", color="white" if i < 3 else "black", fontweight="bold")
        ax.text(xp, 1.45, value, ha="center", va="center", color="black", fontweight="bold")
        if i < len(x) - 1:
            ax.add_patch(FancyArrowPatch((xp + 0.52, 2.35), (x[i + 1] - 0.52, 2.35), arrowstyle="-|>", mutation_scale=12, color=GRAY, linewidth=1.2))
    roles = ["Identity audit", "Reference resource", "Climate eligibility", "Public evidence", "Infrastructure proximity"]
    for xp, role in zip(x, roles): ax.text(xp, 0.65, role, ha="center", va="center", color=NAVY)
    normalize_fonts(fig, {"annotation": 12, "panel_label": 12})
    fig.savefig(OUT / "graphical_abstract.pdf", dpi=300)
    fig.savefig(OUT / "graphical_abstract.png", dpi=300)
    plt.close(fig)


def main() -> None:
    data = load()
    draw_fig1(data); draw_fig2(data); draw_fig3(data); draw_fig4(data); draw_fig5(data)
    draw_supplementary(data); draw_graphical_abstract(data)
    print(f"inventory_rows={len(data['frame'])}")
    print(f"outputs={OUT}")


if __name__ == "__main__":
    main()
