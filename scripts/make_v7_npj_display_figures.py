#!/usr/bin/env python3
# BioEpi-SAFE-LSTM reproducibility code
# Maintainer: Jianyi Zhang

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


MM = 1 / 25.4

SAFE = "bioepi_safe_lstm_common_scale"
BASE = "bioepi_sgq_lstm_no_digital"
FULL = "bioepi_sgq_lstm_full_raw"
PERSIST = "persistence"
GB = "gb_full_raw"
RIDGE_CN = "ridge_raw_cn_search"
RIDGE_NO_DIGITAL = "ridge_no_digital"

MODEL_LABEL = {
    SAFE: "BioEpi-SAFE-LSTM",
    BASE: "Surveillance SGQ-LSTM",
    FULL: "Full digital SGQ-LSTM",
    PERSIST: "Persistence",
    GB: "Digital gradient-boosting expert",
    RIDGE_CN: "China local-search Ridge",
    RIDGE_NO_DIGITAL: "No-digital Ridge",
}

COMPARATOR_LABEL = {
    PERSIST: "Persistence",
    BASE: "Surveillance SGQ-LSTM",
    FULL: "Full digital SGQ-LSTM",
    GB: "Digital gradient boosting",
}

PALETTE = {
    "safe": "#0072B2",
    "surveillance": "#009E73",
    "full": "#56B4E9",
    "persistence": "#6E6E6E",
    "seasonal": "#BDBDBD",
    "local_search": "#D55E00",
    "gb": "#CC79A7",
    "observed": "#111111",
    "interval": "#A6CEE3",
    "orange": "#D55E00",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def save_figure(fig: mpl.figure.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.svg")
    fig.savefig(out_dir / f"{stem}.png", dpi=600)
    plt.close(fig)


def panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.06, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def read_tables(root: Path) -> dict[str, pd.DataFrame]:
    tables = root / "results" / "tables"
    qc = root / "results" / "qc"
    return {
        "headline": pd.read_csv(tables / "v7_headline_external_common_scale.csv"),
        "bootstrap": pd.read_csv(tables / "v7_common_scale_bootstrap.csv"),
        "delay_macro": pd.read_csv(tables / "v7_common_scale_delay_macro.csv"),
        "delay_country": pd.read_csv(tables / "v7_common_scale_delay_by_country.csv"),
        "china_delay2": pd.read_csv(tables / "v7_china_delay2_bootstrap.csv"),
        "regional": pd.read_csv(tables / "v7_external_regional_by_region.csv"),
        "ablation": pd.read_csv(tables / "v7_common_scale_ablation.csv"),
        "ablation_boot": pd.read_csv(tables / "v7_ablation_bootstrap.csv"),
        "national_country": pd.read_csv(tables / "v7_external_national_by_country.csv"),
        "events": pd.read_csv(tables / "v7_event_metrics.csv"),
        "weights": pd.read_csv(tables / "v7_weight_summary.csv"),
        "pred": pd.read_csv(tables / "v7_common_scale_predictions.csv.gz", parse_dates=["origin_week_start", "target_week_start"]),
        "source_availability": pd.read_csv(qc / "country_source_availability.csv"),
        "target_coverage": pd.read_csv(qc / "coverage_qc_by_country_region.csv"),
        "feature_missingness": pd.read_csv(qc / "feature_missingness_in_model_table.csv"),
    }


def reduction_from_diff(diff: pd.Series, denom: pd.Series) -> pd.Series:
    return -100 * diff / denom


def add_reduction_ci(df: pd.DataFrame, denom: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["denominator_WIS"] = denom.to_numpy(dtype=float)
    out["wis_reduction_pct"] = reduction_from_diff(out["mean_diff_A_minus_B"], out["denominator_WIS"])
    out["ci_low_reduction_pct"] = reduction_from_diff(out["ci_high"], out["denominator_WIS"])
    out["ci_high_reduction_pct"] = reduction_from_diff(out["ci_low"], out["denominator_WIS"])
    return out


def lookup_wis(metric: pd.DataFrame, delay_col: str, delay: int, model: str, horizon: int, country: str | None = None) -> float:
    q = (metric[delay_col] == delay) & (metric["model"] == model) & (metric["horizon"] == horizon)
    if country is not None:
        q &= metric["country"] == country
    vals = metric.loc[q, "WIS"].dropna()
    if vals.empty:
        raise KeyError((delay, model, horizon, country))
    return float(vals.iloc[0])


def build_display_tables(data: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    headline = data["headline"].copy()
    boot = data["bootstrap"].copy()
    delay_macro = data["delay_macro"].copy()
    delay_country = data["delay_country"].copy()
    china_boot = data["china_delay2"].copy()
    regional = data["regional"].copy()
    ablation = data["ablation"].copy()
    abboot = data["ablation_boot"].copy()
    national_country = data["national_country"].copy()

    # Figure 3: each model versus persistence, 8-week moving blocks. This table
    # is built from the bundled expert-prediction object by
    # `build_v7_persistence_bootstrap_for_display.py`.
    fig3_path = out_dir / "figure3_models_vs_persistence_bootstrap.csv"
    if fig3_path.exists():
        fig3 = pd.read_csv(fig3_path)
    else:
        fig3 = boot[
            (boot["evaluation"] == "temporal_external_2025_2026")
            & (boot["delay"] == 0)
            & (boot["model_A"] == SAFE)
            & (boot["model_B"].isin([PERSIST, BASE, FULL, GB]))
            & (boot["block_weeks"] == 8)
        ].copy()
        denoms = [lookup_wis(headline, "official_delay_weeks", 0, r.model_B, int(r.horizon)) for r in fig3.itertuples()]
        fig3 = add_reduction_ci(fig3, pd.Series(denoms, index=fig3.index))
        fig3["model_label"] = "BioEpi-SAFE-LSTM"
        fig3["WIS_reduction_vs_persistence_pct"] = fig3["wis_reduction_pct"]
        fig3["ci_low_reduction_pct"] = fig3["ci_low_reduction_pct"]
        fig3["ci_high_reduction_pct"] = fig3["ci_high_reduction_pct"]
        fig3.to_csv(fig3_path, index=False, encoding="utf-8-sig")

    # Figure 4b: SAFE versus surveillance backbone across reporting-delay scenarios.
    fig4_delay = boot[
        (boot["evaluation"] == "temporal_external_2025_2026")
        & (boot["model_A"] == SAFE)
        & (boot["model_B"] == BASE)
        & (boot["block_weeks"] == 8)
    ].copy()
    denoms = [lookup_wis(delay_macro, "official_delay_weeks", int(r.delay), BASE, int(r.horizon)) for r in fig4_delay.itertuples()]
    fig4_delay = add_reduction_ci(fig4_delay, pd.Series(denoms, index=fig4_delay.index))
    fig4_delay.to_csv(out_dir / "figure4_delay_safe_vs_surveillance_bootstrap.csv", index=False, encoding="utf-8-sig")

    # Figure 4c: China delay-2 comparison.
    china_keep = (
        china_boot["model_A"].isin([SAFE, RIDGE_CN])
        & china_boot["model_B"].isin([BASE, RIDGE_NO_DIGITAL])
        & (china_boot["block_weeks"] == 8)
    )
    fig4_china = china_boot[china_keep].copy()
    denoms = []
    for r in fig4_china.itertuples():
        denoms.append(lookup_wis(delay_country, "official_delay_weeks", 2, r.model_B, int(r.horizon), country="CHN"))
    fig4_china = add_reduction_ci(fig4_china, pd.Series(denoms, index=fig4_china.index))
    fig4_china["comparison"] = np.where(fig4_china["model_A"] == RIDGE_CN, "China local-search Ridge", "SAFE-LSTM")
    fig4_china.to_csv(out_dir / "figure4_china_delay2_bootstrap.csv", index=False, encoding="utf-8-sig")

    # Figure 5: regional relative WIS versus persistence.
    safe_reg = regional[regional["model"] == SAFE][["country", "region", "horizon", "WIS", "coverage_80", "n"]].rename(columns={"WIS": "safe_WIS"})
    ref_reg = regional[regional["model"] == PERSIST][["country", "region", "horizon", "WIS"]].rename(columns={"WIS": "persistence_WIS"})
    fig5 = safe_reg.merge(ref_reg, on=["country", "region", "horizon"], how="inner")
    fig5["relative_WIS_vs_persistence"] = fig5["safe_WIS"] / fig5["persistence_WIS"]
    fig5["WIS_reduction_vs_persistence_pct"] = 100 * (1 - fig5["relative_WIS_vs_persistence"])
    fig5.to_csv(out_dir / "figure5_regional_robustness_relative_wis.csv", index=False, encoding="utf-8-sig")

    # Figure 6: adaptive-fusion ablation.
    fig6 = ablation.copy()
    fig6 = fig6[fig6["variant"].notna()].copy()
    boot8 = abboot[
        (abboot["evaluation"] == "temporal_external_2025_2026")
        & (abboot["delay"] == 0)
        & (abboot["block_weeks"] == 8)
        & (abboot["model_A"] == SAFE)
    ].copy()
    boot8["delta_WIS_vs_full_from_boot"] = -boot8["mean_diff_A_minus_B"]
    boot8["ci_low_delta_WIS_vs_full"] = -boot8["ci_high"]
    boot8["ci_high_delta_WIS_vs_full"] = -boot8["ci_low"]
    fig6 = fig6.merge(
        boot8[["model_B", "horizon", "delta_WIS_vs_full_from_boot", "ci_low_delta_WIS_vs_full", "ci_high_delta_WIS_vs_full"]],
        left_on=["model", "horizon"],
        right_on=["model_B", "horizon"],
        how="left",
    )
    fig6["ci_excludes_zero"] = (
        (fig6["ci_low_delta_WIS_vs_full"] > 0) | (fig6["ci_high_delta_WIS_vs_full"] < 0)
    )
    fig6.to_csv(out_dir / "figure6_ablation_delta_wis.csv", index=False, encoding="utf-8-sig")

    calibration = national_country[
        (national_country["evaluation"] == "temporal_external_2025_2026")
        & (national_country["official_delay_weeks"] == 0)
        & (national_country["model"] == SAFE)
    ][
        [
            "country",
            "horizon",
            "coverage_50",
            "coverage_80",
            "coverage_95",
            "width_50",
            "width_80",
            "width_95",
            "n",
        ]
    ].copy()
    calibration.to_csv(out_dir / "supp_calibration_by_country_horizon.csv", index=False, encoding="utf-8-sig")
    data["events"].to_csv(out_dir / "supp_event_warning_national_only.csv", index=False, encoding="utf-8-sig")

    return {"fig3": fig3, "fig4_delay": fig4_delay, "fig4_china": fig4_china, "fig5": fig5, "fig6": fig6}


def draw_box(ax, xy, wh, text, fc="#F7F7F7", ec="#666666", lw=0.7, fontsize=6.5, color="#111111") -> None:
    x, y = xy
    w, h = wh
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012", fc=fc, ec=ec, lw=lw)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=color)


def draw_arrow(ax, start, end, color="#555555", lw=0.8) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7, lw=lw, color=color, shrinkA=2, shrinkB=2))


def figure1(root: Path, figs: Path) -> None:
    fig, ax = plt.subplots(figsize=(180 * MM, 98 * MM), constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_label(ax, "a", x=0.00, y=0.98)
    ax.text(0.00, 0.235, "b", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
    ax.text(0.43, 0.98, "c", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")
    ax.text(0.84, 0.98, "d", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")

    # Left: compact multinational data cards.
    ax.text(0.045, 0.90, "Multinational surveillance and digital streams", fontsize=7, fontweight="bold", ha="left")
    cards = [
        ("China", "ILI%, North/South", "local search + social + Wiki"),
        ("United States", "wILI%, HHS regions", "Google + Wiki + X"),
        ("Japan", "sentinel influenza", "Google + Wiki"),
    ]
    y_positions = [0.72, 0.56, 0.40]
    for (country, outcome, digital), y in zip(cards, y_positions):
        draw_box(ax, (0.035, y), (0.18, 0.10), country, fc="#EAF4FB", ec=PALETTE["safe"], fontsize=6.6)
        ax.text(0.235, y + 0.068, outcome, fontsize=6.1, ha="left", va="center", color="#222222")
        ax.text(0.235, y + 0.030, digital, fontsize=5.8, ha="left", va="center", color="#666666")

    # Center: SAFE-LSTM as the visual anchor.
    draw_arrow(ax, (0.39, 0.57), (0.47, 0.57), color="#555555", lw=1.0)
    ax.text(0.47, 0.90, "Surveillance-anchored adaptive fusion", fontsize=7, fontweight="bold", ha="left")
    draw_box(ax, (0.46, 0.55), (0.19, 0.13), "Surveillance\nLSTM backbone", fc="#EAF4FB", ec=PALETTE["safe"], fontsize=6.7)
    draw_box(ax, (0.46, 0.35), (0.19, 0.13), "Digital expert pool\nLSTM, Ridge, GB", fc="#FFF4E8", ec=PALETTE["orange"], fontsize=6.4)
    draw_box(ax, (0.69, 0.53), (0.14, 0.12), "Activate only\nafter resolved gain", fc="#F7F7F7", ec="#777777", fontsize=5.9)
    draw_box(ax, (0.69, 0.37), (0.14, 0.10), "Auxiliary\nweight <= 50%", fc="#F7F7F7", ec="#777777", fontsize=5.9)
    draw_arrow(ax, (0.65, 0.615), (0.69, 0.59), color=PALETTE["safe"])
    draw_arrow(ax, (0.65, 0.415), (0.69, 0.42), color=PALETTE["orange"])
    draw_arrow(ax, (0.83, 0.59), (0.88, 0.57))
    draw_arrow(ax, (0.83, 0.42), (0.88, 0.50))

    # Right: output and evaluation.
    draw_box(ax, (0.88, 0.47), (0.10, 0.15), "1-4 week\nquantile\nforecasts", fc="#FFFFFF", ec=PALETTE["safe"], fontsize=6.0)
    ax.text(0.88, 0.34, "Primary external endpoint", fontsize=6.4, fontweight="bold", ha="left")
    ax.text(0.88, 0.29, "country-balanced WIS", fontsize=6.0, ha="left")
    ax.text(0.88, 0.25, "paired block bootstrap", fontsize=6.0, ha="left")
    ax.text(0.88, 0.21, "coverage and interval width", fontsize=6.0, ha="left")

    # Bottom: restrained timeline band.
    ax.text(0.045, 0.20, "Evaluation design", fontsize=6.7, fontweight="bold", ha="left")
    x0, y0, h = 0.045, 0.10, 0.055
    segments = [
        (0.25, "training", "#EDEDED"),
        (0.13, "selection", "#D9F0EA"),
        (0.05, "purge", "#FFFFFF"),
        (0.15, "calibration", "#EAF4FB"),
        (0.23, "2025-2026 external evaluation", "#F9E7D6"),
    ]
    x = x0
    for w, label, color in segments:
        ax.add_patch(Rectangle((x, y0), w, h, fc=color, ec="#777777", lw=0.55))
        ax.text(x + w / 2, y0 + h / 2, label, ha="center", va="center", fontsize=5.8)
        x += w
    ax.text(0.045, 0.045, "Fold-2 selected SAFE parameters; rolling fold-3 was locked validation.", fontsize=5.9, ha="left", color="#444444")
    save_figure(fig, figs, "Figure_1_study_design_model")


def plot_forecast_series(ax: mpl.axes.Axes, pred: pd.DataFrame, country: str, horizon: int, country_label: str, title: str | None = None) -> None:
    s = pred[(pred["country"] == country) & (pred["horizon"] == horizon)].sort_values("target_week_start")
    ax.fill_between(
        s["target_week_start"].to_numpy(),
        s["q0.1"].to_numpy(dtype=float),
        s["q0.9"].to_numpy(dtype=float),
        color=PALETTE["interval"],
        alpha=0.45,
        lw=0,
    )
    ax.plot(s["target_week_start"], s["y_true"], color=PALETTE["observed"], lw=0.9)
    ax.plot(s["target_week_start"], s["q0.5"], color=PALETTE["safe"], lw=1.05)
    ax.axhline(0, color="#BBBBBB", lw=0.5, zorder=0)
    ax.grid(axis="y", color="#EAEAEA", lw=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    if title:
        ax.set_title(title, pad=2)
    ax.set_ylabel(country_label)


def figure2(data: dict[str, pd.DataFrame], figs: Path) -> None:
    pred = data["pred"]
    z = pred[
        (pred["evaluation"] == "temporal_external_2025_2026")
        & (pred["official_delay_weeks"] == 0)
        & (pred["model"] == SAFE)
        & (pred["region"] == "national")
        & (pred["horizon"].isin([1, 4]))
    ].copy()
    fig, axes = plt.subplots(3, 2, figsize=(180 * MM, 122 * MM), sharex=True, constrained_layout=True)
    countries = [("CHN", "China"), ("USA", "United States"), ("JPN", "Japan")]
    letters = [["a", "b"], ["c", "d"], ["e", "f"]]
    for i, (country, label) in enumerate(countries):
        ax1 = axes[i, 0]
        ax4 = axes[i, 1]
        plot_forecast_series(ax1, z, country, 1, f"{label}\nactivity", "1-week ahead" if i == 0 else None)
        plot_forecast_series(ax4, z, country, 4, "", "4-week ahead" if i == 0 else None)
        panel_label(ax1, letters[i][0])
        panel_label(ax4, letters[i][1])
        if i < 2:
            ax1.tick_params(labelbottom=False)
            ax4.tick_params(labelbottom=False)
        else:
            ax1.set_xlabel("Target week")
            ax4.set_xlabel("Target week")
        ax4.set_ylabel("")

    forecast_handles = [
        Line2D([0], [0], color=PALETTE["observed"], lw=1.0, label="Observed"),
        Line2D([0], [0], color=PALETTE["safe"], lw=1.1, label="SAFE median"),
        Rectangle((0, 0), 1, 1, fc=PALETTE["interval"], alpha=0.45, ec="none", label="80% interval"),
    ]
    fig.legend(handles=forecast_handles, loc="upper center", bbox_to_anchor=(0.53, 1.02), ncol=3, frameon=False)
    save_figure(fig, figs, "Figure_2_external_probabilistic_forecasts")


def short_model_label(label: str) -> str:
    return {
        "BioEpi-SAFE-LSTM": "SAFE-LSTM",
        "Surveillance SGQ-LSTM": "Surv-LSTM",
        "Full digital SGQ-LSTM": "Full-digital LSTM",
        "Digital gradient boosting": "GB expert",
        "Seasonal naive": "Seasonal naive",
    }.get(label, label)


def figure3(display: dict[str, pd.DataFrame], figs: Path) -> None:
    perf = display["fig3"].copy()
    order = [
        "Digital gradient boosting",
        "Full digital SGQ-LSTM",
        "Surveillance SGQ-LSTM",
        "BioEpi-SAFE-LSTM",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(180 * MM, 78 * MM), sharey=True, constrained_layout=True)
    x_left, x_right = -42, 56
    for idx, h in enumerate([1, 2, 3, 4]):
        ax = axes[idx]
        panel_label(ax, chr(ord("a") + idx))
        s = perf[(perf["horizon"] == h) & (perf["model_label"].isin(order))].copy()
        vals = s.set_index("model_label").reindex(order)
        y = np.arange(len(order))
        ax.axvline(0, color="#777777", lw=0.7)
        for yi, model_label in zip(y, order):
            row = vals.loc[model_label]
            x = float(row["WIS_reduction_vs_persistence_pct"])
            lo = float(row["ci_low_reduction_pct"])
            hi = float(row["ci_high_reduction_pct"])
            color = PALETTE["safe"] if model_label == "BioEpi-SAFE-LSTM" else "#8A8A8A"
            face = color if model_label == "BioEpi-SAFE-LSTM" else "#FFFFFF"
            ax.errorbar(
                x,
                yi,
                xerr=np.array([[x - lo], [hi - x]]),
                fmt="o",
                color=color,
                ecolor=color,
                markerfacecolor=face,
                markeredgecolor=color,
                elinewidth=0.9,
                capsize=2.0,
                markersize=4.0 if model_label == "BioEpi-SAFE-LSTM" else 3.4,
            )
        ax.set_xlim(x_left, x_right)
        ax.set_title(f"h = {h}", pad=2)
        ax.set_xlabel("WIS reduction vs persistence, %")
        ax.grid(axis="x", color="#EAEAEA", lw=0.5)
        if idx == 0:
            ax.set_yticks(y, [short_model_label(v) for v in order])
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
    save_figure(fig, figs, "Figure_3_main_probabilistic_performance")


def figure4(data: dict[str, pd.DataFrame], display: dict[str, pd.DataFrame], figs: Path) -> None:
    delay = display["fig4_delay"].copy()
    china = display["fig4_china"].copy()
    weights = data["weights"].copy()
    fig = plt.figure(figsize=(180 * MM, 118 * MM), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    ax = axes[0]
    panel_label(ax, "a")
    s = delay[(delay["delay"] == 0)].sort_values("horizon")
    ax.axhline(0, color="#777777", lw=0.7)
    ax.errorbar(
        s["horizon"],
        s["wis_reduction_pct"],
        yerr=np.vstack([s["wis_reduction_pct"] - s["ci_low_reduction_pct"], s["ci_high_reduction_pct"] - s["wis_reduction_pct"]]),
        fmt="o",
        color=PALETTE["safe"],
        capsize=2,
        lw=0.9,
    )
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Forecast horizon, weeks")
    ax.set_ylabel("WIS reduction vs Surv-LSTM, %")
    ax.set_title("No delay", pad=2)
    ax.grid(axis="y", color="#EAEAEA", lw=0.5)

    ax = axes[1]
    panel_label(ax, "b")
    delay_mat = delay.pivot(index="delay", columns="horizon", values="wis_reduction_pct").reindex([0, 1, 2])
    im = ax.imshow(delay_mat.to_numpy(float), aspect="auto", cmap=LinearSegmentedColormap.from_list("white_blue", ["#F7F7F7", PALETTE["safe"]]), vmin=0, vmax=max(10, float(np.nanmax(delay_mat.to_numpy(float)))))
    ax.set_xticks(range(4), [1, 2, 3, 4])
    ax.set_yticks(range(3), ["0-week", "1-week", "2-week"])
    ax.set_xlabel("Forecast horizon, weeks")
    ax.set_ylabel("Official-report delay")
    ax.set_title("Reporting-delay gain", pad=2)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.015)
    cbar.set_label("WIS reduction, %")

    ax = axes[2]
    panel_label(ax, "c")
    s = china[china["comparison"] == "China local-search Ridge"].sort_values("horizon")
    y = np.arange(len(s))
    x = s["wis_reduction_pct"].to_numpy(float)
    lo = s["ci_low_reduction_pct"].to_numpy(float)
    hi = s["ci_high_reduction_pct"].to_numpy(float)
    ax.axvline(0, color="#777777", lw=0.7)
    ax.errorbar(
        x,
        y,
        xerr=np.vstack([x - lo, hi - x]),
        fmt="s",
        color=PALETTE["local_search"],
        ecolor=PALETTE["local_search"],
        markerfacecolor=PALETTE["local_search"],
        capsize=2.0,
        markersize=4.0,
    )
    ax.set_yticks(y, [f"h = {h}" for h in s["horizon"]])
    ax.invert_yaxis()
    ax.set_xlabel("WIS reduction vs no-digital Ridge, %")
    ax.set_title("China local search, 2-week delay", pad=2)
    ax.grid(axis="x", color="#EAEAEA", lw=0.5)

    ax = axes[3]
    panel_label(ax, "d")
    w = weights[
        (weights["evaluation"] == "temporal_external_2025_2026")
        & (weights["official_delay_weeks"] == 0)
        & (weights["region"] == "national")
        & (weights["variant"] == "full")
    ].copy()
    aux = (
        w[w["expert"] != BASE]
        .groupby(["country", "horizon"])["mean"]
        .sum()
        .unstack("horizon")
        .reindex(["CHN", "USA", "JPN"])
    )
    im = ax.imshow(100 * aux.to_numpy(float), aspect="auto", cmap=LinearSegmentedColormap.from_list("white_blue", ["#F7F7F7", PALETTE["safe"]]), vmin=0, vmax=50)
    ax.set_xticks(range(4), [1, 2, 3, 4])
    ax.set_yticks(range(3), ["China", "United States", "Japan"])
    ax.set_xlabel("Forecast horizon, weeks")
    ax.set_title("Mean digital-expert activation", pad=2)
    cbar = fig.colorbar(im, ax=ax, shrink=0.83, pad=0.015)
    cbar.set_label("% total SAFE weight")
    save_figure(fig, figs, "Figure_4_conditional_digital_utility")


def figure5(display: dict[str, pd.DataFrame], figs: Path) -> None:
    df = display["fig5"].copy()
    ab = display["fig6"].copy()
    cmap = LinearSegmentedColormap.from_list("blue_white_orange", ["#0072B2", "#F7F7F7", "#D55E00"])
    vals = df["relative_WIS_vs_persistence"].to_numpy(float)
    vmin = min(0.55, float(np.nanmin(vals)))
    vmax = max(1.20, float(np.nanmax(vals)))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    fig = plt.figure(figsize=(180 * MM, 96 * MM), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.65, 0.72, 1.55])
    ax_usa = fig.add_subplot(gs[0, 0])
    ax_chn = fig.add_subplot(gs[0, 1])
    ax_comp = fig.add_subplot(gs[0, 2])

    panel_label(ax_usa, "a")
    usa = df[df["country"] == "USA"].pivot(index="region", columns="horizon", values="relative_WIS_vs_persistence")
    usa = usa.reindex([f"hhs{i}" for i in range(1, 11)])
    im = ax_usa.imshow(usa.to_numpy(float), aspect="auto", cmap=cmap, norm=norm)
    ax_usa.set_xticks(range(4), [1, 2, 3, 4])
    ax_usa.set_yticks(range(len(usa.index)), [x.upper() for x in usa.index])
    ax_usa.set_xlabel("Forecast horizon, weeks")
    ax_usa.set_title("United States HHS", pad=2)

    panel_label(ax_chn, "b")
    chn = df[df["country"] == "CHN"].pivot(index="region", columns="horizon", values="relative_WIS_vs_persistence")
    chn = chn.reindex(["northern_provinces", "southern_provinces"])
    im2 = ax_chn.imshow(chn.to_numpy(float), aspect="auto", cmap=cmap, norm=norm)
    ax_chn.set_xticks(range(4), [1, 2, 3, 4])
    ax_chn.set_yticks(range(len(chn.index)), ["North", "South"])
    ax_chn.set_xlabel("Forecast horizon, weeks")
    ax_chn.set_title("China", pad=2)
    cbar = fig.colorbar(im2, ax=[ax_usa, ax_chn], shrink=0.83, pad=0.013)
    cbar.set_label("Relative WIS vs persistence")

    panel_label(ax_comp, "c")
    keep = [
        "base_only",
        "static_equal_digital_blend",
        "no_gradient_boosting",
        "no_source_specific_ridge",
        "no_full_lstm_expert",
    ]
    comp = ab[(ab["horizon"] == 2) & (ab["variant"].isin(keep))].copy()
    comp["variant"] = pd.Categorical(comp["variant"], categories=keep, ordered=True)
    comp = comp.sort_values("variant")
    y = np.arange(len(comp))
    x = comp["delta_WIS_vs_full"].to_numpy(float)
    lo = comp["ci_low_delta_WIS_vs_full"].to_numpy(float)
    hi = comp["ci_high_delta_WIS_vs_full"].to_numpy(float)
    xerr = np.vstack([x - lo, hi - x])
    colors = [PALETTE["safe"] if val > 0 else PALETTE["orange"] for val in x]
    ax_comp.axvline(0, color="#777777", lw=0.7)
    for yi, xv, xe, color in zip(y, x, xerr.T, colors):
        ax_comp.errorbar(xv, yi, xerr=np.array([[xe[0]], [xe[1]]]), fmt="o", color=color, ecolor=color, capsize=2.0, markersize=3.7)
    compact_labels = {
        "base_only": "Backbone\nonly",
        "static_equal_digital_blend": "Equal\nblend",
        "no_gradient_boosting": "No GB\nexpert",
        "no_source_specific_ridge": "No Ridge\nexperts",
        "no_full_lstm_expert": "No full-digital\nexpert",
    }
    ax_comp.set_yticks(y, [compact_labels.get(v, v) for v in comp["variant"].astype(str)])
    ax_comp.invert_yaxis()
    ax_comp.set_xlabel("Delta WIS vs SAFE-LSTM")
    ax_comp.set_title("Key components, h = 2", pad=2)
    ax_comp.grid(axis="x", color="#EAEAEA", lw=0.5)
    save_figure(fig, figs, "Figure_5_regional_components")


def variant_label(v: str) -> str:
    return {
        "base_only": "Backbone only",
        "static_equal_digital_blend": "Static equal blend",
        "no_full_lstm_expert": "No full-digital\nLSTM expert",
        "no_source_specific_ridge": "No source-specific\nRidge experts",
        "no_gradient_boosting": "No gradient\nboosting",
        "no_google_wiki": "No Google/\nWikipedia",
        "no_china_local_search": "No China\nlocal search",
        "no_social": "No social/\nopen experts",
    }.get(v, v)


def feature_label(feature: str, max_len: int = 34) -> str:
    label = str(feature)
    replacements = {
        "流行性感冒": "influenza",
        "流感": "flu",
        "甲流": "A-flu",
        "乙流": "B-flu",
        "疫苗": "vaccine",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    label = (
        label.replace("toutiao_raw__CHN__", "Toutiao raw: ")
        .replace("toutiao_zglobal__CHN__", "Toutiao z: ")
        .replace("douyin_raw__CHN__", "Douyin raw: ")
        .replace("douyin_zglobal__CHN__", "Douyin z: ")
        .replace("weibo_raw__CN__", "Weibo raw: ")
        .replace("weibo_zglobal__CN__", "Weibo z: ")
        .replace("wiki_global__", "Wiki: ")
        .replace("google_global__", "Google: ")
        .replace("_search_index", "")
        .replace("_", " ")
    )
    return label[:max_len]


def figure6(display: dict[str, pd.DataFrame], figs: Path) -> None:
    df = display["fig6"].copy()
    order = [
        "base_only",
        "static_equal_digital_blend",
        "no_full_lstm_expert",
        "no_source_specific_ridge",
        "no_gradient_boosting",
        "no_google_wiki",
        "no_china_local_search",
        "no_social",
    ]
    df["variant"] = pd.Categorical(df["variant"], categories=order, ordered=True)
    mat = df.pivot(index="variant", columns="horizon", values="delta_WIS_vs_full").reindex(order)
    sig = df.pivot(index="variant", columns="horizon", values="ci_excludes_zero").reindex(order).fillna(False)
    cmap = LinearSegmentedColormap.from_list("orange_white_blue", ["#D55E00", "#F7F7F7", "#0072B2"])
    vmax = max(0.02, float(np.nanmax(np.abs(mat.to_numpy(float)))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(180 * MM, 88 * MM), constrained_layout=True)
    panel_label(ax, "a", x=-0.04)
    im = ax.imshow(mat.to_numpy(float), aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(4), [1, 2, 3, 4])
    ax.set_yticks(range(len(order)), [variant_label(v) for v in order])
    ax.set_xlabel("Forecast horizon, weeks")
    ax.set_ylabel("Alternative model or component removal")
    for i, v in enumerate(order):
        for j, h in enumerate([1, 2, 3, 4]):
            val = mat.loc[v, h]
            if np.isfinite(val):
                txt = f"{val:+.3f}"
                if bool(sig.loc[v, h]):
                    txt += "*"
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.6)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.012)
    cbar.set_label("Delta WIS vs SAFE-LSTM")
    save_figure(fig, figs, "Extended_Data_Figure_2_ablation_components")


def extended_data_coverage(data: dict[str, pd.DataFrame], figs: Path) -> None:
    src = data["source_availability"].copy()
    cov = data["target_coverage"].copy()
    miss = data["feature_missingness"].copy()
    platform_cols = [
        ("google_weeks", "Google"),
        ("wiki_weeks", "Wikipedia"),
        ("baidu_weeks", "Baidu"),
        ("toutiao_weeks", "Toutiao"),
        ("douyin_weeks", "Douyin"),
        ("weibo_weeks", "Weibo"),
        ("twitter_weeks", "X/Twitter"),
    ]
    countries = ["CHN", "USA", "JPN"]
    mat = src.set_index("country")[[c for c, _ in platform_cols]].div(src.set_index("country")["weeks"], axis=0).reindex(countries) * 100

    fig = plt.figure(figsize=(180 * MM, 100 * MM), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.20, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    im = ax.imshow(mat.to_numpy(float), aspect="auto", cmap=LinearSegmentedColormap.from_list("white_blue", ["#F7F7F7", PALETTE["safe"]]), vmin=0, vmax=100)
    ax.set_xticks(range(len(platform_cols)), [label for _, label in platform_cols], rotation=45, ha="right")
    ax.set_yticks(range(len(countries)), ["China", "United States", "Japan"])
    ax.set_title("Digital-source weekly availability", pad=2)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=5.4, color="#111111")
    cbar = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.018)
    cbar.set_label("% weeks")

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b")
    order = (
        cov.assign(
            sort_key=lambda x: x["country"].map({"CHN": 0, "JPN": 1, "USA": 2}).fillna(9),
            label=lambda x: x["country"] + " " + x["region"].str.replace("_", " ", regex=False),
        )
        .sort_values(["sort_key", "region"])
        .reset_index(drop=True)
    )
    y = np.arange(len(order))
    colors = [PALETTE["safe"] if c == "CHN" else (PALETTE["orange"] if c == "JPN" else "#8A8A8A") for c in order["country"]]
    ax.barh(y, order["coverage_percent"], color=colors, alpha=0.82)
    ax.set_yticks(y, order["label"])
    ax.invert_yaxis()
    ax.set_xlim(95, 100.4)
    ax.set_xlabel("Observed target coverage, %")
    ax.set_title("Gold-standard outcome completeness", pad=2)
    ax.grid(axis="x", color="#EAEAEA", lw=0.5)

    ax = fig.add_subplot(gs[0, 2])
    panel_label(ax, "c")
    m = miss[~miss["feature"].str.endswith("_has_data")].copy()
    m = m.sort_values("missing_rate", ascending=False).head(12).iloc[::-1]
    labels = [feature_label(x) for x in m["feature"]]
    ax.barh(np.arange(len(m)), 100 * m["missing_rate"], color="#9A9A9A", alpha=0.8)
    ax.set_yticks(np.arange(len(m)), labels)
    ax.set_xlabel("Missing rows, %")
    ax.set_title("Highest feature missingness", pad=2)
    ax.grid(axis="x", color="#EAEAEA", lw=0.5)
    save_figure(fig, figs, "Extended_Data_Figure_1_data_coverage")


def extended_data_events(data: dict[str, pd.DataFrame], figs: Path) -> None:
    ev = data["events"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 105 * MM), constrained_layout=True)
    country_colors = {"CHN": PALETTE["safe"], "USA": "#7A7A7A", "JPN": PALETTE["orange"]}
    panels = [
        ("sensitivity", "Sensitivity"),
        ("PPV", "Positive predictive value"),
        ("F1", "F1 score"),
    ]
    for ax, (col, title), label in zip(axes.flat[:3], panels, ["a", "b", "c"]):
        panel_label(ax, label)
        for country, group in ev.groupby("country"):
            group = group.sort_values("horizon")
            ax.plot(group["horizon"], group[col], marker="o", lw=1.0, color=country_colors.get(country, "#999999"), label=country)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Forecast horizon, weeks")
        ax.set_ylabel(title)
        ax.grid(axis="y", color="#EAEAEA", lw=0.5)
        ax.set_title(title, pad=2)
    axes.flat[0].legend(frameon=False, loc="lower left")

    ax = axes.flat[3]
    panel_label(ax, "d")
    width = 0.24
    xbase = np.arange(1, 5)
    for k, country in enumerate(["CHN", "USA", "JPN"]):
        group = ev[ev["country"] == country].sort_values("horizon")
        ax.bar(xbase + (k - 1) * width, group["n_alarms"] / group["n_events"], width=width, color=country_colors[country], alpha=0.78, label=country)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Forecast horizon, weeks")
    ax.set_ylabel("Alarms per event")
    ax.set_title("Alarm burden", pad=2)
    ax.grid(axis="y", color="#EAEAEA", lw=0.5)
    save_figure(fig, figs, "Extended_Data_Figure_3_event_warning")


def extended_data_bootstrap_sensitivity(data: dict[str, pd.DataFrame], figs: Path) -> None:
    boot = data["bootstrap"].copy()
    headline = data["headline"].copy()
    rows = []
    for comp, comp_label in [(BASE, "Surveillance SGQ-LSTM"), (FULL, "Full digital SGQ-LSTM")]:
        s = boot[
            (boot["evaluation"] == "temporal_external_2025_2026")
            & (boot["delay"] == 0)
            & (boot["model_A"] == SAFE)
            & (boot["model_B"] == comp)
        ].copy()
        for r in s.itertuples():
            denom = lookup_wis(headline, "official_delay_weeks", 0, comp, int(r.horizon))
            rows.append(
                {
                    "comparison": comp_label,
                    "horizon": int(r.horizon),
                    "block_weeks": int(r.block_weeks),
                    "reduction": -100 * float(r.mean_diff_A_minus_B) / denom,
                    "lo": -100 * float(r.ci_high) / denom,
                    "hi": -100 * float(r.ci_low) / denom,
                }
            )
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 78 * MM), sharey=True, constrained_layout=True)
    colors = {4: PALETTE["safe"], 8: "#4C9F70", 13: PALETTE["orange"]}
    offsets = {4: -0.12, 8: 0.0, 13: 0.12}
    for ax, comp, label in zip(axes, ["Surveillance SGQ-LSTM", "Full digital SGQ-LSTM"], ["a", "b"]):
        panel_label(ax, label)
        ax.axhline(0, color="#777777", lw=0.7)
        for block in [4, 8, 13]:
            s = df[(df["comparison"] == comp) & (df["block_weeks"] == block)].sort_values("horizon")
            x = s["horizon"].to_numpy(float) + offsets[block]
            y = s["reduction"].to_numpy(float)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - s["lo"].to_numpy(float), s["hi"].to_numpy(float) - y]),
                fmt="o",
                color=colors[block],
                capsize=1.8,
                markersize=3.3,
                label=f"{block}-week blocks",
            )
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("Forecast horizon, weeks")
        ax.set_title(f"SAFE vs {comp}", pad=2)
        ax.grid(axis="y", color="#EAEAEA", lw=0.5)
    axes[0].set_ylabel("WIS reduction, %")
    axes[1].legend(frameon=False, loc="lower left")
    save_figure(fig, figs, "Extended_Data_Figure_4_bootstrap_sensitivity")


def extended_data_source_weights(data: dict[str, pd.DataFrame], figs: Path) -> None:
    w = data["weights"].copy()
    w = w[
        (w["evaluation"] == "temporal_external_2025_2026")
        & (w["official_delay_weeks"] == 0)
        & (w["region"] == "national")
        & (w["variant"] == "full")
    ].copy()
    expert_order = [
        BASE,
        FULL,
        GB,
        "ridge_raw_cn_search",
        "ridge_google_wiki",
        "ridge_global_open",
        "ridge_social_raw",
    ]
    expert_colors = {
        BASE: "#C7C7C7",
        FULL: "#56B4E9",
        GB: "#CC79A7",
        "ridge_raw_cn_search": PALETTE["local_search"],
        "ridge_google_wiki": "#009E73",
        "ridge_global_open": "#8A8A8A",
        "ridge_social_raw": "#E69F00",
    }
    expert_labels = w.drop_duplicates("expert").set_index("expert")["expert_label"].to_dict()
    fig, ax = plt.subplots(figsize=(180 * MM, 82 * MM), constrained_layout=True)
    panel_label(ax, "a", x=-0.035)
    x_positions = []
    tick_labels = []
    x = 0.0
    for country in ["CHN", "USA", "JPN"]:
        for h in [1, 2, 3, 4]:
            x_positions.append(x)
            tick_labels.append(f"{country}\nh{h}")
            bottom = 0.0
            for expert in expert_order:
                val = w[(w["country"] == country) & (w["horizon"] == h) & (w["expert"] == expert)]["mean"].sum()
                if val > 0:
                    ax.bar(x, val, bottom=bottom, width=0.72, color=expert_colors[expert], edgecolor="white", linewidth=0.35)
                    bottom += val
            x += 1.0
        x += 0.65
    ax.set_xticks(x_positions, tick_labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Mean SAFE ensemble weight")
    ax.set_title("Country- and horizon-specific source/expert weights", pad=2)
    ax.grid(axis="y", color="#EAEAEA", lw=0.5)
    handles = [Rectangle((0, 0), 1, 1, fc=expert_colors[e], ec="none", label=expert_labels.get(e, e)) for e in expert_order]
    ax.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save_figure(fig, figs, "Extended_Data_Figure_5_source_expert_weights")


def extended_data_regional_calibration(data: dict[str, pd.DataFrame], figs: Path) -> None:
    reg = data["regional"]
    reg = reg[
        (reg["evaluation"] == "temporal_external_2025_2026")
        & (reg["official_delay_weeks"] == 0)
        & (reg["model"] == SAFE)
    ].copy()
    cmap = LinearSegmentedColormap.from_list("orange_white_blue", [PALETTE["orange"], "#F7F7F7", PALETTE["safe"]])
    norm = mpl.colors.Normalize(vmin=0.45, vmax=1.0)
    fig = plt.figure(figsize=(180 * MM, 110 * MM), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 0.65], height_ratios=[1, 1])
    specs = [
        ("USA", "coverage_80", "a", "United States HHS, 80% interval", [f"hhs{i}" for i in range(1, 11)], [x.upper() for x in [f"hhs{i}" for i in range(1, 11)]], 80),
        ("CHN", "coverage_80", "b", "China regions, 80% interval", ["northern_provinces", "southern_provinces"], ["North", "South"], 80),
        ("USA", "coverage_95", "c", "United States HHS, 95% interval", [f"hhs{i}" for i in range(1, 11)], [x.upper() for x in [f"hhs{i}" for i in range(1, 11)]], 95),
        ("CHN", "coverage_95", "d", "China regions, 95% interval", ["northern_provinces", "southern_provinces"], ["North", "South"], 95),
    ]
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    last_im = None
    for ax, (country, col, label, title, region_order, region_labels, target) in zip(axes, specs):
        panel_label(ax, label)
        mat = reg[reg["country"] == country].pivot(index="region", columns="horizon", values=col).reindex(region_order)
        last_im = ax.imshow(mat.to_numpy(float), aspect="auto", cmap=cmap, norm=norm)
        ax.set_xticks(range(4), [1, 2, 3, 4])
        ax.set_yticks(range(len(region_order)), region_labels)
        ax.set_xlabel("Forecast horizon, weeks")
        ax.set_title(title, pad=2)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = 100 * mat.iloc[i, j]
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=5.4)
    cbar = fig.colorbar(last_im, ax=axes, shrink=0.78, pad=0.012)
    cbar.set_label("Empirical coverage")
    save_figure(fig, figs, "Extended_Data_Figure_6_regional_calibration")


def extended_data_national_calibration(data: dict[str, pd.DataFrame], figs: Path) -> None:
    headline = data["headline"].copy()
    national = data["national_country"].copy()
    cal = headline[
        (headline["evaluation"] == "temporal_external_2025_2026")
        & (headline["official_delay_weeks"] == 0)
        & (headline["model"] == SAFE)
    ].sort_values("horizon")
    by_country = national[
        (national["evaluation"] == "temporal_external_2025_2026")
        & (national["official_delay_weeks"] == 0)
        & (national["model"] == SAFE)
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 78 * MM), constrained_layout=True)
    ax = axes[0]
    panel_label(ax, "a")
    coverage_specs = [
        ("coverage_50", 50, "#9E9E9E", "50%"),
        ("coverage_80", 80, PALETTE["safe"], "80%"),
        ("coverage_95", 95, PALETTE["orange"], "95%"),
    ]
    for col, nominal, color, label in coverage_specs:
        ax.axhline(nominal, color=color, lw=0.55, ls="--", alpha=0.55)
        ax.plot(cal["horizon"], 100 * cal[col], marker="o", lw=1.0, color=color, label=label)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(35, 102)
    ax.set_xlabel("Forecast horizon, weeks")
    ax.set_ylabel("Empirical coverage, %")
    ax.set_title("Country-balanced coverage", pad=2)
    ax.grid(axis="y", color="#EAEAEA", lw=0.5)
    ax.legend(frameon=False, loc="upper right", title="Nominal")

    ax = axes[1]
    panel_label(ax, "b")
    country_colors = {"CHN": PALETTE["safe"], "USA": "#7A7A7A", "JPN": PALETTE["orange"]}
    for country, group in by_country.groupby("country"):
        group = group.sort_values("horizon")
        ax.plot(group["horizon"], 100 * group["coverage_80"], marker="o", lw=1.0, color=country_colors.get(country, "#999999"), label=country)
    ax.axhline(80, color="#777777", lw=0.65, ls="--")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(35, 102)
    ax.set_xlabel("Forecast horizon, weeks")
    ax.set_ylabel("80% interval coverage, %")
    ax.set_title("National coverage by country", pad=2)
    ax.grid(axis="y", color="#EAEAEA", lw=0.5)
    ax.legend(frameon=False, loc="lower left")
    save_figure(fig, figs, "Extended_Data_Figure_7_national_calibration")


def write_readme(root: Path) -> None:
    text = """# Breathable npj-Style Display Figures for BioEpi-SAFE-LSTM v7

This addendum redraws the frozen v7 results into a cleaner five-main-figure structure plus Extended Data. It does not rerun, retune, or change any model.

## Output folders

- `results/figures_npj/`: PDF, SVG, and 600 dpi PNG versions of main Figures 1-5 plus Extended Data Figures 1-7.
- `results/tables/npj_display/`: tidy plotting tables used by the figures.
- `scripts/make_v7_npj_display_figures.py`: reproducible plotting script.
- `scripts/build_v7_persistence_bootstrap_for_display.py`: optional builder for the model-vs-persistence bootstrap table used in Figure 3.

## Main figures

1. `Figure_1_study_design_model`: data architecture, evaluation timeline, and SAFE-LSTM schematic.
2. `Figure_2_external_probabilistic_forecasts`: external probabilistic forecast curves only.
3. `Figure_3_main_probabilistic_performance`: model-wise WIS reduction versus persistence with 8-week paired moving-block bootstrap CIs; seasonal naive is moved out of the main forest plot.
4. `Figure_4_conditional_digital_utility`: no-delay digital gain, reporting-delay heatmap, China local-search value, and digital-expert activation.
5. `Figure_5_regional_components`: USA/China regional robustness plus compact key component checks.

## Extended Data

- `Extended_Data_Figure_1_data_coverage`: digital-source availability, gold-standard outcome coverage, and feature missingness.
- `Extended_Data_Figure_2_ablation_components`: full adaptive-fusion ablation matrix; asterisks mark 8-week bootstrap CIs excluding zero.
- `Extended_Data_Figure_3_event_warning`: event-warning sensitivity, PPV, F1, and alarm burden.
- `Extended_Data_Figure_4_bootstrap_sensitivity`: 4/8/13-week block-bootstrap sensitivity for SAFE versus key comparators.
- `Extended_Data_Figure_5_source_expert_weights`: full source/expert weights by country and horizon.
- `Extended_Data_Figure_6_regional_calibration`: regional 80% and 95% interval coverage heatmaps.
- `Extended_Data_Figure_7_national_calibration`: country-balanced and country-specific national coverage.

## Figure legend notes

- Error bars are 95% paired moving-block bootstrap intervals using 8-week origin blocks unless otherwise stated.
- Figure 3 WIS reductions are calculated as `(persistence WIS - model WIS) / persistence WIS * 100`.
- Figure 4 WIS reductions are calculated against the comparator named in the panel heading.
- Forecast time-series panels use the common target scale used in v7 cross-model scoring.
- Regional heatmaps use relative WIS versus persistence; values below 1 favor SAFE-LSTM.
"""
    (root / "NPJ_DISPLAY_FIGURES_README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    set_style()
    figs = root / "results" / "figures_npj"
    display_tables = root / "results" / "tables" / "npj_display"
    figs.mkdir(parents=True, exist_ok=True)
    for pattern in ["Figure_*.pdf", "Figure_*.svg", "Figure_*.png", "Extended_Data_Figure_*.pdf", "Extended_Data_Figure_*.svg", "Extended_Data_Figure_*.png"]:
        for old in figs.glob(pattern):
            old.unlink()
    data = read_tables(root)
    display = build_display_tables(data, display_tables)
    figure1(root, figs)
    figure2(data, figs)
    figure3(display, figs)
    figure4(data, display, figs)
    figure5(display, figs)
    extended_data_coverage(data, figs)
    figure6(display, figs)
    extended_data_events(data, figs)
    extended_data_bootstrap_sensitivity(data, figs)
    extended_data_source_weights(data, figs)
    extended_data_regional_calibration(data, figs)
    extended_data_national_calibration(data, figs)
    write_readme(root)
    manifest = {
        "version": "v7_npj_breathable_display_figures",
        "main_figures": sorted(p.name for p in figs.glob("Figure_*.pdf")),
        "extended_figures": sorted(p.name for p in figs.glob("Extended_Data_Figure_*.pdf")),
        "tables": sorted(p.name for p in display_tables.glob("*.csv")),
        "note": "Breathable five-main-figure display redraw only; no model retraining or retuning.",
    }
    (root / "results" / "metadata" / "manifest_v7_npj_display_figures.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
