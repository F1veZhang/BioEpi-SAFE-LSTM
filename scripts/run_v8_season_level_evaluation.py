#!/usr/bin/env python3
# BioEpi-SAFE-LSTM reproducibility code
# Maintainer: Jianyi Zhang

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd


MM = 1 / 25.4


def load_v7_module():
    path = Path(__file__).with_name("run_v7_common_scale_safe.py")
    spec = importlib.util.spec_from_file_location("v7safe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


v7 = load_v7_module()
SAFE = v7.ADAPT
BASE = v7.BASE
FULL = v7.FULL
PERSIST = "persistence"
GB = "gb_full_raw"
RIDGE_CN = "ridge_raw_cn_search"
RIDGE_NO_DIGITAL = "ridge_no_digital"

MODEL_SHORT = {
    SAFE: "SAFE-LSTM",
    BASE: "Surv-LSTM",
    FULL: "Full-digital LSTM",
    GB: "GB expert",
    PERSIST: "Persistence",
    "seasonal_naive": "Seasonal naive",
    RIDGE_CN: "CN search Ridge",
    RIDGE_NO_DIGITAL: "No-digital Ridge",
}

PALETTE = {
    "safe": "#0072B2",
    "gray": "#8A8A8A",
    "orange": "#D55E00",
    "green": "#009E73",
    "interval": "#A6CEE3",
    "black": "#111111",
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: mpl.figure.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.svg")
    fig.savefig(out_dir / f"{stem}.png", dpi=600)
    plt.close(fig)


def panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def add_season_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    target = pd.to_datetime(out["target_week_start"])
    iso = target.dt.isocalendar()
    start_year = np.where(iso.week.to_numpy() >= 27, iso.year.to_numpy(), iso.year.to_numpy() - 1)
    out["season_start_year"] = start_year.astype(int)
    out["test_season"] = [f"{y}/{y + 1}" for y in out["season_start_year"]]
    return out


def season_bounds(start_year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (
        pd.Timestamp.fromisocalendar(int(start_year), 27, 1),
        pd.Timestamp.fromisocalendar(int(start_year) + 1, 26, 1),
    )


def build_fold_calendar(pred: pd.DataFrame) -> pd.DataFrame:
    z = add_season_columns(pred)
    rows = []
    for sy, g in z.groupby("season_start_year"):
        s0, s1 = season_bounds(int(sy))
        source = (
            g[["evaluation", "fold"]]
            .drop_duplicates()
            .sort_values(["evaluation", "fold"])
            .assign(source=lambda x: x["evaluation"] + ":fold" + x["fold"].astype(str))
        )
        rows.append(
            {
                "test_season": f"{int(sy)}/{int(sy) + 1}",
                "season_start": s0.date().isoformat(),
                "season_end": s1.date().isoformat(),
                "available_target_start": pd.to_datetime(g["target_week_start"]).min().date().isoformat(),
                "available_target_end": pd.to_datetime(g["target_week_start"]).max().date().isoformat(),
                "available_weeks": int(pd.to_datetime(g["target_week_start"]).nunique()),
                "source_prediction_folds": ";".join(source["source"].tolist()),
                "complete_available_season": bool(
                    pd.to_datetime(g["target_week_start"]).min() <= s0
                    and pd.to_datetime(g["target_week_start"]).max() >= s1
                ),
                "purge_gap_weeks": 4,
                "note": "Season assignment is target-week based; underlying expert predictions are frozen rolling-origin outputs.",
            }
        )
    return pd.DataFrame(rows).sort_values("test_season")


def metric_summary(g: pd.DataFrame) -> pd.Series:
    return v7.summary(g)


METRIC_COLS = [
    "WIS",
    "MAE",
    "RMSE",
    "Pearson",
    "Spearman",
    "coverage_50",
    "coverage_80",
    "coverage_95",
    "width_50",
    "width_80",
    "width_95",
]


def country_balanced_metrics(pred: pd.DataFrame, group_cols: list[str], season_balanced: bool = False) -> pd.DataFrame:
    inner = group_cols + ["country"]
    if season_balanced and "test_season" not in inner:
        inner = group_cols + ["test_season", "country"]
    by_country = pred.groupby(inner, dropna=False).apply(metric_summary, include_groups=False).reset_index()
    out = by_country.groupby(group_cols, dropna=False)[METRIC_COLS].mean().reset_index()
    counts = by_country.groupby(group_cols, dropna=False).agg(
        n_total=("n", "sum"),
        n_countries=("country", "nunique"),
    ).reset_index()
    if "test_season" not in group_cols:
        counts = counts.merge(
            by_country.groupby(group_cols, dropna=False).agg(n_test_seasons=("test_season", "nunique")).reset_index(),
            on=group_cols,
            how="left",
        )
    return out.merge(counts, on=group_cols, how="left")


def add_relative_to_reference(metrics: pd.DataFrame, keys: list[str], ref_model: str, suffix: str) -> pd.DataFrame:
    out = metrics.copy()
    ref = out[out["model"] == ref_model][keys + ["WIS"]].rename(columns={"WIS": f"{suffix}_WIS"})
    out = out.merge(ref, on=keys, how="left")
    out[f"relative_WIS_vs_{suffix}"] = out["WIS"] / out[f"{suffix}_WIS"]
    out[f"WIS_reduction_vs_{suffix}_pct"] = 100 * (1 - out[f"relative_WIS_vs_{suffix}"])
    return out


def block_boot_pair(pred: pd.DataFrame, model_a: str, model_b: str, horizon: int, delay: int, block: int, reps: int, seed: int, country: str | None = None) -> dict:
    z = pred[(pred["horizon"] == horizon) & (pred["official_delay_weeks"] == delay)]
    if country is not None:
        z = z[z["country"] == country]
    keys = ["country", "origin_week_start", "target_week_start"]
    a = z[z["model"] == model_a][keys + ["WIS"]].rename(columns={"WIS": "a"})
    b = z[z["model"] == model_b][keys + ["WIS"]].rename(columns={"WIS": "b"})
    x = a.merge(b, on=keys, how="inner")
    if x.empty:
        return {"mean_diff_A_minus_B": np.nan, "ci_low": np.nan, "ci_high": np.nan, "prob_A_better": np.nan, "n_pairs": 0}
    obs = float(x.groupby("country", dropna=False).apply(lambda q: (q["a"] - q["b"]).mean(), include_groups=False).mean())
    rng = np.random.default_rng(seed)
    boot = []
    for _, g in x.groupby("country", dropna=False):
        d = (g.sort_values("origin_week_start")["a"] - g.sort_values("origin_week_start")["b"]).to_numpy(float)
        n = len(d)
        n_blocks = int(math.ceil(n / block))
        max_start = max(1, n - block + 1)
        starts = rng.integers(0, max_start, size=(reps, n_blocks))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]
        idx = np.minimum(idx, n - 1)
        boot.append(d[idx].mean(axis=1))
    bs = np.mean(np.column_stack(boot), axis=1)
    return {
        "mean_diff_A_minus_B": obs,
        "ci_low": float(np.quantile(bs, 0.025)),
        "ci_high": float(np.quantile(bs, 0.975)),
        "prob_A_better": float(np.mean(bs < 0)),
        "n_pairs": int(len(x)),
    }


def build_bootstrap(pred: pd.DataFrame, metrics: pd.DataFrame, reps: int) -> pd.DataFrame:
    rows = []
    comparisons = [
        (SAFE, PERSIST, "persistence", None, [0]),
        (SAFE, BASE, "surveillance", None, [0, 1, 2]),
        (SAFE, FULL, "full_digital", None, [0]),
        (SAFE, GB, "gb_expert", None, [0]),
        (RIDGE_CN, RIDGE_NO_DIGITAL, "no_digital_ridge", "CHN", [2]),
    ]
    for model_a, model_b, denom_label, country, delays in comparisons:
        for delay in delays:
            for h in [1, 2, 3, 4]:
                for block in [4, 8, 13]:
                    res = block_boot_pair(pred, model_a, model_b, h, delay, block, reps, 10000 + h * 100 + delay * 10 + block, country)
                    if country is not None:
                        denom_vals = pred[
                            (pred["official_delay_weeks"] == delay)
                            & (pred["model"] == model_b)
                            & (pred["horizon"] == h)
                            & (pred["country"] == country)
                        ]["WIS"]
                        denom = float(denom_vals.mean()) if len(denom_vals) else np.nan
                    else:
                        denom_q = (metrics["official_delay_weeks"] == delay) & (metrics["model"] == model_b) & (metrics["horizon"] == h)
                        denom_vals = metrics.loc[denom_q, "WIS"]
                        denom = float(denom_vals.iloc[0]) if len(denom_vals) else np.nan
                    rows.append(
                        {
                            "model_A": model_a,
                            "model_B": model_b,
                            "comparison": f"{MODEL_SHORT.get(model_a, model_a)} vs {MODEL_SHORT.get(model_b, model_b)}",
                            "country_filter": country or "all",
                            "official_delay_weeks": delay,
                            "horizon": h,
                            "block_weeks": block,
                            **res,
                            "denominator_WIS": denom,
                            "WIS_reduction_pct": -100 * res["mean_diff_A_minus_B"] / denom if np.isfinite(denom) else np.nan,
                            "ci_low_reduction_pct": -100 * res["ci_high"] / denom if np.isfinite(denom) else np.nan,
                            "ci_high_reduction_pct": -100 * res["ci_low"] / denom if np.isfinite(denom) else np.nan,
                            "reduction_denominator": denom_label,
                        }
                    )
    return pd.DataFrame(rows)


def event_warning_metrics_rolling(pred: pd.DataFrame, root: Path) -> pd.DataFrame:
    threshold_path = root / "results" / "tables" / "v7_event_thresholds.csv"
    if threshold_path.exists():
        thresholds = pd.read_csv(threshold_path)
        threshold_source = threshold_path.relative_to(root).as_posix()
    else:
        thresholds = pd.DataFrame(
            [{"country": c, "horizon": h, "probability_threshold": 0.50} for c in ["CHN", "JPN", "USA"] for h in [1, 2, 3, 4]]
        )
        threshold_source = "fallback_0.50"
    cut_lookup = thresholds.set_index(["country", "horizon"])["probability_threshold"].to_dict()
    work = pred[
        (pred["official_delay_weeks"] == 0)
        & (pred["region"] == "national")
        & (pred["model"] == SAFE)
    ].copy()
    if work.empty:
        return pd.DataFrame()
    work["origin_week_start"] = pd.to_datetime(work["origin_week_start"])
    work["target_week_start"] = pd.to_datetime(work["target_week_start"])

    rows = []

    def add_row(scope: str, season: str, g: pd.DataFrame) -> None:
        country = str(g["country"].iloc[0])
        horizon = int(g["horizon"].iloc[0])
        cut = float(cut_lookup.get((country, horizon), 0.50))
        stats = v7.event_one(g, cut)
        rows.append(
            {
                "scope": scope,
                "test_season": season,
                "country": country,
                "horizon": horizon,
                "model": SAFE,
                "official_delay_weeks": 0,
                "probability_threshold": cut,
                "threshold_source": threshold_source,
                "available_target_start": g["target_week_start"].min().date().isoformat(),
                "available_target_end": g["target_week_start"].max().date().isoformat(),
                "available_weeks": int(g["target_week_start"].nunique()),
                **stats,
            }
        )

    for (_, _, season), g in work.groupby(["country", "horizon", "test_season"], sort=False):
        add_row("test_season", str(season), g)
    for (_, _), g in work.groupby(["country", "horizon"], sort=False):
        add_row("all_available_seasons", "ALL_AVAILABLE_SEASONS", g)
    return pd.DataFrame(rows)


def generate_all_predictions(store, params):
    national = {"CHN": ["national"], "JPN": ["national"], "USA": ["national"]}
    regional = {"CHN": ["northern_provinces", "southern_provinces"], "USA": [f"hhs{i}" for i in range(1, 11)]}
    eval_folds = (
        store.df[["evaluation", "fold"]]
        .drop_duplicates()
        .sort_values(["evaluation", "fold"])
        .itertuples(index=False, name=None)
    )
    eval_folds = list(eval_folds)

    weights = []
    safe = []
    for ev, fold in eval_folds:
        delays = sorted(
            store.df[(store.df["evaluation"] == ev) & (store.df["fold"] == int(fold))]["official_delay_weeks"].dropna().unique()
        )
        for delay in delays:
            safe.append(v7.generate(store, params, ev, int(fold), delay, national, weights=weights))
    safe = pd.concat(safe, ignore_index=True)
    comps = [BASE, FULL, PERSIST, "seasonal_naive", GB, RIDGE_CN, RIDGE_NO_DIGITAL, "ridge_google_wiki", "ridge_social_raw", "ridge_global_open"]
    national_all = pd.concat([safe, store.model_rows(safe, comps)], ignore_index=True)

    ablation_frames = [safe[safe["official_delay_weeks"] == 0]]
    for ev, fold in eval_folds:
        for variant in [
            "base_only",
            "no_full_lstm_expert",
            "no_gradient_boosting",
            "no_source_specific_ridge",
            "no_china_local_search",
            "no_google_wiki",
            "no_social",
        ]:
            ablation_frames.append(v7.generate(store, params, ev, int(fold), 0, national, variant=variant, weights=weights))
        ablation_frames.append(v7.generate(store, params, ev, int(fold), 0, national, update=False, static_blend=True, weights=weights))
    ablation = pd.concat(ablation_frames, ignore_index=True)

    regional_weights = []
    regional_safe = []
    for ev, fold in eval_folds:
        regional_safe.append(v7.generate(store, params, ev, int(fold), 0, regional, weights=regional_weights))
    regional_safe = pd.concat(regional_safe, ignore_index=True)
    regional_all = pd.concat([regional_safe, store.model_rows(regional_safe, [BASE, PERSIST])], ignore_index=True)

    return national_all, ablation, pd.concat(weights, ignore_index=True), regional_all, pd.concat(regional_weights, ignore_index=True)


def figure1_calendar(calendar: pd.DataFrame, figs: Path) -> None:
    fig, ax = plt.subplots(figsize=(180 * MM, 70 * MM), constrained_layout=True)
    ax.set_axis_off()
    panel_label(ax, "a", x=0.0, y=0.89)
    ax.set_xlim(pd.Timestamp("2015-01-01").toordinal(), pd.Timestamp("2026-07-01").toordinal())
    ax.set_ylim(-0.7, len(calendar) + 0.8)
    ax.hlines(len(calendar) + 0.25, pd.Timestamp("2015-01-01").toordinal(), pd.Timestamp("2026-07-01").toordinal(), color="#555555", lw=0.8)
    for year in range(2015, 2027):
        x = pd.Timestamp(f"{year}-01-01").toordinal()
        ax.vlines(x, len(calendar) + 0.08, len(calendar) + 0.42, color="#777777", lw=0.55)
        ax.text(x, len(calendar) + 0.52, str(year), ha="center", va="bottom", fontsize=6)
    for i, row in calendar.reset_index(drop=True).iterrows():
        y = len(calendar) - i - 1
        start = pd.Timestamp(row["available_target_start"]).toordinal()
        end = pd.Timestamp(row["available_target_end"]).toordinal()
        color = PALETTE["safe"] if row["complete_available_season"] else "#9E9E9E"
        ax.hlines(y, start, end, color=color, lw=5.0, alpha=0.9)
        ax.text(start - 20, y, row["test_season"], ha="right", va="center", fontsize=6.5)
        ax.text(end + 18, y, "complete" if row["complete_available_season"] else "partial", ha="left", va="center", fontsize=5.8, color="#555555")
    ax.text(pd.Timestamp("2015-01-01").toordinal(), -0.45, "Target-week seasons are ISO week 27 to week 26; 4-week purge is retained in source rolling-origin experiments.", fontsize=6.1, ha="left", color="#444444")
    save_figure(fig, figs, "Figure_1_v8_rolling_season_calendar")


def figure3_performance(metrics: pd.DataFrame, boot: pd.DataFrame, season_metrics: pd.DataFrame, figs: Path) -> None:
    order = [GB, FULL, BASE, SAFE]
    fig, axes = plt.subplots(1, 4, figsize=(180 * MM, 78 * MM), sharey=True, constrained_layout=True)
    boot8 = boot[(boot["official_delay_weeks"] == 0) & (boot["block_weeks"] == 8) & (boot["model_B"] == PERSIST)]
    points = season_metrics[(season_metrics["official_delay_weeks"] == 0) & (season_metrics["model"].isin(order + [PERSIST]))]
    for idx, h in enumerate([1, 2, 3, 4]):
        ax = axes[idx]
        panel_label(ax, chr(ord("a") + idx))
        ax.axvline(0, color="#777777", lw=0.7)
        for yi, model in enumerate(order):
            row = boot8[(boot8["model_A"] == model) & (boot8["horizon"] == h)]
            if model == SAFE:
                row = boot8[(boot8["model_A"] == SAFE) & (boot8["model_B"] == PERSIST) & (boot8["horizon"] == h)]
            if row.empty and model != SAFE:
                # Only SAFE-vs-comparator bootstrap is needed for inference; other model points use season summaries.
                m = metrics[(metrics["official_delay_weeks"] == 0) & (metrics["model"] == model) & (metrics["horizon"] == h)]
                x = float(m["WIS_reduction_vs_persistence_pct"].iloc[0])
                lo, hi = np.nan, np.nan
            else:
                r = row.iloc[0]
                x, lo, hi = float(r["WIS_reduction_pct"]), float(r["ci_low_reduction_pct"]), float(r["ci_high_reduction_pct"])
            if model != SAFE:
                m = metrics[(metrics["official_delay_weeks"] == 0) & (metrics["model"] == model) & (metrics["horizon"] == h)]
                x = float(m["WIS_reduction_vs_persistence_pct"].iloc[0])
            season_ref = points[(points["model"] == PERSIST) & (points["horizon"] == h)][["test_season", "WIS"]].rename(columns={"WIS": "ref_WIS"})
            season_model = points[(points["model"] == model) & (points["horizon"] == h)][["test_season", "WIS"]]
            sp = season_model.merge(season_ref, on="test_season", how="inner")
            sx = 100 * (1 - sp["WIS"] / sp["ref_WIS"])
            ax.scatter(sx, np.repeat(yi, len(sx)) + np.linspace(-0.12, 0.12, len(sx)), s=6, color="#B0B0B0", alpha=0.55, zorder=1)
            color = PALETTE["safe"] if model == SAFE else PALETTE["gray"]
            face = color if model == SAFE else "#FFFFFF"
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(x, yi, xerr=np.array([[x - lo], [hi - x]]), fmt="o", color=color, ecolor=color, markerfacecolor=face, capsize=2, markersize=4, zorder=3)
            else:
                ax.plot(x, yi, "o", color=color, markerfacecolor=face, markersize=3.6, zorder=3)
        ax.set_xlim(-45, 58)
        ax.set_title(f"h = {h}", pad=2)
        ax.set_xlabel("WIS reduction vs persistence, %")
        ax.grid(axis="x", color="#EAEAEA", lw=0.5)
        if idx == 0:
            ax.set_yticks(range(len(order)), [MODEL_SHORT[m] for m in order])
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
    save_figure(fig, figs, "Figure_3_v8_rolling_season_performance")


def copy_latest_period_forecast_figure(root: Path, figs: Path) -> None:
    src_dir = root / "results" / "figures_npj"
    for ext in ["pdf", "svg", "png"]:
        src = src_dir / f"Figure_2_external_probabilistic_forecasts.{ext}"
        dst = figs / f"Figure_2_v8_latest_period_probabilistic_forecasts.{ext}"
        if src.exists():
            shutil.copy2(src, dst)


def figure4_delay(metrics: pd.DataFrame, boot: pd.DataFrame, weights: pd.DataFrame, figs: Path) -> None:
    fig = plt.figure(figsize=(180 * MM, 112 * MM), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    s = boot[(boot["model_A"] == SAFE) & (boot["model_B"] == BASE) & (boot["official_delay_weeks"] == 0) & (boot["block_weeks"] == 8)].sort_values("horizon")
    x = s["WIS_reduction_pct"].to_numpy(float)
    ax.axvline(0, color="#777777", lw=0.7)
    ax.errorbar(x, np.arange(len(s)), xerr=np.vstack([x - s["ci_low_reduction_pct"].to_numpy(float), s["ci_high_reduction_pct"].to_numpy(float) - x]), fmt="o", color=PALETTE["safe"], ecolor=PALETTE["safe"], capsize=2)
    ax.set_yticks(np.arange(len(s)), [f"h = {h}" for h in s["horizon"]])
    ax.invert_yaxis()
    ax.set_xlabel("WIS reduction vs Surv-LSTM, %")
    ax.set_title("No delay", pad=2)
    ax.grid(axis="x", color="#EAEAEA", lw=0.5)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b")
    m = metrics[(metrics["model"] == SAFE)].pivot(index="official_delay_weeks", columns="horizon", values="WIS_reduction_vs_surveillance_pct").reindex([0, 1, 2])
    im = ax.imshow(m.to_numpy(float), aspect="auto", cmap=LinearSegmentedColormap.from_list("white_blue", ["#F7F7F7", PALETTE["safe"]]), vmin=0, vmax=max(10, float(np.nanmax(m.to_numpy(float)))))
    ax.set_xticks(range(4), [1, 2, 3, 4])
    ax.set_yticks(range(3), ["0-week", "1-week", "2-week"])
    ax.set_xlabel("Forecast horizon, weeks")
    ax.set_ylabel("Official-report delay")
    ax.set_title("Reporting-delay gain", pad=2)
    cbar = fig.colorbar(im, ax=ax, shrink=0.83, pad=0.015)
    cbar.set_label("WIS reduction, %")

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "c")
    c = boot[(boot["model_A"] == RIDGE_CN) & (boot["model_B"] == RIDGE_NO_DIGITAL) & (boot["official_delay_weeks"] == 2) & (boot["block_weeks"] == 8)].sort_values("horizon")
    x = c["WIS_reduction_pct"].to_numpy(float)
    ax.axvline(0, color="#777777", lw=0.7)
    ax.errorbar(x, np.arange(len(c)), xerr=np.vstack([x - c["ci_low_reduction_pct"].to_numpy(float), c["ci_high_reduction_pct"].to_numpy(float) - x]), fmt="s", color=PALETTE["orange"], ecolor=PALETTE["orange"], capsize=2)
    ax.set_yticks(np.arange(len(c)), [f"h = {h}" for h in c["horizon"]])
    ax.invert_yaxis()
    ax.set_xlabel("WIS reduction vs no-digital Ridge, %")
    ax.set_title("China local search, 2-week delay", pad=2)
    ax.grid(axis="x", color="#EAEAEA", lw=0.5)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "d")
    w = weights[(weights["official_delay_weeks"] == 0) & (weights["region"] == "national") & (weights["variant"] == "full")]
    base_weight = w[w["expert"] == BASE].groupby(["country", "horizon"])["weight"].mean().unstack("horizon").reindex(["CHN", "USA", "JPN"])
    aux = 1 - base_weight
    im = ax.imshow(100 * aux.to_numpy(float), aspect="auto", cmap=LinearSegmentedColormap.from_list("white_blue", ["#F7F7F7", PALETTE["safe"]]), vmin=0, vmax=50)
    ax.set_xticks(range(4), [1, 2, 3, 4])
    ax.set_yticks(range(3), ["China", "United States", "Japan"])
    ax.set_xlabel("Forecast horizon, weeks")
    ax.set_title("Mean digital-expert activation", pad=2)
    cbar = fig.colorbar(im, ax=ax, shrink=0.83, pad=0.015)
    cbar.set_label("% total SAFE weight")
    save_figure(fig, figs, "Figure_4_v8_conditional_digital_utility")


def variant_label(v: str) -> str:
    return {
        "bioepi_safe_lstm_common_scale__base_only": "Backbone\nonly",
        "bioepi_safe_lstm_static_blend": "Equal\nblend",
        "bioepi_safe_lstm_common_scale__no_gradient_boosting": "No GB\nexpert",
        "bioepi_safe_lstm_common_scale__no_source_specific_ridge": "No Ridge\nexperts",
        "bioepi_safe_lstm_common_scale__no_full_lstm_expert": "No full-digital\nexpert",
    }.get(v, v.replace("bioepi_safe_lstm_common_scale__", "").replace("_", " "))


def figure5_regional(regional: pd.DataFrame, ablation_metrics: pd.DataFrame, figs: Path) -> None:
    z = regional[(regional["official_delay_weeks"] == 0)]
    safe = z[z["model"] == SAFE][["country", "region", "horizon", "WIS"]].rename(columns={"WIS": "safe_WIS"})
    ref = z[z["model"] == PERSIST][["country", "region", "horizon", "WIS"]].rename(columns={"WIS": "persistence_WIS"})
    rel = safe.merge(ref, on=["country", "region", "horizon"], how="inner")
    rel["relative_WIS"] = rel["safe_WIS"] / rel["persistence_WIS"]
    vals = rel["relative_WIS"].to_numpy(float)
    cmap = LinearSegmentedColormap.from_list("blue_white_orange", ["#0072B2", "#F7F7F7", "#D55E00"])
    norm = TwoSlopeNorm(vmin=min(0.55, float(np.nanmin(vals))), vcenter=1.0, vmax=max(1.15, float(np.nanmax(vals))))
    fig = plt.figure(figsize=(180 * MM, 96 * MM), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.65, 0.72, 1.55])
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    usa = rel[rel["country"] == "USA"].pivot(index="region", columns="horizon", values="relative_WIS").reindex([f"hhs{i}" for i in range(1, 11)])
    im = ax.imshow(usa.to_numpy(float), aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(4), [1, 2, 3, 4])
    ax.set_yticks(range(len(usa.index)), [x.upper() for x in usa.index])
    ax.set_xlabel("Forecast horizon, weeks")
    ax.set_title("United States HHS", pad=2)
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b")
    chn = rel[rel["country"] == "CHN"].pivot(index="region", columns="horizon", values="relative_WIS").reindex(["northern_provinces", "southern_provinces"])
    im2 = ax.imshow(chn.to_numpy(float), aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(4), [1, 2, 3, 4])
    ax.set_yticks(range(len(chn.index)), ["North", "South"])
    ax.set_xlabel("Forecast horizon, weeks")
    ax.set_title("China", pad=2)
    cbar = fig.colorbar(im2, ax=fig.axes[:2], shrink=0.83, pad=0.013)
    cbar.set_label("Relative WIS vs persistence")

    ax = fig.add_subplot(gs[0, 2])
    panel_label(ax, "c")
    keep = [
        "bioepi_safe_lstm_common_scale__base_only",
        "bioepi_safe_lstm_static_blend",
        "bioepi_safe_lstm_common_scale__no_gradient_boosting",
        "bioepi_safe_lstm_common_scale__no_source_specific_ridge",
        "bioepi_safe_lstm_common_scale__no_full_lstm_expert",
    ]
    comp = ablation_metrics[(ablation_metrics["horizon"] == 2) & (ablation_metrics["model"].isin(keep))].copy()
    full = float(ablation_metrics[(ablation_metrics["horizon"] == 2) & (ablation_metrics["model"] == SAFE)]["WIS"].iloc[0])
    comp["delta_WIS_vs_SAFE"] = comp["WIS"] - full
    comp["model"] = pd.Categorical(comp["model"], categories=keep, ordered=True)
    comp = comp.sort_values("model")
    y = np.arange(len(comp))
    colors = [PALETTE["safe"] if v > 0 else PALETTE["orange"] for v in comp["delta_WIS_vs_SAFE"]]
    ax.axvline(0, color="#777777", lw=0.7)
    ax.scatter(comp["delta_WIS_vs_SAFE"], y, color=colors, s=18)
    ax.set_yticks(y, [variant_label(str(v)) for v in comp["model"]])
    ax.invert_yaxis()
    ax.set_xlabel("Delta WIS vs SAFE-LSTM")
    ax.set_title("Key components, h = 2", pad=2)
    ax.grid(axis="x", color="#EAEAEA", lw=0.5)
    save_figure(fig, figs, "Figure_5_v8_regional_components")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    args = ap.parse_args()
    root = args.root.resolve()
    inp = args.input or (root / "data" / "expert_predictions_v6_common_scale_test.pkl.gz")
    tables = root / "results" / "tables" / "v8_season_level"
    figs = root / "results" / "figures_v8"
    meta = root / "results" / "metadata"
    for d in [tables, figs, meta]:
        d.mkdir(parents=True, exist_ok=True)
    set_style()

    store = v7.Store(pd.read_pickle(inp))
    selected = meta / "v7_common_scale_selected_params.json"
    if selected.exists():
        params = v7.Params(**json.loads(selected.read_text(encoding="utf-8")))
        tuning_note = "Loaded fixed v7 SAFE parameters; no tuning on v8 season summaries."
    else:
        params, _ = v7.tune_fold2(store)
        tuning_note = "Selected SAFE parameters using the original fold-2 v7 safety grid."

    national_all, ablation, weights, regional_all, regional_weights = generate_all_predictions(store, params)
    national_all = add_season_columns(national_all)
    ablation = add_season_columns(ablation)
    weights = add_season_columns(weights)
    regional_all = add_season_columns(regional_all)
    regional_weights = add_season_columns(regional_weights)

    calendar = build_fold_calendar(national_all)
    calendar.to_csv(tables / "fold_calendar.csv", index=False, encoding="utf-8-sig")
    keep_cols = [
        "test_season",
        "season_start_year",
        "evaluation",
        "fold",
        "official_delay_weeks",
        "country",
        "region",
        "origin_week_start",
        "target_week_start",
        "horizon",
        "model",
        "period",
        "y_true",
        "target_raw",
        "event_threshold_z_train_q80",
        *v7.QCOLS,
        "WIS",
        "abs_error",
        "covered_50",
        "covered_80",
        "covered_95",
    ]
    national_all[keep_cols].to_csv(tables / "predictions_rolling_season_cv.csv.gz", index=False, compression="gzip")

    metrics = country_balanced_metrics(national_all, ["official_delay_weeks", "model", "horizon"], season_balanced=True)
    metrics = add_relative_to_reference(metrics, ["official_delay_weeks", "horizon"], PERSIST, "persistence")
    metrics = add_relative_to_reference(metrics, ["official_delay_weeks", "horizon"], BASE, "surveillance")
    metrics.to_csv(tables / "metrics_rolling_cv_country_season_balanced.csv", index=False, encoding="utf-8-sig")

    by_season = country_balanced_metrics(national_all, ["test_season", "season_start_year", "official_delay_weeks", "model", "horizon"])
    by_season = add_relative_to_reference(by_season, ["test_season", "season_start_year", "official_delay_weeks", "horizon"], PERSIST, "persistence")
    by_season.to_csv(tables / "metrics_by_test_season.csv", index=False, encoding="utf-8-sig")

    by_phase = country_balanced_metrics(national_all, ["period", "official_delay_weeks", "model", "horizon"], season_balanced=True)
    by_phase = add_relative_to_reference(by_phase, ["period", "official_delay_weeks", "horizon"], PERSIST, "persistence")
    by_phase.to_csv(tables / "metrics_by_phase.csv", index=False, encoding="utf-8-sig")

    events = event_warning_metrics_rolling(national_all, root)
    events.to_csv(tables / "event_warning_metrics_rolling_cv.csv", index=False, encoding="utf-8-sig")

    delay = metrics[metrics["model"].isin([SAFE, BASE, FULL, PERSIST, RIDGE_CN, RIDGE_NO_DIGITAL])].copy()
    delay.to_csv(tables / "delay_scenario_metrics_all_folds.csv", index=False, encoding="utf-8-sig")

    boot = build_bootstrap(national_all, metrics, args.bootstrap_reps)
    boot.to_csv(tables / "bootstrap_wis_delta_by_model.csv", index=False, encoding="utf-8-sig")

    ab_metrics = country_balanced_metrics(ablation, ["model", "horizon"], season_balanced=True)
    ab_metrics.to_csv(tables / "ablation_metrics_rolling_season.csv", index=False, encoding="utf-8-sig")

    reg_metrics = regional_all.groupby(["country", "region", "official_delay_weeks", "model", "horizon"], dropna=False).apply(metric_summary, include_groups=False).reset_index()
    reg_metrics.to_csv(tables / "regional_robustness_metrics.csv", index=False, encoding="utf-8-sig")

    latest = national_all[(national_all["evaluation"] == "temporal_external_2025_2026") & (national_all["official_delay_weeks"] == 0)].copy()
    latest.to_csv(tables / "latest_period_temporal_evaluation_predictions.csv.gz", index=False, compression="gzip")
    weights.to_csv(tables / "source_expert_weights_rolling_season.csv.gz", index=False, compression="gzip")

    figure1_calendar(calendar, figs)
    copy_latest_period_forecast_figure(root, figs)
    figure3_performance(metrics, boot, by_season, figs)
    figure4_delay(delay, boot, weights, figs)
    figure5_regional(reg_metrics, ab_metrics, figs)

    manifest = {
        "version": "v8_season_level_evaluation_layer",
        "created_from": str(inp),
        "evaluation_note": "Season-level summaries are target-week based and use frozen fold-specific expert predictions from v6/v7. This is not a complete re-training of one outer fold per influenza season.",
        "season_definition": "ISO week 27 to ISO week 26 of the following ISO year",
        "primary_metric": "country-season-balanced macro WIS",
        "latest_period_role": "latest-period temporal evaluation, not the sole main result",
        "delay_availability_note": "Delay 0 is available for rolling folds and latest-period external evaluation. Delay 1 and 2 are available only in the frozen latest-period external predictions.",
        "tuning_note": tuning_note,
        "safe_params": params.__dict__,
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_blocks": [4, 8, 13],
        "tables": sorted(p.name for p in tables.glob("*")),
        "figures": sorted(p.name for p in figs.glob("*.pdf")),
    }
    (meta / "manifest_v8_season_level_evaluation.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
