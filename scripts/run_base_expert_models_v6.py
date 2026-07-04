#!/usr/bin/env python
# BioEpi-SAFE-LSTM reproducibility code
# Maintainer: Jianyi Zhang

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HORIZONS = [1, 2, 3, 4]
QUANTILES = np.array([0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975], dtype=float)
QCOLS = [f"q{q:g}" for q in QUANTILES]
QINDEX = {f"q{q:g}": i for i, q in enumerate(QUANTILES)}
LAG_WEEKS = [1, 2, 3, 4, 8, 13, 26, 52]
ROLL_WINDOWS = [4, 8, 13, 26]
SEEDS_FAST = [101]
SEEDS_FULL = [101, 202, 303, 404, 505]
TARGET_EVENT_QUANTILE = 0.80
FULL_LSTM_MODEL = "bioepi_sgq_lstm_full_raw"
NO_DIGITAL_LSTM_MODEL = "bioepi_sgq_lstm_no_digital"
SURVEILLANCE_HISTORY_MODEL = "bioepi_sgq_lstm_surveillance_history_seasonality_only"
DIGITAL_DISEASE_CORE_MODEL = "bioepi_sgq_lstm_digital_disease_core_plus_surveillance"
NO_REGION_INDICATORS_MODEL = "bioepi_sgq_lstm_no_region_indicators"
ROLLING_EVALUATION = "rolling_window_cv"
CORE_EVENT_MODELS = [FULL_LSTM_MODEL, "ridge_raw_cn_search", "ridge_full_raw", "persistence"]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def qcol(q: float) -> str:
    return f"q{q:g}"


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    return (upper - lower) + (2.0 / alpha) * (lower - y) * (y < lower) + (2.0 / alpha) * (y - upper) * (y > upper)


def wis_from_matrix(y: np.ndarray, qmat: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    qmat = np.asarray(qmat, dtype=float)
    score = 0.5 * np.abs(y - qmat[:, QINDEX["q0.5"]])
    for alpha in [0.50, 0.20, 0.05]:
        score += (alpha / 2.0) * interval_score(
            y,
            qmat[:, QINDEX[qcol(alpha / 2.0)]],
            qmat[:, QINDEX[qcol(1.0 - alpha / 2.0)]],
            alpha,
        )
    return score / 3.5


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["y_pred"] = out["q0.5"]
    out["y_pred_raw"] = out["q0.5_raw"]
    out["abs_error"] = (out["y_true"] - out["y_pred"]).abs()
    out["sq_error"] = (out["y_true"] - out["y_pred"]) ** 2
    out["WIS"] = wis_from_matrix(out["y_true"].to_numpy(dtype=float), out[QCOLS].to_numpy(dtype=float))
    for label, alpha in [("50", 0.50), ("80", 0.20), ("95", 0.05)]:
        out[f"covered_{label}"] = (
            (out["y_true"] >= out[qcol(alpha / 2.0)]) & (out["y_true"] <= out[qcol(1.0 - alpha / 2.0)])
        ).astype(int)
    return out


def corr(y: np.ndarray, p: np.ndarray, kind: str) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() < 3 or np.nanstd(y[mask]) == 0 or np.nanstd(p[mask]) == 0:
        return np.nan
    if kind == "pearson":
        return float(pearsonr(y[mask], p[mask])[0])
    return float(spearmanr(y[mask], p[mask]).correlation)


def summarise(g: pd.DataFrame) -> pd.Series:
    y = g["y_true"].to_numpy(dtype=float)
    p = g["y_pred"].to_numpy(dtype=float)
    return pd.Series(
        {
            "n": int(len(g)),
            "MAE": float(np.nanmean(np.abs(y - p))),
            "RMSE": float(math.sqrt(np.nanmean((y - p) ** 2))),
            "Pearson": corr(y, p, "pearson"),
            "Spearman": corr(y, p, "spearman"),
            "WIS": float(np.nanmean(g["WIS"])),
            "coverage_50": float(np.nanmean(g["covered_50"])),
            "coverage_80": float(np.nanmean(g["covered_80"])),
            "coverage_95": float(np.nanmean(g["covered_95"])),
        }
    )


def make_period(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return pd.Series(
        np.select(
            [
                d < pd.Timestamp("2020-01-01"),
                (d >= pd.Timestamp("2020-01-01")) & (d < pd.Timestamp("2022-01-01")),
                (d >= pd.Timestamp("2022-01-01")) & (d < pd.Timestamp("2023-01-01")),
                d >= pd.Timestamp("2023-01-01"),
            ],
            ["pre_covid_2015_2019", "npi_2020_2021", "reopening_2022", "post_covid_2023_2026"],
            default="unknown",
        ),
        index=dates.index,
    )


@dataclass(frozen=True)
class SplitSpec:
    evaluation: str
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def rolling_splits(target_weeks: list[pd.Timestamp]) -> list[SplitSpec]:
    weeks = sorted(pd.to_datetime(pd.Series(target_weeks).dropna().unique()))
    out: list[SplitSpec] = []
    i = 0
    fold = 1
    train_w, val_w, test_w, step_w = 156, 52, 104, 104
    while i + train_w + val_w + test_w <= len(weeks):
        out.append(
            SplitSpec(
                ROLLING_EVALUATION,
                fold,
                weeks[i],
                weeks[i + train_w - 1],
                weeks[i + train_w],
                weeks[i + train_w + val_w - 1],
                weeks[i + train_w + val_w],
                weeks[i + train_w + val_w + test_w - 1],
            )
        )
        fold += 1
        i += step_w
    return out


def external_split(target_weeks: list[pd.Timestamp]) -> SplitSpec:
    weeks = sorted(pd.to_datetime(pd.Series(target_weeks).dropna().unique()))
    return SplitSpec(
        "temporal_external_2025_2026",
        1,
        pd.Timestamp("2015-01-05"),
        pd.Timestamp("2023-12-25"),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-12-23"),
        pd.Timestamp("2024-12-30"),
        weeks[-1],
    )


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "processed": root / "data" / "processed",
        "tables": root / "results" / "tables",
        "figures": root / "results" / "figures",
        "qc": root / "results" / "qc",
        "metadata": root / "results" / "metadata",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def make_origin_wide(design: pd.DataFrame) -> pd.DataFrame:
    design = design.copy()
    for c in ["week_start", "week_end", "target_week_start"]:
        design[c] = pd.to_datetime(design[c])
    base_drop = {"horizon", "target_week_start", "y", "period"}
    base_cols = [c for c in design.columns if c not in base_drop]
    base = (
        design[design["horizon"] == 1][base_cols]
        .sort_values(["country", "region", "week_start"])
        .drop_duplicates(["country", "region", "week_start"])
        .reset_index(drop=True)
    )
    keys = ["country", "region", "week_start"]
    y_wide = design.set_index(keys + ["horizon"])["y"].unstack("horizon")
    target_week_wide = design.set_index(keys + ["horizon"])["target_week_start"].unstack("horizon")
    period_wide = design.set_index(keys + ["horizon"])["period"].unstack("horizon")
    for h in HORIZONS:
        y_wide = y_wide.rename(columns={h: f"y_h{h}"})
        target_week_wide = target_week_wide.rename(columns={h: f"target_week_h{h}"})
        period_wide = period_wide.rename(columns={h: f"period_h{h}"})
    wide = base.merge(y_wide.reset_index(), on=keys, how="left")
    wide = wide.merge(target_week_wide.reset_index(), on=keys, how="left")
    wide = wide.merge(period_wide.reset_index(), on=keys, how="left")
    for h in HORIZONS:
        wide[f"target_week_h{h}"] = pd.to_datetime(wide[f"target_week_h{h}"])
    return wide.sort_values(["country", "region", "week_start"]).reset_index(drop=True)


def apply_official_delay(wide: pd.DataFrame, delay: int) -> pd.DataFrame:
    out = wide.copy()
    out["official_delay_weeks"] = int(delay)
    out = out.sort_values(["country", "region", "week_start"]).reset_index(drop=True)
    if delay == 0:
        return out
    for _, g in out.groupby(["country", "region"], sort=False):
        idx = g.index
        s = pd.to_numeric(g["target"], errors="coerce")
        out.loc[idx, "target_current"] = s.shift(delay).to_numpy()
        for lag in LAG_WEEKS:
            out.loc[idx, f"lag{lag}"] = s.shift(lag + delay).to_numpy()
        shifted = s.shift(1 + delay)
        for window in ROLL_WINDOWS:
            out.loc[idx, f"rollmean{window}"] = shifted.rolling(window, min_periods=2).mean().to_numpy()
            out.loc[idx, f"rollsd{window}"] = shifted.rolling(window, min_periods=3).std(ddof=0).to_numpy()
    return out


def make_feature_sets(wide: pd.DataFrame, out_path: Path) -> dict[str, list[str]]:
    cols = wide.columns.tolist()
    context_cols = [c for c in cols if c.startswith(("phase_", "region_"))]
    phase_cols = [c for c in cols if c.startswith("phase_")]
    surveillance_history_cols = (
        ["target_current"]
        + [f"lag{lag}" for lag in LAG_WEEKS]
        + [f"rollmean{w}" for w in ROLL_WINDOWS]
        + [f"rollsd{w}" for w in ROLL_WINDOWS]
        + ["sin1", "cos1", "sin2", "cos2", "sin3", "cos3", "iso_week_numeric"]
    )
    surveillance_history_cols = [c for c in surveillance_history_cols if c in cols and pd.api.types.is_numeric_dtype(wide[c])]
    epi_cols = surveillance_history_cols + [c for c in context_cols if pd.api.types.is_numeric_dtype(wide[c])]
    epi_no_region_cols = surveillance_history_cols + [c for c in phase_cols if pd.api.types.is_numeric_dtype(wide[c])]
    epi_cols = [c for c in epi_cols if c in cols and pd.api.types.is_numeric_dtype(wide[c])]
    epi_no_region_cols = [c for c in epi_no_region_cols if c in cols and pd.api.types.is_numeric_dtype(wide[c])]

    raw_cols = [c for c in cols if "_raw__" in c and pd.api.types.is_numeric_dtype(wide[c])]
    google_wiki_cols = [
        c
        for c in cols
        if c.startswith(("google_gt_yearwise__", "wiki_global__")) and pd.api.types.is_numeric_dtype(wide[c])
    ]
    flags = [
        c
        for c in ["baidu_has_data", "toutiao_has_data", "douyin_has_data", "weibo_has_data", "x_twitter_has_data", "google_has_data", "wiki_has_data", "n_sources_available"]
        if c in cols and pd.api.types.is_numeric_dtype(wide[c])
    ]
    cn_search = [c for c in raw_cols if c.startswith(("baidu_raw__", "toutiao_raw__", "douyin_raw__"))]
    cn_search_flags = [c for c in flags if c.startswith(("baidu", "toutiao", "douyin"))]
    google_wiki_flags = [c for c in flags if c.startswith(("google", "wiki"))]
    digital_disease_core = (
        [c for c in google_wiki_cols if c.endswith("__disease_core")]
        + [c for c in raw_cols if c.startswith("baidu_raw__") and any(term in c for term in ["流感", "甲流", "乙流"])]
        + [c for c in raw_cols if c.startswith("toutiao_raw__") and any(term in c for term in ["流感", "流行性感冒", "甲流"])]
        + [c for c in raw_cols if c.startswith("douyin_raw__") and "流感" in c]
    )
    digital_disease_core_flags = [
        c
        for c in flags
        if c.startswith(("google", "wiki", "baidu", "toutiao", "douyin"))
    ]
    social_raw = [c for c in raw_cols if c.startswith(("weibo_raw__", "x_twitter_raw__", "douyin_raw__", "toutiao_raw__"))]
    strict_social_raw = [c for c in raw_cols if c.startswith(("weibo_raw__", "x_twitter_raw__"))]
    strict_social_flags = [c for c in flags if c.startswith(("weibo", "x_twitter"))]
    global_open = google_wiki_cols + [c for c in raw_cols if c.startswith("x_twitter_raw__")]
    full_features = epi_cols + raw_cols + google_wiki_cols + flags
    feature_sets = {
        "no_digital": epi_cols,
        "full_raw_plus_open": full_features,
        "raw_cn_search": epi_cols + cn_search + cn_search_flags,
        "google_wiki": epi_cols + google_wiki_cols + google_wiki_flags,
        "social_raw": epi_cols + social_raw + [c for c in flags if c.startswith(("weibo", "x_twitter", "douyin", "toutiao"))],
        "global_open": epi_cols + global_open + [c for c in flags if c.startswith(("google", "wiki", "x_twitter"))],
        "full_no_china_local_search": [c for c in full_features if c not in set(cn_search + cn_search_flags)],
        "full_no_google_wiki": [c for c in full_features if c not in set(google_wiki_cols + google_wiki_flags)],
        "full_no_social_media": [c for c in full_features if c not in set(strict_social_raw + strict_social_flags)],
        "full_no_region_indicators": epi_no_region_cols + raw_cols + google_wiki_cols + flags,
        "surveillance_history_seasonality_only": surveillance_history_cols,
        "digital_disease_core_plus_surveillance": epi_cols + digital_disease_core + digital_disease_core_flags,
    }
    feature_sets = {k: list(dict.fromkeys(v)) for k, v in feature_sets.items()}
    out_path.write_text(json.dumps(feature_sets, ensure_ascii=False, indent=2), encoding="utf-8")
    return feature_sets


def fit_feature_matrix(wide: pd.DataFrame, features: list[str], train_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    x = wide[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    raw_cols = [c for c in features if "_raw__" in c]
    for c in raw_cols:
        vals = x[c].to_numpy(dtype=float)
        x[c] = np.where(vals >= 0, np.log1p(vals), vals)
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_mask.sum() < 5:
        raise ValueError("too few training rows for feature scaling")
    med = x.loc[train_mask].median().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = x.fillna(med).fillna(0.0)
    scaler = StandardScaler().fit(x.loc[train_mask].to_numpy(dtype=float))
    x_scaled = scaler.transform(x.to_numpy(dtype=float)).astype("float32")
    info = {"n_features": len(features), "n_raw_log_features": len(raw_cols)}
    return x_scaled, info


def date_mask(dates: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    dd = pd.to_datetime(dates)
    return ((dd >= start) & (dd <= end)).to_numpy()


def horizon_split_masks(wide: pd.DataFrame, spec: SplitSpec, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dcol = f"target_week_h{horizon}"
    ycol = f"y_h{horizon}"
    valid = (wide[ycol].notna().to_numpy().copy() & wide["target_current"].notna().to_numpy().copy())
    return (
        date_mask(wide[dcol], spec.train_start, spec.train_end) & valid,
        date_mask(wide[dcol], spec.val_start, spec.val_end) & valid,
        date_mask(wide[dcol], spec.test_start, spec.test_end) & valid,
    )


def all_horizon_masks(wide: pd.DataFrame, spec: SplitSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = wide["target_current"].notna().to_numpy().copy()
    for h in HORIZONS:
        valid = valid & wide[f"y_h{h}"].notna().to_numpy().copy()
    masks = []
    for start, end in [(spec.train_start, spec.train_end), (spec.val_start, spec.val_end), (spec.test_start, spec.test_end)]:
        m = valid.copy()
        for h in HORIZONS:
            m &= date_mask(wide[f"target_week_h{h}"], start, end)
        masks.append(m)
    return masks[0], masks[1], masks[2]


def split_validation_mask_by_dates(
    wide: pd.DataFrame,
    val_mask: np.ndarray,
    date_col: str,
    min_rows: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    val_mask = np.asarray(val_mask, dtype=bool)
    dates = pd.to_datetime(wide.loc[val_mask, date_col], errors="coerce").dropna().sort_values().unique()
    if len(dates) < 4:
        return val_mask.copy(), val_mask.copy()
    cut = pd.Timestamp(dates[len(dates) // 2 - 1])
    all_dates = pd.to_datetime(wide[date_col], errors="coerce")
    select = val_mask & (all_dates <= cut).to_numpy()
    calib = val_mask & (all_dates > cut).to_numpy()
    if select.sum() < min_rows or calib.sum() < min_rows:
        return val_mask.copy(), val_mask.copy()
    return select, calib


def split_all_horizon_validation_mask_purged(
    wide: pd.DataFrame,
    val_mask: np.ndarray,
    min_rows: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    val_mask = np.asarray(val_mask, dtype=bool)
    dates = pd.to_datetime(wide.loc[val_mask, "target_week_h1"], errors="coerce").dropna().sort_values().unique()
    if len(dates) < 4:
        raise ValueError("too few validation target weeks for purged selection/calibration split")
    cut = pd.Timestamp(dates[len(dates) // 2 - 1])
    h1_dates = pd.to_datetime(wide["target_week_h1"], errors="coerce")
    h4_dates = pd.to_datetime(wide["target_week_h4"], errors="coerce")
    select = val_mask & (h4_dates <= cut).to_numpy()
    calib = val_mask & (h1_dates > cut).to_numpy()
    embargo = val_mask & ~(select | calib)
    if select.sum() < min_rows or calib.sum() < min_rows:
        raise ValueError(
            f"insufficient purged selection/calibration rows: select={int(select.sum())}, "
            f"calib={int(calib.sum())}, embargo={int(embargo.sum())}"
        )
    info = {
        "selection_calibration_split": "purged_all_horizon_target_h4_le_cut_h1_gt_cut",
        "selection_calibration_cut": cut.strftime("%Y-%m-%d"),
        "n_validation_embargo": int(embargo.sum()),
        "n_validation_total": int(val_mask.sum()),
        "n_validation_select_mask": int(select.sum()),
        "n_validation_calib_mask": int(calib.sum()),
    }
    return select, calib, info


def fit_region_scalers_vector(wide: pd.DataFrame, train_mask: np.ndarray, y_col: str) -> dict[tuple[str, str], tuple[float, float]]:
    scalers: dict[tuple[str, str], tuple[float, float]] = {}
    global_vals = pd.to_numeric(wide.loc[train_mask, y_col], errors="coerce").to_numpy(dtype=float)
    gmu = float(np.nanmean(global_vals))
    gsd = float(np.nanstd(global_vals))
    if not np.isfinite(gsd) or gsd < 1e-8:
        gsd = 1.0
    for (country, region), g in wide.groupby(["country", "region"], sort=False):
        vals = pd.to_numeric(g.loc[train_mask[g.index], y_col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size >= 8:
            mu = float(np.nanmean(vals))
            sd = float(np.nanstd(vals))
        else:
            mu, sd = gmu, gsd
        if not np.isfinite(sd) or sd < 1e-8:
            sd = 1.0
        scalers[(country, region)] = (mu, sd)
    return scalers


def fit_region_scalers_matrix(wide: pd.DataFrame, train_mask: np.ndarray, y_cols: list[str]) -> dict[tuple[str, str], tuple[float, float]]:
    scalers: dict[tuple[str, str], tuple[float, float]] = {}
    global_vals = wide.loc[train_mask, y_cols].to_numpy(dtype=float).ravel()
    global_vals = global_vals[np.isfinite(global_vals)]
    gmu = float(np.nanmean(global_vals))
    gsd = float(np.nanstd(global_vals))
    if not np.isfinite(gsd) or gsd < 1e-8:
        gsd = 1.0
    for (country, region), g in wide.groupby(["country", "region"], sort=False):
        vals = g.loc[train_mask[g.index], y_cols].to_numpy(dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size >= 16:
            mu = float(np.nanmean(vals))
            sd = float(np.nanstd(vals))
        else:
            mu, sd = gmu, gsd
        if not np.isfinite(sd) or sd < 1e-8:
            sd = 1.0
        scalers[(country, region)] = (mu, sd)
    return scalers


def scaler_arrays(wide: pd.DataFrame, scalers: dict[tuple[str, str], tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    mu = np.zeros(len(wide), dtype=float)
    sd = np.ones(len(wide), dtype=float)
    for i, r in enumerate(wide[["country", "region"]].itertuples(index=False)):
        mu[i], sd[i] = scalers[(r.country, r.region)]
    return mu, sd


def residual_quantile_matrix(point: np.ndarray, residual: np.ndarray) -> np.ndarray:
    residual = np.asarray(residual, dtype=float)
    residual = residual[np.isfinite(residual)]
    if residual.size < 20:
        residual = np.r_[residual, np.zeros(20 - residual.size)]
    qs = np.quantile(residual, QUANTILES)
    return np.sort(point[:, None] + qs[None, :], axis=1)


def conformal_adjustment(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return 0.0
    level = min(1.0, math.ceil((scores.size + 1) * (1.0 - alpha)) / scores.size)
    return float(np.quantile(scores, level, method="higher"))


def enforce_monotone(qmat: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(np.asarray(qmat, dtype=float), axis=-1)


def conformal_calibrate(
    val_q: np.ndarray,
    val_y: np.ndarray,
    target_q: np.ndarray,
    context: dict,
    rows: list[dict],
) -> np.ndarray:
    out = np.asarray(target_q, dtype=float).copy()
    for label, alpha in [("50", 0.50), ("80", 0.20), ("95", 0.05)]:
        lo = QINDEX[qcol(alpha / 2.0)]
        hi = QINDEX[qcol(1.0 - alpha / 2.0)]
        scores = np.maximum.reduce([val_q[:, lo] - val_y, val_y - val_q[:, hi], np.zeros_like(val_y, dtype=float)])
        adj = conformal_adjustment(scores, alpha)
        out[:, lo] -= adj
        out[:, hi] += adj
        row = dict(context)
        row.update({"interval": label, "alpha": alpha, "conformal_adjustment_z": adj, "n_calibration": int(np.isfinite(scores).sum())})
        rows.append(row)
    return enforce_monotone(out)


def conformal_calibrate_mondrian(
    val_q: np.ndarray,
    val_y: np.ndarray,
    target_q: np.ndarray,
    val_groups: np.ndarray,
    target_groups: np.ndarray,
    context: dict,
    rows: list[dict],
    min_group_n: int = 24,
) -> np.ndarray:
    out = np.asarray(target_q, dtype=float).copy()
    val_groups = np.asarray(val_groups, dtype=object)
    target_groups = np.asarray(target_groups, dtype=object)
    unique_targets = pd.Series(target_groups).dropna().unique().tolist()
    for label, alpha in [("50", 0.50), ("80", 0.20), ("95", 0.05)]:
        lo = QINDEX[qcol(alpha / 2.0)]
        hi = QINDEX[qcol(1.0 - alpha / 2.0)]
        pooled_scores = np.maximum.reduce([val_q[:, lo] - val_y, val_y - val_q[:, hi], np.zeros_like(val_y, dtype=float)])
        pooled_adj = conformal_adjustment(pooled_scores, alpha)
        for group in unique_targets:
            val_mask = val_groups == group
            target_mask = target_groups == group
            group_scores = pooled_scores[val_mask]
            n_group = int(np.isfinite(group_scores).sum())
            if n_group >= min_group_n:
                group_adj = conformal_adjustment(group_scores, alpha)
                adj = max(group_adj, 0.50 * pooled_adj)
                calibration_level = "mondrian_region"
            else:
                adj = pooled_adj
                calibration_level = "pooled_fallback"
            out[target_mask, lo] -= adj
            out[target_mask, hi] += adj
            row = dict(context)
            row.update(
                {
                    "interval": label,
                    "alpha": alpha,
                    "calibration_group": group,
                    "calibration_level": calibration_level,
                    "conformal_adjustment_z": adj,
                    "pooled_adjustment_z": pooled_adj,
                    "n_calibration": n_group,
                }
            )
            rows.append(row)
    return enforce_monotone(out)


def conformal_calibrate_cube(
    val_q: np.ndarray,
    val_y: np.ndarray,
    target_q: np.ndarray,
    context: dict,
    rows: list[dict],
) -> np.ndarray:
    out = np.asarray(target_q, dtype=float).copy()
    for h_idx, horizon in enumerate(HORIZONS):
        out[:, h_idx, :] = conformal_calibrate(
            val_q[:, h_idx, :],
            val_y[:, h_idx],
            out[:, h_idx, :],
            {**context, "horizon": horizon},
            rows,
        )
    return enforce_monotone(out)


def conformal_calibrate_cube_mondrian(
    val_q: np.ndarray,
    val_y: np.ndarray,
    target_q: np.ndarray,
    val_groups: np.ndarray,
    target_groups: np.ndarray,
    context: dict,
    rows: list[dict],
) -> np.ndarray:
    out = np.asarray(target_q, dtype=float).copy()
    for h_idx, horizon in enumerate(HORIZONS):
        out[:, h_idx, :] = conformal_calibrate_mondrian(
            val_q[:, h_idx, :],
            val_y[:, h_idx],
            out[:, h_idx, :],
            val_groups,
            target_groups,
            {**context, "horizon": horizon},
            rows,
        )
    return enforce_monotone(out)


def seasonal_raw_prediction(wide: pd.DataFrame, horizon: int) -> np.ndarray:
    lookup = {
        (r.country, r.region, pd.Timestamp(r.week_start)): float(r.target)
        for r in wide[["country", "region", "week_start", "target"]].itertuples(index=False)
        if pd.notna(r.target)
    }
    vals = []
    for r in wide[["country", "region", "target_current", f"target_week_h{horizon}"]].itertuples(index=False):
        prev = pd.Timestamp(getattr(r, f"target_week_h{horizon}")) - pd.Timedelta(weeks=52)
        vals.append(lookup.get((r.country, r.region, prev), float(r.target_current)))
    return np.asarray(vals, dtype=float)


def prediction_rows_for_horizon(
    wide: pd.DataFrame,
    row_mask: np.ndarray,
    qmat: np.ndarray,
    y_z: np.ndarray,
    horizon: int,
    model: str,
    spec: SplitSpec,
    sample: str,
    delay: int,
    mu: np.ndarray,
    sd: np.ndarray,
    event_threshold: np.ndarray,
    extra: dict,
) -> pd.DataFrame:
    part = wide.loc[row_mask].copy().reset_index(drop=True)
    out = pd.DataFrame(
        {
            "sample": sample,
            "evaluation": spec.evaluation,
            "fold": spec.fold,
            "official_delay_weeks": delay,
            "country": part["country"].values,
            "region": part["region"].values,
            "horizon": horizon,
            "model": model,
            "origin_week_start": part["week_start"].values,
            "target_week_start": part[f"target_week_h{horizon}"].values,
            "period": part[f"period_h{horizon}"].values,
            "y_true": y_z[row_mask],
            "target_raw": part[f"y_h{horizon}"].values,
            "target_current_raw": part["target_current"].values,
            "scale_mu": mu[row_mask],
            "scale_sd": sd[row_mask],
            "event_threshold_z_train_q80": event_threshold[row_mask],
            "train_start": spec.train_start,
            "train_end": spec.train_end,
            "val_start": spec.val_start,
            "val_end": spec.val_end,
            "test_start": spec.test_start,
            "test_end": spec.test_end,
        }
    )
    for i, c in enumerate(QCOLS):
        out[c] = qmat[:, i]
        out[f"{c}_raw"] = np.maximum(out[c] * out["scale_sd"] + out["scale_mu"], 0.0)
    for k, v in extra.items():
        out[k] = v
    return out


def run_tabular_model(
    wide: pd.DataFrame,
    features: list[str],
    spec: SplitSpec,
    horizon: int,
    model: str,
    model_kind: str,
    delay: int,
    calibration_rows: list[dict],
) -> tuple[pd.DataFrame, dict]:
    train, val_all, test = horizon_split_masks(wide, spec, horizon)
    select, calib = split_validation_mask_by_dates(wide, val_all, f"target_week_h{horizon}")
    if train.sum() < 40 or select.sum() < 8 or calib.sum() < 8 or test.sum() < 10:
        raise ValueError("insufficient split rows")
    scalers = fit_region_scalers_vector(wide, train, f"y_h{horizon}")
    mu, sd = scaler_arrays(wide, scalers)
    y_raw = pd.to_numeric(wide[f"y_h{horizon}"], errors="coerce").to_numpy(dtype=float)
    y_z = (y_raw - mu) / sd
    p_z = (pd.to_numeric(wide["target_current"], errors="coerce").to_numpy(dtype=float) - mu) / sd

    if model_kind == "persistence":
        train_point = p_z[train]
        calib_point = p_z[calib]
        test_point = p_z[test]
        info = {}
    elif model_kind == "seasonal":
        s_raw = seasonal_raw_prediction(wide, horizon)
        s_z = (s_raw - mu) / sd
        train_point = s_z[train]
        calib_point = s_z[calib]
        test_point = s_z[test]
        info = {}
    else:
        x_all, finfo = fit_feature_matrix(wide, features, train)
        if model_kind == "ridge":
            reg = RidgeCV(alphas=np.logspace(-4, 4, 17))
        elif model_kind == "gb":
            reg = HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=180,
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=0.02,
                random_state=20260624 + horizon + delay,
            )
        else:
            raise ValueError(model_kind)
        reg.fit(x_all[train], y_z[train])
        train_point = reg.predict(x_all[train])
        calib_point = reg.predict(x_all[calib])
        test_point = reg.predict(x_all[test])
        info = {"estimator": reg.__class__.__name__, **finfo}
        if hasattr(reg, "alpha_"):
            info["alpha"] = float(reg.alpha_)

    train_resid = y_z[train] - train_point
    calib_q0 = residual_quantile_matrix(calib_point, train_resid)
    test_q0 = residual_quantile_matrix(test_point, train_resid)
    context = {
        "evaluation": spec.evaluation,
        "fold": spec.fold,
        "official_delay_weeks": delay,
        "model": model,
        "horizon": horizon,
        "calibration_scope": "mondrian_region_horizon_independent_calibration",
    }
    groups = (wide["country"].astype(str) + "::" + wide["region"].astype(str)).to_numpy(dtype=object)
    calib_q = conformal_calibrate_mondrian(calib_q0, y_z[calib], calib_q0, groups[calib], groups[calib], context, calibration_rows)
    test_q = conformal_calibrate_mondrian(calib_q0, y_z[calib], test_q0, groups[calib], groups[test], context, calibration_rows)
    thresholds = np.zeros(len(wide), dtype=float)
    for (country, region), g in wide.groupby(["country", "region"], sort=False):
        idx = g.index.to_numpy()
        vals = y_z[idx][train[idx]]
        vals = vals[np.isfinite(vals)]
        thr = float(np.nanquantile(vals, TARGET_EVENT_QUANTILE)) if vals.size else np.nan
        thresholds[idx] = thr

    calib_frame = prediction_rows_for_horizon(
        wide, calib, calib_q, y_z, horizon, model, spec, "calib", delay, mu, sd, thresholds, info
    )
    test_frame = prediction_rows_for_horizon(
        wide, test, test_q, y_z, horizon, model, spec, "test", delay, mu, sd, thresholds, info
    )
    meta = {
        "evaluation": spec.evaluation,
        "fold": spec.fold,
        "official_delay_weeks": delay,
        "country": wide["country"].iloc[0],
        "horizon": horizon,
        "model": model,
        "model_kind": model_kind,
        "n_train": int(train.sum()),
        "n_model_select": int(select.sum()),
        "n_calib": int(calib.sum()),
        "n_val_total": int(val_all.sum()),
        "n_test": int(test.sum()),
        "n_features": len(features),
        **info,
    }
    return pd.concat([calib_frame, test_frame], ignore_index=True), meta


def feature_gate_groups(features: list[str], no_digital_features: list[str]) -> tuple[list[int], dict[str, list[int]]]:
    no_digital_set = set(no_digital_features)
    epi_idx = [i for i, c in enumerate(features) if c in no_digital_set]
    groups: dict[str, list[int]] = {
        "china_local_search": [],
        "google_wiki": [],
        "social_media": [],
        "other_digital": [],
    }
    for i, c in enumerate(features):
        if c in no_digital_set:
            continue
        if c.startswith(("baidu_raw__", "toutiao_raw__", "douyin_raw__", "baidu_has_data", "toutiao_has_data", "douyin_has_data")):
            groups["china_local_search"].append(i)
        elif c.startswith(("google_gt_yearwise__", "wiki_global__", "google_has_data", "wiki_has_data")):
            groups["google_wiki"].append(i)
        elif c.startswith(("weibo_raw__", "x_twitter_raw__", "weibo_has_data", "x_twitter_has_data")):
            groups["social_media"].append(i)
        else:
            groups["other_digital"].append(i)
    groups = {k: v for k, v in groups.items() if v}
    return epi_idx, groups


class SourceGatedQuantileLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float, epi_idx: list[int], gate_groups: dict[str, list[int]]):
        super().__init__()
        self.has_gate = len(gate_groups) > 0 and len(epi_idx) > 0
        if self.has_gate:
            self.register_buffer("epi_idx", torch.tensor(epi_idx, dtype=torch.long))
            gate_hidden = max(4, min(32, len(epi_idx) // 2))
            self.group_names = list(gate_groups.keys())
            self.group_buffers: dict[str, str] = {}
            self.gates = nn.ModuleDict()
            for name, idx in gate_groups.items():
                buffer_name = f"{name}_idx"
                self.register_buffer(buffer_name, torch.tensor(idx, dtype=torch.long))
                self.group_buffers[name] = buffer_name
                self.gates[name] = nn.Sequential(nn.Linear(len(epi_idx), gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1))
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, len(HORIZONS) * len(QUANTILES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_gate:
            x = x.clone()
            epi = torch.index_select(x, -1, self.epi_idx)
            for name in self.group_names:
                idx = getattr(self, self.group_buffers[name])
                gate = torch.sigmoid(self.gates[name](epi))
                x[:, :, idx] = x[:, :, idx] * gate
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :]).view(-1, len(HORIZONS), len(QUANTILES))

    def gate_values(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.has_gate:
            return {}
        epi = torch.index_select(x, -1, self.epi_idx)
        return {name: torch.sigmoid(self.gates[name](epi)).squeeze(-1) for name in self.group_names}


def pinball_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    q = torch.tensor(QUANTILES, dtype=pred.dtype, device=pred.device).view(1, 1, -1)
    err = target.unsqueeze(-1) - pred
    loss = torch.maximum(q * err, (q - 1.0) * err).mean()
    crossing = torch.relu(pred[:, :, :-1] - pred[:, :, 1:]).mean()
    return loss + 0.05 * crossing


def make_sequences(wide: pd.DataFrame, x_all: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    d = wide.copy().reset_index(drop=True)
    d["_row"] = np.arange(len(d))
    xs: list[np.ndarray] = []
    idxs: list[int] = []
    for _, g in d.sort_values("week_start").groupby(["country", "region"], sort=False):
        rows = g["_row"].to_numpy(dtype=int)
        for pos in range(lookback - 1, len(rows)):
            xs.append(x_all[rows[pos - lookback + 1 : pos + 1], :])
            idxs.append(rows[pos])
    return np.stack(xs).astype("float32"), np.asarray(idxs, dtype=int)


def predict_lstm(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    outs = []
    loader = DataLoader(TensorDataset(torch.tensor(x)), batch_size=2048)
    with torch.no_grad():
        for (xb,) in loader:
            outs.append(model(xb).numpy())
    return np.concatenate(outs, axis=0)


def predict_lstm_gates(model: nn.Module, x: np.ndarray, batch_size: int = 2048) -> dict[str, np.ndarray]:
    if not isinstance(model, SourceGatedQuantileLSTM) or not model.has_gate:
        return {}
    model.eval()
    outs: dict[str, list[np.ndarray]] = {}
    loader = DataLoader(TensorDataset(torch.tensor(x)), batch_size=batch_size)
    with torch.no_grad():
        for (xb,) in loader:
            for name, value in model.gate_values(xb).items():
                outs.setdefault(name, []).append(value.detach().cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in outs.items() if parts}


def gate_interpretation(group_name: str, country: str) -> tuple[bool, bool, str]:
    if group_name == "china_local_search":
        available = country == "CHN"
        return available, available, "China-local search platform; interpret only for CHN."
    if group_name == "google_wiki":
        return True, True, "Open Google/Wikipedia signal group."
    if group_name == "social_media":
        available = country in {"CHN", "USA"}
        return available, available, "Social-media signal group; JPN has no local social stream in this package."
    if group_name == "other_digital":
        return False, False, "Contains availability/count features such as n_sources_available; not an independent source contribution."
    return False, False, "Unrecognized gate group; keep raw coefficient for audit only."


def append_gate_summary_rows(
    rows: list[dict],
    wide: pd.DataFrame,
    row_ids: np.ndarray,
    gate_values: dict[str, np.ndarray],
    context: dict,
    candidate_rank: int,
    seed: int,
) -> None:
    if not gate_values:
        return
    meta = wide.iloc[row_ids][["country", "region", "week_start"]].copy()
    meta["origin_period"] = make_period(meta["week_start"])
    for group_name, values in gate_values.items():
        d = meta.copy()
        d["gate_weight"] = np.nanmean(values, axis=1)
        grouped = (
            d.groupby(["country", "region", "origin_period"], dropna=False)["gate_weight"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        for r in grouped.itertuples(index=False):
            platform_available, interpretable_gate, note = gate_interpretation(group_name, str(r.country))
            raw_mean = float(r.mean)
            raw_sd = float(r.std) if pd.notna(r.std) else np.nan
            masked_mean = raw_mean if interpretable_gate else np.nan
            masked_sd = raw_sd if interpretable_gate else np.nan
            rows.append(
                {
                    **context,
                    "horizon": "all",
                    "gate_group": group_name,
                    "candidate_rank": candidate_rank,
                    "seed": seed,
                    "country": r.country,
                    "region": r.region,
                    "origin_period": r.origin_period,
                    "platform_available": bool(platform_available),
                    "interpretable_gate": bool(interpretable_gate),
                    "mean_gate_coefficient": masked_mean,
                    "sd_gate_coefficient": masked_sd,
                    "mean_gate_coefficient_raw": raw_mean,
                    "sd_gate_coefficient_raw": raw_sd,
                    "mean_gate_weight": masked_mean,
                    "sd_gate_weight": masked_sd,
                    "n_sequences": int(r.count),
                    "gate_interpretation_note": note,
                }
            )


def train_quantile_lstm(
    x_seq: np.ndarray,
    target_resid: np.ndarray,
    train_seq: np.ndarray,
    val_seq: np.ndarray,
    cfg: dict,
    seed: int,
    epi_idx: list[int],
    gate_groups: dict[str, list[int]],
) -> tuple[nn.Module, dict]:
    seed_all(seed)
    model = SourceGatedQuantileLSTM(
        n_features=x_seq.shape[2],
        hidden=cfg["hidden"],
        dropout=cfg["dropout"],
        epi_idx=epi_idx,
        gate_groups=gate_groups,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    tr_x = torch.tensor(x_seq[train_seq])
    tr_y = torch.tensor(target_resid[train_seq].astype("float32"))
    va_x = torch.tensor(x_seq[val_seq])
    va_y = torch.tensor(target_resid[val_seq].astype("float32"))
    dl = DataLoader(TensorDataset(tr_x, tr_y), batch_size=cfg["batch_size"], shuffle=True)
    best = np.inf
    best_state = None
    bad = 0
    best_epoch = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = pinball_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(pinball_loss(model(va_x), va_y).item())
        if val_loss < best - 1e-6:
            best = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
        if bad >= cfg["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_loss": best, "best_epoch": best_epoch}


def balanced_sequence_wis(wide: pd.DataFrame, row_ids: np.ndarray, y_z: np.ndarray, qcube: np.ndarray) -> float:
    parts = []
    regions = wide.iloc[row_ids]["region"].to_numpy()
    for h_idx, horizon in enumerate(HORIZONS):
        score = wis_from_matrix(y_z[row_ids, h_idx], qcube[:, h_idx, :])
        d = pd.DataFrame({"region": regions, "horizon": horizon, "score": score})
        parts.append(d.groupby(["region", "horizon"])["score"].mean().reset_index())
    region_h = pd.concat(parts, ignore_index=True)
    return float(region_h["score"].mean())


def run_sgq_lstm(
    wide: pd.DataFrame,
    features: list[str],
    no_digital_features: list[str],
    spec: SplitSpec,
    model: str,
    delay: int,
    configs: list[dict],
    seeds: list[int],
    calibration_rows: list[dict],
    ensemble_top_k: int,
    gate_summary_rows: list[dict] | None = None,
    use_gates: bool = True,
    use_conformal: bool = True,
) -> tuple[pd.DataFrame, dict]:
    train, val_all, test = all_horizon_masks(wide, spec)
    select, calib, split_info = split_all_horizon_validation_mask_purged(wide, val_all)
    if train.sum() < 40 or select.sum() < 8 or calib.sum() < 8 or test.sum() < 10:
        raise ValueError("insufficient all-horizon split rows")
    y_cols = [f"y_h{h}" for h in HORIZONS]
    scalers = fit_region_scalers_matrix(wide, train, y_cols)
    mu, sd = scaler_arrays(wide, scalers)
    y_raw = wide[y_cols].to_numpy(dtype=float)
    y_z = (y_raw - mu[:, None]) / sd[:, None]
    p_z = (pd.to_numeric(wide["target_current"], errors="coerce").to_numpy(dtype=float) - mu) / sd
    target_resid = y_z - p_z[:, None]

    x_all, finfo = fit_feature_matrix(wide, features, train)
    max_lookback = max(c["lookback"] for c in configs)
    x_seq, row_idxs = make_sequences(wide, x_all, max_lookback)
    row_pos = {row: i for i, row in enumerate(row_idxs)}
    train_seq = np.asarray([row_pos[i] for i in np.where(train)[0] if i in row_pos], dtype=int)
    select_seq = np.asarray([row_pos[i] for i in np.where(select)[0] if i in row_pos], dtype=int)
    calib_seq = np.asarray([row_pos[i] for i in np.where(calib)[0] if i in row_pos], dtype=int)
    test_seq = np.asarray([row_pos[i] for i in np.where(test)[0] if i in row_pos], dtype=int)
    if min(train_seq.size, select_seq.size, calib_seq.size, test_seq.size) < 8:
        raise ValueError("too few sequence rows")

    epi_idx, gate_groups = feature_gate_groups(features, no_digital_features)
    if not use_gates:
        gate_groups = {}
    digital_idx = sorted([i for idxs in gate_groups.values() for i in idxs])
    seq_target = target_resid[row_idxs]
    candidates = []
    for cfg in configs:
        trim = max_lookback - cfg["lookback"]
        x_use = x_seq[:, trim:, :] if trim else x_seq
        for seed in seeds:
            fit_model, fit_info = train_quantile_lstm(x_use, seq_target, train_seq, select_seq, cfg, seed, epi_idx, gate_groups)
            select_resid_q = predict_lstm(fit_model, x_use[select_seq])
            select_rows = row_idxs[select_seq]
            select_q = enforce_monotone(select_resid_q + p_z[select_rows, None, None])
            select_wis = balanced_sequence_wis(wide, select_rows, y_z, select_q)
            calib_rows = row_idxs[calib_seq]
            calib_q = enforce_monotone(predict_lstm(fit_model, x_use[calib_seq]) + p_z[calib_rows, None, None])
            test_rows = row_idxs[test_seq]
            test_q = enforce_monotone(predict_lstm(fit_model, x_use[test_seq]) + p_z[test_rows, None, None])
            cand = {
                "select_wis": select_wis,
                "cfg": cfg,
                "seed": seed,
                "fit_info": fit_info,
                "calib_rows": calib_rows,
                "test_rows": test_rows,
                "calib_q0": calib_q,
                "test_q0": test_q,
                "fit_model": fit_model,
                "trim": trim,
            }
            candidates.append(cand)
    if not candidates:
        raise ValueError("no LSTM candidates were trained")
    candidates = sorted(candidates, key=lambda x: x["select_wis"])
    selected = candidates[: max(1, min(ensemble_top_k, len(candidates)))]
    calib_rows = selected[0]["calib_rows"]
    test_rows = selected[0]["test_rows"]
    calib_q0 = enforce_monotone(np.mean([c["calib_q0"] for c in selected], axis=0))
    test_q0 = enforce_monotone(np.mean([c["test_q0"] for c in selected], axis=0))
    context = {
        "evaluation": spec.evaluation,
        "fold": spec.fold,
        "official_delay_weeks": delay,
        "model": model,
        "calibration_scope": "mondrian_region_multi_horizon_independent_calibration" if use_conformal else "none_raw_quantiles",
    }
    if gate_summary_rows is not None and use_gates and gate_groups:
        for rank, cand in enumerate(selected, start=1):
            x_gate = x_seq[:, cand["trim"] :, :] if cand["trim"] else x_seq
            gate_values = predict_lstm_gates(cand["fit_model"], x_gate[test_seq])
            append_gate_summary_rows(
                gate_summary_rows,
                wide,
                row_idxs[test_seq],
                gate_values,
                context,
                rank,
                cand["seed"],
            )
    groups = (wide["country"].astype(str) + "::" + wide["region"].astype(str)).to_numpy(dtype=object)
    if use_conformal:
        calib_q = conformal_calibrate_cube_mondrian(calib_q0, y_z[calib_rows], calib_q0, groups[calib_rows], groups[calib_rows], context, calibration_rows)
        test_q = conformal_calibrate_cube_mondrian(calib_q0, y_z[calib_rows], test_q0, groups[calib_rows], groups[test_rows], context, calibration_rows)
    else:
        calib_q = calib_q0
        test_q = test_q0

    threshold = np.full((len(wide), len(HORIZONS)), np.nan, dtype=float)
    for (country, region), g in wide.groupby(["country", "region"], sort=False):
        idx = g.index.to_numpy()
        for h_idx in range(len(HORIZONS)):
            vals = y_z[idx, h_idx][train[idx]]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                threshold[idx, h_idx] = float(np.nanquantile(vals, TARGET_EVENT_QUANTILE))

    frames = []
    calib_mask = np.zeros(len(wide), dtype=bool)
    calib_mask[calib_rows] = True
    test_mask = np.zeros(len(wide), dtype=bool)
    test_mask[test_rows] = True
    extra = {
        "estimator": "SourceGatedQuantileLSTM" if use_gates else "QuantileLSTM_no_source_gates",
        "ensemble_size": len(selected),
        "selected_seeds": json.dumps([c["seed"] for c in selected]),
        "selected_model_select_wis": float(np.mean([c["select_wis"] for c in selected])),
        "best_candidate_model_select_wis": selected[0]["select_wis"],
        "selected_configs": json.dumps([c["cfg"] for c in selected], sort_keys=True),
        "selected_best_epochs": json.dumps([c["fit_info"]["best_epoch"] for c in selected]),
        "gate_groups": json.dumps({k: len(v) for k, v in gate_groups.items()}, sort_keys=True),
        "uses_source_gates": bool(use_gates),
        "uses_conformal_calibration": bool(use_conformal),
        **split_info,
        **finfo,
    }
    for h_idx, horizon in enumerate(HORIZONS):
        frames.append(
            prediction_rows_for_horizon(
                wide,
                calib_mask,
                calib_q[:, h_idx, :],
                y_z[:, h_idx],
                horizon,
                model,
                spec,
                "calib",
                delay,
                mu,
                sd,
                threshold[:, h_idx],
                extra,
            )
        )
        frames.append(
            prediction_rows_for_horizon(
                wide,
                test_mask,
                test_q[:, h_idx, :],
                y_z[:, h_idx],
                horizon,
                model,
                spec,
                "test",
                delay,
                mu,
                sd,
                threshold[:, h_idx],
                extra,
            )
        )
    meta = {
        "evaluation": spec.evaluation,
        "fold": spec.fold,
        "official_delay_weeks": delay,
        "country": wide["country"].iloc[0],
        "horizon": "1-4",
        "model": model,
        "model_kind": "source_gated_multi_horizon_quantile_lstm" if use_gates else "multi_horizon_quantile_lstm_no_source_gates",
        "n_train": int(train_seq.size),
        "n_model_select": int(select_seq.size),
        "n_calib": int(calib_seq.size),
        "n_val_total": int(select_seq.size + calib_seq.size),
        "n_test": int(test_seq.size),
        "n_features": len(features),
        "n_digital_features": len(digital_idx),
        "n_gate_groups": len(gate_groups),
        "uses_source_gates": bool(use_gates),
        "uses_conformal_calibration": bool(use_conformal),
        "ensemble_size": len(selected),
        "selected_seeds": json.dumps([c["seed"] for c in selected]),
        "selected_model_select_wis": float(np.mean([c["select_wis"] for c in selected])),
        "best_candidate_model_select_wis": selected[0]["select_wis"],
        "selected_configs": json.dumps([c["cfg"] for c in selected], sort_keys=True),
        "selected_best_epochs": json.dumps([c["fit_info"]["best_epoch"] for c in selected]),
        "gate_groups": json.dumps({k: len(v) for k, v in gate_groups.items()}, sort_keys=True),
        **split_info,
    }
    return pd.concat(frames, ignore_index=True), meta


def add_relative_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    keys = ["evaluation", "official_delay_weeks", "horizon"]
    base = out[out["model"].isin(["persistence", FULL_LSTM_MODEL])][keys + ["model", "WIS"]]
    wide = base.pivot_table(index=keys, columns="model", values="WIS", aggfunc="mean").reset_index()
    wide = wide.rename(columns={"persistence": "WIS_persistence", FULL_LSTM_MODEL: "WIS_sgq_full"})
    out = out.merge(wide, on=keys, how="left")
    if "WIS_persistence" not in out.columns:
        out["WIS_persistence"] = np.nan
    if "WIS_sgq_full" not in out.columns:
        out["WIS_sgq_full"] = np.nan
    out["relative_WIS_vs_persistence"] = out["WIS"] / out["WIS_persistence"]
    out["delta_WIS_vs_sgq_full"] = out["WIS"] - out["WIS_sgq_full"]
    return out


def make_tables(pred: pd.DataFrame, out_paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    test = pred[pred["sample"] == "test"].copy()
    summary = (
        test.groupby(["evaluation", "official_delay_weeks", "model", "horizon"], group_keys=False)
        .apply(summarise)
        .reset_index()
    )
    summary = add_relative_metrics(summary)
    summary.to_csv(out_paths["tables"] / "metrics_summary_row_weighted.csv", index=False, encoding="utf-8-sig")

    country = (
        test.groupby(["evaluation", "official_delay_weeks", "country", "model", "horizon"], group_keys=False)
        .apply(summarise)
        .reset_index()
    )
    country.to_csv(out_paths["tables"] / "metrics_by_country_model_horizon.csv", index=False, encoding="utf-8-sig")

    macro = (
        country.groupby(["evaluation", "official_delay_weeks", "model", "horizon"])
        .agg(
            n=("n", "sum"),
            n_countries=("country", "nunique"),
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            Pearson=("Pearson", "mean"),
            Spearman=("Spearman", "mean"),
            WIS=("WIS", "mean"),
            coverage_50=("coverage_50", "mean"),
            coverage_80=("coverage_80", "mean"),
            coverage_95=("coverage_95", "mean"),
        )
        .reset_index()
    )
    macro = add_relative_metrics(macro)
    macro.to_csv(out_paths["tables"] / "metrics_macro_country_balanced.csv", index=False, encoding="utf-8-sig")

    region = (
        test.groupby(["evaluation", "official_delay_weeks", "country", "region", "model", "horizon"], group_keys=False)
        .apply(summarise)
        .reset_index()
    )
    region.to_csv(out_paths["tables"] / "metrics_by_region_model_horizon.csv", index=False, encoding="utf-8-sig")

    period = (
        test.groupby(["evaluation", "official_delay_weeks", "period", "country", "model", "horizon"], group_keys=False)
        .apply(summarise)
        .reset_index()
    )
    period.to_csv(out_paths["tables"] / "metrics_by_period_country_model_horizon.csv", index=False, encoding="utf-8-sig")

    common = make_common_sample_tables(test, out_paths)
    bootstrap = paired_block_bootstrap_common(common["pred"], out_paths)
    bootstrap_national = paired_block_bootstrap_common(
        common["national_pred"],
        out_paths,
        filename="paired_block_bootstrap_external_common_sample_national_only.csv",
    )
    return {
        "summary": summary,
        "country": country,
        "macro": macro,
        "region": region,
        "period": period,
        "common_macro": common["macro"],
        "common_national_macro": common["national_macro"],
        "common_regional_macro": common["regional_macro"],
        "bootstrap": bootstrap,
        "bootstrap_national": bootstrap_national,
    }


def common_key_columns() -> list[str]:
    return ["evaluation", "official_delay_weeks", "country", "region", "origin_week_start", "target_week_start", "horizon"]


def write_common_metrics_subset(
    common_pred: pd.DataFrame,
    out_paths: dict[str, Path],
    country_filename: str,
    macro_filename: str,
    region_filename: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if common_pred.empty:
        empty = pd.DataFrame()
        empty.to_csv(out_paths["tables"] / country_filename, index=False, encoding="utf-8-sig")
        empty.to_csv(out_paths["tables"] / macro_filename, index=False, encoding="utf-8-sig")
        if region_filename is not None:
            empty.to_csv(out_paths["tables"] / region_filename, index=False, encoding="utf-8-sig")
        return empty, empty, empty

    region = pd.DataFrame()
    if region_filename is not None:
        region = (
            common_pred.groupby(["evaluation", "official_delay_weeks", "country", "region", "model", "horizon"], group_keys=False)
            .apply(summarise)
            .reset_index()
        )
        region.to_csv(out_paths["tables"] / region_filename, index=False, encoding="utf-8-sig")

    country = (
        common_pred.groupby(["evaluation", "official_delay_weeks", "country", "model", "horizon"], group_keys=False)
        .apply(summarise)
        .reset_index()
    )
    country.to_csv(out_paths["tables"] / country_filename, index=False, encoding="utf-8-sig")
    macro = (
        country.groupby(["evaluation", "official_delay_weeks", "model", "horizon"])
        .agg(
            n=("n", "sum"),
            n_countries=("country", "nunique"),
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            Pearson=("Pearson", "mean"),
            Spearman=("Spearman", "mean"),
            WIS=("WIS", "mean"),
            coverage_50=("coverage_50", "mean"),
            coverage_80=("coverage_80", "mean"),
            coverage_95=("coverage_95", "mean"),
        )
        .reset_index()
    )
    macro = add_relative_metrics(macro)
    macro.to_csv(out_paths["tables"] / macro_filename, index=False, encoding="utf-8-sig")
    return country, macro, region


def make_common_sample_tables(test: pd.DataFrame, out_paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    key_cols = common_key_columns()
    common_parts = []
    for (evaluation, delay, horizon), g in test.groupby(["evaluation", "official_delay_weeks", "horizon"], sort=False):
        models = sorted(g["model"].dropna().unique())
        if len(models) < 2:
            continue
        counts = g.groupby(key_cols)["model"].nunique().reset_index(name="n_models")
        keep = counts[counts["n_models"] == len(models)][key_cols]
        if keep.empty:
            continue
        common_parts.append(g.merge(keep, on=key_cols, how="inner"))
    if common_parts:
        common_pred = pd.concat(common_parts, ignore_index=True)
    else:
        common_pred = test.iloc[0:0].copy()
    common_pred.to_csv(out_paths["tables"] / "predictions_common_sample_test_v6.csv", index=False, encoding="utf-8-sig")

    country, macro, _ = write_common_metrics_subset(
        common_pred,
        out_paths,
        "metrics_common_sample_by_country.csv",
        "metrics_common_sample_macro_country_balanced.csv",
    )
    national_pred = common_pred[common_pred["region"] == "national"].copy() if not common_pred.empty else common_pred.copy()
    national_country, national_macro, _ = write_common_metrics_subset(
        national_pred,
        out_paths,
        "metrics_common_sample_national_only_by_country.csv",
        "metrics_common_sample_national_only_macro.csv",
    )
    regional_pred = common_pred[common_pred["region"] != "national"].copy() if not common_pred.empty else common_pred.copy()
    regional_country, regional_macro, regional_region = write_common_metrics_subset(
        regional_pred,
        out_paths,
        "metrics_common_sample_regional_robustness_by_country.csv",
        "metrics_common_sample_regional_robustness_macro.csv",
        "metrics_common_sample_regional_robustness_by_region.csv",
    )
    return {
        "pred": common_pred,
        "country": country,
        "macro": macro,
        "national_pred": national_pred,
        "national_country": national_country,
        "national_macro": national_macro,
        "regional_pred": regional_pred,
        "regional_country": regional_country,
        "regional_macro": regional_macro,
        "regional_region": regional_region,
    }


def paired_block_bootstrap_common(
    common_pred: pd.DataFrame,
    out_paths: dict[str, Path],
    n_boot: int = 2000,
    filename: str = "paired_block_bootstrap_external_common_sample.csv",
) -> pd.DataFrame:
    if common_pred.empty:
        out = pd.DataFrame()
        out.to_csv(out_paths["tables"] / filename, index=False, encoding="utf-8-sig")
        return out
    rng = np.random.default_rng(20260624)
    comparisons = [
        (FULL_LSTM_MODEL, "persistence"),
        (FULL_LSTM_MODEL, "ridge_raw_cn_search"),
        (FULL_LSTM_MODEL, NO_DIGITAL_LSTM_MODEL),
        (FULL_LSTM_MODEL, "bioepi_sgq_lstm_no_source_gates"),
        ("ridge_raw_cn_search", "ridge_no_digital"),
        ("ridge_raw_cn_search", "persistence"),
    ]
    rows = []
    key_cols = common_key_columns()
    work = common_pred[common_pred["evaluation"] == "temporal_external_2025_2026"].copy()
    if work.empty:
        out = pd.DataFrame()
        out.to_csv(out_paths["tables"] / filename, index=False, encoding="utf-8-sig")
        return out
    work["origin_week_start"] = pd.to_datetime(work["origin_week_start"])
    min_origin = work["origin_week_start"].min()
    work["origin_block4"] = ((work["origin_week_start"] - min_origin).dt.days // 28).astype(int)
    for (delay, horizon), g in work.groupby(["official_delay_weeks", "horizon"], sort=False):
        pivot = g.pivot_table(index=key_cols + ["origin_block4"], columns="model", values="WIS", aggfunc="mean").reset_index()
        for model_a, model_b in comparisons:
            if model_a not in pivot.columns or model_b not in pivot.columns:
                continue
            pair = pivot.dropna(subset=[model_a, model_b]).copy()
            if pair.empty:
                continue
            pair["diff_wis"] = pair[model_a] - pair[model_b]
            for country_scope in ["macro"] + sorted(pair["country"].unique().tolist()):
                if country_scope == "macro":
                    dscope = pair
                else:
                    dscope = pair[pair["country"] == country_scope].copy()
                if dscope.empty:
                    continue
                obs_country = dscope.groupby("country")["diff_wis"].mean()
                obs = float(obs_country.mean())
                block_means = dscope.groupby(["country", "origin_block4"])["diff_wis"].mean().reset_index()
                boot_vals = []
                for _ in range(n_boot):
                    country_means = []
                    for country, cg in block_means.groupby("country", sort=False):
                        vals = cg["diff_wis"].to_numpy(dtype=float)
                        if vals.size == 0:
                            continue
                        sampled = vals[rng.integers(0, vals.size, vals.size)]
                        country_means.append(float(np.mean(sampled)))
                    if country_means:
                        boot_vals.append(float(np.mean(country_means)))
                if boot_vals:
                    rows.append(
                        {
                            "evaluation": "temporal_external_2025_2026",
                            "official_delay_weeks": delay,
                            "horizon": horizon,
                            "country_scope": country_scope,
                            "model_a": model_a,
                            "model_b": model_b,
                            "diff_WIS_a_minus_b": obs,
                            "relative_diff_vs_b": obs / float(dscope[model_b].mean()) if float(dscope[model_b].mean()) != 0 else np.nan,
                            "ci025": float(np.quantile(boot_vals, 0.025)),
                            "ci975": float(np.quantile(boot_vals, 0.975)),
                            "n_boot": len(boot_vals),
                            "n_pairs": int(len(dscope)),
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(out_paths["tables"] / filename, index=False, encoding="utf-8-sig")
    return out


def exceedance_probability(qvals: np.ndarray, threshold: float) -> float:
    qvals = np.asarray(qvals, dtype=float)
    if not np.isfinite(threshold) or not np.isfinite(qvals).all():
        return np.nan
    if threshold <= qvals[0]:
        return 0.975
    if threshold >= qvals[-1]:
        return 0.025
    cdf = float(np.interp(threshold, qvals, QUANTILES))
    return float(np.clip(1.0 - cdf, 0.0, 1.0))


def suppress_refractory(alarms: pd.DataFrame, refractory_weeks: int = 4) -> pd.DataFrame:
    if alarms.empty:
        return alarms
    keep = []
    last = None
    for row in alarms.sort_values("origin_week_start").itertuples(index=False):
        d = pd.Timestamp(row.origin_week_start)
        if last is None or (d - last).days >= refractory_weeks * 7:
            keep.append(True)
            last = d
        else:
            keep.append(False)
    return alarms.sort_values("origin_week_start").loc[keep].copy()


def event_peaks(g: pd.DataFrame, threshold_col: str = "event_threshold_z_train_q80") -> pd.DataFrame:
    d = g.sort_values("target_week_start").reset_index(drop=True)
    if len(d) < 5:
        return d.iloc[0:0]
    y = d["y_true"].to_numpy(dtype=float)
    thr = d[threshold_col].to_numpy(dtype=float)
    peaks = []
    for i in range(1, len(d) - 1):
        if np.isfinite(y[i]) and np.isfinite(thr[i]) and y[i] >= thr[i] and y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            peaks.append(i)
    return d.iloc[peaks].copy()


def match_alarm_metrics(events: pd.DataFrame, alarms: pd.DataFrame, lead_window_weeks: int = 6) -> dict:
    matched = 0
    used: set[int] = set()
    leads = []
    events = events.sort_values("target_week_start").reset_index(drop=True)
    alarms = alarms.sort_values("origin_week_start").reset_index(drop=True)
    for ev in events.itertuples(index=False):
        peak = pd.Timestamp(ev.target_week_start)
        candidates = []
        for i, al in enumerate(alarms.itertuples(index=False)):
            if i in used:
                continue
            alarm_date = pd.Timestamp(al.origin_week_start)
            lead = (peak - alarm_date).days / 7.0
            if 0 <= lead <= lead_window_weeks:
                candidates.append((lead, i))
        if candidates:
            lead, idx = max(candidates, key=lambda x: x[0])
            used.add(idx)
            matched += 1
            leads.append(lead)
    n_events = len(events)
    n_alarms = len(alarms)
    sensitivity = matched / n_events if n_events else np.nan
    ppv = matched / n_alarms if n_alarms else np.nan
    f1 = 2 * sensitivity * ppv / (sensitivity + ppv) if np.isfinite(sensitivity) and np.isfinite(ppv) and sensitivity + ppv > 0 else np.nan
    return {
        "true_events": int(n_events),
        "alarms": int(n_alarms),
        "matched": int(matched),
        "sensitivity": sensitivity,
        "PPV": ppv,
        "F1": f1,
        "mean_lead_weeks": float(np.nanmean(leads)) if leads else np.nan,
        "sum_lead_weeks": float(np.nansum(leads)) if leads else 0.0,
    }


def compute_event_metrics(pred: pd.DataFrame, out_paths: dict[str, Path]) -> None:
    rows = []
    thresholds = []
    event_models = CORE_EVENT_MODELS
    work = pred[
        (pred["evaluation"] == "temporal_external_2025_2026")
        & (pred["model"].isin(event_models))
    ].copy()
    if work.empty:
        pd.DataFrame().to_csv(out_paths["tables"] / "event_metrics_corrected_by_region.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(out_paths["tables"] / "event_metrics_corrected_national_only.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(out_paths["tables"] / "event_probability_thresholds_from_validation.csv", index=False, encoding="utf-8-sig")
        return
    qmat = work[QCOLS].to_numpy(dtype=float)
    thr = work["event_threshold_z_train_q80"].to_numpy(dtype=float)
    work["event_probability"] = [exceedance_probability(qmat[i, :], thr[i]) for i in range(len(work))]
    for key, g in work.groupby(["evaluation", "fold", "official_delay_weeks", "model", "country", "horizon"], sort=False):
        evaluation, fold, delay, model, country, horizon = key
        val = g[g["sample"].isin(["calib", "val"])].copy()
        test = g[g["sample"] == "test"].copy()
        if val.empty or test.empty:
            continue
        grid = sorted(set(np.linspace(0.10, 0.90, 17).round(2).tolist() + np.nanquantile(val["event_probability"], [0.50, 0.65, 0.80, 0.90]).round(3).tolist()))
        best = None
        for cut in grid:
            region_stats = []
            for (_, region), vr in val.groupby(["country", "region"], sort=False):
                events = event_peaks(vr)
                alarms = suppress_refractory(vr[vr["event_probability"] >= cut])
                region_stats.append(match_alarm_metrics(events, alarms))
            pooled_events = sum(s["true_events"] for s in region_stats)
            pooled_alarms = sum(s["alarms"] for s in region_stats)
            pooled_matched = sum(s["matched"] for s in region_stats)
            sens = pooled_matched / pooled_events if pooled_events else np.nan
            ppv = pooled_matched / pooled_alarms if pooled_alarms else np.nan
            f1 = 2 * sens * ppv / (sens + ppv) if np.isfinite(sens) and np.isfinite(ppv) and sens + ppv > 0 else np.nan
            cand = {"probability_cutoff": float(cut), "validation_sensitivity": sens, "validation_PPV": ppv, "validation_F1": f1}
            eligible = np.isfinite(ppv) and ppv >= 0.50
            score = (1 if eligible else 0, -np.inf if not np.isfinite(f1) else f1, -cut)
            if best is None or score > best["score"]:
                best = {**cand, "score": score}
        if best is None:
            continue
        thresholds.append({k: v for k, v in best.items() if k != "score"} | {"evaluation": evaluation, "fold": fold, "official_delay_weeks": delay, "model": model, "country": country, "horizon": horizon})
        cut = best["probability_cutoff"]
        for (country2, region), tr in test.groupby(["country", "region"], sort=False):
            events = event_peaks(tr)
            alarms = suppress_refractory(tr[tr["event_probability"] >= cut])
            stats = match_alarm_metrics(events, alarms)
            rows.append(
                {
                    "evaluation": evaluation,
                    "fold": fold,
                    "official_delay_weeks": delay,
                    "model": model,
                    "country": country2,
                    "region": region,
                    "horizon": horizon,
                    "probability_cutoff": cut,
                    **stats,
                }
            )
    event_region = pd.DataFrame(rows)
    event_thresholds = pd.DataFrame(thresholds)
    event_region.to_csv(out_paths["tables"] / "event_metrics_corrected_by_region.csv", index=False, encoding="utf-8-sig")
    event_thresholds.to_csv(out_paths["tables"] / "event_probability_thresholds_from_validation.csv", index=False, encoding="utf-8-sig")
    if event_region.empty:
        return
    national = event_region[event_region["region"] == "national"].copy()
    national.to_csv(out_paths["tables"] / "event_metrics_corrected_national_only.csv", index=False, encoding="utf-8-sig")
    country = (
        event_region.groupby(["evaluation", "official_delay_weeks", "model", "country", "horizon"])
        .agg(true_events=("true_events", "sum"), alarms=("alarms", "sum"), matched=("matched", "sum"), sum_lead_weeks=("sum_lead_weeks", "sum"))
        .reset_index()
    )
    country["sensitivity"] = country["matched"] / country["true_events"].replace(0, np.nan)
    country["PPV"] = country["matched"] / country["alarms"].replace(0, np.nan)
    country["F1"] = 2 * country["sensitivity"] * country["PPV"] / (country["sensitivity"] + country["PPV"])
    country["mean_lead_weeks"] = country["sum_lead_weeks"] / country["matched"].replace(0, np.nan)
    country.to_csv(out_paths["tables"] / "event_metrics_corrected_by_country.csv", index=False, encoding="utf-8-sig")


def figure_data_coverage(root: Path, out_paths: dict[str, Path]) -> None:
    path = root / "data" / "processed" / "coverage_qc_by_country_region.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    mat = df.pivot_table(index="country", columns="region", values="coverage_percent", aggfunc="mean")
    plt.figure(figsize=(11, 3.4))
    sns.heatmap(mat, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "target coverage (%)"})
    plt.title("Gold-standard target coverage by country and region")
    plt.tight_layout()
    plt.savefig(out_paths["figures"] / "fig1_gold_standard_coverage.png", dpi=300)
    plt.close()


def figure_external_forecasts(pred: pd.DataFrame, out_paths: dict[str, Path]) -> None:
    d = pred[
        (pred["sample"] == "test")
        & (pred["evaluation"] == "temporal_external_2025_2026")
        & (pred["official_delay_weeks"] == 0)
        & (pred["model"] == FULL_LSTM_MODEL)
        & (pred["region"] == "national")
        & (pred["horizon"].isin([1, 4]))
    ].copy()
    if d.empty:
        return
    countries = [c for c in ["CHN", "USA", "JPN"] if c in set(d["country"])]
    fig, axes = plt.subplots(len(countries), 2, figsize=(13, 7.8), sharex=False)
    if len(countries) == 1:
        axes = np.asarray([axes])
    for i, country in enumerate(countries):
        for j, horizon in enumerate([1, 4]):
            ax = axes[i, j]
            g = d[(d["country"] == country) & (d["horizon"] == horizon)].sort_values("target_week_start")
            x = pd.to_datetime(g["target_week_start"])
            ax.fill_between(x, g["q0.025_raw"].clip(lower=0), g["q0.975_raw"].clip(lower=0), color="#b9c6d4", alpha=0.35, linewidth=0)
            ax.fill_between(x, g["q0.1_raw"].clip(lower=0), g["q0.9_raw"].clip(lower=0), color="#6f91b8", alpha=0.30, linewidth=0)
            ax.plot(x, g["target_raw"], color="#111827", linewidth=1.4, label="Observed")
            ax.plot(x, g["q0.5_raw"], color="#1d4ed8", linewidth=1.4, label="Median forecast")
            ax.set_title(f"{country} h={horizon}")
            ax.set_ylabel("ILI target")
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Temporal external forecasts: calibrated BioEpi-SGQ-LSTM full_raw", y=0.985)
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0.055, 1, 0.94])
    plt.savefig(out_paths["figures"] / "fig2_external_sgq_forecasts.png", dpi=300)
    plt.close()


def figure_external_skill(macro: pd.DataFrame, out_paths: dict[str, Path]) -> None:
    d = macro[(macro["evaluation"] == "temporal_external_2025_2026") & (macro["official_delay_weeks"] == 0)].copy()
    if d.empty:
        return
    keep = ["persistence", "ridge_no_digital", "ridge_full_raw", "ridge_raw_cn_search", "gb_full_raw", NO_DIGITAL_LSTM_MODEL, FULL_LSTM_MODEL]
    d = d[d["model"].isin(keep)]
    plt.figure(figsize=(9.5, 4.5))
    sns.lineplot(data=d, x="horizon", y="relative_WIS_vs_persistence", hue="model", marker="o")
    plt.axhline(1.0, color="#555", linestyle="--", linewidth=1)
    plt.title("Country-balanced temporal external relative WIS")
    plt.ylabel("relative WIS vs persistence")
    plt.xlabel("forecast horizon (weeks)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_paths["figures"] / "fig3_external_macro_relative_wis.png", dpi=300)
    plt.close()


def figure_delay_scenarios(macro: pd.DataFrame, out_paths: dict[str, Path]) -> None:
    d = macro[(macro["evaluation"] == "temporal_external_2025_2026") & (macro["model"].isin(["persistence", "ridge_full_raw", FULL_LSTM_MODEL]))].copy()
    if d.empty:
        return
    d["model_delay"] = d["model"] + " | delay " + d["official_delay_weeks"].astype(str)
    plt.figure(figsize=(10, 5.2))
    sns.lineplot(data=d, x="horizon", y="relative_WIS_vs_persistence", hue="model", style="official_delay_weeks", marker="o")
    plt.axhline(1.0, color="#555", linestyle="--", linewidth=1)
    plt.title("Official-surveillance delay scenarios, external period")
    plt.ylabel("relative WIS vs same-delay persistence")
    plt.xlabel("forecast horizon (weeks)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_paths["figures"] / "fig4_external_delay_scenarios.png", dpi=300)
    plt.close()


def figure_calibration(macro: pd.DataFrame, out_paths: dict[str, Path]) -> None:
    d = macro[(macro["evaluation"] == "temporal_external_2025_2026") & (macro["official_delay_weeks"] == 0)].copy()
    keep = ["persistence", "ridge_full_raw", "gb_full_raw", FULL_LSTM_MODEL]
    d = d[d["model"].isin(keep)]
    if d.empty:
        return
    long = d.melt(
        id_vars=["model", "horizon"],
        value_vars=["coverage_50", "coverage_80", "coverage_95"],
        var_name="interval",
        value_name="coverage",
    )
    long["nominal"] = long["interval"].map({"coverage_50": 0.50, "coverage_80": 0.80, "coverage_95": 0.95})
    plt.figure(figsize=(9.5, 4.8))
    sns.pointplot(data=long, x="horizon", y="coverage", hue="interval", dodge=0.25, errorbar=None)
    for y in [0.50, 0.80, 0.95]:
        plt.axhline(y, color="#6b7280", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.title("External empirical interval coverage after conformal calibration")
    plt.ylabel("country-balanced coverage")
    plt.xlabel("forecast horizon (weeks)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_paths["figures"] / "fig5_external_calibration_coverage.png", dpi=300)
    plt.close()


def figure_ablation(macro: pd.DataFrame, out_paths: dict[str, Path]) -> None:
    d = macro[(macro["evaluation"] == "temporal_external_2025_2026") & (macro["official_delay_weeks"] == 0)].copy()
    d = d[
        d["model"].isin(
            [
                "ridge_no_digital",
                "ridge_full_raw",
                "ridge_raw_cn_search",
                "ridge_google_wiki",
                "ridge_social_raw",
                "ridge_global_open",
                NO_DIGITAL_LSTM_MODEL,
                FULL_LSTM_MODEL,
                "bioepi_sgq_lstm_no_china_local_search",
                "bioepi_sgq_lstm_no_google_wiki",
                "bioepi_sgq_lstm_no_social_media",
                SURVEILLANCE_HISTORY_MODEL,
                DIGITAL_DISEASE_CORE_MODEL,
                "bioepi_sgq_lstm_no_source_gates",
                "bioepi_sgq_lstm_no_conformal",
                NO_REGION_INDICATORS_MODEL,
            ]
        )
    ]
    if d.empty:
        return
    mat = d.pivot_table(index="model", columns="horizon", values="delta_WIS_vs_sgq_full")
    plt.figure(figsize=(9.5, 4.8))
    sns.heatmap(mat, annot=True, fmt=".3f", center=0, cmap="RdBu_r", cbar_kws={"label": "Delta WIS vs SGQ full"})
    plt.title("External ablation: country-balanced WIS difference")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_paths["figures"] / "fig6_external_ablation_delta_wis.png", dpi=300)
    plt.close()


def figure_event_metrics(out_paths: dict[str, Path]) -> None:
    path = out_paths["tables"] / "event_metrics_corrected_national_only.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    d = d[
        (d["evaluation"] == "temporal_external_2025_2026")
        & (d["official_delay_weeks"] == 0)
        & (d["model"] == FULL_LSTM_MODEL)
    ].copy()
    if d.empty:
        return
    long = d.melt(id_vars=["country", "horizon"], value_vars=["sensitivity", "PPV", "F1"], var_name="metric", value_name="value")
    g = sns.catplot(data=long, x="horizon", y="value", hue="metric", col="country", kind="bar", height=3.2, aspect=1.05)
    g.set_axis_labels("horizon", "value")
    g.set_titles("{col_name}")
    g.fig.suptitle("National-only event-warning metrics: first valid alarm before peak", y=1.05)
    g.fig.tight_layout()
    g.fig.savefig(out_paths["figures"] / "fig7_corrected_event_warning_metrics.png", dpi=300)
    plt.close(g.fig)


def make_figures(root: Path, pred: pd.DataFrame, tables: dict[str, pd.DataFrame], out_paths: dict[str, Path]) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    main_macro = tables.get("common_national_macro", pd.DataFrame())
    if main_macro.empty:
        main_macro = tables.get("common_macro", pd.DataFrame())
    if main_macro.empty:
        main_macro = tables["macro"]
    figure_data_coverage(root, out_paths)
    figure_external_forecasts(pred, out_paths)
    figure_external_skill(main_macro, out_paths)
    figure_delay_scenarios(main_macro, out_paths)
    figure_calibration(main_macro, out_paths)
    figure_ablation(main_macro, out_paths)
    figure_event_metrics(out_paths)


def run_experiments(root: Path, fast: bool, scope: str) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    out_paths = ensure_dirs(root)
    design_path = out_paths["processed"] / "forecast_design_final_revision.csv"
    design = pd.read_csv(design_path, parse_dates=["week_start", "week_end", "target_week_start"])
    wide = make_origin_wide(design)
    wide.to_csv(out_paths["processed"] / "forecast_design_origin_wide_v6.csv", index=False, encoding="utf-8-sig")
    feature_sets = make_feature_sets(wide, out_paths["metadata"] / "feature_sets_v6.json")

    target_weeks = []
    for h in HORIZONS:
        target_weeks.extend(wide[f"target_week_h{h}"].dropna().tolist())
    ext = external_split(target_weeks)
    if scope == "external":
        split_specs = [ext]
    elif scope == "rolling":
        split_specs = rolling_splits(target_weeks)
    else:
        split_specs = rolling_splits(target_weeks) + [ext]

    if fast:
        configs = [
            {"lookback": 12, "hidden": 16, "dropout": 0.10, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 20, "patience": 4, "batch_size": 512},
        ]
        seeds = SEEDS_FAST
        ensemble_top_k = 1
    else:
        configs = [
            {"lookback": 12, "hidden": 24, "dropout": 0.10, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 80, "patience": 12, "batch_size": 512},
            {"lookback": 16, "hidden": 32, "dropout": 0.15, "lr": 7e-4, "weight_decay": 1e-4, "epochs": 90, "patience": 12, "batch_size": 512},
            {"lookback": 26, "hidden": 32, "dropout": 0.20, "lr": 5e-4, "weight_decay": 2e-4, "epochs": 100, "patience": 15, "batch_size": 512},
        ]
        seeds = SEEDS_FULL
        ensemble_top_k = 5

    tabular_plan = [
        ("persistence", "no_digital", "persistence"),
        ("seasonal_naive", "no_digital", "seasonal"),
        ("ridge_no_digital", "no_digital", "ridge"),
        ("ridge_full_raw", "full_raw_plus_open", "ridge"),
        ("gb_full_raw", "full_raw_plus_open", "gb"),
        ("ridge_raw_cn_search", "raw_cn_search", "ridge"),
        ("ridge_google_wiki", "google_wiki", "ridge"),
        ("ridge_social_raw", "social_raw", "ridge"),
        ("ridge_global_open", "global_open", "ridge"),
    ]
    lstm_plan = [
        (NO_DIGITAL_LSTM_MODEL, "no_digital", True, True, False),
        (FULL_LSTM_MODEL, "full_raw_plus_open", True, True, False),
        ("bioepi_sgq_lstm_no_china_local_search", "full_no_china_local_search", True, True, True),
        ("bioepi_sgq_lstm_no_google_wiki", "full_no_google_wiki", True, True, True),
        ("bioepi_sgq_lstm_no_social_media", "full_no_social_media", True, True, True),
        (SURVEILLANCE_HISTORY_MODEL, "surveillance_history_seasonality_only", True, True, True),
        (DIGITAL_DISEASE_CORE_MODEL, "digital_disease_core_plus_surveillance", True, True, True),
        (NO_REGION_INDICATORS_MODEL, "full_no_region_indicators", True, True, True),
        ("bioepi_sgq_lstm_no_source_gates", "full_raw_plus_open", False, True, True),
        ("bioepi_sgq_lstm_no_conformal", "full_raw_plus_open", True, False, True),
    ]

    delays = [0, 1, 2] if scope in {"external", "all"} else [0]
    all_frames: list[pd.DataFrame] = []
    logs: list[dict] = []
    calibration_rows: list[dict] = []
    gate_summary_rows: list[dict] = []
    countries = sorted(wide["country"].dropna().unique())
    torch.set_num_threads(2)
    for delay in delays:
        w_delay = apply_official_delay(wide, delay)
        for spec in split_specs:
            if spec.evaluation == ROLLING_EVALUATION and delay != 0:
                continue
            for country in countries:
                wc = w_delay[w_delay["country"] == country].reset_index(drop=True)
                for model_name, fset, kind in tabular_plan:
                    for horizon in HORIZONS:
                        t0 = time.time()
                        try:
                            frame, meta = run_tabular_model(
                                wc,
                                feature_sets[fset],
                                spec,
                                horizon,
                                model_name,
                                kind,
                                delay,
                                calibration_rows,
                            )
                            meta["run_seconds"] = time.time() - t0
                            all_frames.append(frame)
                            logs.append(meta)
                            print(f"[OK] delay={delay} {spec.evaluation} fold={spec.fold} {country} h={horizon} {model_name}", flush=True)
                        except Exception as exc:
                            logs.append(
                                {
                                    "evaluation": spec.evaluation,
                                    "fold": spec.fold,
                                    "official_delay_weeks": delay,
                                    "country": country,
                                    "horizon": horizon,
                                    "model": model_name,
                                    "error": repr(exc),
                                    "run_seconds": time.time() - t0,
                                }
                            )
                            print(f"[WARN] failed delay={delay} {spec.evaluation} {country} h={horizon} {model_name}: {exc}", flush=True)
                for model_name, fset, use_gates, use_conformal, external_delay0_only in lstm_plan:
                    if external_delay0_only and (fast or spec.evaluation != "temporal_external_2025_2026" or delay != 0):
                        continue
                    t0 = time.time()
                    try:
                        frame, meta = run_sgq_lstm(
                            wc,
                            feature_sets[fset],
                            feature_sets["no_digital"],
                            spec,
                            model_name,
                            delay,
                            configs,
                            seeds,
                            calibration_rows,
                            ensemble_top_k,
                            gate_summary_rows,
                            use_gates=use_gates,
                            use_conformal=use_conformal,
                        )
                        meta["run_seconds"] = time.time() - t0
                        all_frames.append(frame)
                        logs.append(meta)
                        print(f"[OK] delay={delay} {spec.evaluation} fold={spec.fold} {country} all-h {model_name}", flush=True)
                    except Exception as exc:
                        logs.append(
                            {
                                "evaluation": spec.evaluation,
                                "fold": spec.fold,
                                "official_delay_weeks": delay,
                                "country": country,
                                "horizon": "1-4",
                                "model": model_name,
                                "error": repr(exc),
                                "run_seconds": time.time() - t0,
                            }
                        )
                        print(f"[WARN] failed delay={delay} {spec.evaluation} {country} {model_name}: {exc}", flush=True)
    pred = add_scores(pd.concat(all_frames, ignore_index=True))
    pred.to_csv(out_paths["tables"] / "predictions_calib_test_all_models_v6.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(logs).to_csv(out_paths["metadata"] / "model_training_log_v6.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(calibration_rows).to_csv(out_paths["metadata"] / "conformal_calibration_adjustments_v6.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(gate_summary_rows).to_csv(out_paths["metadata"] / "source_gate_coefficient_summary_v6.csv", index=False, encoding="utf-8-sig")
    return pred, logs, calibration_rows


def write_readme(root: Path, out_paths: dict[str, Path], tables: dict[str, pd.DataFrame], fast: bool, scope: str) -> None:
    macro = tables["macro"]
    common_macro = tables.get("common_macro", pd.DataFrame())
    national_macro = tables.get("common_national_macro", pd.DataFrame())
    regional_macro = tables.get("common_regional_macro", pd.DataFrame())
    bootstrap = tables.get("bootstrap", pd.DataFrame())
    bootstrap_national = tables.get("bootstrap_national", pd.DataFrame())
    primary_bootstrap = bootstrap_national if not bootstrap_national.empty else bootstrap
    external_national = national_macro[(national_macro["evaluation"] == "temporal_external_2025_2026") & (national_macro["official_delay_weeks"] == 0)].copy() if not national_macro.empty else pd.DataFrame()
    rolling_national = national_macro[(national_macro["evaluation"] == ROLLING_EVALUATION) & (national_macro["official_delay_weeks"] == 0)].copy() if not national_macro.empty else pd.DataFrame()
    regional_external = regional_macro[(regional_macro["evaluation"] == "temporal_external_2025_2026") & (regional_macro["official_delay_weeks"] == 0)].copy() if not regional_macro.empty else pd.DataFrame()
    mixed_external = common_macro[(common_macro["evaluation"] == "temporal_external_2025_2026") & (common_macro["official_delay_weeks"] == 0)].copy() if not common_macro.empty else pd.DataFrame()
    external = external_national if not external_national.empty else macro[(macro["evaluation"] == "temporal_external_2025_2026") & (macro["official_delay_weeks"] == 0)].copy()
    key_models = [
        "persistence",
        "ridge_no_digital",
        "ridge_full_raw",
        "ridge_raw_cn_search",
        "gb_full_raw",
        NO_DIGITAL_LSTM_MODEL,
        FULL_LSTM_MODEL,
    ]
    key = external[external["model"].isin(key_models)].sort_values(["model", "horizon"])
    key_md = key[
        ["model", "horizon", "n_countries", "WIS", "Pearson", "coverage_80", "coverage_95", "relative_WIS_vs_persistence", "delta_WIS_vs_sgq_full"]
    ].to_markdown(index=False, floatfmt=".3f") if not key.empty else "(no key external common-sample rows)"

    best = external.loc[external.groupby("horizon")["WIS"].idxmin()] if not external.empty else pd.DataFrame()
    best_md = best[["horizon", "model", "WIS", "relative_WIS_vs_persistence"]].sort_values("horizon").to_markdown(index=False, floatfmt=".3f") if not best.empty else "(no best-model rows)"

    rolling = rolling_national if not rolling_national.empty else macro[(macro["evaluation"] == ROLLING_EVALUATION) & (macro["official_delay_weeks"] == 0)].copy()
    rolling_md = rolling[rolling["model"].isin(["persistence", "ridge_raw_cn_search", NO_DIGITAL_LSTM_MODEL, FULL_LSTM_MODEL])][
        ["model", "horizon", "n", "n_countries", "WIS", "Pearson", "coverage_80", "coverage_95", "relative_WIS_vs_persistence", "delta_WIS_vs_sgq_full"]
    ].sort_values(["model", "horizon"]).to_markdown(index=False, floatfmt=".3f") if not rolling.empty else "(no rolling-window common-sample rows)"

    ablation_models = [
        FULL_LSTM_MODEL,
        NO_DIGITAL_LSTM_MODEL,
        "bioepi_sgq_lstm_no_china_local_search",
        "bioepi_sgq_lstm_no_google_wiki",
        "bioepi_sgq_lstm_no_social_media",
        SURVEILLANCE_HISTORY_MODEL,
        DIGITAL_DISEASE_CORE_MODEL,
        NO_REGION_INDICATORS_MODEL,
        "bioepi_sgq_lstm_no_source_gates",
        "bioepi_sgq_lstm_no_conformal",
    ]
    ablation_md = external[external["model"].isin(ablation_models)][
        ["model", "horizon", "WIS", "coverage_80", "coverage_95", "relative_WIS_vs_persistence", "delta_WIS_vs_sgq_full"]
    ].sort_values(["model", "horizon"]).to_markdown(index=False, floatfmt=".3f") if not external.empty else "(no strict LSTM ablation rows)"

    delay = macro[(macro["evaluation"] == "temporal_external_2025_2026") & (macro["model"] == FULL_LSTM_MODEL)].copy()
    delay_md = delay[
        ["official_delay_weeks", "horizon", "WIS", "coverage_80", "relative_WIS_vs_persistence"]
    ].sort_values(["official_delay_weeks", "horizon"]).to_markdown(index=False, floatfmt=".3f") if not delay.empty else "(no delay rows)"

    regional_md = regional_external[regional_external["model"].isin(key_models)][
        ["model", "horizon", "n", "n_countries", "WIS", "coverage_80", "coverage_95", "relative_WIS_vs_persistence", "delta_WIS_vs_sgq_full"]
    ].sort_values(["model", "horizon"]).to_markdown(index=False, floatfmt=".3f") if not regional_external.empty else "(no regional robustness rows)"

    mixed_md = mixed_external[mixed_external["model"].isin(key_models)][
        ["model", "horizon", "n", "n_countries", "WIS", "coverage_80", "coverage_95", "relative_WIS_vs_persistence", "delta_WIS_vs_sgq_full"]
    ].sort_values(["model", "horizon"]).to_markdown(index=False, floatfmt=".3f") if not mixed_external.empty else "(no mixed national+regional common-sample rows)"

    common_ext = external_national
    common_md = common_ext[common_ext["model"].isin(key_models)][
        ["model", "horizon", "n", "WIS", "coverage_80", "relative_WIS_vs_persistence", "delta_WIS_vs_sgq_full"]
    ].sort_values(["model", "horizon"]).to_markdown(index=False, floatfmt=".3f") if not common_ext.empty else "(no common-sample rows)"

    boot_md = "(no bootstrap rows)"
    if not primary_bootstrap.empty:
        b = primary_bootstrap[
            (primary_bootstrap["official_delay_weeks"] == 0)
            & (primary_bootstrap["country_scope"] == "macro")
            & (primary_bootstrap["model_a"].isin([FULL_LSTM_MODEL, "ridge_raw_cn_search"]))
            & (
                primary_bootstrap["model_b"].isin(
                    ["persistence", "ridge_raw_cn_search", "ridge_no_digital", NO_DIGITAL_LSTM_MODEL, "bioepi_sgq_lstm_no_source_gates"]
                )
            )
        ].copy()
        if not b.empty:
            boot_md = b[["horizon", "model_a", "model_b", "diff_WIS_a_minus_b", "ci025", "ci975", "n_pairs"]].sort_values(["model_a", "model_b", "horizon"]).to_markdown(index=False, floatfmt=".4f")

    gate_boot_md = "(no source-gate bootstrap rows)"
    if not primary_bootstrap.empty:
        gb = primary_bootstrap[
            (primary_bootstrap["official_delay_weeks"] == 0)
            & (primary_bootstrap["country_scope"] == "macro")
            & (primary_bootstrap["model_a"] == FULL_LSTM_MODEL)
            & (primary_bootstrap["model_b"] == "bioepi_sgq_lstm_no_source_gates")
        ].copy()
        if not gb.empty:
            gate_boot_md = gb[["horizon", "diff_WIS_a_minus_b", "ci025", "ci975", "n_boot", "n_pairs"]].sort_values("horizon").to_markdown(index=False, floatfmt=".4f")

    cn_delay2_md = "(no China delay-2 ridge increment bootstrap rows)"
    if not primary_bootstrap.empty:
        cn = primary_bootstrap[
            (primary_bootstrap["official_delay_weeks"] == 2)
            & (primary_bootstrap["country_scope"] == "CHN")
            & (primary_bootstrap["model_a"] == "ridge_raw_cn_search")
            & (primary_bootstrap["model_b"] == "ridge_no_digital")
        ].copy()
        if not cn.empty:
            cn_delay2_md = cn[["horizon", "diff_WIS_a_minus_b", "ci025", "ci975", "n_boot", "n_pairs"]].sort_values("horizon").to_markdown(index=False, floatfmt=".4f")

    lines = [
        "# BioEpi final revision v6 results package",
        "",
        "This v6 package implements the final pre-submission fixes requested after v5: purged multi-horizon selection/calibration splitting, national-only headline reporting, regional robustness tables, corrected `no_region_indicators` naming, platform-aware source-gate summaries, and paired bootstrap for the source-gate ablation.",
        "",
        "## What changed from v5",
        "",
        "- Splits the multi-horizon LSTM validation period with a purge: model selection uses rows whose h=4 target week is on/before the cut, while conformal calibration uses rows whose h=1 target week is after the cut.",
        "- Reports national-only common-sample macro metrics as the headline results for the three countries; regional rows are now a robustness analysis.",
        "- Renames `bioepi_sgq_lstm_no_region` to `bioepi_sgq_lstm_no_region_indicators`, because the model still uses separate regional sequences, scaling, and Mondrian conformal groups.",
        "- Adds national-only paired country-time-block bootstrap, including `full_raw` versus `no_source_gates`.",
        "- Source-gate output is now described as source-specific gating coefficients. Platform-unavailable groups and `other_digital` are kept as raw audit coefficients but masked from interpretation.",
        "- Uses a formal LSTM profile when `--fast` is not set: 5 random seeds, 3 hyperparameter configurations, 80-100 maximum epochs, and top-5 quantile ensemble.",
        "- The YAML grid is not exhaustively executed; this script executes the fixed 3-configuration formal grid above.",
        "",
        "## Important interpretation boundary",
        "",
        "The 2025-2026 period is reported as a temporal external evaluation, not as an untouched final test after model development, because earlier outputs from that period have already been inspected. V6 avoids further external-period tuning and changes only the validation split/reporting design requested after v5.",
        "",
        "This run still does not add unvalidated new climate, COVID, RSV, holiday, or US regional digital feeds. Google/Wikipedia composites are inherited processed inputs, so claims about fully fold-specific raw preprocessing should be limited to raw local/social signal columns unless raw Google/Wiki exports are rebuilt.",
        "",
        f"Run mode: `fast={fast}`, `scope={scope}`. Non-fast means the v6 formal LSTM grid.",
        "",
        "## Key temporal external metrics, national-only common-sample macro",
        "",
        key_md,
        "",
        "## Best external model by horizon",
        "",
        best_md,
        "",
        "## Rolling-window CV metrics, national-only common-sample macro",
        "",
        rolling_md,
        "",
        "## How to read the v6 result",
        "",
        "The strongest manuscript claim should be about region-aware probabilistic forecasting and conditional digital-signal value, not universal superiority of all-platform fusion. The full SGQ model outperforms persistence, but pooled no-delay digital increments are small and not consistently superior to no-digital LSTM.",
        "",
        "Prediction intervals should still be described cautiously: validation/Mondrian conformal calibration is implemented, but external temporal distribution shift can leave intervals under-dispersed.",
        "",
        "## Strict LSTM ablations, external delay 0",
        "",
        "Negative `delta_WIS_vs_sgq_full` means the ablation has lower WIS than the full SGQ model. Treat these strict ablations as exploratory unless a paired CI is provided; full digital fusion is not uniformly best.",
        "",
        ablation_md,
        "",
        "## National-only common-sample temporal external metrics",
        "",
        common_md,
        "",
        "## Regional robustness common-sample temporal external metrics",
        "",
        regional_md,
        "",
        "## Mixed national+regional common-sample metrics for audit only",
        "",
        mixed_md,
        "",
        "## Paired 4-week block bootstrap on national-only common sample",
        "",
        "Negative `diff_WIS_a_minus_b` means model_a has lower WIS than model_b.",
        "",
        boot_md,
        "",
        "## Source-gate ablation bootstrap",
        "",
        "These rows compare the full source-gated LSTM against the same architecture without source gates.",
        "",
        gate_boot_md,
        "",
        "## China delay-2 local-search increment bootstrap",
        "",
        "These rows directly compare `ridge_raw_cn_search` against the same Ridge family without digital inputs. Negative values favor China local-search features.",
        "",
        cn_delay2_md,
        "",
        "## BioEpi-SGQ-LSTM full_raw delay scenario",
        "",
        delay_md,
        "",
        "## Main files",
        "",
        "- `scripts/run_base_expert_models_v6.py`: reproducible experiment script.",
        "- `data/processed/forecast_design_origin_wide_v6.csv`: origin-week wide table used by the multi-horizon model.",
        "- `results/tables/predictions_calib_test_all_models_v6.csv`: purged-selection/calibration and test quantile predictions.",
        "- `results/tables/metrics_common_sample_national_only_macro.csv`: preferred headline metrics.",
        "- `results/tables/metrics_common_sample_regional_robustness_macro.csv`: regional robustness metrics.",
        "- `results/tables/metrics_common_sample_macro_country_balanced.csv`: mixed national+regional audit metrics.",
        "- `results/tables/paired_block_bootstrap_external_common_sample_national_only.csv`: primary paired country-time-block bootstrap CIs.",
        "- `results/tables/paired_block_bootstrap_external_common_sample.csv`: mixed national+regional bootstrap CIs for audit.",
        "- `results/metadata/source_gate_coefficient_summary_v6.csv`: platform-aware source-gate coefficients by country, region, and origin period; gates are not horizon-specific, so `horizon` is recorded as `all`.",
        "- `results/tables/metrics_summary_row_weighted.csv`: row-weighted metrics for comparison with v1.",
        "- `results/tables/event_metrics_corrected_national_only.csv`: main-text event-warning metrics.",
        "- `results/figures/`: regenerated manuscript-ready figures.",
    ]
    (root / "README_V6_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def copy_run_helpers(root: Path) -> None:
    ps1 = root / "run_v6_formal_all.ps1"
    ps1.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                "$root = Split-Path -Parent $MyInvocation.MyCommand.Path",
                "$py = 'python'",
                "if (!(Test-Path $py)) { $py = 'python' }",
                "& $py \"$root\\scripts\\run_base_expert_models_v6.py\" --scope all --root $root",
            ]
        ),
        encoding="utf-8",
    )
    fast = root / "run_v6_fast_smoke.ps1"
    fast.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                "$root = Split-Path -Parent $MyInvocation.MyCommand.Path",
                "$py = 'python'",
                "if (!(Test-Path $py)) { $py = 'python' }",
                "& $py \"$root\\scripts\\run_base_expert_models_v6.py\" --fast --scope external --root $root",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fast", action="store_true", help="Use a small smoke-test grid instead of the v6 formal grid.")
    parser.add_argument("--scope", choices=["external", "rolling", "all"], default="all")
    parser.add_argument("--postprocess-only", action="store_true", help="Reuse existing predictions and rebuild tables, figures, and README.")
    args = parser.parse_args()
    root = args.root.resolve()
    out_paths = ensure_dirs(root)
    copy_run_helpers(root)
    pred_path = out_paths["tables"] / "predictions_calib_test_all_models_v6.csv"
    if args.postprocess_only:
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        pred = pd.read_csv(pred_path, parse_dates=["origin_week_start", "target_week_start", "train_start", "train_end", "val_start", "val_end", "test_start", "test_end"])
    else:
        pred, _, _ = run_experiments(root, args.fast, args.scope)
    tables = make_tables(pred, out_paths)
    compute_event_metrics(pred, out_paths)
    make_figures(root, pred, tables, out_paths)
    write_readme(root, out_paths, tables, args.fast, args.scope)
    manifest = {
        "created_by": "Codex",
        "script": str((root / "scripts" / "run_base_expert_models_v6.py").resolve()),
        "fast": args.fast,
        "scope": args.scope,
        "outputs": {
            "predictions": "results/tables/predictions_calib_test_all_models_v6.csv",
            "common_sample_national_only_metrics": "results/tables/metrics_common_sample_national_only_macro.csv",
            "common_sample_regional_robustness_metrics": "results/tables/metrics_common_sample_regional_robustness_macro.csv",
            "common_sample_mixed_audit_metrics": "results/tables/metrics_common_sample_macro_country_balanced.csv",
            "paired_bootstrap": "results/tables/paired_block_bootstrap_external_common_sample.csv",
            "paired_bootstrap_national_only": "results/tables/paired_block_bootstrap_external_common_sample_national_only.csv",
            "macro_metrics": "results/tables/metrics_macro_country_balanced.csv",
            "figures": "results/figures",
        },
    }
    (out_paths["metadata"] / "manifest_v6.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] v6 outputs written to {root}", flush=True)


if __name__ == "__main__":
    main()
