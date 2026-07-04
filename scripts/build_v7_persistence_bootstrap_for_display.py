#!/usr/bin/env python3
# BioEpi-SAFE-LSTM reproducibility code
# Maintainer: Jianyi Zhang

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


QCOLS = ["q0.025", "q0.1", "q0.25", "q0.5", "q0.75", "q0.9", "q0.975"]
QRAW = [c + "_raw" for c in QCOLS]
BASE = "bioepi_sgq_lstm_no_digital"
FULL = "bioepi_sgq_lstm_full_raw"
SAFE = "bioepi_safe_lstm_common_scale"
PERSIST = "persistence"
GB = "gb_full_raw"
SEASONAL = "seasonal_naive"
MODELS = [SAFE, BASE, FULL, GB, SEASONAL]
LABELS = {
    SAFE: "BioEpi-SAFE-LSTM",
    BASE: "Surveillance SGQ-LSTM",
    FULL: "Full digital SGQ-LSTM",
    GB: "Digital gradient boosting",
    SEASONAL: "Seasonal naive",
}


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    return (upper - lower) + (2.0 / alpha) * (lower - y) * (y < lower) + (2.0 / alpha) * (y - upper) * (y > upper)


def wis(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    q = np.asarray(q, float)
    out = 0.5 * np.abs(y - q[:, 3])
    for alpha, lower, upper in [(0.50, 2, 4), (0.20, 1, 5), (0.05, 0, 6)]:
        out += (alpha / 2.0) * interval_score(y, q[:, lower], q[:, upper], alpha)
    return out / 3.5


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["WIS"] = wis(out["y_true"].to_numpy(float), out[QCOLS].to_numpy(float))
    return out


def common_scale_expert_rows(raw: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    keys = [
        "evaluation",
        "fold",
        "official_delay_weeks",
        "country",
        "region",
        "horizon",
        "origin_week_start",
        "target_week_start",
    ]
    test = raw[raw["sample"] == "test"].copy()
    base_scale = (
        test[test["model"] == BASE][keys + ["scale_mu", "scale_sd", "event_threshold_z_train_q80"]]
        .drop_duplicates(keys)
        .rename(columns={"scale_mu": "common_mu", "scale_sd": "common_sd", "event_threshold_z_train_q80": "common_event_threshold"})
    )
    test = test.merge(base_scale, on=keys, how="inner", validate="many_to_one")
    test = test[test["model"].isin(models + [PERSIST])].copy()
    test["y_true"] = (test["target_raw"] - test["common_mu"]) / test["common_sd"]
    for c, raw_c in zip(QCOLS, QRAW):
        test[c] = (test[raw_c] - test["common_mu"]) / test["common_sd"]
    return add_scores(test)


def bootstrap_pair(pair: pd.DataFrame, block_weeks: int, reps: int, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    obs = float(pair.groupby("country")["diff"].mean().mean())
    country_boot = []
    for _, g in pair.groupby("country", sort=False):
        d = g.sort_values("origin_week_start")["diff"].to_numpy(float)
        n = len(d)
        if n == 0:
            continue
        n_blocks = int(math.ceil(n / block_weeks))
        max_start = max(1, n - block_weeks + 1)
        starts = rng.integers(0, max_start, size=(reps, n_blocks))
        idx = (starts[:, :, None] + np.arange(block_weeks)[None, None, :]).reshape(reps, -1)[:, :n]
        idx = np.minimum(idx, n - 1)
        country_boot.append(d[idx].mean(axis=1))
    boot = np.mean(np.column_stack(country_boot), axis=1)
    return {
        "diff_WIS_model_minus_persistence": obs,
        "ci_low_diff": float(np.quantile(boot, 0.025)),
        "ci_high_diff": float(np.quantile(boot, 0.975)),
        "prob_model_better": float(np.mean(boot < 0)),
        "n_pairs": int(len(pair)),
    }


def country_balanced_wis(rows: pd.DataFrame, model: str, horizon: int) -> float:
    z = rows[(rows["model"] == model) & (rows["horizon"] == horizon)]
    return float(z.groupby("country")["WIS"].mean().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--block-weeks", type=int, default=8)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = root / "results" / "tables" / "npj_display"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_pickle(root / "data" / "expert_predictions_v6_common_scale_test.pkl.gz")
    raw["origin_week_start"] = pd.to_datetime(raw["origin_week_start"])
    raw["target_week_start"] = pd.to_datetime(raw["target_week_start"])
    experts = common_scale_expert_rows(raw, [BASE, FULL, GB, SEASONAL])
    experts = experts[
        (experts["evaluation"] == "temporal_external_2025_2026")
        & (experts["official_delay_weeks"] == 0)
        & (experts["region"] == "national")
    ].copy()
    keep_cols = [
        "evaluation",
        "official_delay_weeks",
        "country",
        "region",
        "horizon",
        "origin_week_start",
        "target_week_start",
        "model",
        "WIS",
    ]
    experts = experts[keep_cols].copy()
    safe = pd.read_csv(root / "results" / "tables" / "v7_common_scale_predictions.csv.gz", parse_dates=["origin_week_start", "target_week_start"])
    safe = safe[
        (safe["evaluation"] == "temporal_external_2025_2026")
        & (safe["official_delay_weeks"] == 0)
        & (safe["region"] == "national")
        & (safe["model"] == SAFE)
    ][keep_cols].copy()
    rows = pd.concat([safe, experts], ignore_index=True)
    key = ["country", "origin_week_start", "target_week_start", "horizon"]
    persist = rows[rows["model"] == PERSIST][key + ["WIS"]].rename(columns={"WIS": "persistence_WIS"})
    output_rows = []
    for h in [1, 2, 3, 4]:
        persistence_wis = country_balanced_wis(rows, PERSIST, h)
        for model in MODELS:
            model_wis = country_balanced_wis(rows, model, h)
            z = rows[(rows["model"] == model) & (rows["horizon"] == h)][key + ["WIS"]].rename(columns={"WIS": "model_WIS"})
            pair = z.merge(persist[persist["horizon"] == h], on=key, how="inner")
            pair["diff"] = pair["model_WIS"] - pair["persistence_WIS"]
            boot = bootstrap_pair(pair, args.block_weeks, args.reps, seed=20260629 + h * 100 + MODELS.index(model))
            diff = boot["diff_WIS_model_minus_persistence"]
            ci_low = boot["ci_low_diff"]
            ci_high = boot["ci_high_diff"]
            output_rows.append(
                {
                    "evaluation": "temporal_external_2025_2026",
                    "official_delay_weeks": 0,
                    "horizon": h,
                    "model": model,
                    "model_label": LABELS[model],
                    "model_WIS": model_wis,
                    "persistence_WIS": persistence_wis,
                    "block_weeks": args.block_weeks,
                    **boot,
                    "WIS_reduction_vs_persistence_pct": -100.0 * diff / persistence_wis,
                    "ci_low_reduction_pct": -100.0 * ci_high / persistence_wis,
                    "ci_high_reduction_pct": -100.0 * ci_low / persistence_wis,
                }
            )
    out = pd.DataFrame(output_rows)
    out.to_csv(out_dir / "figure3_models_vs_persistence_bootstrap.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
