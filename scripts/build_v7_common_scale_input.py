#!/usr/bin/env python3
# BioEpi-SAFE-LSTM reproducibility code
# Maintainer: Jianyi Zhang

"""Create the compact v7 expert-prediction input from the v6 full prediction export.

The v7 fusion requires raw-unit quantiles because different expert families were fitted
with slightly different target scalers. The downstream v7 script transforms every expert
onto a single no-digital SGQ-LSTM reference scale before fusion and WIS comparison.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

QCOLS = ["q0.025", "q0.1", "q0.25", "q0.5", "q0.75", "q0.9", "q0.975"]
QRAW = [f"{q}_raw" for q in QCOLS]
MODELS = {
    "bioepi_sgq_lstm_full_raw", "bioepi_sgq_lstm_no_digital",
    "gb_full_raw", "persistence", "seasonal_naive",
    "ridge_global_open", "ridge_google_wiki", "ridge_no_digital",
    "ridge_raw_cn_search", "ridge_social_raw",
}
COLS = [
    "sample", "evaluation", "fold", "official_delay_weeks", "country", "region",
    "horizon", "model", "origin_week_start", "target_week_start", "period",
    "y_true", "target_raw", "scale_mu", "scale_sd", "event_threshold_z_train_q80",
    *sum(([q, qr] for q, qr in zip(QCOLS, QRAW)), []),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="v6 predictions_calib_test_all_models_v6.csv")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output .pkl.gz path")
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    parts = []
    for chunk in pd.read_csv(args.source, usecols=COLS, chunksize=args.chunksize):
        chunk = chunk[chunk["sample"].eq("test") & chunk["model"].isin(MODELS)].copy()
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        raise RuntimeError("No test predictions were found in the source file.")
    out = pd.concat(parts, ignore_index=True)

    required = ["target_raw", "scale_mu", "scale_sd", *QRAW]
    missing = [c for c in required if c not in out or out[c].isna().all()]
    if missing:
        raise ValueError(f"Missing required raw-scale fields: {missing}")
    if (out["scale_sd"] <= 0).any():
        raise ValueError("Non-positive target scale detected.")

    # Raw quantiles are authoritative for v7. They may differ from a direct inverse
    # standardization because v6 applied raw-scale non-negativity constraints after
    # inverse transformation (particularly important for Japanese case counts).
    max_transform_or_clipping_difference = 0.0
    for q, qr in zip(QCOLS, QRAW):
        reconstructed = out[q] * out["scale_sd"] + out["scale_mu"]
        max_transform_or_clipping_difference = max(
            max_transform_or_clipping_difference,
            float(np.nanmax(np.abs(reconstructed - out[qr]))),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(args.output, compression="gzip")
    print({
        "rows": len(out),
        "models": int(out["model"].nunique()),
        "evaluations": sorted(out["evaluation"].dropna().unique().tolist()),
        "max_transform_or_nonnegativity_clipping_difference": max_transform_or_clipping_difference,
        "output": str(args.output),
    })


if __name__ == "__main__":
    main()
