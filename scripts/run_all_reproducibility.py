#!/usr/bin/env python3
# BioEpi-SAFE-LSTM reproducibility code
# Maintainer: Jianyi Zhang

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BioEpi-SAFE-LSTM reproducibility workflow.")
    parser.add_argument("--root", default=".", help="Repository root directory.")
    parser.add_argument("--bootstrap-reps", type=int, default=2000, help="Bootstrap replicates for final analyses.")
    parser.add_argument("--skip-safe-fusion", action="store_true", help="Skip SAFE fusion and reuse existing outputs.")
    parser.add_argument("--skip-season-eval", action="store_true", help="Skip season-level evaluation and reuse existing outputs.")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure regeneration.")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    python = sys.executable

    if not args.skip_safe_fusion:
        run_command([
            python, "scripts/run_v7_common_scale_safe.py",
            "--root", str(root),
            "--input", str(root / "data/processed/expert_predictions_v6_common_scale_test.pkl.gz"),
            "--bootstrap-reps", str(args.bootstrap_reps),
        ], root)

    if not args.skip_season_eval:
        run_command([
            python, "scripts/run_v8_season_level_evaluation.py",
            "--root", str(root),
            "--input", str(root / "results/tables/v7_common_scale_predictions.csv.gz"),
            "--bootstrap-reps", str(args.bootstrap_reps),
        ], root)

    if not args.skip_figures:
        run_command([python, "scripts/make_v7_npj_display_figures.py", "--root", str(root)], root)

    print("Workflow completed.")


if __name__ == "__main__":
    main()
