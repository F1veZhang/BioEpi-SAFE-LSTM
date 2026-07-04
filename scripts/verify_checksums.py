#!/usr/bin/env python3
# BioEpi-SAFE-LSTM reproducibility code
# Maintainer: Jianyi Zhang

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify repository checksums.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--checksum-file", default="CHECKSUMS.sha256")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    checksum_path = root / args.checksum_file
    if not checksum_path.exists():
        raise FileNotFoundError(checksum_path)
    failed = []
    checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, rel = line.split(maxsplit=1)
        rel = rel.strip().lstrip("*")
        path = root / rel
        if not path.exists():
            failed.append((rel, "missing"))
            continue
        actual = sha256_file(path)
        checked += 1
        if actual != digest:
            failed.append((rel, "checksum mismatch"))
    if failed:
        print("Checksum verification failed:")
        for rel, reason in failed:
            print(f"  {rel}: {reason}")
        raise SystemExit(1)
    print(f"Checksum verification passed for {checked} files.")


if __name__ == "__main__":
    main()
