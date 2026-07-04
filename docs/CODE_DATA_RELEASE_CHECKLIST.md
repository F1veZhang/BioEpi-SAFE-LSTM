# Code and data release checklist

- [ ] Remove local absolute paths from scripts and notebooks.
- [ ] Confirm no raw social-media posts, identifiers, profile metadata or post text are included.
- [ ] Confirm reviewer data folder access is read-only.
- [ ] Run `python scripts/verify_checksums.py --root .`.
- [ ] Run a smoke test with `--bootstrap-reps 50`.
- [ ] Check that main tables are present in `results/tables/v8_season_level/`.
- [ ] Check that figure files are present in `results/figures_v8/` and `results/figures_npj/`.
- [ ] Replace temporary Google Drive links with a persistent repository DOI before publication.
