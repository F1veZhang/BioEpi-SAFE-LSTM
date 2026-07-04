# Breathable npj-Style Display Figures for BioEpi-SAFE-LSTM v7

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
