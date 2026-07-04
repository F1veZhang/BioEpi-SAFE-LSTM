# Data dictionary

## Core identifiers

- `country`: CHN, JPN or USA.
- `region`: national, China North/South or U.S. HHS region.
- `origin_week_start`: forecast origin week start date.
- `target_week_start`: target week start date.
- `horizon`: forecast horizon in weeks, 1–4.
- `model`: forecasting model or expert label.

## Forecast columns

- `q_0.025`, `q_0.10`, `q_0.25`, `q_0.50`, `q_0.75`, `q_0.90`, `q_0.975`: predictive quantiles where available.
- `WIS`: weighted interval score.
- `MAE`: mean absolute error.
- `RMSE`: root mean squared error.
- `Pearson`: Pearson correlation between median forecasts and observed activity.
- `coverage_50`, `coverage_80`, `coverage_95`: empirical interval coverage.

## SAFE fusion columns

- `expert`: auxiliary or backbone expert name.
- `weight`: ensemble weight.
- `official_delay_weeks`: simulated official-reporting delay.
- `WIS_reduction_pct`: WIS reduction relative to the comparator specified in the table.
