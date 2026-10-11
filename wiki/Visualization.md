# Visualization

All plots are produced by the modules under `src/eruption_forecast/plots/`.
The pipeline auto-renders the most useful plots; you can also invoke each helper directly for ad-hoc figures.

| Module | Re-exported as | Used by |
|--------|----------------|---------|
| `plots/styles.py` | `apply_nature_style`, `setup_nature_style`, `configure_spine`, `get_color`, `get_figure_size`, `NATURE_COLORS`, `OKABE_ITO` | Internal styling for every figure |
| `plots/tremor_plots.py` | `plot_tremor`, `plot_tremor_from_file` | `CalculateTremor` daily + summary plots |
| `plots/feature_plots.py` | `plot_significant_features`, `replot_significant_features`, `plot_frequency_band_contribution`, `plot_common_features_heatmap`, `plot_common_features_correlation`, `plot_features_matrix` | `TrainingModel.fit()` when `plot_features=True`; cross-scenario analysis; ad-hoc feature time-series |
| `plots/forecast_plots.py` | `plot_forecast`, `plot_forecast_from_file` | `PredictionModel.forecast()` |
| `plots/evaluation_plots.py` | `plot_roc_curve`, `plot_precision_recall_curve`, `plot_confusion_matrix`, `plot_threshold_analysis`, `plot_g_mean_curve`, `plot_mcc_curve`, aggregate counterparts (`plot_aggregate_*`), `render_one_plot`, `render_one_aggregate_plot` | `EvaluationModel.evaluate(plot_aggregate=…, plot_per_seed=…)` |
| `plots/explanation_plots.py` | `plot_shap_waterfall`, `plot_shap_beeswarm`, `plot_shap_bar`, `plot_aggregate_shap_bar`, `plot_aggregate_shap_beeswarm`, `plot_classifier_waterfall`, `render_seed_plot` | `ExplanationModel.explain()` + `ExplainerEnsemble.plot_seed/plot_waterfall` |
| `plots/label_plots.py` | `plot_label_distribution`, `plot_label_distribution_from_file`, `plot_label_distribution_comparison`, `plot_label_distribution_comparison_from_files` | `LabelBuilder.plot_distribution()`; cross-scenario label class comparisons |

The most common helpers are re-exported from `eruption_forecast.plots`: the styling helpers, `plot_tremor` / `plot_tremor_from_file`, the five feature heatmap and importance plots, `plot_forecast` / `plot_forecast_from_file`, `plot_roc_curve` / `plot_precision_recall_curve` / `plot_confusion_matrix` / `plot_threshold_analysis` / `plot_feature_importance`, and `plot_label_distribution`:

```python
from eruption_forecast.plots import (
    plot_tremor, plot_forecast, plot_significant_features,
    plot_roc_curve, plot_label_distribution, apply_nature_style,
)
```

Everything else (`plot_features_matrix`, the `plot_aggregate_*` evaluation curves, the SHAP helpers, and the `*_comparison*` label plots) is imported from its module directly, e.g. `from eruption_forecast.plots.feature_plots import plot_features_matrix`.

---

## Tremor - `plot_tremor`

Multi-panel tremor plot with one subplot per detected method (RSAM / DSAR / entropy). By default each series shows a rolling median and a rolling mean; `metrics="median" | "mean" | "all"` restricts that.

```python
from eruption_forecast.plots import plot_tremor, plot_tremor_from_file

fig = plot_tremor(
    tremor_df,
    interval=7, interval_unit="days",          # x-axis tick spacing
    rolling_window="1D",
    eruption_dates=["2025-03-20"],
    selected_columns=["rsam_f2", "dsar_f3-f4", "entropy"],
    filter_dsar_value=10, filter_rsam_value=None, filter_entropy_value=None,  # per-method upper bounds
    rsam_as_log=True,
    colors=None,                               # str | list[str] colour override
    filepath="tremor.png",                     # None → return without saving
)

fig = plot_tremor_from_file("output/VG.OJN.00.EHZ/tremor/VG.OJN.00.EHZ_2025-01-01_2025-08-31.csv", interval=7, interval_unit="days")
```

Auto-rendered by `CalculateTremor` / `fm.calculate(...)`:

```
{station_dir}/tremor/figures/{YYYY-MM-DD}.png               # plot_daily=True — one per day
{station_dir}/tremor/tremor-interpolated_{nslc}_{start}-{end}.{png,pdf}  # save_plot=True — merged summary figure
```

The summary figure honours the `plot_eruption_dates`, `plot_rsam_as_log`, `plot_rolling_window`, and `plot_filter_dsar_value` arguments of `fm.calculate(...)` / `CalculateTremor.run(...)`. `CalculateTremor.run(plot_interval=2)` controls its tick spacing in days. `fm.calculate()` does not expose `plot_interval` yet.

---

## Features - `plot_significant_features` & friends

| Function | Purpose | Output |
|----------|---------|--------|
| `plot_significant_features(df, filepath, top_features, values_column)` | Horizontal bar chart of top-N selected features per seed | `{features_dir}/seed/figures/{seed:05d}.png` |
| `replot_significant_features(...)` | Same chart re-rendered from a saved features CSV | ad-hoc |
| `plot_frequency_band_contribution(df, filepath)` | Bar chart of feature counts per seismic band | `{features_dir}/frequency_band_contribution.png` |
| `plot_common_features_heatmap(top_features_csv, ..., label_style, show_colorbar, max_features)` | Cross-scenario heatmap of shared top features; `max_features` keeps the top-N by total frequency, `show_colorbar=False` hides the bar | `output_path` (default `{cwd}/common_top_features_heatmap.png`) |
| `plot_common_features_correlation(top_features_csv, ...)` | Cross-scenario correlation of shared features | `output_path` |
| `plot_features_matrix(features_path, label_csv, figures_dir, ...)` | One PNG per feature column: raw 10-minute scatter + 12 h / 48 h centred rolling medians, with eruption markers | `{figures_dir}/{column_index}_{slugified_feature}.png` |

Auto-rendered by `fm.train(..., plot_features=True)` (the first three). The cross-scenario heatmaps and `plot_features_matrix` are ad-hoc.

```python
from eruption_forecast.plots.feature_plots import plot_features_matrix

plot_features_matrix(
    features_path="output/.../training/features/stratified-shuffle-split/features-matrix_2025-01-03_2025-03-31.parquet",
    label_csv="output/.../training/features/stratified-shuffle-split/features-label_2025-01-03_2025-03-31.csv",
    figures_dir="output/.../training/features/stratified-shuffle-split/figures/matrix",
    eruption_dates=["2025-03-20"],                       # only dates inside the matrix range are drawn
    select_features="output/.../top_20_features.csv",    # CSV path or list[str]; None plots every column
    number_of_features=10,                               # cap when select_features is a CSV
    figsize=(8, 2), dpi=150, overwrite=False,            # existing PNGs are skipped unless overwrite=True
)
```

---

## Forecast - `plot_forecast`

Three-panel forecast figure consumed by `PredictionModel.forecast()`:

```
┌────────────────────────────────────────────────────┐
│  Panel 1   Consensus max-envelope prediction       │
│            + probability, with threshold line       │
├────────────────────────────────────────────────────┤
│  Panel 2   Per-classifier predictions overlaid     │
│            with the consensus envelope             │
├────────────────────────────────────────────────────┤
│  Panel 3   Per-classifier probabilities overlaid  │
│            with the consensus envelope             │
└────────────────────────────────────────────────────┘
```

Auto-rendered by `fm.predict(plot_threshold=0.7, plot_pdf=True, eruption_dates=[...])`:

```
{station_dir}/prediction/figures/forecast_{basename}.png
{station_dir}/prediction/figures/forecast_{basename}.pdf
```

Key kwargs forwarded via `**plot_kwargs` from `fm.predict(...)`:

| Param | Effect |
|-------|--------|
| `threshold` (`plot_threshold` in `predict`) | Horizontal dashed reference line on every panel |
| `eruption_dates` | Vertical dashed lines on each ground-truth eruption |
| `rolling_window="6h"` | Pandas rolling window applied before plotting (smoothing) |
| `x_days_interval=2` | Major x-tick spacing in days |
| `legend_n_cols=6`, `bbox_to_anchor=(0.5, -0.05)` | Legend positioning |
| `title="..."` | Figure suptitle |
| `start_date="YYYY-MM-DD"`, `end_date="YYYY-MM-DD"` | Optional window slice applied to `df` before envelope extraction and rolling smoothing. `None` on either bound keeps `df.index.min()` / `df.index.max()`. An empty resulting slice logs a warning and falls back to the full range. Also narrows the top segment strip's `forecast_start` / `forecast_end`. From `PredictionModel.forecast()` / `fm.predict()`, pass as the explicit `plot_start_date` / `plot_end_date` kwargs to avoid colliding with the forecast-period `start_date` / `end_date` on those callers. When both are set, the output file is renamed `forecast_{plot_start_date}_{plot_end_date}.{png,pdf}`. |

To re-render a forecast plot from the persisted CSV:

```python
from eruption_forecast.plots import plot_forecast_from_file
fig = plot_forecast_from_file(
    "output/VG.OJN.00.EHZ/forecast-results_2025-07-27_2025-08-22.csv",
    eruption_dates=["2025-08-02"],
)
fig.savefig("forecast.png", dpi=200, bbox_inches="tight")
```

---

## Evaluation - `plot_roc_curve` etc.

These are the aggregate plots `EvaluationModel.evaluate(plot_aggregate=True)` renders per classifier — each call also writes a `{plot_name}.csv` sidecar with the underlying mean / std data so the figure can be re-rendered offline:

| Function | Plot |
|----------|------|
| `plot_roc_curve` / `plot_aggregate_roc_curve` | Mean ROC + ± std band across seeds |
| `plot_precision_recall_curve` / `plot_aggregate_precision_recall_curve` | Mean PR + ± std band |
| `plot_confusion_matrix` | Summed confusion matrix (per-seed only) |
| `plot_threshold_analysis` / `plot_aggregate_threshold_analysis` | Precision, recall, F1, balanced accuracy, G-mean, MCC vs threshold — marks `ERUPTION_PROBABILITY_THRESHOLD` (`config/constants.py`) and the optimal G-mean / MCC thresholds |
| `plot_g_mean_curve` / `plot_aggregate_g_mean_curve` | G-mean vs threshold across seeds |
| `plot_mcc_curve` / `plot_aggregate_mcc_curve` | MCC vs threshold across seeds |

Auto-rendered at:

```
{station_dir}/evaluation/{kind}/classifiers/{Clf}/figures/aggregate/{plot_name}.{png,csv}
{station_dir}/evaluation/{kind}/classifiers/{Clf}/figures/{plot_name}/{seed:05d}.png   # plot_per_seed=True
```

`ClassifierComparator.plot_all()` adds cross-classifier figures under `evaluation/{kind}/comparison/figures/` - see [Evaluation Workflow → Cross-Classifier Comparison](Evaluation-Workflow#cross-classifier-comparison).

---

## Labels - `plot_label_distribution`

Debug-friendly bar plot showing the positive/negative class balance across the labelled window range:

```python
from eruption_forecast.plots import plot_label_distribution
fig = plot_label_distribution(label_df)
fig.savefig("labels.png", dpi=150)
```

Useful when tuning `window_step` and `day_to_forecast` - flips the imbalance immediately visible.

---

## SHAP — `plot_classifier_waterfall` etc.

SHAP plots are produced by the dedicated explanation stage (`ExplanationModel.explain()` → `ExplainerEnsemble.plot_seed/plot_waterfall`) and rendered through `plots/explanation_plots.py`. The legacy `EvaluationModel.evaluate(plot_shap=True)` hook is now a reserved no-op — it accepts the kwarg, logs a warning, and dispatches no figures.

| Function | Plot | Output |
|----------|------|--------|
| `plot_shap_bar(explanation, ...)` | One bar plot for one seed | caller-supplied `save_filepath` |
| `plot_shap_beeswarm(explanation, ...)` | One beeswarm for one seed | caller-supplied `save_filepath` |
| `plot_shap_waterfall(explanation, ...)` | One waterfall for one observation | caller-supplied `save_filepath` |
| `plot_aggregate_shap_bar(classifier_explanation, ...)` | Frequency-weighted aggregate bar across seeds | `(fig, df)` returned; `save_filepath` optional |
| `plot_aggregate_shap_beeswarm(classifier_explanation, ...)` | Stacked-seeds aggregate beeswarm | `(fig, tidy_df)` returned; `save_filepath` optional |
| `plot_classifier_waterfall(classifier_explanation, classifier_ensemble, labels, eruption_dates, ...)` | Per-eruption highest-probability waterfall | `{explanation_dir}/eruptions/{date}/{Clf}_*.png` |

Auto-rendered at:

```
{station_dir}/explanation/{kind}/classifiers/{Clf}/figures/{bar,beeswarm}/{seed:05d}.png    # plot_per_seed=True
{station_dir}/explanation/{kind}/classifiers/{Clf}/figures/aggregate/{bar,beeswarm}.{png,csv}  # plot_aggregate=True
{station_dir}/explanation/{kind}/eruptions/{date}/{Clf}_{datetime}_seed=_index=.png
```

The aggregate `bar.png` ranks features by frequency-weighted mean |SHAP| with `selection_frequency` annotated at the right edge; the aggregate `beeswarm.png` stacks every seed into the NaN-padded union feature space so a single figure summarises the whole ensemble. Each `.png` has a `.csv` sidecar (importance table for the bar, tidy long-form non-NaN cell list for the beeswarm) so the figures can be redrawn offline.

When invoking the SHAP helpers directly, always pass `plot_size=None` to `shap.plots.beeswarm` so SHAP does not override the pre-created `figsize` — the project's `shap_figure` context manager already sets it. See [Explanation Workflow](Explanation-Workflow) for the full surface.

---

## Styling - Nature-Style Defaults

Every figure goes through `apply_nature_style()` from `plots/styles.py`, which sets:

- Serif fonts and small-format figure sizes consistent with Nature/Science columns
- The Okabe–Ito palette by default; a sequential brewer palette for diverging signals
- `mpl.rc("pdf", fonttype=42)` so saved PDFs keep editable text
- `nature_figure(width_in, height_in)` helper for one-line publication-quality sizing

Override per call:

```python
import matplotlib.pyplot as plt
from eruption_forecast.plots import apply_nature_style

apply_nature_style()
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(df.index, df["consensus_eruption_probability"])
fig.savefig("custom.png", dpi=300, bbox_inches="tight")
```
