# Visualization

All plots are produced by the modules under `src/eruption_forecast/plots/`.
The pipeline auto-renders the most useful plots; you can also invoke each helper directly for ad-hoc figures.

| Module | Re-exported as | Used by |
|--------|----------------|---------|
| `plots/styles.py` | `apply_nature_style`, `setup_nature_style`, `configure_spine`, `get_color`, `get_figure_size`, `NATURE_COLORS`, `OKABE_ITO` | Internal styling for every figure |
| `plots/tremor_plots.py` | `plot_tremor` | `CalculateTremor` daily plots |
| `plots/feature_plots.py` | `plot_significant_features`, `replot_significant_features`, `plot_frequency_band_contribution`, `plot_common_features_heatmap`, `plot_common_features_correlation` | `TrainingModel.fit()` when `plot_features=True`; cross-scenario analysis |
| `plots/forecast_plots.py` | `plot_forecast`, `plot_forecast_from_file` | `PredictionModel.forecast()` |
| `plots/evaluation_plots.py` | `plot_roc_curve`, `plot_precision_recall_curve`, `plot_confusion_matrix`, `plot_threshold_analysis`, `plot_g_mean_curve`, `plot_mcc_curve`, aggregate counterparts (`plot_aggregate_*`), `render_one_plot`, `render_one_aggregate_plot` | `EvaluationModel.evaluate(plot_aggregate=…, plot_per_seed=…)` |
| `plots/explanation_plots.py` | `plot_shap_waterfall`, `plot_shap_beeswarm`, `plot_shap_bar`, `plot_aggregate_shap_bar`, `plot_aggregate_shap_beeswarm`, `plot_classifier_waterfall`, `render_seed_plot` | `ExplanationModel.explain()` + `ExplainerEnsemble.plot_seed/plot_waterfall` |
| `plots/label_plots.py` | `plot_label_distribution`, `plot_label_distribution_from_file`, `plot_label_distribution_comparison`, `plot_label_distribution_comparison_from_files` | `LabelBuilder.plot_distribution()`; cross-scenario label class comparisons |

All of these are re-exported from `eruption_forecast.plots`:

```python
from eruption_forecast.plots import (
    plot_tremor, plot_forecast, plot_significant_features,
    plot_roc_curve, plot_label_distribution, apply_nature_style,
)
```

---

## Tremor - `plot_tremor`

Multi-panel band-decomposed tremor plot. One panel per band/method (RSAM/DSAR/entropy).

```python
from eruption_forecast.plots import plot_tremor
fig = plot_tremor(tremor_df, methods=["rsam", "dsar", "entropy"])
```

Auto-rendered when `fm.calculate(plot_daily=True, save_plot=True)`:

```
{station_dir}/tremor/figures/{nslc}_{YYYY-MM-DD}.png
```

---

## Features - `plot_significant_features` & friends

| Function | Purpose | Output |
|----------|---------|--------|
| `plot_significant_features(df, filepath, top_features, values_column)` | Horizontal bar chart of top-N selected features per seed | `{features_dir}/seed/figures/{seed:05d}.png` |
| `replot_significant_features(...)` | Same chart re-rendered from a saved features CSV | ad-hoc |
| `plot_frequency_band_contribution(df, filepath)` | Bar chart of feature counts per seismic band | `{features_dir}/frequency_band_contribution.png` |

Auto-rendered by `fm.train(..., plot_features=True)`.

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
