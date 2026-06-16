# Visualization

All plots are produced by the modules under `src/eruption_forecast/plots/` plus `src/eruption_forecast/label/label_plots.py`. 
The pipeline auto-renders the most useful plots; you can also invoke each helper directly for ad-hoc figures.

| Module | Re-exported as | Used by |
|--------|----------------|---------|
| `plots/styles.py` | `apply_nature_style`, `setup_nature_style`, `configure_spine`, `get_color`, `get_figure_size`, `NATURE_COLORS`, `OKABE_ITO` | Internal styling for every figure |
| `plots/tremor_plots.py` | `plot_tremor` | `CalculateTremor` daily plots |
| `plots/feature_plots.py` | `plot_significant_features`, `replot_significant_features`, `plot_frequency_band_contribution` | `TrainingModel.fit()` when `plot_features=True` |
| `plots/forecast_plots.py` | `plot_forecast`, `plot_forecast_from_file` | `PredictionModel.forecast()` |
| `plots/evaluation_plots.py` | `plot_roc_curve`, `plot_precision_recall_curve`, `plot_confusion_matrix`, `plot_threshold_analysis`, `plot_feature_importance` | `EvaluationModel.evaluate(plot_aggregate=True)` |
| `label/label_plots.py` | `plot_label_distribution` | `LabelBuilder` debugging |

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

These are the aggregate plots `EvaluationModel.evaluate(plot_aggregate=True)` renders per classifier:

| Function | Plot |
|----------|------|
| `plot_roc_curve` | Mean ROC + ± std band across seeds |
| `plot_precision_recall_curve` | Mean PR + ± std band |
| `plot_confusion_matrix` | Summed confusion matrix |
| `plot_threshold_analysis` | Precision, recall, F1, balanced accuracy, G-mean vs threshold - marks `ERUPTION_PROBABILITY_THRESHOLD` (`config/constants.py`) and the optimal G-mean threshold |
| `plot_feature_importance` | Mean feature importance with std error bars |

Auto-rendered at:

```
{station_dir}/evaluation/{kind}/classifiers/{classifier}/figures/*.png
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

## SHAP Status

The current evaluation flow has SHAP plotting stubbed via `evaluate(plot_shap=True)` - the call is accepted 
but emits a warning instead of rendering. Per-seed SHAP plots will return once the follow-up rebuilds them from the `(y_proba, y_pred, y_true)` 
matrices persisted by `MetricsEnsemble`. When that happens, remember to pass `plot_size=None` to `shap.plots.beeswarm` so SHAP does not override the pre-created `figsize`.

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
