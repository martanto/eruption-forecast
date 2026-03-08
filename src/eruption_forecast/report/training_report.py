"""Training report: per-seed metrics, ROC, confusion matrix, seed stability."""

import os
import re
import json
from typing import Any

import numpy as np


# Ordered list of (key_aliases, display_label) for the aggregate table.
# Only these metrics are shown; keys are tried in order until one is found.
_AGGREGATE_DISPLAY: list[tuple[tuple[str, ...], str]] = [
    (("balanced_accuracy",), "Balanced Accuracy"),
    (("precision",), "Precision"),
    (("recall",), "Recall"),
    (("f1", "f1_score"), "F1 Score"),
    (("roc_auc",), "ROC AUC"),
    (("pr_auc", "average_precision"), "PR AUC"),
    (("threshold", "optimal_threshold"), "Optimal Threshold"),
]

from eruption_forecast.report.base_report import BaseReport


class TrainingReport(BaseReport):
    """Generate an interactive HTML report for model training results.

    Builds per-seed ``<details>`` blocks with all metrics, aggregate
    mean ± std rows, best-seed highlighting, and interactive Plotly charts
    for ROC curve, confusion matrix heatmap, seed stability, and threshold
    analysis.

    Training results can be supplied as:

    - A ``ModelEvaluator`` instance (uses its JSON metric files directly)
    - A directory path containing metric JSON files

    Args:
        metrics_dir (str | None): Directory containing ``*_metrics.json``
            files produced by ``ModelTrainer``. Defaults to None.
        classifier_name (str): Human-readable classifier name shown in the
            report header. Defaults to ``"Classifier"``.
        output_dir (str | None): Output directory for the saved report.
        root_dir (str | None): Anchor directory for path resolution.
    """

    def __init__(
        self,
        metrics_dir: str | None = None,
        classifier_name: str = "Classifier",
        output_dir: str | None = None,
        root_dir: str | None = None,
    ) -> None:
        """Initialize TrainingReport from a metrics directory.

        Args:
            metrics_dir (str | None): Path to directory containing per-seed
                ``*_metrics.json`` files. Defaults to None.
            classifier_name (str): Name of the classifier for the report title.
                Defaults to ``"Classifier"``.
            output_dir (str | None): Directory for saved HTML reports.
                Defaults to None.
            root_dir (str | None): Anchor for path resolution. Defaults to None.
        """
        super().__init__(output_dir=output_dir, root_dir=root_dir)
        self._metrics_dir = metrics_dir
        self._classifier_name = classifier_name

    @property
    def title(self) -> str:
        """Report title including classifier name.

        Returns:
            str: Title string for the training report.
        """
        return f"Training Report — {self._classifier_name}"

    @property
    def _template_name(self) -> str:
        """Jinja2 template filename.

        Returns:
            str: Template filename.
        """
        return "training.html.j2"

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_seed_metrics(self) -> list[dict[str, Any]]:
        """Load all per-seed metric JSON files from the metrics directory.

        Each file is expected to contain a dict with metric keys such as
        ``seed``, ``f1``, ``roc_auc``, ``precision``, ``recall``,
        ``accuracy``, ``threshold``.

        Returns:
            list[dict[str, Any]]: List of metric dicts, one per seed, sorted
                by seed number.
        """
        if not self._metrics_dir or not os.path.isdir(self._metrics_dir):
            return []

        rows = []
        for fname in sorted(os.listdir(self._metrics_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(self._metrics_dir, fname)
            try:
                with open(fpath) as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    rows.append(data)
            except Exception:  # noqa: BLE001
                continue
        return sorted(rows, key=lambda x: x.get("seed", 0))

    def _aggregate(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute mean ± std for numeric metrics across all seeds.

        Args:
            rows (list[dict[str, Any]]): Per-seed metric dicts.

        Returns:
            dict[str, Any]: Dict mapping metric name → "mean ± std" string
                for numeric fields.
        """
        if not rows:
            return {}
        numeric_keys = [
            k
            for k, v in rows[0].items()
            if isinstance(v, (int, float)) and k != "seed"
        ]
        result: dict[str, Any] = {}
        for k in numeric_keys:
            vals = [r[k] for r in rows if k in r and r[k] is not None]
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                result[k] = f"{mean:.4f} ± {std:.4f}"
        return result

    def _best_seed_idx(self, rows: list[dict[str, Any]]) -> int:
        """Find the index of the best seed by F1 score.

        Args:
            rows (list[dict[str, Any]]): Per-seed metric dicts.

        Returns:
            int: Index of the row with the highest F1 score, or -1 if none.
        """
        best_idx = -1
        best_f1 = -1.0
        for i, row in enumerate(rows):
            f1 = row.get("f1", row.get("f1_score", 0.0)) or 0.0
            if f1 > best_f1:
                best_f1 = float(f1)
                best_idx = i
        return best_idx

    # ------------------------------------------------------------------
    # Chart helpers
    # ------------------------------------------------------------------

    def _seed_stability_chart(self, rows: list[dict[str, Any]]) -> str:
        """Build Plotly line chart for metric stability across seeds.

        Args:
            rows (list[dict[str, Any]]): Per-seed metric dicts.

        Returns:
            str: Plotly JSON string for the seed stability chart.
        """
        if not rows:
            return json.dumps({"data": [], "layout": {}})

        seeds = [r.get("seed", i) for i, r in enumerate(rows)]
        metrics_to_plot = ["f1_score", "roc_auc", "precision", "recall", "accuracy"]
        _metric_labels = {
            "f1_score": "F1 Score",
            "roc_auc": "ROC AUC",
            "precision": "Precision",
            "recall": "Recall",
            "accuracy": "Accuracy",
        }
        traces = []
        for m in metrics_to_plot:
            vals = [r.get(m) for r in rows]
            if all(v is None for v in vals):
                continue
            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": _metric_labels.get(m, m.upper().replace("_", " ")),
                    "x": seeds,
                    "y": vals,
                    "marker": {"size": 4},
                }
            )

        layout = {
            "height": 300,
            "margin": {"l": 55, "r": 20, "t": 20, "b": 50},
            "xaxis": {"title": "Seed"},
            "yaxis": {"title": "Score", "range": [0, 1.05]},
            "legend": {"orientation": "h", "y": -0.3},
            "paper_bgcolor": "white",
            "plot_bgcolor": "#f7f9fc",
        }
        return json.dumps({"data": traces, "layout": layout})

    def _threshold_chart(self, rows: list[dict[str, Any]]) -> str:
        """Build Plotly line chart for threshold analysis (placeholder).

        In the absence of per-threshold data in the metric files, shows the
        mean F1, precision, and recall as flat reference lines.

        Args:
            rows (list[dict[str, Any]]): Per-seed metric dicts.

        Returns:
            str: Plotly JSON string for the threshold analysis chart.
        """
        if not rows:
            return json.dumps({"data": [], "layout": {}})

        thresholds = [i / 20 for i in range(1, 20)]
        agg = self._aggregate(rows)

        def _mean(key: str) -> float:
            raw = agg.get(key, "0")
            try:
                return float(str(raw).split("±")[0].strip())
            except ValueError:
                return 0.0

        f1_m = _mean("f1")
        prec_m = _mean("precision")
        rec_m = _mean("recall")

        traces = [
            {
                "type": "scatter",
                "mode": "lines",
                "name": "F1 (avg)",
                "x": thresholds,
                "y": [f1_m] * len(thresholds),
                "line": {"dash": "dot"},
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Precision (avg)",
                "x": thresholds,
                "y": [prec_m] * len(thresholds),
                "line": {"dash": "dash"},
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Recall (avg)",
                "x": thresholds,
                "y": [rec_m] * len(thresholds),
            },
        ]
        layout = {
            "height": 280,
            "margin": {"l": 55, "r": 20, "t": 20, "b": 50},
            "xaxis": {"title": "Threshold"},
            "yaxis": {"title": "Score", "range": [0, 1.05]},
            "legend": {"orientation": "h", "y": -0.3},
            "paper_bgcolor": "white",
            "plot_bgcolor": "#f7f9fc",
        }
        return json.dumps({"data": traces, "layout": layout})

    def _collect_data(self) -> dict[str, Any]:
        """Collect all template variables for the training report.

        Returns:
            dict[str, Any]: Context dict with per-seed rows, aggregate stats,
                best-seed index, and Plotly chart JSON strings.
        """
        rows = self._load_seed_metrics()
        aggregate = self._aggregate(rows)
        best_idx = self._best_seed_idx(rows)

        metric_keys = []
        if rows:
            metric_keys = [k for k in rows[0].keys() if k != "seed"]

        # Build filtered, ordered aggregate display list
        aggregate_display = []
        for aliases, label in _AGGREGATE_DISPLAY:
            for key in aliases:
                if key in aggregate:
                    aggregate_display.append({"label": label, "value": aggregate[key]})
                    break

        # Unique ID for Plotly div elements — avoids collisions when multiple
        # TrainingReport sections are embedded in the same pipeline HTML page.
        report_id = re.sub(r"[^a-z0-9]", "-", self._classifier_name.lower())

        return {
            "classifier_name": self._classifier_name,
            "report_id": report_id,
            "n_seeds": len(rows),
            "seed_rows": rows,
            "aggregate": aggregate,
            "aggregate_display": aggregate_display,
            "best_seed_idx": best_idx,
            "metric_keys": metric_keys,
            "stability_chart_json": self._seed_stability_chart(rows),
            "threshold_chart_json": self._threshold_chart(rows),
        }
