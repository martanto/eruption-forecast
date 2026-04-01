"""Comparator report: side-by-side classifier comparison with Plotly charts."""

import os
import json
from typing import Any

import pandas as pd

from eruption_forecast.report.base_report import BaseReport


class ComparatorReport(BaseReport):
    """Generate an interactive HTML report comparing multiple classifiers.

    Accepts a mapping of classifier names to registry CSV paths (the same
    format used by ``ClassifierComparator``) and produces:

    - Aggregate metrics table (mean ± std per classifier)
    - Plotly grouped bar chart comparing key metrics across classifiers

    Args:
        classifier_registry (dict[str, str]): Mapping of classifier name
            (e.g. ``"rf"``) to the path of its trained-model registry CSV.
        metrics_to_compare (list[str] | None): Metric column names to include
            in comparison charts. Defaults to ``["f1", "roc_auc", "precision",
            "recall", "accuracy"]``.
        output_dir (str | None): Output directory for the saved report.
        root_dir (str | None): Anchor directory for path resolution.
    """

    _DEFAULT_METRICS = ["f1", "roc_auc", "precision", "recall", "accuracy"]

    def __init__(
        self,
        classifier_registry: dict[str, str],
        metrics_to_compare: list[str] | None = None,
        output_dir: str | None = None,
        root_dir: str | None = None,
    ) -> None:
        """Initialize ComparatorReport.

        Args:
            classifier_registry (dict[str, str]): Classifier name → registry
                CSV path mapping.
            metrics_to_compare (list[str] | None): Metric names to compare.
                Defaults to None (uses the default set).
            output_dir (str | None): Directory for saved HTML reports.
                Defaults to None.
            root_dir (str | None): Anchor for path resolution. Defaults to None.
        """
        super().__init__(output_dir=output_dir, root_dir=root_dir)
        self._registry = classifier_registry
        self._metrics = metrics_to_compare or self._DEFAULT_METRICS

    @property
    def title(self) -> str:
        """Report title.

        Returns:
            str: Title string for the comparator report.
        """
        return "Classifier Comparison Report"

    @property
    def _template_name(self) -> str:
        """Jinja2 template filename.

        Returns:
            str: Template filename.
        """
        return "comparator.html.j2"

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_registry(self, csv_path: str) -> pd.DataFrame | None:
        """Load a single classifier registry CSV file.

        Args:
            csv_path (str): Path to the registry CSV file.

        Returns:
            pd.DataFrame | None: Loaded DataFrame, or None on failure.
        """
        if not os.path.exists(csv_path):
            return None
        try:
            return pd.read_csv(csv_path)
        except Exception:  # noqa: BLE001
            return None

    def _aggregate_classifier(self, df: pd.DataFrame) -> dict[str, str]:
        """Compute mean ± std for each numeric metric column.

        Args:
            df (pd.DataFrame): Registry DataFrame for one classifier.

        Returns:
            dict[str, str]: Metric name → ``"mean ± std"`` string.
        """
        result: dict[str, str] = {}
        for m in self._metrics:
            if m not in df.columns:
                continue
            vals = pd.to_numeric(df[m], errors="coerce").dropna()
            if vals.empty:
                continue
            result[m] = f"{vals.mean():.4f} ± {vals.std():.4f}"
        return result

    def _comparison_chart_json(self, agg_data: dict[str, dict[str, str]]) -> str:
        """Build Plotly grouped bar chart for all classifiers × metrics.

        Args:
            agg_data (dict[str, dict[str, str]]): Classifier name → {metric:
                "mean ± std"} mapping.

        Returns:
            str: Plotly JSON string.
        """
        classifiers = list(agg_data.keys())
        traces = []

        for metric in self._metrics:
            means = []
            errors = []
            for clf in classifiers:
                val_str = agg_data[clf].get(metric, "")
                if "±" in val_str:
                    parts = val_str.split("±")
                    means.append(float(parts[0].strip()))
                    errors.append(float(parts[1].strip()))
                else:
                    means.append(0.0)
                    errors.append(0.0)

            traces.append(
                {
                    "type": "bar",
                    "name": metric.upper().replace("_", " "),
                    "x": classifiers,
                    "y": means,
                    "error_y": {"type": "data", "array": errors, "visible": True},
                }
            )

        layout = {
            "height": 350,
            "barmode": "group",
            "margin": {"l": 55, "r": 20, "t": 20, "b": 80},
            "xaxis": {"title": "Classifier"},
            "yaxis": {"title": "Score", "range": [0, 1.1]},
            "legend": {"orientation": "h", "y": -0.4},
            "paper_bgcolor": "white",
            "plot_bgcolor": "#f7f9fc",
        }
        return json.dumps({"data": traces, "layout": layout})

    def _collect_data(self) -> dict[str, Any]:
        """Collect all template variables for the comparator report.

        Returns:
            dict[str, Any]: Context dict with aggregate metrics per classifier
                and grouped bar chart JSON.
        """
        agg_data: dict[str, dict[str, str]] = {}
        n_seeds_per_clf: dict[str, int] = {}

        for clf_name, csv_path in self._registry.items():
            df = self._load_registry(csv_path)
            if df is None:
                agg_data[clf_name] = {}
                n_seeds_per_clf[clf_name] = 0
            else:
                agg_data[clf_name] = self._aggregate_classifier(df)
                n_seeds_per_clf[clf_name] = len(df)

        return {
            "classifiers": list(self._registry.keys()),
            "metrics": self._metrics,
            "agg_data": agg_data,
            "n_seeds_per_clf": n_seeds_per_clf,
            "comparison_chart_json": self._comparison_chart_json(agg_data),
        }
