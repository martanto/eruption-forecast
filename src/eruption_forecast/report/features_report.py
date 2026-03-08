"""Features report: feature counts, selection method, and band contributions."""

import os
import json
from typing import Any

import pandas as pd

from eruption_forecast.report.base_report import BaseReport


class FeaturesReport(BaseReport):
    """Generate an interactive HTML report for extracted tsfresh features.

    Produces a self-contained HTML page with:

    - Feature count before and after selection
    - Selection method used
    - Top-N feature importance Plotly bar chart
    - Frequency band contribution chart

    Args:
        features_csv (str): Path to the extracted features CSV file.
        label_csv (str | None): Path to the aligned label CSV file. Used to
            display feature-label alignment counts. Defaults to None.
        significant_features_dir (str | None): Directory containing per-seed
            significant feature CSV files. Used to aggregate feature importance
            across seeds. Defaults to None.
        selection_method (str): Feature selection method used
            (``"tsfresh"``, ``"random_forest"``, or ``"combined"``).
            Defaults to ``"tsfresh"``.
        top_n (int): Number of top features to show in the bar chart.
            Defaults to 20.
        output_dir (str | None): Output directory for the saved report.
        root_dir (str | None): Anchor directory for path resolution.
    """

    def __init__(
        self,
        features_csv: str,
        label_csv: str | None = None,
        significant_features_dir: str | None = None,
        selection_method: str = "tsfresh",
        top_n: int = 20,
        output_dir: str | None = None,
        root_dir: str | None = None,
    ) -> None:
        """Initialize FeaturesReport.

        Stores references to the feature and label CSV paths, and any
        pre-computed significant-features directory.

        Args:
            features_csv (str): Path to extracted features CSV.
            label_csv (str | None): Path to aligned label CSV. Defaults to None.
            significant_features_dir (str | None): Directory with per-seed
                significant feature lists. Defaults to None.
            selection_method (str): Feature selection method name.
                Defaults to ``"tsfresh"``.
            top_n (int): Number of top features to show. Defaults to 20.
            output_dir (str | None): Directory for saved reports. Defaults to None.
            root_dir (str | None): Anchor for path resolution. Defaults to None.
        """
        super().__init__(output_dir=output_dir, root_dir=root_dir)
        self._features_csv = features_csv
        self._label_csv = label_csv
        self._sig_dir = significant_features_dir
        self._selection_method = selection_method
        self._top_n = top_n

    @property
    def title(self) -> str:
        """Report title.

        Returns:
            str: Title string for the features report.
        """
        return "Feature Extraction Report"

    @property
    def _template_name(self) -> str:
        """Jinja2 template filename.

        Returns:
            str: Template filename.
        """
        return "features.html.j2"

    # ------------------------------------------------------------------
    # Data collection helpers
    # ------------------------------------------------------------------

    def _load_significant_features(self) -> dict[str, int]:
        """Load and aggregate significant feature counts across all seed files.

        Reads every CSV file found in ``self._sig_dir`` and tallies how often
        each feature name appears. Returns an empty dict when the directory is
        absent or contains no CSV files.

        Returns:
            dict[str, int]: Mapping of feature name → appearance count.
        """
        if not self._sig_dir or not os.path.isdir(self._sig_dir):
            return {}

        counts: dict[str, int] = {}
        for fname in os.listdir(self._sig_dir):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(self._sig_dir, fname)
            try:
                df = pd.read_csv(fpath)
                col = "feature" if "feature" in df.columns else df.columns[0]
                for feat in df[col].dropna():
                    counts[str(feat)] = counts.get(str(feat), 0) + 1
            except Exception:  # noqa: BLE001
                continue
        return counts

    def _band_contribution(self, feature_counts: dict[str, int]) -> str:
        """Build Plotly bar chart JSON for frequency band contributions.

        Groups features by the band prefix (e.g., ``rsam_f0``, ``dsar_f0-f1``)
        extracted from feature names and sums their selection counts.

        Args:
            feature_counts (dict[str, int]): Feature name → count mapping.

        Returns:
            str: Plotly JSON string for the band contribution bar chart.
        """
        band_totals: dict[str, int] = {}
        for feat, cnt in feature_counts.items():
            parts = feat.split("__")
            band = parts[0] if parts else feat
            band_totals[band] = band_totals.get(band, 0) + cnt

        sorted_bands = sorted(band_totals.items(), key=lambda x: -x[1])
        if not sorted_bands:
            return json.dumps({"data": [], "layout": {}})

        bands, vals = zip(*sorted_bands, strict=False)
        traces = [
            {
                "type": "bar",
                "x": list(bands),
                "y": list(vals),
                "marker": {"color": "#2c5282"},
                "text": [str(v) for v in vals],
                "textposition": "outside",
            }
        ]
        layout = {
            "height": 280,
            "margin": {"l": 50, "r": 20, "t": 20, "b": 100},
            "xaxis": {"title": "Tremor column", "tickangle": -45},
            "yaxis": {"title": "Selection count"},
            "paper_bgcolor": "white",
            "plot_bgcolor": "#f7f9fc",
            "showlegend": False,
        }
        return json.dumps({"data": traces, "layout": layout})

    def _top_features_chart_json(self, feature_counts: dict[str, int]) -> str:
        """Build Plotly horizontal bar chart JSON for the top-N features.

        Args:
            feature_counts (dict[str, int]): Feature name → count mapping.

        Returns:
            str: Plotly JSON string for the top-N features bar chart.
        """
        if not feature_counts:
            return json.dumps({"data": [], "layout": {}})

        top = sorted(feature_counts.items(), key=lambda x: -x[1])[: self._top_n]
        feats, vals = zip(*top, strict=False)

        traces = [
            {
                "type": "bar",
                "orientation": "h",
                "x": list(vals)[::-1],
                "y": [f[:60] for f in list(feats)[::-1]],
                "marker": {"color": "#1a3a5c"},
            }
        ]
        layout = {
            "height": max(300, self._top_n * 22),
            "margin": {"l": 300, "r": 30, "t": 20, "b": 50},
            "xaxis": {"title": "Seed selection count"},
            "paper_bgcolor": "white",
            "plot_bgcolor": "#f7f9fc",
            "showlegend": False,
        }
        return json.dumps({"data": traces, "layout": layout})

    def _collect_data(self) -> dict[str, Any]:
        """Collect all template variables for the features report.

        Returns:
            dict[str, Any]: Context dict with feature counts, selection method,
                and Plotly chart JSON strings.
        """
        df = pd.read_csv(self._features_csv, nrows=1)
        total_features = len(df.columns)

        label_windows: int | None = None
        if self._label_csv and os.path.exists(self._label_csv):
            ldf = pd.read_csv(self._label_csv)
            label_windows = len(ldf)

        feature_counts = self._load_significant_features()
        n_significant = len(feature_counts) if feature_counts else None

        return {
            "features_csv": self._features_csv,
            "total_features": total_features,
            "n_significant": n_significant,
            "label_windows": label_windows,
            "selection_method": self._selection_method,
            "top_n": self._top_n,
            "top_features_chart_json": self._top_features_chart_json(feature_counts),
            "band_contribution_chart_json": self._band_contribution(feature_counts),
        }
