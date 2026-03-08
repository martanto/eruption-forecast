"""Pipeline report: executive summary + all section reports with sidebar nav."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from eruption_forecast.report.base_report import BaseReport
from eruption_forecast.report.label_report import LabelReport
from eruption_forecast.report.tremor_report import TremorReport
from eruption_forecast.report.features_report import FeaturesReport
from eruption_forecast.report.training_report import TrainingReport
from eruption_forecast.report.prediction_report import PredictionReport


if TYPE_CHECKING:
    from eruption_forecast.model.forecast_model import ForecastModel



class PipelineReport(BaseReport):
    """Generate a combined pipeline HTML report with sidebar navigation.

    Renders an executive summary and all available section reports
    (tremor, label, features, training, prediction) into a single
    self-contained HTML page with a sidebar navigation.

    Construct via :meth:`from_forecast_model` or :meth:`from_output_dir`
    rather than calling the constructor directly.

    Args:
        sections (list[str] | None): Section names to include. If None, all
            available sections are included. Defaults to None.
        tremor_report (TremorReport | None): Tremor section report. Defaults to None.
        label_report (LabelReport | None): Label section report. Defaults to None.
        features_report (FeaturesReport | None): Features section report.
            Defaults to None.
        training_report (TrainingReport | None): Training section report.
            Defaults to None.
        prediction_report (PredictionReport | None): Prediction section report.
            Defaults to None.
        summary_meta (dict[str, Any] | None): Extra key-value pairs for the
            executive summary. Defaults to None.
        output_dir (str | None): Output directory for the saved report.
        root_dir (str | None): Anchor directory for path resolution.
    """

    def __init__(
        self,
        sections: list[str] | None = None,
        tremor_report: TremorReport | None = None,
        label_report: LabelReport | None = None,
        features_report: FeaturesReport | None = None,
        training_reports: list[TrainingReport] | None = None,
        prediction_report: PredictionReport | None = None,
        summary_meta: dict[str, Any] | None = None,
        output_dir: str | None = None,
        root_dir: str | None = None,
    ) -> None:
        """Initialize PipelineReport with optional section reports.

        Args:
            sections (list[str] | None): Sections to include. Defaults to None.
            tremor_report (TremorReport | None): Tremor section. Defaults to None.
            label_report (LabelReport | None): Label section. Defaults to None.
            features_report (FeaturesReport | None): Features section. Defaults to None.
            training_reports (list[TrainingReport] | None): One TrainingReport per
                classifier. Defaults to None.
            prediction_report (PredictionReport | None): Prediction section.
                Defaults to None.
            summary_meta (dict[str, Any] | None): Executive summary metadata.
                Defaults to None.
            output_dir (str | None): Directory for saved HTML reports. Defaults to None.
            root_dir (str | None): Anchor for path resolution. Defaults to None.
        """
        super().__init__(output_dir=output_dir, root_dir=root_dir)
        self._sections = sections
        self._tremor_report = tremor_report
        self._label_report = label_report
        self._features_report = features_report
        self._training_reports = training_reports or []
        self._prediction_report = prediction_report
        self._summary_meta = summary_meta or {}

    @property
    def title(self) -> str:
        """Report title.

        Returns:
            str: Title string for the pipeline report.
        """
        return "Eruption Forecast — Full Pipeline Report"

    @property
    def _template_name(self) -> str:
        """Jinja2 template filename.

        Returns:
            str: Template filename.
        """
        return "pipeline.html.j2"

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_forecast_model(
        cls,
        fm: ForecastModel,
        sections: list[str] | None = None,
        output_dir: str | None = None,
    ) -> PipelineReport:
        """Build a PipelineReport from a live ForecastModel instance.

        Reads available artifacts directly from the model instance attributes
        and instantiates whichever section reports have data.

        Args:
            fm (ForecastModel): A ForecastModel instance after one or more
                pipeline stages have been called.
            sections (list[str] | None): Section names to include. None
                includes all available sections. Defaults to None.
            output_dir (str | None): Directory for the saved report. Defaults
                to ``{station_dir}/reports/``.

        Returns:
            PipelineReport: A configured pipeline report instance.
        """
        out = output_dir or os.path.join(fm.station_dir, "reports")

        tremor_rpt: TremorReport | None = None
        if fm.TremorData is not None:
            tremor_rpt = TremorReport(
                tremor_data=fm.TremorData,
                station=fm.station,
                output_dir=out,
            )

        label_rpt: LabelReport | None = None
        eruption_dates: list[str] = []
        if fm.label_csv and os.path.exists(str(fm.label_csv)):
            if fm.LabelBuilder is not None:
                eruption_dates = [
                    str(d) for d in getattr(fm.LabelBuilder, "eruption_dates", [])
                ]
            label_rpt = LabelReport(
                label_data=str(fm.label_csv),
                eruption_dates=eruption_dates,
                output_dir=out,
            )

        features_rpt: FeaturesReport | None = None
        if fm.features_csv and os.path.exists(str(fm.features_csv)):
            features_rpt = FeaturesReport(
                features_csv=str(fm.features_csv),
                label_csv=fm.label_csv,
                selection_method=fm.feature_selection_method,
                output_dir=out,
            )

        # Prefer metrics from the evaluations directory on disk — these exist when
        # train(with_evaluation=True) was run previously, even if the current call
        # was train(with_evaluation=False) for model-only forecasting.
        training_rpts: list[TrainingReport] = []
        evaluations_dir = os.path.join(fm.station_dir, "trainings", "evaluations")
        if os.path.isdir(evaluations_dir):
            for root, dirs, _files in os.walk(evaluations_dir):
                if "metrics" in dirs:
                    m_dir = os.path.join(root, "metrics")
                    # Path: evaluations/{clf_slug}/{cv_slug}/ → metrics/ is inside cv_slug.
                    # Classifier name is one directory above the cv directory.
                    clf_name = os.path.basename(os.path.dirname(root))
                    training_rpts.append(
                        TrainingReport(
                            metrics_dir=m_dir,
                            classifier_name=clf_name,
                            output_dir=out,
                        )
                    )

        # Fall back to live fm.trainers when no evaluations directory exists yet
        if not training_rpts:
            for trainer in fm.trainers:
                clf_name = trainer.get("classifier_name", "Classifier")
                model_trainer = trainer.get("model_trainer")
                metrics_dir: str | None = None
                if model_trainer is not None:
                    metrics_dir = getattr(model_trainer, "metrics_dir", None)
                training_rpts.append(
                    TrainingReport(
                        metrics_dir=metrics_dir,
                        classifier_name=clf_name,
                        output_dir=out,
                    )
                )

        pred_rpt: PredictionReport | None = None
        if fm.prediction_df is not None and not fm.prediction_df.empty:
            pred_rpt = PredictionReport(
                prediction_df=fm.prediction_df,
                eruption_dates=eruption_dates,
                output_dir=out,
            )
        else:
            # Fall back to a predictions CSV saved on disk by a previous forecast() call.
            # ModelPredictor saves to {station_dir}/result_all_model_predictions_*.csv
            for f in sorted(os.listdir(fm.station_dir)):
                if f.startswith("result_all_model_predictions") and f.endswith(".csv"):
                    pred_rpt = PredictionReport(
                        prediction_df=os.path.join(fm.station_dir, f),
                        eruption_dates=eruption_dates,
                        output_dir=out,
                    )
                    break

        summary_meta: dict[str, Any] = {
            "Station": fm.station,
            "Channel": fm.channel,
            "Network": fm.network,
            "Start date": fm.start_date_str,
            "End date": fm.end_date_str,
            "Window size (days)": fm.window_size,
            "Volcano ID": fm.volcano_id,
        }

        return cls(
            sections=sections,
            tremor_report=tremor_rpt,
            label_report=label_rpt,
            features_report=features_rpt,
            training_reports=training_rpts,
            prediction_report=pred_rpt,
            summary_meta=summary_meta,
            output_dir=out,
        )

    @classmethod
    def from_output_dir(
        cls,
        output_dir: str,
        sections: list[str] | None = None,
    ) -> PipelineReport:
        """Build a PipelineReport by scanning an existing output directory.

        Searches for known artifact files (tremor CSV, label CSV, features CSV,
        metrics directory, prediction CSV) and instantiates whichever section
        reports have data available.

        Args:
            output_dir (str): Root output directory to scan (e.g. the
                station-specific directory such as
                ``output/VG.OJN.00.EHZ/``).
            sections (list[str] | None): Sections to include. None includes
                all. Defaults to None.

        Returns:
            PipelineReport: A configured pipeline report instance.
        """
        report_out = os.path.join(output_dir, "reports")

        def _find(pattern: str) -> str | None:
            """Search output_dir recursively for the first file matching pattern.

            Args:
                pattern (str): Substring to search for in filenames.

            Returns:
                str | None: First matching absolute path, or None.
            """
            for root, _dirs, files in os.walk(output_dir):
                for f in sorted(files):
                    if pattern in f:
                        return os.path.join(root, f)
            return None

        tremor_csv = _find("tremor") if not _find("tremor_matrix") else None
        # Prefer non-matrix tremor CSV
        for root, _dirs, files in os.walk(output_dir):
            for f in sorted(files):
                if f.startswith("tremor_") and "matrix" not in f and f.endswith(".csv"):
                    tremor_csv = os.path.join(root, f)
                    break

        label_csv = _find("label_")
        features_csv = _find("features_")
        prediction_csv = _find("prediction")

        tremor_rpt = TremorReport(tremor_csv, output_dir=report_out) if tremor_csv else None
        label_rpt = LabelReport(label_csv, output_dir=report_out) if label_csv else None
        features_rpt = (
            FeaturesReport(features_csv, label_csv=label_csv, output_dir=report_out)
            if features_csv
            else None
        )

        # Find all metrics directories (one per classifier)
        training_rpts: list[TrainingReport] = []
        for root, dirs, _files in os.walk(output_dir):
            if "metrics" in dirs:
                m_dir = os.path.join(root, "metrics")
                clf_name = os.path.basename(root)
                training_rpts.append(
                    TrainingReport(
                        metrics_dir=m_dir,
                        classifier_name=clf_name,
                        output_dir=report_out,
                    )
                )

        pred_rpt = (
            PredictionReport(prediction_csv, output_dir=report_out)
            if prediction_csv
            else None
        )

        return cls(
            sections=sections,
            tremor_report=tremor_rpt,
            label_report=label_rpt,
            features_report=features_rpt,
            training_reports=training_rpts,
            prediction_report=pred_rpt,
            output_dir=report_out,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _collect_data(self) -> dict[str, Any]:
        """Collect all section HTML fragments and sidebar navigation entries.

        Renders each available sub-report to an HTML string and returns a
        context dict for the pipeline template.

        Returns:
            dict[str, Any]: Context dict with rendered section HTML fragments,
                summary metadata, and sidebar navigation entries.
        """
        requested = set(self._sections) if self._sections else None

        section_html: dict[str, str] = {}
        training_html_list: list[dict[str, str]] = []
        nav_sections = [
            {"type": "header", "label": "Pipeline", "anchor": ""},
            {"type": "link", "label": "Summary", "anchor": "summary"},
        ]

        single_sections: list[tuple[str, str, BaseReport | None]] = [
            ("tremor", "Tremor Data", self._tremor_report),
            ("label", "Labels", self._label_report),
            ("features", "Features", self._features_report),
            ("prediction", "Prediction", self._prediction_report),
        ]

        for key, label, rpt in single_sections:
            if (requested is not None and key not in requested) or rpt is None:
                continue
            try:
                section_html[key] = rpt.to_html()
            except Exception as exc:  # noqa: BLE001
                section_html[key] = f"<p>Error rendering {key} section: {exc}</p>"
            nav_sections.append({"type": "link", "label": label, "anchor": key})

        # Training: one entry per classifier
        if requested is None or "training" in requested:
            if self._training_reports:
                nav_sections.append(
                    {"type": "header", "label": "Training", "anchor": ""}
                )
            for rpt in self._training_reports:
                clf_name = rpt._classifier_name  # noqa: SLF001
                anchor = f"training-{clf_name.lower().replace(' ', '-')}"
                try:
                    html = rpt.to_html()
                except Exception as exc:  # noqa: BLE001
                    html = f"<p>Error rendering training section: {exc}</p>"
                training_html_list.append(
                    {"classifier_name": clf_name, "anchor": anchor, "html": html}
                )
                nav_sections.append(
                    {"type": "link", "label": clf_name, "anchor": anchor}
                )

        return {
            "summary_meta": self._summary_meta,
            "section_html": section_html,
            "training_html_list": training_html_list,
            "sections": nav_sections,
            "has_tremor": "tremor" in section_html,
            "has_label": "label" in section_html,
            "has_features": "features" in section_html,
            "has_training": bool(training_html_list),
            "has_prediction": "prediction" in section_html,
        }
