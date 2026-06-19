import os
from typing import Any, Self, Literal
from datetime import datetime

import pandas as pd
import matplotlib

from eruption_forecast.plots import plot_forecast
from eruption_forecast.logger import logger
from eruption_forecast.utils.window import construct_windows
from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.model.base_model import BaseModel
from eruption_forecast.utils.date_utils import set_datetime_index
from eruption_forecast.utils.formatting import pdf_metadata
from eruption_forecast.model.cache_model import CacheModel
from eruption_forecast.ensemble.seed_ensemble import SeedEnsemble
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PredictionModel(BaseModel, CacheModel):
    """Forecast volcanic eruption probabilities from a trained ensemble.

    Loads a trained ensemble produced by ``TrainingModel.fit()`` and runs
    forecast inference over an unlabelled window grid. Builds the grid via
    ``build_label()``, extracts tsfresh features via ``extract_features()``,
    then aggregates per-seed and per-classifier probabilities into a consensus
    forecast via ``forecast()``. Use :meth:`forecast` for the full forecast +
    plotting flow.

    Args:
        model (str | ClassifierEnsemble | SeedEnsemble): Trained model source.
            Accepted forms:

            - ``ClassifierEnsemble`` object — used directly.
            - ``SeedEnsemble`` object — wrapped into a ``ClassifierEnsemble``.
            - Path to ``ClassifierEnsemble.json`` (from ``TrainingModel``).
            - Path to ``ClassifierEnsemble.pkl`` or ``SeedEnsemble_*.pkl``.
            - Path to a trained-model registry ``*.json`` (new, written by
              :func:`~eruption_forecast.utils.ml.save_model_json`; one value
              from ``TrainingModel.results``) or legacy ``*.csv``.
        tremor_data (str | pd.DataFrame): Path to a tremor CSV file or a
            pre-loaded tremor DataFrame.
        start_date (str | datetime): Inclusive start of the forecast period.
        end_date (str | datetime): Inclusive end of the forecast period.
        window_size (int, optional): Sliding window size in days. Defaults to
            ``2``.
        overwrite (bool, optional): Overwrite existing output files when
            ``True``. Defaults to ``False``.
        output_dir (str | None, optional): Root output directory. Resolved
            relative to ``root_dir`` when omitted. Defaults to ``None``.
        root_dir (str | None, optional): Project root directory used to resolve
            ``output_dir``. Defaults to ``None``.
        n_jobs (int, optional): Number of parallel workers. Defaults to ``1``.
        verbose (bool, optional): Emit verbose log messages when ``True``.
            Defaults to ``False``.

    Attributes:
        ClassifierEnsemble (ClassifierEnsemble): Trained ensemble used for
            inference. Always populated after construction.
        kind (Literal["training", "prediction"]): Stage marker consumed by
            ``EvaluationModel`` to dispatch on reuse type. Always
            ``"prediction"``.
        basename (str): ``"{start_date_str}_{end_date_str}"`` — shared
            filename stem for label, feature, plot, and result artefacts.
        prediction_dir (str): Root prediction output directory.
        features_dir (str): Forecast-grid features directory.
        result_dir (str): Per-seed probability output directory.
        labels (pd.Series): Forecast-window index. ``pd.DatetimeIndex`` with
            integer ``id`` values; carries no truth labels. Empty until
            ``build_label()`` is called. May be re-narrowed by
            ``extract_features()`` to drop ids whose tremor windows failed
            ``minimum_completion``.
        labels_csv (str | None): Path to the forecast-grid CSV written by
            ``build_label()`` (and overwritten by ``extract_features()`` when
            the grid is narrowed). ``None`` until ``build_label()`` is called.
        forecast_plot_path (str): Path to the saved forecast plot (PDF when
            ``plot_pdf=True``, otherwise PNG). Set by ``forecast()``.
        results (pd.DataFrame): Forecast results indexed by datetime, with
            one ``{classifier}_{probability|uncertainty|prediction|confidence}``
            column per registered classifier plus the four ``consensus_*``
            columns. Empty until ``forecast()`` is called.

    Example:
        >>> prediction = (
        ...     PredictionModel(
        ...         model="output/.../ClassifierEnsemble.pkl",
        ...         tremor_data="output/.../tremor.csv",
        ...         start_date="2025-03-16",
        ...         end_date="2025-03-22",
        ...     )
        ...     .build_label(window_step=10, window_step_unit="minutes")
        ...     .extract_features()
        ... )
        >>> df_forecast = prediction.forecast()
    """

    def __init__(
        self,
        model: str | ClassifierEnsemble | SeedEnsemble,
        tremor_data: str | pd.DataFrame,
        start_date: str | datetime,
        end_date: str | datetime,
        window_size: int = 2,
        overwrite: bool = False,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """Initialise the prediction model.

        Resolves the trained ensemble from ``model``, delegates shared
        configuration to ``BaseModel.__init__``, computes the output
        subdirectories, and initialises empty placeholders for the forecast
        window grid. Heavy work (CSV reads, window construction, feature
        extraction, inference) is deferred to the corresponding method call.

        Args:
            model (str | ClassifierEnsemble | SeedEnsemble): Trained model
                source. See the class docstring for accepted forms.
            tremor_data (str | pd.DataFrame): Path to a tremor CSV file or a
                pre-loaded tremor DataFrame.
            start_date (str | datetime): Inclusive start of the forecast period.
            end_date (str | datetime): Inclusive end of the forecast period.
            window_size (int, optional): Sliding window size in days. Defaults
                to ``2``.
            overwrite (bool, optional): Overwrite existing output files when
                ``True``. Defaults to ``False``.
            output_dir (str | None, optional): Output directory path. Defaults
                to ``None``.
            root_dir (str | None, optional): Project root directory. Defaults
                to ``None``.
            n_jobs (int, optional): Number of parallel workers. Defaults to ``1``.
            verbose (bool, optional): Emit verbose log messages when ``True``.
                Defaults to ``False``.
        """
        super().__init__(
            tremor_data=tremor_data,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            eruption_dates=None,
            output_dir=output_dir,
            root_dir=root_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.ClassifierEnsemble: ClassifierEnsemble = (
            model
            if isinstance(model, ClassifierEnsemble)
            else ClassifierEnsemble.from_any(model, verbose)
        )

        self.kind: Literal["training", "prediction"] = "prediction"
        self.overwrite = overwrite
        self.basename = (
            f"{self.start_date_str}_{self.end_date_str}_ws-{self.window_size}"
        )

        (
            self.prediction_dir,
            self.features_dir,
            self.result_dir,
        ) = self.set_directories()

        # ``self.labels`` exposes the ``id`` Series for downstream callers;
        # ``self._labels`` keeps the full forecast grid (``id`` + ``is_erupted``
        # all-zero) so prediction labels match the training schema produced by
        # ``construct_windows``.
        self.labels: pd.Series = pd.Series()
        self._labels: pd.DataFrame = pd.DataFrame()
        self.labels_csv: str | None = None

        # Will be set after forecast() called
        self.forecast_plot_path: str | None = None
        self.results: pd.DataFrame = pd.DataFrame()

    def set_directories(self) -> tuple[str, str, str]:
        """Build and return the prediction output directory paths.

        Computes stage-specific paths from ``self.output_dir`` without
        materialising them on disk. Used by ``create_directories()`` to lay
        out the prediction tree.

        Returns:
            tuple[str, str, str]: A three-element tuple
                ``(prediction_dir, features_dir, result_dir)`` rooted under
                ``self.output_dir``.
        """
        prediction_dir = os.path.join(self.output_dir, "prediction")
        features_dir = os.path.join(prediction_dir, "features")
        result_dir = os.path.join(prediction_dir, "results")

        return prediction_dir, features_dir, result_dir

    def create_directories(self) -> None:
        """Create the prediction output directories on disk.

        Materialises ``prediction_dir``, ``features_dir``, and ``result_dir``
        so downstream steps can write into them without additional setup.
        """
        ensure_dir(self.prediction_dir)
        ensure_dir(self.features_dir)
        ensure_dir(self.result_dir)

    def validate(self) -> Self:
        """No-op validation hook required by :class:`BaseModel`.

        All required checks are already enforced by ``BaseModel._validate()``
        (date ordering, ``n_jobs`` clamping). The trained ensemble is
        resolved during ``__init__``, so no additional stage-specific
        validation is needed.

        Returns:
            Self: The current instance, for method chaining.
        """
        return self

    def describe(self) -> str:
        """Return a one-line prose summary of the prediction configuration.

        Builds a compact human-readable string covering the forecast period,
        sliding window size, and registered classifiers. Suitable for CLI
        output and human-facing reports. For raw data use ``to_dict()``; for
        an MCP/LLM-friendly block use ``to_prompt()``.

        Returns:
            str: One-line description such as
                ``"PredictionModel(period=2025-03-16 → 2025-03-22,
                window_size=2d, classifiers=[RandomForestClassifier, XGBClassifier])"``.

        Example:
            >>> print(prediction.describe())
            PredictionModel(period=2025-03-16 → 2025-03-22, window_size=2d, classifiers=[RandomForestClassifier, XGBClassifier])
        """
        classifier_names = ", ".join(self.ClassifierEnsemble.classifiers)
        return (
            f"PredictionModel("
            f"period={self.start_date_str} → {self.end_date_str}, "
            f"window_size={self.window_size}d, "
            f"classifiers=[{classifier_names}]"
            f")"
        )

    def to_dict(self) -> dict:
        """Serialise core prediction parameters to a JSON-friendly dictionary.

        Produces a stable schema for MCP tool responses and downstream LLM
        consumption. Always includes the construction-time configuration
        (``start_date``, ``end_date``, ``window_size``, ``classifiers``,
        ``overwrite``, ``output_dir``, ``root_dir``, ``n_jobs``, ``verbose``,
        and ``basename``). Stage-specific keys (``window_step``,
        ``window_step_unit``, ``features_csv``, ``forecast_plot_path``) are
        added once the corresponding method has been called.

        Returns:
            dict: Mapping of parameter name to its serialised value.

        Example:
            >>> d = prediction.to_dict()
            >>> d["window_size"]
            2
            >>> "classifiers" in d
            True
        """
        result: dict = {
            "start_date": self.start_date_str,
            "end_date": self.end_date_str,
            "window_size": self.window_size,
            "classifiers": self.ClassifierEnsemble.classifiers,
            "overwrite": self.overwrite,
            "output_dir": self.output_dir,
            "root_dir": self.root_dir,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "basename": self.basename,
        }

        if self.window_step is not None:
            result["window_step"] = self.window_step

        if self.window_step_unit is not None:
            result["window_step_unit"] = self.window_step_unit

        if self.features_csv is not None:
            result["features_csv"] = self.features_csv

        if self.forecast_plot_path is not None:
            result["forecast_plot_path"] = self.forecast_plot_path

        return result

    def to_prompt(self) -> str:
        """Return a structured bullet-list block suitable for LLM and MCP prompts.

        Builds a deterministic, template-driven prose block from
        :meth:`to_dict`. Paths are reduced to filenames only to avoid leaking
        absolute system paths into prompts. For human-readable prose use
        :meth:`describe`. For raw data use :meth:`to_dict`.

        Returns:
            str: Prompt-ready block ending with the build state (window grid,
                feature extraction, and forecast plot).

        Example:
            >>> for part in prediction.to_prompt().split(". "):
            ...     print(part)
            Forecast period: 2025-03-16 to 2025-03-22
            Window size: 2 day(s)
            Classifiers: RandomForestClassifier, XGBClassifier
            Window step: 10 minutes
            Features CSV: features-label_2025-03-16_2025-03-22_step-10-minutes.csv
            Forecast plot: forecast_2025-03-16_2025-03-22.pdf
            Overwrite: False
            Output dir: output/VG.OJN.00.EHZ
            Root dir: None
            n_jobs: 4
            Verbose: False
            Basename: 2025-03-16_2025-03-22.
        """
        classifier_names = ", ".join(self.ClassifierEnsemble.classifiers)

        window_step_str = (
            f"{self.window_step} {self.window_step_unit}"
            if self.window_step is not None and self.window_step_unit is not None
            else "not built"
        )
        features_csv_str = (
            os.path.basename(self.features_csv)
            if self.features_csv is not None
            else "not extracted"
        )
        forecast_plot_str = (
            os.path.basename(self.forecast_plot_path)
            if self.forecast_plot_path is not None
            else "not rendered"
        )

        return (
            f"Forecast period: {self.start_date_str} to {self.end_date_str}. "
            f"Window size: {self.window_size} day(s). "
            f"Classifiers: {classifier_names}. "
            f"Window step: {window_step_str}. "
            f"Features CSV: {features_csv_str}. "
            f"Forecast plot: {forecast_plot_str}. "
            f"Overwrite: {self.overwrite}. "
            f"Output dir: {self.output_dir}. "
            f"Root dir: {self.root_dir}. "
            f"n_jobs: {self.n_jobs}. "
            f"Verbose: {self.verbose}. "
            f"Basename: {self.basename}."
        )

    @classmethod
    def build_cache_identity(  # ty:ignore[invalid-method-override]
        cls,
        *,
        nslc: str,
        tremor_df: pd.DataFrame,
        training_hash: str | None,
        start_date: str | datetime,
        end_date: str | datetime,
        window_size: int,
        build_label_params: dict,
        extract_features_params: dict,
    ) -> dict:
        """Return the canonical identity dict that defines this prediction cache entry.

        Bundles the station-channel id, the tremor data fingerprint, the
        upstream training-model hash, and all prediction-specific params into
        a canonical dict. Including ``training_hash`` ensures the prediction
        cache invalidates automatically whenever the trained ensemble it
        depends on changes — even when the prediction inputs themselves are
        unchanged.

        Args:
            nslc (str): ``Network.Station.Location.Channel`` identifier.
            tremor_df (pd.DataFrame): The tremor DataFrame used for the
                forecast. Reduced to a fingerprint via
                :meth:`CacheModel.tremor_fingerprint`.
            training_hash (str | None): Hash of the upstream
                :class:`TrainingModel` identity that produced the loaded
                ensemble. ``None`` only when the ensemble was loaded from a
                source outside the current pipeline run.
            start_date (str | datetime): Forecast period start.
            end_date (str | datetime): Forecast period end.
            window_size (int): Sliding window size in days.
            build_label_params (dict): Kwargs passed to ``build_label()``.
            extract_features_params (dict): Kwargs passed to
                ``extract_features()`` excluding ``overwrite``, ``n_jobs``,
                ``verbose``.

        Returns:
            dict: Canonical identity dict ready for hashing.
        """
        return {
            "class": cls.__name__,
            "nslc": nslc,
            "training_hash": training_hash,
            "tremor": cls.tremor_fingerprint(tremor_df),
            "constructor": {
                "start_date": start_date,
                "end_date": end_date,
                "window_size": window_size,
            },
            "build_label": build_label_params,
            "extract_features": extract_features_params,
        }

    def build_label(
        self, window_step: int, window_step_unit: Literal["minutes", "hours"]
    ) -> Self:
        """Build the unlabelled forecast window grid.

        Constructs evenly spaced windows between ``start_date`` and
        ``end_date`` via ``construct_windows()`` and assigns each row an
        integer ``id``. The result is cached as a CSV under ``features_dir``
        and stored on ``self.labels`` as a ``pd.Series`` with a
        ``pd.DatetimeIndex``. When the CSV already exists and ``overwrite``
        is ``False``, the cached grid is loaded instead of rebuilt.

        Args:
            window_step (int): Step size between consecutive windows. Must be
                strictly greater than ``0``.
            window_step_unit (Literal["minutes", "hours"]): Unit for
                ``window_step``.

        Returns:
            Self: The current instance, for method chaining.

        Raises:
            ValueError: If ``window_step`` is not strictly greater than ``0``.
            ValueError: If a cached CSV is loaded but lacks an ``id`` column.
            ValueError: If the constructed window grid is empty (typically
                caused by an inverted date range or a step larger than the
                forecast period).
        """
        if window_step <= 0:
            raise ValueError("window_step must be > 0.")

        self.window_step = window_step
        self.window_step_unit = window_step_unit

        filename = (
            f"features-label_{self.basename}_step-{window_step}-{window_step_unit}"
        )
        label_csv = os.path.join(self.features_dir, f"{filename}.csv")
        ensure_dir(self.features_dir)

        if os.path.exists(label_csv) and not self.overwrite:
            label_df = pd.read_csv(label_csv, index_col=0, parse_dates=True)

            # Backward compat: legacy cached forecast grids predate the
            # ``is_erupted`` placeholder column.
            if "is_erupted" not in label_df.columns:
                label_df["is_erupted"] = 0

            self._labels = label_df
            self.labels = label_df["id"]
            self.labels_csv = label_csv

            return self

        label_df = construct_windows(
            start_date=self.start_date,
            end_date=self.end_date,
            window_step=window_step,
            window_step_unit=window_step_unit,
        )

        if label_df.empty:
            raise ValueError(
                f"Labels is empty. Check your start date {self.start_date} "
                f"and end date {self.end_date}, window step {window_step} and "
                f"window step unit {window_step_unit}."
            )

        label_df.to_csv(label_csv, index=True)

        # Label with ``is_erupted`` values are 0.
        self._labels = label_df
        self.labels = label_df["id"]
        self.labels_csv = label_csv

        if self.verbose:
            logger.info(f"Label for prediction: {label_csv}")

        return self

    def extract_features(
        self,
        select_tremor_columns: list[str] | None = None,
        save_tremor_matrix_per_method: bool = False,
        exclude_features: list[str] | None = None,
        overwrite: bool = False,
        n_jobs: int | None = None,
        verbose: bool | None = None,
    ) -> Self:
        """Extract tsfresh features over the forecast window grid.

        Builds the windowed tremor matrix aligned to the labels produced by
        :meth:`build_label`, then runs ``FeaturesBuilder`` in prediction mode
        (no relevance filtering). Populates ``self.features_df`` and
        ``self.features_csv``. When features were already extracted on this
        instance, the method early-returns without re-running tsfresh.

        When ``TremorMatrixBuilder`` drops windows that fail
        ``minimum_completion``, the surviving feature ids become a strict
        subset of the label grid built by :meth:`build_label`. The method
        detects the mismatch and narrows ``self._labels`` / ``self.labels``
        to the surviving ids, rewriting ``self.labels_csv`` on disk so the
        invariant *"``self._labels`` and ``self.features_df`` share length
        and id space"* holds for every downstream consumer.

        Args:
            select_tremor_columns (list[str] | None, optional): Tremor column
                names to include. All columns are used when ``None``. Defaults
                to ``None``.
            save_tremor_matrix_per_method (bool, optional): Save per-column
                tremor matrices under ``per_method/`` when ``True``. Defaults
                to ``False``.
            exclude_features (list[str] | None, optional): tsfresh feature
                names to drop after extraction. Defaults to ``None``.
            overwrite (bool, optional): Overwrite existing matrix and feature
                files when ``True``. Defaults to ``False``.
            n_jobs (int | None, optional): Worker count for tsfresh extraction.
                Falls back to ``self.n_jobs`` when ``None``. Defaults to
                ``None``.
            verbose (bool | None, optional): Override ``self.verbose`` for
                this call only. Defaults to ``None``.

        Returns:
            Self: The current instance, for method chaining.

        Raises:
            ValueError: If ``build_label()`` has not been called first.
        """
        if self._labels.empty:
            raise ValueError("Please run build_label() first.")

        if not self.features_df.empty and self.features_csv is not None:
            if self.verbose:
                logger.info(f"Features already extracted: {self.features_csv}")
            return self

        features_builder = self._build_features(
            label_df=self._labels,
            output_dir=self.prediction_dir,
            features_dir=self.features_dir,
            select_tremor_columns=select_tremor_columns,
            save_tremor_matrix_per_method=save_tremor_matrix_per_method,
            save_tremor_matrix_per_id=False,
            overwrite=overwrite,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.features_df = features_builder.extract_features(
            select_tremor_columns=select_tremor_columns,
            exclude_features=exclude_features,
        )

        self.features_csv = features_builder.csv

        # ``TremorMatrixBuilder`` drops windows that fail ``minimum_completion``,
        # so narrow the label grid to the surviving feature ids here — this
        # invariant lets every downstream consumer (forecast, evaluation,
        # explanation) treat ``self._labels`` and ``self.features_df`` as
        # aligned without a separate re-check.
        if len(self._labels) != len(self.features_df):
            self._labels = self._labels[
                self._labels["id"].isin(self.features_df.index)
            ]
            self._labels.to_csv(self.labels_csv, index=True)
            self.labels = self._labels["id"]

        return self

    def forecast(
        self,
        save_seed_result: bool = True,
        plot_threshold: float = 0.5,
        plot_title: str | None = None,
        plot_pdf: bool = True,
        **plot_kwargs,
    ) -> pd.DataFrame:
        """Run forecast inference and render the forecast plot.

        Calls :meth:`ClassifierEnsemble.predict_with_uncertainty` over the
        extracted features, assembles per-classifier and consensus
        probabilities into a single DataFrame indexed by forecast datetime,
        writes the result CSV under ``self.output_dir``, and renders the
        forecast plot via :meth:`_plot_forecast`.

        Args:
            save_seed_result (bool, optional): Persist per-seed probability
                CSVs under ``result_dir/{classifier_name}/`` when ``True``.
                Defaults to ``True``.
            plot_threshold (float, optional): Decision threshold drawn as a
                horizontal reference line on the forecast plot. Defaults to
                ``0.5``.
            plot_title (str | None, optional): Title to render on the forecast
                plot. Defaults to ``None``.
            plot_pdf (bool, optional): Save a PDF copy of the forecast plot
                with embedded TrueType fonts in addition to the PNG. Defaults
                to ``True``.
            **plot_kwargs: Extra keyword arguments forwarded to
                :func:`plot_forecast`.

        Returns:
            pd.DataFrame: Forecast results indexed by datetime with one column
                per ``{classifier}_{probability|uncertainty|prediction|confidence}``
                plus the four ``consensus_*`` columns.

        Raises:
            ValueError: If ``build_label()`` has not been called first.
            ValueError: If ``extract_features()`` has not been called first.
        """
        if self.labels.empty or self._labels.empty or self.labels_csv is None:
            raise ValueError("Please run build_label() first.")

        if self.features_df.empty and self.features_csv is None:
            raise ValueError(
                "Features (matrix) dataframe (features_df) is empty. "
                "Please run extract_features() first."
            )

        self.create_directories()

        results = self.ClassifierEnsemble.predict_with_uncertainty(
            X=self.features_df,
            save=save_seed_result,
            output_dir=self.result_dir,
            overwrite=self.overwrite,
            verbose=self.verbose,
        )

        df_forecast = pd.DataFrame(results, index=self.features_df.index)
        csv_path = os.path.join(
            self.result_dir, f"forecast-results_{self.basename}.csv"
        )

        df_forecast = set_datetime_index(self._labels, df_forecast)

        df_forecast.to_csv(csv_path)
        logger.info(f"Predictions saved to: {csv_path}")

        self._plot_forecast(
            df_forecast,
            plot_threshold,
            title=plot_title,
            plot_pdf=plot_pdf,
            **plot_kwargs,
        )

        self.results = df_forecast

        self.save()

        return df_forecast

    def _plot_forecast(
        self,
        df: pd.DataFrame,
        threshold: float,
        title: str | None = None,
        plot_pdf: bool = False,
        **plot_kwargs: Any,
    ) -> None:
        """Render the forecast plot to PNG and optionally PDF.

        Calls :func:`plot_forecast` to produce the figure, saves it as PNG
        under ``prediction_dir/figures/``, and optionally writes a PDF copy
        with ``pdf.fonttype=42`` so text remains selectable in vector
        editors. Stores the resulting artefact path on
        ``self.forecast_plot_path``.

        Args:
            df (pd.DataFrame): Forecast results with one row per forecast
                datetime and the per-classifier plus consensus probability
                columns produced by :meth:`forecast`.
            threshold (float): Decision threshold drawn as a horizontal
                reference line.
            title (str | None, optional): Title rendered on the plot. Defaults
                to ``None``.
            plot_pdf (bool, optional): Save a PDF copy in addition to the PNG
                when ``True``. Defaults to ``False``.
            **plot_kwargs: Extra keyword arguments forwarded to
                :func:`plot_forecast`.
        """
        fig = plot_forecast(
            df=df,
            label_df=self._labels,
            threshold=threshold,
            title=title,
            **plot_kwargs,
        )

        figure_dir = os.path.join(self.prediction_dir, "figures")
        os.makedirs(figure_dir, exist_ok=True)

        path = os.path.join(figure_dir, f"forecast_{self.basename}.png")
        fig.savefig(
            path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor=None
        )

        logger.info(f"Forecast plot saved to: {path}")

        if plot_pdf:
            path = os.path.join(figure_dir, f"forecast_{self.basename}.pdf")

            # Type 42 embeds TrueType fonts — text stays selectable and
            # renders consistently in all PDF viewers and vector editors.
            with matplotlib.rc_context({"pdf.fonttype": 42}):
                fig.savefig(
                    path,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor=None,
                    metadata=pdf_metadata(
                        title=f"Eruption Forecast: {self.start_date_str} to {self.end_date_str}"
                    ),
                )

        plt.close(fig)
        self.forecast_plot_path = path
