"""Forecast inference using trained classifier ensembles.

Provides :class:`ModelPredictor`, the production inference component of the
eruption forecasting pipeline.  It accepts a trained
:class:`~eruption_forecast.model.seed_ensemble.SeedEnsemble`,
:class:`~eruption_forecast.model.classifier_ensemble.ClassifierEnsemble`, or a
mapping of classifier names to registry CSV paths, and produces eruption
probability forecasts over a specified date range.

Key capabilities:
    - ``predict_proba(tremor_data, start_date, end_date, ...)``: Slice tremor
      data into sliding windows, extract features, run inference through every
      loaded ensemble, and return a DataFrame with per-classifier columns
      (``{name}_eruption_probability``, ``{name}_uncertainty``,
      ``{name}_confidence``, ``{name}_prediction``) plus ``consensus_*``
      aggregates across classifiers.
    - Automatically constructs sliding windows aligned to the requested step
      size (in minutes or hours).
    - Saves forecast CSV and optional plot to the configured output directory.

Design notes:
    - ``predict()`` and ``predict_best()`` have been removed; ``predict_proba()``
      is the single inference entry point.
    - The ``model_name`` constructor parameter is used to label output columns
      and file names (defaults to ``"model"``).
    - The non-interactive matplotlib backend is set at module import time to
      ensure safe use in background worker threads.
"""

import os
from typing import Any, Literal
from datetime import datetime, timedelta

import joblib
import matplotlib

from eruption_forecast.config.constants import MATPLOTLIB_BACKEND
from eruption_forecast.utils.date_utils import set_datetime_index
from eruption_forecast.utils.formatting import slugify, pdf_metadata


matplotlib.use(
    MATPLOTLIB_BACKEND
)  # Must be called before pyplot import — non-interactive backend safe for worker threads
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eruption_forecast.logger import logger
from eruption_forecast.utils.window import construct_windows
from eruption_forecast.utils.pathutils import ensure_dir, resolve_output_dir
from eruption_forecast.utils.date_utils import normalize_dates
from eruption_forecast.tremor.tremor_data import TremorData
from eruption_forecast.model.seed_ensemble import SeedEnsemble
from eruption_forecast.plots.forecast_plots import plot_forecast
from eruption_forecast.features.features_builder import FeaturesBuilder
from eruption_forecast.model.classifier_ensemble import ClassifierEnsemble
from eruption_forecast.features.tremor_matrix_builder import TremorMatrixBuilder


class ModelPredictor:
    """Evaluate or forecast with one or more sets of trained full-dataset models.

    Loads models produced by ``ModelTrainer.train()`` and runs forecast
    inference. Supports single-model or multi-model consensus predictions by
    aggregating across classifiers and seeds with uncertainty quantification.
    Use :meth:`predict_proba` for unlabelled forecast mode.

    Attributes:
        start_date (datetime): Start of the prediction window.
        end_date (datetime): End of the prediction window.
        start_date_str (str): Start date as "YYYY-MM-DD" string.
        end_date_str (str): End date as "YYYY-MM-DD" string.
        trained_models (dict[str, str]): Dictionary mapping classifier names
            to their trained model registry CSV paths.
        overwrite (bool): Whether to re-compute cached files.
        n_jobs (int): Number of parallel jobs for feature extraction.
        output_dir (str): Root directory for prediction outputs.
        tremor_dir (str): Directory for tremor data cache.
        features_dir (str): Directory for feature outputs.
        extracted_dir (str): Directory for extracted features.
        figures_dir (str): Directory for plots.
        verbose (bool): Enable verbose logging.
        debug (bool): Enable debug mode.

    Args:
        start_date (str | datetime): Start of the prediction window (YYYY-MM-DD).
        end_date (str | datetime): End of the prediction window (YYYY-MM-DD).
        trained_models (str | dict[str, str]): Either a single path to a
            ``trained_model_{suffix}.csv`` file produced by ``ModelTrainer.train()``
            or a merged ``.pkl`` file produced by ``ModelTrainer.merge_models()``;
            or a dict mapping classifier name to its CSV or ``.pkl`` path
            (e.g., ``{"rf": "...", "xgb": "...", "voting": "..."}``).
        overwrite (bool, optional): Re-compute cached files if True.
            Defaults to False.
        n_jobs (int, optional): Number of parallel jobs for feature extraction.
            Defaults to 1.
        output_dir (str | None, optional): Root directory for outputs.
            Defaults to ``<cwd>/output/predictions``.
        root_dir (str | None, optional): Project root used to resolve relative
            ``output_dir`` paths. Defaults to None.
        verbose (bool, optional): Print extra progress information.
            Defaults to False.
        debug (bool, optional): Enable debug mode. Defaults to False.

    Raises:
        ValueError: If start_date >= end_date.
        FileNotFoundError: If any trained model registry CSV does not exist.

    Examples:
        >>> predictor = ModelPredictor(
        ...     start_date="2025-03-20",
        ...     end_date="2025-03-22",
        ...     trained_models={
        ...         "rf":     "output/trainings/rf/trained_model_rf.csv",
        ...         "xgb":    "output/trainings/xgb/trained_model_xgb.csv",
        ...         "voting": "output/trainings/voting/trained_model_voting.csv",
        ...     },
        ... )
        >>> df_forecast = predictor.predict_proba(
        ...     tremor_data="output/tremor/tremor.csv",
        ...     window_size=2,
        ...     window_step=6,
        ...     window_step_unit="hours",
        ... )
        >>> # Columns: rf_eruption_probability, xgb_eruption_probability, ...,
        >>> #          consensus_eruption_probability, consensus_confidence, ...
        >>> print(df_forecast[["consensus_probability", "consensus_confidence"]])
    """

    def __init__(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        trained_models: str | dict[str, str],
        overwrite: bool = False,
        model_name: str = "model",
        n_jobs: int = 1,
        output_dir: str | None = None,
        root_dir: str | None = None,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize the ModelPredictor with date range, trained model registry, and output settings.

        Normalises dates to cover full calendar days, resolves output directory structure,
        and stores the trained model registry path(s). No models are loaded until
        predict(), predict_best(), or predict_proba() is called.

        Args:
            start_date (str | datetime): Start of the prediction window in "YYYY-MM-DD"
                format or as a datetime. Time is normalised to 00:00:00.
            end_date (str | datetime): End of the prediction window in "YYYY-MM-DD"
                format or as a datetime. Time is normalised to 23:59:59.
            trained_models (str | dict[str, str]): Path to a single trained model
                registry CSV or merged ``.pkl`` file; or a dict mapping classifier
                names to their registry CSV paths or per-classifier ``.pkl`` paths
                for multi-model consensus mode. Accepted ``.pkl`` types are
                ``ClassifierEnsemble`` (all classifiers bundled), ``SeedEnsemble``
                (single classifier), and plain ``dict[str, SeedEnsemble]``
                (backward-compatible, auto-wrapped into ``ClassifierEnsemble``).
            overwrite (bool, optional): Re-compute cached intermediate files.
                Defaults to False.
            model_name (str, optional): Model name related to trained models.
                Defaults to "model".
            n_jobs (int, optional): Number of parallel jobs for feature extraction.
                Defaults to 1.
            output_dir (str | None, optional): Root directory for prediction outputs.
                Defaults to ``<root_dir>/output/predictions``. Defaults to None.
            root_dir (str | None, optional): Anchor directory for relative path
                resolution. Defaults to None.
            verbose (bool, optional): Emit progress log messages. Defaults to False.
            debug (bool, optional): Emit debug log messages. Defaults to False.

        Raises:
            ValueError: If start_date >= end_date.
            FileNotFoundError: If any trained model registry CSV does not exist.
        """
        # ------------------------------------------------------------------
        # Set DEFAULT parameter
        # ------------------------------------------------------------------
        start_date, end_date, start_date_str, end_date_str = normalize_dates(
            start_date, end_date
        )

        # Output training dir: ``<root_dir>/output/trainings``
        output_dir = resolve_output_dir(
            output_dir,
            root_dir,
            os.path.join("output"),
        )
        output_dir = os.path.join(output_dir, "predictions")

        tremor_dir = os.path.join(output_dir, "tremor")
        features_dir = os.path.join(output_dir, "features")
        extracted_dir = os.path.join(features_dir, "extracted")
        figures_dir = os.path.join(output_dir, "figures")

        ensure_dir(figures_dir)

        # ------------------------------------------------------------------
        # Set DEFAULT properties
        # ------------------------------------------------------------------
        self.start_date: datetime = start_date
        self.end_date: datetime = end_date
        self.overwrite = overwrite
        self.n_jobs = n_jobs
        self.output_dir = output_dir
        self.verbose = verbose
        self.debug = debug

        # ------------------------------------------------------------------
        # Set ADDITIONAL properties (derived values)
        # ------------------------------------------------------------------
        self.start_date_str = start_date_str
        self.end_date_str = end_date_str
        self.tremor_dir = tremor_dir
        self.features_dir = features_dir
        self.extracted_dir = extracted_dir
        self.figures_dir = figures_dir

        # ------------------------------------------------------------------
        # Will be set after get_features_dataframe() method called
        # ------------------------------------------------------------------
        self.FeaturesBuilder: FeaturesBuilder | None = None
        self.features_df: pd.DataFrame | None = None
        self.features_csv: str | None = None

        # ------------------------------------------------------------------
        # Will be set after build_label() method called
        # ------------------------------------------------------------------
        self.window_step: int | None = None
        self.window_step_unit: Literal["minutes", "hours"] | None = None
        self.select_tremor_columns: list[str] | None = None

        # ------------------------------------------------------------------
        # Will be set after get_tremor_date() method called
        # ------------------------------------------------------------------
        self.TremorData: TremorData | None = None
        self.tremor_start_date: datetime | None = None
        self.tremor_end_date: datetime | None = None
        self.tremor_df: pd.DataFrame | None = None
        self.tremor_matrix_df: pd.DataFrame | None = None

        # ------------------------------------------------------------------
        # Will be set after build_prediction_labels() method called
        # ------------------------------------------------------------------
        self.labels_df: pd.DataFrame | pd.Series | None = None
        self.basename = f"{self.start_date_str}_{self.end_date_str}"

        model_name = slugify(model_name).lower()
        self.model_name = model_name

        # Normalize trained_models to one of two internal representations:
        #   - dict[str, SeedEnsemble]     for merged .pkl files (direct path)
        #   - _registry_csv_paths         for CSV registry paths (merged lazily in predict_proba)
        self._classifier_ensemble: ClassifierEnsemble | None = None
        self._registry_csv_paths: dict[str, str] = {}
        if isinstance(trained_models, str):
            if trained_models.endswith(".pkl"):
                loaded = joblib.load(trained_models)
                if isinstance(loaded, ClassifierEnsemble):
                    # Multi-classifier merged pkl
                    self._classifier_ensemble: ClassifierEnsemble = loaded
                    self.trained_models: dict[str, SeedEnsemble] = dict(
                        loaded.ensembles.items()
                    )
                elif isinstance(loaded, SeedEnsemble):
                    # Single-classifier merged pkl
                    self.trained_models = {model_name: loaded}
                elif isinstance(loaded, dict):
                    # Backward-compat: plain dict[str, SeedEnsemble] pkl — auto-wrap
                    self._classifier_ensemble: ClassifierEnsemble = (
                        ClassifierEnsemble.from_seed_ensembles(loaded)
                    )
                    self.trained_models = loaded
                else:
                    raise ValueError(f"Unrecognised object type in pkl: {type(loaded)}")
            else:
                # CSV registry path — defer merging to predict_proba()
                self.trained_models = {}
                self._registry_csv_paths = {model_name: trained_models}
        elif isinstance(trained_models, dict):
            first_val = next(iter(trained_models.values()))
            if isinstance(first_val, str) and first_val.endswith(".pkl"):
                # Values are paths to per-classifier merged pkl files
                self.trained_models = {}
                for model_name, path in trained_models.items():
                    loaded = joblib.load(path)
                    if isinstance(loaded, SeedEnsemble):
                        self.trained_models[model_name] = loaded
                    elif isinstance(loaded, dict):
                        # Unlikely but handle gracefully
                        self.trained_models.update(loaded)
                    else:
                        raise ValueError(
                            f"Unrecognised object type in {path}: {type(loaded)}"
                        )
                self._classifier_ensemble = ClassifierEnsemble.from_seed_ensembles(
                    self.trained_models
                )
            elif isinstance(first_val, str):
                # Values are paths to CSV registry files — defer merging to predict_proba()
                self.trained_models = {}
                self._registry_csv_paths = dict(trained_models)
            else:
                raise ValueError(
                    f"Unrecognised trained_models value type: {type(first_val)}"
                )

        total_models = len(self.trained_models) + len(self._registry_csv_paths)
        self._multi_model: bool = total_models > 1

        # Will be set after predict_proba() is called
        self.df: pd.DataFrame | None = None
        self.forecast_plot_path: str | None = None

        if verbose:
            logger.info(f"Models registered: {total_models}")

    @property
    def model_names(self) -> list[str]:
        """Return the names of all registered classifier types.

        Returns:
            list[str]: Classifier names drawn from ``trained_models`` keys,
                or from ``_registry_csv_paths`` keys when no models are loaded yet.
        """
        return list(self.trained_models.keys()) or list(self._registry_csv_paths.keys())

    def build_prediction_labels(
        self,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
    ) -> pd.DataFrame:
        """Build a future-labels DataFrame with datetime index and sequential IDs.

        Constructs one row per sliding window between ``self.start_date`` and
        ``self.end_date`` using ``construct_windows``, assigns a sequential
        ``id`` column, and saves the result to disk for caching.

        Args:
            window_step (int): Step size between consecutive windows.
            window_step_unit (Literal["minutes", "hours"]): Unit of ``window_step``.

        Returns:
            pd.DataFrame: Labels dataframe with a ``DatetimeIndex`` and an
            ``id`` column assigning a sequential integer to each window.
        """
        self.window_step = window_step
        self.window_step_unit = window_step_unit
        filename = f"{self.basename}_step-{window_step}-{window_step_unit}"

        ensure_dir(self.features_dir)
        futures_labels_filepath = os.path.join(
            self.features_dir,
            f"prediction-labels_{filename}.csv",
        )

        # Skip if file exists
        if os.path.exists(futures_labels_filepath) and not self.overwrite:
            self.labels_df = pd.read_csv(
                futures_labels_filepath, index_col=0, parse_dates=True
            )
            return self.labels_df

        futures_labels_df: pd.DataFrame = construct_windows(
            start_date=self.start_date,
            end_date=self.end_date,
            window_step=window_step,
            window_step_unit=window_step_unit,
        )
        futures_labels_df["id"] = range(len(futures_labels_df))

        self.labels_df = futures_labels_df
        futures_labels_df.to_csv(futures_labels_filepath, index=True)

        return futures_labels_df

    def build_tremor_matrix(
        self,
        tremor_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        window_size: int = 2,
        select_tremor_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Build tremor matrix from tremor dataframe and labels.

        Args:
            tremor_df (pd.DataFrame): Tremor dataframe with a ``DatetimeIndex``.
            labels_df (pd.DataFrame): Labels dataframe produced by
                :meth:`build_prediction_labels`.
            window_size (int, optional): Window size in days. Defaults to 2.
            select_tremor_columns (list[str] | None, optional): Subset of
                tremor columns to include.  Defaults to None (all columns).

        Returns:
            pd.DataFrame: Unified tremor matrix with ``id``, ``datetime``,
            and tremor columns, one row per (window, time-step) pair.

        Raises:
            ValueError: If ``tremor_df`` is empty or ``labels_df`` is None.
        """

        if isinstance(labels_df, pd.Series):
            labels_df = labels_df.to_frame()

        if tremor_df is None:
            raise ValueError("Parameter tremor_df not provided.")

        if tremor_df.empty:
            raise ValueError("Tremor dataframe is empty.")

        if labels_df is None:
            raise ValueError(
                "Parameter labels_df not provided. Please run build_prediction_labels() first."
            )

        ensure_dir(self.tremor_dir)
        tremor_matrix_filename = (
            f"tremor_matrix_unified_{self.basename}_ws-{window_size}.csv"
        )
        filepath = os.path.join(self.tremor_dir, tremor_matrix_filename)

        if os.path.exists(filepath) and not self.overwrite:
            if self.verbose:
                logger.info(f"Loading tremor matrix from {filepath}")
            self.tremor_matrix_df = pd.read_csv(filepath, parse_dates=True)
            return self.tremor_matrix_df

        tremor_df = tremor_df.loc[
            self.start_date - timedelta(days=window_size) : self.end_date
        ]

        tremor_matrix_builder = TremorMatrixBuilder(
            tremor_df=tremor_df,
            label_df=labels_df,
            output_dir=self.tremor_dir,
            window_size=window_size,
            overwrite=self.overwrite,
            verbose=self.verbose,
        ).build(
            select_tremor_columns=select_tremor_columns,
            save_tremor_matrix_per_method=True,
            save_tremor_matrix_per_id=False,
        )

        self.tremor_matrix_df: pd.DataFrame = tremor_matrix_builder.df
        tremor_matrix_builder.df.to_csv(filepath, index=False)

        return self.tremor_matrix_df

    def extract_features(
        self,
        tremor_matrix_df: pd.DataFrame,
        use_relevant_features: bool = True,
        select_tremor_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Extract features from tremor matrix using tsfresh.

        Args:
            tremor_matrix_df (pd.DataFrame): Unified tremor matrix produced by
                :meth:`build_tremor_matrix`.
            use_relevant_features (bool, optional): Whether to filter to
                statistically relevant features only.  Defaults to True.
            select_tremor_columns (list[str] | None, optional): Subset of
                tremor columns to extract features from.  Defaults to None
                (all columns).

        Returns:
            pd.DataFrame: Extracted tsfresh features, one row per window.

        Raises:
            ValueError: If ``self.tremor_matrix_df`` is None (i.e.
                :meth:`build_tremor_matrix` has not been called yet).
        """
        ensure_dir(self.extracted_dir)

        if self.tremor_matrix_df is None:
            raise ValueError(
                "Parameter tremor_matrix_df not provided. Please run build_tremor_matrix() first."
            )

        features_builder = FeaturesBuilder(
            tremor_matrix_df=tremor_matrix_df,
            output_dir=self.features_dir,
            overwrite=self.overwrite,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        extracted_features_df = features_builder.extract_features(
            use_relevant_features=use_relevant_features,
            select_tremor_columns=select_tremor_columns,
            prefix_filename="predictions",
        )

        self.FeaturesBuilder = features_builder
        self.features_csv = features_builder.csv

        return extracted_features_df

    def get_tremor_dataframe(self, tremor_data: str | pd.DataFrame) -> pd.DataFrame:
        """Get tremor dataframe.

        Args:
            tremor_data (str | pd.DataFrame): Tremor data as a filepath or
                pre-loaded DataFrame.

        Returns:
            pd.DataFrame: Full tremor DataFrame loaded from the source, with a
                DatetimeIndex.

        Raises:
            ValueError: If ``tremor_data`` is neither a filepath string nor a
                DataFrame, or if no tremor data falls within the requested
                date range.
            TypeError: If the resulting tremor DataFrame does not have a
                ``DatetimeIndex``.
        """

        if isinstance(tremor_data, str):
            _tremor_data: TremorData = TremorData.from_csv(tremor_data)
        elif isinstance(tremor_data, pd.DataFrame):
            _tremor_data: TremorData = TremorData(df=tremor_data)
        else:
            raise ValueError(
                f"Parameter tremor_data only accepts valid tremor data filepath "
                f"or pandas DataFrame type. Your tremor data type is: {type(tremor_data)}"
            )

        tremor_df = _tremor_data.df
        if not isinstance(tremor_df.index, pd.DatetimeIndex):
            raise TypeError("tremor_df index is not pd.DatetimeIndex")

        tremor_start_date = _tremor_data.start_date
        tremor_end_date = _tremor_data.end_date
        tremor_start_date_str = tremor_start_date.strftime("%Y-%m-%d")
        tremor_end_date_str = tremor_end_date.strftime("%Y-%m-%d")

        if tremor_df.empty:
            raise ValueError(
                f"No tremor data found between your date prediction from {self.start_date_str} to {self.end_date_str}. Available tremor data from {tremor_start_date_str} to {tremor_end_date_str}."
            )

        self.TremorData = _tremor_data
        self.tremor_df = _tremor_data.df
        self.tremor_start_date = tremor_start_date
        self.tremor_end_date = tremor_end_date

        return tremor_df

    def get_features_dataframe(
        self,
        tremor_df: pd.DataFrame,
        window_size: int,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        use_relevant_features: bool = True,
        select_tremor_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Build and extract tsfresh features for unlabelled forecast windows.

        Orchestrates three sequential steps: ``build_prediction_labels`` →
        ``build_tremor_matrix`` → ``extract_features``, returning the final
        features DataFrame ready for model inference.

        Args:
            tremor_df (pd.DataFrame): Tremor dataframe with a ``DatetimeIndex``,
                as returned by :meth:`get_tremor_dataframe`.
            window_size (int): Window size in days.
            window_step (int): Step size between consecutive windows.
            window_step_unit (Literal["minutes", "hours"]): Unit of ``window_step``.
            use_relevant_features (bool, optional): Whether to filter to
                statistically relevant features only.  Defaults to True.
            select_tremor_columns (list[str] | None, optional): Subset of
                tremor columns to use.  Defaults to None (all columns).

        Returns:
            pd.DataFrame: Extracted features dataframe, one row per window.
        """
        future_labels = self.build_prediction_labels(
            window_step=window_step, window_step_unit=window_step_unit
        )

        tremor_matrix_df = self.build_tremor_matrix(
            tremor_df=tremor_df,
            labels_df=future_labels,
            window_size=window_size,
            select_tremor_columns=select_tremor_columns,
        )

        features_df = self.extract_features(
            tremor_matrix_df=tremor_matrix_df,
            use_relevant_features=use_relevant_features,
        )

        self.features_df = features_df

        return features_df

    def predict_proba(
        self,
        tremor_data: str | pd.DataFrame,
        window_size: int,
        window_step: int,
        *,
        window_step_unit: Literal["minutes", "hours"],
        use_relevant_features: bool = True,
        select_tremor_columns: list[str] | None = None,
        save_predictions: bool = True,
        threshold: float = 0.7,
        title: str | None = None,
        plot_pdf: bool = False,
        **plot_kwargs: Any,
    ) -> pd.DataFrame:
        """Forecast eruption probability for UNLABELLED future windows.

        For each classifier and each of its seed models,
        ``model.predict_proba()`` is called on the future feature set.
        Results are aggregated in two stages:

        1. **Within each classifier** — probabilities are averaged across all
           seeds of that classifier to produce per-classifier columns
           ``{name}_eruption_probability``, ``{name}_uncertainty``,
           ``{name}_confidence``, and ``{name}_prediction``.

        2. **Across classifiers (consensus)** — the per-classifier mean
           probabilities are averaged to produce ``consensus_*`` columns.
           Consensus confidence is the fraction of classifiers voting with
           the majority.

        When only a single classifier was registered (``trained_models`` was a
        plain string), per-classifier columns use the prefix ``"model_"`` and
        the consensus columns equal those values exactly. The result is
        saved as ``predictions.csv``.

        Args:
            tremor_data (str | pd.DataFrame): Tremor data in CSV or dataframe.
            window_size (int): Window size to use.
            window_step (int): Window step size to use.
            window_step_unit (Literal["minutes", "hours"]): Window step unit to use.
            use_relevant_features (bool, optional): Whether to use relevant features. Defaults to True.
            select_tremor_columns (list[str] | None): List of tremor columns to use. Defaults to None.
            save_predictions (bool, optional): Save predictions result. Defaults to True.
            threshold (float, optional): Probability threshold for eruption classification in the
                forecast plot. Defaults to 0.7.
            title (str | None, optional): Forecast plot title. Defaults to None.
            plot_pdf (bool, optional): Also save the forecast plot as a PDF file
                alongside the PNG. ``dpi`` is ignored for PDF (vector format);
                ``bbox_inches``, ``facecolor``, and ``edgecolor`` still apply.
                Defaults to False.
            **plot_kwargs: Additional keyword arguments forwarded to
                :func:`~eruption_forecast.plots.forecast_plots.plot_forecast`
                (e.g. ``fig_width``, ``fig_height``, ``rolling_window``,
                ``x_days_interval``, ``eruption_dates``, ``y_max``, ``legend_n_cols``).

        Returns:
            pd.DataFrame: Index matches the future features index.  Columns
            include per-classifier metrics and consensus metrics.
        """
        tremor_data = self.get_tremor_dataframe(tremor_data)

        features_df = self.get_features_dataframe(
            tremor_df=tremor_data,
            window_size=window_size,
            window_step=window_step,
            window_step_unit=window_step_unit,
            use_relevant_features=use_relevant_features,
            select_tremor_columns=select_tremor_columns,
        )

        cols: dict[str, np.ndarray] = {}
        result_dir = os.path.join(self.output_dir, "results")

        if self._classifier_ensemble is None and self._registry_csv_paths:
            ensembles: dict[str, SeedEnsemble] = {}
            for name, csv_path in self._registry_csv_paths.items():
                ensembles[name] = SeedEnsemble.from_registry(csv_path)
            self._classifier_ensemble = ClassifierEnsemble.from_seed_ensembles(
                ensembles
            )

        (
            consensus_probability,
            consensus_uncertainty,
            consensus_prediction,
            consensus_confidence,
        ) = self._forecast_with_classifier_ensemble(
            features_df=features_df,
            cols=cols,
            save_predictions=save_predictions,
            result_dir=result_dir,
        )

        cols["consensus_probability"] = consensus_probability
        cols["consensus_uncertainty"] = consensus_uncertainty
        cols["consensus_prediction"] = consensus_prediction
        cols["consensus_confidence"] = consensus_confidence

        df_forecast = pd.DataFrame(cols, index=features_df.index)
        csv_path = os.path.join(
            self.output_dir, f"result_all_model_predictions_{self.basename}.csv"
        )

        # Replace "id" index with "datetime" index
        if self.labels_df is not None:
            df_forecast = set_datetime_index(self.labels_df, df_forecast)

        # Always save prediction results
        df_forecast.to_csv(csv_path)
        logger.info(f"Predictions saved to: {csv_path}")

        self._plot_forecast(
            df_forecast, threshold, title=title, plot_pdf=plot_pdf, **plot_kwargs
        )

        self.df = df_forecast
        return df_forecast

    def _forecast_with_classifier_ensemble(
        self,
        features_df: pd.DataFrame,
        cols: dict[str, np.ndarray],
        save_predictions: bool = False,
        result_dir: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Delegate inference and consensus to a ClassifierEnsemble.

        Calls ``ClassifierEnsemble.predict_with_uncertainty`` for the fast
        path where all classifiers are bundled in a single object.

        Args:
            features_df (pd.DataFrame): Future feature matrix.
            cols (dict[str, np.ndarray]): Mutable column accumulator for the
                forecast DataFrame.
            save_predictions (bool, optional): If ``True``, save per-seed
                predictions to CSV files. Defaults to ``False``.
            result_dir (str | None, optional): Base directory for per-seed CSV
                output. Used only when ``save_predictions`` is ``True``.
                Defaults to ``None``.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple of
            ``(consensus_probability, consensus_uncertainty, consensus_prediction, consensus_confidence)``.
        """
        assert self._classifier_ensemble is not None
        seeds_dir = os.path.join(result_dir, "seeds") if result_dir else None
        (
            consensus_probability,  # mean P(eruption) averaged across all classifiers
            consensus_uncertainty,  # std of P(eruption) across classifiers
            consensus_prediction,  # mean of per-classifier binary votes (continuous, [0, 1])
            consensus_confidence,  # CI-like metric: 1.96 * sqrt(p * (1 - p) / n_classifiers)
            clf_results,  # a dict with keys ``"probability"``, ``"uncertainty"``, ``"prediction"``,``"confidence"``
        ) = self._classifier_ensemble.predict_with_uncertainty(
            features_df,
            save=save_predictions,
            output_dir=seeds_dir,
            overwrite=self.overwrite,
            verbose=self.verbose,
        )

        for model_name, clf_result in clf_results.items():
            cols[f"{model_name}_probability"] = clf_result["probability"]
            cols[f"{model_name}_uncertainty"] = clf_result["uncertainty"]
            cols[f"{model_name}_prediction"] = clf_result["prediction"]
            cols[f"{model_name}_confidence"] = clf_result["confidence"]
            logger.info(
                f"{model_name} — mean probability: {clf_result['probability'].mean():.4f}, "
                f"mean prediction: {clf_result['prediction'].mean():.4f}"
            )

        return (
            consensus_probability,
            consensus_uncertainty,
            consensus_prediction,
            consensus_confidence,
        )

    def _plot_forecast(
        self,
        df: pd.DataFrame,
        threshold: float,
        title: str | None = None,
        plot_pdf: bool = False,
        **plot_kwargs: Any,
    ) -> None:
        """Save a time-series probability + confidence plot.

        Delegates to :func:`eruption_forecast.plots.forecast_plots.plot_forecast`
        for figure construction, then saves to the figures directory.

        Args:
            df (pd.DataFrame): Forecast consensus dataframe with predictions and confidence.
            threshold (float): Probability threshold for the eruption classification line.
            title (str | None, optional): Forecast plot title. Defaults to None.
            plot_pdf (bool, optional): Also save the figure as a PDF alongside the PNG.
                ``dpi`` is ignored for PDF (vector format); ``bbox_inches``,
                ``facecolor``, and ``edgecolor`` still apply. Defaults to False.
            **plot_kwargs: Extra keyword arguments forwarded verbatim to
                :func:`~eruption_forecast.plots.forecast_plots.plot_forecast`
                (e.g. ``eruption_dates``, ``fig_width``).
        """
        if self.labels_df is None:
            raise ValueError("No labels dataframe provided.")

        fig = plot_forecast(
            df=df,
            label_df=self.labels_df,
            threshold=threshold,
            title=title,
            **plot_kwargs,
        )

        path = os.path.join(self.figures_dir, f"forecast_{self.basename}.png")
        fig.savefig(
            path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor=None
        )

        logger.info(f"Forecast plot saved to: {path}")

        if plot_pdf:
            path = os.path.join(self.figures_dir, f"forecast_{self.basename}.pdf")

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
