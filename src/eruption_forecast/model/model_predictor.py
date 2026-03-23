import os
from typing import Literal
from datetime import datetime, timedelta

import joblib
import matplotlib

from eruption_forecast.config.constants import MATPLOTLIB_BACKEND
from eruption_forecast.utils.formatting import slugify


matplotlib.use(
    MATPLOTLIB_BACKEND
)  # Must be called before pyplot import — non-interactive backend safe for worker threads
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eruption_forecast.config import ERUPTION_PROBABILITY_THRESHOLD
from eruption_forecast.logger import logger
from eruption_forecast.utils.ml import compute_model_probabilities
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
        >>> print(df_forecast[["consensus_eruption_probability", "consensus_confidence"]])
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
        self.start_date = start_date
        self.end_date = end_date
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
        # Will be set after build_future_labels() method called
        # ------------------------------------------------------------------
        self.labels_df: pd.DataFrame | pd.Series | None = None
        self.basename = f"{self.start_date_str}_{self.end_date_str}"

        model_name = slugify(model_name).lower()
        self.model_name = model_name

        # Normalize trained_models to one of two internal representations:
        #   - dict[str, pd.DataFrame]   for CSV-based registry (existing path)
        #   - dict[str, SeedEnsemble]   for merged .pkl files (new path)
        self._classifier_ensemble: ClassifierEnsemble | None = None
        if isinstance(trained_models, str):
            if trained_models.endswith(".pkl"):
                loaded = joblib.load(trained_models)
                if isinstance(loaded, ClassifierEnsemble):
                    # New multi-classifier merged pkl
                    self._classifier_ensemble: ClassifierEnsemble = loaded
                    self.trained_models: dict[str, pd.DataFrame | SeedEnsemble] = dict(
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
                self.trained_models = {
                    model_name: pd.read_csv(trained_models, index_col=0)
                }
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
            elif isinstance(first_val, str):
                # Values are paths to CSV registry files
                logger.info("Models prediction using CSV.")
                self.trained_models = {
                    model_name: pd.read_csv(path, index_col=0)
                    for model_name, path in trained_models.items()
                }
            else:
                raise ValueError(
                    f"Unrecognised trained_models value type: {type(first_val)}"
                )

        self._multi_model: bool = len(self.trained_models) > 1

        # Will be set after predict_proba() is called
        self.df: pd.DataFrame | None = None

        if verbose:
            logger.info(f"Models registered: {len(self.trained_models)}")

    @property
    def model_names(self) -> list[str]:
        """Names of registered classifier types."""
        return list(self.trained_models.keys())

    def build_future_labels(
        self,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
    ) -> pd.DataFrame:
        """Build future labels dataframe with datetime as index and id as columns.

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
            f"future-labels_{filename}.csv",
        )

        # Skip if file exists
        if os.path.exists(futures_labels_filepath) and not self.overwrite:
            self.labels_df = pd.read_csv(
                futures_labels_filepath, index_col=0, parse_dates=True
            )
            return self.labels_df

        futures_labels_df = construct_windows(
            start_date=self.start_date,
            end_date=self.end_date,
            window_step=window_step,
            window_step_unit=window_step_unit,
        )
        futures_labels_df["id"] = range(len(futures_labels_df))

        self.labels_df = futures_labels_df
        self.labels_df.to_csv(futures_labels_filepath, index=True)

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
                :meth:`build_future_labels`.
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
                "Parameter labels_df not provided. Please run build_future_labels() first."
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

        self.tremor_matrix_df = tremor_matrix_builder.df
        self.tremor_matrix_df.to_csv(filepath, index=False)

        return tremor_matrix_builder.df

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
            pd.DataFrame: Tremor dataframe sliced to the predictor date range.

        Raises:
            ValueError: If ``tremor_data`` is neither a filepath string nor a
                DataFrame, or if no tremor data falls within the requested
                date range.
            TypeError: If the resulting tremor DataFrame does not have a
                ``DatetimeIndex``.
        """
        tremor_df = None

        _tremor_data = TremorData()
        if isinstance(tremor_data, str):
            tremor_df = _tremor_data.from_csv(tremor_data)

        if isinstance(tremor_data, pd.DataFrame):
            _tremor_data = TremorData(df=tremor_data)
            tremor_df = _tremor_data.df

        if tremor_df is None:
            raise ValueError(
                f"Parameter tremor_data only accepts valid tremor data filepath "
                f"or pandas DataFrame type. Your tremor data type is: {type(tremor_data)}"
            )

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
        """Get future features dataframe.

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
        future_labels = self.build_future_labels(
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
        window_step_unit: Literal["minutes", "hours"],
        use_relevant_features: bool = True,
        select_tremor_columns: list[str] | None = None,
        threshold: float = ERUPTION_PROBABILITY_THRESHOLD,
        save_predictions: bool = True,
        plot: bool = True,
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
        the consensus columns equal those values exactly.

        The result is saved as ``predictions.csv``.  If *plot* is True a
        time-series figure is also saved.

        Args:
            tremor_data (str | pd.DataFrame): Tremor data in CSV or dataframe.
            window_size (int): Window size to use.
            window_step (int): Window step size to use.
            window_step_unit (Literal["minutes", "hours"]): Window step unit to use.
            use_relevant_features (bool, optional): Whether to use relevant features. Defaults to True.
            select_tremor_columns (list[str] | None): List of tremor columns to use. Defaults to None.
            threshold (float, optional): Probability threshold for eruption classification.
                Defaults to ``ERUPTION_PROBABILITY_THRESHOLD`` which value is 0.5.
            save_predictions (bool, optional): Save predictions result. Defaults to True.
            plot (bool, optional): Save a probability time-series plot.
                Defaults to True.

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
        model_mean_probabilities: dict[str, np.ndarray] = {}
        result_dir = os.path.join(self.output_dir, "results")

        if isinstance(self._classifier_ensemble, ClassifierEnsemble):
            consensus_mean, consensus_std, consensus_conf, consensus_pred = (
                self._forecast_with_classifier_ensemble(
                    features_df,
                    cols,
                    model_mean_probabilities,
                    threshold=threshold,
                )
            )
        else:
            self._forecast_with_trained_models(
                features_df=features_df,
                result_dir=result_dir,
                threshold=threshold,
                save_predictions=save_predictions,
                cols=cols,
                model_mean_probabilities=model_mean_probabilities,
            )
            consensus_mean, consensus_std, consensus_conf, consensus_pred = (
                self._compute_consensus(cols, model_mean_probabilities)
            )

        cols["consensus_eruption_probability"] = consensus_mean
        cols["consensus_uncertainty"] = consensus_std
        cols["consensus_confidence"] = consensus_conf
        cols["consensus_prediction"] = consensus_pred

        df_forecast = pd.DataFrame(cols, index=features_df.index)

        if save_predictions:
            csv_path = os.path.join(
                self.output_dir, f"result_all_model_predictions_{self.basename}.csv"
            )
            df_forecast.to_csv(csv_path)
            logger.info(f"Predictions saved to: {csv_path}")

        self._log_forecast_summary(df_forecast, model_mean_probabilities)

        if plot:
            self._plot_forecast(df_forecast, model_mean_probabilities)

        self.df = df_forecast
        return df_forecast

    @staticmethod
    def _store_classifier_proba(
        model_name: str,
        mean: np.ndarray,
        std: np.ndarray,
        confidence: np.ndarray,
        prediction: np.ndarray,
        cols: dict[str, np.ndarray],
        model_mean_probabilities: dict[str, np.ndarray],
    ) -> None:
        """Store per-classifier probability arrays and log a one-line summary.

        Writes the four per-classifier result columns into ``cols``, records
        the mean probability array in ``model_mean_probabilities``, and emits
        a log line with mean eruption probability and mean confidence.

        Args:
            model_name (str): Classifier identifier used as column prefix.
            mean (np.ndarray): Mean eruption probability per window.
            std (np.ndarray): Standard deviation of probability across seeds.
            confidence (np.ndarray): Fraction of seeds agreeing with the majority vote.
            prediction (np.ndarray): Binary prediction per window (0 or 1).
            cols (dict[str, np.ndarray]): Mutable column accumulator for the
                forecast DataFrame.
            model_mean_probabilities (dict[str, np.ndarray]): Mutable accumulator
                for cross-classifier consensus computation.
        """
        model_mean_probabilities[model_name] = mean
        cols[f"{model_name}_eruption_probability"] = mean
        cols[f"{model_name}_uncertainty"] = std
        cols[f"{model_name}_confidence"] = confidence
        cols[f"{model_name}_prediction"] = prediction
        logger.info(
            f"  {model_name} — mean P(eruption): {mean.mean():.4f}, "
            f"mean confidence: {confidence.mean():.4f}"
        )

    def _forecast_with_classifier_ensemble(
        self,
        features_df: pd.DataFrame,
        cols: dict[str, np.ndarray],
        model_mean_probabilities: dict[str, np.ndarray],
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Delegate inference and consensus to a ClassifierEnsemble.

        Calls ``ClassifierEnsemble.predict_with_uncertainty`` for the fast
        path where all classifiers are bundled in a single object, then
        populates ``cols`` and ``model_mean_probabilities`` via
        :meth:`_store_classifier_proba`.

        Args:
            features_df (pd.DataFrame): Future feature matrix.
            cols (dict[str, np.ndarray]): Mutable column accumulator for the
                forecast DataFrame.
            model_mean_probabilities (dict[str, np.ndarray]): Mutable
                accumulator for cross-classifier consensus computation.
            threshold (float, optional): Threshold for classifying eruption
                probability as positive.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple of
            ``(consensus_mean, consensus_std, consensus_conf, consensus_pred)``.
        """
        assert self._classifier_ensemble is not None
        consensus_mean, consensus_std, consensus_conf, consensus_pred, per_clf = (
            self._classifier_ensemble.predict_with_uncertainty(
                features_df, threshold=threshold
            )
        )
        for model_name, clf_result in per_clf.items():
            self._store_classifier_proba(
                model_name=model_name,
                mean=clf_result["mean"],
                std=clf_result["std"],
                confidence=clf_result["confidence"],
                prediction=clf_result["prediction"],
                cols=cols,
                model_mean_probabilities=model_mean_probabilities,
            )
        return consensus_mean, consensus_std, consensus_conf, consensus_pred

    def _get_classifier_proba(
        self,
        df_models: pd.DataFrame | SeedEnsemble,
        model_name: str,
        features_df: pd.DataFrame,
        model_output_dir: str,
        threshold: float,
        save_predictions: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(mean, std, confidence, prediction)`` for one classifier.

        Dispatches to ``SeedEnsemble.predict_with_uncertainty`` for in-memory
        models or to :func:`compute_model_probabilities` for CSV-registry models.

        Args:
            df_models (pd.DataFrame | SeedEnsemble): Seed data source.
            model_name (str): Classifier identifier used for output paths.
            features_df (pd.DataFrame): Future feature matrix.
            model_output_dir (str): Directory for per-classifier result output.
            threshold (float): Probability threshold for binary prediction.
            save_predictions (bool): Whether to persist per-seed predictions to disk.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Arrays of
            ``(mean_probability, std_probability, confidence, prediction)``.
        """
        if isinstance(df_models, SeedEnsemble):
            return df_models.predict_with_uncertainty(features_df, threshold)
        return compute_model_probabilities(
            df_models=df_models,
            features_df=features_df,
            classifier_name=model_name,
            output_dir=model_output_dir,
            save_predictions=save_predictions,
            overwrite=self.overwrite,
        )

    def _forecast_single_classifier(
        self,
        model_name: str,
        df_models: pd.DataFrame | SeedEnsemble,
        features_df: pd.DataFrame,
        result_dir: str,
        threshold: float,
        save_predictions: bool,
        cols: dict[str, np.ndarray],
        model_mean_probabilities: dict[str, np.ndarray],
    ) -> None:
        """Run inference for a single classifier and store results.

        Calls :meth:`_get_classifier_proba` to obtain aggregated probability
        arrays then delegates storage to :meth:`_store_classifier_proba`.

        Args:
            model_name (str): Classifier identifier used for logging and output paths.
            df_models (pd.DataFrame | SeedEnsemble): Either a ``SeedEnsemble``
                or a registry ``DataFrame`` with model file paths.
            features_df (pd.DataFrame): Future feature matrix.
            result_dir (str): Root directory for per-classifier result output.
            threshold (float): Probability threshold for binary prediction.
            save_predictions (bool): Whether to persist per-seed predictions to disk.
            cols (dict[str, np.ndarray]): Mutable column accumulator for the
                forecast DataFrame.
            model_mean_probabilities (dict[str, np.ndarray]): Mutable
                accumulator for cross-classifier consensus computation.
        """
        logger.info(f"Forecasting with classifier: {model_name}")

        model_output_dir = os.path.join(result_dir, model_name)
        ensure_dir(model_output_dir)

        mean_probability, std_probability, confidence, prediction = (
            self._get_classifier_proba(
                df_models=df_models,
                model_name=model_name,
                features_df=features_df,
                model_output_dir=model_output_dir,
                threshold=threshold,
                save_predictions=save_predictions,
            )
        )

        self._store_classifier_proba(
            model_name=model_name,
            mean=mean_probability,
            std=std_probability,
            confidence=confidence,
            prediction=prediction,
            cols=cols,
            model_mean_probabilities=model_mean_probabilities,
        )

    def _forecast_with_trained_models(
        self,
        features_df: pd.DataFrame,
        result_dir: str,
        threshold: float,
        save_predictions: bool,
        cols: dict[str, np.ndarray],
        model_mean_probabilities: dict[str, np.ndarray],
    ) -> None:
        """Iterate over all registered classifiers and run per-classifier inference.

        Calls :meth:`_forecast_single_classifier` for each entry in
        ``self.trained_models``.

        Args:
            features_df (pd.DataFrame): Future feature matrix.
            result_dir (str): Root directory for per-classifier result output.
            threshold (float): Probability threshold for binary prediction.
            save_predictions (bool): Whether to persist per-seed predictions to disk.
            cols (dict[str, np.ndarray]): Mutable column accumulator for the
                forecast DataFrame.
            model_mean_probabilities (dict[str, np.ndarray]): Mutable
                accumulator for cross-classifier consensus computation.
        """
        for model_name, df_models in self.trained_models.items():
            self._forecast_single_classifier(
                model_name=model_name,
                df_models=df_models,
                features_df=features_df,
                result_dir=result_dir,
                threshold=threshold,
                save_predictions=save_predictions,
                cols=cols,
                model_mean_probabilities=model_mean_probabilities,
            )

    def _compute_consensus(
        self,
        cols: dict[str, np.ndarray],
        model_mean_probabilities: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute cross-classifier consensus probability, uncertainty, confidence, and prediction.

        Averages per-classifier mean probabilities to form the consensus
        probability, derives uncertainty as the standard deviation across
        classifiers, and computes confidence as the fraction of classifiers
        whose binary prediction agrees with the majority vote.

        Args:
            cols (dict[str, np.ndarray]): Column accumulator containing
                ``{name}_prediction`` arrays for each registered classifier.
            model_mean_probabilities (dict[str, np.ndarray]): Mean eruption
                probability array per classifier.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple of
            ``(consensus_mean, consensus_std, consensus_conf, consensus_pred)``.
        """
        all_model_means = np.stack(list(model_mean_probabilities.values()), axis=0)
        consensus_mean = all_model_means.mean(axis=0)
        consensus_std = all_model_means.std(axis=0)
        consensus_pred = (consensus_mean >= ERUPTION_PROBABILITY_THRESHOLD).astype(int)

        # Fraction of classifiers agreeing with the majority vote
        n_classifiers = all_model_means.shape[0]
        clf_preds = np.stack(
            [cols[f"{n}_prediction"] for n in self.trained_models], axis=0
        )
        votes_with_majority = np.where(
            consensus_pred == 1,
            (clf_preds == 1).sum(axis=0),
            (clf_preds == 0).sum(axis=0),
        )
        consensus_conf = votes_with_majority / n_classifiers
        return consensus_mean, consensus_std, consensus_conf, consensus_pred

    def _log_forecast_summary(
        self,
        df: pd.DataFrame,
        model_mean_probabilities: dict[str, np.ndarray],
    ) -> None:
        """Log a per-classifier + consensus forecast summary.

        Args:
            df (pd.DataFrame): Forecast dataframe with predictions.
            model_mean_probabilities (dict[str, np.ndarray]): Mean probabilities per model.
        """
        logger.info("=" * 60)
        logger.info("Forecast Summary")
        logger.info("=" * 60)
        for name, mean_p in model_mean_probabilities.items():
            pred = df[f"{name}_prediction"].to_numpy()
            conf = df[f"{name}_confidence"].to_numpy()
            logger.info(
                f"  {name:12s}  eruption_windows={int(pred.sum()):4d}  "
                f"mean_P={mean_p.mean():.4f}  mean_conf={conf.mean():.4f}"
            )
        if self._multi_model:
            cp = df["consensus_eruption_probability"].to_numpy()
            cc = df["consensus_confidence"].to_numpy()
            pred_c = df["consensus_prediction"].to_numpy()
            logger.info("-" * 60)
            logger.info(
                f"  {'consensus':12s}  eruption_windows={int(pred_c.sum()):4d}  "
                f"mean_P={cp.mean():.4f}  mean_conf={cc.mean():.4f}"
            )
        logger.info("=" * 60)

    def _plot_forecast(
        self,
        df: pd.DataFrame,
        model_mean_probabilities: dict[str, np.ndarray],
    ) -> None:
        """Save a time-series probability + confidence plot.

        Delegates to :func:`eruption_forecast.plots.forecast_plots.plot_forecast`
        for figure construction, then saves to the figures directory.

        Args:
            df (pd.DataFrame): Forecast dataframe with predictions and confidence.
            model_mean_probabilities (dict[str, np.ndarray]): Mean probabilities per model.
        """
        fig = plot_forecast(
            df=df,
            model_names=list(model_mean_probabilities.keys()),
            multi_model=self._multi_model,
            figsize=(14, 8),
            dpi=300,
        )
        path = os.path.join(self.figures_dir, f"forecast_{self.basename}.png")
        fig.savefig(
            path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        plt.close(fig)
        logger.info(f"Forecast plot saved to: {path}")
