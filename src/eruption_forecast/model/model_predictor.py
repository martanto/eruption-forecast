import os
from typing import Literal
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eruption_forecast import TremorData, FeaturesBuilder, TremorMatrixBuilder
from eruption_forecast.utils import (
    to_datetime,
    construct_windows,
    resolve_output_dir,
    compute_model_probabilities,
)
from eruption_forecast.logger import logger
from eruption_forecast.features.constants import (
    ID_COLUMN,
    ERUPTED_COLUMN,
    DATETIME_COLUMN,
)
from eruption_forecast.model.model_evaluator import ModelEvaluator


_METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "f1_score",
    "precision",
    "recall",
]


class ModelPredictor:
    """Evaluate or forecast with one or more sets of trained full-dataset models.

    Loads models produced by ``ModelTrainer.train()``.  Accepts either a single
    registry CSV or a dict of multiple registries (one per classifier type),
    enabling **consensus** aggregation across classifiers.

    Two operating modes are supported:

    * **Evaluation mode** (``future_labels_csv`` provided): evaluates every
      seed of every classifier on labelled future data and aggregates metrics
      per classifier and across classifiers.  Use :meth:`predict` and
      :meth:`predict_best`.

    * **Forecast mode** (no ``future_labels_csv``): aggregates eruption
      probability across seeds *and* classifiers.  Use :meth:`predict_proba`.

    Args:
        start_date (str | datetime): Start of the prediction window.
        end_date (str | datetime): End of the prediction window.
        trained_models (str | dict[str, str]): Either a single path to a
            ``trained_model_{suffix}.csv`` file produced by
            ``ModelTrainer.train()``; or a dict mapping a classifier name to its
            CSV path (e.g. ``{"rf": "...", "xgb": "...", "voting": "..."}``)
            for multi-model consensus.
        overwrite (bool, optional): Re-compute cached files if True.
            Defaults to False.
        n_jobs (int, optional): Number of parallel jobs for feature extraction.
            Defaults to 1.
        output_dir (str | None, optional): Root directory for outputs.
            Defaults to ``<cwd>/output/predictions``.
        root_dir (str | None, optional): Project root used to resolve
            relative ``output_dir`` paths.  Defaults to None.
        verbose (bool, optional): Print extra progress information.
            Defaults to False.

    Example — single model, evaluation mode::

        >>> predictor = ModelPredictor(
        ...     start_date="2025-03-20",
        ...     end_date="2025-03-22",
        ...     trained_models="output/trainings/rf/trained_model_rf.csv",
        ... )
        >>> df_metrics = predictor.predict(
        ...     future_features_csv="output/features/future_features.csv",
        ...     future_labels_csv="output/features/future_labels.csv",
        ... )
        >>> evaluator = predictor.predict_best(
        ...     future_features_csv="output/features/future_features.csv",
        ...     future_labels_csv="output/features/future_labels.csv",
        ... )
        >>> print(evaluator.summary())

    Example — multi-model consensus, forecast mode::

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
        n_jobs: int = 1,
        output_dir: str | None = None,
        root_dir: str | None = None,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        # ------------------------------------------------------------------
        # Set DEFAULT parameter
        # ------------------------------------------------------------------
        start_date = to_datetime(start_date).replace(hour=0, minute=0, second=0)
        end_date = to_datetime(end_date).replace(hour=23, minute=59, second=59)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        # Output training dir: ``<root_dir>/output/trainings``
        output_dir = resolve_output_dir(
            output_dir,
            root_dir,
            os.path.join("output", "predictions"),
        )

        output_dir = output_dir
        tremor_dir = os.path.join(output_dir, "tremor")
        features_dir = os.path.join(output_dir, "features")
        extracted_dir = os.path.join(features_dir, "extracted")
        figures_dir = os.path.join(output_dir, "figures")

        os.makedirs(figures_dir, exist_ok=True)

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

        # Backward-compat alias for single-model usage
        # Normalize to dict[name, df_models]
        if isinstance(trained_models, str):
            self.trained_models: dict[str, pd.DataFrame] = {
                "model": pd.read_csv(trained_models, index_col=0)
            }
        else:
            self.trained_models = {
                model_name: pd.read_csv(path, index_col=0)
                for model_name, path in trained_models.items()
            }

        self._multi_model: bool = len(self.trained_models) > 1

        # Will be set after predict() is called
        self.predict_all_metrics: list[dict] | None = None
        self.predict_evaluators: list[ModelEvaluator] | None = None

        # Will be set after predict_proba() is called
        self.df: pd.DataFrame | None = None

        if verbose:
            print(f"Models registered: {len(self.trained_models)}")

    @property
    def model_names(self) -> list[str]:
        """Names of registered classifier types."""
        return list(self.trained_models.keys())

    def predict(
        self, future_features_csv: str, future_labels_csv: str, plot: bool = False
    ) -> pd.DataFrame:
        """Evaluate every trained model on labelled future data.

        Iterates over all classifiers and all seeds within each classifier,
        computing evaluation metrics for each (classifier, seed) pair.

        For multi-model usage a ``classifier`` column is added so rows can be
        grouped per classifier.  A consensus summary (mean across all
        classifiers) is logged and saved separately.

        Args:
            future_features_csv (str): Extracted features dataframe from future tremor.
            future_labels_csv (str): Path to a labels CSV
                matching ``future_features_csv``.  Required for :meth:`predict` /
                :meth:`predict_best`.
            plot (bool, optional): Save per-seed plots to the output
                directory. Defaults to False.

        Returns:
            pd.DataFrame: One row per (classifier, seed) with all evaluation
            metrics plus a ``classifier`` column.

        Raises:
            RuntimeError: If no ``future_labels_csv`` was provided.
        """
        features_df = pd.read_csv(future_features_csv, index_col=0)

        _df = pd.read_csv(future_labels_csv)
        if ID_COLUMN in _df.columns:
            _df = _df.set_index(ID_COLUMN)
        if DATETIME_COLUMN in _df.columns:
            _df = _df.drop(DATETIME_COLUMN, axis=1)

        labels_df = _df[ERUPTED_COLUMN]

        if labels_df.empty:
            raise RuntimeError(
                "predict() requires labels. "
                "Pass future_labels_csv= at construction, or use predict_proba() "
                "for unlabelled forecast mode."
            )

        all_metrics: list[dict] = []
        evaluators: list[ModelEvaluator] = []

        for model_name, df_models in self.trained_models.items():
            logger.info(f"Evaluating classifier: {model_name}")

            model_output_dir = os.path.join(self.output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            for random_state, row in df_models.iterrows():
                significant_features_csv: str = row["significant_features_csv"]
                model_filepath: str = row["trained_model_filepath"]
                model_name = f"{model_name}_seed_{random_state:05d}"

                df_sig = pd.read_csv(significant_features_csv, index_col=0)
                feature_names: list[str] = df_sig.index.tolist()

                seed_output_dir = os.path.join(model_output_dir, model_name)

                evaluator = ModelEvaluator.from_files(
                    model_path=model_filepath,
                    X_test=features_df,
                    y_test=labels_df,
                    selected_features=feature_names,
                    model_name=model_name,
                    output_dir=seed_output_dir if plot else model_output_dir,
                )

                metrics = evaluator.get_metrics()
                assert isinstance(metrics, dict)
                metrics["classifier"] = model_name
                metrics["random_state"] = random_state
                all_metrics.append(metrics)
                evaluators.append(evaluator)

                if plot:
                    evaluator.plot_all()

                logger.info(
                    f"  Seed {random_state:05d} — balanced_accuracy: "
                    f"{metrics['balanced_accuracy']:.4f}"
                )

        self.predict_log_metrics_summary(all_metrics)
        self.predict_all_metrics = all_metrics
        self.predict_evaluators = evaluators

        df_result = pd.DataFrame(all_metrics)
        df_result.to_csv(
            os.path.join(self.output_dir, f"all_metrics_{self.basename}.csv"),
            index=False,
        )
        return df_result

    def predict_best(
        self,
        future_features_csv: str,
        future_labels_csv: str,
        criterion: str = "balanced_accuracy",
        plot: bool = False,
    ) -> ModelEvaluator:
        """Return the ``ModelEvaluator`` for the best (classifier, seed) pair.

        Searches across all classifiers and seeds to find the single model
        with the highest value for *criterion*.

        Args:
            future_features_csv (str): Extracted features dataframe from future tremor.
            future_labels_csv (str): Path to a labels CSV
                matching ``future_features_csv``.  Required for :meth:`predict` /
                :meth:`predict_best`.
            criterion (str, optional): Metric name to optimise.
                Defaults to ``"balanced_accuracy"``.
            plot (bool, optional): Passed to :meth:`predict` when called
                internally.

        Returns:
            ModelEvaluator: Evaluator for the best (classifier, seed).

        Raises:
            RuntimeError: If no ``future_labels_csv`` was provided.

        Example:
            >>> best = predictor.predict_best(criterion="f1_score")
            >>> print(best.summary())
        """
        if self.predict_all_metrics is None or self.predict_evaluators is None:
            self.predict(
                future_features_csv=future_features_csv,
                future_labels_csv=future_labels_csv,
                plot=plot,
            )

        assert self.predict_all_metrics is not None
        assert self.predict_evaluators is not None

        best_idx = max(
            range(len(self.predict_all_metrics)),
            key=lambda i: self.predict_all_metrics[i].get(criterion, float("-inf")),  # type: ignore[index]
        )

        best_evaluator = self.predict_evaluators[best_idx]
        best: dict = self.predict_all_metrics[best_idx]
        logger.info(
            f"Best model: classifier={best.get('classifier')} "
            f"seed={best.get('random_state')} "
            f"({criterion}={best.get(criterion):.4f})"
        )
        return best_evaluator

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
        filename = f"{self.basename}_ws-{window_step}-{window_step_unit}"

        os.makedirs(self.features_dir, exist_ok=True)
        futures_labels_filepath = os.path.join(
            self.features_dir,
            f"future_labels_{filename}.csv",
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

        os.makedirs(self.tremor_dir, exist_ok=True)
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
        os.makedirs(self.extracted_dir, exist_ok=True)

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
        save_predictions: bool = False,
        plot: bool = True,
    ) -> pd.DataFrame:
        """Forecast eruption probability for unlabelled future windows.

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

        for model_name, df_models in self.trained_models.items():
            logger.info(f"Forecasting with classifier: {model_name}")

            model_output_dir = os.path.join(self.output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            mean_probability, std_probability, confidence, prediction = (
                compute_model_probabilities(
                    df_models=df_models,
                    features_df=features_df,
                    classifier_name=model_name,
                    output_dir=model_output_dir,
                    save_predictions=save_predictions,
                    overwrite=self.overwrite,
                )
            )

            model_mean_probabilities[model_name] = mean_probability

            cols[f"{model_name}_eruption_probability"] = mean_probability
            cols[f"{model_name}_uncertainty"] = std_probability
            cols[f"{model_name}_confidence"] = confidence
            cols[f"{model_name}_prediction"] = prediction

            logger.info(
                f"  {model_name} — mean P(eruption): {mean_probability.mean():.4f}, "
                f"mean confidence: {confidence.mean():.4f}"
            )

        # Consensus across classifiers
        all_model_means = np.stack(list(model_mean_probabilities.values()), axis=0)
        consensus_mean: np.ndarray = all_model_means.mean(axis=0)
        consensus_std: np.ndarray = all_model_means.std(axis=0)
        consensus_pred: np.ndarray = (consensus_mean >= 0.5).astype(int)

        # Consensus confidence: fraction of classifiers agreeing with majority
        n_classifiers = all_model_means.shape[0]
        clf_preds = np.stack(
            [cols[f"{n}_prediction"] for n in self.trained_models], axis=0
        )
        votes_with_majority = np.where(
            consensus_pred == 1,
            (clf_preds == 1).sum(axis=0),
            (clf_preds == 0).sum(axis=0),
        )
        consensus_conf: np.ndarray = votes_with_majority / n_classifiers

        cols["consensus_eruption_probability"] = consensus_mean
        cols["consensus_uncertainty"] = consensus_std
        cols["consensus_confidence"] = consensus_conf
        cols["consensus_prediction"] = consensus_pred

        df_forecast = pd.DataFrame(cols, index=features_df.index)

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

    def predict_log_metrics_summary(self, all_metrics: list[dict]) -> None:
        """Log per-classifier and consensus metrics summary.

        Args:
            all_metrics (list[dict]): List of metric dictionaries from all predictions.
        """
        df = pd.DataFrame(all_metrics)

        logger.info("=" * 60)
        logger.info("Evaluation Metrics Summary (mean ± std across seeds)")
        logger.info("=" * 60)

        for model_name in self.model_names:
            sub = (
                df[df["classifier"] == model_name] if "classifier" in df.columns else df
            )
            logger.info(f"  Classifier: {model_name}")
            for metric in _METRIC_KEYS:
                if metric in sub.columns:
                    logger.info(
                        f"    {metric:20s}: {sub[metric].mean():.4f} ± {sub[metric].std():.4f}"
                    )

        if self._multi_model:
            logger.info("-" * 60)
            logger.info("  Consensus (mean across classifiers):")
            # Average the per-classifier means
            for metric in _METRIC_KEYS:
                if metric in df.columns:
                    per_clf_means = [
                        df[df["classifier"] == n][metric].mean()
                        for n in self.model_names
                    ]
                    consensus_mean = float(np.mean(per_clf_means))
                    consensus_std = float(np.std(per_clf_means))
                    logger.info(
                        f"    {metric:20s}: {consensus_mean:.4f} ± {consensus_std:.4f}"
                    )

        logger.info("=" * 60)

        # Save summary
        summary_path = os.path.join(self.output_dir, "all_metrics_summary.csv")
        df.describe().T.to_csv(summary_path)
        logger.info(f"Metrics summary saved to: {summary_path}")

    def _plot_forecast(
        self,
        df: pd.DataFrame,
        model_mean_probabilities: dict[str, np.ndarray],
    ) -> None:
        """Save a time-series probability + confidence plot with enhanced styling.

        Multi-model: draws each classifier as a separate line plus the
        consensus with a shaded ±1 std band.
        Single-model: same styling without multi-model comparison.

        Args:
            df (pd.DataFrame): Forecast dataframe with predictions and confidence.
            model_mean_probabilities (dict[str, np.ndarray]): Mean probabilities per model.
        """
        # Define custom color palette
        custom_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        # Create figure with better proportions
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(14, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [1.2, 1]},
        )

        index = range(len(df)) if not hasattr(df.index, "to_pydatetime") else df.index

        # ========== TOP PANEL: Eruption Probability ==========
        # Per-classifier probability lines (subtle)
        for i, name in enumerate(model_mean_probabilities):
            col = f"{name}_eruption_probability"
            ax1.plot(
                index,
                df[col],
                linewidth=1.5,
                alpha=0.5,
                color=custom_colors[i % len(custom_colors)],
                linestyle="--",
                label=f"{name.upper()}",
                marker="o",
                markersize=2,
                markevery=max(1, len(df) // 20),
            )

        # Consensus with uncertainty band (prominent)
        cp = df["consensus_eruption_probability"]
        cu = df["consensus_uncertainty"]
        ax1.fill_between(
            index,
            (cp - cu).clip(0, 1),
            (cp + cu).clip(0, 1),
            alpha=0.25,
            # color="#34495e",
            label="Uncertainty (±1σ)",
            edgecolor="#2c3e50",
            linewidth=0.5,
        )
        ax1.plot(
            index,
            cp,
            color="#2c3e50",
            linewidth=3.0,
            label="Consensus",
            marker="o",
            markersize=3,
            markevery=max(1, len(df) // 20),
            zorder=5,
        )

        # Threshold line with better styling
        ax1.axhline(
            0.5,
            color="#e74c3c",
            linestyle="--",
            linewidth=2,
            label="Threshold (0.5)",
            alpha=0.8,
            zorder=3,
        )

        # High-risk zone shading
        # ax1.axhspan(0.5, 1.0, alpha=0.05, color="#e74c3c", zorder=0)

        # Styling
        ax1.set_ylabel("Eruption Probability", fontsize=12, fontweight="bold")
        ax1.set_ylim(-0.02, 1.02)
        ax1.legend(
            loc="upper left",
            fontsize=9,
            frameon=True,
            shadow=True,
            fancybox=True,
            ncol=2 if len(model_mean_probabilities) > 3 else 1,
        )
        ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
        ax1.set_title(
            "Volcanic Eruption Forecast"
            + (" — Multi-Model Consensus" if self._multi_model else ""),
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # ========== BOTTOM PANEL: Model Confidence ==========
        # Per-classifier confidence lines
        for i, name in enumerate(model_mean_probabilities):
            ax2.plot(
                index,
                df[f"{name}_confidence"],
                # linewidth=1.5,
                # alpha=0.5,
                color=custom_colors[i % len(custom_colors)],
                # linestyle="--",
                label=f"{name.upper()}",
                # marker="s",
                # markersize=2,
                markevery=max(1, len(df) // 20),
            )

        # Consensus confidence (prominent)
        ax2.plot(
            index,
            df["consensus_confidence"],
            # color="#2c3e50",
            linewidth=3.0,
            label="Consensus",
            # marker="s",
            # markersize=3,
            markevery=max(1, len(df) // 20),
            zorder=5,
        )

        # Reference lines
        ax2.axhline(0.5, color="#95a5a6", linestyle=":", linewidth=1.5, alpha=0.7)
        ax2.axhline(
            0.75,
            color="red",
            linestyle=":",
            linewidth=1,
            # alpha=0.5,
            label="High confidence",
        )

        # Styling
        ax2.set_ylabel("Model Confidence", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Time Window", fontsize=12, fontweight="bold")
        ax2.set_ylim(-0.02, 1.02)
        ax2.legend(
            loc="upper left",
            fontsize=9,
            frameon=False,
            shadow=False,
            fancybox=False,
            ncol=2 if len(model_mean_probabilities) > 3 else 1,
        )
        ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # Final layout and save
        plt.tight_layout()
        path = os.path.join(self.figures_dir, f"eruption_forecast_{self.basename}.png")
        fig.savefig(
            path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        plt.close(fig)
        logger.info(f"Forecast plot saved to: {path}")
