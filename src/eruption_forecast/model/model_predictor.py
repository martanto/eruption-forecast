import os

import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.features.constants import (
    ID_COLUMN,
    ERUPTED_COLUMN,
    DATETIME_COLUMN,
)
from eruption_forecast.model.model_evaluator import ModelEvaluator


class ModelPredictor:
    """Evaluate trained full-dataset models against a future feature set.

    Loads all models produced by ``TrainModel.fit()``, evaluates each one on
    "future" (held-out) features and labels, and aggregates metrics across
    seeds — mirroring the aggregation done by ``TrainModel._aggregate_metrics()``.

    Args:
        trained_models_csv (str): Path to the ``trained_model_{suffix}.csv`` file
            produced by ``TrainModel.fit()``.  Columns: ``random_state``,
            ``significant_features_csv``, ``trained_model_filepath``.
        future_features_csv (str): Path to a features CSV whose rows correspond
            to the "future" windows to predict on.
        future_labels_csv (str): Path to a labels CSV matching ``future_features_csv``.
        output_dir (str | None, optional): Root directory for prediction outputs.
            Defaults to ``<cwd>/output/predictions``.

    Example:
        >>> predictor = ModelPredictor(
        ...     trained_models_csv=trainer.csv,
        ...     future_features_csv="output/features/future_features.csv",
        ...     future_labels_csv="output/features/future_labels.csv",
        ... )
        >>> df_metrics = predictor.predict()
        >>> evaluator = predictor.predict_best()
        >>> print(evaluator.summary())
    """

    def __init__(
        self,
        trained_models_csv: str,
        future_features_csv: str,
        future_labels_csv: str,
        output_dir: str | None = None,
    ) -> None:
        df_models = pd.read_csv(trained_models_csv, index_col=0)

        df_features = pd.read_csv(future_features_csv, index_col=0)

        df_labels = pd.read_csv(future_labels_csv)
        if ID_COLUMN in df_labels.columns:
            df_labels = df_labels.set_index(ID_COLUMN)
        if DATETIME_COLUMN in df_labels.columns:
            df_labels = df_labels.drop(DATETIME_COLUMN, axis=1)
        df_labels = df_labels[ERUPTED_COLUMN]

        output_dir = output_dir or os.path.join(os.getcwd(), "output", "predictions")
        metrics_dir = os.path.join(output_dir, "metrics")
        figures_dir = os.path.join(output_dir, "figures")

        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        self.df_models = df_models
        self.df_features = df_features
        self.df_labels = df_labels
        self.output_dir = output_dir
        self.metrics_dir = metrics_dir
        self.figures_dir = figures_dir

        # Will be set after predict() is called
        self._all_metrics: list[dict] | None = None
        self._evaluators: list[ModelEvaluator] | None = None

    def predict(self, plot: bool = False) -> pd.DataFrame:
        """Evaluate every trained model on the future dataset.

        Args:
            plot (bool, optional): Save per-seed plots to the output directory. Defaults to False.

        Returns:
            pd.DataFrame: One row per seed with all evaluation metrics.
        """
        all_metrics = []
        evaluators = []

        for random_state, row in self.df_models.iterrows():
            significant_features_csv: str = row["significant_features_csv"]
            model_filepath: str = row["trained_model_filepath"]
            model_name = f"seed_{random_state:05d}"

            # Load feature names from the significant-features CSV
            df_sig = pd.read_csv(significant_features_csv, index_col=0)
            feature_names: list[str] = df_sig.index.tolist()

            # Evaluator output dir for this seed
            seed_output_dir = os.path.join(self.output_dir, model_name)

            evaluator = ModelEvaluator.from_files(
                model_path=model_filepath,
                X_test=self.df_features,
                y_test=self.df_labels,
                selected_features=feature_names,
                model_name=model_name,
                output_dir=seed_output_dir if plot else self.output_dir,
            )

            metrics = evaluator.get_metrics()
            assert isinstance(metrics, dict)
            metrics["random_state"] = random_state
            all_metrics.append(metrics)
            evaluators.append(evaluator)

            if plot:
                evaluator.plot_all()

            logger.info(
                f"Seed {random_state:05d} — balanced_accuracy: "
                f"{metrics['balanced_accuracy']:.4f}"
            )

        self._aggregate_metrics(all_metrics)

        self._all_metrics = all_metrics
        self._evaluators = evaluators

        return pd.DataFrame(all_metrics)

    def predict_best(
        self,
        criterion: str = "balanced_accuracy",
        plot: bool = False,
    ) -> ModelEvaluator:
        """Return the ``ModelEvaluator`` for the best-performing seed.

        Calls ``predict()`` if it has not been run yet, then finds the seed
        with the highest value for *criterion* and returns its evaluator.

        Args:
            criterion (str, optional): Metric name to optimise.
                Defaults to ``"balanced_accuracy"``.
            plot (bool, optional): Passed to ``predict()`` when called internally.
            save_reports (bool, optional): Passed to ``predict()`` when called internally.

        Returns:
            ModelEvaluator: Evaluator for the best seed.

        Example:
            >>> best = predictor.predict_best(criterion="f1_score")
            >>> print(best.summary())
        """
        if self._all_metrics is None or self._evaluators is None:
            self.predict(plot=plot)

        assert self._all_metrics is not None
        assert self._evaluators is not None

        best_idx = max(
            range(len(self._all_metrics)),
            key=lambda i: self._all_metrics[i].get(criterion, float("-inf")),  # type: ignore[index]
        )

        best_evaluator = self._evaluators[best_idx]
        best_score = self._all_metrics[best_idx].get(criterion)
        logger.info(
            f"Best seed: {self._all_metrics[best_idx].get('random_state')} "
            f"({criterion}={best_score:.4f})"
        )

        return best_evaluator

    def _aggregate_metrics(self, all_metrics: list[dict]) -> None:
        """Aggregate metrics across all seeds and save summary files.

        Computes mean and std for each numeric metric, logs a summary table,
        and saves ``all_metrics.csv`` and ``metrics_summary.csv`` to
        ``self.output_dir``.

        Args:
            all_metrics (list[dict]): List of metric dicts from each seed.
        """
        df_metrics = pd.DataFrame(all_metrics)

        summary = df_metrics.describe().T
        summary_filepath = os.path.join(self.output_dir, "metrics_summary.csv")
        summary.to_csv(summary_filepath)

        all_metrics_filepath = os.path.join(self.output_dir, "all_metrics.csv")
        df_metrics.to_csv(all_metrics_filepath, index=False)

        logger.info("=" * 60)
        logger.info("Prediction Metrics Summary (mean ± std across seeds)")
        logger.info("=" * 60)
        for metric in [
            "accuracy",
            "balanced_accuracy",
            "f1_score",
            "precision",
            "recall",
        ]:
            if metric in df_metrics.columns:
                mean = df_metrics[metric].mean()
                std = df_metrics[metric].std()
                logger.info(f"{metric:20s}: {mean:.4f} ± {std:.4f}")
        logger.info("=" * 60)

        logger.info(f"Summary metrics saved to: {summary_filepath}")
        logger.info(f"All metrics saved to: {all_metrics_filepath}")
