import os

import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

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

    Loads models produced by ``TrainModel.fit()``.  Accepts either a single
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
        trained_models (str | dict[str, str]): Either a single path to a
            ``trained_model_{suffix}.csv`` file produced by
            ``TrainModel.fit()``; or a dict mapping a classifier name to its
            CSV path (e.g. ``{"rf": "...", "xgb": "...", "voting": "..."}``)
            for multi-model consensus.
        future_features_csv (str): Path to a features CSV whose rows
            correspond to the future windows to predict on.
        future_labels_csv (str | None, optional): Path to a labels CSV
            matching ``future_features_csv``.  Required for :meth:`predict` /
            :meth:`predict_best`; omit for :meth:`predict_proba`.
            Defaults to ``None``.
        output_dir (str | None, optional): Root directory for outputs.
            Defaults to ``<cwd>/output/predictions``.

    Example — single model, evaluation mode::

        >>> predictor = ModelPredictor(
        ...     trained_models=trainer.csv,
        ...     future_features_csv="output/features/future_features.csv",
        ...     future_labels_csv="output/features/future_labels.csv",
        ... )
        >>> df_metrics = predictor.predict()
        >>> evaluator = predictor.predict_best()
        >>> print(evaluator.summary())

    Example — multi-model consensus, forecast mode::

        >>> predictor = ModelPredictor(
        ...     trained_models={
        ...         "rf":     "output/trainings/rf/trained_model_rf.csv",
        ...         "xgb":    "output/trainings/xgb/trained_model_xgb.csv",
        ...         "voting": "output/trainings/voting/trained_model_voting.csv",
        ...     },
        ...     future_features_csv="output/features/future_features.csv",
        ... )
        >>> df_forecast = predictor.predict_proba()
        >>> # Columns: rf_eruption_probability, xgb_eruption_probability, ...,
        >>> #          consensus_eruption_probability, consensus_confidence, ...
        >>> print(df_forecast[["consensus_eruption_probability", "consensus_confidence"]])
    """

    def __init__(
        self,
        trained_models: str | dict[str, str],
        future_features_csv: str,
        future_labels_csv: str | None = None,
        output_dir: str | None = None,
    ) -> None:
        # Normalize to dict[name, df_models]
        if isinstance(trained_models, str):
            self._models_registry: dict[str, pd.DataFrame] = {
                "model": pd.read_csv(trained_models, index_col=0)
            }
        else:
            self._models_registry = {
                name: pd.read_csv(path, index_col=0)
                for name, path in trained_models.items()
            }

        self._multi_model: bool = len(self._models_registry) > 1

        df_features = pd.read_csv(future_features_csv, index_col=0)

        df_labels: pd.Series | None = None
        if future_labels_csv is not None:
            _df = pd.read_csv(future_labels_csv)
            if ID_COLUMN in _df.columns:
                _df = _df.set_index(ID_COLUMN)
            if DATETIME_COLUMN in _df.columns:
                _df = _df.drop(DATETIME_COLUMN, axis=1)
            df_labels = _df[ERUPTED_COLUMN]

        output_dir = output_dir or os.path.join(os.getcwd(), "output", "predictions")
        metrics_dir = os.path.join(output_dir, "metrics")
        figures_dir = os.path.join(output_dir, "figures")

        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        self.df_features = df_features
        self.df_labels = df_labels
        self.output_dir = output_dir
        self.metrics_dir = metrics_dir
        self.figures_dir = figures_dir

        # Backward-compat alias for single-model usage
        if not self._multi_model:
            self.df_models = next(iter(self._models_registry.values()))

        # Will be set after predict() is called
        self._all_metrics: list[dict] | None = None
        self._evaluators: list[ModelEvaluator] | None = None

        # Will be set after predict_proba() is called
        self._forecast: pd.DataFrame | None = None

    @property
    def model_names(self) -> list[str]:
        """Names of registered classifier types."""
        return list(self._models_registry.keys())

    # ------------------------------------------------------------------
    # Evaluation mode
    # ------------------------------------------------------------------

    def predict(self, plot: bool = False) -> pd.DataFrame:
        """Evaluate every trained model on labelled future data.

        Iterates over all classifiers and all seeds within each classifier,
        computing evaluation metrics for each (classifier, seed) pair.

        For multi-model usage a ``classifier`` column is added so rows can be
        grouped per classifier.  A consensus summary (mean across all
        classifiers) is logged and saved separately.

        Args:
            plot (bool, optional): Save per-seed plots to the output
                directory. Defaults to False.

        Returns:
            pd.DataFrame: One row per (classifier, seed) with all evaluation
            metrics plus a ``classifier`` column.

        Raises:
            RuntimeError: If no ``future_labels_csv`` was provided.
        """
        if self.df_labels is None:
            raise RuntimeError(
                "predict() requires labels. "
                "Pass future_labels_csv= at construction, or use predict_proba() "
                "for unlabelled forecast mode."
            )

        all_metrics: list[dict] = []
        evaluators: list[ModelEvaluator] = []

        for clf_name, df_models in self._models_registry.items():
            logger.info(f"Evaluating classifier: {clf_name}")

            for random_state, row in df_models.iterrows():
                significant_features_csv: str = row["significant_features_csv"]
                model_filepath: str = row["trained_model_filepath"]
                model_name = f"{clf_name}_seed_{random_state:05d}"

                df_sig = pd.read_csv(significant_features_csv, index_col=0)
                feature_names: list[str] = df_sig.index.tolist()

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
                metrics["classifier"] = clf_name
                metrics["random_state"] = random_state
                all_metrics.append(metrics)
                evaluators.append(evaluator)

                if plot:
                    evaluator.plot_all()

                logger.info(
                    f"  Seed {random_state:05d} — balanced_accuracy: "
                    f"{metrics['balanced_accuracy']:.4f}"
                )

        self._log_metrics_summary(all_metrics)
        self._all_metrics = all_metrics
        self._evaluators = evaluators

        df_result = pd.DataFrame(all_metrics)
        df_result.to_csv(
            os.path.join(self.output_dir, "all_metrics.csv"), index=False
        )
        return df_result

    def predict_best(
        self,
        criterion: str = "balanced_accuracy",
        plot: bool = False,
    ) -> ModelEvaluator:
        """Return the ``ModelEvaluator`` for the best (classifier, seed) pair.

        Searches across all classifiers and seeds to find the single model
        with the highest value for *criterion*.

        Args:
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
        if self._all_metrics is None or self._evaluators is None:
            self.predict(plot=plot)

        assert self._all_metrics is not None
        assert self._evaluators is not None

        best_idx = max(
            range(len(self._all_metrics)),
            key=lambda i: self._all_metrics[i].get(criterion, float("-inf")),  # type: ignore[index]
        )

        best_evaluator = self._evaluators[best_idx]
        best = self._all_metrics[best_idx]
        logger.info(
            f"Best model: classifier={best.get('classifier')} "
            f"seed={best.get('random_state')} "
            f"({criterion}={best.get(criterion):.4f})"
        )
        return best_evaluator

    # ------------------------------------------------------------------
    # Forecast mode (unlabelled)
    # ------------------------------------------------------------------

    def predict_proba(self, plot: bool = True) -> pd.DataFrame:
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
            plot (bool, optional): Save a probability time-series plot.
                Defaults to True.

        Returns:
            pd.DataFrame: Index matches the future features index.  Columns
            include per-classifier metrics and consensus metrics.
        """
        cols: dict[str, np.ndarray] = {}
        model_mean_probas: dict[str, np.ndarray] = {}

        for clf_name, df_models in self._models_registry.items():
            logger.info(f"Forecasting with classifier: {clf_name}")

            mean_p, std_p, conf, pred = self._compute_model_proba(
                clf_name, df_models
            )
            model_mean_probas[clf_name] = mean_p

            cols[f"{clf_name}_eruption_probability"] = mean_p
            cols[f"{clf_name}_uncertainty"] = std_p
            cols[f"{clf_name}_confidence"] = conf
            cols[f"{clf_name}_prediction"] = pred

            logger.info(
                f"  {clf_name} — mean P(eruption): {mean_p.mean():.4f}, "
                f"mean confidence: {conf.mean():.4f}"
            )

        # Consensus across classifiers
        all_model_means = np.stack(list(model_mean_probas.values()), axis=0)
        consensus_mean: np.ndarray = all_model_means.mean(axis=0)
        consensus_std: np.ndarray = all_model_means.std(axis=0)
        consensus_pred: np.ndarray = (consensus_mean >= 0.5).astype(int)

        # Consensus confidence: fraction of classifiers agreeing with majority
        n_classifiers = all_model_means.shape[0]
        clf_preds = np.stack(
            [cols[f"{n}_prediction"] for n in self._models_registry], axis=0
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

        df_forecast = pd.DataFrame(cols, index=self.df_features.index)

        csv_path = os.path.join(self.output_dir, "predictions.csv")
        df_forecast.to_csv(csv_path)
        logger.info(f"Predictions saved to: {csv_path}")

        self._log_forecast_summary(df_forecast, model_mean_probas)

        if plot:
            self._plot_forecast(df_forecast, model_mean_probas)

        self._forecast = df_forecast
        return df_forecast

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_model_proba(
        self,
        clf_name: str,
        df_models: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate probabilities across all seeds of a single classifier.

        Args:
            clf_name: Classifier name (used in log messages only).
            df_models: Registry DataFrame for this classifier.

        Returns:
            Tuple of (mean_proba, std_proba, confidence, prediction) arrays
            of shape ``(n_windows,)``.
        """
        seed_probas: list[np.ndarray] = []

        for random_state, row in df_models.iterrows():
            significant_features_csv: str = row["significant_features_csv"]
            model_filepath: str = row["trained_model_filepath"]

            df_sig = pd.read_csv(significant_features_csv, index_col=0)
            feature_names: list[str] = df_sig.index.tolist()

            model = joblib.load(model_filepath)
            X = self.df_features[feature_names]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                p_eruption: np.ndarray = proba[:, 1] if proba.ndim > 1 else proba
            elif hasattr(model, "decision_function"):
                scores: np.ndarray = model.decision_function(X)
                p_eruption = 1.0 / (1.0 + np.exp(-scores))
            else:
                raise RuntimeError(
                    f"[{clf_name}] Model at {model_filepath} supports neither "
                    "predict_proba nor decision_function."
                )

            seed_probas.append(p_eruption)
            logger.debug(
                f"  [{clf_name}] Seed {random_state:05d} — "
                f"mean P(eruption): {p_eruption.mean():.4f}"
            )

        proba_matrix = np.stack(seed_probas, axis=0)  # (n_seeds, n_windows)

        mean_proba: np.ndarray = proba_matrix.mean(axis=0)
        std_proba: np.ndarray = proba_matrix.std(axis=0)
        prediction: np.ndarray = (mean_proba >= 0.5).astype(int)

        votes_for_eruption: np.ndarray = (proba_matrix >= 0.5).sum(axis=0)
        n_seeds = proba_matrix.shape[0]
        majority_votes = np.where(
            prediction == 1, votes_for_eruption, n_seeds - votes_for_eruption
        )
        confidence: np.ndarray = majority_votes / n_seeds

        return mean_proba, std_proba, confidence, prediction

    def _log_forecast_summary(
        self,
        df: pd.DataFrame,
        model_mean_probas: dict[str, np.ndarray],
    ) -> None:
        """Log a per-classifier + consensus forecast summary."""
        logger.info("=" * 60)
        logger.info("Forecast Summary")
        logger.info("=" * 60)
        for name, mean_p in model_mean_probas.items():
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

    def _log_metrics_summary(self, all_metrics: list[dict]) -> None:
        """Log per-classifier and consensus metrics summary."""
        df = pd.DataFrame(all_metrics)

        logger.info("=" * 60)
        logger.info("Evaluation Metrics Summary (mean ± std across seeds)")
        logger.info("=" * 60)

        for clf_name in self.model_names:
            sub = df[df["classifier"] == clf_name] if "classifier" in df.columns else df
            logger.info(f"  Classifier: {clf_name}")
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
        summary_path = os.path.join(self.output_dir, "metrics_summary.csv")
        df.describe().T.to_csv(summary_path)
        logger.info(f"Metrics summary saved to: {summary_path}")

    def _plot_forecast(
        self,
        df: pd.DataFrame,
        model_mean_probas: dict[str, np.ndarray],
    ) -> None:
        """Save a time-series probability + confidence plot.

        Multi-model: draws each classifier as a separate line plus the
        consensus with a shaded ±1 std band.
        Single-model: same as before.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        index = range(len(df)) if not hasattr(df.index, "to_pydatetime") else df.index

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Per-classifier probability lines
        for i, name in enumerate(model_mean_probas):
            col = f"{name}_eruption_probability"
            ax1.plot(
                index,
                df[col],
                linewidth=1.0,
                alpha=0.6,
                color=colors[i % len(colors)],
                linestyle="--",
                label=name,
            )

        # Consensus with uncertainty band
        cp = df["consensus_eruption_probability"]
        cu = df["consensus_uncertainty"]
        ax1.fill_between(
            index,
            (cp - cu).clip(0, 1),
            (cp + cu).clip(0, 1),
            alpha=0.2,
            color="black",
            label="consensus ±1 std",
        )
        ax1.plot(index, cp, color="black", linewidth=2.0, label="consensus")
        ax1.axhline(0.5, color="red", linestyle="--", linewidth=1, label="threshold 0.5")
        ax1.set_ylabel("Eruption Probability")
        ax1.set_ylim(0, 1)
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(alpha=0.3)
        ax1.set_title("Eruption Forecast" + (" — Consensus" if self._multi_model else ""))

        # Per-classifier confidence lines
        for i, name in enumerate(model_mean_probas):
            ax2.plot(
                index,
                df[f"{name}_confidence"],
                linewidth=1.0,
                alpha=0.6,
                color=colors[i % len(colors)],
                linestyle="--",
                label=name,
            )

        ax2.plot(
            index,
            df["consensus_confidence"],
            color="black",
            linewidth=2.0,
            label="consensus",
        )
        ax2.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        ax2.set_ylabel("Confidence")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(alpha=0.3)
        ax2.set_xlabel("Window")

        plt.tight_layout()
        path = os.path.join(self.figures_dir, "eruption_forecast.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Forecast plot saved to: {path}")
