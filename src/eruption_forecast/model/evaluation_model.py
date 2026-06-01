"""Evaluation of pre-trained ensemble models on labeled seismic data.

Provides :class:`EvaluationModel`, the final stage of the
**Training → Prediction → Evaluate** pipeline.  It evaluates an already-trained
``ClassifierEnsemble`` against ground-truth labels **without re-extracting
features or re-fitting any model**.

Two operating modes:

- **Training reuse** (``model`` is a ``TrainingModel``): features, the loaded
  ``ClassifierEnsemble``, the window grid, and ground-truth labels are taken
  directly from the training stage.  ``eruption_dates`` is not required.
- **Prediction reuse** (``model`` is a ``PredictionModel``): features, the
  loaded ``ClassifierEnsemble``, and the window grid are taken from the
  prediction stage.  ``eruption_dates`` is required so that true labels can be
  built on the prediction window grid.

The constructor expects a live ``TrainingModel`` or ``PredictionModel``
instance.  Use :meth:`EvaluationModel.from_file` to construct an instance from
a saved ``.pkl`` produced by ``TrainingModel.save()`` or
``PredictionModel.save()``.  Output artefacts are written to
``{output_dir}/evaluation/{kind}/`` so that training-window and
prediction-window results never share the same directory.
"""

import os
import json
from typing import Self, Literal

import joblib
import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.model.base_model import BaseModel
from eruption_forecast.label.label_builder import LabelBuilder
from eruption_forecast.model.training_model import TrainingModel
from eruption_forecast.model.model_evaluator import ModelEvaluator
from eruption_forecast.model.prediction_model import PredictionModel
from eruption_forecast.model.classifier_comparator import ClassifierComparator
from eruption_forecast.model.multi_model_evaluator import MultiModelEvaluator
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


class EvaluationModel(BaseModel):
    """Evaluate a pre-trained ensemble against ground-truth labels.

    Reuses the extracted features and in-memory seed models produced upstream
    and computes per-seed metrics (via ``model.predict`` only — never a re-fit),
    then aggregates mean ± std across seeds for each classifier.

    Two operating modes:

    - **Training reuse** — pass a completed ``TrainingModel`` as ``model``.
      Features, the loaded ``ClassifierEnsemble``, the window grid, and
      ground-truth labels are taken directly from the training stage.
      ``eruption_dates`` is not required.
    - **Prediction reuse** — pass a completed ``PredictionModel`` as ``model``.
      Features, the loaded ``ClassifierEnsemble``, and the window grid are
      taken from the prediction stage.  ``eruption_dates`` is required so that
      true labels can be built on the prediction window grid.

    The window step is always taken from ``model.window_step`` /
    ``model.window_step_unit``.  Use :meth:`from_file` to construct an instance
    from a saved ``.pkl`` produced by ``TrainingModel.save()`` or
    ``PredictionModel.save()``.

    Output artefacts are written to ``{output_dir}/evaluation/{kind}/`` so that
    training-window and prediction-window results can coexist in the same
    ``output_dir`` without metric-file collisions.

    Args:
        model (TrainingModel | PredictionModel): Live trained model source. Use
            :meth:`from_file` to load from a ``.pkl`` path.
        eruption_dates (list[str] | None): Ground-truth eruption dates in
            ``YYYY-MM-DD`` format.  Required for prediction reuse.  May be
            ``None`` for training reuse (truth is embedded in the training
            labels). Defaults to ``None``.
        overwrite (bool): Re-run and overwrite cached output files. Defaults to
            ``False``.
        output_dir (str | None): Root output directory. Derived from ``model``
            when ``None``. Defaults to ``None``.
        root_dir (str | None): Project root used for path resolution. Defaults
            to ``None``.
        n_jobs (int): Number of parallel workers. Defaults to ``1``.
        verbose (bool): Emit detailed progress logs. Defaults to ``False``.

    Example:
        >>> # Training reuse — evaluate on the training window
        >>> em = EvaluationModel(model=training_model)
        >>> metrics = em.evaluate()

        >>> # Prediction reuse — evaluate on the forecast window
        >>> em = EvaluationModel(
        ...     model=prediction_model,
        ...     eruption_dates=["2025-03-20"],
        ... )
        >>> metrics = em.evaluate()
    """

    def __init__(
        self,
        model: TrainingModel | PredictionModel,
        eruption_dates: list[str] | None = None,
        overwrite: bool = False,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> None:
        """Initialize EvaluationModel in training-reuse or prediction-reuse mode.

        Reuses ``tremor_data``, ``start_date``, ``end_date``, ``window_size``,
        ``window_step`` / ``window_step_unit``, the loaded
        ``ClassifierEnsemble``, and the extracted ``features_df`` from
        ``model``.  ``output_dir`` falls back to ``model.output_dir`` when not
        explicitly provided.  ``build_label`` is invoked at the end of
        construction so the instance is ready for :meth:`evaluate` immediately.

        Args:
            model (TrainingModel | PredictionModel): Live trained model source.
            eruption_dates (list[str] | None): Ground-truth eruption dates in
                ``YYYY-MM-DD`` format.  Optional for training reuse; required
                for prediction reuse. Defaults to ``None``.
            overwrite (bool): Overwrite cached outputs. Defaults to ``False``.
            output_dir (str | None): Root output directory. Derived from
                ``model`` when ``None``. Defaults to ``None``.
            root_dir (str | None): Project root. Defaults to ``None``.
            n_jobs (int): Parallel workers. Defaults to ``1``.
            verbose (bool): Verbose logging. Defaults to ``False``.

        Raises:
            ValueError: If ``eruption_dates`` is ``None`` when ``model`` is a
                ``PredictionModel``.
            ValueError: If ``model.ClassifierEnsemble`` is ``None``.
            ValueError: If ``model.window_step`` or ``model.window_step_unit``
                is ``None``.
        """
        model_kind = model.kind

        if model_kind == "prediction" and eruption_dates is None:
            raise ValueError(
                "eruption_dates for evaluation is required when model is a PredictionModel."
            )

        if model.ClassifierEnsemble is None:
            raise ValueError(
                f"ClassifierEnsemble is not found for {type(model).__name__}"
            )

        if model.window_step is None or model.window_step_unit is None:
            raise ValueError(
                f"window_step and window_step_unit are required for {type(model).__name__}"
            )

        super().__init__(
            tremor_data=model._tremor_data,
            start_date=model.start_date,
            end_date=model.end_date,
            window_size=model.window_size,
            eruption_dates=eruption_dates,
            overwrite=overwrite,
            output_dir=output_dir if output_dir is not None else model.output_dir,
            root_dir=root_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.model: TrainingModel | PredictionModel = model
        self.model_kind = model_kind
        self.ClassifierEnsemble: ClassifierEnsemble = model.ClassifierEnsemble
        self.features_df = model.features_df
        self.window_step: int = model.window_step
        self.window_step_unit: Literal["minutes", "hours"] = model.window_step_unit
        self.basename = f"{self.start_date_str}_{self.end_date_str}"

        # ``self.y_true`` always holds ground-truth labels (0/1) when populated.
        # Training reuse: ``TrainingModel.labels`` is already truth on
        # ``pd.RangeIndex``, so assign directly. Prediction reuse: no truth
        # exists upstream, so start empty; ``evaluate()`` calls
        # ``build_label()`` which constructs truth from a fresh
        # ``LabelBuilder`` joined against ``self.model.labels`` (the window-id
        # mapping).
        if self.model_kind == "training":
            self.y_true: pd.Series = model.labels
        else:
            self.y_true: pd.Series = pd.Series(dtype=int, name="is_erupted")

        (
            self.evaluation_dir,
            self.classifiers_dir,
        ) = self.set_directories()

        # Will be set after evaluate() called
        self.metrics: dict[str, pd.DataFrame] = {}

        self.validate()

    @classmethod
    def from_file(
        cls,
        filepath: str,
        eruption_dates: list[str] | None = None,
        overwrite: bool = False,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> "EvaluationModel":
        """Construct an EvaluationModel from a saved ``TrainingModel`` or ``PredictionModel`` ``.pkl``.

        Loads the pickle, dispatches on its ``kind`` attribute, and forwards
        every remaining argument to :meth:`__init__`.  Use this instead of the
        constructor when the upstream stage was persisted to disk via
        :meth:`BaseModel.save`.

        Args:
            filepath (str): Path to a ``.pkl`` file produced by
                ``TrainingModel.save()`` or ``PredictionModel.save()``.
            eruption_dates (list[str] | None): Ground-truth eruption dates in
                ``YYYY-MM-DD`` format.  Optional for training reuse; required
                for prediction reuse. Defaults to ``None``.
            overwrite (bool): Overwrite cached outputs. Defaults to ``False``.
            output_dir (str | None): Root output directory. Derived from the
                loaded model when ``None``. Defaults to ``None``.
            root_dir (str | None): Project root. Defaults to ``None``.
            n_jobs (int): Parallel workers. Defaults to ``1``.
            verbose (bool): Verbose logging. Defaults to ``False``.

        Returns:
            EvaluationModel: A fully constructed instance ready for
                :meth:`evaluate`.

        Raises:
            FileNotFoundError: If ``filepath`` does not exist on disk.
            ValueError: If the loaded object is not a ``TrainingModel`` or
                ``PredictionModel``.

        Example:
            >>> em = EvaluationModel.from_file(
            ...     "output/VG.OJN.00.EHZ/PredictionModel_2025-03-16_2025-03-22.pkl",
            ...     eruption_dates=["2025-03-20"],
            ... )
            >>> metrics = em.evaluate()
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        model = joblib.load(filepath)

        if not isinstance(model, (TrainingModel, PredictionModel)):
            raise ValueError(
                f"Loaded .pkl is not a TrainingModel or PredictionModel "
                f"(got {type(model).__name__})."
            )

        return cls(
            model=model,
            eruption_dates=eruption_dates,
            overwrite=overwrite,
            output_dir=output_dir,
            root_dir=root_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def set_directories(self) -> tuple[str, str]:
        """Build the output directory paths for evaluation artefacts.

        Output paths are namespaced by mode — ``evaluation/training/`` or
        ``evaluation/prediction/`` — so that results from the two modes written
        to the same ``output_dir`` never share a metrics directory.  No
        directories are created here; call :meth:`create_directories` for that.

        Returns:
            tuple: A three-element tuple containing:

                - evaluation_dir (str): Root evaluation output path.
                - classifiers_dir (str): Classifier-level results subdirectory.
        """
        evaluation_dir = os.path.join(self.output_dir, "evaluation", self.model_kind)
        classifiers_dir = os.path.join(evaluation_dir, "classifiers")

        return evaluation_dir, classifiers_dir

    def create_directories(self) -> None:
        """Create the evaluation output directories on disk.

        Idempotent: existing directories are left untouched.  Called once at
        the start of :meth:`evaluate` so that label rebuilding in
        :meth:`build_label` and classifier output writes never race on a
        missing parent directory.
        """
        ensure_dir(self.evaluation_dir)
        ensure_dir(self.classifiers_dir)

    def validate(self) -> Self:
        """Run model-specific validation hooks (currently a no-op).

        Implemented to satisfy the :class:`BaseModel` interface.  All
        validation needed by ``EvaluationModel`` is performed inline in
        :meth:`__init__` against the upstream ``model``.

        Returns:
            Self: The current instance, enabling method chaining.
        """
        return self

    def describe(self) -> str:
        """Return a one-line human-readable description of this instance.

        Returns:
            str: Prose description naming the operating mode, the configured
                classifiers, and the evaluation date range.
        """
        classifiers = ", ".join(self.ClassifierEnsemble.classifiers)
        return (
            f"EvaluationModel ({self.model_kind} mode) evaluating [{classifiers}] on "
            f"{self.start_date_str} to {self.end_date_str}."
        )

    def to_dict(self) -> dict:
        """Return a structured dictionary of all configuration fields.

        ``eruption_dates`` are emitted as a plain list of ``YYYY-MM-DD``
        strings (the format returned by :func:`sort_dates`).  An empty list is
        used when no eruption dates were provided.

        Returns:
            dict: Mapping of field names to their current values.
        """
        return {
            "mode": self.model_kind,
            "start_date": self.start_date_str,
            "end_date": self.end_date_str,
            "window_size": self.window_size,
            "window_step": self.window_step,
            "window_step_unit": self.window_step_unit,
            "eruption_dates": list(self.eruption_dates) if self.eruption_dates else [],
            "classifiers": self.ClassifierEnsemble.classifiers,
            "output_dir": self.output_dir,
        }

    def to_prompt(self) -> str:
        """Return a structured text block suitable for LLM prompt input.

        Each key/value pair from :meth:`to_dict` is emitted as a single
        ``"- key: value"`` line, joined with newlines.

        Returns:
            str: Formatted bullet-list block derived from :meth:`to_dict`.
        """
        data = self.to_dict()
        lines = [f"- {k}: {v}" for k, v in data.items()]
        return "\n".join(lines)

    def build_label(
        self,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
    ) -> Self:
        """Build ground-truth labels for the evaluation window grid.

        In **training reuse** mode this is a no-op: ``self.y_true`` was
        populated directly from ``TrainingModel.labels`` at construction.  In
        **prediction reuse** mode a fresh :class:`LabelBuilder` is run over
        the evaluation period and its ``is_erupted`` flags are joined onto
        the prediction window-id mapping (``self.model.labels``) by datetime,
        then re-indexed by window id and assigned to ``self.y_true``.

        Args:
            window_step (int): Step size between consecutive windows.
            window_step_unit (Literal["minutes", "hours"]): Unit of
                ``window_step``.

        Returns:
            Self: The current instance, enabling method chaining.

        Raises:
            ValueError: If ``self.eruption_dates`` is ``None`` while prediction
                reuse requires rebuilding labels.
        """
        if self.model_kind == "training":
            return self

        if self.eruption_dates is None:
            raise ValueError("No eruption dates provided.")

        label_builder = LabelBuilder(
            start_date=self.start_date,
            end_date=self.end_date,
            window_step=window_step,
            window_step_unit=window_step_unit,
            eruption_dates=self.eruption_dates,
            day_to_forecast=self.window_size,
            verbose=self.verbose,
        ).build(
            save_label=False,
            plot_distribution=False,
        )

        id_series = self.model.labels
        true_label_df = label_builder.df
        merged = id_series.to_frame().join(true_label_df["is_erupted"], how="left")
        y_true = merged.set_index("id")["is_erupted"].fillna(0).astype(int)
        y_true.name = "is_erupted"

        self.y_true = y_true

        return self

    def extract_features(self) -> Self:
        """Return immediately — features are always reused from the upstream stage.

        Implemented to satisfy the :class:`BaseModel` interface and to allow
        ``EvaluationModel`` to be dropped into method chains that call
        ``extract_features()``.  The upstream ``TrainingModel`` or
        ``PredictionModel`` already carries the extracted ``features_df``; no
        tsfresh re-run is ever performed here.

        Returns:
            Self: The current instance, enabling method chaining.
        """
        if self.verbose:
            logger.info(
                f"{self.model_kind} reuse: features already available, "
                "skipping tsfresh extraction."
            )

        return self

    def evaluate(
        self,
        plot_per_seed: bool = False,
        plot_aggregate: bool = True,
        plot_shap: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Evaluate every seed model against the true labels and aggregate.

        For each classifier and seed in the loaded ``ClassifierEnsemble``,
        slices the reused feature matrix to the seed's selected features and
        runs :class:`ModelEvaluator` to compute per-seed metrics.
        ``ModelEvaluator`` calls ``model.predict`` only — models are **never
        re-fit**.  Per-seed metrics are cached as JSON under
        ``{classifiers_dir}/{classifier}/metrics/json/`` and reused on
        subsequent runs unless ``self.overwrite`` is ``True``.  Per-seed
        metrics are aggregated to mean ± std per classifier and the resulting
        ``DataFrame`` is also assigned to ``self.metrics``.

        Args:
            plot_per_seed (bool): Render per-seed evaluation plots via
                ``ModelEvaluator.plot_all``.  Expensive across many seeds.
                Defaults to ``False``.
            plot_aggregate (bool): Render aggregate plots per classifier via
                :class:`MultiModelEvaluator`. Defaults to ``True``.
            plot_shap (bool): Enable SHAP plots within per-seed
                ``ModelEvaluator`` (only relevant when ``plot_per_seed=True``).
                Defaults to ``False``.

        Returns:
            dict[str, pd.DataFrame]: Mapping of classifier name to a DataFrame
                of per-seed metrics (one row per seed).

        Raises:
            ValueError: If ``self.eruption_dates`` is ``None`` in prediction
                reuse mode (raised by ``build_label()``).
            ValueError: If ``self.y_true`` is still empty after
                ``build_label()`` — usually means the upstream
                ``TrainingModel`` was constructed without running
                ``extract_features``.
            ValueError: If ``self.features_df`` is empty (the upstream model
                was passed in without running ``extract_features``).
        """
        self.create_directories()

        self.build_label(
            window_step=self.window_step,
            window_step_unit=self.window_step_unit,
        )

        if self.y_true.empty:
            raise ValueError(
                "Ground-truth labels (self.y_true) are empty after build_label(). "
                "This usually means the upstream TrainingModel was constructed "
                "without running extract_features()."
            )
        if self.features_df.empty:
            raise ValueError(
                "Features are empty. Ensure the source model ran extract_features() "
                "before being passed to EvaluationModel."
            )

        metrics_per_classifier: dict[str, pd.DataFrame] = {}

        for classifier_name, seed_ensemble in self.ClassifierEnsemble.ensembles.items():
            clf_dir = os.path.join(self.classifiers_dir, classifier_name)
            metrics_dir = os.path.join(clf_dir, "metrics")
            json_dir = os.path.join(metrics_dir, "json")
            figures_dir = os.path.join(clf_dir, "figures")
            ensure_dir(json_dir)

            all_metrics: list[dict] = []

            for seed_record in seed_ensemble.seeds:
                missing_keys: list[str] = []
                for key in ("random_state", "model", "feature_names"):
                    if key not in seed_record.keys():
                        missing_keys.append(key)

                if len(missing_keys) > 0:
                    raise ValueError(
                        f"SeedEnsemble.seeds missing keys: {missing_keys}. "
                        f"Available keys: {', '.join(seed_record.keys())}"
                    )

                random_state: int = seed_record["random_state"]
                seed_model = seed_record["model"]
                feature_names: list[str] = seed_record["feature_names"]

                metrics_filepath = os.path.join(json_dir, f"{random_state:05d}.json")

                if not self.overwrite and os.path.isfile(metrics_filepath):
                    with open(metrics_filepath) as f:
                        metrics: dict = json.load(f)
                    all_metrics.append(metrics)
                    if self.verbose:
                        logger.info(
                            f"{classifier_name}/{random_state:05d}: cached metrics loaded, skipping."
                        )
                    continue

                available_features = [
                    feature_name
                    for feature_name in feature_names
                    if feature_name in self.features_df.columns
                ]

                if not available_features:
                    logger.warning(
                        f"{classifier_name}/{random_state:05d}: none of the "
                        "seed features are present in the data — skipping."
                    )
                    continue

                X_test = self.features_df[available_features]
                common_index = X_test.index.intersection(self.y_true.index)
                X_test = X_test.loc[common_index]
                y_true = self.y_true.loc[common_index]
                y_pred = seed_model.predict(X_test)

                evaluator = ModelEvaluator(
                    model=seed_model,
                    X_test=X_test,
                    y_test=y_true,
                    model_name=classifier_name,
                    output_dir=figures_dir,
                    selected_features=available_features,
                    random_state=random_state,
                    plot_shap=plot_shap,
                    verbose=self.verbose,
                )

                metrics = evaluator.get_metrics()
                metrics["random_state"] = random_state

                with open(metrics_filepath, "w") as f:
                    json.dump(metrics, f, indent=4)

                if plot_per_seed:
                    try:
                        evaluator.plot_all(dpi=150)
                    except Exception as e:
                        logger.warning(
                            f"{classifier_name}/{random_state:05d}: "
                            f"plot_all() skipped. Reason: {e}"
                        )

                if self.verbose:
                    logger.info(
                        f"{classifier_name}/{random_state:05d}: "
                        f"balanced_accuracy="
                        f"{metrics.get('balanced_accuracy', float('nan')):.4f} "
                        f"(y_pred positives={int(y_pred.sum())}, "
                        f"y_true positives={int(y_true.sum())})"
                    )

                all_metrics.append(metrics)

            if not all_metrics:
                logger.warning(f"{classifier_name}: no seed results — skipping.")
                continue

            self._aggregate_metrics(
                all_metrics=all_metrics,
                suffix_filename=self.basename,
                classifier_name=classifier_name,
                classifier_dir=clf_dir,
            )

            if plot_aggregate:
                self._plot_aggregate(metrics_dir, classifier_name, clf_dir)

            metrics_per_classifier[classifier_name] = pd.DataFrame(all_metrics)

        self.metrics = metrics_per_classifier
        return metrics_per_classifier

    def compare(
        self,
        metrics: str | list[str] | None = None,
        output_dir: str | None = None,
    ) -> ClassifierComparator:
        """Build a :class:`ClassifierComparator` from the per-classifier results.

        Constructs one ``MultiModelEvaluator`` per classifier from the metrics
        JSON directory written by :meth:`evaluate` and returns a
        ``ClassifierComparator`` that can produce cross-classifier rankings and
        comparison plots.  The live ``ClassifierEnsemble`` plus
        ``self.features_df`` and ``self.y_true`` are forwarded as
        ``ensemble_source`` so ROC plotting uses in-memory models and never
        requires a trained-model registry CSV on disk.

        Output of the returned comparator is written under
        ``{evaluation_dir}/comparison/`` (i.e. peer to ``classifiers/``).

        Args:
            metrics (str | list[str] | None, optional): Metric or ordered list
                of metrics used for ranking and plots.  When None, the
                comparator falls back to its own default metrics list.
                Defaults to None.

        Returns:
            ClassifierComparator: Comparator wired to the per-classifier metrics
                directories and the in-memory ensemble.

        Examples:
            >>> em = EvaluationModel(model=training_model)
            >>> em.evaluate()
            >>> comparator = em.compare()
            >>> ranking = comparator.get_ranking()
            >>> figures = comparator.plot_all()
        """
        evaluators: dict[str, MultiModelEvaluator] = {
            name: MultiModelEvaluator(
                metrics_dir=os.path.join(self.classifiers_dir, name, "metrics"),
                classifier_name=name,
                output_dir=output_dir,
            )
            for name in self.ClassifierEnsemble.ensembles
        }

        ensemble_source = (self.ClassifierEnsemble, self.features_df, self.y_true)

        return ClassifierComparator.from_evaluators(
            evaluators=evaluators,
            output_dir=self.evaluation_dir,
            metrics=metrics,
            ensemble_source=ensemble_source,
        )

    def _plot_aggregate(
        self, metrics_dir: str, classifier_name: str, output_dir: str | None = None
    ) -> None:
        """Render aggregate plots for one classifier from per-seed metrics JSON.

        Delegates to :class:`MultiModelEvaluator`, which reads ``*.json`` files
        from ``{metrics_dir}/json/``.  Failures are logged and swallowed so a
        plotting error never aborts the evaluation run.

        Args:
            metrics_dir (str): Per-classifier metrics directory whose ``json/``
                subfolder holds the per-seed metric files.
            classifier_name (str): Classifier name used in the warning log
                message when plotting fails.
        """
        output_dir = output_dir or self.output_dir

        try:
            MultiModelEvaluator(
                output_dir=output_dir,
                metrics_dir=metrics_dir,
                classifier_name=classifier_name,
            ).plot_all()
        except Exception as exc:
            logger.warning(f"{classifier_name}: aggregate plots skipped. Reason: {exc}")

    def _aggregate_metrics(
        self,
        all_metrics: list[dict],
        suffix_filename: str,
        classifier_name: str,
        classifier_dir: str,
    ) -> None:
        """Aggregate per-seed metrics into summary statistics for one classifier.

        Computes ``DataFrame.describe()`` statistics across all seed rows and
        writes both the per-seed table and the summary to CSV under
        ``classifier_dir``.  A condensed mean ± std line is logged for the
        common metrics (``accuracy``, ``balanced_accuracy``, ``f1_score``,
        ``precision``, ``recall``) when they are present in the records.

        Args:
            all_metrics (list[dict]): One metrics dictionary per seed.
            suffix_filename (str): Suffix appended to the output filenames
                (typically ``self.basename``).
            classifier_name (str): Classifier name used in the log header.
            classifier_dir (str): Directory where ``metrics_summary_*.csv`` and
                ``all_metrics_*.csv`` are written.
        """
        ensure_dir(classifier_dir)

        df_metrics = pd.DataFrame(all_metrics)

        summary = df_metrics.describe().T
        summary_filepath = os.path.join(
            classifier_dir, f"metrics_summary_{suffix_filename}.csv"
        )
        summary.to_csv(summary_filepath)

        all_metrics_filepath = os.path.join(
            classifier_dir, f"all_metrics_{suffix_filename}.csv"
        )
        df_metrics = df_metrics.set_index("random_state")
        df_metrics.to_csv(all_metrics_filepath, index=True)

        logger.info("=" * 60)
        logger.info(f"Metrics Summary — {classifier_name} (mean ± std across seeds)")
        logger.info("=" * 60)
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "f1_score",
            "precision",
            "recall",
        ):
            if metric not in df_metrics.columns:
                continue
            mean = df_metrics[metric].mean()
            std = df_metrics[metric].std()
            logger.info(f"{metric:20s}: {mean:.4f} ± {std:.4f}")
        logger.info("=" * 60)

        if self.verbose:
            logger.info(f"Summary metrics saved to: {summary_filepath}")
            logger.info(f"All metrics saved to: {all_metrics_filepath}")
