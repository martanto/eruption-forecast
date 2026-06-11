"""Stage class for Explanatory Model Analysis (DALEX/SHAP) over tree classifiers.

Sibling to :class:`~eruption_forecast.model.evaluation_model.EvaluationModel`.
Reuses the extracted features and in-memory seed models produced upstream by
:class:`~eruption_forecast.model.training_model.TrainingModel` or
:class:`~eruption_forecast.model.prediction_model.PredictionModel` and delegates
the per-classifier × per-seed DALEX loop to
:class:`~eruption_forecast.ensemble.dalex_explainer_ensemble.DalexExplainerEnsemble`.
"""

import os
from typing import Self, Literal

import joblib
import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.model.base_model import BaseModel
from eruption_forecast.label.label_builder import LabelBuilder
from eruption_forecast.model.training_model import TrainingModel
from eruption_forecast.model.prediction_model import PredictionModel
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble
from eruption_forecast.ensemble.dalex_explainer_ensemble import (
    TREE_CLASSIFIERS,
    DalexExplainerEnsemble,
)


class ExplanationModel(BaseModel):
    """Run DALEX-based Explanatory Model Analysis over a fitted ensemble.

    Two operating modes mirror
    :class:`~eruption_forecast.model.evaluation_model.EvaluationModel`:

    - **Training reuse** — pass a completed ``TrainingModel`` as ``model``.
      Features, the loaded ``ClassifierEnsemble``, the window grid, and
      ground-truth labels are taken directly from the training stage.
      ``eruption_dates`` is not required.
    - **Prediction reuse** — pass a completed ``PredictionModel`` as ``model``.
      Features, the loaded ``ClassifierEnsemble``, and the window grid are
      taken from the prediction stage. ``eruption_dates`` is required so that
      true labels can be built on the prediction window grid.

    Restricted to tree classifiers (:data:`TREE_CLASSIFIERS`). Non-tree
    classifiers in the ensemble are skipped with one INFO-level log line each
    by the underlying
    :class:`~eruption_forecast.ensemble.dalex_explainer_ensemble.DalexExplainerEnsemble`.

    Output artefacts are written to ``{output_dir}/explanation/{kind}/`` so
    training-window and prediction-window results can coexist in the same
    ``output_dir`` without colliding with each other or with the metrics
    artefacts under ``{output_dir}/evaluation/{kind}/``.

    Args:
        model (TrainingModel | PredictionModel): Live trained model source. Use
            :meth:`from_file` to load from a ``.pkl`` path.
        eruption_dates (list[str] | None): Ground-truth eruption dates in
            ``YYYY-MM-DD`` format. Required for prediction reuse. May be
            ``None`` for training reuse (truth is embedded in the training
            labels). Defaults to ``None``.
        n_seeds_to_explain (int): Number of seeds sampled per classifier.
            Defaults to ``10``.
        n_observations_to_explain (int): Number of observations fed to
            ``predict_parts`` per seed. Defaults to ``5``.
        top_k_features (int): Number of top-ranked features fed to
            ``model_profile`` per seed. Defaults to ``5``.
        overwrite (bool): Re-run and overwrite cached output files. Defaults
            to ``False``.
        output_dir (str | None): Root output directory. Derived from ``model``
            when ``None``. Defaults to ``None``.
        root_dir (str | None): Project root used for path resolution. Defaults
            to ``None``.
        n_jobs (int): Number of parallel workers (reserved for future use).
            Defaults to ``1``.
        verbose (bool): Emit detailed progress logs. Defaults to ``False``.

    Example:
        >>> # Training reuse — in-sample explanations
        >>> em = ExplanationModel(model=training_model)
        >>> dalex_ensemble = em.explain()

        >>> # Prediction reuse — forecast-window explanations
        >>> em = ExplanationModel(
        ...     model=prediction_model,
        ...     eruption_dates=["2025-03-20"],
        ...     n_seeds_to_explain=5,
        ... )
        >>> dalex_ensemble = em.explain()
    """

    def __init__(
        self,
        model: TrainingModel | PredictionModel,
        eruption_dates: list[str] | None = None,
        n_seeds_to_explain: int = 10,
        n_observations_to_explain: int = 5,
        top_k_features: int = 5,
        overwrite: bool = False,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> None:
        """Initialise ``ExplanationModel`` in training- or prediction-reuse mode.

        Reuses ``tremor_data``, ``start_date``, ``end_date``, ``window_size``,
        ``window_step`` / ``window_step_unit``, the loaded
        ``ClassifierEnsemble``, and the extracted ``features_df`` from
        ``model``. ``output_dir`` falls back to ``model.output_dir`` when not
        explicitly provided.

        Args:
            model (TrainingModel | PredictionModel): Live trained model source.
            eruption_dates (list[str] | None): See class docstring.
            n_seeds_to_explain (int): See class docstring.
            n_observations_to_explain (int): See class docstring.
            top_k_features (int): See class docstring.
            overwrite (bool): See class docstring.
            output_dir (str | None): See class docstring.
            root_dir (str | None): See class docstring.
            n_jobs (int): See class docstring.
            verbose (bool): See class docstring.

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
                "eruption_dates for explanation is required when model is a "
                "PredictionModel."
            )

        if model.ClassifierEnsemble is None:
            raise ValueError(
                f"ClassifierEnsemble is not found for {type(model).__name__}"
            )

        if model.window_step is None or model.window_step_unit is None:
            raise ValueError(
                f"window_step and window_step_unit are required for "
                f"{type(model).__name__}"
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
        self.basename: str = f"{self.start_date_str}_{self.end_date_str}"
        self.n_seeds_to_explain = n_seeds_to_explain
        self.n_observations_to_explain = n_observations_to_explain
        self.top_k_features = top_k_features

        # Training reuse: ``TrainingModel.labels`` is already truth on
        # ``pd.RangeIndex``. Prediction reuse: no truth exists upstream, so
        # start empty; ``explain()`` calls ``build_label()`` which constructs
        # truth from a fresh ``LabelBuilder``.
        if self.model_kind == "training":
            self.y_true: pd.Series = model.labels
        else:
            self.y_true = pd.Series(dtype=int, name="is_erupted")

        self.explanation_dir, self.classifiers_dir = self.set_directories()

        # Populated by :meth:`explain`.
        self.DalexExplainerEnsemble: DalexExplainerEnsemble | None = None

        self.validate()

    @classmethod
    def from_file(
        cls,
        filepath: str,
        eruption_dates: list[str] | None = None,
        n_seeds_to_explain: int = 10,
        n_observations_to_explain: int = 5,
        top_k_features: int = 5,
        overwrite: bool = False,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> "ExplanationModel":
        """Build an ``ExplanationModel`` from a saved ``Training/PredictionModel`` ``.pkl``.

        Args:
            filepath (str): Path to a ``.pkl`` file produced by
                ``TrainingModel.save()`` or ``PredictionModel.save()``.
            eruption_dates (list[str] | None): See class docstring.
            n_seeds_to_explain (int): See class docstring.
            n_observations_to_explain (int): See class docstring.
            top_k_features (int): See class docstring.
            overwrite (bool): See class docstring.
            output_dir (str | None): See class docstring.
            root_dir (str | None): See class docstring.
            n_jobs (int): See class docstring.
            verbose (bool): See class docstring.

        Returns:
            ExplanationModel: A fully constructed instance ready for
                :meth:`explain`.

        Raises:
            FileNotFoundError: If ``filepath`` does not exist on disk.
            TypeError: If the loaded object is not a ``TrainingModel`` or
                ``PredictionModel``.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        model = joblib.load(filepath)

        if not isinstance(model, (TrainingModel, PredictionModel)):
            raise TypeError(
                f"Loaded .pkl is not a TrainingModel or PredictionModel "
                f"(got {type(model).__name__})."
            )

        return cls(
            model=model,
            eruption_dates=eruption_dates,
            n_seeds_to_explain=n_seeds_to_explain,
            n_observations_to_explain=n_observations_to_explain,
            top_k_features=top_k_features,
            overwrite=overwrite,
            output_dir=output_dir,
            root_dir=root_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def set_directories(self) -> tuple[str, str]:
        """Build the output directory paths for explanation artefacts.

        Returns:
            tuple: ``(explanation_dir, classifiers_dir)``.
        """
        explanation_dir = os.path.join(
            self.output_dir, "explanation", self.model_kind
        )
        classifiers_dir = os.path.join(explanation_dir, "classifiers")
        return explanation_dir, classifiers_dir

    def create_directories(self) -> None:
        """Create the explanation output directories on disk.

        Idempotent: existing directories are left untouched.
        """
        ensure_dir(self.explanation_dir)
        ensure_dir(self.classifiers_dir)

    def validate(self) -> Self:
        """Warn early when the ensemble holds no tree classifiers.

        DALEX will silently no-op in that case (no classifiers pass the
        :data:`TREE_CLASSIFIERS` filter), so we flag it at construction time.

        Returns:
            Self: The current instance, enabling method chaining.
        """
        tree = [n for n in self.ClassifierEnsemble.classifiers if n in TREE_CLASSIFIERS]
        if not tree:
            logger.warning(
                "ExplanationModel: no tree classifiers in ensemble; available "
                f"= {self.ClassifierEnsemble.classifiers}. DALEX explanations "
                "will be skipped entirely."
            )
        return self

    def describe(self) -> str:
        """Return a one-line description of this instance.

        Returns:
            str: Prose description naming the operating mode, the configured
                classifiers, and the date range.
        """
        classifiers = ", ".join(self.ClassifierEnsemble.classifiers)
        return (
            f"ExplanationModel ({self.model_kind} mode) explaining "
            f"[{classifiers}] on {self.start_date_str} to {self.end_date_str}."
        )

    def to_dict(self) -> dict:
        """Return a structured dictionary of all configuration fields.

        Returns:
            dict: Mapping of field names to their current values. The
                ``eruption_dates`` field is emitted as a list of strings even
                when none were supplied (an empty list).
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
            "tree_classifiers": [
                n
                for n in self.ClassifierEnsemble.classifiers
                if n in TREE_CLASSIFIERS
            ],
            "n_seeds_to_explain": self.n_seeds_to_explain,
            "n_observations_to_explain": self.n_observations_to_explain,
            "top_k_features": self.top_k_features,
            "output_dir": self.output_dir,
        }

    def to_prompt(self) -> str:
        """Return a bullet-list block suitable for LLM prompt input.

        Returns:
            str: Formatted block derived from :meth:`to_dict`.
        """
        return "\n".join(f"- {k}: {v}" for k, v in self.to_dict().items())

    def build_label(
        self,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
    ) -> Self:
        """Build ground-truth labels for the explanation window grid.

        In **training reuse** mode this is a no-op: ``self.y_true`` was
        populated directly from ``TrainingModel.labels`` at construction. In
        **prediction reuse** mode a fresh :class:`LabelBuilder` is run over
        the explanation period and its ``is_erupted`` flags are joined onto
        the prediction window-id mapping (``self.model.labels``) by datetime,
        then re-indexed by window id and assigned to ``self.y_true``.

        Args:
            window_step (int): Step size between consecutive windows.
            window_step_unit (Literal["minutes", "hours"]): Unit of
                ``window_step``.

        Returns:
            Self: The current instance, enabling method chaining.

        Raises:
            ValueError: If ``self.eruption_dates`` is ``None`` while
                prediction reuse requires rebuilding labels.
        """
        if self.model_kind == "training":
            return self

        if self.eruption_dates is None:
            raise ValueError("No eruption dates provided.")

        y_true_df = (
            LabelBuilder(
                start_date=self.start_date,
                end_date=self.end_date,
                window_step=window_step,
                window_step_unit=window_step_unit,
                eruption_dates=self.eruption_dates,
                day_to_forecast=self.window_size,
                verbose=self.verbose,
            )
            .build(
                save_label=False,
                plot_distribution=False,
            )
            .df
        )

        id_series = self.model.labels
        merged = id_series.to_frame().join(y_true_df["is_erupted"], how="left")
        y_true = merged.set_index("id")["is_erupted"].fillna(0).astype(int)
        y_true.name = "is_erupted"

        y_true_dir = os.path.join(self.explanation_dir, "labels")
        ensure_dir(y_true_dir)
        y_true_filepath = os.path.join(y_true_dir, "y_true.csv")
        y_true.to_csv(y_true_filepath, index=True)

        self.y_true = y_true

        if self.verbose:
            logger.info(f"Label y_true saved to {y_true_filepath}")

        return self

    def extract_features(self) -> Self:
        """Return immediately — features are always reused from the upstream stage.

        Implemented to satisfy the :class:`BaseModel` interface.

        Returns:
            Self: The current instance, enabling method chaining.
        """
        if self.verbose:
            logger.info(
                f"{self.model_kind} reuse: features already available, "
                "skipping tsfresh extraction."
            )
        return self

    def explain(
        self,
        plot_local: bool = True,
        plot_global: bool = True,
        plot_profile: bool = True,
    ) -> DalexExplainerEnsemble:
        """Run DALEX explanations across the tree classifiers in the ensemble.

        Materialises output directories, rebuilds labels in prediction-reuse
        mode, then delegates the per-classifier × per-seed loop to a freshly
        constructed
        :class:`~eruption_forecast.ensemble.dalex_explainer_ensemble.DalexExplainerEnsemble`.
        The fitted worker is stored on ``self.DalexExplainerEnsemble``.

        Args:
            plot_local (bool): Save per-seed SHAP HTML plots. Defaults to
                ``True``.
            plot_global (bool): Save per-seed permutation-importance HTML
                plots. Defaults to ``True``.
            plot_profile (bool): Save per-feature PDP HTML plots. Defaults to
                ``True``.

        Returns:
            DalexExplainerEnsemble: The worker with all ``*_results`` dicts
                populated.

        Raises:
            ValueError: If ``self.y_true`` is empty after ``build_label()`` —
                same failure modes as ``EvaluationModel.evaluate()``.
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
                "Ground-truth labels (self.y_true) are empty after "
                "build_label(). For training reuse, the upstream TrainingModel "
                "did not run build_label/extract_features before being passed "
                "in. For prediction reuse, LabelBuilder produced no "
                "overlapping windows over the explanation period — check "
                "eruption_dates and the prediction window grid."
            )
        if self.features_df.empty:
            raise ValueError(
                "Features are empty. Ensure the source model ran "
                "extract_features() before being passed to ExplanationModel."
            )

        explainer_ensemble = DalexExplainerEnsemble(
            classifier_ensemble=self.ClassifierEnsemble,
            features_df=self.features_df,
            y_true=self.y_true,
            kind=self.model_kind,
            output_dir=self.explanation_dir,
            n_seeds_to_explain=self.n_seeds_to_explain,
            n_observations_to_explain=self.n_observations_to_explain,
            top_k_features=self.top_k_features,
            overwrite=self.overwrite,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        ).compute(
            plot_local=plot_local,
            plot_global=plot_global,
            plot_profile=plot_profile,
        )

        self.DalexExplainerEnsemble = explainer_ensemble
        return explainer_ensemble
