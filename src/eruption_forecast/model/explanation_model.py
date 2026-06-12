"""SHAP-based explanation stage for the eruption-forecast pipeline.

Mirrors :class:`~eruption_forecast.model.evaluation_model.EvaluationModel`'s
upstream-model-as-input shape, but inherits
:class:`~eruption_forecast.model.cache_model.CacheModel` because SHAP runs are
expensive (per-classifier × per-seed × per-observation) and benefit from a
content-addressable cache keyed on the upstream training hash.
"""

import os
from typing import Self, Literal

import shap  # noqa: F401  # imported for explicit dependency declaration
import numpy as np
import joblib
import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.model.base_model import BaseModel
from eruption_forecast.model.cache_model import CacheModel
from eruption_forecast.model.training_model import TrainingModel
from eruption_forecast.model.prediction_model import PredictionModel
from eruption_forecast.ensemble.explainer_ensemble import (
    ExplainerEnsemble,
    _is_tree_classifier,
)
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


class ExplanationModel(BaseModel, CacheModel):
    """Explain a fitted ``ClassifierEnsemble`` via SHAP TreeExplainer.

    Reuses the extracted features and in-memory seed models produced upstream.
    No tsfresh re-run or model re-fit is ever performed — the worker simply
    feeds features into ``shap.TreeExplainer`` for every tree-based seed.

    Two operating modes:

    - **Training reuse** — pass a completed ``TrainingModel`` as ``model``.
      Features, the loaded ``ClassifierEnsemble``, and the window grid are
      taken directly from the training stage.
    - **Prediction reuse** — pass a completed ``PredictionModel`` as
      ``model``.  Features, the loaded ``ClassifierEnsemble``, and the window
      grid are taken from the prediction stage.

    Output artefacts are written to ``{output_dir}/explanation/{kind}/`` so
    that training-window and prediction-window explanations coexist in the
    same ``output_dir`` without collisions.

    Example:
        >>> em = ExplanationModel(model=prediction_model)
        >>> em.explain()
    """

    def __init__(
        self,
        model: TrainingModel | PredictionModel,
        n_observations_to_explain: int = 10,
        method: Literal["shap"] = "shap",
        feature_perturbation: Literal[
            "tree_path_dependent", "interventional"
        ] = "tree_path_dependent",
        model_output: Literal["raw", "probability", "log_loss"] = "raw",
        background_size: int = 100,
        check_additivity: bool = True,
        selection: Literal["top_proba", "near_threshold"] = "top_proba",
        overwrite: bool = False,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> None:
        """Initialise ExplanationModel in training-reuse or prediction-reuse mode.

        Reuses ``tremor_data``, ``start_date``, ``end_date``, ``window_size``,
        the loaded ``ClassifierEnsemble``, and the extracted ``features_df``
        from ``model``.  ``output_dir`` falls back to ``model.output_dir`` when
        not explicitly provided.

        Args:
            model (TrainingModel | PredictionModel): Live trained model
                source.
            n_observations_to_explain (int, optional): Top-N observations
                to explain per classifier. Defaults to ``10``.
            method (Literal["shap"], optional): Explanation method.
                Reserved for future additions. Defaults to ``"shap"``.
            feature_perturbation (Literal["tree_path_dependent",
                "interventional"], optional): SHAP perturbation mode.
                Defaults to ``"tree_path_dependent"``.
            model_output (Literal["raw", "probability", "log_loss"],
                optional): SHAP output unit. Defaults to ``"raw"``.
            background_size (int, optional): Background sample size for
                interventional mode. Defaults to ``100``.
            check_additivity (bool, optional): Forwarded to the inner
                ``explainer(X, ...)`` call. Defaults to ``True``.
            selection (Literal["top_proba", "near_threshold"], optional):
                Observation-ranking strategy. ``"top_proba"`` picks the
                highest predicted positive-class probability (alarm
                cases); ``"near_threshold"`` picks the windows closest
                to ``0.5`` (borderline cases). Defaults to ``"top_proba"``.
            overwrite (bool, optional): Overwrite cached artefacts.
                Defaults to ``False``.
            output_dir (str | None, optional): Root output directory.
                Derived from ``model`` when ``None``. Defaults to
                ``None``.
            root_dir (str | None, optional): Project root used for path
                resolution. Defaults to ``None``.
            n_jobs (int, optional): Parallel workers. Defaults to ``1``.
            verbose (bool, optional): Emit detailed progress logs.
                Defaults to ``False``.

        Raises:
            ValueError: If ``model.ClassifierEnsemble`` is ``None``.
            ValueError: If ``model_output`` is ``"probability"`` or
                ``"log_loss"`` while ``feature_perturbation`` is
                ``"tree_path_dependent"`` (validated by
                :class:`ExplainerEnsemble`).
        """
        if model.ClassifierEnsemble is None:
            raise ValueError(
                f"ClassifierEnsemble is not found for {type(model).__name__}"
            )

        super().__init__(
            tremor_data=model._tremor_data,
            start_date=model.start_date,
            end_date=model.end_date,
            window_size=model.window_size,
            eruption_dates=None,
            overwrite=overwrite,
            output_dir=output_dir if output_dir is not None else model.output_dir,
            root_dir=root_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.model: TrainingModel | PredictionModel = model
        self.model_kind: Literal["training", "prediction"] = model.kind
        self.kind: Literal["explanation"] = "explanation"
        self.ClassifierEnsemble: ClassifierEnsemble = model.ClassifierEnsemble
        self.features_df = model.features_df
        self.basename = f"{self.start_date_str}_{self.end_date_str}"

        self.n_observations_to_explain = n_observations_to_explain
        self.method: Literal["shap"] = method
        self.feature_perturbation: Literal["tree_path_dependent", "interventional"] = (
            feature_perturbation
        )
        self.model_output: Literal["raw", "probability", "log_loss"] = model_output
        self.background_size = background_size
        self.check_additivity = check_additivity
        self.selection: Literal["top_proba", "near_threshold"] = selection

        self.explanation_dir, self.classifiers_dir = self.set_directories()

        #  Populated by :meth:`explain`.
        self.ExplainerEnsemble: ExplainerEnsemble | None = None
        self.results: pd.DataFrame = pd.DataFrame()

        self.validate()

    @classmethod
    def from_file(
        cls,
        filepath: str,
        n_observations_to_explain: int = 10,
        method: Literal["shap"] = "shap",
        feature_perturbation: Literal[
            "tree_path_dependent", "interventional"
        ] = "tree_path_dependent",
        model_output: Literal["raw", "probability", "log_loss"] = "raw",
        background_size: int = 100,
        check_additivity: bool = True,
        selection: Literal["top_proba", "near_threshold"] = "top_proba",
        overwrite: bool = False,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> "ExplanationModel":
        """Construct an ``ExplanationModel`` from a saved upstream ``.pkl``.

        Loads the pickle, validates that it is a ``TrainingModel`` or
        ``PredictionModel``, and forwards every remaining argument to the
        constructor.

        Args:
            filepath (str): Path to a ``.pkl`` produced by
                :meth:`TrainingModel.save` or :meth:`PredictionModel.save`.
            n_observations_to_explain (int, optional): Forwarded to
                :meth:`__init__`. Defaults to ``10``.
            method (Literal["shap"], optional): Forwarded to
                :meth:`__init__`. Defaults to ``"shap"``.
            feature_perturbation (Literal["tree_path_dependent",
                "interventional"], optional): Forwarded to
                :meth:`__init__`. Defaults to ``"tree_path_dependent"``.
            model_output (Literal["raw", "probability", "log_loss"],
                optional): Forwarded to :meth:`__init__`. Defaults to
                ``"raw"``.
            background_size (int, optional): Forwarded to
                :meth:`__init__`. Defaults to ``100``.
            check_additivity (bool, optional): Forwarded to
                :meth:`__init__`. Defaults to ``True``.
            selection (Literal["top_proba", "near_threshold"], optional):
                Forwarded to :meth:`__init__`. Defaults to
                ``"top_proba"``.
            overwrite (bool, optional): Forwarded to :meth:`__init__`.
                Defaults to ``False``.
            output_dir (str | None, optional): Forwarded to
                :meth:`__init__`. Defaults to ``None``.
            root_dir (str | None, optional): Forwarded to
                :meth:`__init__`. Defaults to ``None``.
            n_jobs (int, optional): Forwarded to :meth:`__init__`.
                Defaults to ``1``.
            verbose (bool, optional): Forwarded to :meth:`__init__`.
                Defaults to ``False``.

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
            n_observations_to_explain=n_observations_to_explain,
            method=method,
            feature_perturbation=feature_perturbation,
            model_output=model_output,
            background_size=background_size,
            check_additivity=check_additivity,
            selection=selection,
            overwrite=overwrite,
            output_dir=output_dir,
            root_dir=root_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def set_directories(self) -> tuple[str, str]:
        """Build the output directory paths for explanation artefacts.

        Paths are namespaced by upstream-stage marker —
        ``explanation/training/`` or ``explanation/prediction/`` — so that
        results from the two modes never share a classifiers directory.

        Returns:
            tuple[str, str]: A 2-tuple of
                ``(explanation_dir, classifiers_dir)``.
        """
        explanation_dir = os.path.join(self.output_dir, "explanation", self.model_kind)
        classifiers_dir = os.path.join(explanation_dir, "classifiers")
        return explanation_dir, classifiers_dir

    def create_directories(self) -> None:
        """Create the explanation output directories on disk.

        Idempotent: existing directories are left untouched.
        """
        ensure_dir(self.explanation_dir)
        ensure_dir(self.classifiers_dir)

    def validate(self) -> Self:
        """Run model-specific validation (currently a no-op).

        All validation needed by ``ExplanationModel`` is performed inline
        in :meth:`__init__` against the upstream ``model`` and forwarded
        to :class:`ExplainerEnsemble`.

        Returns:
            Self: The current instance, enabling method chaining.
        """
        return self

    def describe(self) -> str:
        """Return a one-line human-readable description of this instance.

        Returns:
            str: Prose description naming the upstream-stage mode, the
                configured classifiers, the selection strategy, and the
                evaluation date range.
        """
        classifiers = ", ".join(self.ClassifierEnsemble.classifiers)
        return (
            f"ExplanationModel ({self.model_kind} mode) explaining "
            f"[{classifiers}] with method={self.method!r}, "
            f"selection={self.selection!r}, n_obs={self.n_observations_to_explain} "
            f"on {self.start_date_str} to {self.end_date_str}."
        )

    def to_dict(self) -> dict:
        """Return a structured dictionary of all configuration fields.

        Returns:
            dict: Mapping of field names to their current values.
        """
        data: dict = {
            "kind": self.kind,
            "model_kind": self.model_kind,
            "start_date": self.start_date_str,
            "end_date": self.end_date_str,
            "window_size": self.window_size,
            "classifiers": self.ClassifierEnsemble.classifiers,
            "method": self.method,
            "feature_perturbation": self.feature_perturbation,
            "model_output": self.model_output,
            "background_size": self.background_size,
            "check_additivity": self.check_additivity,
            "selection": self.selection,
            "n_observations_to_explain": self.n_observations_to_explain,
            "output_dir": self.output_dir,
        }
        if not self.results.empty:
            data["explained_observation_ids"] = {
                col: self.results[col].dropna().astype(int).tolist()
                for col in self.results.columns
            }
        return data

    def to_prompt(self) -> str:
        """Return a structured text block suitable for LLM prompt input.

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
        """No-op label-build hook.

        ``ExplanationModel`` does not need labels — SHAP explains
        predictions, not ground-truth correctness. The hook is
        implemented to satisfy the :class:`BaseModel` interface and to
        let the stage be dropped into method chains that call
        ``build_label()``.

        Args:
            window_step (int): Step size between consecutive windows.
                Accepted but ignored.
            window_step_unit (Literal["minutes", "hours"]): Unit of
                ``window_step``. Accepted but ignored.

        Returns:
            Self: The current instance, enabling method chaining.
        """
        return self

    def extract_features(self) -> Self:
        """Return immediately — features are reused from the upstream stage.

        Returns:
            Self: The current instance, enabling method chaining.
        """
        if self.verbose:
            logger.info(
                f"{self.model_kind} reuse: features already available, "
                "skipping tsfresh extraction."
            )
        return self

    @classmethod
    def build_cache_identity(  # ty:ignore[invalid-method-override]
        cls,
        *,
        nslc: str,
        upstream_hash: str,
        explain_params: dict,
    ) -> dict:
        """Return the canonical identity dict for this explanation run.

        The ``upstream_hash`` ties the cache to the exact trained
        ensemble — a re-trained ``TrainingModel`` or freshly-forecast
        ``PredictionModel`` invalidates the explanation cache. Same
        pattern as :meth:`PredictionModel.build_cache_identity` threading
        a ``training_hash``.

        Args:
            nslc (str): Station identifier (NSLC string).
            upstream_hash (str): Hash of the upstream model used as the
                source of features and seed models.
            explain_params (dict): Explanation-stage knobs (``method``,
                ``feature_perturbation``, ``model_output``,
                ``background_size``, ``check_additivity``, ``selection``,
                ``n_observations_to_explain``).

        Returns:
            dict: Canonical identity dict ready for hashing.
        """
        return {
            "nslc": nslc,
            "upstream_hash": upstream_hash,
            "explain_params": explain_params,
        }

    def explain(
        self,
        plot_aggregate: bool = True,
        plot_per_seed: bool = False,
        plot_waterfall: bool = True,
        overwrite: bool | None = None,
    ) -> Self:
        """Pick top-N observations per classifier and dispatch SHAP computation.

        For every tree-based classifier in the ensemble, ranks rows of
        ``self.features_df`` per the configured ``selection`` strategy and
        forwards the top-N window ids to :class:`ExplainerEnsemble`. Builds
        the worker, computes the per-seed SHAP values, optionally renders
        plots, and writes a top-level summary CSV of which observations
        were explained per classifier.

        Args:
            plot_aggregate (bool, optional): Render aggregate plots via
                :meth:`ExplainerEnsemble.plot_aggregate`. Defaults to
                ``True``.
            plot_per_seed (bool, optional): Render per-seed bar /
                beeswarm plots via
                :meth:`ExplainerEnsemble.plot_seed`. Expensive across many
                seeds. Defaults to ``False``.
            plot_waterfall (bool, optional): When ``plot_per_seed`` is
                ``True``, also render per-(seed, observation) waterfall
                plots. Defaults to ``True``.
            overwrite (bool | None, optional): Overwrite existing plot
                files. ``None`` inherits from ``self.overwrite``.
                Defaults to ``None``.

        Returns:
            Self: The current instance, enabling method chaining.
        """
        self.create_directories()

        #  ClassifierEnsemble.predict_per_classifier returns
        #  `dict[str, dict[str, np.ndarray]]` keyed by classifier name —
        #  inner keys are "probability", "uncertainty", "prediction",
        #  "confidence". Each ndarray indexes positionally with
        #  ``self.features_df.index``.
        proba_per_clf = self.ClassifierEnsemble.predict_per_classifier(self.features_df)
        observation_ids: dict[str, list[int]] = {}
        index = self.features_df.index

        for name, arrays in proba_per_clf.items():
            seed_ensemble = self.ClassifierEnsemble.ensembles[name]
            if not _is_tree_classifier(seed_ensemble):
                if self.verbose:
                    logger.info(
                        f"ExplanationModel: skipping non-tree classifier {name}"
                    )
                continue

            probs = arrays["probability"]
            if self.selection == "top_proba":
                order = np.argsort(-probs)
            else:
                order = np.argsort(np.abs(probs - 0.5))

            top = order[: self.n_observations_to_explain]
            observation_ids[name] = index[top].tolist()

        resolved_overwrite = overwrite if overwrite is not None else self.overwrite

        self.ExplainerEnsemble = ExplainerEnsemble(
            classifier_ensemble=self.ClassifierEnsemble,
            features_df=self.features_df,
            observation_ids=observation_ids,
            method=self.method,
            feature_perturbation=self.feature_perturbation,
            model_output=self.model_output,
            background_size=self.background_size,
            check_additivity=self.check_additivity,
            kind=self.model_kind,
            output_dir=self.explanation_dir,
            overwrite=resolved_overwrite,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        ).compute()

        if self.ExplainerEnsemble:
            if plot_aggregate:
                self.ExplainerEnsemble.plot_aggregate()
            if plot_per_seed:
                self.ExplainerEnsemble.plot_seed(plot_waterfall=plot_waterfall)

        #  Top-level summary of which observations were explained per
        #  classifier — DataFrame.from_dict with orient="index" + .T
        #  produces one column per classifier where the rows are the
        #  explained window ids in selection order.
        if observation_ids:
            self.results = pd.DataFrame.from_dict(observation_ids, orient="index").T
            results_path = os.path.join(
                self.explanation_dir,
                f"result_all_model_explanations_{self.basename}.csv",
            )
            self.results.to_csv(results_path, index=False)

        return self
