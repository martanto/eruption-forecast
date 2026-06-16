import os
from typing import Self, Literal

import joblib

from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.model.base_model import BaseModel
from eruption_forecast.model.cache_model import CacheModel
from eruption_forecast.model.training_model import TrainingModel
from eruption_forecast.model.prediction_model import PredictionModel
from eruption_forecast.ensemble.explainer_ensemble import (
    ExplainerEnsemble,
    ClassifierExplanation,
)
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


class ExplanationModel(BaseModel, CacheModel):
    def __init__(
        self,
        model: TrainingModel | PredictionModel,
        overwrite: bool = False,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        if model.ClassifierEnsemble is None:
            raise ValueError(
                f"ClassifierEnsemble is not found for {type(model).__name__}"
            )

        super().__init__(
            tremor_data=model._tremor_data,
            start_date=model.start_date,
            end_date=model.end_date,
            window_size=model.window_size,
            overwrite=overwrite,
            output_dir=output_dir,
            root_dir=root_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.kind: Literal["explanation"] = "explanation"
        self.model: TrainingModel | PredictionModel = model
        self.model_kind: Literal["training", "prediction"] = model.kind
        self.ClassifierEnsemble: ClassifierEnsemble = model.ClassifierEnsemble
        self.features_df = model.features_df
        self.basename = (
            f"{type(model).__name__}_{self.start_date_str}_{self.end_date_str}"
        )

        self.explanation_dir, self.classifiers_dir = self.set_directories()

        self.ExplainerEnsemble = ExplainerEnsemble(
            classifier_ensemble=self.ClassifierEnsemble,
            features_df=self.features_df,
            kind=self.model_kind,
            output_dir=self.classifiers_dir,
            overwrite=self.overwrite,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        self.explanations: list[ClassifierExplanation] = []
        self.validate()

        if verbose:
            logger.info(f"Explaining on {type(model).__name__}")

    @classmethod
    def build_cache_identity(  # ty:ignore[invalid-method-override]
        cls,
        *,
        upstream_hash: str,
        explain_params: dict,
    ) -> dict:
        """Return the canonical identity dict for this explanation run.

        The ``upstream_hash`` ties the cache to the exact upstream model:
        when the trained ``ClassifierEnsemble``, the feature matrix, or the
        modelling window change, the existing explanation cache is
        invalidated automatically. Station identity is implicit in the
        per-station ``output_dir`` so it is not part of the identity.

        Args:
            upstream_hash (str): Hash of the upstream model fingerprint
                (model kind, classifier names, features shape and columns,
                date range). Produced by :meth:`_upstream_hash`.
            explain_params (dict): Stage-level knobs that change the
                produced artefact — currently ``save_per_seed``.

        Returns:
            dict: Canonical identity dict ready for hashing.
        """
        return {
            "class": cls.__name__,
            "upstream_hash": upstream_hash,
            "explain_params": explain_params,
        }

    @classmethod
    def from_file(
        cls,
        filepath: str,
        overwrite: bool = False,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> "ExplanationModel":

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
            f"[{classifiers}] on {self.start_date_str} to {self.end_date_str}."
        )

    def to_dict(self) -> dict:
        """Return a structured dictionary of identifying configuration.

        Returns:
            dict: Mapping of stage identity, upstream mode, window range,
                classifier names, and output directory.
        """
        return {
            "kind": self.kind,
            "model_kind": self.model_kind,
            "start_date": self.start_date_str,
            "end_date": self.end_date_str,
            "window_size": self.window_size,
            "classifiers": list(self.ClassifierEnsemble.classifiers),
            "output_dir": self.output_dir,
        }

    def to_prompt(self) -> str:
        """Return a structured prompt block suitable for LLM and MCP prompts.

        Builds a deterministic, template-driven prose block from the
        instance's identity fields. Paths are reduced to filenames only to
        avoid leaking absolute system paths into prompts. For human-readable
        prose use :meth:`describe`. For raw data use :meth:`to_dict`.

        Returns:
            str: Prompt-ready block ending with the build state (number of
                explained classifiers and the basename).

        Example:
            >>> for part in explanation.to_prompt().split(". "):
            ...     print(part)
            Explanation period: 2025-03-16 to 2025-03-22
            Window size: 2 day(s)
            Upstream stage: prediction
            Classifiers: RandomForestClassifier, XGBClassifier
            Explained classifiers: 2
            Overwrite: False
            Output dir: VG.OJN.00.EHZ
            Root dir: None
            n_jobs: 4
            Verbose: False
            Basename: PredictionModel_2025-03-16_2025-03-22.
        """
        classifier_names = ", ".join(self.ClassifierEnsemble.classifiers)
        output_dir_str = (
            os.path.basename(self.output_dir.rstrip(os.sep))
            if self.output_dir
            else "not set"
        )

        return (
            f"Explanation period: {self.start_date_str} to {self.end_date_str}. "
            f"Window size: {self.window_size} day(s). "
            f"Upstream stage: {self.model_kind}. "
            f"Classifiers: {classifier_names}. "
            f"Explained classifiers: {len(self.explanations)}. "
            f"Overwrite: {self.overwrite}. "
            f"Output dir: {output_dir_str}. "
            f"Root dir: {self.root_dir}. "
            f"n_jobs: {self.n_jobs}. "
            f"Verbose: {self.verbose}. "
            f"Basename: {self.basename}."
        )

    def _upstream_hash(self) -> str:
        """Hash the upstream-model fingerprint so the explanation cache invalidates
        whenever the trained ensemble, the feature matrix, or the modelling
        window changes."""
        fingerprint = {
            "model_kind": self.model_kind,
            "classifiers": list(self.ClassifierEnsemble.classifiers),
            "features_shape": list(self.features_df.shape),
            "features_columns": sorted(map(str, self.features_df.columns)),
            "start_date": self.start_date_str,
            "end_date": self.end_date_str,
            "window_size": self.window_size,
        }
        return CacheModel.compute_hash(fingerprint)

    def explain(self, save_per_seed: bool = True, plot_per_seed: bool = True) -> Self:
        identity = type(self).build_cache_identity(
            upstream_hash=self._upstream_hash(),
            explain_params={"save_per_seed": save_per_seed},
        )

        if not self.overwrite:
            cached = type(self).load_from_cache(self.output_dir, identity)
            if cached is not None:
                self.explanations = cached.explanations
                self.ExplainerEnsemble.explanations = cached.explanations
                if plot_per_seed:
                    self.ExplainerEnsemble.plot_seed()
                return self

        self.create_directories()
        self.ExplainerEnsemble.explain(save_per_seed=save_per_seed)

        if plot_per_seed:
            self.ExplainerEnsemble.plot_seed()

        self.explanations = self.ExplainerEnsemble.explanations
        self.save_to_cache(identity)

        return self
