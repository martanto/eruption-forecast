import os
from typing import Self, Literal

from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import ensure_dir, load_pickle
from eruption_forecast.model.base_model import BaseModel
from eruption_forecast.model.cache_model import CacheModel
from eruption_forecast.model.training_model import TrainingModel
from eruption_forecast.model.prediction_model import PredictionModel
from eruption_forecast.config.explanation_config import ExplanationConfig
from eruption_forecast.ensemble.explainer_ensemble import ExplainerEnsemble
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble
from eruption_forecast.dataclass.classifier_explanation import ClassifierExplanation


class ExplanationModel(BaseModel, CacheModel):
    """Per-classifier SHAP explanation stage built on a fitted upstream model.

    Reuses ``tremor_data``, ``start_date``, ``end_date``, ``window_size``,
    the loaded ``ClassifierEnsemble``, and the extracted ``features_df``
    from the supplied ``TrainingModel`` or ``PredictionModel``. No tsfresh
    re-run or model re-fit is performed.

    Output is namespaced by upstream-stage marker so that training and
    prediction explanations never share a classifiers directory:

    - ``{output_dir}/explanation/training/`` for training-reuse mode
    - ``{output_dir}/explanation/prediction/`` for prediction-reuse mode

    Attributes:
        kind (Literal["explanation"]): Stage identity tag.
        model (TrainingModel | PredictionModel): Upstream model being explained.
        model_kind (Literal["training", "prediction"]): Upstream stage marker.
        ClassifierEnsemble (ClassifierEnsemble): Fitted ensemble pulled
            from ``model``.
        features_df (pd.DataFrame): Feature matrix pulled from ``model``.
        eruption_dates (list[str] | None): Ground-truth eruption dates.
        explanation_dir (str): Root explanation directory.
        classifiers_dir (str): Per-classifier output directory.
        ExplainerEnsemble (ExplainerEnsemble): SHAP worker instance.
        explanations (list[ClassifierExplanation]): Populated by
            :meth:`explain`.
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
    ):
        """Initialize ExplanationModel in training-reuse or prediction-reuse mode.

        Args:
            model (TrainingModel | PredictionModel): Live fitted model
                source. Must carry a non-``None`` ``ClassifierEnsemble``.
            eruption_dates (list[str] | None): Ground-truth eruption dates
                in ``"YYYY-MM-DD"`` format. Optional for training reuse
                (the dates are read from ``model.eruption_dates``); required
                for prediction reuse because ``PredictionModel`` does not
                carry them. Defaults to ``None``.
            overwrite (bool): Overwrite cached outputs. Defaults to
                ``False``.
            output_dir (str | None): Root output directory. Derived from
                ``model`` when ``None``. Defaults to ``None``.
            root_dir (str | None): Project root used to anchor relative
                output paths. Defaults to ``None``.
            n_jobs (int): Parallel workers. Defaults to ``1``.
            verbose (bool): Verbose logging. Defaults to ``False``.

        Raises:
            ValueError: If ``model.ClassifierEnsemble`` is ``None``.
            ValueError: If ``eruption_dates`` is ``None`` when ``model`` is
                a ``PredictionModel``.
        """
        if model.ClassifierEnsemble is None:
            raise ValueError(
                f"ClassifierEnsemble is not found for {type(model).__name__}"
            )

        if model.kind == "prediction" and eruption_dates is None:
            raise ValueError(
                "Parameter `eruption_dates` cannot be None. Please set eruption_dates=['2025-01-01', ...]"
            )

        # In prediction reuse the upstream ``PredictionModel`` does not
        # carry eruption_dates, so the caller-supplied value is the only
        # source. In training reuse we prefer the upstream model's own
        # eruption_dates so explain() does not require the caller to pass
        # them a second time.
        resolved_eruption_dates = (
            eruption_dates if model.kind == "prediction" else model.eruption_dates
        )

        super().__init__(
            tremor_data=model._tremor_data,
            start_date=model.start_date,
            end_date=model.end_date,
            window_size=model.window_size,
            eruption_dates=resolved_eruption_dates,
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
            explanation_dir=self.explanation_dir,
            overwrite=self.overwrite,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        self.explanations: list[ClassifierExplanation] = []
        self.validate()

        # Save model config
        self._config: ExplanationConfig = ExplanationConfig(
            eruption_dates=list(resolved_eruption_dates)
            if resolved_eruption_dates
            else [],
            overwrite=overwrite,
            output_dir=output_dir,
            root_dir=root_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )

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
        eruption_dates: list[str] | None = None,
        overwrite: bool = False,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> "ExplanationModel":
        """Construct an ExplanationModel from a saved upstream ``.pkl``.

        Loads the pickle, validates it is a ``TrainingModel`` or
        ``PredictionModel``, and forwards every argument to
        :meth:`__init__`. Use this when the upstream stage was persisted
        to disk via :meth:`BaseModel.save`.

        Args:
            filepath (str): Path to a ``.pkl`` file produced by
                ``TrainingModel.save()`` or ``PredictionModel.save()``.
            eruption_dates (list[str] | None): Ground-truth eruption dates
                in ``"YYYY-MM-DD"`` format. Required when the loaded model
                is a ``PredictionModel``. Defaults to ``None``.
            overwrite (bool): Overwrite cached outputs. Defaults to
                ``False``.
            output_dir (str | None): Root output directory. Defaults to
                ``None``.
            root_dir (str | None): Project root used to anchor relative
                output paths. Defaults to ``None``.
            n_jobs (int): Parallel workers. Defaults to ``1``.
            verbose (bool): Verbose logging. Defaults to ``False``.

        Returns:
            ExplanationModel: Ready-to-explain instance.

        Raises:
            FileNotFoundError: If ``filepath`` does not exist.
            TypeError: If the loaded pickle is not a ``TrainingModel`` or
                ``PredictionModel``.
        """
        model = load_pickle(filepath)

        if not isinstance(model, (TrainingModel, PredictionModel)):
            raise TypeError(
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
            "eruption_dates": list(self.eruption_dates) if self.eruption_dates else [],
            "overwrite": self.overwrite,
            "output_dir": self.output_dir,
            "root_dir": self.root_dir,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
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

    def save_config(
        self,
        path: str | None = None,
        fmt: Literal["yaml", "json"] = "yaml",
    ) -> str:
        """Persist the captured ``ExplanationModel`` init configuration to disk.

        Writes the parameter snapshot captured during ``__init__`` so a
        standalone explanation run can save its constructor surface without
        going through :class:`~eruption_forecast.model.forecast_model.ForecastModel`.

        Args:
            path (str | None): Destination file path. ``None`` resolves to
                ``{explanation_dir}/explanation.config.{fmt}`` — already
                namespaced by upstream stage (``explanation/training/`` or
                ``explanation/prediction/``) so the two reuse modes never
                collide. Defaults to ``None``.
            fmt (Literal["yaml", "json"]): Output format. Defaults to
                ``"yaml"``.

        Returns:
            str: The absolute path the configuration was written to.

        Example:
            >>> path = explanation.save_config()
            >>> path  # doctest: +SKIP
            'output/VG.OJN.00.EHZ/explanation/prediction/explanation.config.yaml'
        """
        if path is None:
            path = os.path.join(self.explanation_dir, f"explanation.config.{fmt}")
        return self._config.save(path, fmt)

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

    def explain(
        self,
        save_per_seed: bool = True,
        check_additivity: bool = False,
        overwrite_classifier_explanation: bool = False,
    ) -> Self:
        """Compute per-classifier SHAP explanations for every seed.

        Delegates to :meth:`ExplainerEnsemble.explain` and caches the
        result via :class:`CacheModel`. On a cache hit the stored
        ``explanations`` list is restored without re-running SHAP.

        Args:
            save_per_seed (bool): Persist each per-seed
                ``shap.Explanation`` to disk so a subsequent run can
                short-circuit recomputation. Defaults to ``True``.
            check_additivity (bool): Forwarded to ``shap.TreeExplainer``
                to verify SHAP additivity against the model output.
                Defaults to ``False``.
            overwrite_classifier_explanation (bool): Overwrite the cached
                per-classifier ``ClassifierExplanation.pkl`` artefact.
                Falls back to ``self.overwrite`` when ``False``. Defaults
                to ``False``.

        Returns:
            Self: The current instance, enabling method chaining.
        """
        identity = type(self).build_cache_identity(
            upstream_hash=self._upstream_hash(),
            explain_params={"save_per_seed": save_per_seed},
        )

        if not self.overwrite:
            cached = type(self).load_from_cache(self.output_dir, identity)
            if cached is not None:
                self.explanations = cached.explanations
                self.ExplainerEnsemble.explanations = cached.explanations
                return self

        self.create_directories()
        self.ExplainerEnsemble.explain(
            save_per_seed=save_per_seed,
            check_additivity=check_additivity,
            overwrite_classifier_explanation=overwrite_classifier_explanation
            or self.overwrite,
        )

        self.explanations = self.ExplainerEnsemble.explanations
        self.save_to_cache(identity)

        self.save()

        try:
            self.save_config()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to save explanation config: {exc}")

        return self

    def plot(
        self,
        figsize: tuple[float, float] | None = None,
        max_display: int = 20,
        group_remaining_features: bool = False,
        dpi: int = 150,
        plot_per_seed: bool = True,
        plot_aggregate: bool = True,
    ):
        """Render every SHAP plot the stage produces.

        Renders per-eruption waterfall plots (when ``eruption_dates`` is
        available) and, optionally, per-seed bar and beeswarm plots plus
        per-classifier aggregate bar and beeswarm plots.

        Args:
            figsize (tuple[float, float] | None): Figure size in inches
                for SHAP plots. ``None`` auto-sizes from ``max_display``.
                Defaults to ``None``.
            max_display (int): Maximum number of features to display.
                Defaults to ``20``.
            group_remaining_features (bool): Forwarded to
                ``shap.plots.beeswarm`` to group features beyond
                ``max_display``. Defaults to ``False``.
            dpi (int): Figure resolution in dots per inch. Defaults to
                ``150``.
            plot_per_seed (bool): Render per-seed bar and beeswarm plots
                in parallel under ``classifiers/{name}/figures/``. Defaults
                to ``True``.
            plot_aggregate (bool): Render per-classifier aggregate bar
                and beeswarm plots under
                ``classifiers/{name}/figures/aggregate/``. Stacks every
                seed into the union feature space (NaN-padded) so a single
                figure summarises the whole ensemble. Defaults to ``True``.

        Raises:
            ValueError: If :meth:`explain` has not yet been called.
        """
        if len(self.explanations) == 0:
            raise ValueError("No explanations found. Please run explain() first.")

        explainer_ensemble = self.ExplainerEnsemble

        if self.eruption_dates is None:
            logger.warning(
                "Eruption dates not found. Please add eruption_dates when init the model."
            )
        else:
            explainer_ensemble.plot_waterfall(
                labels=self.model.labels,
                eruption_dates=self.eruption_dates,
                figsize=figsize,
                max_display=max_display,
                dpi=dpi,
            )

        # Plot SHAP beeswarm and bar per seed
        if plot_per_seed:
            explainer_ensemble.plot_seed(
                max_display=max_display,
                group_remaining_features=group_remaining_features,
                dpi=dpi,
            )

        # Aggregate plots collapse all seeds into a single per-classifier view
        # over the NaN-padded union feature space.
        if plot_aggregate:
            explainer_ensemble.plot_aggregate(
                max_display=max_display,
                top_n=max_display,
                figsize=figsize,
                dpi=dpi,
                group_remaining_features=group_remaining_features,
            )
