"""Per-classifier × per-seed SHAP explanation worker.

Mirrors the role of
:class:`~eruption_forecast.ensemble.metrics_ensemble.MetricsEnsemble`: takes a
fitted :class:`~eruption_forecast.ensemble.classifier_ensemble.ClassifierEnsemble`
plus the feature matrix and the observations selected upstream, then runs
SHAP TreeExplainer over every (tree-based classifier, seed) pair, persists the
per-classifier explanation dicts to disk, and renders bar / beeswarm / waterfall
plots.

Only tree-based classifiers (``RandomForestClassifier``, ``XGBClassifier``,
``GradientBoostingClassifier`` and any subclass thereof — covers ``lite-rf``
transparently) are explained. Non-tree classifiers in the ensemble emit one
INFO log line on dispatch and are silently skipped.
"""

import os
from typing import Self, Literal
from os.path import dirname

import shap
import numpy as np
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import ensure_dir, resolve_output_dir
from eruption_forecast.ensemble.seed_ensemble import SeedEnsemble
from eruption_forecast.plots.explanation_plots import (
    PER_SEED_PLOT_DISPATCHER,
    AGGREGATE_PLOT_DISPATCHER,
    AGGREGATE_PLOT_INPUT_KIND,
    render_one_seed_plot,
    render_one_aggregate_plot,
    render_one_waterfall_plot,
)
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


#  isinstance tuple (not a set of class names) — keeps lite-rf transparent and
#  survives sklearn-API subclasses like BalancedRandomForestClassifier.
TREE_CLASSIFIERS: tuple[type, ...] = (
    RandomForestClassifier,
    XGBClassifier,
    GradientBoostingClassifier,
)


def _is_tree_classifier(seed_ensemble: SeedEnsemble) -> bool:
    """Return ``True`` when the first seed model is a tree-based classifier.

    Args:
        seed_ensemble (SeedEnsemble): Per-classifier ensemble whose first
            seed instance is sampled.

    Returns:
        bool: ``True`` when ``seed_ensemble.seeds[0]["model"]`` is an
            instance of any of :data:`TREE_CLASSIFIERS`.
    """
    if not seed_ensemble.seeds:
        return False
    return isinstance(seed_ensemble.seeds[0]["model"], TREE_CLASSIFIERS)


def _normalise_shap_values(
    explanation: shap.Explanation,
) -> tuple[np.ndarray, float]:
    """Reduce a binary-classifier ``Explanation`` to a 2-D positive-class slice.

    Modern SHAP returns different ranks depending on backend:

    * sklearn ``RandomForestClassifier`` / ``GradientBoostingClassifier``
      → ``values`` shape ``(n_obs, n_features, 2)`` with
      ``base_values`` shape ``(n_obs, 2)``.
    * ``XGBClassifier`` (binary, single-margin output)
      → ``values`` shape ``(n_obs, n_features)`` with
      ``base_values`` scalar or ``(n_obs,)``.

    Always reduce to the positive-class slice so downstream storage and plots
    operate on a uniform 2-D matrix regardless of backend.

    Args:
        explanation (shap.Explanation): Raw SHAP explanation produced by
            ``TreeExplainer(X)``.

    Returns:
        tuple[np.ndarray, float]: A 2-tuple of ``(shap_values, base_value)``
            where ``shap_values`` has shape ``(n_obs, n_features)`` and
            ``base_value`` is the positive-class baseline collapsed to a
            scalar via mean over observations.
    """
    vals = np.asarray(explanation.values)
    shap_values = vals[..., 1] if vals.ndim == 3 else vals

    base = np.asarray(explanation.base_values)
    if base.ndim >= 2:
        base_value = float(base[..., 1].mean())
    elif base.ndim == 1:
        base_value = float(base.mean())
    else:
        base_value = float(base)

    return shap_values, base_value


def _explain_seed(
    seed: dict,
    X: pd.DataFrame,
    X_full: pd.DataFrame,
    observation_ids: list[int],
    feature_perturbation: Literal["tree_path_dependent", "interventional"],
    model_output: Literal["raw", "probability", "log_loss"],
    background_size: int,
    check_additivity: bool,
) -> dict:
    """Run SHAP TreeExplainer on one seed model and return the persisted dict.

    Module-level so ``joblib`` workers under the ``loky`` backend can pickle
    it. Slices ``X`` and ``X_full`` to the seed's feature subset, builds the
    explainer with the configured perturbation / output mode, applies the
    binary-class shape normalisation in :func:`_normalise_shap_values`, and
    caches the predicted positive-class probability alongside the SHAP
    values so callers can rank observations without re-running ``predict``.

    Args:
        seed (dict): Per-seed record with keys ``"random_state"``,
            ``"model"``, ``"feature_names"`` (as written by
            :meth:`SeedEnsemble.from_any`).
        X (pd.DataFrame): Full feature matrix (all rows). Only rows in
            ``observation_ids`` are explained, but the full matrix is the
            source of background samples when
            ``feature_perturbation="interventional"``.
        X_full (pd.DataFrame): Background-distribution pool. Currently the
            same as ``X``; kept as a separate parameter so a future
            refactor can decouple the explanation grid from the
            background pool.
        observation_ids (list[int]): Index values into ``X`` of the rows
            to explain.
        feature_perturbation (Literal["tree_path_dependent",
            "interventional"]): SHAP perturbation mode.
        model_output (Literal["raw", "probability", "log_loss"]): SHAP
            output unit.
        background_size (int): Number of background samples for the
            interventional path.
        check_additivity (bool): Forwarded as the
            ``explainer(X, check_additivity=...)`` kwarg.

    Returns:
        dict: ``{"shap_values", "base_value", "feature_values",
            "feature_names", "predicted_proba"}`` — all positive-class.
    """
    feature_names = list(seed["feature_names"])
    model = seed["model"]
    random_state = int(seed["random_state"])

    X_seed = X.loc[observation_ids, feature_names]
    feature_values = X_seed.to_numpy()

    background = None
    if feature_perturbation == "interventional":
        n_bg = min(background_size, len(X_full))
        background = X_full[feature_names].sample(n=n_bg, random_state=random_state)

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation=feature_perturbation,
        model_output=model_output,
    )
    explanation = explainer(X_seed, check_additivity=check_additivity)
    shap_values, base_value = _normalise_shap_values(explanation)

    proba_full = model.predict_proba(X_seed)
    predicted_proba = (
        proba_full[:, 1]
        if proba_full.ndim == 2 and proba_full.shape[1] >= 2
        else np.asarray(proba_full).ravel()
    )

    return {
        "shap_values": shap_values,
        "base_value": base_value,
        "feature_values": feature_values,
        "feature_names": feature_names,
        "predicted_proba": np.asarray(predicted_proba),
    }


def _explain_seed_ensemble(
    seed_ensemble: SeedEnsemble,
    X: pd.DataFrame,
    observation_ids: list[int],
    feature_perturbation: Literal["tree_path_dependent", "interventional"],
    model_output: Literal["raw", "probability", "log_loss"],
    background_size: int,
    check_additivity: bool,
) -> dict[int, dict]:
    """Run :func:`_explain_seed` over every seed in one ``SeedEnsemble``.

    Args:
        seed_ensemble (SeedEnsemble): Tree-based per-classifier ensemble.
        X (pd.DataFrame): Full feature matrix (all rows). Used both as the
            row source for the explained observations and as the
            background-distribution pool when interventional.
        observation_ids (list[int]): Index values into ``X`` of the rows
            to explain for this classifier.
        feature_perturbation (Literal["tree_path_dependent",
            "interventional"]): SHAP perturbation mode.
        model_output (Literal["raw", "probability", "log_loss"]): SHAP
            output unit.
        background_size (int): Background sample size for interventional
            mode.
        check_additivity (bool): Forwarded to the inner explainer call.

    Returns:
        dict[int, dict]: Per-seed explanation dicts keyed by
            ``random_state``.
    """
    per_seed: dict[int, dict] = {}
    for seed in seed_ensemble.seeds:
        result = _explain_seed(
            seed=seed,
            X=X,
            X_full=X,
            observation_ids=observation_ids,
            feature_perturbation=feature_perturbation,
            model_output=model_output,
            background_size=background_size,
            check_additivity=check_additivity,
        )
        per_seed[int(seed["random_state"])] = result
    return per_seed


class ExplainerEnsemble:
    """Per-seed SHAP explanation engine over a fitted ``ClassifierEnsemble``.

    Wraps a trained ``ClassifierEnsemble`` together with the feature matrix
    and the per-classifier list of observations to explain. :meth:`compute`
    runs SHAP TreeExplainer over every (tree-based classifier, seed) pair in
    parallel and persists one joblib dict per classifier under
    ``{output_dir}/classifiers/{classifier_name}/shap/explanations.pkl``.

    Idempotent: once :attr:`results` is populated, subsequent ``compute()``
    calls return immediately. Persistence of the full instance is available
    via :meth:`save` / :meth:`load`.

    Attributes:
        ClassifierEnsemble (ClassifierEnsemble): Fitted ensemble.
        features_df (pd.DataFrame): Feature matrix.
        observation_ids (dict[str, list[int]]): Per-classifier list of
            window ids to explain.
        method (Literal["shap"]): Explanation method. Reserved for future
            additions.
        feature_perturbation (Literal["tree_path_dependent",
            "interventional"]): SHAP perturbation mode.
        model_output (Literal["raw", "probability", "log_loss"]): SHAP
            output unit.
        background_size (int): Background sample size for interventional
            mode.
        check_additivity (bool): Forwarded to the inner explainer call.
        kind (Literal["training", "prediction"]): Upstream-stage marker.
            Drives the default output subpath.
        output_dir (str): Base directory for all artefacts.
        classifiers_dir (str): ``{output_dir}/classifiers`` — per-classifier
            artefact root.
        n_jobs (int): Outer joblib worker count.
        overwrite (bool): When ``True``, regenerate plot artefacts even
            if destination files already exist.
        verbose (bool): When ``True``, emits progress logs.
        results (dict[str, dict]): Per-classifier explanation dicts keyed
            by classifier name. Populated by :meth:`compute`.

    Example:
        >>> ee = ExplainerEnsemble(
        ...     classifier_ensemble=ce,
        ...     features_df=features_df,
        ...     observation_ids={"RandomForestClassifier": [0, 1, 2]},
        ... )
        >>> ee.compute().plot_aggregate()
    """

    def __init__(
        self,
        classifier_ensemble: ClassifierEnsemble,
        features_df: pd.DataFrame,
        observation_ids: dict[str, list[int]],
        method: Literal["shap"] = "shap",
        feature_perturbation: Literal[
            "tree_path_dependent", "interventional"
        ] = "tree_path_dependent",
        model_output: Literal["raw", "probability", "log_loss"] = "raw",
        background_size: int = 100,
        check_additivity: bool = True,
        kind: Literal["training", "prediction"] = "prediction",
        output_dir: str | None = None,
        root_dir: str | None = None,
        overwrite: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> None:
        """Bind a fitted ensemble to the data needed to explain its predictions.

        Validates the ``model_output`` / ``feature_perturbation`` combination
        and rejects the invalid pairs at construction time.

        Args:
            classifier_ensemble (ClassifierEnsemble): Fitted ensemble.
            features_df (pd.DataFrame): Feature matrix used both as the
                source of rows to explain and as the background pool for
                interventional perturbation.
            observation_ids (dict[str, list[int]]): Per-classifier list of
                window ids to explain (subset of
                ``features_df.index``).
            method (Literal["shap"], optional): Explanation method.
                Reserved for future additions. Defaults to ``"shap"``.
            feature_perturbation (Literal["tree_path_dependent",
                "interventional"], optional): SHAP perturbation mode.
                Defaults to ``"tree_path_dependent"``.
            model_output (Literal["raw", "probability", "log_loss"],
                optional): SHAP output unit. ``"probability"`` and
                ``"log_loss"`` require ``feature_perturbation
                ="interventional"``. Defaults to ``"raw"``.
            background_size (int, optional): Background sample size for
                the interventional path. Ignored when
                ``feature_perturbation="tree_path_dependent"``. Defaults
                to ``100``.
            check_additivity (bool, optional): Forwarded to the inner
                explainer call. Set ``False`` to silence the
                ``ExplainerError: Additivity check failed`` raised by
                some XGB configurations. Defaults to ``True``.
            kind (Literal["training", "prediction"], optional):
                Upstream-stage marker. Drives the default ``output_dir``
                subpath. Defaults to ``"prediction"``.
            output_dir (str | None, optional): Explicit base directory.
                ``None`` resolves to ``{root_dir}/output/explanation/{kind}``.
                Defaults to ``None``.
            root_dir (str | None, optional): Project root used to resolve
                ``output_dir`` when no explicit path is given. Defaults to
                ``None``.
            overwrite (bool, optional): When ``True``, regenerate plot
                artefacts even if destination files already exist.
                Defaults to ``False``.
            n_jobs (int, optional): Outer joblib worker count for the
                per-classifier loop. Defaults to ``1``.
            verbose (bool, optional): Emit progress logs. Defaults to
                ``False``.

        Raises:
            ValueError: If ``model_output`` is ``"probability"`` or
                ``"log_loss"`` while ``feature_perturbation`` is
                ``"tree_path_dependent"``.
        """
        if (
            model_output in ("probability", "log_loss")
            and feature_perturbation != "interventional"
        ):
            raise ValueError(
                f"model_output={model_output!r} requires feature_perturbation="
                f"'interventional' (needs a background distribution for the "
                f"link-function integration), got "
                f"feature_perturbation={feature_perturbation!r}."
            )

        output_dir = resolve_output_dir(
            output_dir=output_dir,
            root_dir=root_dir,
            default_subpath=os.path.join("output", "explanation", kind),
        )

        self.ClassifierEnsemble = classifier_ensemble
        self.features_df = features_df
        self.observation_ids = observation_ids
        self.method: Literal["shap"] = method
        self.feature_perturbation: Literal["tree_path_dependent", "interventional"] = (
            feature_perturbation
        )
        self.model_output: Literal["raw", "probability", "log_loss"] = model_output
        self.background_size = background_size
        self.check_additivity = check_additivity
        self.kind: Literal["training", "prediction"] = kind
        self.output_dir = output_dir
        self.classifiers_dir = os.path.join(output_dir, "classifiers")
        self.overwrite = overwrite
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.results: dict[str, dict] = {}

        if verbose:
            logger.info(
                f"ExplainerEnsemble: feature_perturbation={feature_perturbation!r}, "
                f"model_output={model_output!r}, check_additivity={check_additivity}. "
                f"Note: under model_output='raw' the SHAP units differ across "
                f"backends — sklearn RF/GB return probability space while "
                f"XGB returns margin (log-odds)."
            )

    @classmethod
    def load(cls, path: str) -> "ExplainerEnsemble":
        """Reconstitute a saved ``ExplainerEnsemble`` from a joblib ``.pkl``.

        Args:
            path (str): Path to a ``.pkl`` previously written by
                :meth:`save`.

        Returns:
            ExplainerEnsemble: The deserialised instance with
                :attr:`results` populated as of the save.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            TypeError: If the loaded object is not an
                ``ExplainerEnsemble``.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ExplainerEnsemble file not found: {path}")

        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object is not an ExplainerEnsemble (got {type(obj).__name__})."
            )
        return obj

    def compute(self) -> Self:
        """Materialise per-classifier per-seed SHAP values.

        Filters the classifier ensemble to tree-based classifiers via
        :func:`_is_tree_classifier`, dispatches the per-classifier
        explanation job to :func:`_explain_seed_ensemble` under
        ``joblib.Parallel(backend="loky")``, then writes one joblib dict
        per classifier under
        ``{classifiers_dir}/{classifier_name}/shap/explanations.pkl``.
        Idempotent — once :attr:`results` is populated, subsequent calls
        return immediately.

        Returns:
            Self: This instance, with :attr:`results` populated.
        """
        if self.results:
            if self.verbose:
                logger.info(
                    "ExplainerEnsemble.compute(): results already populated; "
                    "skipping recomputation."
                )
            return self

        tree_jobs: list[tuple[str, SeedEnsemble, list[int]]] = []
        for name, seed_ensemble in self.ClassifierEnsemble.ensembles.items():
            if not _is_tree_classifier(seed_ensemble):
                logger.info(f"ExplainerEnsemble: skipping non-tree classifier {name}")
                continue
            obs_ids = self.observation_ids.get(name)
            if not obs_ids:
                logger.info(
                    f"ExplainerEnsemble: no observation_ids supplied for "
                    f"{name}; skipping."
                )
                continue
            tree_jobs.append((name, seed_ensemble, list(obs_ids)))

        if not tree_jobs:
            logger.warning("ExplainerEnsemble: no tree-based classifiers to explain.")
            return self

        per_classifier_seeds: list[dict[int, dict]] = joblib.Parallel(
            n_jobs=self.n_jobs, backend="loky"
        )(
            joblib.delayed(_explain_seed_ensemble)(
                seed_ensemble=seed_ensemble,
                X=self.features_df,
                observation_ids=obs_ids,
                feature_perturbation=self.feature_perturbation,
                model_output=self.model_output,
                background_size=self.background_size,
                check_additivity=self.check_additivity,
            )
            for _, seed_ensemble, obs_ids in tree_jobs
        )

        for (name, _, obs_ids), seeds in zip(
            tree_jobs, per_classifier_seeds, strict=True
        ):
            self.results[name] = {
                "method": self.method,
                "feature_perturbation": self.feature_perturbation,
                "model_output": self.model_output,
                "observation_ids": list(obs_ids),
                "seeds": seeds,
            }

        self._persist_explanations()
        return self

    def plot_aggregate(
        self,
        include_plots: list[str] | None = None,
        exclude_plots: list[str] | None = None,
    ) -> list[str]:
        """Render aggregate plots registered in the dispatcher.

        Auto-runs :meth:`compute` first if explanations have not been
        materialised. For each ``(classifier_name, plot_name)`` pair,
        builds the frequency-weighted aggregate DataFrame via
        :meth:`_aggregate_importance`, then dispatches to
        :func:`render_one_aggregate_plot`.

        Args:
            include_plots (list[str] | None): Positive opt-in list of
                plot names. When ``None``, starts from the full
                dispatcher registry. Defaults to ``None``.
            exclude_plots (list[str] | None): Plot names to omit.
                Applied after ``include_plots``. Defaults to ``None``.

        Returns:
            list[str]: Saved figure filepath stems, one per executed job.
                The PNG figure and the CSV data table share each stem.
        """
        if not self.results:
            self.compute()

        plots = list(AGGREGATE_PLOT_DISPATCHER.keys())
        if include_plots:
            plots = [name for name in plots if name in include_plots]
        if exclude_plots:
            plots = [name for name in plots if name not in exclude_plots]

        jobs: list[tuple[str, str]] = [
            (clf, name) for clf in self.results for name in plots
        ]

        #  Build each plot's input on the orchestrator side so the workers
        #  remain pure renderers. The input shape varies per plot:
        #  ``"dataframe"`` for the bar, ``"explanation"`` for the beeswarm.
        return joblib.Parallel(n_jobs=self.n_jobs, backend="loky")(
            joblib.delayed(render_one_aggregate_plot)(
                classifier_name=clf,
                plot_name=name,
                aggregate_input=(
                    self._aggregate_importance(clf)
                    if AGGREGATE_PLOT_INPUT_KIND[name] == "dataframe"
                    else self._aggregate_explanation(clf)
                ),
                output_dir=self.classifiers_dir,
                overwrite=self.overwrite,
                verbose=self.verbose,
            )
            for clf, name in jobs
        )

    def plot_seed(
        self,
        include_plots: list[str] | None = None,
        exclude_plots: list[str] | None = None,
        plot_waterfall: bool = True,
    ) -> list[str]:
        """Render per-seed and per-observation plots in parallel.

        Per-seed plots (``bar``, ``beeswarm``) iterate over every seed.
        Waterfall plots additionally iterate over every observation; the
        cross-product cardinality is ``n_classifiers × n_seeds × n_obs``,
        so callers should keep ``n_observations_to_explain`` modest.

        Args:
            include_plots (list[str] | None): Positive opt-in list of
                per-seed plot names. When ``None``, starts from the full
                dispatcher registry. Defaults to ``None``.
            exclude_plots (list[str] | None): Per-seed plot names to
                omit. Applied after ``include_plots``. Defaults to
                ``None``.
            plot_waterfall (bool, optional): Whether to render
                per-(seed, obs) waterfall plots in addition to the
                per-seed bar / beeswarm plots. Defaults to ``True``.

        Returns:
            list[str]: Saved figure filepath stems, one per executed job.
        """
        if not self.results:
            self.compute()

        plots = list(PER_SEED_PLOT_DISPATCHER.keys())
        if include_plots:
            plots = [name for name in plots if name in include_plots]
        if exclude_plots:
            plots = [name for name in plots if name not in exclude_plots]

        seed_jobs: list[tuple] = []
        waterfall_jobs: list[tuple] = []
        for clf, payload in self.results.items():
            obs_ids = payload["observation_ids"]
            for random_state, seed_data in payload["seeds"].items():
                for plot_name in plots:
                    seed_jobs.append((clf, int(random_state), plot_name, seed_data))
                if plot_waterfall:
                    for obs_index, obs_id in enumerate(obs_ids):
                        waterfall_jobs.append(
                            (clf, int(random_state), obs_index, int(obs_id), seed_data)
                        )

        results: list[str] = []
        if seed_jobs:
            results.extend(
                joblib.Parallel(n_jobs=self.n_jobs, backend="loky")(
                    joblib.delayed(render_one_seed_plot)(
                        classifier_name=clf,
                        random_state=random_state,
                        plot_name=plot_name,
                        shap_values=seed_data["shap_values"],
                        base_value=seed_data["base_value"],
                        feature_values=seed_data["feature_values"],
                        feature_names=seed_data["feature_names"],
                        output_dir=self.classifiers_dir,
                        overwrite=self.overwrite,
                        verbose=self.verbose,
                    )
                    for clf, random_state, plot_name, seed_data in seed_jobs
                )
            )

        if waterfall_jobs:
            results.extend(
                joblib.Parallel(n_jobs=self.n_jobs, backend="loky")(
                    joblib.delayed(render_one_waterfall_plot)(
                        classifier_name=clf,
                        random_state=random_state,
                        obs_index=obs_index,
                        obs_id=obs_id,
                        shap_values=seed_data["shap_values"],
                        base_value=seed_data["base_value"],
                        feature_values=seed_data["feature_values"],
                        feature_names=seed_data["feature_names"],
                        output_dir=self.classifiers_dir,
                        overwrite=self.overwrite,
                        verbose=self.verbose,
                    )
                    for clf, random_state, obs_index, obs_id, seed_data in waterfall_jobs
                )
            )

        return results

    def save(self, path: str | None = None) -> str:
        """Persist the full instance to disk via joblib.

        Writes a single ``.pkl`` containing everything — the embedded
        ``ClassifierEnsemble``, ``features_df``, ``observation_ids``, and
        the populated :attr:`results` dict — so :meth:`load` can
        reconstitute the explainer without recomputation.

        Args:
            path (str | None, optional): Destination ``.pkl`` path.
                ``None`` resolves to
                ``{self.output_dir}/ExplainerEnsemble.pkl``. Defaults to
                ``None``.

        Returns:
            str: The absolute path the instance was written to.
        """
        if path is None:
            ensure_dir(self.output_dir)
            path = os.path.join(self.output_dir, "ExplainerEnsemble.pkl")
        else:
            parent = dirname(path)
            if parent:
                ensure_dir(parent)

        joblib.dump(self, path)
        if self.verbose:
            logger.info(f"Saved ExplainerEnsemble to {path}")

        return path

    def _aggregate_importance(self, classifier_name: str) -> pd.DataFrame:
        """Return the frequency-weighted aggregate-importance DataFrame.

        For each feature in the union of per-seed feature subsets, computes
        the mean of ``|SHAP|`` **only over seeds that selected the
        feature** (no zero-fill) and the fraction of seeds that selected
        it. Result is sorted by ``mean_abs_shap`` descending.

        Args:
            classifier_name (str): Classifier whose per-seed results are
                aggregated.

        Returns:
            pd.DataFrame: Columns ``feature``, ``mean_abs_shap``,
                ``selection_frequency``, ``n_seeds_selected``.
        """
        payload = self.results[classifier_name]
        seeds = payload["seeds"]
        total_seeds = len(seeds)

        accum: dict[str, list[float]] = {}
        for seed_data in seeds.values():
            shap_values = seed_data["shap_values"]
            feature_names = seed_data["feature_names"]
            abs_mean = np.abs(shap_values).mean(axis=0)
            for feature, value in zip(feature_names, abs_mean, strict=True):
                accum.setdefault(feature, []).append(float(value))

        rows: list[dict] = []
        for feature, values in accum.items():
            rows.append(
                {
                    "feature": feature,
                    "mean_abs_shap": float(np.mean(values)),
                    "selection_frequency": len(values) / total_seeds,
                    "n_seeds_selected": len(values),
                }
            )

        df = (
            pd.DataFrame(rows)
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        return df

    def _aggregate_explanation(
        self, classifier_name: str
    ) -> tuple[shap.Explanation, list[int], list[int]]:
        """Stack per-seed SHAP into one union-of-features ``shap.Explanation``.

        For each seed, the per-seed ``(n_obs, n_features_seed)`` block is
        placed into the corresponding columns of a ``(n_obs, |union|)``
        NaN-filled block; the per-seed blocks are then stacked vertically to
        form a ``(n_seeds × n_obs, |union|)`` matrix. Feature axis order
        matches :meth:`_aggregate_importance` (``mean_abs_shap`` descending)
        so the aggregate bar and beeswarm rank features identically.

        Args:
            classifier_name (str): Classifier whose per-seed results are
                aggregated.

        Returns:
            tuple[shap.Explanation, list[int], list[int]]: The stacked
                explanation, the seed identifier for each row, and the
                observation id for each row. Row alignment lets callers
                rebuild a tidy long-form table without re-walking the
                per-seed dicts.
        """
        payload = self.results[classifier_name]
        seeds: dict[int, dict] = payload["seeds"]
        obs_ids: list[int] = list(payload["observation_ids"])
        n_obs = len(obs_ids)

        #  Sort feature axis to match the bar plot so the two aggregate
        #  views rank features the same way.
        union_features: list[str] = (
            self._aggregate_importance(classifier_name)["feature"].tolist()
        )
        col_index = {name: i for i, name in enumerate(union_features)}

        seed_blocks: list[np.ndarray] = []
        data_blocks: list[np.ndarray] = []
        base_blocks: list[np.ndarray] = []
        row_seed: list[int] = []
        row_obs: list[int] = []

        for random_state, seed_data in seeds.items():
            shap_values = seed_data["shap_values"]
            feature_values = seed_data["feature_values"]
            feature_names: list[str] = list(seed_data["feature_names"])
            base_value = float(seed_data["base_value"])

            cols = np.array([col_index[name] for name in feature_names], dtype=int)

            block_values = np.full((n_obs, len(union_features)), np.nan)
            block_data = np.full((n_obs, len(union_features)), np.nan)
            block_values[:, cols] = shap_values
            block_data[:, cols] = feature_values

            seed_blocks.append(block_values)
            data_blocks.append(block_data)
            base_blocks.append(np.full(n_obs, base_value))
            row_seed.extend([int(random_state)] * n_obs)
            row_obs.extend(int(obs_id) for obs_id in obs_ids)

        values = np.vstack(seed_blocks)
        data = np.vstack(data_blocks)
        base_values = np.concatenate(base_blocks)

        explanation = shap.Explanation(
            values=values,
            base_values=base_values,
            data=data,
            feature_names=union_features,
        )
        return explanation, row_seed, row_obs

    def _persist_explanations(self) -> None:
        """Write per-classifier explanation pickles + aggregate CSVs.

        For every entry in :attr:`results`, writes
        ``{classifiers_dir}/{classifier_name}/shap/explanations.pkl`` (the
        full per-seed dict) and ``aggregate_importance.csv`` (the
        frequency-weighted aggregate) so downstream consumers can inspect
        SHAP outputs without loading the worker pickle.
        """
        for classifier_name, payload in self.results.items():
            shap_dir = os.path.join(self.classifiers_dir, classifier_name, "shap")
            ensure_dir(shap_dir)

            pkl_path = os.path.join(shap_dir, "explanations.pkl")
            joblib.dump(payload, pkl_path)

            agg_df = self._aggregate_importance(classifier_name)
            csv_path = os.path.join(shap_dir, "aggregate_importance.csv")
            agg_df.to_csv(csv_path, index=False)

            if self.verbose:
                logger.info(f"{classifier_name}: SHAP explanations saved to {shap_dir}")
