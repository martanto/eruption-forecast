"""Per-classifier × per-seed SHAP explanation worker."""

import os
from typing import Self, Literal

import shap
import numpy as np
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import (
    save_data,
    ensure_dir,
    save_figure,
    resolve_output_dir,
)
from eruption_forecast.utils.formatting import shorten_feature_name
from eruption_forecast.ensemble.seed_ensemble import SeedEnsemble
from eruption_forecast.plots.explanation_plots import (
    render_seed_plot,
    plot_aggregate_shap_bar,
    plot_classifier_waterfall,
    plot_aggregate_shap_beeswarm,
)
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble
from eruption_forecast.dataclass.classifier_explanation import (
    SeedExplanation,
    ClassifierExplanation,
)


TREE_CLASSIFIERS: tuple[type, ...] = (
    RandomForestClassifier,
    XGBClassifier,
    GradientBoostingClassifier,
)


class ExplainerEnsemble:
    """Per-seed SHAP explanation engine over a fitted ``ClassifierEnsemble``."""

    def __init__(
        self,
        classifier_ensemble: ClassifierEnsemble,
        features_df: pd.DataFrame,
        kind: Literal["training", "prediction"] = "prediction",
        output_dir: str | None = None,
        explanation_dir: str | None = None,
        root_dir: str | None = None,
        overwrite: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> None:
        """Bind a fitted ensemble to the data needed to explain its predictions.

        Args:
            classifier_ensemble (ClassifierEnsemble): Fitted ensemble to
                explain.
            features_df (pd.DataFrame): Feature matrix the ensemble was
                trained or predicted on.
            kind (Literal["training", "prediction"]): Upstream-stage
                marker used to namespace the default output directory.
                Defaults to ``"prediction"``.
            output_dir (str | None): Per-classifier output directory.
                Houses ``{classifier_name}/`` subdirectories with
                ``ClassifierExplanation_*.pkl`` and per-seed figures.
                Defaults to ``None``.
            explanation_dir (str | None): Parent explanation directory,
                used as the root for per-eruption waterfall plots so they
                land alongside ``classifiers/`` rather than nested inside
                it. ``None`` falls back to ``os.path.dirname(output_dir)``.
                Defaults to ``None``.
            root_dir (str | None): Project root used to anchor relative
                paths when ``output_dir`` is ``None``. Defaults to
                ``None``.
            overwrite (bool): Overwrite cached SHAP outputs. Defaults to
                ``False``.
            n_jobs (int): Parallel workers for per-seed plotting.
                Defaults to ``1``.
            verbose (bool): Verbose logging. Defaults to ``False``.
        """
        output_dir = resolve_output_dir(
            output_dir=output_dir,
            root_dir=root_dir,
            default_subpath=os.path.join("output", "explanation", kind),
        )

        self.kind: Literal["training", "prediction"] = kind
        self.ClassifierEnsemble = classifier_ensemble
        self.features_df = features_df
        self.output_dir = output_dir
        self.explanation_dir = (
            explanation_dir
            if explanation_dir is not None
            else os.path.dirname(output_dir.rstrip(os.sep))
        )
        self.overwrite = overwrite
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.explanations: list[ClassifierExplanation] = []

    @staticmethod
    def normalise_shap_values(
        explanation: shap.Explanation,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project SHAP values onto the positive class for binary classifiers.

        ``shap.TreeExplainer`` emits a three-dimensional ``values`` array
        of shape ``(n_samples, n_features, 2)`` for binary classifiers and
        a flat 2-D array for the regression-like path. This helper picks
        the positive-class slice when present so downstream code can treat
        either shape uniformly. ``base_values`` is normalised the same way.

        Args:
            explanation (shap.Explanation): Raw SHAP explanation produced
                by ``shap.TreeExplainer``.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(shap_values, base_values)``
                with the positive-class slice extracted when applicable.
        """
        vals = np.asarray(explanation.values)
        shap_values = vals[..., 1] if vals.ndim == 3 else vals

        base = np.asarray(explanation.base_values)
        base_values = base[..., 1] if base.ndim >= 2 else base

        return shap_values, base_values

    @staticmethod
    def explain_seed(
        seed: dict,
        features_df: pd.DataFrame,
        save_per_seed: bool = False,
        check_additivity: bool = False,
        seed_explanation_filepath: str | None = None,
    ) -> shap.Explanation:
        """Compute the SHAP explanation for a single seed model.

        Builds a ``shap.TreeExplainer`` over the seed's selected feature
        subset, runs it on the supplied ``features_df``, normalises the
        result via :meth:`normalise_shap_values`, and shortens feature
        names so downstream plots stay readable.

        Args:
            seed (dict): Seed record carrying ``"model"``,
                ``"feature_names"``, and ``"random_state"`` entries.
            features_df (pd.DataFrame): Full feature matrix. Sliced to the
                seed's selected columns before SHAP runs.
            save_per_seed (bool): When ``True`` persist the resulting
                ``shap.Explanation`` to ``seed_explanation_filepath`` via
                joblib. Defaults to ``False``.
            check_additivity (bool): Forwarded to the explainer call to
                verify SHAP additivity against the model output. Defaults
                to ``False``.
            seed_explanation_filepath (str | None): Destination path used
                when ``save_per_seed`` is ``True``. Defaults to ``None``.

        Returns:
            shap.Explanation: Positive-class SHAP values with shortened
                feature names.

        Raises:
            ValueError: If SHAP fails to explain the seed (wrapped to
                include the seed identifier).
        """
        model = seed["model"]
        feature_names: list[str] = seed["feature_names"]
        selected_features_df = features_df[feature_names]

        try:
            explainer = shap.TreeExplainer(
                model,
                selected_features_df,
                feature_perturbation="tree_path_dependent",
                model_output="raw",
            )

            explanation: shap.Explanation = explainer(
                selected_features_df, check_additivity=check_additivity
            )

            shap_values, base_value = ExplainerEnsemble.normalise_shap_values(
                explanation
            )

            explanation = shap.Explanation(
                values=shap_values,
                base_values=base_value,
                data=explanation.data,
                feature_names=[shorten_feature_name(name) for name in feature_names],
            )

            if save_per_seed and seed_explanation_filepath:
                joblib.dump(explanation, seed_explanation_filepath)

            return explanation
        except Exception as e:
            raise ValueError(
                f"Explaining Seed {seed['random_state']}: {selected_features_df.shape}. {e}"
            ) from e

    @staticmethod
    def explain_classifier(
        seed_ensemble: SeedEnsemble,
        features_df: pd.DataFrame,
        save_per_seed: bool = False,
        kind: Literal["training", "prediction"] = "prediction",
        check_additivity: bool = False,
        output_dir: str | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> ClassifierExplanation:
        """Explain every seed of a single classifier's ``SeedEnsemble``.

        Iterates over every seed in ``seed_ensemble`` and either loads its
        cached ``shap.Explanation`` from disk (when one exists and
        ``overwrite`` is ``False``) or computes a fresh explanation via
        :meth:`explain_seed`. Per-seed ``.pkl`` files land under
        ``{output_dir}/{classifier_name}/shap_values/`` when
        ``save_per_seed`` is ``True``.

        Args:
            seed_ensemble (SeedEnsemble): Trained ensemble for a single
                classifier.
            features_df (pd.DataFrame): Feature matrix the seeds were
                trained on.
            save_per_seed (bool): Persist each per-seed ``shap.Explanation``
                to disk. Defaults to ``False``.
            kind (Literal["training", "prediction"]): Upstream-stage
                marker used only to build the default ``output_dir``.
                Defaults to ``"prediction"``.
            check_additivity (bool): Forwarded to ``shap.TreeExplainer``.
                Defaults to ``False``.
            output_dir (str | None): Per-classifier root directory.
                Defaults to ``output/explanation/{kind}/classifiers`` under
                the current working directory.
            overwrite (bool): Re-compute and overwrite cached per-seed
                ``.pkl`` files. Defaults to ``False``.
            verbose (bool): Verbose logging. Defaults to ``False``.

        Returns:
            ClassifierExplanation: One entry per seed.
        """
        classifier_name = seed_ensemble.classifier_name
        seeds = seed_ensemble.seeds

        classifier_explanation = ClassifierExplanation(
            classifier_name=classifier_name,
        )

        if verbose:
            logger.info(
                f"Explaining SeedEnsemble: {classifier_name} with {len(seeds)} seeds"
            )

        output_dir = ensure_dir(
            output_dir
            or os.path.join(os.getcwd(), "output", "explanation", kind, "classifiers")
        )

        output_shap_dir = os.path.join(output_dir, classifier_name, "shap_values")
        if save_per_seed:
            ensure_dir(output_shap_dir)

        for seed in seeds:
            seed_idx = seed["random_state"]

            seed_explanation_filepath = os.path.join(
                output_shap_dir,
                f"{seed_idx:05d}.pkl",
            )

            if not overwrite and os.path.exists(seed_explanation_filepath):
                if verbose:
                    logger.info(
                        f"SeedExplanation {classifier_name}/{seed_idx:05d} exists."
                    )
                seed_explanation: shap.Explanation = joblib.load(
                    seed_explanation_filepath
                )
                classifier_explanation.seeds.append(
                    SeedExplanation(
                        random_state=int(seed_idx),
                        shap_values=seed_explanation,
                    )
                )
                continue

            if verbose:
                logger.info(f"Explaining {classifier_name}/{seed_idx:05d}")

            seed_explanation: shap.Explanation = ExplainerEnsemble.explain_seed(
                seed=seed,
                features_df=features_df,
                save_per_seed=save_per_seed,
                check_additivity=check_additivity,
                seed_explanation_filepath=seed_explanation_filepath,
            )

            if save_per_seed and verbose:
                logger.info(
                    f"Done. Explanation {classifier_name}/{seed['random_state']}: "
                    f"{seed_explanation_filepath}"
                )

            classifier_explanation.seeds.append(
                SeedExplanation(
                    random_state=int(seed_idx),
                    shap_values=seed_explanation,
                )
            )

        return classifier_explanation

    def explain(
        self,
        save_per_seed: bool = True,
        check_additivity: bool = False,
        overwrite_classifier_explanation: bool = False,
    ) -> Self:
        """Explain every classifier in the wrapped ``ClassifierEnsemble``.

        Iterates over each ``SeedEnsemble``, loads any cached
        ``ClassifierExplanation.pkl`` when present, and otherwise
        delegates to :meth:`explain_classifier`. Non-tree classifiers are
        skipped with a warning because ``shap.TreeExplainer`` does not
        apply.

        Args:
            save_per_seed (bool): Persist each per-seed
                ``shap.Explanation`` to disk. Defaults to ``True``.
            check_additivity (bool): Forwarded to ``shap.TreeExplainer``.
                Defaults to ``False``.
            overwrite_classifier_explanation (bool): Overwrite the cached
                per-classifier ``ClassifierExplanation.pkl`` artefact.
                ``or``-ed with ``self.overwrite``. Defaults to ``False``.

        Returns:
            Self: The current instance, enabling method chaining.
        """
        overwrite = overwrite_classifier_explanation or self.overwrite

        for seed_ensemble in self.ClassifierEnsemble.ensembles.values():
            classifier_name = seed_ensemble.classifier_name

            save_filepath = os.path.join(
                self.output_dir,
                classifier_name,
                f"ClassifierExplanation_{classifier_name}.pkl",
            )

            ensure_dir(os.path.dirname(save_filepath))

            if os.path.exists(save_filepath) and not overwrite:
                classifier_explanation = joblib.load(save_filepath)

                if self.verbose:
                    logger.info(
                        f"[{classifier_name}]: Loaded from cache: {save_filepath}"
                    )

                self.explanations.append(classifier_explanation)
                continue

            if not seed_ensemble.seeds:
                logger.warning(
                    f"Skipping {classifier_name}: no seed models in ensemble."
                )
                continue

            if not isinstance(seed_ensemble.seeds[0]["model"], TREE_CLASSIFIERS):
                logger.warning(
                    f"Skipping {classifier_name}: non-tree classifier, "
                    f"SHAP TreeExplainer does not apply."
                )
                continue

            if self.verbose:
                logger.info(f"Explaining {classifier_name}")

            classifier_explanation: ClassifierExplanation = self.explain_classifier(
                seed_ensemble=seed_ensemble,
                features_df=self.features_df,
                save_per_seed=save_per_seed,
                check_additivity=check_additivity,
                output_dir=self.output_dir,
                overwrite=self.overwrite,
                verbose=self.verbose,
            )

            joblib.dump(classifier_explanation, save_filepath)
            logger.info(
                f"[{classifier_name}]: ClassifierExplanation saved to: {save_filepath}"
            )

            self.explanations.append(classifier_explanation)

        logger.info(f"Explained {len(self.explanations)} classifiers")

        return self

    def plot_seed(
        self,
        max_display: int = 20,
        group_remaining_features: bool = False,
        dpi: int = 150,
    ):
        """Render per-seed bar and beeswarm SHAP plots in parallel.

        Plots land under
        ``{output_dir}/{classifier_name}/figures/{kind}/{seed:05d}.png``
        where ``kind`` is ``"bar"`` or ``"beeswarm"``. Existing files are
        skipped when ``self.overwrite`` is ``False``.

        Args:
            max_display (int): Maximum number of features to display.
                Defaults to ``20``.
            group_remaining_features (bool): Forwarded to
                ``shap.plots.beeswarm``. Defaults to ``False``.
            dpi (int): Figure resolution in dots per inch. Defaults to
                ``150``.

        Raises:
            ValueError: If :meth:`explain` has not yet been called.
        """
        if len(self.explanations) == 0:
            raise ValueError("Please run explain() first")

        jobs: list[tuple] = []
        for explanation in self.explanations:
            classifier_name = explanation.classifier_name
            figures_dir = os.path.join(self.output_dir, classifier_name, "figures")

            for seed in explanation.seeds:
                seed_id = int(seed.random_state)
                shap_values = seed.shap_values
                title = f"{classifier_name}[{seed_id:05d}]"

                for plot_kind in ("beeswarm", "bar"):
                    save_filepath = os.path.join(
                        figures_dir, plot_kind, f"{seed_id:05d}.png"
                    )

                    if os.path.exists(save_filepath) and not self.overwrite:
                        continue

                    jobs.append(
                        (
                            plot_kind,
                            shap_values,
                            save_filepath,
                            title,
                            max_display,
                            group_remaining_features,
                            dpi,
                            self.verbose,
                        )
                    )

        if not jobs:
            return None

        joblib.Parallel(n_jobs=self.n_jobs, backend="loky")(
            joblib.delayed(render_seed_plot)(*job) for job in jobs
        )

        return None

    def plot_waterfall(
        self,
        labels: pd.Series | pd.DataFrame,
        eruption_dates: list[str],
        figsize: tuple[float, float] | None = None,
        max_display: int = 20,
        dpi: int = 150,
    ):
        """Render the per-eruption SHAP waterfall plot for each classifier.

        Writes plots under
        ``{explanation_dir}/eruptions/{eruption_date}/`` — a sibling of
        ``classifiers/`` — so the eruption-centric artefacts stay outside
        the per-classifier tree.

        Args:
            labels (pd.Series | pd.DataFrame): Ground-truth label series
                indexed by window id; used by
                :func:`~eruption_forecast.utils.ml.build_classifier_ensemble_summary`
                to pick the highest-probability window per eruption.
            eruption_dates (list[str]): Ground-truth eruption dates in
                ``"YYYY-MM-DD"`` format.
            figsize (tuple[float, float] | None): Figure size in inches.
                ``None`` auto-sizes from ``max_display``. Defaults to
                ``None``.
            max_display (int): Maximum number of features to display.
                Defaults to ``20``.
            dpi (int): Figure resolution in dots per inch. Defaults to
                ``150``.

        Raises:
            ValueError: If :meth:`explain` has not yet been called.
        """
        if len(self.explanations) == 0:
            raise ValueError("Please run explain() first")

        eruption_dir = os.path.join(self.explanation_dir, "eruptions")

        for classifier_explanation in self.explanations:
            plot_classifier_waterfall(
                classifier_explanation=classifier_explanation,
                classifier_ensemble=self.ClassifierEnsemble,
                labels=labels,
                eruption_dates=eruption_dates,
                figsize=figsize,
                max_display=max_display,
                output_dir=eruption_dir,
                dpi=dpi,
                verbose=self.verbose,
            )

    @staticmethod
    def _aggregate_importance(
        classifier_explanation: ClassifierExplanation,
    ) -> pd.DataFrame:
        """Compute a per-feature aggregate importance table across all seeds.

        For each feature appearing in any seed's ``shap.Explanation``, records
        the mean ``|SHAP|`` averaged over the seeds that selected the feature
        (the conditional importance), the count of seeds that selected it, and
        the corresponding ``selection_frequency``. Seed values are reduced
        from 3D to 2D by slicing the positive class (mirrors the per-seed
        beeswarm rendering at
        :func:`~eruption_forecast.plots.explanation_plots.plot_shap_beeswarm`)
        before the per-seed mean ``|SHAP|`` is taken.

        Args:
            classifier_explanation (ClassifierExplanation): Per-classifier
                bundle of per-seed ``SeedExplanation`` payloads.

        Returns:
            pd.DataFrame: Frequency-weighted importance table sorted by
                ``mean_abs_shap`` descending. Columns are ``feature``,
                ``mean_abs_shap``, ``selection_frequency``, and
                ``n_seeds_selected``. The schema matches the input contract
                of
                :func:`~eruption_forecast.plots.explanation_plots.plot_aggregate_shap_bar`.
        """
        n_total_seeds = len(classifier_explanation.seeds)
        accumulator: dict[str, list[float]] = {}
        for seed in classifier_explanation.seeds:
            explanation = seed.shap_values
            values = np.asarray(explanation.values)  # noqa: PD011
            if values.ndim == 3:
                values = values[..., 1]
            mean_abs = np.abs(values).mean(axis=0)
            for name, val in zip(
                list(explanation.feature_names), mean_abs, strict=True
            ):
                accumulator.setdefault(name, []).append(float(val))

        rows = [
            {
                "feature": feature,
                "mean_abs_shap": float(np.mean(seed_values)),
                "selection_frequency": len(seed_values) / n_total_seeds,
                "n_seeds_selected": len(seed_values),
            }
            for feature, seed_values in accumulator.items()
        ]
        importance_df = pd.DataFrame(
            rows,
            columns=[
                "feature",
                "mean_abs_shap",
                "selection_frequency",
                "n_seeds_selected",
            ],
        )
        return importance_df.sort_values(
            "mean_abs_shap", ascending=False
        ).reset_index(drop=True)

    @staticmethod
    def _aggregate_explanation(
        classifier_explanation: ClassifierExplanation,
    ) -> tuple[shap.Explanation, list[int], list[int]]:
        """Stack per-seed ``shap.Explanation``s into a union feature space.

        Builds the ordered union of feature names across every seed
        (preserving first-seen order so re-runs are deterministic), then
        NaN-pads each seed's ``values`` and ``data`` matrices from its
        per-seed columns into the union space and concatenates along the
        sample axis. Cells where a seed did not select a feature carry
        ``np.nan`` so SHAP's beeswarm internals skip them — preserving the
        "this seed didn't pick it" signal instead of inflating the swarm
        with fake zeros. 3D ``values`` are reduced to the positive class
        before padding (matches the per-seed beeswarm path at
        :func:`~eruption_forecast.plots.explanation_plots.plot_shap_beeswarm`).

        Args:
            classifier_explanation (ClassifierExplanation): Per-classifier
                bundle of per-seed ``SeedExplanation`` payloads.

        Returns:
            tuple[shap.Explanation, list[int], list[int]]: A 3-tuple of:

                - **explanation** (``shap.Explanation``): Stacked explanation
                  of shape ``(n_seeds * n_obs, |union features|)`` with
                  ``feature_names`` set to the union list.
                - **row_seed** (``list[int]``): ``random_state`` for each
                  row of ``explanation.values``.
                - **row_obs** (``list[int]``): Window id for each row of
                  ``explanation.values`` (the row index within the
                  per-seed explanation, since per-seed explanations are
                  aligned positionally to ``features_df``).
        """
        all_names: list[str] = []
        seen: set[str] = set()
        for seed in classifier_explanation.seeds:
            for name in seed.shap_values.feature_names:
                if name not in seen:
                    seen.add(name)
                    all_names.append(name)
        name_to_idx = {name: i for i, name in enumerate(all_names)}
        n_features = len(all_names)

        values_blocks: list[np.ndarray] = []
        data_blocks: list[np.ndarray] = []
        row_seed: list[int] = []
        row_obs: list[int] = []

        for seed in classifier_explanation.seeds:
            explanation = seed.shap_values
            values = np.asarray(explanation.values)  # noqa: PD011
            if values.ndim == 3:
                values = values[..., 1]
            raw_data = getattr(explanation, "data", None)
            data = (
                np.asarray(raw_data)
                if raw_data is not None
                else np.full_like(values, np.nan)
            )
            n_samples = values.shape[0]

            padded_values = np.full((n_samples, n_features), np.nan)
            padded_data = np.full((n_samples, n_features), np.nan)
            for j, name in enumerate(explanation.feature_names):
                col_idx = name_to_idx[name]
                padded_values[:, col_idx] = values[:, j]
                padded_data[:, col_idx] = data[:, j]

            values_blocks.append(padded_values)
            data_blocks.append(padded_data)
            row_seed.extend([int(seed.random_state)] * n_samples)
            row_obs.extend(range(n_samples))

        merged_values = np.concatenate(values_blocks, axis=0)
        merged_data = np.concatenate(data_blocks, axis=0)

        aggregated = shap.Explanation(
            values=merged_values,
            data=merged_data,
            feature_names=all_names,
        )
        return aggregated, row_seed, row_obs

    def plot_aggregate(
        self,
        max_display: int = 20,
        top_n: int = 20,
        figsize: tuple[float, float] | None = None,
        dpi: int = 150,
        group_remaining_features: bool = False,
    ) -> None:
        """Render aggregate bar and beeswarm SHAP plots per classifier.

        For every ``ClassifierExplanation`` on :attr:`explanations`, builds
        the frequency-weighted importance table via
        :meth:`_aggregate_importance` and the NaN-padded stacked explanation
        via :meth:`_aggregate_explanation`, then feeds them to
        :func:`~eruption_forecast.plots.explanation_plots.plot_aggregate_shap_bar`
        and
        :func:`~eruption_forecast.plots.explanation_plots.plot_aggregate_shap_beeswarm`.
        Outputs land under
        ``{output_dir}/{classifier_name}/figures/aggregate/`` as
        ``bar.{png,csv}`` and ``beeswarm.{png,csv}`` — the ``.csv`` files
        are the durable sidecar artefacts (importance table for the bar
        plot, tidy long-form non-NaN cell list for the beeswarm).

        Args:
            max_display (int): Maximum number of features to display in
                the beeswarm. Defaults to ``20``.
            top_n (int): Maximum number of features to display in the bar
                plot. Defaults to ``20``.
            figsize (tuple[float, float] | None): Figure size in inches.
                ``None`` auto-sizes inside each renderer. Defaults to
                ``None``.
            dpi (int): Figure resolution in dots per inch. Defaults to
                ``150``.
            group_remaining_features (bool): Reserved for parity with the
                per-seed renderer signature.
                ``plot_aggregate_shap_beeswarm`` does not forward this
                kwarg to ``shap.plots.beeswarm`` because the NaN-padded
                stack uses SHAP's default grouping behaviour. Defaults to
                ``False``.

        Raises:
            ValueError: If :meth:`explain` has not yet been called.
        """
        if len(self.explanations) == 0:
            raise ValueError("Please run explain() first")

        del group_remaining_features  # Reserved for signature parity.

        for classifier_explanation in self.explanations:
            classifier_name = classifier_explanation.classifier_name
            aggregate_dir = os.path.join(
                self.output_dir, classifier_name, "figures", "aggregate"
            )
            ensure_dir(aggregate_dir)

            importance_df = self._aggregate_importance(classifier_explanation)
            bar_path = os.path.join(aggregate_dir, "bar")
            bar_fig, _ = plot_aggregate_shap_bar(
                aggregate_df=importance_df,
                top_n=top_n,
                figsize=figsize,
                dpi=dpi,
                title=f"{classifier_name} — Aggregate Top-{top_n} SHAP Importances",
            )
            save_figure(bar_fig, bar_path, dpi, verbose=self.verbose)
            save_data(importance_df, bar_path, filetype="csv")

            aggregated, row_seed, row_obs = self._aggregate_explanation(
                classifier_explanation
            )
            beeswarm_path = os.path.join(aggregate_dir, "beeswarm")
            beeswarm_fig, tidy_df = plot_aggregate_shap_beeswarm(
                explanation=aggregated,
                row_seed=row_seed,
                row_obs=row_obs,
                max_display=max_display,
                figsize=figsize,
                dpi=dpi,
                title=f"{classifier_name} — Aggregate SHAP Summary",
            )
            save_figure(beeswarm_fig, beeswarm_path, dpi, verbose=self.verbose)
            save_data(tidy_df, beeswarm_path, filetype="csv")
