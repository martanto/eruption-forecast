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
from eruption_forecast.utils.ml import build_classifier_ensemble_summary
from eruption_forecast.utils.pathutils import ensure_dir, resolve_output_dir
from eruption_forecast.utils.formatting import shorten_feature_name
from eruption_forecast.ensemble.seed_ensemble import SeedEnsemble
from eruption_forecast.plots.explanation_plots import render_seed_plot
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
        root_dir: str | None = None,
        overwrite: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> None:
        """Bind a fitted ensemble to the data needed to explain its predictions."""
        output_dir = resolve_output_dir(
            output_dir=output_dir,
            root_dir=root_dir,
            default_subpath=os.path.join("output", "explanation", kind),
        )

        self.kind: Literal["training", "prediction"] = kind
        self.ClassifierEnsemble = classifier_ensemble
        self.features_df = features_df
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.explanations: list[ClassifierExplanation] = []

    @staticmethod
    def normalise_shap_values(
        explanation: shap.Explanation,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        self, labels: pd.Series | pd.DataFrame, eruption_dates: list[str]
    ):
        if len(self.explanations) == 0:
            raise ValueError("Please run explain() first")

        for classifier_explanation in self.explanations:
            classifier_name = classifier_explanation.classifier_name
            seed_ensemble = self.ClassifierEnsemble.ensembles[classifier_name]

            classifier_ensemble_summary = build_classifier_ensemble_summary(
                seed_ensemble=seed_ensemble,
                labels=labels,
                eruption_dates=eruption_dates,
            )
