"""Per-classifier × per-seed DALEX worker over a fitted ``ClassifierEnsemble``.

Mirrors :class:`~eruption_forecast.ensemble.metrics_ensemble.MetricsEnsemble`'s
iteration pattern but produces explanatory artefacts instead of classification
metrics. Restricted to tree classifiers (``RandomForestClassifier``,
``XGBClassifier``, ``GradientBoostingClassifier``); other classifiers in the
ensemble are skipped with one INFO-level log line.

Three DALEX analyses run per sampled seed:

- :py:meth:`dx.Explainer.predict_parts` (``type='shap'``) — local SHAP
  attribution on the observations selected by :meth:`_select_observations`.
- :py:meth:`dx.Explainer.model_parts` (``loss_function='1-auc'``) — global
  permutation variable importance.
- :py:meth:`dx.Explainer.model_profile` (``type='partial'``) — Partial
  Dependence Profiles for the top-``k`` features ranked by mean dropout loss.
"""

import os
import warnings
from typing import Self, Literal

import dalex as dx
import numpy as np
import joblib
import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import save_data, ensure_dir
from eruption_forecast.utils.formatting import slugify
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


TREE_CLASSIFIERS: frozenset[str] = frozenset(
    {
        "LiteRandomForestClassifier",
        "RandomForestClassifier",
        "XGBClassifier",
        "GradientBoostingClassifier",
    }
)
"""Whitelist of classifier names that DALEX explanations are restricted to."""


def _predict_proba_positive(model, data) -> np.ndarray:
    """Return ``P(class=1)`` for sklearn-compatible binary classifiers.

    Used as the ``predict_function`` for :class:`dalex.Explainer` so that
    ``model_parts`` / ``predict_parts`` / ``model_profile`` operate on the
    eruption-positive probability instead of the default class label.

    Args:
        model: Fitted estimator with a ``predict_proba`` method returning a
            ``(n_samples, 2)`` array.
        data: Feature matrix to score. Anything ``model.predict_proba`` accepts.

    Returns:
        np.ndarray: 1-D array of length ``n_samples`` with ``P(class=1)``.
    """
    proba = model.predict_proba(data)
    if proba.ndim == 2 and proba.shape[1] == 2:
        return proba[:, 1]
    return np.asarray(proba).ravel()


class DalexExplainerEnsemble:
    """Run DALEX explanations on every tree-classifier seed in a ``ClassifierEnsemble``.

    The constructor binds the ensemble to a feature matrix and ground-truth
    labels; :meth:`compute` then iterates over each tree classifier and an
    evenly-spaced subset of its seeds, building one :class:`dx.Explainer` per
    seed and running SHAP / variable importance / partial dependence analyses.
    Per-seed CSVs and HTML plots are written under
    ``{classifiers_dir}/{classifier_name}/{shap,variable_importance,partial_dependence}/seeds/``;
    aggregates across the sampled seeds are written alongside.

    Plotly figures are saved as standalone HTML (``include_plotlyjs="cdn"``)
    rather than PNG so the implementation does not depend on the
    ``kaleido``/Chromium static-image stack, which is fragile on Windows.

    Attributes:
        ClassifierEnsemble (ClassifierEnsemble): Fitted ensemble whose tree
            classifiers drive the explanation loop.
        features_df (pd.DataFrame): Feature matrix aligned with :attr:`y_true`.
        y_true (np.ndarray): Ground-truth binary labels of length ``n_samples``.
        kind (Literal["training", "prediction"]): Reuse mode; propagated from
            the upstream stage.
        output_dir (str): Resolved explanation root for this instance.
        classifiers_dir (str): ``{output_dir}/classifiers`` — per-classifier
            artefact root.
        n_seeds_to_explain (int): Number of seeds sampled per classifier.
        n_observations_to_explain (int): Number of observations fed to
            ``predict_parts`` per seed.
        top_k_features (int): Number of top-ranked features fed to
            ``model_profile``.
        permutation_B (int): Number of permutation rounds for ``model_parts``.
        shap_B (int): Number of random paths for ``predict_parts(type='shap')``.
        n_jobs (int): Reserved for future parallelism — sequential today.
        overwrite (bool): Reserved for future skip-if-exists logic.
        verbose (bool): Emit progress logs when ``True``.
        shap_results (dict): ``{clf_name: {random_state: result_df}}`` populated
            by :meth:`compute`.
        vi_results (dict): Same layout for variable importance results.
        pdp_results (dict): Same layout for partial dependence results;
            each value is a dict with ``"result"`` and ``"variables"`` keys.
    """

    def __init__(
        self,
        classifier_ensemble: ClassifierEnsemble,
        features_df: pd.DataFrame,
        y_true: pd.Series | np.ndarray,
        kind: Literal["training", "prediction"] = "prediction",
        output_dir: str | None = None,
        n_seeds_to_explain: int = 10,
        n_observations_to_explain: int = 5,
        top_k_features: int = 5,
        permutation_B: int = 10,
        shap_B: int = 25,
        overwrite: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> None:
        """Bind a fitted ensemble to the data needed for DALEX explanations.

        Args:
            classifier_ensemble (ClassifierEnsemble): Fitted ensemble.
            features_df (pd.DataFrame): Feature matrix aligned with ``y_true``.
            y_true (pd.Series | np.ndarray): Ground-truth binary labels.
                ``pd.Series`` inputs are converted to ``np.ndarray``.
            kind (Literal["training", "prediction"]): Reuse mode. Defaults to
                ``"prediction"``.
            output_dir (str | None): Explanation root. ``None`` resolves to
                ``./output/explanation/{kind}``. Defaults to ``None``.
            n_seeds_to_explain (int): Seeds sampled per classifier via
                even-stride selection over sorted ``random_state``. Defaults
                to ``10``.
            n_observations_to_explain (int): Observations fed to
                ``predict_parts`` per seed. Defaults to ``5``.
            top_k_features (int): Number of top features ranked by VI fed to
                ``model_profile``. Defaults to ``5``.
            permutation_B (int): Permutation rounds for ``model_parts``.
                Defaults to ``10``.
            shap_B (int): Random paths for ``predict_parts(type='shap')``.
                Defaults to ``25``.
            overwrite (bool): Reserved for future skip-if-exists logic.
                Defaults to ``False``.
            n_jobs (int): Reserved for future parallelism. Defaults to ``1``.
            verbose (bool): Emit progress logs. Defaults to ``False``.
        """
        if isinstance(y_true, pd.Series):
            y_true = y_true.to_numpy()

        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "output", "explanation", kind)

        self.ClassifierEnsemble = classifier_ensemble
        self.features_df = features_df
        self.y_true: np.ndarray = np.asarray(y_true).astype(int)
        self.kind: Literal["training", "prediction"] = kind
        self.output_dir = output_dir
        self.classifiers_dir = os.path.join(output_dir, "classifiers")
        self.n_seeds_to_explain = n_seeds_to_explain
        self.n_observations_to_explain = n_observations_to_explain
        self.top_k_features = top_k_features
        self.permutation_B = permutation_B
        self.shap_B = shap_B
        self.overwrite = overwrite
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.shap_results: dict[str, dict[int, pd.DataFrame]] = {}
        self.vi_results: dict[str, dict[int, pd.DataFrame]] = {}
        self.pdp_results: dict[str, dict[int, dict]] = {}

    @property
    def tree_classifiers(self) -> list[str]:
        """Return the tree-classifier names present in the bound ensemble.

        Returns:
            list[str]: Subset of ``ClassifierEnsemble.classifiers`` whose
                names appear in :data:`TREE_CLASSIFIERS`, preserving the
                ensemble's insertion order.
        """
        return [n for n in self.ClassifierEnsemble.classifiers if n in TREE_CLASSIFIERS]

    @property
    def skipped_classifiers(self) -> list[str]:
        """Return classifier names that fall outside the tree-only whitelist.

        Returns:
            list[str]: Classifier names that DALEX will not explain.
        """
        return [
            n for n in self.ClassifierEnsemble.classifiers if n not in TREE_CLASSIFIERS
        ]

    def _sample_seeds(self, seed_records: list[dict]) -> list[dict]:
        """Pick an evenly-spaced subset of seeds sorted by ``random_state``.

        Args:
            seed_records (list[dict]): The full per-seed records from a
                ``SeedEnsemble`` (each dict carries ``random_state``,
                ``model``, ``feature_names``).

        Returns:
            list[dict]: ``min(len(seed_records), n_seeds_to_explain)`` records.
                Selection is deterministic across calls because indices are
                drawn from :func:`numpy.linspace` over the sorted list.
        """
        n_total = len(seed_records)
        if n_total <= self.n_seeds_to_explain:
            return seed_records
        sorted_seeds = sorted(seed_records, key=lambda r: int(r["random_state"]))
        indices = np.linspace(0, n_total - 1, self.n_seeds_to_explain).astype(int)
        return [sorted_seeds[i] for i in indices]

    def _select_observations(self, X_seed: pd.DataFrame, model) -> pd.DataFrame:
        """Choose observations for local SHAP attribution.

        Positive-class samples (``y_true == 1``) come first; if fewer than
        ``n_observations_to_explain`` are available, the shortfall is filled
        with the top-probability rows under the seed's own ``predict_proba``.

        Args:
            X_seed (pd.DataFrame): Feature matrix sliced to the seed's
                ``feature_names``.
            model: The fitted seed estimator used for ranking when there are
                not enough positive-class observations.

        Returns:
            pd.DataFrame: ``n_observations_to_explain`` rows from ``X_seed``,
                or fewer when the matrix itself is shorter.
        """
        n_target = min(self.n_observations_to_explain, len(X_seed))
        pos_mask = self.y_true == 1
        pos_idx = np.where(pos_mask)[0]
        if len(pos_idx) >= n_target:
            return X_seed.iloc[pos_idx[:n_target]]

        proba = _predict_proba_positive(model, X_seed)
        order = np.argsort(-proba)
        selected: list[int] = list(pos_idx)
        for idx in order:
            if idx in selected:
                continue
            selected.append(int(idx))
            if len(selected) >= n_target:
                break
        return X_seed.iloc[selected[:n_target]]

    def compute(
        self,
        plot_local: bool = True,
        plot_global: bool = True,
        plot_profile: bool = True,
    ) -> Self:
        """Run all enabled DALEX analyses across the sampled seeds.

        Iterates classifier-by-classifier, skipping non-tree classifiers with
        one INFO-level log line each. For each tree classifier, samples a
        deterministic subset of seeds via :meth:`_sample_seeds` and feeds
        them through :meth:`_compute_seed`. After all seeds for a classifier
        finish, :meth:`_aggregate_and_persist` writes the cross-seed summary.

        Args:
            plot_local (bool): Save per-seed SHAP attribution HTML plots.
                Defaults to ``True``.
            plot_global (bool): Save per-seed permutation-importance HTML
                plots. Defaults to ``True``.
            plot_profile (bool): Save per-feature PDP HTML plots. Defaults to
                ``True``.

        Returns:
            Self: This instance, with the ``*_results`` dicts populated.
        """
        for skipped in self.skipped_classifiers:
            logger.info(f"Skipping {skipped}: not a tree model")

        for clf_name in self.tree_classifiers:
            seed_ensemble = self.ClassifierEnsemble.ensembles[clf_name]
            sampled = self._sample_seeds(seed_ensemble.seeds)

            if self.verbose:
                logger.info(
                    f"[{clf_name}] DALEX on {len(sampled)} / "
                    f"{len(seed_ensemble.seeds)} seeds"
                )

            self.shap_results.setdefault(clf_name, {})
            self.vi_results.setdefault(clf_name, {})
            self.pdp_results.setdefault(clf_name, {})

            for seed_record in sampled:
                self._compute_seed(
                    clf_name=clf_name,
                    seed_record=seed_record,
                    plot_local=plot_local,
                    plot_global=plot_global,
                    plot_profile=plot_profile,
                )

            self._aggregate_and_persist(clf_name)

        return self

    def _compute_seed(
        self,
        clf_name: str,
        seed_record: dict,
        plot_local: bool,
        plot_global: bool,
        plot_profile: bool,
    ) -> None:
        """Run SHAP / VI / PDP for a single seed and persist per-seed artefacts.

        Each of the three DALEX calls is wrapped in a ``try``/``except`` so
        that a single failure does not abort the whole loop — the failure is
        logged at WARNING level and the loop continues.

        Args:
            clf_name (str): Classifier name (e.g. ``"RandomForestClassifier"``).
            seed_record (dict): A single record from ``SeedEnsemble.seeds``
                with ``random_state``, ``model``, and ``feature_names`` keys.
            plot_local (bool): Save the SHAP HTML plot.
            plot_global (bool): Save the variable-importance HTML plot.
            plot_profile (bool): Save per-feature PDP HTML plots.
        """
        random_state = int(seed_record["random_state"])
        estimator = seed_record["model"]
        feature_names = list(seed_record["feature_names"])

        missing = [f for f in feature_names if f not in self.features_df.columns]
        if missing:
            logger.warning(
                f"[{clf_name} seed {random_state:05d}] Skipping — "
                f"{len(missing)} required features missing from features_df: "
                f"{missing[:3]}..."
            )
            return

        X_seed = self.features_df[feature_names]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                explainer = dx.Explainer(
                    model=estimator,
                    data=X_seed,
                    y=self.y_true,
                    label=f"{clf_name}_{random_state:05d}",
                    model_type="classification",
                    predict_function=_predict_proba_positive,
                    verbose=False,
                )
            except Exception as exc:
                logger.warning(
                    f"[{clf_name} seed {random_state:05d}] Explainer construction "
                    f"failed: {exc}"
                )
                return

        self._run_shap(
            clf_name=clf_name,
            random_state=random_state,
            explainer=explainer,
            X_seed=X_seed,
            estimator=estimator,
            plot_local=plot_local,
        )
        self._run_variable_importance(
            clf_name=clf_name,
            random_state=random_state,
            explainer=explainer,
            plot_global=plot_global,
        )
        self._run_partial_dependence(
            clf_name=clf_name,
            random_state=random_state,
            explainer=explainer,
            X_seed=X_seed,
            feature_names=feature_names,
            plot_profile=plot_profile,
        )

    def _run_shap(
        self,
        clf_name: str,
        random_state: int,
        explainer: dx.Explainer,
        X_seed: pd.DataFrame,
        estimator,
        plot_local: bool,
    ) -> None:
        """SHAP local attribution for each selected observation.

        ``dalex.Explainer.predict_parts(type='shap')`` accepts one observation
        per call, so this iterates over the selection from
        :meth:`_select_observations`, concatenates the per-row results into
        a single DataFrame tagged with ``observation_index``, and writes one
        HTML plot per row.

        Args:
            clf_name (str): Classifier name.
            random_state (int): Seed identifier.
            explainer (dx.Explainer): Constructed DALEX explainer.
            X_seed (pd.DataFrame): Feature matrix sliced to the seed's columns.
            estimator: Fitted seed estimator (used to rank observations when
                positive-class rows are scarce).
            plot_local (bool): Persist the SHAP HTML plot when ``True``.
        """
        seeds_dir = os.path.join(self.classifiers_dir, clf_name, "shap", "seeds")
        observations = self._select_observations(X_seed, estimator)
        if observations.empty:
            return

        per_obs_frames: list[pd.DataFrame] = []
        for obs_pos in range(len(observations)):
            obs_row = observations.iloc[[obs_pos]]
            row_label = str(observations.index[obs_pos])
            try:
                result = explainer.predict_parts(
                    new_observation=obs_row,
                    type="shap",
                    B=self.shap_B,
                    random_state=random_state,
                )
            except Exception as exc:
                logger.warning(
                    f"[{clf_name} seed {random_state:05d} obs {row_label}] "
                    f"SHAP predict_parts failed: {exc}"
                )
                continue

            tagged = result.result.copy()
            tagged["observation_index"] = row_label
            per_obs_frames.append(tagged)

            if plot_local:
                try:
                    fig = result.plot(show=False)
                except Exception as exc:
                    logger.warning(
                        f"[{clf_name} seed {random_state:05d} obs {row_label}] "
                        f"SHAP plot failed: {exc}"
                    )
                    continue
                self._save_plotly_html(
                    fig=fig,
                    path=os.path.join(seeds_dir, f"{random_state:05d}_obs_{row_label}"),
                )

        if not per_obs_frames:
            return

        combined = pd.concat(per_obs_frames, axis=0, ignore_index=True)
        self.shap_results[clf_name][random_state] = combined
        save_data(combined, os.path.join(seeds_dir, f"{random_state:05d}"))

    def _run_variable_importance(
        self,
        clf_name: str,
        random_state: int,
        explainer: dx.Explainer,
        plot_global: bool,
    ) -> None:
        """Permutation-based global variable importance.

        Args:
            clf_name (str): Classifier name.
            random_state (int): Seed identifier.
            explainer (dx.Explainer): Constructed DALEX explainer.
            plot_global (bool): Persist the VI HTML plot when ``True``.
        """
        seeds_dir = os.path.join(
            self.classifiers_dir, clf_name, "variable_importance", "seeds"
        )
        try:
            result = explainer.model_parts(
                loss_function="1-auc",
                type="variable_importance",
                B=self.permutation_B,
                random_state=random_state,
            )
        except Exception as exc:
            logger.warning(
                f"[{clf_name} seed {random_state:05d}] model_parts failed: {exc}"
            )
            return

        self.vi_results[clf_name][random_state] = result.result.copy()
        save_data(result.result, os.path.join(seeds_dir, f"{random_state:05d}"))
        if plot_global:
            self._save_plotly_html(
                fig=result.plot(show=False),
                path=os.path.join(seeds_dir, f"{random_state:05d}"),
            )

    def _run_partial_dependence(
        self,
        clf_name: str,
        random_state: int,
        explainer: dx.Explainer,
        X_seed: pd.DataFrame,
        feature_names: list[str],
        plot_profile: bool,
    ) -> None:
        """Partial Dependence Profiles for the top-``k`` features.

        Top features are selected from the matching VI result when available;
        otherwise the first ``top_k_features`` from ``feature_names`` are used.

        Args:
            clf_name (str): Classifier name.
            random_state (int): Seed identifier.
            explainer (dx.Explainer): Constructed DALEX explainer.
            X_seed (pd.DataFrame): Feature matrix used to size the PDP grid.
            feature_names (list[str]): Full per-seed feature list (fallback
                source when VI ranking is unavailable).
            plot_profile (bool): Persist per-feature PDP HTML plots when ``True``.
        """
        top_features = self._pick_top_k_features(
            self.vi_results.get(clf_name, {}).get(random_state),
            feature_names,
        )
        try:
            result = explainer.model_profile(
                type="partial",
                variables=top_features,
                N=min(300, len(X_seed)),
                random_state=random_state,
                verbose=False,
            )
        except Exception as exc:
            logger.warning(
                f"[{clf_name} seed {random_state:05d}] model_profile failed: {exc}"
            )
            return

        self.pdp_results[clf_name][random_state] = {
            "result": result.result.copy(),
            "variables": top_features,
        }
        save_data(
            result.result,
            os.path.join(
                self.classifiers_dir,
                clf_name,
                "partial_dependence",
                "seeds",
                f"{random_state:05d}",
            ),
        )
        if plot_profile:
            for feature in top_features:
                try:
                    fig = result.plot(variables=[feature], show=False)
                except Exception as exc:
                    logger.warning(
                        f"[{clf_name} seed {random_state:05d}] PDP plot for "
                        f"{feature} failed: {exc}"
                    )
                    continue
                self._save_plotly_html(
                    fig=fig,
                    path=os.path.join(
                        self.classifiers_dir,
                        clf_name,
                        "partial_dependence",
                        slugify(feature, hyphen="_") or "feature",
                        "seeds",
                        f"{random_state:05d}",
                    ),
                )

    def _pick_top_k_features(
        self,
        vi_df: pd.DataFrame | None,
        feature_names: list[str],
    ) -> list[str]:
        """Rank features by mean dropout loss, dropping DALEX sentinel rows.

        DALEX's ``model_parts`` result includes ``_baseline_`` and
        ``_full_model_`` rows that must be filtered out before ranking.

        Args:
            vi_df (pd.DataFrame | None): The seed's ``model_parts`` result, or
                ``None`` when no VI result is available.
            feature_names (list[str]): The seed's full feature list, used as
                a fallback when ``vi_df`` is missing or malformed.

        Returns:
            list[str]: Up to ``top_k_features`` feature names.
        """
        if vi_df is None or "variable" not in vi_df.columns:
            return feature_names[: self.top_k_features]

        try:
            sentinels = {"_baseline_", "_full_model_"}
            ranked = (
                vi_df[
                    vi_df["variable"].isin(feature_names)
                    & ~vi_df["variable"].isin(sentinels)
                ]
                .groupby("variable")["dropout_loss"]
                .mean()
                .sort_values(ascending=False)
            )
            top = ranked.head(self.top_k_features).index.tolist()
            return top if top else feature_names[: self.top_k_features]
        except Exception:
            return feature_names[: self.top_k_features]

    def _aggregate_and_persist(self, clf_name: str) -> None:
        """Concatenate per-seed results and write the aggregate CSVs.

        For SHAP and VI, every per-seed DataFrame is tagged with its
        ``random_state`` and concatenated. The VI aggregate additionally
        produces a mean ± std table grouped by ``variable`` for quick
        ranking. PDP per-seed CSVs are already on disk; no aggregate is
        materialised today because cross-seed PDP averaging requires
        interpolating onto a shared grid.

        Args:
            clf_name (str): Classifier name to aggregate.
        """
        if self.shap_results.get(clf_name):
            frames = []
            for rs, df in self.shap_results[clf_name].items():
                tagged = df.copy()
                tagged["random_state"] = rs
                frames.append(tagged)
            agg = pd.concat(frames, axis=0, ignore_index=True)
            save_data(
                agg,
                os.path.join(self.classifiers_dir, clf_name, "shap", "aggregate"),
            )

        if self.vi_results.get(clf_name):
            frames = []
            for rs, df in self.vi_results[clf_name].items():
                tagged = df.copy()
                tagged["random_state"] = rs
                frames.append(tagged)
            agg = pd.concat(frames, axis=0, ignore_index=True)
            sentinels = {"_baseline_", "_full_model_"}
            mean_vi = (
                agg[~agg["variable"].isin(sentinels)]
                .groupby("variable")["dropout_loss"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("mean", ascending=False)
            )
            save_data(
                agg,
                os.path.join(
                    self.classifiers_dir,
                    clf_name,
                    "variable_importance",
                    "aggregate_long",
                ),
            )
            save_data(
                mean_vi,
                os.path.join(
                    self.classifiers_dir,
                    clf_name,
                    "variable_importance",
                    "aggregate",
                ),
            )

    @staticmethod
    def _save_plotly_html(fig, path: str) -> str:
        """Persist a Plotly figure as standalone HTML.

        Uses ``include_plotlyjs="cdn"`` so the file stays small and renders
        in any browser without bundled dependencies. Bypasses the
        ``kaleido``/Chromium pipeline entirely.

        Args:
            fig: Plotly figure returned by ``result.plot(show=False)``.
            path (str): Destination path WITHOUT a file extension.

        Returns:
            str: The full ``.html`` path the figure was written to.
        """
        full_path = f"{path}.html"
        ensure_dir(os.path.dirname(full_path))
        fig.write_html(full_path, include_plotlyjs="cdn")
        return full_path

    def save(self, path: str | None = None) -> str:
        """Persist the full instance to a single ``.pkl`` via joblib.

        Round-trips through :meth:`load` so the populated ``*_results`` dicts
        survive without re-running DALEX.

        Args:
            path (str | None): Destination ``.pkl`` path. ``None`` resolves to
                ``{self.output_dir}/DalexExplainerEnsemble.pkl``. Defaults to
                ``None``.

        Returns:
            str: The path the instance was written to.
        """
        if path is None:
            ensure_dir(self.output_dir)
            path = os.path.join(self.output_dir, "DalexExplainerEnsemble.pkl")
        else:
            parent = os.path.dirname(path)
            if parent:
                ensure_dir(parent)
        joblib.dump(self, path)
        if self.verbose:
            logger.info(f"Saved DalexExplainerEnsemble to {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "DalexExplainerEnsemble":
        """Reconstitute a saved instance from a joblib ``.pkl``.

        Args:
            path (str): Path to a ``.pkl`` previously written by :meth:`save`.

        Returns:
            DalexExplainerEnsemble: The deserialised instance.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            TypeError: If the loaded object is not a ``DalexExplainerEnsemble``.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"DalexExplainerEnsemble file not found: {path}")
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object is not a DalexExplainerEnsemble "
                f"(got {type(obj).__name__})."
            )
        return obj
