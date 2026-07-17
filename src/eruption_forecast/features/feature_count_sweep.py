"""Post-hoc feature-count sweep — pick optimal top-N by re-scoring existing seeds.

Reuses per-seed p-value rankings, resampled ids, and tuned hyperparameters
already written by :class:`~eruption_forecast.model.training_model.TrainingModel`,
then re-scores each seed at several candidate ``N`` values under one of two
modes:

- ``mode="forecast"`` (default) — refits each seed at candidate ``N`` on the
  trainer's already-resampled subset restricted to the top-``N`` features and
  scores against the prediction-window features + ground-truth ``y_true``
  carried by an :class:`~eruption_forecast.model.evaluation_model.EvaluationModel`
  in prediction-reuse mode. This matches the project's real forecasting
  evaluation. **Missing features in the prediction matrix raise ``KeyError``
  — never silently skipped.**
- ``mode="cv"`` — the sklearn ``RFECV``-analog: cross-validation *inside* the
  trainer's resampled subset. Fast proxy, but does not measure forecast-window
  generalisation.

The recommendation is the ``N`` with the highest mean score across seeds,
optionally broken toward smaller ``N`` when several values are within one
standard error of the peak (default) — the "highest score, fewest features"
tie-break.

Two entry points:

- :class:`FeatureCountSweep` — the general engine. Accepts any ranking source
  plus a per-seed estimator dict.
- :func:`sweep_feature_count` — the convenience wrapper that discovers the
  ranking, resampled ids, and tuned estimators from a completed training
  directory (or a live :class:`TrainingModel` instance), and — in forecast
  mode — pulls ``X_test`` / ``y_test`` from an ``EvaluationModel``.
"""

from __future__ import annotations

import os
import glob
import json
from typing import TYPE_CHECKING, Any, Self, Literal

import numpy as np
import joblib
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    BaseCrossValidator,
)

from eruption_forecast.logger import logger
from eruption_forecast.utils.ml import resample as _resample
from eruption_forecast.utils.dataframe import load_label_csv
from eruption_forecast.utils.pathutils import ensure_dir, load_pickle
from eruption_forecast.plots.feature_plots import plot_feature_count_curve


if TYPE_CHECKING:
    from eruption_forecast.model.training_model import TrainingModel
    from eruption_forecast.model.evaluation_model import EvaluationModel


DEFAULT_N_CANDIDATES: list[int] = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]


class FeatureCountSweep:
    """Sweep top-N over an existing feature ranking and pick the CV-optimal N.

    Does not re-rank features. Consumes rankings already produced by the
    per-seed FDR selector — either the aggregated ``top_features.csv``
    (strategy ``"shared"``) or the per-seed ``{seed:05d}.csv`` files
    (strategy ``"per-seed"``, default).

    For each candidate ``N`` and each seed, refits ``clone(seed_estimator)``
    inside a fresh CV splitter on the seed's resampled subset restricted to
    the top-``N`` features from that seed's ranking. The aggregated summary
    is exposed on :attr:`cv_scores_`; the full ``(N × seed)`` matrix is
    preserved on :attr:`cv_scores_raw_` for diagnostics.

    Args:
        estimator (BaseEstimator | None): Fallback estimator used when
            ``per_seed_inputs`` is not supplied to :meth:`fit`. Ignored in
            the post-hoc path — each seed brings its own tuned estimator
            via ``per_seed_inputs[seed]["estimator"]``.
        strategy (Literal["shared", "per-seed"]): Ranking source. Only
            meaningful for the standalone ``fit(X, y, shared_ranking=...)``
            or ``fit(X, y, per_seed_rankings=...)`` calls. Ignored in the
            post-hoc path (which always uses per-seed inputs). Defaults to
            ``"per-seed"``.
        n_candidates (list[int] | None): Candidate ``N`` values. Defaults
            to ``[5, 10, 15, 20, 25, 30, 40, 50, 75, 100]``.
        cv (BaseCrossValidator | int): CV splitter or fold count. Only used
            when ``mode="cv"``. Integer values dispatch to
            ``StratifiedKFold`` when labels are binary/multiclass, otherwise
            ``KFold``. Defaults to ``5``.
        scoring (str): sklearn scoring name. Defaults to
            ``"average_precision"`` (best for eruption-style imbalance).
        parsimony (bool): When ``True``, break ties among competitive
            ``N`` values by picking the **smallest**. Defaults to ``True``.
        parsimony_tolerance (float | None): Controls what "competitive"
            means. ``None`` → 1-SE rule (adaptive; tolerance = 1 standard
            error of the mean at the peak). A float in ``(0, 1)`` →
            fractional tolerance (``peak × (1 − t)``). Ignored when
            ``parsimony=False``. Defaults to ``None``.
        resample_method (Literal["under", "over", "auto"] | None): Applied
            inside each fold on the training split only. Mirrors
            :meth:`TrainingModel.fit`. Ignored in ``mode="forecast"`` — the
            per-seed ``ids`` already point at the trainer's resampled
            subset, so no further resampling is needed. Defaults to
            ``None``.
        minority_threshold (float): Threshold used when
            ``resample_method="auto"``. Defaults to ``0.15``.
        estimator_mode (Literal["default", "tuned"]): How the per-seed
            estimator is prepared before cloning into each fit.
            ``"default"`` (recommended) drops the tuned hyperparameters and
            uses the class defaults — matching sklearn ``RFECV`` /
            yellowbrick, so every candidate ``N`` is compared with the same
            base learner. ``"tuned"`` keeps the ``GridSearchCV``-picked
            params, which biases the curve toward the trainer's
            ``top_n_features`` because those params were selected at that
            specific ``N``. Only meaningful in the post-hoc path (see
            :func:`sweep_feature_count`); standalone modes carry whatever
            estimator the caller supplied. Defaults to ``"default"``.
        mode (Literal["cv", "forecast"]): Scoring strategy.
            ``"forecast"`` (default) refits each seed on the trainer's
            resampled subset restricted to the top-``N`` features and
            scores against the prediction-window ``X_test`` / ``y_test``
            passed to :meth:`fit` (typically sourced from an
            :class:`~eruption_forecast.model.evaluation_model.EvaluationModel`
            in prediction-reuse mode). This matches the project's real
            forecasting evaluation. Missing per-seed features in ``X_test``
            raise ``KeyError``. ``"cv"`` runs the sklearn
            ``RFECV``-analog: cross-validation *inside* the trainer's
            resampled subset — a fast proxy that does not measure
            forecast-window generalisation. Defaults to ``"forecast"``.
        n_jobs (int): Parallel workers. Applied over CV folds when
            ``mode="cv"``; unused when ``mode="forecast"`` (one fit per
            ``(seed, N)`` — parallelism is left to the trainer / scorer).
            Defaults to ``1``.
        random_state (int): Seed for CV splitters and resamplers.
            Defaults to ``42``.
        verbose (bool): Emit per-``N`` progress logs. Defaults to
            ``False``.

    Attributes:
        cv_scores_ (pd.DataFrame): Aggregated summary indexed by ``N``
            with columns ``mean``, ``std``, ``n_seeds``. Populated by
            :meth:`fit`. In ``mode="cv"`` each cell is the mean fold
            score; in ``mode="forecast"`` each cell is the held-out
            forecast-window score (single value per seed).
        cv_scores_raw_ (pd.DataFrame): Full ``(N × seed)`` score matrix.
            Kept so aggregation and diagnostics operate on the same data.
        n_features_ (int): Recommended ``N*``. See "Picking N*" in the
            design doc.
        seed_argmax_ (pd.Series): Per-seed argmax ``N`` — free
            diagnostic that reveals whether the shared-``N`` assumption
            holds.
        support_ (dict[int | str, list[str]]): For each seed, the
            top-``N*`` feature list from that seed's ranking.
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        *,
        strategy: Literal["shared", "per-seed"] = "per-seed",
        n_candidates: list[int] | None = None,
        cv: BaseCrossValidator | int = 5,
        scoring: str = "average_precision",
        parsimony: bool = True,
        parsimony_tolerance: float | None = None,
        resample_method: Literal["under", "over", "auto"] | None = None,
        minority_threshold: float = 0.15,
        estimator_mode: Literal["default", "tuned"] = "default",
        mode: Literal["cv", "forecast"] = "forecast",
        n_jobs: int = 1,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.estimator = estimator
        self.strategy = strategy
        self.n_candidates: list[int] = list(
            n_candidates if n_candidates is not None else DEFAULT_N_CANDIDATES
        )
        self.cv = cv
        self.scoring = scoring
        self.parsimony = parsimony
        self.parsimony_tolerance = parsimony_tolerance
        self.resample_method = resample_method
        self.minority_threshold = minority_threshold
        self.estimator_mode = estimator_mode
        self.mode = mode
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # Populated by fit()
        self.cv_scores_: pd.DataFrame = pd.DataFrame()
        self.cv_scores_raw_: pd.DataFrame = pd.DataFrame()
        self.n_features_: int | None = None
        self.seed_argmax_: pd.Series = pd.Series(dtype=int)
        self.support_: dict[int | str, list[str]] = {}
        self._classifier_name: str | None = None  # informational, set by wrapper

        self._validate_init()

    def _validate_init(self) -> None:
        """Validate constructor arguments early to fail fast."""
        if not self.n_candidates:
            raise ValueError("n_candidates must be non-empty.")
        if any(n < 1 for n in self.n_candidates):
            raise ValueError(f"n_candidates must be positive; got {self.n_candidates}.")
        if self.parsimony_tolerance is not None:
            if not 0.0 < self.parsimony_tolerance < 1.0:
                raise ValueError(
                    "parsimony_tolerance must be in (0, 1); "
                    f"got {self.parsimony_tolerance!r}."
                )
        if self.estimator_mode not in ("default", "tuned"):
            raise ValueError(
                'estimator_mode must be "default" or "tuned"; '
                f"got {self.estimator_mode!r}."
            )
        if self.mode not in ("cv", "forecast"):
            raise ValueError(
                'mode must be "cv" or "forecast"; '
                f"got {self.mode!r}."
            )
        if self.n_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1; got {self.n_jobs}.")

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        per_seed_inputs: dict[int, dict[str, Any]] | None = None,
        shared_ranking: pd.Series | pd.Index | list[str] | None = None,
        per_seed_rankings: dict[int, pd.Series | pd.Index | list[str]]
        | None = None,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
    ) -> Self:
        """Run the sweep and populate the result attributes.

        Three input modes are supported (mutually exclusive):

        1. ``per_seed_inputs`` — post-hoc mode used by
           :func:`sweep_feature_count`. A mapping
           ``{seed: {"ranking": list[str], "ids": pd.Index,
           "estimator": BaseEstimator}}`` where ``estimator`` is the tuned
           model from ``GridSearchCV`` (cloned per fold / per fit).
        2. ``per_seed_rankings`` — standalone strategy ``(B)``. A mapping
           ``{seed: ranking}``. The seed's rows are ``X`` in full; the
           shared ``self.estimator`` is cloned per fold / per fit.
        3. ``shared_ranking`` — standalone strategy ``(A)``. A single
           ranking used for every seed (there is only one "seed" in this
           mode). The shared ``self.estimator`` is cloned per fold / per fit.

        When ``self.mode == "forecast"``, ``X_test`` and ``y_test`` are
        required — the sweep refits each seed at each candidate ``N`` on
        the seed's training subset and scores against the held-out
        forecast window. Missing per-seed features in ``X_test`` raise
        ``KeyError`` — never silently skipped.

        Args:
            X (pd.DataFrame): Full training feature matrix. Must be
                id-indexed when ``per_seed_inputs`` is used so the ``ids``
                slice can resolve.
            y (pd.Series): Full training label series aligned to
                ``X.index``.
            per_seed_inputs (dict | None): Post-hoc per-seed payloads.
            shared_ranking (pd.Series | pd.Index | list[str] | None): Single
                ranking for the shared strategy.
            per_seed_rankings (dict | None): Per-seed rankings for strategy (B).
            X_test (pd.DataFrame | None): Held-out feature matrix used for
                scoring when ``self.mode == "forecast"``. Column set must
                cover every seed's top-``N`` selection. Ignored when
                ``self.mode == "cv"``.
            y_test (pd.Series | None): Ground-truth labels aligned to
                ``X_test.index``. Required when ``self.mode == "forecast"``.

        Returns:
            Self: The current instance, enabling method chaining.

        Raises:
            ValueError: If no input mode is provided, if the shared
                fallback estimator is missing when required, if the
                ranking iterators yield no usable seeds, or if
                ``self.mode == "forecast"`` and ``X_test`` / ``y_test``
                are missing.
            KeyError: In forecast mode, if a seed's top-``N`` ranking
                contains features absent from ``X_test.columns``.
        """
        seed_iter = self._build_seed_iter(
            X,
            per_seed_inputs=per_seed_inputs,
            shared_ranking=shared_ranking,
            per_seed_rankings=per_seed_rankings,
        )
        if not seed_iter:
            raise ValueError(
                "FeatureCountSweep.fit: no seeds to score — check the input mode."
            )

        if self.mode == "forecast":
            if X_test is None or y_test is None:
                raise ValueError(
                    'FeatureCountSweep.fit: mode="forecast" requires X_test '
                    "and y_test."
                )

        # Clamp candidates to the shortest available ranking so head(N)
        # never over-runs a seed.
        max_available = min(len(data["ranking"]) for data in seed_iter.values())
        candidates: list[int] = sorted({n for n in self.n_candidates if n <= max_available})
        clamped = sorted(set(self.n_candidates) - set(candidates))
        if clamped:
            logger.warning(
                f"FeatureCountSweep: clamped n_candidates {clamped} — shortest "
                f"per-seed ranking has {max_available} features."
            )
        if not candidates:
            raise ValueError(
                "FeatureCountSweep: every n_candidate exceeds the shortest ranking "
                f"({max_available} features). Reduce n_candidates or re-train "
                "with a larger top_n_features."
            )

        raw = pd.DataFrame(
            index=candidates,
            columns=list(seed_iter.keys()),
            dtype=float,
        )
        raw.index.name = "N"

        scorer = get_scorer(self.scoring) if self.mode == "forecast" else None

        for n_val in candidates:
            for seed_key, seed_data in seed_iter.items():
                ranking = seed_data["ranking"]
                ids = seed_data["ids"]
                tuned_est = seed_data["estimator"]

                features_n = list(ranking[:n_val])
                x_seed = X.loc[ids, features_n]
                y_seed = y.loc[ids]

                if self.mode == "forecast":
                    # Narrowing carried inline — the top-of-fit guard rejects
                    # None but ty does not propagate the check through the loop.
                    if X_test is None or y_test is None or scorer is None:
                        raise ValueError(
                            "FeatureCountSweep.fit: X_test / y_test missing "
                            'inside mode="forecast" loop — this is a bug.'
                        )
                    missing = [f for f in features_n if f not in X_test.columns]
                    if missing:
                        preview = missing[:5]
                        raise KeyError(
                            f"FeatureCountSweep[forecast]: seed {seed_key!r} at "
                            f"N={n_val} needs {len(missing)} feature(s) absent "
                            f"from X_test — first: {preview}"
                            + ("..." if len(missing) > 5 else "")
                        )
                    estimator = clone(tuned_est).fit(x_seed, y_seed)
                    raw.loc[n_val, seed_key] = float(
                        scorer(estimator, X_test[features_n], y_test)
                    )
                else:
                    cv_splitter = self._resolve_cv(y_seed)
                    fold_scores = Parallel(n_jobs=self.n_jobs, backend="loky")(
                        delayed(_score_fold)(
                            clone(tuned_est),
                            x_seed.iloc[tr],
                            y_seed.iloc[tr],
                            x_seed.iloc[te],
                            y_seed.iloc[te],
                            self.scoring,
                            self.resample_method,
                            self.minority_threshold,
                            self.random_state,
                        )
                        for tr, te in cv_splitter.split(x_seed, y_seed)
                    )
                    raw.loc[n_val, seed_key] = float(np.mean(fold_scores))

            if self.verbose:
                logger.info(
                    f"FeatureCountSweep[{self.mode}]: N={n_val:3d}  "
                    f"mean={raw.loc[n_val].mean():.4f}  "
                    f"std={raw.loc[n_val].std():.4f}  "
                    f"seeds={raw.shape[1]}"
                )

        self.cv_scores_raw_ = raw
        self.cv_scores_ = pd.DataFrame(
            {
                "mean": raw.mean(axis=1),
                "std": raw.std(axis=1),
                "n_seeds": raw.count(axis=1),
            }
        )
        self.seed_argmax_ = raw.idxmax(axis=0).astype(int)
        self.n_features_ = _pick_best(
            self.cv_scores_,
            parsimony=self.parsimony,
            parsimony_tolerance=self.parsimony_tolerance,
        )
        self.support_ = {
            seed_key: list(seed_data["ranking"][: self.n_features_])
            for seed_key, seed_data in seed_iter.items()
        }
        return self

    def _build_seed_iter(
        self,
        X: pd.DataFrame,
        *,
        per_seed_inputs: dict[int, dict[str, Any]] | None,
        shared_ranking: pd.Series | pd.Index | list[str] | None,
        per_seed_rankings: dict[int, pd.Series | pd.Index | list[str]] | None,
    ) -> dict[int | str, dict[str, Any]]:
        """Normalise the three input modes into a single seed → payload dict.

        Payload shape is uniform downstream: ``{"ranking", "ids", "estimator"}``.
        """
        modes_provided = sum(
            src is not None
            for src in (per_seed_inputs, shared_ranking, per_seed_rankings)
        )
        if modes_provided == 0:
            raise ValueError(
                "FeatureCountSweep.fit: pass one of per_seed_inputs, "
                "per_seed_rankings, or shared_ranking."
            )
        if modes_provided > 1:
            raise ValueError(
                "FeatureCountSweep.fit: pass only one input mode at a time."
            )

        if per_seed_inputs is not None:
            return {int(k): v for k, v in per_seed_inputs.items()}

        if self.estimator is None:
            raise ValueError(
                "FeatureCountSweep.fit: standalone modes (shared_ranking / "
                "per_seed_rankings) require the constructor's `estimator` "
                "argument to be set."
            )

        if shared_ranking is not None:
            return {
                "__shared__": {
                    "ranking": _to_list(shared_ranking),
                    "ids": X.index,
                    "estimator": self.estimator,
                }
            }

        assert per_seed_rankings is not None  # narrowing for ty
        return {
            int(k): {
                "ranking": _to_list(r),
                "ids": X.index,
                "estimator": self.estimator,
            }
            for k, r in per_seed_rankings.items()
        }

    def _resolve_cv(self, y: pd.Series) -> BaseCrossValidator:
        """Turn ``self.cv`` into a concrete splitter.

        Integer values become ``StratifiedKFold(n_splits=self.cv)`` when the
        target has more than one class, else ``KFold(n_splits=self.cv)``.
        Splitter instances are returned unchanged.
        """
        if isinstance(self.cv, int):
            n_splits = self.cv
            is_multiclass = len(y.unique()) > 1
            if is_multiclass:
                return StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
            return KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
        return self.cv

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the fitted sweep to a single ``.pkl`` file via joblib.

        Args:
            path (str): Destination file path (should end with ``.pkl``).
        """
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        joblib.dump(self, path)
        logger.info(f"[FeatureCountSweep] Saved to: {path}")

    @classmethod
    def load(cls, path: str) -> Self:
        """Restore a previously-saved sweep from ``.pkl``."""
        obj = load_pickle(path)
        logger.info(f"[FeatureCountSweep] Loaded from: {path}")
        return obj


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _to_list(ranking: pd.Series | pd.Index | list[str]) -> list[str]:
    """Coerce any of Series / Index / list to a plain ``list[str]``."""
    if isinstance(ranking, (pd.Series, pd.Index)):
        return [str(x) for x in ranking]
    return list(ranking)


def _score_fold(
    estimator: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    scoring: str,
    resample_method: Literal["under", "over", "auto"] | None,
    minority_threshold: float,
    random_state: int,
) -> float:
    """Fit ``estimator`` on the (possibly resampled) train split and score on test.

    Resampling happens **inside** the fold on the training split only — never
    before the split — to avoid leaking test-row information into the
    resampler.
    """
    if resample_method == "auto":
        minority_share = y_train.value_counts(normalize=True).min()
        resample_method = "under" if minority_share <= minority_threshold else None

    if resample_method is not None:
        x_train, y_train = _resample(
            features=x_train,
            labels=y_train,
            method=resample_method,
            random_state=random_state,
        )

    estimator.fit(x_train, y_train)  # ty:ignore[unresolved-attribute]
    scorer = get_scorer(scoring)
    return float(scorer(estimator, x_test, y_test))


def _pick_best(
    cv_scores: pd.DataFrame,
    *,
    parsimony: bool = True,
    parsimony_tolerance: float | None = None,
) -> int:
    """Mean-across-seeds argmax, with optional tie-breaking toward smaller N.

    ``parsimony=False`` → strict argmax on the mean (no tie-break).
    ``parsimony=True, parsimony_tolerance=None`` → 1-SE rule (adaptive).
    ``parsimony=True, parsimony_tolerance=t`` → fractional tolerance
    ``peak × (1 − t)``.

    Args:
        cv_scores (pd.DataFrame): Indexed by ``N`` with columns
            ``mean``, ``std``, ``n_seeds``.
        parsimony (bool): Enable tie-breaking toward smaller ``N``.
        parsimony_tolerance (float | None): Fractional-tolerance override.

    Returns:
        int: Recommended ``N*``.

    Raises:
        ValueError: If ``parsimony_tolerance`` is outside ``(0, 1)``.
    """
    means = cv_scores["mean"].to_numpy(dtype=float)
    stds = cv_scores["std"].to_numpy(dtype=float)
    seeds = cv_scores["n_seeds"].to_numpy(dtype=float)
    n_values = cv_scores.index.to_numpy()

    peak_idx = int(np.argmax(means))
    peak_n = int(n_values[peak_idx])
    if not parsimony:
        return peak_n

    peak_mean = float(means[peak_idx])
    if parsimony_tolerance is None:
        peak_seeds = float(seeds[peak_idx])
        peak_std = float(stds[peak_idx])
        # Single-seed sweeps produce NaN std by pandas convention; treat the
        # 1-SE band as zero so the peak N is still selected.
        if peak_seeds > 1 and np.isfinite(peak_std):
            peak_sem = peak_std / np.sqrt(peak_seeds)
        else:
            peak_sem = 0.0
        threshold = peak_mean - peak_sem
    else:
        if not 0.0 < parsimony_tolerance < 1.0:
            raise ValueError(
                "parsimony_tolerance must be in (0, 1); "
                f"got {parsimony_tolerance!r}."
            )
        threshold = peak_mean * (1.0 - parsimony_tolerance)

    within = [int(n) for n, mean in zip(n_values, means, strict=True) if mean >= threshold]
    return min(within)


# ---------------------------------------------------------------------------
# sweep_feature_count — public entry point
# ---------------------------------------------------------------------------


def sweep_feature_count(
    source: str | os.PathLike | TrainingModel,
    *,
    mode: Literal["cv", "forecast"] = "forecast",
    evaluation_source: EvaluationModel | None = None,
    classifier_name: str | None = None,
    n_candidates: list[int] | None = None,
    cv: BaseCrossValidator | int = 5,
    scoring: str = "average_precision",
    parsimony: bool = True,
    parsimony_tolerance: float | None = None,
    resample_method: Literal["under", "over", "auto"] | None = None,
    minority_threshold: float = 0.15,
    estimator_mode: Literal["default", "tuned"] = "default",
    n_jobs: int = 1,
    random_state: int = 42,
    output_dir: str | None = None,
    save: bool = True,
    verbose: bool = False,
) -> FeatureCountSweep | dict[str, FeatureCountSweep]:
    """Recommend the optimal ``top_n_features`` by post-hoc CV re-scoring.

    Harvests per-seed p-value rankings, resampled ids, and tuned
    hyperparameters from a completed training run, then delegates to
    :class:`FeatureCountSweep`. Runs one bare ``clone(best_model).fit`` per
    ``(seed, N, fold)`` — no ``GridSearchCV``, no re-selection.

    ``source`` may be a path to an existing ``training/`` directory (cold
    flow — primary interface) or a fitted
    :class:`~eruption_forecast.model.training_model.TrainingModel` instance
    (live flow — convenience). Both dispatch through the same
    directory-based harvester internally.

    Args:
        source (str | os.PathLike | TrainingModel): Training directory or
            fitted trainer.
        mode (Literal["cv", "forecast"]): Scoring strategy. ``"forecast"``
            (default) refits each seed at each candidate ``N`` on the
            trainer's resampled subset restricted to the top-``N`` features
            and scores against ``evaluation_source.features_df`` /
            ``evaluation_source.y_true`` — the project's real forecasting
            evaluation. Missing per-seed features in the prediction matrix
            raise ``KeyError`` (never silently skipped). ``"cv"`` runs
            cross-validation *inside* the trainer's resampled subset
            (sklearn ``RFECV`` analog). Defaults to ``"forecast"``.
        evaluation_source (EvaluationModel | None): Held-out data source
            for ``mode="forecast"``. Must be an ``EvaluationModel`` in
            prediction-reuse mode (``model_kind == "prediction"``) with
            ``y_true`` already built (call
            ``EvaluationModel.evaluate()`` first, or invoke
            ``EvaluationModel.build_label(...)`` directly). Required when
            ``mode="forecast"``; ignored when ``mode="cv"``. Defaults to
            ``None``.
        classifier_name (str | None): Sweep this classifier only. When
            ``None``, sweep every classifier discovered under
            ``training/classifiers/`` and return a ``{name: sweep}`` dict.
        n_candidates (list[int] | None): Candidate ``N`` values. The
            discovered ``top_n_features`` is automatically inserted so the
            curve always includes the trained baseline. Defaults to
            ``[5, 10, 15, 20, 25, 30, 40, 50, 75, 100]``.
        cv (BaseCrossValidator | int): CV splitter or fold count.
        scoring (str): sklearn scoring name. Defaults to
            ``"average_precision"``.
        parsimony (bool): Enable smallest-N tie-break. Defaults to ``True``.
        parsimony_tolerance (float | None): Fractional-tolerance override
            for parsimony. Defaults to ``None`` (→ 1-SE rule).
        resample_method (Literal["under", "over", "auto"] | None): Applied
            inside each fold on the training split only. Defaults to
            ``None``. Recommend matching whatever the trainer used.
        minority_threshold (float): Threshold for ``resample_method="auto"``.
        estimator_mode (Literal["default", "tuned"]): How each seed's
            estimator is prepared. ``"default"`` (recommended) drops the
            ``GridSearchCV`` hyperparameters and uses the class defaults —
            matching sklearn's ``RFECV`` convention so every candidate ``N``
            is compared with the same untuned base learner. ``"tuned"``
            keeps the trainer's tuned params, which biases the curve
            toward the trained ``top_n_features``. Defaults to
            ``"default"``.
        n_jobs (int): Parallel workers over CV folds. Defaults to ``1``.
        random_state (int): Seed for CV splitters. Defaults to ``42``.
        output_dir (str | None): Destination root for artefacts. When
            ``None``, resolves to
            ``{training_dir}/features/{cv-slug}/sweep/{classifier-slug}/``.
        save (bool): Persist ``cv_scores.csv``, ``cv_scores_raw.csv``,
            ``seed_argmax_hist.csv``, ``support.json``, and
            ``FeatureCountSweep.pkl``. Defaults to ``True``.
        verbose (bool): Emit per-``N`` progress. Defaults to ``False``.

    Returns:
        FeatureCountSweep | dict[str, FeatureCountSweep]: A single sweep
            when ``classifier_name`` is given, else a dict keyed by the
            discovered classifier names.

    Raises:
        FileNotFoundError: When ``source`` is a missing directory.
        TypeError: When ``source`` is neither a path nor a
            :class:`TrainingModel`.
        RuntimeError: When no per-seed models can be discovered.
        ValueError: When ``mode="forecast"`` but ``evaluation_source`` is
            missing, points at a training-reuse evaluation, or has empty
            ``y_true`` / ``features_df``.
        KeyError: When a per-seed top-``N`` ranking contains features
            absent from the prediction feature matrix (forecast mode).
    """
    x_test, y_test = _resolve_forecast_inputs(mode, evaluation_source)

    training_dir, features_df, labels = _resolve_training_context(source)
    cv_slug, features_dir = _discover_cv_dir(training_dir)

    if features_df is None:
        features_df = _load_features_matrix(features_dir)
    if labels is None:
        labels = _load_labels(features_dir)

    features_seed_dir = os.path.join(features_dir, "seed")
    features_resampled_dir = os.path.join(features_dir, "resampled")

    trained_top_n = _discover_trained_top_n(features_seed_dir)

    if classifier_name is not None:
        classifier_names = [classifier_name]
    else:
        classifier_names = _discover_classifier_names(training_dir, cv_slug)

    if not classifier_names:
        raise RuntimeError(
            f"No classifiers found under {os.path.join(training_dir, 'classifiers')}."
        )

    # Merge trained top_n into the candidate list so the curve always shows
    # the baseline the operator actually trained at.
    base_candidates = list(n_candidates if n_candidates is not None else DEFAULT_N_CANDIDATES)
    if trained_top_n is not None and trained_top_n not in base_candidates:
        base_candidates.append(trained_top_n)
    base_candidates = sorted(set(base_candidates))

    results: dict[str, FeatureCountSweep] = {}
    for clf_name in classifier_names:
        per_seed_inputs = _harvest_per_seed_inputs(
            training_dir=training_dir,
            cv_slug=cv_slug,
            classifier_name=clf_name,
            features_seed_dir=features_seed_dir,
            features_resampled_dir=features_resampled_dir,
            estimator_mode=estimator_mode,
        )
        if not per_seed_inputs:
            logger.warning(
                f"sweep_feature_count[{clf_name}]: no seed models found — skipping."
            )
            continue

        sweep = FeatureCountSweep(
            n_candidates=base_candidates,
            cv=cv,
            scoring=scoring,
            parsimony=parsimony,
            parsimony_tolerance=parsimony_tolerance,
            resample_method=resample_method,
            minority_threshold=minority_threshold,
            estimator_mode=estimator_mode,
            mode=mode,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        sweep._classifier_name = clf_name
        sweep.fit(
            features_df,
            labels,
            per_seed_inputs=per_seed_inputs,
            X_test=x_test,
            y_test=y_test,
        )

        if save:
            resolved_output = output_dir or os.path.join(
                features_dir, "sweep", mode, clf_name
            )
            _persist_sweep(sweep, resolved_output)

        results[clf_name] = sweep

    if classifier_name is not None:
        return results[classifier_name]
    return results


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------


def _resolve_training_context(
    source: str | os.PathLike | TrainingModel,
) -> tuple[str, pd.DataFrame | None, pd.Series | None]:
    """Normalise ``source`` into ``(training_dir, features_df?, labels?)``.

    A live :class:`TrainingModel` may already carry an in-memory features
    matrix and labels; a path source defers loading to
    :func:`_load_features_matrix` / :func:`_load_labels`.
    """
    # Local import to avoid the cycle at module import time.
    from eruption_forecast.model.training_model import TrainingModel

    if isinstance(source, TrainingModel):
        training_dir = source.training_dir
        features_df = source.features_df if not source.features_df.empty else None
        labels = source.labels if not source.labels.empty else None
        return training_dir, features_df, labels

    if isinstance(source, (str, os.PathLike)):
        training_dir = os.fspath(source)
        if not os.path.isdir(training_dir):
            raise FileNotFoundError(f"Training directory not found: {training_dir}")
        return training_dir, None, None

    raise TypeError(
        f"source must be str, PathLike, or TrainingModel; got {type(source).__name__}."
    )


def _discover_cv_dir(training_dir: str) -> tuple[str, str]:
    """Find the sole CV-slug directory under ``training/features/``.

    Returns ``(cv_slug, absolute_features_cv_dir)``. Raises when the layout
    contains anything other than exactly one CV subdirectory.
    """
    features_root = os.path.join(training_dir, "features")
    if not os.path.isdir(features_root):
        raise FileNotFoundError(f"Missing features directory: {features_root}")
    entries = [e for e in os.listdir(features_root) if os.path.isdir(os.path.join(features_root, e))]
    if len(entries) != 1:
        raise RuntimeError(
            f"Expected exactly one CV subdirectory under {features_root}; "
            f"found {entries}. Multi-CV runs need an explicit selector."
        )
    cv_slug = entries[0]
    return cv_slug, os.path.join(features_root, cv_slug)


def _load_features_matrix(features_dir: str) -> pd.DataFrame:
    """Glob and read the single ``features-matrix_*.parquet`` under ``features_dir``."""
    matches = sorted(glob.glob(os.path.join(features_dir, "features-matrix_*.parquet")))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one features-matrix parquet under {features_dir}; "
            f"found {matches}."
        )
    return pd.read_parquet(matches[0])


def _load_labels(features_dir: str) -> pd.Series:
    """Glob and read the single ``features-label_*.csv`` under ``features_dir``."""
    matches = sorted(glob.glob(os.path.join(features_dir, "features-label_*.csv")))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one features-label csv under {features_dir}; "
            f"found {matches}."
        )
    return load_label_csv(matches[0])


def _discover_trained_top_n(features_seed_dir: str) -> int | None:
    """Peek at the first per-seed CSV to learn the trainer's ``top_n_features``.

    Returns ``None`` when the directory is empty (standalone runs without
    a full training pipeline behind them).
    """
    seed_csvs = sorted(glob.glob(os.path.join(features_seed_dir, "*.csv")))
    if not seed_csvs:
        return None
    return len(pd.read_csv(seed_csvs[0], index_col=0).index)


def _discover_classifier_names(training_dir: str, cv_slug: str) -> list[str]:
    """List classifier directories containing per-seed model pickles.

    Classifier layout: ``training/classifiers/{name}/{cv_slug}/models/``.
    Skips classifier directories that never produced a model.
    """
    classifiers_root = os.path.join(training_dir, "classifiers")
    if not os.path.isdir(classifiers_root):
        return []
    names: list[str] = []
    for entry in sorted(os.listdir(classifiers_root)):
        models_dir = os.path.join(classifiers_root, entry, cv_slug, "models")
        if os.path.isdir(models_dir) and any(
            f.endswith(".pkl") for f in os.listdir(models_dir)
        ):
            names.append(entry)
    return names


def _harvest_per_seed_inputs(
    *,
    training_dir: str,
    cv_slug: str,
    classifier_name: str,
    features_seed_dir: str,
    features_resampled_dir: str,
    estimator_mode: Literal["default", "tuned"] = "default",
) -> dict[int, dict[str, Any]]:
    """Read per-seed rankings + resampled ids + estimators into the payload dict.

    When ``estimator_mode == "default"`` (the recommended sklearn ``RFECV``
    convention), the tuned ``best_model.pkl`` is reset to a fresh instance
    of its own class so every candidate ``N`` is compared with the same
    untuned base learner. ``"tuned"`` keeps the ``GridSearchCV``-picked
    hyperparameters, which biases the sweep toward the trained
    ``top_n_features``.

    Any seed missing one of the three inputs is skipped with a warning —
    the sweep proceeds over surviving seeds.
    """
    models_dir = os.path.join(
        training_dir, "classifiers", classifier_name, cv_slug, "models"
    )
    if not os.path.isdir(models_dir):
        return {}

    payloads: dict[int, dict[str, Any]] = {}
    for model_path in sorted(glob.glob(os.path.join(models_dir, "*.pkl"))):
        seed_str = os.path.splitext(os.path.basename(model_path))[0]
        try:
            seed_int = int(seed_str)
        except ValueError:
            logger.warning(f"sweep: non-integer seed filename {model_path!r}, skipping.")
            continue

        ranking_path = os.path.join(features_seed_dir, f"{seed_str}.csv")
        resampled_path = os.path.join(features_resampled_dir, f"{seed_str}.csv")
        if not os.path.isfile(ranking_path):
            logger.warning(
                f"sweep[{classifier_name}][{seed_int:05d}]: ranking CSV missing, skipping."
            )
            continue
        if not os.path.isfile(resampled_path):
            logger.warning(
                f"sweep[{classifier_name}][{seed_int:05d}]: resampled CSV missing, skipping."
            )
            continue

        ranking = pd.read_csv(ranking_path, index_col=0).index.tolist()
        ids = pd.read_csv(resampled_path, index_col=0).index
        estimator = joblib.load(model_path)
        if estimator_mode == "default":
            estimator = estimator.__class__()
        payloads[seed_int] = {
            "ranking": ranking,
            "ids": ids,
            "estimator": estimator,
        }
    return payloads


def _resolve_forecast_inputs(
    mode: Literal["cv", "forecast"],
    evaluation_source: EvaluationModel | None,
) -> tuple[pd.DataFrame | None, pd.Series | None]:
    """Validate ``evaluation_source`` and extract ``(X_test, y_test)``.

    Returns ``(None, None)`` in ``mode="cv"``. In ``mode="forecast"`` the
    ``EvaluationModel`` must be in prediction-reuse mode with ``y_true``
    and ``features_df`` already populated — call
    ``EvaluationModel.evaluate()`` (or at minimum
    ``EvaluationModel.build_label(...)`` on prediction-reuse) before
    passing it in.
    """
    if mode == "cv":
        return None, None

    if evaluation_source is None:
        raise ValueError(
            'sweep_feature_count: mode="forecast" requires evaluation_source '
            "(an EvaluationModel in prediction-reuse mode)."
        )
    if evaluation_source.model_kind != "prediction":
        raise ValueError(
            "sweep_feature_count: evaluation_source must be in prediction-reuse "
            f"mode (got model_kind={evaluation_source.model_kind!r}). "
            "Training-reuse labels are the training labels themselves — they "
            "do not form a valid held-out test set."
        )
    if evaluation_source.y_true.empty:
        raise ValueError(
            "sweep_feature_count: evaluation_source.y_true is empty. Run "
            "EvaluationModel.evaluate() or EvaluationModel.build_label(...) "
            "before passing it in."
        )
    if evaluation_source.features_df.empty:
        raise ValueError(
            "sweep_feature_count: evaluation_source.features_df is empty. "
            "The upstream PredictionModel must have run extract_features() "
            "before evaluation."
        )
    return evaluation_source.features_df, evaluation_source.y_true


def _persist_sweep(sweep: FeatureCountSweep, output_dir: str) -> None:
    """Write the fitted sweep's artefacts under ``output_dir``.

    Produces:

    - ``cv_scores.csv`` — aggregated summary (N, mean, std, n_seeds).
    - ``cv_scores_raw.csv`` — full (N × seed) matrix.
    - ``seed_argmax_hist.csv`` — per-seed argmax N.
    - ``support.json`` — ``{"n_features": N*, "seeds": {seed: [features…]}}``.
    - ``FeatureCountSweep.pkl`` — the full instance via joblib.
    """
    ensure_dir(output_dir)
    sweep.cv_scores_.to_csv(os.path.join(output_dir, "cv_scores.csv"))
    sweep.cv_scores_raw_.to_csv(os.path.join(output_dir, "cv_scores_raw.csv"))
    sweep.seed_argmax_.rename("argmax_N").to_csv(
        os.path.join(output_dir, "seed_argmax_hist.csv"), header=True
    )
    support_payload = {
        "n_features": sweep.n_features_,
        "seeds": {str(k): v for k, v in sweep.support_.items()},
    }
    with open(os.path.join(output_dir, "support.json"), "w") as f:
        json.dump(support_payload, f, indent=2)

    if sweep.n_features_ is not None:
        try:
            plot_feature_count_curve(
                cv_scores=sweep.cv_scores_,
                seed_argmax=sweep.seed_argmax_,
                n_features_star=sweep.n_features_,
                filepath=os.path.join(output_dir, "curve.png"),
                title=(
                    f"Feature-count sweep [{sweep.mode}] — {sweep._classifier_name}"
                    if sweep._classifier_name
                    else f"Feature-count sweep [{sweep.mode}]"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to render sweep curve: {exc}")

    sweep.save(os.path.join(output_dir, "FeatureCountSweep.pkl"))
