import os
from typing import Self, Literal
from datetime import datetime

import numpy as np
import joblib
import pandas as pd

from eruption_forecast.utils.ml import build_y_true, compute_seed
from eruption_forecast.utils.pathutils import resolve_output_dir
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


class MetricsEnsemble:
    def __init__(
        self,
        classifier_ensemble: ClassifierEnsemble,
        features_df: pd.DataFrame,
        y_true: pd.Series | np.ndarray,
        kind: Literal["prediction", "training"] = "prediction",
        basename: str | None = None,
        overwrite: bool = False,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        output_dir = resolve_output_dir(
            output_dir=output_dir,
            root_dir=root_dir,
            default_subpath=os.path.join("output", "evaluation"),
        )

        output_dir = os.path.join(output_dir, kind)

        if isinstance(y_true, pd.Series):
            y_true = y_true.to_numpy()

        self.ClassifierEnsemble = classifier_ensemble
        self.features_df = features_df
        self.y_true: np.ndarray = y_true
        self.n_jobs = n_jobs
        self.output_dir = output_dir
        self.classifiers_dir = os.path.join(output_dir, "classifiers")
        self.basename = basename
        self.overwrite = overwrite
        self.verbose = verbose

        self.metrics: dict[str, pd.DataFrame] = {}

    @classmethod
    def from_file(
        cls,
        model_filepath: str,
        features_csv: str,
        features_label_csv: str,
        eruption_dates: list[str] | list[datetime],
        kind: Literal["prediction", "training"] = "prediction",
        basename: str | None = None,
        overwrite: bool = False,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> "MetricsEnsemble":
        for label, path in (
            ("Model Filepath", model_filepath),
            ("Features CSV", features_csv),
            ("Features Label CSV", features_label_csv),
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{label} file not found: {path}")

        classifier_ensemble = ClassifierEnsemble.from_any(
            model_filepath, verbose=verbose
        )
        features_df = pd.read_csv(features_csv, index_col=0)
        y_true = build_y_true(features_label_csv, eruption_dates)

        return cls(
            classifier_ensemble=classifier_ensemble,
            features_df=features_df,
            y_true=y_true,
            kind=kind,
            basename=basename,
            overwrite=overwrite,
            output_dir=output_dir,
            root_dir=root_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def compute(self) -> Self:
        y_true = self.y_true
        if isinstance(y_true, np.ndarray):
            y_true = pd.Series(
                np.asarray(y_true), index=self.features_df.index, name="is_erupted"
            )

        common_index = self.features_df.index.intersection(y_true.index)
        if len(common_index) == 0:
            raise ValueError(
                "features_df and y_true have no overlapping index entries. "
                "Check your features_df.index and y_true.index."
            )

        features_df = self.features_df.loc[common_index]

        y_true = y_true.loc[common_index].astype(int).to_numpy()

        self.metrics = self._compute(X=features_df, y_true=y_true)

        return self

    def _compute(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
    ) -> dict[str, pd.DataFrame]:
        """Handling parallel job to calculate metrics.

        Args:
            X (pd.DataFrame): Extracted features DataFrame of shape
                ``(n_samples, n_features)``.
            y_true (np.ndarray | pd.Series): Ground-truth binary labels
                aligned positionally with ``X``. Length must equal
                ``n_samples``.
        """
        classifier_names = list(self.ClassifierEnsemble.ensembles.keys())
        seed_ensembles = list(self.ClassifierEnsemble.ensembles.values())

        results = joblib.Parallel(n_jobs=self.n_jobs, backend="loky")(
            joblib.delayed(compute_seed)(
                seed_ensemble=seed_ensemble,
                X=X,
                y_true=y_true,
                output_dir=self.classifiers_dir,
                overwrite=self.overwrite,
            )
            for seed_ensemble in seed_ensembles
        )

        if len(results) == 0:
            raise ValueError(f"No metrics were calculated. For {classifier_names}")

        return dict(zip(classifier_names, results, strict=True))
