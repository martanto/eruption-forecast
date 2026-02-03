import pandas as pd
import os
from typing import Optional, Union, Self
from eruption_forecast.utils import random_under_sampler
from tsfresh.transformers import FeatureSelector


class TrainModel:
    def __init__(
        self,
        features_csv: str,
        label_csv: str,
        n_jobs: int = 1,
        random_state: int = 42,
        total_seed: int = 458,
        sampling_strategy: str = "auto",
        output_dir: Optional[str] = None,
        verbose: bool = False,
        debug: bool = False,
    ):
        # Set DEFAULT parameter
        df_features = pd.read_csv(features_csv, index_col=0)
        df_labels = pd.read_csv(label_csv, index_col=1)

        if "datetime" in df_labels.columns:
            df_labels = df_labels.drop("datetime", axis=1)

        df_labels = df_labels["is_erupted"]
        output_dir = output_dir or os.path.join(os.getcwd(), "output", "predict")

        # Set DEFAULT properties
        self.df_features = df_features
        self.df_label = df_labels
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)
        self.total_seed = int(total_seed)
        self.sampling_strategy = sampling_strategy
        self.output_dir = output_dir
        self.verbose = verbose
        self.debug = debug

        # Set ADDITIONAL properties
        self.seeds: list[int] = range(total_seed)

        # Validate
        self.validate()

    def validate(self):
        """Validate parameters."""
        len_features = self.df_features.shape[0]
        len_labels = self.df_label.shape[0]

        assert len_features > 0, ValueError(
            f"Features cannot be empty. Check your features CSV file,"
        )
        assert len_labels > 0, ValueError(
            f"Labels cannot be empty. Check your labels CSV file"
        )

        assert len_features == len_labels, ValueError(
            f"Length of features and labels do not match. "
            f"Length of features: {len_features}, labels: {len_labels}. "
            f"Features CSV should be located under output/_nslc_/features,"
            f"with filename starts with extracted_features_(start_date)_(end_date).csv, or"
            f"extracted_relevant_(start_date)_(end_date).csv"
        )

        assert self.random_state > 0, ValueError(f"Random seed must be greater than 0")
        assert self.total_seed > 0, ValueError(f"Total seed must be greater than 0")

        os.makedirs(self.output_dir, exist_ok=True)

    def build_model(self, sampling_strategy: Optional[Union[str, float]] = None):
        sampling_strategy = sampling_strategy or self.sampling_strategy

        for seed in self.seeds:
            random_state = self.random_state + seed

            # Balancing class using random under sampler
            features, labels = random_under_sampler(
                features=self.df_features,
                labels=self.df_label,
                sampling_strategy=sampling_strategy,
                random_state=random_state,
            )

            feature_selector = FeatureSelector(
                n_jobs=self.n_jobs,
                ml_task="classification",
            )

            selected_features = feature_selector.fit_transform(X=features, y=labels)  # type: ignore
