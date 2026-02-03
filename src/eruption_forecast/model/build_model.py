import pandas as pd
import os
from typing import Optional, Union
from eruption_forecast.utils import (
    random_under_sampler,
    get_significant_features,
)
from eruption_forecast.plot import plot_significant_features
from eruption_forecast.logger import logger
from multiprocessing import Pool


class TrainModel:
    def __init__(
        self,
        features_csv: str,
        label_csv: str,
        output_dir: Optional[str] = None,
        overwrite: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
        debug: bool = False,
    ):
        # Set DEFAULT parameter
        df_features = pd.read_csv(features_csv, index_col=0)
        df_labels = pd.read_csv(label_csv, index_col=1)

        if "datetime" in df_labels.columns:
            df_labels = df_labels.drop("datetime", axis=1)

        df_labels = df_labels["is_erupted"]
        output_dir = output_dir or os.path.join(os.getcwd(), "output", "predictions")
        significant_features_dir = os.path.join(output_dir, "significant_features")
        all_features_dir = os.path.join(output_dir, "all_features")

        figures_dir = os.path.join(output_dir, "figures")
        all_figures_dir = os.path.join(figures_dir, "all_figures")

        # Set DEFAULT properties
        self.df_features = df_features
        self.df_labels = df_labels
        self.n_jobs = int(n_jobs)
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.verbose = verbose
        self.debug = debug

        # Set ADDITIONAL properties
        self.significant_features_dir = significant_features_dir
        self.all_features_dir = all_features_dir
        self.figures_dir = figures_dir
        self.all_figures_dir = all_figures_dir
        self.csvs: list[str] = []
        self.df_significant_features: pd.DataFrame = pd.DataFrame()

        # Validate
        self.validate()

    def validate(self):
        """Validate parameters."""
        len_features = self.df_features.shape[0]
        len_labels = self.df_labels.shape[0]

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

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.significant_features_dir, exist_ok=True)

    def concat_significant_features(self) -> pd.DataFrame:
        """Concatenate significant features.

        Returns:
            pd.DataFrame: Significant features.
        """
        if len(self.csvs) == 0:
            raise ValueError(
                f"No significant features CSV file inside directory {self.significant_features_dir}"
            )

        df = pd.concat([pd.read_csv(file) for file in self.csvs], ignore_index=True)

        if df.empty:
            raise ValueError(f"No data found inside csv files.")

        df.groupby(by="features").count()

        df.to_csv(
            os.path.join(self.output_dir, "significant_features.csv"), index=False
        )

        return df

    def _train(
        self,
        seed: int,
        random_state: int,
        number_of_significant_features: int = 20,
        sampling_strategy: Union[str, float] = 0.75,
        overwrite: bool = False,
        save_features: bool = False,
        plot_features: bool = False,
    ):
        state = random_state + seed

        significant_filepath = os.path.join(
            self.significant_features_dir, f"{state:05d}.csv"
        )
        all_features_filepath = os.path.join(self.all_features_dir, f"{state:05d}.csv")
        all_figures_filepath = os.path.join(self.all_figures_dir, f"{state:05d}.jpg")

        # Skip if files aready exists
        can_skip = (
            not overwrite
            and os.path.isfile(significant_filepath)
            and (save_features or os.path.isfile(all_features_filepath))
            and (plot_features or os.path.isfile(all_figures_filepath))
        )

        if can_skip:
            logger.info(f"Features {seed:05d}.csv exists.")
            return significant_filepath

        logger.info(f"Training Seed: {seed:05d}")

        # Balancing class using random under sampler
        features, labels = random_under_sampler(
            features=self.df_features,
            labels=self.df_labels,
            sampling_strategy=sampling_strategy,
            random_state=state,
        )

        significant_features = get_significant_features(
            features=features,
            labels=labels,
        )

        significant_features.head(number_of_significant_features).to_csv(
            significant_filepath, index=True
        )

        if save_features:
            significant_features.to_csv(all_features_filepath, index=True)
            if plot_features:
                plot_significant_features(
                    df=pd.DataFrame(significant_features).reset_index(),
                    filepath=all_figures_filepath,
                    overwrite=overwrite,
                    dpi=72,
                )

        if self.verbose:
            logger.info(f"Features {state:05d} saved to: {significant_filepath}")

        return significant_filepath

    def train(
        self,
        random_state: int = 0,
        total_seed: int = 500,
        number_of_significant_features: int = 20,
        sampling_strategy: Union[str, float] = 0.75,
        save_features: bool = False,
        plot_features: bool = False,
        overwrite: bool = False,
        verbose: bool = False,
    ):
        overwrite = overwrite or self.overwrite

        if save_features:
            os.makedirs(self.all_features_dir, exist_ok=True)

        if plot_features:
            os.makedirs(self.all_figures_dir, exist_ok=True)

        jobs = [
            (
                seed,
                random_state,
                number_of_significant_features,
                sampling_strategy,
                overwrite,
                save_features,
                plot_features,
            )
            for seed in range(total_seed)
        ]

        if self.n_jobs == 1:
            for job in jobs:
                csv = self._train(*job)
                self.csvs.append(csv)

        if self.n_jobs > 1:
            logger.info(f"Running on {self.n_jobs} job(s)")
            with Pool(self.n_jobs) as pool:
                csvs = pool.starmap(self._train, jobs)
                self.csvs.extend(csvs)

        self.df_significant_features = self.concat_significant_features()

        return None
