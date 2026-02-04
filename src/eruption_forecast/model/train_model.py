# Standard library imports
import os
from multiprocessing import Pool

# Third party imports
import pandas as pd

# Project imports
from eruption_forecast.features.constants import (
    DATETIME_COLUMN,
    ERUPTED_COLUMN,
    ID_COLUMN,
    SIGNIFICANT_FEATURES_FILENAME,
)
from eruption_forecast.logger import logger
from eruption_forecast.plot import plot_significant_features as plot_significant
from eruption_forecast.utils import (
    get_significant_features,
    random_under_sampler,
)


class TrainModel:
    """Train feature-selection models over multiple random seeds.

    Loads pre-extracted features and labels, runs randomised under-sampling
    followed by tsfresh feature selection for each seed, and collects the
    top-N significant features per seed into a single output CSV.

    Args:
        features_csv (str): Path to the extracted features CSV file.
        label_csv (str): Path to the label CSV file.
        output_dir (str, optional): Directory for output files. Defaults to
            ``<cwd>/output/trainings``.
        overwrite (bool, optional): Overwrite existing output files.
            Defaults to False.
        n_jobs (int, optional): Number of parallel workers. Defaults to 1.
        verbose (bool, optional): Verbose logging. Defaults to False.
        debug (bool, optional): Debug mode. Defaults to False.

    Raises:
        ValueError: If features or labels are empty, or their lengths
            do not match.
    """

    def __init__(
        self,
        features_csv: str,
        label_csv: str,
        output_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        # Set DEFAULT parameter
        df_features = pd.read_csv(features_csv, index_col=0)
        df_labels = pd.read_csv(label_csv)
        if ID_COLUMN in df_labels.columns:
            df_labels = df_labels.set_index(ID_COLUMN)

        if DATETIME_COLUMN in df_labels.columns:
            df_labels = df_labels.drop(DATETIME_COLUMN, axis=1)

        df_labels = df_labels[ERUPTED_COLUMN]
        output_dir = output_dir or os.path.join(os.getcwd(), "output", "trainings")
        training_dir = output_dir or os.path.join(
            os.getcwd(), "output", "models", "trainings"
        )

        significant_features_dir = os.path.join(output_dir, "significant_features")
        all_features_dir = os.path.join(output_dir, "all_features")

        figures_dir = os.path.join(output_dir, "figures")
        all_figures_dir = os.path.join(figures_dir, "all_figures")

        # Set DEFAULT properties
        self.df_features = df_features
        self.df_labels = df_labels
        self.n_jobs = int(n_jobs)
        self.output_dir = output_dir
        self.verbose = verbose
        self.debug = debug

        # Set ADDITIONAL properties
        self.significant_features_dir = significant_features_dir
        self.training_dir = training_dir
        self.all_features_dir = all_features_dir
        self.figures_dir = figures_dir
        self.all_figures_dir = all_figures_dir
        self.csvs: list[str] = []
        self.df_significant_features: pd.DataFrame = pd.DataFrame()

        # Validate and create directories
        self.validate()
        self.create_directories()

    def validate(self) -> None:
        """Validate that features and labels are non-empty and aligned.

        Raises:
            ValueError: If features or labels are empty, or their row
                counts do not match.
        """
        len_features = self.df_features.shape[0]
        len_labels = self.df_labels.shape[0]
        logger.debug(
            f"Validating: {len_features} feature rows, {len_labels} label rows"
        )

        if len_features == 0:
            raise ValueError("Features cannot be empty. Check your features CSV file.")
        if len_labels == 0:
            raise ValueError("Labels cannot be empty. Check your labels CSV file.")
        if len_features != len_labels:
            raise ValueError(
                f"Length of features and labels do not match. "
                f"Length of features: {len_features}, labels: {len_labels}. "
                f"Features CSV should be located under output/_nslc_/features, "
                f"with filename starting with extracted_features_(start_date)_(end_date).csv or "
                f"extracted_relevant_(start_date)_(end_date).csv"
            )

    def create_directories(self) -> None:
        """Create required output directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.training_dir, exist_ok=True)
        os.makedirs(self.significant_features_dir, exist_ok=True)

    def concat_significant_features(
        self, number_of_significant_features: int | None = None
    ) -> pd.DataFrame:
        """Concatenate significant features.

        Args:
            number_of_significant_features (int, optional): Number of significant features.

        Returns:
            pd.DataFrame: Significant features.
        """
        if len(self.csvs) == 0:
            raise ValueError(
                f"No significant features CSV file inside directory {self.significant_features_dir}"
            )

        df = pd.concat([pd.read_csv(file) for file in self.csvs], ignore_index=True)

        if df.empty:
            raise ValueError("No data found inside csv files.")

        df.to_csv(
            os.path.join(self.output_dir, SIGNIFICANT_FEATURES_FILENAME), index=False
        )

        if (
            number_of_significant_features is not None
            and number_of_significant_features > 0
        ):
            pass

        return df

    def _train(
        self,
        seed: int,
        random_state: int,
        number_of_significant_features: int = 20,
        sampling_strategy: str | float = 0.75,
        overwrite: bool = False,
        save_features: bool = False,
        plot_features: bool = False,
    ) -> str:
        state = random_state + seed
        logger.debug(f"_train: seed={seed}, random_state={random_state}, state={state}")

        significant_filepath = os.path.join(
            self.significant_features_dir, f"{state:05d}.csv"
        )
        all_features_filepath = os.path.join(self.all_features_dir, f"{state:05d}.csv")
        all_figures_filepath = os.path.join(self.all_figures_dir, f"{state:05d}.jpg")

        # Skip if files already exist
        can_skip = (
            not overwrite
            and os.path.isfile(significant_filepath)
            and (not save_features or os.path.isfile(all_features_filepath))
            and (not plot_features or os.path.isfile(all_figures_filepath))
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
                plot_significant(
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
        sampling_strategy: str | float = 0.75,
        save_all_features: bool = False,
        plot_significant_features: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Train the model.

        Args:
            random_state (int, optional): Initial random state. Defaults to 0.
            total_seed (int, optional): Total random state. Defaults to 500.
            number_of_significant_features (int, optional): Number of significant features.
            sampling_strategy (str, optional): Sampling strategy. Defaults to 0.75.
            save_all_features (bool, optional): Whether to save all features. Defaults to False.
            plot_significant_features (bool, optional): Whether to plot significant features. Defaults to False.
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.


        """
        if save_all_features:
            os.makedirs(self.all_features_dir, exist_ok=True)

        if plot_significant_features:
            os.makedirs(self.all_figures_dir, exist_ok=True)

        jobs = [
            (
                seed,
                random_state,
                number_of_significant_features,
                sampling_strategy,
                overwrite,
                save_all_features,
                plot_significant_features,
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
