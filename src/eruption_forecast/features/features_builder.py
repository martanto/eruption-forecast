# Standard library imports
import os
from typing import Any

# Third party imports
import pandas as pd
from tsfresh import extract_features as tsfresh_extract_features
from tsfresh import (
    extract_relevant_features,
)
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute

# Project imports
from eruption_forecast.features.constants import (
    DATETIME_COLUMN,
    ERUPTED_COLUMN,
    ID_COLUMN,
)
from eruption_forecast.logger import logger
from eruption_forecast.utils import concat_features as utils_concat_features
from eruption_forecast.utils import validate_columns


class FeaturesBuilder:
    """Builds feature matrices from tremor and label data.

    Slices tremor time-series into fixed-size windows aligned with label
    windows, then concatenates them into a single feature matrix suitable
    for downstream model training with tsfresh.

    The feature matrix contains one row per time sample within each window,
    with columns for window ID, datetime, and tremor metrics (RSAM/DSAR).

    Args:
        tremor_matrix_df (pd.DataFrame): Tremor dataframe with DatetimeIndex.
        label_df (pd.DataFrame): Label dataframe with DatetimeIndex and
            columns 'id' and 'is_erupted'.
        output_dir (str): Output directory path for saved CSVs.
        overwrite (bool, optional): Overwrite existing output files.
            Defaults to False.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 1.
        verbose (bool, optional): Verbose logging. Defaults to False.
        debug (bool, optional): Debug mode. Defaults to False.

    Raises:
        TypeError: If tremor_df or label_df index is not a DatetimeIndex.
        ValueError: If required label columns are missing or requested
            tremor columns do not exist.

    Example:
        >>> import pandas as pd
        >>> # Prepare tremor data (10-minute intervals)
        >>> tremor_df = pd.read_csv("tremor.csv", index_col=0, parse_dates=True)
        >>> label_df = pd.read_csv("labels.csv", index_col=0, parse_dates=True)
        >>> builder = FeaturesBuilder(
        ...     tremor_df=tremor_df,
        ...     label_df=label_df,
        ...     output_dir="output/features",
        ...     window_size=1,  # 1-day windows
        ...     select_tremor_columns=["rsam_f0", "rsam_f1", "dsar_f0-f1"],
        ... )
        >>> features_matrix = builder.extract_features(save_per_method=True)
        >>> print(features_matrix.shape)
        (14400, 5)  # 100 windows × 144 samples/day, 5 columns
    """

    def __init__(
        self,
        tremor_matrix_df: pd.DataFrame,
        label_df: pd.DataFrame,
        output_dir: str | None = None,
        overwrite: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        # Set DEFAULT parameter

        # Set DEFAULT properties
        self.tremor_matrix_df = tremor_matrix_df
        self.label_df = label_df
        self.output_dir = output_dir or os.path.join(os.getcwd(), "output", "features")
        self.overwrite = overwrite
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Set ADDITIONAL properties
        # Initialize feature parameters
        self.default_fc_parameters, self.excludes_features = (
            self._initialize_feature_parameters()
        )

        # Will be set after extract_features() called
        self.use_relevant_features: bool = False
        self.all_features_csvs: set[str] = set()
        self.relevant_features_csvs: set[str] = set()
        self.csv: str | None = None
        self.df: pd.DataFrame = pd.DataFrame()
        self.label_features_csv: str | None = None
        self.unique_ids: list[int] = []

        # Will be set after concat_features() called
        self.all_features_csv: str | None = None
        self.relevant_features_csv: str | None = None
        self.df_all_features: pd.DataFrame = pd.DataFrame()
        self.df_relevant_features: pd.DataFrame = pd.DataFrame()

        # Validate
        self.validate()

        # Verbose and debugging
        if debug:
            logger.info("⚠️ Debug mode is ON")

    def validate(self) -> None:
        """Validate tremor matrix columns and date ranges.

        Raises:
            ValueError: If required label columns are missing or requested
                tremor columns do not exist in the tremor dataframe.
        """
        if not isinstance(self.tremor_matrix_df, pd.DataFrame):
            raise ValueError(
                f"Tremor matrix must be pd.DataFrame. "
                f"Your tremor matrix type is {type(self.tremor_matrix_df)}"
            )
        if not isinstance(self.label_df, pd.DataFrame):
            raise ValueError(
                f"Label dataframe must be pd.DataFrame. "
                f"Your label data type is {type(self.tremor_matrix_df)}"
            )
        validate_columns(self.tremor_matrix_df, [ID_COLUMN, DATETIME_COLUMN])
        validate_columns(self.label_df, [ID_COLUMN, ERUPTED_COLUMN])

    @staticmethod
    def _initialize_feature_parameters() -> tuple[ComprehensiveFCParameters, set[str]]:
        """Initialize tsfresh feature extraction parameters with defaults.

        Sets up default feature calculators from tsfresh's ComprehensiveFCParameters
        and defines a set of features to exclude from calculation.

        Returns:
            Tuple of (default_fc_parameters, excludes_features)
        """
        default_fc_parameters = ComprehensiveFCParameters()
        excludes_features: set[str] = {
            "agg_linear_trend",
            "linear_trend_timewise",
            "length",
            "has_duplicate_max",
            "has_duplicate_min",
            "has_duplicate",
        }

        return default_fc_parameters, excludes_features

    def exclude_features(
        self, excludes_features: list[str]
    ) -> ComprehensiveFCParameters:
        """Exclude features from calulation.

        Args:
            excludes_features (list[str]): List of features to exclude from calculation.

        Returns:
            ComprehensiveFCParameters: tsfresh ComprehensiveFCParameters

        """
        default_fc_parameters = self.default_fc_parameters
        self.excludes_features.update(excludes_features)

        if len(self.excludes_features) > 0:
            default_fc_parameters_data = default_fc_parameters.data
            for feature in self.excludes_features:
                if feature in list(default_fc_parameters_data.keys()):
                    default_fc_parameters.pop(feature)

        self.default_fc_parameters = default_fc_parameters
        return default_fc_parameters

    def _prepare_extraction_parameters(
        self,
        exclude_features: list[str] | bool | None,
    ) -> dict[str, Any]:
        """Prepare parameters for tsfresh feature extraction.

        Handles feature exclusion logic and builds the parameter dictionary
        for tsfresh feature extraction functions.

        Args:
            exclude_features: Features to exclude from calculation

        Returns:
            Dictionary of extraction parameters for tsfresh
        """
        # Handle feature exclusion
        default_fc_parameters = self.default_fc_parameters

        if exclude_features is not None:
            if isinstance(exclude_features, list):
                default_fc_parameters = self.exclude_features(exclude_features)
            elif isinstance(exclude_features, bool) and not exclude_features:
                self.excludes_features = set()

        # Build extraction parameters
        return {
            "column_id": ID_COLUMN,
            "column_sort": DATETIME_COLUMN,
            "n_jobs": self.n_jobs,
            "default_fc_parameters": default_fc_parameters,
        }

    def _extract_features_for_column(
        self,
        tremor_matrix_df: pd.DataFrame,
        column_method: str,
        y: pd.Series,
        extract_params: dict[str, Any],
        use_relevant_features: bool,
        prefix_filename: str,
        extract_features_dir: str,
    ) -> str | None:
        """Extract features for a single tremor column.

        Performs tsfresh feature extraction for one column, either using
        all features or only relevant features based on correlation with labels.

        Args:
            tremor_matrix_df: Features dataframe with id, datetime, and tremor columns
            column_method: Column method to extract features from. Example: rsam_f0, etc
            y: Target labels
            extract_params: Parameters for tsfresh extraction
            use_relevant_features: Whether to use relevant features only
            prefix_filename: Prefix for output filename
            extract_features_dir: Directory to save extracted features

        Returns:
            Path to extracted features CSV, or None if skipped
        """
        extracted_csv = os.path.join(
            extract_features_dir, f"{prefix_filename}_{column_method}.csv"
        )

        # Skip if already exists and not overwriting
        if not self.overwrite and os.path.exists(extracted_csv):
            if self.verbose:
                logger.info(
                    f"Extracted features for {column_method} already exist: {extracted_csv}"
                )
            return extracted_csv

        # Prepare data for extraction
        df = tremor_matrix_df[[ID_COLUMN, DATETIME_COLUMN, column_method]]

        if self.verbose:
            logger.info(f"Extracting features for {column_method}")

        # Extract features
        if use_relevant_features:
            extracted_features = extract_relevant_features(df, y, **extract_params)
        else:
            extracted_features = tsfresh_extract_features(
                df,
                impute_function=impute,
                **extract_params,
            )

        # Save to CSV
        extracted_features.index.name = ID_COLUMN
        extracted_features.to_csv(extracted_csv, index=True)

        logger.info(f"Extracted features for {column_method} saved: {extracted_csv}")

        return extracted_csv

    def concat_features(
        self, csv_list: list[str], filename: str
    ) -> tuple[str, pd.DataFrame]:
        """Concatenate features from calculation.

        Args:
            csv_list (list[str]): List of csv files to concat.
            filename: Prefix for output filename

        Returns:
             filepath (str) : CSV output file
             extracted_df (pd.DataFrame) : Features extracted from calculation
        """
        if len(csv_list) == 0:
            raise FileNotFoundError("Extracted features CSV not found.")

        if self.verbose:
            logger.info("Concatenating extracted features from calculation.")

        filepath = os.path.join(self.output_dir, f"{filename}.csv")
        features_csv, features_df = utils_concat_features(list(csv_list), filepath)

        if self.use_relevant_features:
            self.relevant_features_csv = features_csv
            self.df_relevant_features = features_df
        else:
            self.all_features_csv = features_csv
            self.df_all_features = features_df

        self.csv = features_csv
        self.df = features_df

        return features_csv, features_df

    def extract_features(
        self,
        use_relevant_features: bool = False,
        select_tremor_columns: list[str] | None = None,
        exclude_features: list[str] | None = None,
    ) -> pd.DataFrame:
        """Extract features from tremor data using tsfresh.

        Applies time-series feature extraction to each tremor column, either
        extracting all comprehensive features or only statistically relevant
        features based on correlation with eruption labels.

        Args:
            select_tremor_columns (list[str], optional): Subset of tremor columns to
                use. If None, all columns are used. Defaults to None.
            exclude_features (Optional[list[str]]): List features calculator to be excluded.
            use_relevant_features (bool): If True, extract features using relevant features.
        """
        label_df = self.label_df

        # Select column matrix
        tremor_matrix_df = self.tremor_matrix_df
        if select_tremor_columns is not None:
            validate_columns(tremor_matrix_df, select_tremor_columns)
            tremor_matrix_df = self.tremor_matrix_df[
                [ID_COLUMN, DATETIME_COLUMN, *select_tremor_columns]
            ]

        # Get labels based on unique IDs from tremor matrix
        unique_ids: list[int] = tremor_matrix_df[ID_COLUMN].unique().tolist()
        label_df = label_df[label_df[ID_COLUMN].isin(unique_ids)]

        start_date_str = label_df.index[0].strftime("%Y-%m-%d")
        end_date_str = label_df.index[-1].strftime("%Y-%m-%d")
        dates_str = f"{start_date_str}-{end_date_str}"

        # Setup extraction directory
        extract_features_dir = os.path.join(self.output_dir, "extracted")
        os.makedirs(extract_features_dir, exist_ok=True)
        prefix_filename = (
            f"relevant_features_{dates_str}"
            if use_relevant_features
            else f"all_features_{dates_str}"
        )

        if use_relevant_features and self.verbose:
            self.use_relevant_features = use_relevant_features
            logger.info("Extracting features using relevant features")

        # Prepare extraction parameters
        extract_params = self._prepare_extraction_parameters(exclude_features)

        # Prepare target labels
        y = label_df[ERUPTED_COLUMN]
        y.index = label_df[ID_COLUMN]

        # Extract features for each column
        extracted_csvs = set()
        for column_method in tremor_matrix_df.columns.tolist():
            if column_method in [ID_COLUMN, DATETIME_COLUMN]:
                continue

            csv_path = self._extract_features_for_column(
                tremor_matrix_df=tremor_matrix_df,
                column_method=column_method,
                y=y,
                extract_params=extract_params,
                use_relevant_features=use_relevant_features,
                prefix_filename=prefix_filename,
                extract_features_dir=extract_features_dir,
            )

            if csv_path:
                extracted_csvs.add(csv_path)

        # Update tracked CSVs
        if use_relevant_features:
            self.relevant_features_csvs.update(extracted_csvs)
        else:
            self.all_features_csvs.update(extracted_csvs)

        self.csv, self.df = self.concat_features(
            csv_list=list(extracted_csvs), filename=prefix_filename
        )

        # Save label CSV after success extraction
        label_csv = os.path.join(
            self.output_dir,
            f"label_features_{start_date_str}-{end_date_str}.csv",
        )
        label_df.to_csv(label_csv, index=True)

        # Set variable that will be use for TrainingLabel
        self.label_features_csv = label_csv
        self.unique_ids = unique_ids

        return self.df
