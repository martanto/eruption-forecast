import os
from typing import Any

import pandas as pd
from tsfresh import (
    extract_features as tsfresh_extract_features,
    extract_relevant_features,
)
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute

from eruption_forecast.utils import (
    concat_features as utils_concat_features,
    validate_columns,
)
from eruption_forecast.logger import logger
from eruption_forecast.features.constants import (
    ID_COLUMN,
    ERUPTED_COLUMN,
    DATETIME_COLUMN,
)


class FeaturesBuilder:
    """Extract tsfresh features from a pre-built tremor matrix.

    Accepts a tremor matrix produced by :class:`TremorMatrixBuilder` (one row
    per time sample, grouped by window ``id``) and runs tsfresh feature
    extraction on each tremor metric column independently.  Results are saved
    as individual CSVs and then concatenated into a single feature matrix
    ready for :class:`ModelTrainer`.

    Two operating modes are supported:

    * **Training mode** (``label_df`` provided): filters windows to those
      present in the label DataFrame and saves a matching label CSV alongside
      the feature CSV.
    * **Prediction mode** (``label_df`` omitted / ``None``): extracts all
      features without label filtering; ``use_relevant_features`` is forced
      to ``False``.

    Args:
        tremor_matrix_df (pd.DataFrame): Tremor matrix produced by
            ``TremorMatrixBuilder.build()``.  Must contain ``id`` and
            ``datetime`` columns plus one or more tremor metric columns
            (``rsam_*``, ``dsar_*``).
        output_dir (str | None, optional): Directory for saved CSVs.
            Defaults to ``<cwd>/output/features``.
        label_df (pd.DataFrame | None, optional): Label DataFrame with a
            ``DatetimeIndex`` and columns ``id`` and ``is_erupted``.
            Pass ``None`` (default) for prediction mode.
        overwrite (bool, optional): If ``True``, re-extract even when output
            files already exist. Defaults to ``False``.
        n_jobs (int, optional): Parallel jobs passed to tsfresh. Defaults to 1.
        verbose (bool, optional): Emit progress log messages. Defaults to ``False``.
        debug (bool, optional): Emit debug log messages. Defaults to ``False``.

    Raises:
        ValueError: If ``tremor_matrix_df`` or ``label_df`` is not a
            ``pd.DataFrame``, or if required columns (``id``, ``datetime``,
            ``is_erupted``) are missing.

    Example:
        >>> tremor_matrix_df = TremorMatrixBuilder(...).build().df
        >>> label_df = LabelBuilder(...).build().df
        >>> builder = FeaturesBuilder(
        ...     tremor_matrix_df=tremor_matrix_df,
        ...     label_df=label_df,
        ...     output_dir="output/features",
        ... )
        >>> features_df = builder.extract_features(
        ...     select_tremor_columns=["rsam_f0", "rsam_f1", "dsar_f0-f1"],
        ... )
        >>> print(features_df.shape)
        (100, 1500)  # 100 windows × ~750 features per column
    """

    def __init__(
        self,
        tremor_matrix_df: pd.DataFrame,
        output_dir: str | None = None,
        label_df: pd.DataFrame | None = None,
        overwrite: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        # =========================
        # Set DEFAULT parameter
        # =========================
        output_dir = output_dir or os.path.join(os.getcwd(), "output", "features")
        label_df = pd.DataFrame() if label_df is None else label_df

        # =========================
        # Set DEFAULT properties
        # =========================
        self.tremor_matrix_df = tremor_matrix_df
        self.output_dir = output_dir
        self.label_df = label_df
        self.overwrite = overwrite
        self.n_jobs = n_jobs
        self.verbose = verbose

        # =========================
        # Set ADDITIONAL properties (derived values)
        # =========================
        # Initialize feature parameters
        self.default_fc_parameters, self.excludes_features = (
            self._initialize_feature_parameters()
        )

        # =========================
        # Will be set after extract_features() called
        # =========================
        self.use_relevant_features: bool = False
        self.all_features_csvs: set[str] = set()
        self.relevant_features_csvs: set[str] = set()
        self.csv: str | None = None
        self.df: pd.DataFrame = pd.DataFrame()
        self.label_features_csv: str | None = None
        self.unique_ids: list[int] = []

        # =========================
        # Will be set after concat_features() called
        # =========================
        self.all_features_csv: str | None = None
        self.relevant_features_csv: str | None = None
        self.df_all_features: pd.DataFrame = pd.DataFrame()
        self.df_relevant_features: pd.DataFrame = pd.DataFrame()

        # =========================
        # Validate and create directories
        # =========================
        self.validate()

        # =========================
        # Verbose and logging
        # =========================
        if debug:
            logger.info("⚠️ Debug mode is ON")

    def validate(self) -> None:
        """Validate input DataFrame types and required column presence.

        Checks that both ``tremor_matrix_df`` and ``label_df`` are
        ``pd.DataFrame`` instances, that the tremor matrix contains ``id``
        and ``datetime`` columns, and (when a non-empty label DataFrame is
        provided) that it contains ``id`` and ``is_erupted`` columns.

        Raises:
            ValueError: If either DataFrame is not a ``pd.DataFrame``, or if
                required columns are missing.

        Returns:
            None
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

        # If label df is not empty, it must have "id" and "is_erupted" columns.
        if not self.label_df.empty:
            validate_columns(self.label_df, [ID_COLUMN, ERUPTED_COLUMN])

        return None

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
            "has_duplicate_max",
            "has_duplicate_min",
            "has_duplicate",
            "linear_trend_timewise",
            "length",
            "sum_of_reoccurring_data_points",
            "sum_of_reoccurring_values",
            "value_count",
        }

        return default_fc_parameters, excludes_features

    def exclude_features(
        self, excludes_features: list[str]
    ) -> ComprehensiveFCParameters:
        """Exclude specific features from tsfresh feature calculation.

        Updates the internal set of excluded features and removes them from
        the default feature calculation parameters.

        Args:
            excludes_features (list[str]): List of tsfresh feature names to
                exclude from calculation (e.g., ["length", "has_duplicate"]).

        Returns:
            ComprehensiveFCParameters: Updated tsfresh feature calculation
                parameters with specified features removed.

        Example:
            >>> builder = FeaturesBuilder(...)
            >>> params = builder.exclude_features(["length", "has_duplicate"])
            >>> # Features will be extracted without these calculators
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
        y: pd.DataFrame,
        extract_params: dict[str, Any],
        use_relevant_features: bool,
        prefix_filename: str,
        extract_features_dir: str,
    ) -> str:
        """Extract features for a single tremor column.

        Performs tsfresh feature extraction for one column, either using
        all features or only relevant features based on correlation with labels.

        Args:
            tremor_matrix_df: Features dataframe with id, datetime, and tremor columns
            column_method: Column method to extract features from. Example: rsam_f0, etc
            y: Target labels. If empty, extract all features.
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
        if use_relevant_features and not y.empty:
            # Prepare target labels: Series with window ID as index
            y_index = y[ID_COLUMN]
            y = y[ERUPTED_COLUMN]
            y.index = y_index

            # Extract Relevant Features
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
        """Concatenate extracted features from multiple CSV files.

        Merges per-column feature CSVs produced by
        :meth:`_extract_features_for_column` into a single unified features
        DataFrame and saves it to disk.  Also updates the relevant
        ``self.all_features_csv`` / ``self.relevant_features_csv`` attributes
        depending on the current :attr:`use_relevant_features` flag.

        Args:
            csv_list (list[str]): Paths to per-column extracted feature CSVs.
                Must contain at least 2 files.
            filename (str): Output filename stem (without ``.csv`` extension).

        Returns:
            tuple[str, pd.DataFrame]: ``(filepath, features_df)`` where
                *filepath* is the path of the saved concatenated CSV and
                *features_df* is the resulting DataFrame.

        Raises:
            FileNotFoundError: If ``csv_list`` is empty.
            ValueError: If ``csv_list`` contains fewer than 2 files (raised by
                the underlying :func:`concat_features` utility).

        Example:
            >>> builder = FeaturesBuilder(...)
            >>> csv_files = ["features_rsam_f0.csv", "features_rsam_f1.csv"]
            >>> filepath, df = builder.concat_features(csv_files, "all_features")
            >>> print(df.shape)
            (100, 1500)  # 100 windows, 1500 features
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
        prefix_filename: str | None = None,
    ) -> pd.DataFrame:
        """Extract time-series features from the tremor matrix using tsfresh.

        Iterates over each tremor metric column, runs tsfresh feature
        extraction independently, saves per-column CSVs to
        ``<output_dir>/extracted/``, then concatenates all columns into a
        single feature DataFrame saved in ``output_dir``.

        When ``label_df`` was supplied at construction time, the method also
        saves a filtered label CSV (``label_features_<dates>.csv``) that is
        aligned to the extracted windows and stores its path in
        :attr:`label_features_csv` for downstream use by :class:`ModelTrainer`.

        If no ``label_df`` was provided, ``use_relevant_features`` is
        automatically forced to ``False``.

        Args:
            use_relevant_features (bool, optional): If ``True``, uses
                ``tsfresh.extract_relevant_features()`` to keep only features
                with a statistically significant correlation to eruption
                labels.  Requires ``label_df`` to be non-empty; silently
                falls back to ``False`` otherwise.  Defaults to ``False``.
            select_tremor_columns (list[str] | None, optional): Restrict
                extraction to this subset of tremor columns
                (e.g. ``["rsam_f0", "dsar_f0-f1"]``).  If ``None``, all
                tremor columns in the matrix are used.  Defaults to ``None``.
            exclude_features (list[str] | None, optional): tsfresh calculator
                names to skip (e.g. ``["length", "has_duplicate"]``).
                Defaults to ``None``.
            prefix_filename (str | None, optional): Additional prefix prepended
                to the auto-generated filename
                (``[all|relevant]_features_<dates>``).  Defaults to ``None``.

        Returns:
            pd.DataFrame: Feature matrix of shape ``(n_windows, n_features)``
                with window IDs as index.  Also stored as :attr:`df` and
                :attr:`csv`.

        Raises:
            ValueError: If ``select_tremor_columns`` contains column names
                that do not exist in ``tremor_matrix_df``.

        Example:
            >>> builder = FeaturesBuilder(
            ...     tremor_matrix_df=tremor_matrix_df,
            ...     label_df=label_df,
            ...     output_dir="output/features",
            ... )
            >>> features_df = builder.extract_features(
            ...     select_tremor_columns=["rsam_f0", "rsam_f1"],
            ...     exclude_features=["length"],
            ... )
            >>> print(features_df.shape)
            (100, 1500)  # 100 windows × ~750 features per column
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
        self.unique_ids: list[int] = tremor_matrix_df[ID_COLUMN].unique().tolist()

        # Override use_relevant_features value to False if label not provided.
        # Calculate relevant features need label to process.
        if label_df.empty:
            logger.info(
                "No labels provided. Using relevant features will be disabled. "
                "All features will be extracted."
            )
            use_relevant_features = False
            tremor_matrix_dates = (
                tremor_matrix_df[DATETIME_COLUMN].sort_values().unique().tolist()
            )
            start_date_str = tremor_matrix_dates[0].strftime("%Y-%m-%d")
            end_date_str = tremor_matrix_dates[-1].strftime("%Y-%m-%d")
            dates_str = f"{start_date_str}-{end_date_str}"
        else:
            label_df = label_df[label_df[ID_COLUMN].isin(self.unique_ids)]
            start_date_str = label_df.index[0].strftime("%Y-%m-%d")
            end_date_str = label_df.index[-1].strftime("%Y-%m-%d")
            dates_str = f"{start_date_str}-{end_date_str}"

            # Save label CSV after success extraction
            label_csv = os.path.join(
                self.output_dir,
                f"label_features_{start_date_str}-{end_date_str}.csv",
            )
            label_df.to_csv(label_csv, index=True)

            # Set variables that will be used for model training
            self.label_features_csv = label_csv

        # Get params for features extraction
        extract_params = self._prepare_extraction_parameters(exclude_features)

        # Setup extraction directory
        extract_features_dir = os.path.join(self.output_dir, "extracted")
        os.makedirs(extract_features_dir, exist_ok=True)
        _prefix_filename = (
            f"relevant_features_{dates_str}"
            if use_relevant_features
            else f"all_features_{dates_str}"
        )
        prefix_filename = (
            f"{_prefix_filename}"
            if prefix_filename is None
            else f"{prefix_filename}_{_prefix_filename}"
        )

        if use_relevant_features and self.verbose:
            logger.info("Extracting features using relevant features")

        # Extract features for each column
        extracted_csvs = set()
        for column_method in tremor_matrix_df.columns.tolist():
            if column_method in [ID_COLUMN, DATETIME_COLUMN]:
                continue

            extracted_csv_path = self._extract_features_for_column(
                tremor_matrix_df=tremor_matrix_df,
                column_method=column_method,
                y=label_df,
                extract_params=extract_params,
                use_relevant_features=use_relevant_features,
                prefix_filename=prefix_filename,
                extract_features_dir=extract_features_dir,
            )

            if extracted_csv_path:
                extracted_csvs.add(extracted_csv_path)

        # Update tracked CSVs
        if use_relevant_features:
            self.relevant_features_csvs.update(extracted_csvs)
        else:
            self.all_features_csvs.update(extracted_csvs)

        self.csv, self.df = self.concat_features(
            csv_list=list(extracted_csvs), filename=prefix_filename
        )

        # Update value for self.use_relevant_features
        self.use_relevant_features = use_relevant_features

        return self.df
