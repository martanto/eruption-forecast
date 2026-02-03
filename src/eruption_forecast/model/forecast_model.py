# Standard library imports
from typing import Any
import os
from datetime import datetime, timedelta
from typing import Literal, Optional, Self, Union

# Third party imports
import pandas as pd
from tsfresh import extract_features as tsfresh_extract_features
from tsfresh import (
    extract_relevant_features,
)
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute

# Project imports
from eruption_forecast.features.features_builder import FeaturesBuilder
from eruption_forecast.label.label_builder import LabelBuilder
from eruption_forecast.logger import logger
from eruption_forecast.tremor.calculate_tremor import CalculateTremor
from eruption_forecast.tremor.tremor_data import TremorData
from eruption_forecast.utils import (
    to_datetime,
    validate_columns,
    validate_date_ranges,
    construct_windows,
    concat_features as utils_concat_features,
)


class ForecastModel:
    """Create forecast model from seismic data.

    CalculateTremor: Calculate Tremor data from seismic data.
    Build Label and extract features for training.
    Predict based on training data.

    Args:
        station (str): Seismic station code.
        channel (str): Seismic channel code.
        start_date (str | datetime): Start date in YYYY-MM-DD format.
        end_date (str | datetime): End date in YYYY-MM-DD format.
        window_size (int): Window size in days. Used to create label and training data.
        volcano_id (str): Volcano ID. To set and forecast ID.
        network (str): Seismic network code. Defaults to "VG".
        location (str): Seismic location code. Defaults to "00".
        output_dir (str): Directory for output files. Defaults to "output".
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
        n_jobs (int): Number of parallel jobs to use. Defaults to 1.
        verbose (bool): If True, enables verbose logging. Defaults to False.
        debug (bool): If True, enables debug mode. Defaults to False.
    """

    def __init__(
        self,
        station: str,
        channel: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        window_size: int,
        volcano_id: str,
        network: str = "VG",
        location: str = "00",
        output_dir: Optional[str] = None,
        overwrite: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
        debug: bool = False,
    ):
        # Normalize dates
        start_date, end_date, start_date_str, end_date_str = self._normalize_dates(
            start_date, end_date
        )

        # Setup directories
        network = network or "VG"
        location = location or "00"
        nslc, output_dir, station_dir, features_dir = self._setup_directories(
            network, station, location, channel, output_dir
        )

        # Set DEFAULT properties (core parameters)
        self.station = station
        self.channel = channel
        self.start_date: datetime = start_date
        self.end_date: datetime = end_date
        self.window_size: int = window_size
        self.volcano_id = volcano_id
        self.network = network
        self.location = location
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.debug = debug

        # Set ADDITIONAL properties (derived values)
        self.start_date_minus_window_size = start_date - timedelta(days=window_size)
        self.start_date_str = start_date_str
        self.end_date_str = end_date_str
        self.nslc = nslc
        self.station_dir = station_dir
        self.features_dir = features_dir

        # Initialize feature parameters
        self.default_fc_parameters, self.excludes_features = (
            self._initialize_feature_parameters()
        )

        # Initialize state properties (set during lifecycle)
        # Will be set after calculate() method called
        self.CalculateTremor: Optional[CalculateTremor] = None
        self.TremorData: Optional[TremorData] = None
        self.tremor_data: pd.DataFrame = pd.DataFrame()
        self.tremor_csv: Optional[str] = None

        # Will be set after build_label() method called
        self.LabelBuilder: Optional[LabelBuilder] = None
        self.label_data: pd.DataFrame = pd.DataFrame()
        self.label_csv: Optional[str] = None
        self.total_eruption_class: Optional[int] = None
        self.total_non_eruption_class: Optional[int] = None

        # Will be set after build_features() called
        self.FeaturesBuilder: Optional[FeaturesBuilder] = None
        self.features_data: pd.DataFrame = pd.DataFrame()
        self.features_csv: Optional[str] = None

        # Will be set after extract_features() called
        self.extract_features_csvs: set[str] = set()
        self.relevant_features_csvs: set[str] = set()

        # Will be set after concat_features() called
        self.extracted_features_csv: Optional[str] = None
        self.extracted_relevant_csv: Optional[str] = None

        # Will be set after predict() called
        self.prediction_features_csvs: set[str] = set()

        # Base filename without extension
        self.basename: Optional[str] = None

        # Validate
        self.validate()

        # Verbose and debugging
        if debug:
            logger.info("⚠️ Forecast Model :: Debug mode is ON")

        if verbose:
            logger.info(f"Start Date (YYYY-MM-DD): {start_date_str}")
            logger.info(f"End Date (YYYY-MM-DD): {end_date_str}")
            logger.info(f"Volcano ID: {self.volcano_id}")
            logger.info(f"Network: {self.network}")
            logger.info(f"Station: {self.station}")
            logger.info(f"Location: {self.location}")
            logger.info(f"Channel: {self.channel}")
            logger.info(f"Output Dir: {self.output_dir}")

    def _normalize_dates(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> tuple[datetime, datetime, str, str]:
        """Normalize start and end dates to standard format.

        Converts date strings to datetime objects and formats them consistently.
        Start date is set to 00:00:00, end date is set to 23:59:59.

        Args:
            start_date: Start date in YYYY-MM-DD format or datetime object.
            end_date: End date in YYYY-MM-DD format or datetime object.

        Returns:
            Tuple of (start_date, end_date, start_date_str, end_date_str)
        """
        start_date = to_datetime(start_date).replace(hour=0, minute=0, second=0)
        end_date = to_datetime(end_date).replace(hour=23, minute=59, second=59)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        return start_date, end_date, start_date_str, end_date_str

    def _setup_directories(
        self,
        network: str,
        station: str,
        location: str,
        channel: str,
        output_dir: Optional[str],
    ) -> tuple[str, str, str, str]:
        """Setup directory structure for forecast model outputs.

        Creates the NSLC (Network.Station.Location.Channel) identifier and
        builds the directory structure for storing model outputs.

        Args:
            network: Network code
            station: Station code
            location: Location code
            channel: Channel code
            output_dir: Base output directory. Defaults to 'output' in current directory.

        Returns:
            Tuple of (nslc, output_dir, station_dir, features_dir)
        """
        nslc = f"{network}.{station}.{location}.{channel}"
        output_dir = output_dir or os.path.join(os.getcwd(), "output")
        station_dir = os.path.join(output_dir, nslc)
        features_dir = os.path.join(station_dir, "features")

        return nslc, output_dir, station_dir, features_dir

    def _initialize_feature_parameters(
        self,
    ) -> tuple[ComprehensiveFCParameters, set[str]]:
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

    def _setup_calculate_tremor(
        self,
        methods: Optional[str],
        filename_prefix: Optional[str],
        remove_outlier_method: Literal["all", "maximum"],
        interpolate: bool,
        value_multiplier: Optional[float],
        cleanup_tmp_dir: bool,
        plot_tmp: bool,
        save_plot: bool,
        overwrite_plot: bool,
        verbose: Optional[bool],
        debug: Optional[bool],
    ) -> CalculateTremor:
        """Setup CalculateTremor instance with configuration.

        Creates and configures a CalculateTremor instance with all necessary
        parameters. Updates start_date to include window_size buffer.

        Args:
            methods: Calculation methods to apply
            filename_prefix: Prefix for generated filenames
            remove_outlier_method: Method for outlier removal
            interpolate: Whether to interpolate data
            value_multiplier: Scaling factor for values
            cleanup_tmp_dir: Whether to clean temporary directory
            plot_tmp: Whether to plot temporary results
            save_plot: Whether to save plots
            overwrite_plot: Whether to overwrite existing plots
            verbose: Enable verbose logging
            debug: Enable debug mode

        Returns:
            Configured CalculateTremor instance
        """
        verbose = verbose or self.verbose
        debug = debug or self.debug

        # Create CalculateTremor with explicit parameters for type safety
        calculate = CalculateTremor(
            station=self.station,
            channel=self.channel,
            network=self.network,
            location=self.location,
            start_date=self.start_date_minus_window_size,
            end_date=self.end_date,
            output_dir=self.output_dir,
            overwrite=self.overwrite,
            n_jobs=self.n_jobs,
            methods=methods,
            filename_prefix=filename_prefix,
            remove_outlier_method=remove_outlier_method,
            interpolate=interpolate,
            value_multiplier=value_multiplier,
            cleanup_tmp_dir=cleanup_tmp_dir,
            plot_tmp=plot_tmp,
            save_plot=save_plot,
            overwrite_plot=overwrite_plot,
        )

        if verbose:
            calculate.verbose = True

        if debug:
            calculate.debug = True

        return calculate

    def _calculate_from_sds(
        self, calculate: CalculateTremor, sds_dir: Optional[str]
    ) -> CalculateTremor:
        """Calculate tremor from SDS data source.

        Args:
            calculate: CalculateTremor instance
            sds_dir: Path to SDS directory

        Returns:
            Calculated CalculateTremor instance with data

        Raises:
            ValueError: If sds_dir is None or doesn't exist
        """
        if sds_dir is None:
            raise ValueError(
                "You chose 'sds' as source, please provide 'sds_dir' parameter. "
                "Example: calculate(source='sds', sds_dir='converted')"
            )

        if not os.path.isdir(sds_dir):
            raise ValueError(f"SDS dir {sds_dir} not exists.")

        return calculate.from_sds(sds_dir=sds_dir).run()

    def _calculate_from_fdsn(
        self, calculate: CalculateTremor, client_url: str
    ) -> CalculateTremor:
        """Calculate tremor from FDSN data source.

        Args:
            calculate: CalculateTremor instance
            client_url: FDSN service URL

        Returns:
            Calculated CalculateTremor instance with data

        Raises:
            NotImplementedError: FDSN source is not yet supported
        """
        logger.error(f"FDSN source is not yet supported. Client url: {client_url}")
        raise NotImplementedError("FDSN source is not yet supported")

    def _adjust_dates_to_tremor_range(self, tremor_data: TremorData) -> None:
        """Adjust start_date and end_date to match available tremor data.

        Updates self.start_date, self.end_date, and their string representations
        if they fall outside the tremor data range. Logs changes if verbose mode
        is enabled.

        Args:
            tremor_data: TremorData instance with date range
        """
        # Adjust start date if earlier than tremor start
        if self.start_date_minus_window_size < tremor_data.start_date:
            self.start_date = tremor_data.start_date
            self.start_date_str = tremor_data.start_date_str
            if self.verbose:
                logger.info(
                    f"start_date parameter: {self.start_date_minus_window_size} updated to "
                    f"tremor start date: {tremor_data.start_date}"
                )

        # Adjust end date if later than tremor end
        if self.end_date > tremor_data.end_date:
            self.end_date = tremor_data.end_date
            self.end_date_str = tremor_data.end_date_str
            if self.verbose:
                logger.info(
                    f"end_date parameter: {self.end_date} updated to "
                    f"tremor end date: {tremor_data.end_date}"
                )

    def _prepare_features_data(
        self,
        tremor_columns: Optional[list[str]],
    ) -> pd.DataFrame:
        """Prepare features data by filtering columns if specified.

        Validates that specified columns exist and returns a filtered dataframe
        containing only id, datetime, and the specified tremor columns.

        Args:
            tremor_columns: Specific columns to extract, or None for all

        Returns:
            Filtered features dataframe

        Raises:
            ValueError: If specified columns don't exist in features_data
        """
        features_data = self.features_data

        if tremor_columns is not None:
            validate_columns(self.features_data, tremor_columns)
            features_data = features_data[["id", "datetime", *tremor_columns]]

        return features_data

    def _prepare_extraction_parameters(
        self,
        exclude_features: Optional[Union[list[str], bool]],
        n_jobs: Optional[int],
    ) -> dict[str, Any]:
        """Prepare parameters for tsfresh feature extraction.

        Handles feature exclusion logic and builds the parameter dictionary
        for tsfresh feature extraction functions.

        Args:
            exclude_features: Features to exclude from calculation
            n_jobs: Number of parallel jobs

        Returns:
            Dictionary of extraction parameters for tsfresh
        """
        # Handle feature exclusion
        default_fc_parameters = self.default_fc_parameters

        if exclude_features is not None:
            if isinstance(exclude_features, list):
                default_fc_parameters = self.drop_features(exclude_features)
            elif isinstance(exclude_features, bool) and not exclude_features:
                self.excludes_features = set()

        # Build extraction parameters
        return {
            "column_id": "id",
            "column_sort": "datetime",
            "n_jobs": n_jobs or self.n_jobs,
            "default_fc_parameters": default_fc_parameters,
        }

    def _extract_features_for_column(
        self,
        features_data: pd.DataFrame,
        column: str,
        y: pd.Series,
        extract_params: dict[str, Any],
        use_relevant_features: bool,
        prefix_filename: str,
        extract_features_dir: str,
        overwrite: bool,
    ) -> Optional[str]:
        """Extract features for a single tremor column.

        Performs tsfresh feature extraction for one column, either using
        all features or only relevant features based on correlation with labels.

        Args:
            features_data: Features dataframe with id, datetime, and tremor columns
            column: Column name to extract features from
            y: Target labels
            extract_params: Parameters for tsfresh extraction
            use_relevant_features: Whether to use relevant features only
            prefix_filename: Prefix for output filename
            extract_features_dir: Directory to save extracted features
            overwrite: Whether to overwrite existing files

        Returns:
            Path to extracted features CSV, or None if skipped
        """
        extracted_csv = os.path.join(
            extract_features_dir, f"{prefix_filename}_{column}.csv"
        )

        # Skip if already exists and not overwriting
        if not overwrite and os.path.isfile(extracted_csv):
            if self.verbose:
                logger.info(
                    f"Extracted features for {column} already exist: {extracted_csv}"
                )
            return extracted_csv

        # Prepare data for extraction
        df = features_data[["id", "datetime", column]]

        if self.verbose:
            logger.info(f"Extracting features for {column}")

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
        extracted_features.index.name = "id"
        extracted_features.to_csv(extracted_csv, index=True)

        logger.info(f"Extracted features for {column} saved: {extracted_csv}")

        return extracted_csv

    def _validate_tremor_for_labeling(
        self,
        tremor_data: pd.DataFrame,
        tremor_columns: Optional[list[str]],
    ) -> None:
        """Validate tremor data is available for label building.

        Checks that tremor data is loaded and that specified columns exist.

        Args:
            tremor_data: Tremor dataframe
            tremor_columns: Columns to validate

        Raises:
            ValueError: If tremor data is not loaded or columns are invalid
        """
        if not isinstance(tremor_data, pd.DataFrame) or len(tremor_data) == 0:
            raise ValueError(
                "Tremor data not found/loaded. "
                "Please run calculate() or load_tremor_data() method first."
            )

        if tremor_columns:
            validate_columns(tremor_data, tremor_columns)

    def _prepare_tremor_for_labeling(
        self,
        tremor_columns: Optional[list[str]],
    ) -> pd.DataFrame:
        """Prepare tremor data for label building.

        Creates a copy of tremor data and optionally selects specific columns.
        Sorts the data by datetime index.

        Args:
            tremor_columns: Specific columns to select, or None for all

        Returns:
            Prepared tremor dataframe
        """
        df_tremor = self.tremor_data.copy()

        if tremor_columns is not None:
            df_tremor = df_tremor[tremor_columns]

        df_tremor.sort_index(ascending=True, inplace=True)

        return df_tremor

    def _validate_label_tremor_date_range(
        self,
        df_label: pd.DataFrame,
        df_tremor: pd.DataFrame,
    ) -> None:
        """Validate label date range falls within tremor data range.

        Ensures that the label windows are completely covered by available
        tremor data, preventing gaps in feature extraction.

        Args:
            df_label: Label dataframe with datetime index
            df_tremor: Tremor dataframe with datetime index

        Raises:
            ValueError: If label dates fall outside tremor data range
        """
        label_start_date: pd.Timestamp = df_label.index[0]
        label_end_date: pd.Timestamp = df_label.index[-1]
        tremor_start_date: pd.Timestamp = df_tremor.index[0]
        tremor_end_date: pd.Timestamp = df_tremor.index[-1]

        if tremor_start_date > label_start_date:
            raise ValueError(
                f"Training start date ({label_start_date}) should be after/equal "
                f"to tremor start date ({tremor_start_date}). "
                f"Change your training start date to after/equal {tremor_start_date}."
            )

        if tremor_end_date < label_end_date:
            raise ValueError(
                f"Training end date ({label_end_date}) should be before/equal "
                f"to tremor end date ({tremor_end_date}). "
                f"Change your training end date to before/equal {tremor_end_date}."
            )

    def _calculate_eruption_statistics(
        self,
        label_builder: LabelBuilder,
    ) -> None:
        """Calculate and log eruption class statistics.

        Computes the number of eruption and non-eruption windows, and their
        ratio. Logs statistics if verbose mode is enabled.

        Args:
            label_builder: LabelBuilder instance with built labels
        """
        self.total_eruption_class = len(label_builder.df_eruption)
        self.total_non_eruption_class = (
            len(label_builder.df) - self.total_eruption_class
        )
        class_ratio: float = self.total_eruption_class / self.total_non_eruption_class

        if self.verbose:
            logger.info(
                f"Total number of eruptions: {self.total_eruption_class}. "
                f"Total number of non-eruptions: {self.total_non_eruption_class}. "
                f"Class ratio (eruption vs non-eruptions): {class_ratio}"
            )

    def validate(self) -> None:
        """Validate initialization parameters.

        Validates that all required parameters are properly set and creates
        necessary directories. Follows the pattern used in LabelBuilder and
        FeaturesBuilder classes.

        Raises:
            ValueError: If any parameters are invalid
        """
        # Validate window size
        if self.window_size <= 0:
            raise ValueError(
                f"window_size must be greater than 0. Got: {self.window_size}"
            )

        # Validate n_jobs
        if self.n_jobs <= 0:
            raise ValueError(f"n_jobs must be greater than 0. Got: {self.n_jobs}")

        # Validate date ranges
        validate_date_ranges(self.start_date, self.end_date)

        # Validate strings are not empty
        if not self.station.strip():
            raise ValueError("station cannot be empty")

        if not self.channel.strip():
            raise ValueError("channel cannot be empty")

        if not self.volcano_id.strip():
            raise ValueError("volcano_id cannot be empty")

        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.station_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)

    def load_tremor_data(self, tremor_csv: str) -> Self:
        """Load calculate tremor data from CSV file

        Args:
            tremor_csv (str): Tremor CSV file

        Returns:
            self (Self)
        """
        tremor_data = TremorData()
        self.TremorData = tremor_data
        self.tremor_data = tremor_data.from_csv(tremor_csv)
        self.TremorData.csv = tremor_csv
        self.tremor_csv = tremor_csv
        return self

    def calculate(
        self,
        source: Literal["sds", "fdsn"] = "sds",
        methods: Optional[str] = None,
        filename_prefix: Optional[str] = None,
        remove_outlier_method: Literal["all", "maximum"] = "maximum",
        interpolate: bool = True,
        value_multiplier: Optional[float] = None,
        cleanup_tmp_dir: bool = False,
        plot_tmp: bool = True,
        save_plot: bool = True,
        overwrite_plot: bool = False,
        sds_dir: Optional[str] = None,
        client_url: str = "https://service.iris.edu",
        verbose: Optional[bool] = None,
        debug: Optional[bool] = None,
    ) -> Self:
        """Calculate Tremor Data from seismic data source.

        Args:
            source (optional, Literal["sds", "fdsn"]): Seismic data source
            methods (Optional[str]): Calculation methods to apply.
            filename_prefix (Optional[str]): Prefix for generated filenames.
            remove_outlier_method (bool): If True, removes outliers from the data. Defaults to True.
            interpolate (bool): If True, interpolates the data. Defaults to True.
            value_multiplier (Optional[float]): Scaling factor for seismic values.
            cleanup_tmp_dir (bool): If True, deletes temporary directory after use. Defaults to False.
            plot_tmp (bool): If True, plot temporary results for quick view.
            save_plot (bool): If True, save tremor results for quick view.
            overwrite_plot (bool): If True, overwrite existing plot files. Defaults to False.
            sds_dir (str): SDS directory location. Must be provided if source is 'sds'.
            client_url (str): URL to FDSN service. Default to https://service.iris.edu
            verbose (bool): If True, enables verbose logging. Defaults to False.
            debug (bool): If True, enables debug mode. Defaults to False.

        Returns:
            self (Self): ForecastModel object
        """
        # Setup CalculateTremor instance
        calculate = self._setup_calculate_tremor(
            methods=methods,
            filename_prefix=filename_prefix,
            remove_outlier_method=remove_outlier_method,
            interpolate=interpolate,
            value_multiplier=value_multiplier,
            cleanup_tmp_dir=cleanup_tmp_dir,
            plot_tmp=plot_tmp,
            save_plot=save_plot,
            overwrite_plot=overwrite_plot,
            verbose=verbose,
            debug=debug,
        )

        self.CalculateTremor = calculate

        # Calculate from appropriate source
        if source == "sds":
            calculate = self._calculate_from_sds(calculate, sds_dir)
        elif source == "fdsn":
            calculate = self._calculate_from_fdsn(calculate, client_url)

        # Wrap calculated data
        tremor_data = TremorData(calculate.df)
        self.TremorData = tremor_data
        self.TremorData.csv = calculate.csv
        self.tremor_csv = calculate.csv

        # Adjust dates to match tremor data availability
        self._adjust_dates_to_tremor_range(tremor_data)

        # Slice tremor data to adjusted date range
        self.tremor_data = tremor_data.df.loc[self.start_date : self.end_date]

        return self

    def drop_features(self, excludes_features: list[str]) -> ComprehensiveFCParameters:
        """Drop features from calculation.

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

    def concat_features(self) -> Self:
        """Concatenate features from calculation."""
        if len(self.extract_features_csvs) > 0:
            csv_list = list(self.extract_features_csvs)
            if self.verbose:
                logger.info(f"Concatenating extracted features from calculation.")
            filepath = os.path.join(
                self.features_dir,
                f"extracted_features_{self.start_date_str}-{self.end_date_str}.csv",
            )
            self.extracted_features_csv = utils_concat_features(
                csv_list, filepath, return_as_filepath=True
            )

        if len(self.relevant_features_csvs) > 0:
            csv_list = list(self.relevant_features_csvs)
            if self.verbose:
                logger.info(
                    f"Concatenating relevant extracted features from calculation."
                )
            filepath = os.path.join(
                self.features_dir,
                f"extracted_relevant_{self.start_date_str}-{self.end_date_str}.csv",
            )
            self.extracted_relevant_csv = utils_concat_features(
                csv_list, filepath, return_as_filepath=True
            )

        return self

    def extract_features(
        self,
        exclude_features: Optional[Union[list[str], bool]] = None,
        tremor_columns: Optional[list[str]] = None,
        use_relevant_features: bool = False,
        overwrite: bool = False,
        concat_features: bool = True,
        n_jobs: Optional[int] = None,
    ) -> Self:
        """Extract features from tremor data using tsfresh.

        Applies time-series feature extraction to each tremor column, either
        extracting all comprehensive features or only statistically relevant
        features based on correlation with eruption labels.

        Args:
            exclude_features (Optional[list[str]]): List features calculator to be excluded.
            tremor_columns (list[str]): List of tremor columns to extract.
            use_relevant_features (bool): If True, extract features using relevant features.
            overwrite (bool): If True, overwrite existing feature files. Defaults to False.
            concat_features (bool): If True, concat all features
            n_jobs (int): Number of parallel jobs. Defaults to None.

        Returns:
            self (Self): ForecastModel object
        """
        # Prepare data
        features_data = self._prepare_features_data(tremor_columns)

        # Setup parameters
        overwrite = overwrite or self.overwrite
        label_data = self.label_data
        prefix_filename = (
            "extracted_relevant" if use_relevant_features else "extracted_features"
        )

        if use_relevant_features and self.verbose:
            logger.info("Extracting features using relevant features")

        # Prepare target labels
        y = label_data["is_erupted"]
        y.index = label_data["id"]

        # Setup extraction directory
        extract_features_dir = os.path.join(self.features_dir, "extract_features")
        os.makedirs(extract_features_dir, exist_ok=True)

        # Prepare extraction parameters
        extract_params = self._prepare_extraction_parameters(exclude_features, n_jobs)

        # Extract features for each column
        extracted_csvs = set()
        for column in features_data.columns.tolist():
            if column in ["id", "datetime"]:
                continue

            csv_path = self._extract_features_for_column(
                features_data=features_data,
                column=column,
                y=y,
                extract_params=extract_params,
                use_relevant_features=use_relevant_features,
                prefix_filename=prefix_filename,
                extract_features_dir=extract_features_dir,
                overwrite=overwrite,
            )

            if csv_path:
                extracted_csvs.add(csv_path)

        # Update tracked CSVs
        if use_relevant_features:
            self.relevant_features_csvs.update(extracted_csvs)
        else:
            self.extract_features_csvs.update(extracted_csvs)

        # Concatenate if requested
        if concat_features:
            self.concat_features()

        return self

    def build_features(
        self,
        output_dir: Optional[str] = None,
        tremor_columns: Optional[list[str]] = None,
        save_per_method: bool = True,
        save_tmp_feature: bool = False,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> Self:
        """Build features from tremor data.

        Args:
            output_dir (Optional[str]): Directory to save features to. Defaults to None.
            tremor_columns (list[str]): List of tremor columns to extract. Defaults to None.
            save_tmp_feature (bool): If True, save features temporarily. Defaults to False.
            save_per_method (bool): If True, save features per method. Defaults to True.
            overwrite (bool): If True, overwrite existing feature files. Defaults to False.
            verbose (bool): If True, show progress. Defaults to False.

        Returns:
            self (Self): ForecastModel object
        """
        label_data = self.label_data
        output_dir = output_dir or self.features_dir
        verbose = verbose or self.verbose

        features_builder = FeaturesBuilder(
            df_tremor=self.tremor_data,
            df_label=label_data,
            output_dir=output_dir,
            window_size=self.window_size,
            tremor_columns=tremor_columns,
            overwrite=overwrite or self.overwrite,
            verbose=verbose or self.verbose,
        )

        self.FeaturesBuilder = features_builder
        features_filename = f"features_{self.start_date_str}-{self.end_date_str}_ws-{self.window_size}.csv"
        features_data = features_builder.build(
            save_tmp_feature=save_tmp_feature,
            save_per_method=save_per_method,
            filename=features_filename,
        )

        # Sync label with features matrix
        if len(features_builder.unique_ids) == 0:
            raise ValueError(f"Features builder does not have unique ids.")

        label_data = label_data[label_data["id"].isin(features_builder.unique_ids)]

        label_csv = os.path.join(
            self.features_dir,
            f"label_{self.start_date_str}-{self.end_date_str}.csv",
        )
        label_data.to_csv(label_csv, index=True)

        self.features_data = features_data
        self.features_csv = features_builder.csv
        self.label_data = label_data
        self.label_csv = label_csv

        return self

    def build_label(
        self,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        day_to_forecast: int,
        eruption_dates: list[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        output_dir: Optional[str] = None,
        tremor_columns: Optional[list[str]] = None,
        verbose: Optional[bool] = None,
        debug: Optional[bool] = None,
    ) -> Self:
        """Build labels for eruption forecasting.

        Creates labeled time windows for training machine learning models.
        Each window is labeled as erupted (1) or not erupted (0) based on
        eruption dates and forecast horizon.

        Args:
            window_step (int): Window step size.
            window_step_unit (Literal["minutes", "hours"]): Unit of window step.
            day_to_forecast (int): Day to forecast in days.
            eruption_dates (list[str]): Eruption dates in YYYY-MM-DD format.
            start_date (str, optional): Override self.start_date.
            end_date (str, optional): Override self.end_date.
            output_dir (Optional[str], optional): Output directory. Defaults to None.
            tremor_columns (Optional[list[str]], optional): Columns to select. Defaults to None.
            verbose (bool): If True, enables verbose logging. Defaults to False.
            debug (bool): If True, enables debug mode. Defaults to False.

        Returns:
            self (Self): ForecastModel object
        """
        # Setup parameters
        tremor_data = self.tremor_data
        train_start_date = start_date or self.start_date
        train_end_date = end_date or self.end_date
        verbose = verbose or self.verbose
        debug = debug or self.debug
        output_dir = output_dir or self.station_dir

        # Validate inputs
        validate_date_ranges(train_start_date, train_end_date)
        self._validate_tremor_for_labeling(tremor_data, tremor_columns)

        # Build labels
        label_builder = LabelBuilder(
            start_date=to_datetime(train_start_date),
            end_date=to_datetime(train_end_date),
            window_size=self.window_size,
            window_step=window_step,
            window_step_unit=window_step_unit,
            day_to_forecast=day_to_forecast,
            eruption_dates=eruption_dates,
            volcano_id=self.volcano_id,
            output_dir=output_dir,
            verbose=verbose,
            debug=debug,
        ).build()

        # Prepare tremor data
        df_tremor = self._prepare_tremor_for_labeling(tremor_columns)
        df_label = label_builder.df

        # Validate date ranges
        self._validate_label_tremor_date_range(df_label, df_tremor)

        # Set label properties
        self.LabelBuilder = label_builder
        self.label_csv = label_builder.csv
        self.basename = os.path.basename(label_builder.csv).split(".csv")[0]

        # Filter labels from start_date onwards
        df_label = df_label.loc[self.start_date :]

        if df_label.empty:
            raise ValueError(f"Label from start date {self.start_date} is empty.")

        self.label_data = df_label

        # Calculate and log statistics
        self._calculate_eruption_statistics(label_builder)

        return self

    def predict(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        output_dir: Optional[str] = None,
        verbose: Optional[bool] = None,
    ):
        verbose = verbose or self.verbose

        start_date = to_datetime(start_date)
        end_date = to_datetime(end_date)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        output_dir = output_dir or os.path.join(self.station_dir, "predict")
        os.makedirs(output_dir, exist_ok=True)

        filename = f"predict_window_{start_date_str}-{end_date_str}_ws-{window_step}{window_step_unit}.csv"
        predict_window_csv = os.path.join(output_dir, filename)

        df_predict_window = construct_windows(
            start_date=start_date,
            end_date=end_date,
            window_step=window_step,
            window_step_unit=window_step_unit,
        )

        df_predict_window.to_csv(predict_window_csv, index=True)

        if verbose:
            logger.info(f"Prediction window saved to: {predict_window_csv}")

        return self
