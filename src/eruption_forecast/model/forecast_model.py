# Standard library imports
import os
from datetime import datetime, timedelta
from typing import Any, Literal, Self

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
    ID_COLUMN,
)
from eruption_forecast.features.features_builder import FeaturesBuilder
from eruption_forecast.features.tremor_matrix_builder import TremorMatrixBuilder
from eruption_forecast.label.label_builder import LabelBuilder
from eruption_forecast.logger import logger
from eruption_forecast.model.train_model import TrainModel
from eruption_forecast.tremor.calculate_tremor import CalculateTremor
from eruption_forecast.tremor.tremor_data import TremorData
from eruption_forecast.utils import (
    construct_windows,
    normalize_dates,
    to_datetime,
    validate_columns,
    validate_date_ranges,
)


class ForecastModel:
    """Create forecast model from seismic data.

    Orchestrates the complete eruption forecasting pipeline:
    1. Calculate tremor data from seismic data (RSAM/DSAR metrics)
    2. Build labels for supervised learning
    3. Extract time-series features using tsfresh
    4. Train classification models
    5. Generate predictions

    Args:
        station (str): Seismic station code (e.g., "OJN").
        channel (str): Seismic channel code (e.g., "EHZ").
        start_date (str | datetime): Start date in YYYY-MM-DD format.
        end_date (str | datetime): End date in YYYY-MM-DD format.
        window_size (int): Window size in days for training data windows.
        volcano_id (str): Volcano identifier for output naming.
        network (str): Seismic network code. Defaults to "VG".
        location (str): Seismic location code. Defaults to "00".
        output_dir (str): Directory for output files. Defaults to "output".
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
        n_jobs (int): Number of parallel jobs to use. Defaults to 1.
        verbose (bool): If True, enables verbose logging. Defaults to False.
        debug (bool): If True, enables debug mode. Defaults to False.

    Example:
        >>> model = ForecastModel(
        ...     station="OJN",
        ...     channel="EHZ",
        ...     start_date="2024-01-01",
        ...     end_date="2024-06-30",
        ...     window_size=1,
        ...     volcano_id="LEWOTOBI",
        ... )
        >>> model.calculate(source="sds", sds_dir="data/sds")
        >>> model.build_label(
        ...     window_step=12,
        ...     window_step_unit="hours",
        ...     day_to_forecast=2,
        ...     eruption_dates=["2024-03-15", "2024-05-20"],
        ... )
        >>> model.extract_features().train()
    """

    def __init__(
        self,
        station: str,
        channel: str,
        start_date: str | datetime,
        end_date: str | datetime,
        window_size: int,
        volcano_id: str,
        network: str = "VG",
        location: str = "00",
        output_dir: str | None = None,
        overwrite: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        # Normalize dates
        start_date, end_date, start_date_str, end_date_str = normalize_dates(
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
        self.CalculateTremor: CalculateTremor | None = None
        self.TremorData: TremorData | None = None
        self.tremor_data: pd.DataFrame = pd.DataFrame()
        self.tremor_csv: str | None = None

        # Will be set after build_label() method called
        self.LabelBuilder: LabelBuilder | None = None
        self.label_data: pd.DataFrame = pd.DataFrame()
        self.label_csv: str | None = None
        self.total_eruption_class: int | None = None
        self.total_non_eruption_class: int | None = None

        # Will be set after extract_features() called
        self.TremorMatrixBuilder: TremorMatrixBuilder | None = None
        self.tremor_matrix_df: pd.DataFrame = pd.DataFrame()
        self.tremor_matrix_csv: str | None = None
        self.FeaturesBuilder: FeaturesBuilder | None = None
        self.features_df: pd.DataFrame = pd.DataFrame()
        self.features_csv: str | None = None
        self.use_relevant_features: bool = False

        # Will be set after predict() called
        self.prediction_features_csvs: set[str] = set()

        # Will be set after train() called
        self.TrainModel: TrainModel | None = None

        # Base filename without extension
        self.basename: str | None = None

        # Validate and create directories
        self.validate()
        self.create_directories()

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

    @staticmethod
    def _setup_directories(
        network: str,
        station: str,
        location: str,
        channel: str,
        output_dir: str | None,
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

    def _setup_calculate_tremor(
        self,
        methods: str | None,
        filename_prefix: str | None,
        remove_outlier_method: Literal["all", "maximum"],
        interpolate: bool,
        value_multiplier: float | None,
        cleanup_tmp_dir: bool,
        plot_tmp: bool,
        save_plot: bool,
        overwrite_plot: bool,
        n_jobs: int | None = None,
        verbose: bool = False,
        debug: bool = False,
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
            n_jobs: Number of jobs to run in parallel. Isolated on this method only
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
            n_jobs=n_jobs or self.n_jobs,
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

    @staticmethod
    def _calculate_from_sds(
        calculate: CalculateTremor, sds_dir: str | None
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
        # TODO: Calculate using FDSN
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
    ) -> str | None:
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
        df = features_data[[ID_COLUMN, DATETIME_COLUMN, column]]

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
        extracted_features.index.name = ID_COLUMN
        extracted_features.to_csv(extracted_csv, index=True)

        logger.info(f"Extracted features for {column} saved: {extracted_csv}")

        return extracted_csv

    @staticmethod
    def _validate_tremor_for_labeling(
        tremor_data: pd.DataFrame,
        tremor_columns: list[str] | None,
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
        tremor_columns: list[str] | None,
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

    @staticmethod
    def _validate_label_tremor_date_range(
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

        Ensures that window_size is positive, date ranges are valid,
        and required string parameters (station, channel, volcano_id)
        are not empty.

        Raises:
            ValueError: If window_size is <= 0, date ranges are invalid
                (start_date >= end_date), or required string parameters
                (station, channel, volcano_id) are empty.

        Example:
            >>> model = ForecastModel(...)
            >>> model.validate()  # Called automatically in __init__
        """
        # Validate window size
        if self.window_size <= 0:
            raise ValueError(
                f"window_size must be greater than 0. Got: {self.window_size}"
            )

        # Validate date ranges
        validate_date_ranges(self.start_date, self.end_date)

        # Validate strings are not empty
        if not self.station.strip():
            raise ValueError("station cannot be empty")

        if not self.channel.strip():
            raise ValueError("channel cannot be empty")

        if not self.volcano_id.strip():
            raise ValueError("volcano_id cannot be empty")

    def create_directories(self) -> None:
        """Create required output directory structure.

        Creates the main output directory, station-specific directory,
        and features subdirectory. Called automatically during initialization.

        Example:
            >>> model = ForecastModel(...)
            >>> model.create_directories()  # Called in __init__
            >>> # Creates: output/, output/VG.OJN.00.EHZ/, output/VG.OJN.00.EHZ/features/
        """
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.station_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)

    def load_tremor_data(self, tremor_csv: str) -> Self:
        """Load pre-calculated tremor data from CSV file.

        Loads tremor data from a previously calculated tremor CSV file
        instead of recalculating from raw seismic data. Use this method
        when you already have tremor metrics calculated.

        Args:
            tremor_csv (str): Path to the tremor CSV file containing
                columns like rsam_f0, rsam_f1, dsar_f0-f1, etc.

        Returns:
            Self: ForecastModel instance for method chaining.
                Sets self.tremor_data, self.TremorData, and self.tremor_csv.

        Example:
            >>> model = ForecastModel(
            ...     station="OJN",
            ...     channel="EHZ",
            ...     start_date="2025-01-01",
            ...     end_date="2025-06-30",
            ...     window_size=1,
            ...     volcano_id="LEWOTOBI"
            ... )
            >>> model.load_tremor_data("output/VG.OJN.00.EHZ/tremor/tremor.csv")
            >>> # Now ready to build labels and extract features
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
        methods: str | None = None,
        filename_prefix: str | None = None,
        remove_outlier_method: Literal["all", "maximum"] = "maximum",
        interpolate: bool = True,
        value_multiplier: float | None = None,
        cleanup_tmp_dir: bool = False,
        plot_tmp: bool = False,
        save_plot: bool = False,
        overwrite_plot: bool = False,
        sds_dir: str | None = None,
        client_url: str = "https://service.iris.edu",
        n_jobs: int | None = None,
        verbose: bool = False,
        debug: bool = False,
    ) -> Self:
        """Calculate Tremor Data from seismic data source.

        Args:
            source (optional, Literal["sds", "fdsn"]): Seismic data source
            methods (Optional[str]): Calculation methods to apply.
            filename_prefix (Optional[str]): Prefix for generated filenames.
            remove_outlier_method ("all" or "maximum"): Method for outlier removal. Defaults to "maximum".
            interpolate (bool): If True, interpolates the data. Defaults to True.
            value_multiplier (Optional[float]): Scaling factor for seismic values.
            cleanup_tmp_dir (bool): If True, deletes temporary directory after use. Defaults to False.
            plot_tmp (bool): If True, plot temporary results for quick view.
            save_plot (bool): If True, save tremor results for quick view.
            overwrite_plot (bool): If True, overwrite existing plot files. Defaults to False.
            sds_dir (str): SDS directory location. Must be provided if source is 'sds'.
            client_url (str): URL to FDSN service. Default to https://service.iris.edu
            n_jobs: Number of jobs to run in parallel. Isolated on this method only
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
            n_jobs=n_jobs or self.n_jobs,
            verbose=verbose or self.verbose,
            debug=debug or self.debug,
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

    def extract_features(
        self,
        select_tremor_columns: list[str] | None = None,
        save_tremor_matrix_per_method: bool = True,
        save_tremor_matrix_per_id: bool = False,
        exclude_features: list[str] | None = None,
        use_relevant_features: bool = False,
        overwrite: bool = False,
        n_jobs: int | None = None,
        verbose: bool | None = None,
    ) -> Self:
        """Extract features from tremor data using tsfresh.

        Applies time-series feature extraction to each tremor column, either
        extracting all comprehensive features or only statistically relevant
        features based on correlation with eruption labels.

        Args:
            select_tremor_columns (list[str]): List of tremor columns to extract.
            save_tremor_matrix_per_method (bool, optional): Save separate CSV per tremor
                column. Defaults to True.
            save_tremor_matrix_per_id (bool, optional): BE CAREFULL, IT WILL GENERATE A LOT OF FILES.
                Save individual windowed tremor CSVs for debugging. Defaults to False.
            exclude_features (Optional[list[str]]): List features calculator to be excluded.
            use_relevant_features (bool): If True, extract features using relevant features.
            overwrite (bool): If True, overwrite existing feature files. Defaults to False.
            n_jobs (int): Number of parallel jobs. Defaults to None.
            verbose (bool): If True, enables verbose mode. Defaults to False.

        Returns:
            self (Self): ForecastModel object
        """
        tremor_matrix_builder = TremorMatrixBuilder(
            tremor_df=self.tremor_data,
            label_df=self.label_data,
            output_dir=self.features_dir,
            window_size=self.window_size,
            overwrite=overwrite or self.overwrite,
            verbose=verbose or self.verbose,
        ).build(
            select_tremor_columns=select_tremor_columns,
            save_tremor_matrix_per_method=save_tremor_matrix_per_method,
            save_tremor_matrix_per_id=save_tremor_matrix_per_id,
        )

        tremor_matrix_df = tremor_matrix_builder.df

        features_builder = FeaturesBuilder(
            tremor_matrix_df=tremor_matrix_df,
            label_df=self.label_data,
            output_dir=self.features_dir,
            overwrite=overwrite or self.overwrite,
            n_jobs=n_jobs or self.n_jobs,
        )

        extracted_features_df = features_builder.extract_features(
            use_relevant_features=use_relevant_features,
            select_tremor_columns=select_tremor_columns,
            exclude_features=exclude_features,
        )

        self.TremorMatrixBuilder = tremor_matrix_builder
        self.tremor_matrix_df = tremor_matrix_df
        self.tremor_matrix_csv = tremor_matrix_builder.csv

        self.FeaturesBuilder = features_builder
        self.features_df = extracted_features_df
        self.features_csv = features_builder.csv
        self.label_csv = features_builder.label_features_csv
        self.use_relevant_features = use_relevant_features

        return self

    def build_label(
        self,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        day_to_forecast: int,
        eruption_dates: list[str],
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
        output_dir: str | None = None,
        tremor_columns: list[str] | None = None,
        verbose: bool | None = None,
        debug: bool | None = None,
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

    def train(
        self,
        classifier: Literal[
            "svm", "knn", "dt", "rf", "gb", "nn", "nb", "lr", "voting"
        ] = "rf",
        cv_strategy: Literal["shuffle", "stratified", "timeseries"] = "shuffle",
        grid_params: dict[str, Any] | None = None,
        random_state: int = 0,
        total_seed: int = 500,
        number_of_significant_features: int = 20,
        sampling_strategy: str | float = 0.75,
        save_all_features: bool = False,
        plot_significant_features: bool = False,
        extracted_features_csv: str | None = None,
        output_dir: str | None = None,
        n_jobs: int | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> Self:
        """Training model using extracted features and labels.

        Args:
            classifier (str, optional): Classifier type ("rf", "gb", "svm", "lr", "nn",
                "dt", "knn", "nb", "voting"). Defaults to "rf".
            cv_strategy (str, optional): Cross-validation strategy ("shuffle", "stratified",
                "timeseries"). Defaults to "shuffle".
            grid_params (dict[str, any], optional): Grid-search parameters. Defaults to None.
            random_state (int, optional): Initiate random seed. Defaults to 0.
            total_seed (int, optional): Total random seed. Defaults to 500.
            number_of_significant_features (int, optional): Number of significant features. Defaults to 20.
            sampling_strategy (str, optional): Sampling strategy. Defaults to 0.75.
            save_all_features (bool, optional): Whether to save ALL features. Defaults to False.
            plot_significant_features (bool, optional): Whether to plot each significant feature. Defaults to False.
            extracted_features_csv (str | None): Path to extracted features.
            output_dir (str, optional): Path to output directory. Defaults to None.
            n_jobs (int, optional): Number of jobs. Defaults to 1.
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.

        Returns:
            self (Self): ForecastModel object
        """
        if verbose or self.verbose:
            print("=" * 50)
            print("| Training model")
            if self.use_relevant_features:
                print("|- Using Relevant features")
            print("=" * 50)

        features_csv = extracted_features_csv or self.features_csv

        if features_csv is None or not os.path.exists(features_csv):
            error_msg = f"Features CSV not found: {features_csv}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        label_csv = self.label_csv
        if label_csv is None or not os.path.exists(label_csv):
            error_msg = f"Label CSV not found: {label_csv}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        output_dir = output_dir or os.path.join(self.station_dir, "trainings")
        os.makedirs(output_dir, exist_ok=True)

        train_model = TrainModel(
            features_csv=features_csv,
            label_csv=label_csv,
            output_dir=output_dir,
            classifier=classifier,
            cv_strategy=cv_strategy,
            grid_params=grid_params,
            number_of_significant_features=number_of_significant_features,
            overwrite=overwrite or self.overwrite,
            n_jobs=n_jobs or self.n_jobs,
            verbose=verbose or self.verbose,
        )

        train_model.train(
            random_state=random_state,
            total_seed=total_seed,
            sampling_strategy=sampling_strategy,
            save_all_features=save_all_features,
            plot_significant_features=plot_significant_features,
        )

        self.TrainModel = train_model

        return self

    def predict(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        output_dir: str | None = None,
        verbose: bool | None = None,
    ) -> Self:
        """Generate prediction windows for eruption forecasting.

        Constructs sliding time windows over the given date range and saves
        them to a CSV file.  Note: actual model inference is not yet
        implemented — this method only prepares the prediction window layout.

        Args:
            start_date: Start date for prediction windows.
            end_date: End date for prediction windows.
            window_step: Step size between consecutive windows.
            window_step_unit: Unit of the window step ("minutes" or "hours").
            output_dir: Directory to write the prediction CSV.
                Defaults to ``<station_dir>/predictions``.
            verbose: Override instance verbose flag.

        Returns:
            Self for method chaining.
        """
        verbose = verbose or self.verbose

        if verbose:
            logger.info("predict() started")

        start_date = to_datetime(start_date)
        end_date = to_datetime(end_date)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        output_dir = output_dir or os.path.join(self.station_dir, "predictions")
        os.makedirs(output_dir, exist_ok=True)

        filename = f"predict_window_{start_date_str}-{end_date_str}_step-{window_step}{window_step_unit}.csv"
        predict_window_csv = os.path.join(output_dir, filename)

        df_predict_window = construct_windows(
            start_date=start_date,
            end_date=end_date,
            window_step=window_step,
            window_step_unit=window_step_unit,
        )

        logger.debug(f"Total preditcted windows generated: {len(df_predict_window)}")

        df_predict_window.to_csv(predict_window_csv, index=True)

        if verbose:
            logger.info(f"Prediction window saved to: {predict_window_csv}")

        return self
