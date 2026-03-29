import os
import json
from typing import Any, Self, Literal
from datetime import datetime, timedelta

import joblib
import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.model.constants import CLASSIFIERS
from eruption_forecast.utils.pathutils import ensure_dir, resolve_output_dir
from eruption_forecast.utils.date_utils import (
    to_datetime,
    normalize_dates,
)
from eruption_forecast.utils.validation import validate_columns, validate_date_ranges
from eruption_forecast.tremor.tremor_data import TremorData
from eruption_forecast.label.label_builder import LabelBuilder
from eruption_forecast.model.model_trainer import ModelTrainer
from eruption_forecast.model.model_predictor import ModelPredictor
from eruption_forecast.config.pipeline_config import (
    ModelConfig,
    TrainConfig,
    ForecastConfig,
    PipelineConfig,
    CalculateConfig,
    BuildLabelConfig,
    ExtractFeaturesConfig,
)
from eruption_forecast.report.pipeline_report import PipelineReport
from eruption_forecast.tremor.calculate_tremor import CalculateTremor
from eruption_forecast.features.features_builder import FeaturesBuilder
from eruption_forecast.label.dynamic_label_builder import DynamicLabelBuilder
from eruption_forecast.features.tremor_matrix_builder import TremorMatrixBuilder


class ForecastModel:
    """Orchestrate the complete volcanic eruption forecasting pipeline.

    Provides a high-level interface to coordinate all pipeline stages from
    raw seismic data to trained models and predictions. Uses method chaining
    for fluent API design.

    Pipeline stages:

    1. **Calculate tremor**: Process seismic waveforms → RSAM/DSAR metrics
    2. **Build labels**: Create time windows with eruption labels
    3. **Extract features**: Apply tsfresh to generate ML features
    4. **Train models**: Multi-seed training with cross-validation
    5. **Generate predictions**: Forecast eruptions on new data

    Attributes:
        station (str): Seismic station code.
        channel (str): Seismic channel code.
        start_date (datetime | None): Tremor calculation start date; set by ``calculate()``.
        end_date (datetime | None): Tremor calculation end date; set by ``calculate()``.
        window_size (int): Window size in days for training data windows.
        volcano_id (str): Volcano identifier for output naming.
        network (str): Seismic network code.
        location (str): Seismic location code.
        output_dir (str): Root directory for all outputs.
        root_dir (str | None): Anchor directory for path resolution.
        overwrite (bool): Whether to overwrite existing files.
        n_jobs (int): Number of parallel jobs.
        verbose (bool): Enable verbose logging.
        debug (bool): Enable debug mode.
        start_date_minus_window_size (datetime): Adjusted start date for tremor calculation.
        start_date_str (str): Start date as "YYYY-MM-DD".
        end_date_str (str): End date as "YYYY-MM-DD".
        nslc (str): Combined network.station.location.channel identifier.
        station_dir (str): Station-specific output directory.
        features_dir (str): Features output directory.
        basename (str | None): Base filename for outputs.
        CalculateTremor (CalculateTremor | None): Tremor calculation instance.
        TremorData (TremorData | None): Loaded tremor data wrapper.
        tremor_data (pd.DataFrame): Tremor metrics DataFrame.
        tremor_csv (str | None): Path to tremor CSV file.
        LabelBuilder (LabelBuilder | None): Label builder instance.
        label_data (pd.DataFrame): Labels DataFrame.
        label_csv (str | None): Path to labels CSV file.
        total_eruption_class (int | None): Count of eruption windows.
        total_non_eruption_class (int | None): Count of non-eruption windows.
        class_ratio (float | None): Ratio of eruption to non-eruption windows.
        TremorMatrixBuilder (TremorMatrixBuilder | None): Tremor matrix builder instance.
        tremor_matrix_df (pd.DataFrame): Windowed tremor matrix.
        tremor_matrix_csv (str | None): Path to tremor matrix CSV.
        FeaturesBuilder (FeaturesBuilder | None): Feature extraction instance.
        features_df (pd.DataFrame): Extracted features DataFrame.
        features_csv (str | None): Path to features CSV file.
        use_relevant_features (bool): Whether relevance filtering was applied.
        select_tremor_columns (list[str] | None): Selected tremor columns for processing.
        feature_selection_method (str): Feature selection method.
        trained_models (dict[str, str]): Mapping of classifier name to trained model
            registry CSV path. Populated after ``train()``; used by ``forecast()``.
        ModelPredictor (ModelPredictor | None): Predictor instance.
        prediction_df (pd.DataFrame): Prediction results DataFrame.

    Args:
        station (str): Seismic station code (e.g., "OJN").
        channel (str): Seismic channel code (e.g., "EHZ").
        window_size (int): Window size in days for training data windows.
        volcano_id (str): Volcano identifier for output naming.
        network (str): Seismic network code (e.g., "VG").
        location (str): Seismic location code (e.g., "00").
        output_dir (str | None, optional): Directory for output files. If None,
            defaults to ``root_dir/output``. Relative paths are resolved against
            ``root_dir`` (or ``os.getcwd()`` when ``root_dir`` is None). Absolute
            paths are used as-is. Defaults to None.
        root_dir (str | None, optional): Anchor directory for resolving relative
            ``output_dir`` values. Relative ``root_dir`` values are immediately
            normalized to an absolute path via ``os.path.abspath``. Defaults to
            None (uses ``os.getcwd()``).
        overwrite (bool, optional): Whether to overwrite existing files.
            Defaults to False.
        n_jobs (int, optional): Number of parallel jobs to use. Defaults to 1.
        verbose (bool, optional): If True, enables verbose logging. Defaults to False.
        debug (bool, optional): If True, enables debug mode. Defaults to False.

    Raises:
        ValueError: If window_size < 1 or network/location are empty.

    Examples:
        >>> # Complete pipeline example
        >>> model = ForecastModel(
        ...     station="OJN",
        ...     channel="EHZ",
        ...     window_size=1,
        ...     volcano_id="LEWOTOBI",
        ...     network="VG",
        ...     location="00",
        ...     root_dir=r"D:\\Projects\\eruption-forecast",
        ... )
        >>> model.calculate(
        ...     start_date="2024-01-01",
        ...     end_date="2024-06-30",
        ...     source="sds",
        ...     sds_dir="data/sds",
        ... )
        >>> model.build_label(
        ...     window_step=12,
        ...     window_step_unit="hours",
        ...     day_to_forecast=2,
        ...     eruption_dates=["2024-03-15", "2024-05-20"],
        ... )
        >>> model.extract_features().train(classifier="rf", total_seed=100)

        >>> # Method chaining example
        >>> model = ForecastModel(...).calculate(...).build_label(...).extract_features()
        >>> model.train(classifier=["rf", "xgb"], random_state=42, total_seed=100)
    """

    def __init__(
        self,
        station: str,
        channel: str,
        network: str,
        location: str,
        window_size: int,
        volcano_id: str,
        output_dir: str | None = None,
        root_dir: str | None = None,
        overwrite: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize the ForecastModel pipeline orchestrator.

        Resolves all output directory paths and initialises lifecycle state
        attributes. Dates are not required at construction time — pass them to
        ``calculate()`` when fetching tremor data. No data is loaded or
        processed until the pipeline stage methods (calculate, build_label,
        extract_features, train, forecast) are called.

        Args:
            station (str): Seismic station code (e.g., "OJN").
            channel (str): Channel code (e.g., "EHZ").
            network (str): Seismic network code (e.g., "VG").
            location (str): Location code (e.g., "00").
            window_size (int): Window size in days used for matrix building and forecasting.
            volcano_id (str): Unique volcano identifier used for labelling.
            output_dir (str | None, optional): Base output directory. Defaults to None
                (resolved from root_dir or os.getcwd()).
            root_dir (str | None, optional): Root project directory used to anchor
                relative output paths. Defaults to None.
            overwrite (bool, optional): Overwrite existing output files. Defaults to False.
            n_jobs (int, optional): Number of parallel jobs for tremor calculation
                and feature extraction. Defaults to 1.
            verbose (bool, optional): Emit progress log messages. Defaults to False.
            debug (bool, optional): Emit debug log messages. Defaults to False.
        """
        # ------------------------------------------------------------------
        # Set DEFAULT parameter
        # ------------------------------------------------------------------
        root_dir = os.path.abspath(root_dir) if root_dir is not None else None
        nslc, output_dir, station_dir, features_dir = self._setup_directories(
            network, station, location, channel, output_dir, root_dir
        )

        # ------------------------------------------------------------------
        # Set DEFAULT properties
        # ------------------------------------------------------------------
        self.station = station
        self.channel = channel
        self.window_size: int = window_size
        self.volcano_id = volcano_id
        self.network = network
        self.location = location
        self.output_dir = output_dir
        self.root_dir = root_dir
        self.overwrite = overwrite
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.debug = debug

        # ------------------------------------------------------------------
        # Set ADDITIONAL properties (derived values)
        # ------------------------------------------------------------------
        self.start_date: datetime | None = None
        self.end_date: datetime | None = None
        self.start_date_minus_window_size: datetime | None = None
        self.start_date_str: str | None = None
        self.end_date_str: str | None = None
        self.nslc = nslc
        self.station_dir = station_dir
        self.features_dir = features_dir
        self.include_eruption_date: bool = False
        self.basename: str | None = None

        # ------------------------------------------------------------------
        # Initialize state properties (set during lifecycle)
        # ------------------------------------------------------------------
        # Will be set after calculate() method called
        # ------------------------------------------------------------------
        self.CalculateTremor: CalculateTremor | None = None
        self.TremorData: TremorData | None = None
        self.tremor_data: pd.DataFrame = pd.DataFrame()
        self.tremor_csv: str | None = None

        # ------------------------------------------------------------------
        # Will be set after build_label() method called
        # ------------------------------------------------------------------
        self.LabelBuilder: LabelBuilder | None = None
        self.label_data: pd.DataFrame = pd.DataFrame()
        self.label_csv: str | None = None
        self.total_eruption_class: int | None = None
        self.total_non_eruption_class: int | None = None
        self.class_ratio: float | None = None

        # ------------------------------------------------------------------
        # Will be set after extract_features() called
        # ------------------------------------------------------------------
        self.TremorMatrixBuilder: TremorMatrixBuilder | None = None
        self.tremor_matrix_df: pd.DataFrame = pd.DataFrame()
        self.tremor_matrix_csv: str | None = None
        self.FeaturesBuilder: FeaturesBuilder | None = None
        self.features_df: pd.DataFrame = pd.DataFrame()
        self.features_csv: str | None = None
        self.label_features_csv: str | None = None
        self.use_relevant_features: bool = False
        self.select_tremor_columns: list[str] | None = None

        # ------------------------------------------------------------------
        # Will be updated after set_feature_selection_method() called
        # ------------------------------------------------------------------
        self.feature_selection_method: Literal[
            "tsfresh", "random_forest", "combined"
        ] = "tsfresh"

        # ------------------------------------------------------------------
        # Will be set after train() called
        # ------------------------------------------------------------------
        self._plot_shap = False
        self.trained_models: dict[str, str] = {}
        self.merged_models: dict[str, str] = {}

        # ------------------------------------------------------------------
        # Will be set after predict() called
        # ------------------------------------------------------------------
        self.ModelPredictor: ModelPredictor | None = None
        self.prediction_df: pd.DataFrame = pd.DataFrame()

        # ------------------------------------------------------------------
        # Pipeline configuration (accumulates params as each stage is called)
        # ------------------------------------------------------------------
        self._config = PipelineConfig(
            model=ModelConfig(
                station=station,
                channel=channel,
                window_size=window_size,
                volcano_id=volcano_id,
                network=network,
                location=location,
                output_dir=self.output_dir,
                root_dir=self.root_dir,
                overwrite=overwrite,
                n_jobs=n_jobs,
                verbose=verbose,
                debug=debug,
            )
        )
        # Set when the instance is restored via from_config()
        self._loaded_config: PipelineConfig | None = None

        # ------------------------------------------------------------------
        # Validate and create directories
        # ------------------------------------------------------------------
        self.validate()
        self.create_directories()

        # ------------------------------------------------------------------
        # Verbose and logging
        # ------------------------------------------------------------------
        if debug:
            logger.info("⚠️ Forecast Model :: Debug mode is ON")

        if verbose:
            logger.info(f"Volcano ID: {self.volcano_id}")
            logger.info(f"NSLC: {self.nslc}")
            logger.info(f"Output Dir: {self.output_dir}")

    @staticmethod
    def _setup_directories(
        network: str,
        station: str,
        location: str,
        channel: str,
        output_dir: str | None,
        root_dir: str | None = None,
    ) -> tuple[str, str, str, str]:
        """Set up directory structure for forecast model outputs.

        Creates the NSLC (Network.Station.Location.Channel) identifier and
        builds the directory structure for storing model outputs.

        Args:
            network (str): Network code (e.g., "VG").
            station (str): Station code (e.g., "OJN").
            location (str): Location code (e.g., "00").
            channel (str): Channel code (e.g., "EHZ").
            output_dir (str | None): Base output directory. If None, defaults to
                "output" relative to root_dir.
            root_dir (str | None, optional): Anchor directory for resolving relative
                paths. If None, falls back to ``os.getcwd()``. Defaults to None.

        Returns:
            tuple[str, str, str, str]: A 4-tuple containing:

                - **nslc** (str): Combined identifier (e.g., "VG.OJN.00.EHZ")
                - **output_dir** (str): Resolved output directory path
                - **station_dir** (str): Station-specific directory path
                - **features_dir** (str): Features directory path

        Examples:
            >>> nslc, out, station, features = ForecastModel._setup_directories(
            ...     "VG", "OJN", "00", "EHZ", None, "/project"
            ... )
            >>> print(nslc)  # "VG.OJN.00.EHZ"
        """
        nslc = f"{network}.{station}.{location}.{channel}"
        output_dir = resolve_output_dir(output_dir, root_dir, "output")
        station_dir = os.path.join(output_dir, nslc)
        features_dir = os.path.join(station_dir, "features")

        return nslc, output_dir, station_dir, features_dir

    def _setup_calculate_tremor(
        self,
        methods: list[str] | None,
        filename_prefix: str | None,
        remove_outlier_method: Literal["all", "maximum"],
        interpolate: bool,
        value_multiplier: float | None,
        cleanup_daily_dir: bool,
        plot_daily: bool,
        save_plot: bool,
        overwrite_plot: bool,
        remove_tremor_anomalies: bool = False,
        n_jobs: int | None = None,
        verbose: bool = False,
        debug: bool = False,
    ) -> CalculateTremor:
        """Set up CalculateTremor instance with configuration.

        Creates and configures a CalculateTremor instance with all necessary
        parameters. Automatically adjusts start_date to include window_size buffer.

        Args:
            methods (list[str] | None): Calculation methods to apply (e.g., "rsam,dsar").
            filename_prefix (str | None): Prefix for generated filenames.
            remove_outlier_method (Literal["all", "maximum"]): Method for outlier removal.
            interpolate (bool): Whether to interpolate missing data.
            value_multiplier (float | None): Scaling factor for tremor values.
            cleanup_daily_dir (bool): Whether to clean up daily temporary directory.
            plot_daily (bool): Whether to plot daily results.
            save_plot (bool): Whether to save plots to disk.
            overwrite_plot (bool): Whether to overwrite existing plots.
            remove_tremor_anomalies (bool, optional): Remove anomalies after tremor calculated.
                Using Z-score analysis to determine the anomalies. Defaults to False.
            n_jobs (int | None, optional): Number of jobs to run in parallel.
                Isolated to this method only. If None, uses ``self.n_jobs``.
                Defaults to None.
            verbose (bool, optional): Enable verbose logging. Defaults to False.
            debug (bool, optional): Enable debug mode. Defaults to False.

        Returns:
            CalculateTremor: Configured CalculateTremor instance ready for use.

        Examples:
            >>> calc = model._setup_calculate_tremor(
            ...     methods=["rsam","dsar","entropy"]
            ...     filename_prefix="tremor",
            ...     remove_outlier_method="maximum",
            ...     interpolate=True,
            ...     value_multiplier=None,
            ...     cleanup_daily_dir=True,
            ...     plot_daily=False,
            ...     save_plot=False,
            ...     overwrite_plot=False,
            ... )
        """
        verbose = verbose or self.verbose
        debug = debug or self.debug

        if self.start_date_minus_window_size is None:
            raise ValueError("self.start_date_minus_window_size cannot be None")
        if self.end_date is None:
            raise ValueError("self.end_date cannot be None")

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
            remove_tremor_anomalies=remove_tremor_anomalies,
            interpolate=interpolate,
            value_multiplier=value_multiplier,
            cleanup_daily_dir=cleanup_daily_dir,
            plot_daily=plot_daily,
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
        calculate: CalculateTremor, sds_dir: str
    ) -> CalculateTremor:
        """Calculate tremor from an SDS archive.

        Configures the calculator for SDS mode, validates the directory, and
        runs the full tremor calculation workflow.

        Args:
            calculate (CalculateTremor): Pre-configured CalculateTremor instance.
            sds_dir (str): Root path to the SDS archive directory.

        Returns:
            CalculateTremor: The same instance after running the calculation.

        Raises:
            ValueError: If sds_dir is None or does not exist on disk.
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
        """Calculate tremor from an FDSN web service.

        Configures the calculator for FDSN mode and runs the full tremor
        calculation workflow, downloading and caching data as needed.

        Args:
            calculate (CalculateTremor): Pre-configured CalculateTremor instance.
            client_url (str): FDSN web-service base URL.

        Returns:
            CalculateTremor: The same instance after running the calculation.
        """
        if self.verbose:
            logger.info(f"Calculating tremor using FDSN with client URL: {client_url}")

        return calculate.from_fdsn(client_url=client_url).run()

    def _adjust_dates_to_tremor_range(self, tremor_data: TremorData) -> None:
        """Adjust start_date and end_date to match available tremor data.

        Updates self.start_date, self.end_date, and their string representations
        if they fall outside the tremor data range. Logs changes when verbose mode
        is enabled.

        Args:
            tremor_data (TremorData): Loaded tremor data wrapper with date range info.
        """
        # Adjust start date if earlier than tremor start
        if (
            isinstance(self.start_date_minus_window_size, datetime)
            and self.start_date_minus_window_size < tremor_data.start_date
        ):
            self.start_date = tremor_data.start_date
            self.start_date_str = tremor_data.start_date_str
            if self.verbose:
                logger.info(
                    f"start_date parameter: {self.start_date_minus_window_size} updated to "
                    f"tremor start date: {self.start_date_str}"
                )

        # Adjust end date if later than tremor end
        if isinstance(self.end_date, datetime) and self.end_date > tremor_data.end_date:
            self.end_date = tremor_data.end_date
            self.end_date_str = tremor_data.end_date_str
            if self.verbose:
                logger.info(
                    f"end_date parameter: {self.end_date} updated to "
                    f"tremor end date: {self.end_date_str}"
                )

    @staticmethod
    def _validate_tremor_for_labeling(
        tremor_data: pd.DataFrame,
        tremor_columns: list[str] | None,
    ) -> None:
        """Validate that tremor data is available for label building.

        Checks that tremor data is a non-empty DataFrame and that all specified
        column names are present.

        Args:
            tremor_data (pd.DataFrame): Tremor metrics DataFrame to validate.
            tremor_columns (list[str] | None): Column names to check, or None to
                skip column validation.

        Raises:
            ValueError: If tremor_data is not a DataFrame, is empty, or any
                column in tremor_columns is missing.
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

        Creates a sorted copy of tremor data and optionally subsets it to the
        specified columns.

        Args:
            tremor_columns (list[str] | None): Column names to retain, or None
                to keep all columns.

        Returns:
            pd.DataFrame: Sorted copy of the tremor DataFrame.
        """
        df_tremor = self.tremor_data.copy()

        if tremor_columns is not None:
            df_tremor = df_tremor[tremor_columns]

        df_tremor = df_tremor.sort_index(ascending=True)

        return df_tremor

    @staticmethod
    def _validate_label_tremor_date_range(
        df_label: pd.DataFrame,
        df_tremor: pd.DataFrame,
    ) -> None:
        """Validate that label date range falls within the tremor data range.

        Ensures label windows are fully covered by available tremor data,
        preventing silent gaps during feature extraction.

        Args:
            df_label (pd.DataFrame): Label DataFrame with DatetimeIndex.
            df_tremor (pd.DataFrame): Tremor DataFrame with DatetimeIndex.

        Raises:
            ValueError: If label start date precedes tremor start date, or
                label end date exceeds tremor end date.
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

        Computes the count of eruption and non-eruption windows and their ratio,
        storing results in ``self.total_eruption_class``,
        ``self.total_non_eruption_class``, and ``self.class_ratio``.
        Logs the statistics when verbose mode is enabled.

        Args:
            label_builder (LabelBuilder): Fully built LabelBuilder instance.
        """
        self.total_eruption_class = len(label_builder.df_eruption)
        self.total_non_eruption_class = (
            len(label_builder.df) - self.total_eruption_class
        )
        self.class_ratio: float = (
            self.total_eruption_class / self.total_non_eruption_class
        )

        if self.verbose:
            logger.info(
                f"Total number of eruptions: {self.total_eruption_class}. "
                f"Total number of non-eruptions: {self.total_non_eruption_class}. "
                f"Class ratio (eruption vs non-eruptions): {self.class_ratio}"
            )

    def validate(self) -> None:
        """Validate initialization parameters.

        Ensures that window_size is positive and required string parameters
        (station, channel, volcano_id, network, location) are not empty.
        Date range validation is deferred to ``calculate()``.

        Raises:
            ValueError: If window_size is <= 0, or required string parameters
                (station, channel, volcano_id, network, location) are empty.

        Example:
            >>> model = ForecastModel(...)
            >>> model.validate()  # Called automatically in __init__
        """
        # Validate window size
        if self.window_size <= 0:
            raise ValueError(
                f"window_size must be greater than 0. Got: {self.window_size}"
            )

        # Validate strings are not empty
        if not self.station.strip():
            raise ValueError("station cannot be empty")

        if not self.channel.strip():
            raise ValueError("channel cannot be empty")

        if not self.volcano_id.strip():
            raise ValueError("volcano_id cannot be empty")

        if not self.network.strip():
            raise ValueError("network cannot be empty")

        if not self.location.strip():
            raise ValueError("location cannot be empty")

        if self.n_jobs <= 0:
            raise ValueError(f"n_jobs must be greater than 0. Got: {self.n_jobs}")

    def create_directories(self) -> None:
        """Create required output directory structure.

        Creates the main output directory, station-specific directory,
        and features subdirectory. Called automatically during initialization.

        Example:
            >>> model = ForecastModel(...)
            >>> model.create_directories()  # Called in __init__
            >>> # Creates: output/, output/VG.OJN.00.EHZ/, output/VG.OJN.00.EHZ/features/
        """
        ensure_dir(self.output_dir)
        ensure_dir(self.station_dir)
        ensure_dir(self.features_dir)

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
                Sets ``self.tremor_data``, ``self.TremorData``, ``self.tremor_csv``,
                ``self.start_date``, ``self.end_date``, ``self.start_date_str``,
                ``self.end_date_str``, and ``self.start_date_minus_window_size``
                from the loaded tremor DataFrame's datetime index.

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

        self.start_date = self.tremor_data.index[0].to_pydatetime()
        self.end_date = self.tremor_data.index[-1].to_pydatetime()
        self.start_date_str = self.start_date.strftime("%Y-%m-%d")
        self.end_date_str = self.end_date.strftime("%Y-%m-%d")
        self.start_date_minus_window_size = self.start_date - timedelta(
            days=self.window_size
        )

        return self

    def calculate(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        source: Literal["sds", "fdsn"] = "sds",
        methods: list[str] | None = None,
        filename_prefix: str | None = None,
        remove_outlier_method: Literal["all", "maximum"] = "maximum",
        remove_tremor_anomalies: bool = False,
        interpolate: bool = True,
        value_multiplier: float | None = None,
        cleanup_daily_dir: bool = False,
        plot_daily: bool = False,
        save_plot: bool = False,
        overwrite_plot: bool = False,
        sds_dir: str | None = None,
        client_url: str = "https://service.iris.edu",
        n_jobs: int | None = None,
        verbose: bool = False,
        debug: bool = False,
    ) -> Self:
        """Calculate tremor metrics from a seismic data source.

        Delegates to ``CalculateTremor`` to compute RSAM, DSAR, and Shannon
        Entropy metrics from either a local SDS archive or an FDSN web service.
        The resulting tremor DataFrame is stored in ``self.tremor_data`` and
        the pipeline date range is clipped to the available data.

        Args:
            start_date (str | datetime): Start date for the tremor calculation period
                in "YYYY-MM-DD" format or as a datetime object.
            end_date (str | datetime): End date for the tremor calculation period
                in "YYYY-MM-DD" format or as a datetime object.
            source (Literal["sds", "fdsn"]): Seismic data source. Defaults to "sds".
            methods (list[str] | None): Calculation methods to apply. Defaults to None.
            filename_prefix (str | None): Prefix for generated filenames. Defaults to None.
            remove_outlier_method (Literal["all", "maximum"]): Outlier removal method.
                Defaults to "maximum".
            remove_tremor_anomalies (bool, optional): Remove anomalies after tremor calculated.
                Using Z-score analysis to determine the anomalies. Defaults to False.
            interpolate (bool): If True, interpolates gaps in the data. Defaults to True.
            value_multiplier (float | None): Scaling factor for seismic values. Defaults to None.
            cleanup_daily_dir (bool): If True, deletes the daily directory after merging.
                Defaults to False.
            plot_daily (bool): If True, plots each daily result for a quick visual check.
                Defaults to False.
            save_plot (bool): If True, saves the final tremor plot. Defaults to False.
            overwrite_plot (bool): If True, overwrites existing plot files. Defaults to False.
            sds_dir (str | None): Path to the SDS data directory. Required when
                ``source="sds"``. Defaults to None.
            client_url (str): FDSN web-service URL. Defaults to
                ``"https://service.iris.edu"``.
            n_jobs (int | None): Parallel workers for this call only. Overrides the
                instance-level ``n_jobs`` when provided. Defaults to None.
            verbose (bool): If True, enables verbose logging. Defaults to False.
            debug (bool): If True, enables debug mode. Defaults to False.

        Returns:
            Self: ForecastModel instance for method chaining.
        """
        # Normalise and store dates on self
        _start, _end, _start_str, _end_str = normalize_dates(start_date, end_date)
        validate_date_ranges(_start, _end)
        self.start_date = _start
        self.end_date = _end
        self.start_date_str = _start_str
        self.end_date_str = _end_str
        self.start_date_minus_window_size = _start - timedelta(days=self.window_size)

        if verbose or self.verbose:
            logger.info(f"Start Date: {_start_str}")
            logger.info(f"End Date: {_end_str}")

        # Setup CalculateTremor instance
        calculate = self._setup_calculate_tremor(
            methods=methods,
            filename_prefix=filename_prefix,
            remove_outlier_method=remove_outlier_method,
            remove_tremor_anomalies=remove_tremor_anomalies,
            interpolate=interpolate,
            value_multiplier=value_multiplier,
            cleanup_daily_dir=cleanup_daily_dir,
            plot_daily=plot_daily,
            save_plot=save_plot,
            overwrite_plot=overwrite_plot,
            n_jobs=n_jobs or self.n_jobs,
            verbose=verbose or self.verbose,
            debug=debug or self.debug,
        )

        self.CalculateTremor = calculate

        # Calculate from appropriate source
        if source == "sds" and sds_dir is not None:
            calculate = self._calculate_from_sds(calculate, sds_dir)
        elif source == "fdsn":
            calculate = self._calculate_from_fdsn(calculate, client_url)
        else:
            raise ValueError(f"Unknown source '{source}'. Choose 'sds' or 'fdsn'.")

        # Wrap calculated data
        tremor_data = TremorData(calculate.df)
        self.TremorData = tremor_data
        self.TremorData.csv = calculate.csv
        self.tremor_csv = calculate.csv

        # Adjust dates to match tremor data availability
        self._adjust_dates_to_tremor_range(tremor_data)

        # Slice tremor data to adjusted date range
        self.tremor_data = tremor_data.df.loc[self.start_date : self.end_date]

        self._config.calculate = CalculateConfig(
            start_date=_start_str,
            end_date=_end_str,
            source=source,
            sds_dir=sds_dir,
            methods=methods,
            filename_prefix=filename_prefix,
            remove_outlier_method=remove_outlier_method,
            remove_tremor_anomalies=remove_tremor_anomalies,
            interpolate=interpolate,
            value_multiplier=value_multiplier,
            cleanup_daily_dir=cleanup_daily_dir,
            plot_daily=plot_daily,
            save_plot=save_plot,
            overwrite_plot=overwrite_plot,
            client_url=client_url,
            n_jobs=n_jobs,
            verbose=verbose,
            debug=debug,
        )

        return self

    def extract_features(
        self,
        select_tremor_columns: list[str] | None = None,
        save_tremor_matrix_per_method: bool = True,
        save_tremor_matrix_per_id: bool = False,
        exclude_features: list[str] | None = None,
        use_relevant_features: bool = False,
        output_dir: str | None = None,
        overwrite: bool = False,
        n_jobs: int | None = None,
        verbose: bool | None = None,
    ) -> Self:
        """Extract features from tremor data using tsfresh.

        Applies time-series feature extraction to each tremor column, either
        extracting all comprehensive features or only statistically relevant
        features based on correlation with eruption labels.

        Args:
            select_tremor_columns (list[str] | None): Tremor columns to use for feature
                extraction. Uses all available columns when None. Defaults to None.
            save_tremor_matrix_per_method (bool): Save a separate tremor-matrix CSV for
                each tremor column. Defaults to True.
            save_tremor_matrix_per_id (bool): **WARNING: generates one file per label
                window** — use only for debugging. Defaults to False.
            exclude_features (list[str] | None): tsfresh feature calculator names to skip.
                Defaults to None.
            use_relevant_features (bool): If True, run tsfresh with relevance filtering
                (requires labels). Defaults to False.
            output_dir (str | None): Output directory for feature files. Defaults to
                ``self.features_dir``.
            overwrite (bool): If True, overwrite existing feature files. Defaults to False.
            n_jobs (int | None): Parallel workers for tsfresh extraction. Overrides the
                instance-level ``n_jobs`` when provided. Defaults to None.
            verbose (bool | None, optional): If True, enables verbose logging. Defaults to None.

        Returns:
            Self: ForecastModel instance for method chaining.
        """
        output_dir = output_dir or self.features_dir
        ensure_dir(output_dir)

        tremor_matrix_builder = TremorMatrixBuilder(
            tremor_df=self.tremor_data,
            label_df=self.label_data,
            output_dir=output_dir,
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
            label_features_basename=self.basename,
            output_dir=self.features_dir,
            overwrite=overwrite or self.overwrite,
            n_jobs=n_jobs or self.n_jobs,
            verbose=verbose or self.verbose,
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
        self.label_features_csv = features_builder.label_features_csv
        self.use_relevant_features = use_relevant_features
        self.select_tremor_columns = select_tremor_columns

        self._config.extract_features = ExtractFeaturesConfig(
            select_tremor_columns=select_tremor_columns,
            save_tremor_matrix_per_method=save_tremor_matrix_per_method,
            save_tremor_matrix_per_id=save_tremor_matrix_per_id,
            exclude_features=exclude_features,
            use_relevant_features=use_relevant_features,
            overwrite=overwrite,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        return self

    @staticmethod
    def _label_builder(
        builder: Literal["standard", "dynamic"],
        label_kwargs: dict[str, Any],
        days_before_eruption: int | None = None,
        train_start_date: str | datetime | None = None,
        train_end_date: str | datetime | None = None,
    ) -> LabelBuilder:
        """Instantiate and build a label builder of the requested type.

        Dispatches to either ``DynamicLabelBuilder`` (one window per eruption)
        or ``LabelBuilder`` (one global window spanning the full date range)
        based on the ``builder`` argument, then calls ``.build()`` and returns
        the resulting instance.

        Args:
            builder (Literal["standard", "dynamic"]): Label builder variant.
                ``"standard"`` uses a single global window; ``"dynamic"``
                generates one window per eruption event.
            label_kwargs (dict[str, Any]): Shared keyword arguments forwarded
                to the builder constructor (e.g. window_step, eruption_dates).
            days_before_eruption (int | None): Days before each eruption to
                start its window. Required when ``builder="dynamic"``.
                Defaults to None.
            train_start_date (str | datetime | None): Label window start date.
                Required when ``builder="standard"``. Defaults to None.
            train_end_date (str | datetime | None): Label window end date.
                Required when ``builder="standard"``. Defaults to None.

        Returns:
            LabelBuilder: A fully built label builder instance.

        Raises:
            ValueError: If ``builder="dynamic"`` and ``days_before_eruption``
                is None, or if ``builder="standard"`` and either
                ``train_start_date`` or ``train_end_date`` is None.
        """
        if builder == "dynamic":
            if days_before_eruption is None:
                raise ValueError(
                    "days_before_eruption is required when builder='dynamic'."
                )
            label_builder = DynamicLabelBuilder(
                days_before_eruption=days_before_eruption,
                **label_kwargs,
            ).build()

            return label_builder

        if train_start_date is None:
            raise ValueError(
                "start_date is required for builder='standard'. "
                "Either call calculate() first or pass start_date explicitly."
            )
        if train_end_date is None:
            raise ValueError(
                "end_date is required for builder='standard'. "
                "Either call calculate() first or pass end_date explicitly."
            )

        validate_date_ranges(train_start_date, train_end_date)
        label_builder = LabelBuilder(
            start_date=to_datetime(train_start_date),
            end_date=to_datetime(train_end_date),
            **label_kwargs,
        ).build()

        return label_builder

    def build_label(
        self,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        day_to_forecast: int,
        eruption_dates: list[str],
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
        include_eruption_date: bool = False,
        output_dir: str | None = None,
        tremor_columns: list[str] | None = None,
        builder: Literal["standard", "dynamic"] = "standard",
        days_before_eruption: int | None = None,
        verbose: bool | None = None,
        debug: bool | None = None,
    ) -> Self:
        """Build labels for eruption forecasting.

        Creates labeled time windows for training machine learning models.
        Each window is labeled as erupted (1) or not erupted (0) based on
        eruption dates and forecast horizon.

        Two builder modes are supported:

        - ``"standard"``: one global window spanning ``start_date`` to ``end_date``.
        - ``"dynamic"``: one window per eruption, each spanning
          ``days_before_eruption`` days before the event.

        Args:
            window_step (int): Window step size.
            window_step_unit (Literal["minutes", "hours"]): Unit of window step.
            day_to_forecast (int): Day to forecast in days.
            eruption_dates (list[str]): Eruption dates in YYYY-MM-DD format.
            start_date (str, optional): Override self.start_date. Used only when
                ``builder="standard"``.
            end_date (str, optional): Override self.end_date. Used only when
                ``builder="standard"``.
            include_eruption_date (bool, optional): Date of eruption will marked
                as an eruption (not excluded). Defaults to False.
            output_dir (Optional[str], optional): Output directory. Defaults to None.
            tremor_columns (Optional[list[str]], optional): Columns to select. Defaults to None.
            builder (Literal["standard", "dynamic"]): Label builder type. Defaults to
                ``"standard"``.
            days_before_eruption (int | None): Days before each eruption to start its
                window. Required when ``builder="dynamic"``. Defaults to None.
            verbose (bool | None, optional): If True, enables verbose logging. Defaults to None.
            debug (bool | None, optional): If True, enables debug mode. Defaults to None.

        Returns:
            Self: ForecastModel instance for method chaining.

        Raises:
            ValueError: If ``builder="dynamic"`` and ``days_before_eruption`` is None.
        """
        # Setup parameters
        tremor_data = self.tremor_data
        train_start_date = start_date or self.start_date
        train_end_date = end_date or self.end_date
        verbose = verbose or self.verbose
        debug = debug or self.debug

        output_dir = output_dir or self.station_dir
        ensure_dir(output_dir)

        # Validate inputs
        self._validate_tremor_for_labeling(tremor_data, tremor_columns)

        # Shared kwargs for both builders
        label_kwargs = {
            "window_step": window_step,
            "window_step_unit": window_step_unit,
            "day_to_forecast": day_to_forecast,
            "eruption_dates": eruption_dates,
            "volcano_id": self.volcano_id,
            "include_eruption_date": include_eruption_date,
            "output_dir": output_dir,
            "root_dir": self.root_dir,
            "verbose": verbose,
            "debug": debug,
        }

        # Build labels
        label_builder = self._label_builder(
            builder=builder,
            label_kwargs=label_kwargs,
            days_before_eruption=days_before_eruption,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
        )

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
        if builder == "standard" and train_start_date is not None:
            df_label = df_label.loc[train_start_date:]

        if df_label.empty:
            raise ValueError(f"Label from start date {self.start_date} is empty.")

        self.label_data = df_label

        # Calculate and log statistics
        self._calculate_eruption_statistics(label_builder)

        self._config.build_label = BuildLabelConfig(
            window_step=window_step,
            window_step_unit=window_step_unit,
            day_to_forecast=day_to_forecast,
            eruption_dates=eruption_dates,
            start_date=str(train_start_date) if start_date is not None else None,
            end_date=str(train_end_date) if end_date is not None else None,
            include_eruption_date=include_eruption_date,
            tremor_columns=tremor_columns,
            builder=builder,
            days_before_eruption=days_before_eruption,
            verbose=bool(verbose),
            debug=bool(debug),
        )

        return self

    def set_feature_selection_method(
        self,
        using: Literal["tsfresh", "random_forest", "combined"],
    ) -> Self:
        """Change the feature selection method used during training.

        Updates ``self.feature_selection_method``, which is read by ``train()``
        when it constructs the ``ModelTrainer``. Call this before ``train()``
        to switch strategies without re-extracting features.

        Args:
            using (Literal["tsfresh", "random_forest", "combined"]): Feature
                selection strategy:
                - ``"tsfresh"``: Statistical significance filtering only.
                - ``"random_forest"``: Permutation importance only.
                - ``"combined"``: Two-stage — tsfresh then RandomForest.

        Returns:
            Self: ForecastModel instance for method chaining.
        """
        self.feature_selection_method = using
        return self

    def _train_all_classifiers(
        self,
        classifiers: list[str],
        cv_strategy: Literal[
            "shuffle", "stratified", "shuffle-stratified", "timeseries"
        ],
        features_csv: str,
        label_csv: str,
        output_dir: str,
        train_params: dict[str, Any],
        with_evaluation: bool = False,
        number_of_significant_features: int = 20,
        grid_params: dict[str, Any] | None = None,
        n_jobs: int = 1,
        grid_search_n_jobs: int = 1,
        overwrite: bool = False,
        use_gpu: bool = False,
        gpu_id: int = 0,
        verbose: bool = False,
    ) -> dict[str, str]:
        """Train all classifiers with shared under-sampling and feature selection.

        Creates a single ``ModelTrainer`` for all classifiers so that under-sampling
        and feature selection are performed once per seed and reused across classifiers,
        eliminating redundant computation. Runs ``fit()`` on the shared trainer, then
        builds and persists per-classifier training results in ``self.trained_models``
        as a mapping of classifier name to trained-model registry CSV path, and
        returns the same mapping.

        Args:
            classifiers (list[str]): List of classifier keys (e.g. ``["rf", "xgb"]``).
            cv_strategy (Literal["shuffle", "stratified", "shuffle-stratified", "timeseries"]):
                Cross-validation strategy passed to ``ModelTrainer``.
            features_csv (str): Path to the extracted features CSV file.
            label_csv (str): Path to the aligned labels CSV file.
            output_dir (str): Root directory for training outputs.
            train_params (dict[str, Any]): Keyword arguments forwarded to
                ``ModelTrainer.fit()`` (e.g. ``random_state``, ``total_seed``).
            with_evaluation (bool, optional): If True, performs an 80/20 train/test
                split and computes evaluation metrics. Defaults to False.
            number_of_significant_features (int, optional): Top-N features retained
                per seed. Defaults to 20.
            grid_params (dict[str, Any] | None, optional): Custom hyperparameter grid
                for GridSearchCV applied to all classifiers. If None, each classifier's
                default grid is used. Defaults to None.
            n_jobs (int, optional): Parallel workers for the outer seed loop.
                Defaults to 1.
            grid_search_n_jobs (int, optional): Number of parallel jobs inside each
                GridSearchCV call (inner loop). Defaults to 1.
            overwrite (bool, optional): If True, overwrites existing output files.
                Defaults to False.
            verbose (bool, optional): If True, enables verbose logging. Defaults to False.
            use_gpu (bool, optional): Enable GPU acceleration for XGBoost. Defaults to False.
            gpu_id (int, optional): GPU device index to use when use_gpu is True.
                Defaults to 0.

        Returns:
            dict[str, str]: Mapping of classifier name to trained-model registry CSV path.
        """
        train_model = ModelTrainer(
            extracted_features_csv=features_csv,
            label_features_csv=label_csv,
            output_dir=output_dir,
            classifiers=classifiers,
            cv_strategy=cv_strategy,
            number_of_significant_features=number_of_significant_features,
            feature_selection_method=self.feature_selection_method,
            overwrite=overwrite,
            plot_shap=self._plot_shap,
            n_jobs=n_jobs,
            grid_search_n_jobs=grid_search_n_jobs,
            verbose=verbose,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
        )

        # Override default grid search parameters for all classifiers
        if grid_params is not None:
            for clf_model in train_model.classifier_models:
                clf_model.grid = grid_params

        merged_models = train_model.fit(
            with_evaluation=with_evaluation, **train_params
        ).merge_models()

        # Build result mapping: classifier_name -> registry CSV path
        trained_models: dict[str, str] = {}
        for clf_model in train_model.classifier_models:
            clf_name = clf_model.name
            clf_slug = clf_model.slug_name
            csv_path = train_model.csv.get(clf_slug)
            if csv_path is not None:
                trained_models[clf_name] = csv_path

        self.merged_models = merged_models

        return trained_models

    def train(
        self,
        classifier: str | list[str] = "rf",
        cv_strategy: Literal[
            "shuffle", "stratified", "shuffle-stratified", "timeseries"
        ] = "shuffle-stratified",
        random_state: int = 0,
        total_seed: int = 500,
        with_evaluation: bool = False,
        grid_params: dict[str, Any] | None = None,
        number_of_significant_features: int = 20,
        sampling_strategy: str | float = 0.75,
        save_all_features: bool = False,
        plot_significant_features: bool = False,
        extracted_features_csv: str | None = None,
        output_dir: str | None = None,
        n_jobs: int | None = None,
        grid_search_n_jobs: int = 1,
        plot_shap: bool = False,
        overwrite: bool = False,
        use_gpu: bool = False,
        gpu_id: int = 0,
        verbose: bool = False,
        save_model: bool = True,
    ) -> Self:
        """Training model using extracted features and labels.

        Supported classifiers:
            - svm: Support Vector Machine (SVC with balanced class weights)
            - knn: K-Nearest Neighbors
            - dt: Decision Tree (with balanced class weights)
            - rf: Random Forest (with balanced class weights)
            - gb: Gradient Boosting (handles imbalanced data well)
            - nn: Multi-Layer Perceptron Neural Network
            - nb: Gaussian Naive Bayes
            - lr: Logistic Regression (with balanced class weights)
            - xgb: XGBoost classifier (excellent for imbalanced data)
            - voting: Ensemble VotingClassifier combining rf and xgb
            - lite-rf: Random Forest but faster with more simple grid parameters

        Args:
            classifier (str | list[str], optional): Classifier type or list of classifier
                types to train sequentially. Supported values: ``"svm"``, ``"knn"``,
                ``"dt"``, ``"rf"``, ``"gb"``, ``"xgb"``, ``"nn"``, ``"nb"``,
                ``"lr"``, ``"voting"``, ``"lite-rf"``. A comma-separated string
                (e.g. ``"rf,xgb"``) is also accepted. Defaults to ``"rf"``.
            cv_strategy (str, optional): Cross-validation strategy ("shuffle", "stratified",
                "shuffle-stratified", "timeseries"). Defaults to "shuffle-stratified".
            random_state (int, optional): Initial random seed. Defaults to 0.
            total_seed (int, optional): Total number of random seeds. Defaults to 500.
            with_evaluation (bool, optional): If True, performs an 80/20 train/test split
                and computes evaluation metrics per seed. Requires labels. Set to False
                to train on the full dataset without metrics. Defaults to False.
            grid_params (dict[str, Any], optional): Override default hyperparameter grid
                for GridSearchCV. Defaults to None.
            number_of_significant_features (int, optional): Number of top features to retain
                per seed. Defaults to 20.
            sampling_strategy (str | float, optional): Under-sampling ratio for balancing
                classes. Defaults to 0.75.
            save_all_features (bool, optional): Whether to save ALL features. Defaults to False.
            plot_significant_features (bool, optional): Whether to plot each significant feature. Defaults to False.
            extracted_features_csv (str | None): Path to extracted features.
            output_dir (str, optional): Path to output directory. Defaults to None.
            n_jobs (int, optional): Number of jobs. Defaults to None.
            grid_search_n_jobs (int, optional): Number of parallel jobs inside each
                GridSearchCV call (inner loop). Safe to set > 1 because the loky
                backend handles nested parallelism without deadlocks. Combine with
                ``n_jobs`` to control the total CPU budget:
                ``n_jobs × grid_search_n_jobs ≤ total_cores``. Defaults to 1.
            plot_shap (bool, optional): Plot SHAP explanation value. Defaults to False.
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
            save_model (bool, optional): If True, serialises the full ``ForecastModel``
                instance to ``{station_dir}/trainings/{evaluations/predictions_dir}/forecast_model.pkl`` via ``save_model()``.
                Defaults to True.
            use_gpu (bool, optional): Enable GPU acceleration for XGBoost. Has no effect
                for other classifiers. Defaults to False.
            gpu_id (int, optional): GPU device index to use when use_gpu is True
                (e.g. 0 for the first GPU, 1 for the second). Defaults to 0.

        Returns:
            Self: ForecastModel instance for method chaining.
        """
        if isinstance(classifier, str):
            classifiers: list[str] = list(classifier.split(","))
        elif isinstance(classifier, list):
            for _classifier in classifier:
                if _classifier not in CLASSIFIERS:
                    raise ValueError(
                        f"Classifier {_classifier} not supported. Choose from {CLASSIFIERS}"
                    )
            classifiers = classifier
        else:
            raise TypeError(
                f"Classifier ({classifier}) type `{type(classifier)}` not supported."
            )

        if verbose or self.verbose:
            logger.info("=" * 50)
            logger.info(f"| Training model using: {classifier}")
            if self.use_relevant_features:
                logger.info("|- Relevant Features selected.")
            # Model evaluation only works if self.label_data is not empty
            if not self.label_data.empty and with_evaluation:
                logger.info("|- Training model for evaluation.")
            if self.label_data.empty and with_evaluation:
                logger.info("|- Label is empty. Model evaluation will be set to False.")
                with_evaluation = False
            if not with_evaluation:
                logger.info("|- Training for prediction.")
            logger.info("=" * 50)

        features_csv = extracted_features_csv or self.features_csv

        if features_csv is None or not os.path.exists(features_csv):
            error_msg = f"Features CSV not found: {features_csv}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        label_csv = self.label_features_csv
        if label_csv is None or not os.path.exists(label_csv):
            error_msg = f"Label features CSV not found: {label_csv}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        output_dir = output_dir or self.station_dir

        train_params: dict[str, Any] = {
            "random_state": random_state,
            "total_seed": total_seed,
            "sampling_strategy": sampling_strategy,
            "save_all_features": save_all_features,
            "plot_significant_features": plot_significant_features,
        }

        # Train all classifiers together — shared under-sampling and feature selection
        # Reset accumulated state from any prior train() call
        self._plot_shap = plot_shap
        trained_models: dict[str, str] = self._train_all_classifiers(
            classifiers=classifiers,
            cv_strategy=cv_strategy,
            features_csv=features_csv,
            label_csv=label_csv,
            output_dir=output_dir,
            train_params=train_params,
            with_evaluation=with_evaluation,
            number_of_significant_features=number_of_significant_features,
            grid_params=grid_params,
            n_jobs=n_jobs or self.n_jobs,
            grid_search_n_jobs=grid_search_n_jobs,
            overwrite=overwrite or self.overwrite,
            verbose=verbose or self.verbose,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
        )

        prefix = "evaluations" if with_evaluation else "predictions"

        # `<cwd>/output/<station_dir>/trainings/<prefix>
        trainings_dir = os.path.join(output_dir, "trainings", prefix)
        ensure_dir(trainings_dir)

        # Save trained models JSON
        # Example format:
        # {
        #     "RandomForestClassifier": "/path/to/trained_model_rf.csv",
        #     "XGBClassifier": "/path/to/trained_model_xgb.csv"
        # }
        trained_models_filepath = os.path.join(
            trainings_dir, f"{prefix}_trained_models.json"
        )
        with open(trained_models_filepath, "w") as f:
            json.dump(trained_models, f, indent=4)

        self._config.train = TrainConfig(
            classifiers=classifiers,
            cv_strategy=cv_strategy,
            random_state=random_state,
            total_seed=total_seed,
            with_evaluation=with_evaluation,
            number_of_significant_features=number_of_significant_features,
            sampling_strategy=float(sampling_strategy)
            if isinstance(sampling_strategy, (int, float))
            else 0.75,
            save_all_features=save_all_features,
            plot_significant_features=plot_significant_features,
            n_jobs=n_jobs,
            grid_search_n_jobs=grid_search_n_jobs,
            overwrite=overwrite,
            verbose=verbose,
            plot_shap=plot_shap,
            save_model=save_model,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
        )

        self.trained_models = trained_models

        if save_model:
            self.save_config(os.path.join(trainings_dir, f"{prefix}_config.yaml"))

            self.save_model(
                path=os.path.join(trainings_dir, f"{prefix}_forecast_model.pkl")
            )

        return self

    def forecast(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        save_predictions: bool = True,
        threshold: float = 0.7,
        save_plot: bool = True,
        output_dir: str | None = None,
        n_jobs: int | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> Self:
        """Generate probabilistic eruption forecasts for a future date range.

        Constructs a ``ModelPredictor`` from the trained model registry, runs
        ``predict_proba()`` on the provided tremor data for the specified future
        window, and stores results in ``self.prediction_df``. Saves the pipeline
        configuration to ``config.yaml`` after completion.

        Args:
            start_date (str | datetime): Start date for forecasting windows.
            end_date (str | datetime): End date for forecasting windows.
            window_step (int): Step size between consecutive forecast windows.
            window_step_unit (Literal["minutes", "hours"]): Unit of window step.
            save_predictions (bool, optional): If True, saves the prediction DataFrame
                to a CSV file. Defaults to True.
            threshold (float, optional): Threshold for classifying eruption
                probability as positive. Defaults to 0.7.
            save_plot (bool, optional): If True, saves the forecast probability plot.
                Defaults to True.
            output_dir (str | None, optional): Directory for forecast output files.
                Defaults to ``self.station_dir``.
            n_jobs (int | None, optional): Parallel workers for feature extraction.
                Defaults to None (uses ``self.n_jobs``).
            overwrite (bool, optional): If True, overwrites existing output files.
                Defaults to False.
            verbose (bool, optional): If True, enables verbose logging. Defaults to False.

        Returns:
            Self: ForecastModel instance for method chaining.
        """
        verbose = verbose or self.verbose
        output_dir = output_dir or self.station_dir
        overwrite = overwrite or self.overwrite
        n_jobs = n_jobs or self.n_jobs

        trained_models = self.trained_models
        if len(self.merged_models) > 0:
            trained_models = self.merged_models

        if trained_models is None or len(trained_models) == 0:
            raise ValueError(
                "Trained models are not provided. Run train() method first or provide trained_models parameter."
                "Example: ForecastModel(...).forecast(trained_models={'rf': 'path to trained model CSV'}"
            )

        if verbose:
            logger.info("Starting Prediction...")

        model_predictor = ModelPredictor(
            start_date=start_date,
            end_date=end_date,
            trained_models=trained_models,
            output_dir=output_dir,
            n_jobs=n_jobs,
            verbose=verbose,
            overwrite=overwrite,
        )

        df_prediction = model_predictor.predict_proba(
            tremor_data=self.tremor_data,
            window_size=self.window_size,
            window_step=window_step,
            window_step_unit=window_step_unit,
            select_tremor_columns=self.select_tremor_columns,
            threshold=threshold,
            save_predictions=save_predictions,
            plot=save_plot,
        )

        self.ModelPredictor = model_predictor
        self.prediction_df = df_prediction

        self._config.forecast = ForecastConfig(
            start_date=str(to_datetime(start_date).date()),
            end_date=str(to_datetime(end_date).date()),
            window_step=window_step,
            window_step_unit=window_step_unit,
            save_predictions=save_predictions,
            threshold=threshold,
            save_plot=save_plot,
            n_jobs=n_jobs,
            overwrite=overwrite,
            verbose=verbose,
        )

        self.save_config()

        return self

    # ------------------------------------------------------------------
    # Configuration persistence helpers
    # ------------------------------------------------------------------

    def save_config(
        self,
        path: str | None = None,
        fmt: Literal["yaml", "json"] = "yaml",
    ) -> str:
        """Save the accumulated pipeline configuration to disk.

        Writes all parameters that have been set so far (model init plus any
        stage that has been called) to a YAML or JSON file.

        Args:
            path (str | None): Destination file path. Defaults to
                ``{station_dir}/config.yaml`` (or ``config.json`` when
                ``fmt="json"``).
            fmt (Literal["yaml", "json"]): Output format. Defaults to
                ``"yaml"``.

        Returns:
            str: The path where the file was written.
        """
        if path is None:
            ext = "json" if fmt == "json" else "yaml"
            path = os.path.join(self.station_dir, f"config_forecast.{ext}")
        return self._config.save(path, fmt=fmt)

    @classmethod
    def from_config(cls, path: str) -> Self:
        """Construct a ``ForecastModel`` from a previously saved config file.

        Loads the YAML or JSON config, creates a new ``ForecastModel`` from
        the ``model`` section, and attaches the full config so that
        ``run()`` can replay all stages.

        Args:
            path (str): Path to the config file produced by ``save_config()``.

        Returns:
            Self: A new instance initialised from the saved config.
                Call ``run()`` on the result to replay the full pipeline.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        config = PipelineConfig.load(path)
        fm = cls(**config.model.to_dict())
        fm._loaded_config = config
        return fm

    def save_model(self, path: str | None = None) -> str:
        """Serialise the full ``ForecastModel`` instance to disk via joblib.

        Saving the entire object allows resuming from any pipeline stage
        without re-running earlier steps. The default destination is
        ``{station_dir}/forecast_model.pkl``.

        Args:
            path (str | None): Destination file path. Defaults to
                ``{station_dir}/forecast_model.pkl``.

        Returns:
            str: The path where the file was written.
        """
        path = path or os.path.join(self.station_dir, "forecast_model.pkl")
        joblib.dump(self, path)
        return path

    @classmethod
    def load_model(cls, path: str) -> Self:
        """Restore a ``ForecastModel`` instance from a joblib pickle file.

        All pipeline state (tremor data, labels, features, trained models)
        is restored from the saved object, so you can call further pipeline
        methods (e.g. ``forecast()``) without re-running prior stages.

        Args:
            path (str): Path to a ``forecast_model.pkl`` file produced by
                ``save_model()``.

        Returns:
            Self: The restored instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)

    def run(self) -> Self:
        """Replay the full pipeline from a loaded configuration.

        Calls each pipeline stage in order (``calculate`` →
        ``build_label`` → ``extract_features`` → ``train`` →
        ``forecast``) using the parameters stored in the loaded config.
        Stages whose config section is absent are skipped.

        This method is only meaningful when the instance was created via
        ``from_config()``.

        Returns:
            Self: This instance after all stages have been executed.

        Raises:
            RuntimeError: If the instance was not created via ``from_config()``.
        """
        config = self._loaded_config
        if config is None:
            raise RuntimeError(
                "run() is only available when the instance was created via "
                "ForecastModel.from_config(path)."
            )

        if config.calculate is not None:
            self.calculate(**config.calculate.to_dict())
        if config.build_label is not None:
            self.build_label(**config.build_label.to_dict())
        if config.extract_features is not None:
            self.extract_features(**config.extract_features.to_dict())
        if config.train is not None:
            self.train(**config.train.to_dict())
        if config.forecast is not None:
            self.forecast(**config.forecast.to_dict())

        return self

    def generate_report(
        self,
        sections: list[str] | None = None,
        output_dir: str | None = None,
        fmt: str = "html",
    ) -> Self:
        """Generate an interactive HTML (or PDF) report for the current pipeline state.

        Reads artifacts directly from this ``ForecastModel`` instance and builds a
        :class:`~eruption_forecast.report.PipelineReport` containing whichever
        sections have data available. The report is saved to disk and this method
        returns ``self`` for method chaining.

        Args:
            sections (list[str] | None): Section names to include. If None, all
                available sections are included. Valid values: ``"tremor"``,
                ``"label"``, ``"features"``, ``"training"``, ``"prediction"``.
                Defaults to None.
            output_dir (str | None): Directory for the saved report files.
                Defaults to ``{station_dir}/reports/``.
            fmt (str): Output format — ``"html"`` or ``"pdf"`` (PDF requires
                ``weasyprint``). Defaults to ``"html"``.

        Returns:
            Self: This ``ForecastModel`` instance, for method chaining.

        Examples:
            >>> fm.calculate(...).build_label(...).train(...).generate_report()
            >>> fm.generate_report(sections=["tremor", "label"], output_dir="reports/")
        """
        pipeline = PipelineReport.from_forecast_model(
            self, sections=sections, output_dir=output_dir
        )

        if fmt == "html":
            path = pipeline.save("pipeline_report.html")
        elif fmt == "pdf":
            html_path = pipeline.save("pipeline_report.html")
            path = pipeline.to_pdf(html_path.replace(".html", ".pdf"))
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Choose 'html' or 'pdf'.")

        logger.info(f"Report saved to {path}")
        return self
