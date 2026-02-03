# Standard library imports
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
        # Set DEFAULT parameter
        start_date = to_datetime(start_date).replace(hour=0, minute=0, second=0)
        end_date = to_datetime(end_date).replace(hour=23, minute=59, second=59)
        start_date_str: str = start_date.strftime("%Y-%m-%d")
        end_date_str: str = end_date.strftime("%Y-%m-%d")
        output_dir = output_dir or os.path.join(os.getcwd(), "output")
        network = network or "VG"
        location = location or "00"
        nslc = f"{network}.{station}.{location}.{channel}"
        output_dir = output_dir or os.path.join(os.getcwd(), "output")
        station_dir = os.path.join(output_dir, nslc)
        training_dir = os.path.join(station_dir, "training")
        features_dir = os.path.join(station_dir, "features")

        # Set DEFAULT properties
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

        # Set ADDITIONAL properties
        self.start_date_minus_window_size = start_date - timedelta(days=window_size)
        self.start_date_str = start_date_str
        self.end_date_str = end_date_str
        self.nslc = nslc
        self.station_dir = station_dir
        self.training_dir = training_dir
        self.features_dir = features_dir
        self.kwargs = {
            "station": station,
            "channel": channel,
            "network": network,
            "location": location,
            "start_date": start_date,
            "end_date": end_date,
            "output_dir": output_dir,
            "overwrite": overwrite,
            "n_jobs": n_jobs,
        }

        # Default feature calculator (fc) parameters
        self.default_fc_parameters = ComprehensiveFCParameters()
        self.excludes_features: set[str] = {
            "agg_linear_trend",
            "linear_trend_timewise",
            "length",
            "has_duplicate_max",
            "has_duplicate_min",
            "has_duplicate",
        }

        # Will be set after calculate() method called
        self.CalculateTremor: Optional[CalculateTremor] = None
        self.TremorData: Optional[TremorData] = None
        self.tremor_data: pd.DataFrame = pd.DataFrame()
        self.tremor_csv: Optional[str] = None

        # WIll be set after train() method called
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
        self.extracted_features_csv: str = None
        self.extracted_relevant_csv: str = None

        # Will be set after predict() called
        self.prediction_features_csvs = set()

        # Base filename without extension
        self.basename: Optional[str] = None

        # Validate

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
        """Calculate Tremor Data

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
        verbose = verbose or self.verbose
        debug = debug or self.debug

        # Update "start_date" key arguments
        kwargs = self.kwargs
        kwargs["start_date"] = self.start_date_minus_window_size

        calculate = CalculateTremor(
            methods=methods,
            filename_prefix=filename_prefix,
            remove_outlier_method=remove_outlier_method,
            interpolate=interpolate,
            value_multiplier=value_multiplier,
            cleanup_tmp_dir=cleanup_tmp_dir,
            plot_tmp=plot_tmp,
            save_plot=save_plot,
            overwrite_plot=overwrite_plot,
            **kwargs,
        )

        if verbose:
            calculate.verbose = True

        if debug:
            calculate.debug = True

        self.CalculateTremor = calculate

        if source == "sds":
            assert sds_dir is not None, ValueError(
                f"You chose 'sds' as source, please provide 'sds_dir' parameter. "
                f"Example: calculate(source='sds', sds_dir='converted')"
            )
            assert os.path.isdir(sds_dir), ValueError(f"SDS dir {sds_dir} not exists.")

            calculate = calculate.from_sds(sds_dir=sds_dir).run()

        # TODO: Get data from FDSN services
        if source == "fdsn":
            # calculate = calculate.from_fdsn(client_url=client_url).run()
            logger.error(f"FDSN source is not yet supported. Client url: {client_url}")
            raise

        tremor_data = TremorData(calculate.df)
        df_tremor = tremor_data.df
        self.TremorData = tremor_data
        self.TremorData.csv = calculate.csv
        self.tremor_csv = calculate.csv

        # Update self.start_date and self.end_date based on calculated tremor data
        if self.start_date_minus_window_size < tremor_data.start_date:
            self.start_date = tremor_data.start_date
            self.start_date_str = tremor_data.start_date_str
            if self.verbose:
                logger.info(
                    f"start_date parameter: {self.start_date_minus_window_size} updated to "
                    f"tremor start date: {tremor_data.start_date}"
                )

        if self.end_date > tremor_data.end_date:
            self.end_date = tremor_data.end_date
            self.end_date_str = tremor_data.end_date_str
            if self.verbose:
                logger.info(
                    f"end_date parameter: {self.end_date} updated to "
                    f"tremor end date: {tremor_data.end_date}"
                )

        # Update tremor data based on update self.start_date or self.end_date
        self.tremor_data = df_tremor.loc[self.start_date : self.end_date]

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
        """Extract features from tremor data.

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
        features_data = self.features_data
        if tremor_columns is not None:
            validate_columns(self.features_data, tremor_columns)
            features_data = features_data[["id", "datetime", *tremor_columns]]

        overwrite = overwrite or self.overwrite
        label_data = self.label_data
        prefix_filename = (
            "extracted_relevant" if use_relevant_features else "extracted_features"
        )

        if use_relevant_features and self.verbose:
            logger.info(f"Extracting features using relevant features")

        # Get label data as a target
        y = label_data["is_erupted"]
        y.index = label_data["id"]

        # Extract features per method
        extract_features_dir = os.path.join(self.features_dir, "extract_features")
        os.makedirs(extract_features_dir, exist_ok=True)

        # Exclude features from calculation
        default_fc_parameters = self.default_fc_parameters

        if exclude_features is not None:
            if isinstance(exclude_features, list):
                default_fc_parameters = self.drop_features(exclude_features)
            if isinstance(exclude_features, bool):
                if not exclude_features:
                    self.excludes_features = {}

        extract_params = {
            "column_id": "id",
            "column_sort": "datetime",
            "n_jobs": n_jobs or self.n_jobs,
            "default_fc_parameters": default_fc_parameters,
        }

        traning_features_csvs = set()
        for column in features_data.columns.tolist():
            if column in ["id", "datetime"]:
                continue

            extracted_csv = os.path.join(
                extract_features_dir, f"{prefix_filename}_{column}.csv"
            )

            # Skip if extracted features already exists
            if not overwrite and os.path.isfile(extracted_csv):
                if self.verbose:
                    logger.info(
                        f"Extracted features for {column} saved in: {extracted_csv} "
                    )
                traning_features_csvs.add(extracted_csv)
                continue

            df = features_data[["id", "datetime", column]]

            if self.verbose:
                logger.info(f"Extracting {column} features. ")

            if use_relevant_features:
                extracted_features = extract_relevant_features(df, y, **extract_params)
            else:
                extracted_features = tsfresh_extract_features(
                    df,
                    impute_function=impute,
                    **extract_params,
                )

            extracted_features.index.name = "id"
            extracted_features.to_csv(extracted_csv, index=True)

            # Add extracted csvs to list
            traning_features_csvs.add(extracted_csv)

            logger.info(f"Extracted features for {column} saved in: {extracted_csv} ")

        if use_relevant_features:
            self.relevant_features_csvs.update(traning_features_csvs)
        else:
            self.extract_features_csvs.update(traning_features_csvs)

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
        """Build label.

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
        tremor_data = self.tremor_data
        train_start_date: Union[str, datetime] = start_date or self.start_date
        train_end_date: Union[str, datetime] = end_date or self.end_date

        # Validating
        validate_date_ranges(train_start_date, train_end_date)

        assert isinstance(tremor_data, pd.DataFrame) and (
            len(tremor_data) > 0
        ), ValueError(
            f"Tremor data not found/loaded. "
            f"Please run calculate() or load_tremor_data() method first."
        )
        if tremor_columns:
            validate_columns(tremor_data, tremor_columns)

        verbose = verbose or self.verbose
        debug = debug or self.debug

        output_dir = output_dir or self.station_dir

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

        df_label = label_builder.df

        # Build output directory
        basename = os.path.basename(label_builder.csv).split(".csv")[0]

        # Load tremor and select specific columns
        df_tremor = tremor_data.copy()
        if tremor_columns is not None:
            df_tremor = df_tremor[tremor_columns]
        df_tremor.sort_index(ascending=True, inplace=True)

        # Ensuring label data is within tremor data range
        label_start_date_obj: pd.Timestamp = df_label.index[0]
        label_end_date_obj: pd.Timestamp = df_label.index[-1]
        tremor_start_date_obj: pd.Timestamp = df_tremor.index[0]
        tremor_end_date_obj: pd.Timestamp = df_tremor.index[-1]

        assert tremor_start_date_obj <= label_start_date_obj, ValueError(
            f"Training start date ({tremor_start_date_obj}) should be after/equal "
            f"to tremor start date ({label_start_date_obj}). "
            f"Change your training start date after/equal {tremor_start_date_obj}."
        )
        assert tremor_end_date_obj >= label_end_date_obj, ValueError(
            f"Training end date ({tremor_end_date_obj}) should be before/equal "
            f"to tremor end date ({label_start_date_obj}). "
            f"Change your training end date before/equal {tremor_end_date_obj}."
        )

        # Set properties
        self.LabelBuilder = label_builder
        self.label_csv = label_builder.csv
        self.basename = basename

        # Omitting label data (df) based on window step
        df_label = df_label.loc[self.start_date :]

        if df_label.empty:
            raise ValueError(f"Label from start date {self.start_date} is empty.")

        self.label_data = df_label

        # Get target class numbers. Check if the data is balanced or not
        self.total_eruption_class = len(label_builder.df_eruption)
        self.total_non_eruption_class = (
            len(label_builder.df) - self.total_eruption_class
        )
        class_ratio: float = self.total_eruption_class / self.total_non_eruption_class

        if verbose:
            logger.info(
                f"Total number of eruptions: {self.total_eruption_class}. "
                f"Total number of non-eruptions: {self.total_non_eruption_class}. "
                f"Class ratio (eruption againts non eruptions): {class_ratio}"
            )

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
