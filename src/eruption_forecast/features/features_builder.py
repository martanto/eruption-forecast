# Standard library imports
import os
from datetime import timedelta
from typing import Optional, Self

# Third party imports
import pandas as pd

# Project imports
from eruption_forecast.logger import logger


class FeaturesBuilder:
    """Features builder class

    Args:
        df_tremor (str): Tremor dataframe.
        df_label (str): Label dataframe.
        output_dir (str): Output directory path.
        window_size (int): Window size.
        tremor_columns (list): Select tremor columns.
        verbose (bool, optional): Verbose mode. Defaults to False.
        debug (bool, optional): Debug mode. Defaults to False.
    """

    def __init__(
        self,
        df_tremor: pd.DataFrame,
        df_label: pd.DataFrame,
        output_dir: str,
        window_size: int,
        tremor_columns: Optional[list[str]] = None,
        overwrite: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ):
        # Set DEFAULT parameter
        assert isinstance(df_tremor.index, pd.DatetimeIndex), ValueError(
            f"df_tremor.index is not a DatetimeIndex"
        )
        assert isinstance(df_label.index, pd.DatetimeIndex), ValueError(
            f"df_label.index is not a DatetimeIndex"
        )
        features_tmp_dir = os.path.join(output_dir, "tmp")
        df_tremor = df_tremor.sort_index()
        df_label.sort_index(inplace=True)

        # Set DEFAULT properties
        self.df_tremor: pd.DataFrame = df_tremor
        self.df_label: pd.DataFrame = df_label
        self.output_dir = output_dir
        self.window_size = window_size
        self.tremor_columns = tremor_columns
        self.overwrite = overwrite
        self.verbose = verbose
        self.debug = debug

        # Set ADDITIONAL properties
        self.features_tmp_dir = features_tmp_dir
        self.start_date = df_label.index[0] - timedelta(days=self.window_size)
        self.end_date = df_tremor.index[-1]
        self.features_matrix: pd.DataFrame = pd.DataFrame()
        self.unique_ids: list[int] = []
        self.csv: Optional[str] = None

        # Validate
        self.validate()

        # Verbose and debugging
        if self.debug:
            logger.info("⚠️ Debug mode is ON")

        if self.verbose:
            logger.info(f"Test")

    def validate(self):
        """Validate parameters"""
        label_columns = self.df_label.columns.tolist()
        tremor_columns = self.df_tremor.columns.tolist()

        assert "id" in label_columns, ValueError(
            f"Column 'id' not found, in Label dataframe"
        )
        assert "is_erupted" in label_columns, ValueError(
            f"Column 'is_erupted' not found, in Label dataframe"
        )

        if self.tremor_columns is not None:
            for column in self.tremor_columns:
                assert column in tremor_columns, ValueError(
                    f"Column '{column}' not found in Tremor dataframe. "
                    f"Columns available: {tremor_columns}"
                )
            self.df_tremor = self.df_tremor[self.tremor_columns]

        # Ensuring label date within range of tremor date
        tremor_start_date = self.df_tremor.index[0]
        tremor_end_date = self.df_tremor.index[-1]

        if self.start_date < tremor_start_date:
            self.start_date = tremor_start_date
            if self.verbose:
                logger.info(f"start_date updated to: {self.start_date}")

        if self.end_date > tremor_end_date:
            self.end_date = tremor_end_date
            if self.verbose:
                logger.info(f"end_date updated to: {self.end_date}")

        os.makedirs(self.output_dir, exist_ok=True)

    def save_features_per_method(self, features_matrix: pd.DataFrame) -> Self:
        feature_method_dir = os.path.join(self.output_dir, "method")
        os.makedirs(feature_method_dir, exist_ok=True)

        for column in features_matrix.columns.tolist():
            if column in ["id", "datetime"]:
                continue

            feature_matrix_filename = f"features_{column}.csv"
            feature_matrix_filepath = os.path.join(
                feature_method_dir, feature_matrix_filename
            )

            # Skip if exists
            if not self.overwrite and (os.path.isfile(feature_matrix_filepath)):
                continue

            df_feature_matrix = features_matrix[["id", "datetime", column]]
            df_feature_matrix.to_csv(feature_matrix_filepath, index=True)

            if self.verbose:
                logger.info(f"Features {column} is saved to: {feature_matrix_filepath}")

        return self

    def build(
        self,
        save_tmp_feature: bool = False,
        save_per_method: bool = True,
        filename: Optional[str] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        df_label = self.df_label.loc[self.start_date : self.end_date]
        df_tremor = self.df_tremor
        window_size = self.window_size
        verbose = verbose or self.verbose
        filename = filename or "features.csv"

        columns = df_tremor.columns.tolist()
        feature_csv = os.path.join(self.output_dir, filename)

        # Skip if exists
        if not self.overwrite and (os.path.isfile(feature_csv)):
            self.features_matrix = pd.read_csv(feature_csv)
            self.unique_ids = self.features_matrix["id"].unique()
            return self.features_matrix

        if save_tmp_feature:
            os.makedirs(self.features_tmp_dir, exist_ok=True)

        if verbose:
            logger.info(f"Group features per ID label")

        features: list[pd.DataFrame] = []

        # TODO: Check sampling period consistency
        tremor_sampling_period = (df_tremor.index[1] - df_tremor.index[0]).seconds

        for datetime_index, column_id, column_eruption in df_label.itertuples():
            start_datetime = datetime_index - timedelta(days=window_size)
            end_datetime = datetime_index - timedelta(milliseconds=1)
            total_window = int(window_size * 24 * 60 * 60 / tremor_sampling_period)

            df_tremor_sliced = df_tremor.loc[start_datetime:end_datetime]

            if len(df_tremor_sliced) == total_window:
                df_tremor_sliced = df_tremor_sliced.sort_index(ascending=True)
                df_tremor_sliced.reset_index(inplace=True)
                df_tremor_sliced["id"] = column_id
                df_tremor_sliced = df_tremor_sliced[["id", "datetime", *columns]]

                features.append(df_tremor_sliced)

                if save_tmp_feature:
                    start_date_str = start_datetime.strftime("%Y-%m-%d--%H-%M-%S")
                    end_date_str = end_datetime.strftime("%Y-%m-%d_%H--%H-%M-%S")
                    feature_tmp_filename = f"{column_id:05}_{start_date_str}_{end_date_str}_eruption-{column_eruption}.csv"
                    feature_tmp_filepath = os.path.join(
                        self.features_tmp_dir, feature_tmp_filename
                    )
                    df_tremor_sliced.to_csv(feature_tmp_filepath, index=False)
                    df_label["feature_csv"] = feature_tmp_filename

        # Concat features matrix
        features_matrix = pd.concat(features, ignore_index=False)

        if features_matrix.empty:
            logger.error(f"Features matrix is empty")
            raise Exception(f"Features matrix is empty")

        # Save all features and labels for training
        features_matrix.to_csv(feature_csv, index=False)

        # Save features per method/columns
        if save_per_method:
            self.save_features_per_method(features_matrix)

        self.features_matrix = features_matrix
        self.csv = feature_csv
        self.unique_ids = features_matrix["id"].unique()

        logger.info(f"Features matrix saved to: {feature_csv}")

        return features_matrix
