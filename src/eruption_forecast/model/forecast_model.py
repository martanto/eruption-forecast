import os
from typing import Self, Literal
from datetime import datetime, timedelta

import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import setup_nslc_directories
from eruption_forecast.utils.date_utils import to_datetime
from eruption_forecast.tremor.tremor_data import TremorData
from eruption_forecast.model.training_model import TrainingModel
from eruption_forecast.model.prediction_model import PredictionModel
from eruption_forecast.tremor.calculate_tremor import CalculateTremor
from eruption_forecast.model.classifier_ensemble import ClassifierEnsemble


class ForecastModel:
    """Orchestrate the complete volcanic eruption forecasting pipeline.

    Provides a high-level interface to coordinate all pipeline stages from
    raw seismic data to trained models and predictions. Uses method chaining
    for fluent API design.
    """

    def __init__(
        self,
        station: str,
        channel: str,
        network: str,
        location: str = "",
        day_to_forecast: int = 2,
        output_dir: str | None = None,
        root_dir: str | None = None,
        overwrite: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """Initialize the ForecastModel pipeline orchestrator."""
        root_dir = os.path.abspath(root_dir) if root_dir is not None else None
        nslc, output_dir, station_dir = setup_nslc_directories(
            network, station, location, channel, output_dir, root_dir
        )

        self.station = station.upper()
        self.channel = channel.upper()
        self.network = network.upper()
        self.location = location.upper()
        self.day_to_forecast = day_to_forecast
        self.output_dir = output_dir
        self.root_dir = root_dir
        self.overwrite = overwrite
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.nslc = nslc
        self.station_dir = station_dir

        # Will be set after calculate() run
        self.CalculateTremor: CalculateTremor | None = None
        self.tremor_df: pd.DataFrame = pd.DataFrame()
        self.tremor_start_date: datetime | None = None
        self.tremor_end_date: datetime | None = None

        # Will be set after train() run
        self.TrainingModel: TrainingModel | None = None
        self.classifier_ensemble: ClassifierEnsemble | None = None
        self.select_tremor_columns: list[str] | None = None
        self.save_tremor_matrix_per_method: bool = False
        self.exclude_features: list[str] | None = None

        # Will be set after predict() run
        self.PredictionModel: PredictionModel | None = None
        self.results: pd.DataFrame = pd.DataFrame()

    def calculate(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        source: Literal["sds", "fdsn"] = "sds",
        methods: str | list[str] | None = None,
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
        minimum_completion_ratio: float = 0.3,
        overwrite: bool | None = None,
        n_jobs: int | None = None,
        verbose: bool | None = None,
    ) -> Self:

        start_date = to_datetime(start_date) - timedelta(days=self.day_to_forecast)
        methods = [methods] if isinstance(methods, str) else methods
        verbose = verbose if verbose is not None else self.verbose
        overwrite = overwrite if overwrite is not None else self.overwrite

        if verbose:
            logger.info(
                f"Adjusting tremor calculation start date by reducing to "
                f"day_to_forecast={self.day_to_forecast}: {start_date}"
            )

        calculate = CalculateTremor(
            start_date=start_date,
            end_date=end_date,
            station=self.station,
            channel=self.channel,
            network=self.network,
            location=self.location,
            methods=methods,
            output_dir=self.output_dir,
            remove_outlier_method=remove_outlier_method,
            remove_tremor_anomalies=remove_tremor_anomalies,
            interpolate=interpolate,
            value_multiplier=value_multiplier,
            cleanup_daily_dir=cleanup_daily_dir,
            plot_daily=plot_daily,
            save_plot=save_plot,
            overwrite_plot=overwrite_plot,
            overwrite=overwrite or self.overwrite,
            minimum_completion_ratio=minimum_completion_ratio,
            n_jobs=n_jobs if n_jobs is not None else self.n_jobs,
            verbose=verbose,
        )

        if source.upper() == "SDS":
            if sds_dir is None:
                raise ValueError(
                    "You chose 'sds' as source, please provide 'sds_dir' parameter. "
                    "Example: calculate(source='sds', sds_dir='converted')"
                )
            calculate = calculate.from_sds(sds_dir=sds_dir).run()
        if source.upper() == "FDSN":
            calculate = calculate.from_fdsn(client_url=client_url).run()

        self.CalculateTremor = calculate

        if verbose:
            start_date_str = start_date.strftime("%Y-%m-%d")
            logger.info(f"Calculate Tremor from {start_date_str} to {end_date}")

        tremor_data = TremorData(calculate.df)
        self.tremor_df = tremor_data.df
        self.tremor_start_date = tremor_data.start_date
        self.tremor_end_date = tremor_data.end_date

        return self

    def train(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        eruption_dates: list[str],
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        label_builder: Literal["standard", "dynamic"] = "standard",
        days_before_eruption: int | None = None,
        classifiers: str | list[str] = "rf",
        cv_strategy: Literal[
            "shuffle", "stratified", "shuffle-stratified"
        ] = "shuffle-stratified",
        cv_splits: int = 5,
        number_of_features: int = 20,
        include_eruption_date: bool = True,
        select_tremor_columns: list[str] | None = None,
        save_tremor_matrix_per_method: bool = True,
        exclude_features: list[str] | None = None,
        minimum_completion: float = 1.0,
        seeds: int = 10,
        resample_method: Literal["under", "over", "auto"] | None = "auto",
        minority_threshold: float = 0.15,
        sampling_strategy: str | float = 0.75,
        plot_features: bool = True,
        output_dir: str | None = None,
        overwrite: bool | None = None,
        n_jobs: int | None = None,
        n_grids: int = 1,
        verbose: bool | None = None,
    ) -> Self:
        if self.CalculateTremor is None:
            raise ValueError("Tremor data not found. Please run calculate() first.")

        n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        verbose = verbose if verbose is not None else self.verbose
        overwrite = overwrite if overwrite is not None else self.overwrite

        training_model = (
            TrainingModel(
                tremor_data=self.tremor_df,
                start_date=start_date,
                end_date=end_date,
                classifiers=classifiers,
                eruption_dates=eruption_dates,
                window_size=self.day_to_forecast,
                cv_strategy=cv_strategy,
                cv_splits=cv_splits,
                number_of_features=number_of_features,
                output_dir=output_dir or self.station_dir,
                overwrite=overwrite,
                n_jobs=n_jobs,
                n_grids=n_grids,
                verbose=verbose,
            )
            .build_label(
                window_step=window_step,
                window_step_unit=window_step_unit,
                builder=label_builder,
                include_eruption_date=include_eruption_date,
                days_before_eruption=days_before_eruption,
                verbose=verbose,
            )
            .extract_features(
                select_tremor_columns=select_tremor_columns,
                save_tremor_matrix_per_method=save_tremor_matrix_per_method,
                exclude_features=exclude_features,
                minimum_completion=minimum_completion,
                overwrite=overwrite,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            .fit(
                seeds=seeds,
                resample_method=resample_method,
                minority_threshold=minority_threshold,
                sampling_strategy=sampling_strategy,
                plot_features=plot_features,
            )
        )

        self.TrainingModel = training_model
        self.classifier_ensemble = training_model.model
        self.select_tremor_columns = select_tremor_columns
        self.save_tremor_matrix_per_method = save_tremor_matrix_per_method
        self.exclude_features = exclude_features

        return self

    def predict(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        save_seed_result: bool = True,
        plot_threshold: float = 0.5,
        plot_title: str | None = None,
        plot_pdf: bool = True,
        output_dir: str | None = None,
        overwrite: bool | None = None,
        n_jobs: int | None = None,
        verbose: bool | None = None,
        **plot_kwargs,
    ) -> Self:
        if self.TrainingModel is None or self.classifier_ensemble is None:
            raise ValueError("Training model not found. Please run train() first.")

        n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        verbose = verbose if verbose is not None else self.verbose
        overwrite = overwrite if overwrite is not None else self.overwrite

        prediction_model = (
            PredictionModel(
                model=self.classifier_ensemble,
                tremor_data=self.tremor_df,
                start_date=start_date,
                end_date=end_date,
                window_size=self.day_to_forecast,
                output_dir=output_dir or self.station_dir,
                overwrite=overwrite,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            .build_label(
                window_step=window_step,
                window_step_unit=window_step_unit,
            )
            .extract_features(
                select_tremor_columns=self.select_tremor_columns,
                save_tremor_matrix_per_method=self.save_tremor_matrix_per_method,
                exclude_features=self.exclude_features,
                overwrite=overwrite,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        )

        self.PredictionModel = prediction_model
        self.results = prediction_model.forecast(
            save_seed_result=save_seed_result,
            plot_threshold=plot_threshold,
            plot_title=plot_title,
            plot_pdf=plot_pdf,
            **plot_kwargs,
        )

        return self
