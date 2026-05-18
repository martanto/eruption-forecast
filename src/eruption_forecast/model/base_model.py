import os
import multiprocessing
from abc import ABC, abstractmethod
from typing import Self, Literal
from datetime import datetime
from functools import cached_property

import pandas as pd

from eruption_forecast import (
    TremorData,
    LabelBuilder,
    FeaturesBuilder,
    DynamicLabelBuilder,
    TremorMatrixBuilder,
)
from eruption_forecast.utils import validate_date_ranges
from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import resolve_output_dir
from eruption_forecast.utils.date_utils import sort_dates, to_datetime


class BaseModel(ABC):
    """Abstract base class for eruption forecast model stages.

    Attributes:
        start_date (datetime): Inclusive start of the modelling period.
        end_date (datetime): Inclusive end of the modelling period.
        window_size (int): Sliding window size in days.
        eruption_dates (list[datetime] | None): Sorted eruption datetimes, or
            ``None`` when no eruption dates are provided.
        output_dir (str): Resolved path to the output directory.
        root_dir (str | None): Optional project root used to resolve
            ``output_dir``.
        n_jobs (int): Number of parallel workers.
        verbose (bool): Whether to emit verbose log messages.
        start_date_str (str): ``start_date`` formatted as ``YYYY-MM-DD``.
        end_date_str (str): ``end_date`` formatted as ``YYYY-MM-DD``.
        n_days (int): Total number of days in the modelling period.
        total_cpu (int): Number of logical CPUs on the current machine.
    """

    def __init__(
        self,
        tremor_data: str | pd.DataFrame,
        start_date: str | datetime,
        end_date: str | datetime,
        window_size: int = 2,
        eruption_dates: list[str] | None = None,
        overwrite: bool = False,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """Initialize the base model.

        Args:
            tremor_data (str | DataFrame): Path to a tremor CSV file or a
                pre-loaded DataFrame.
            start_date (str | datetime): Inclusive start of the modelling
                period.
            end_date (str | datetime): Inclusive end of the modelling period.
            window_size (int, optional): Sliding window size in days. Defaults
                to 2.
            eruption_dates (list[str] | None, optional): Known eruption dates
                in ``YYYY-MM-DD`` format. Defaults to ``None``.
            overwrite (bool, optional): Overwrite existing output files when
                ``True``. Defaults to ``False``.
            output_dir (str | None, optional): Output directory path. Resolved
                relative to ``root_dir`` when omitted. Defaults to ``None``.
            root_dir (str | None, optional): Project root directory used to
                resolve ``output_dir``. Defaults to ``None``.
            n_jobs (int, optional): Number of parallel workers. Capped at
                ``cpu_count - 2`` when too high. Defaults to ``1``.
            verbose (bool, optional): Emit verbose log messages when ``True``.
                Defaults to ``False``.

        Example:
            >>> model = ConcreteModel(
            ...     tremor_data="tremor.csv",
            ...     start_date="2025-01-01",
            ...     end_date="2025-01-31",
            ... )
        """

        # Set properties
        output_dir = resolve_output_dir(
            output_dir,
            root_dir,
            os.path.join("output"),
        )

        # Set default properties
        self._tremor_data = tremor_data
        self.start_date = to_datetime(start_date)
        self.end_date = to_datetime(end_date)
        self.window_size = window_size
        self.eruption_dates = sort_dates(eruption_dates) if eruption_dates else None
        self.overwrite = overwrite
        self.output_dir = output_dir
        self.root_dir = root_dir
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Set additional properties
        self.start_date_str = self.start_date.strftime("%Y-%m-%d")
        self.end_date_str = self.end_date.strftime("%Y-%m-%d")
        self.use_relevant_features: bool = True
        self.n_days = (self.end_date - self.start_date).days + 1
        self.total_cpu = multiprocessing.cpu_count()

        # WIll be set after extract_features() called
        self.features_df: pd.DataFrame = pd.DataFrame()
        self.features_csv: str | None = None

        self._validate()

    def _validate(self) -> Self:
        """Validate base model parameters and clamp ``n_jobs`` to a safe range.

        Returns:
            Self: The current instance, for method chaining.
        """
        validate_date_ranges(self.start_date, self.end_date)

        if self.n_jobs >= self.total_cpu:
            n_jobs = self.total_cpu - 2
            if self.verbose:
                logger.warning(
                    f"Value of n_jobs ({self.n_jobs}) is more than {self.total_cpu} CPU. "
                    f"Update n_jobs to {n_jobs} CPU."
                )
            self.n_jobs = n_jobs

        return self

    def _sync_dates_to_tremor(self) -> None:
        """Clamp ``start_date`` and ``end_date`` to the tremor data range.

        Adjusts ``start_date`` and ``end_date`` when they fall outside the span
        of the loaded tremor CSV. Called lazily from ``build_label()`` to avoid
        triggering a CSV read at construction time.
        """
        tremor_data: TremorData = self.tremor_data
        tremor_start_date = tremor_data.start_date
        tremor_end_date = tremor_data.end_date
        if self.start_date < tremor_start_date:
            self.start_date = tremor_start_date
            logger.info(
                f"Training start date adjusted to tremor start date: "
                f"{tremor_start_date.strftime('%Y-%m-%d')}"
            )
        if self.end_date > tremor_end_date:
            self.end_date = tremor_end_date
            logger.info(
                f"Training end date adjusted to tremor end date: "
                f"{tremor_end_date.strftime('%Y-%m-%d')}"
            )

    @cached_property
    def tremor_data(self) -> TremorData:
        """The validated TremorData instance loaded from the configured source.

        Accepts a filesystem path or a pre-loaded DataFrame. The result is
        cached after the first access so subsequent reads are free.

        Returns:
            TremorData: Validated tremor data container.

        Raises:
            TypeError: If ``_tremor_data`` is neither a ``str`` nor a
                ``DataFrame``.
        """
        if not isinstance(self._tremor_data, str | pd.DataFrame):
            raise TypeError(
                f"tremor_data should have an instance of `str` or `pd.DataFrame` "
                f"instead of {type(self._tremor_data)}"
            )

        if isinstance(self._tremor_data, str):
            return TremorData.from_csv(self._tremor_data)

        return TremorData(self._tremor_data)

    @abstractmethod
    def set_directories(self) -> tuple:
        """Build and return all directory paths needed for this model."""

    @abstractmethod
    def create_directories(self) -> None:
        """Create the output directories required by this model."""

    @abstractmethod
    def validate(self) -> Self:
        """Validate model-specific parameters."""

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable prose description of this instance."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Return a structured dictionary of all configuration and derived fields."""

    @abstractmethod
    def to_prompt(self) -> str:
        """Return a structured text block suitable for LLM and MCP prompt input.

        Builds a deterministic, template-driven bullet-list block from
        ``to_dict()`` that is stable across calls. Paths are reduced to
        filenames only to avoid leaking absolute system paths into prompts.
        For human-readable prose use ``describe()``. For raw data use
        ``to_dict()``.

        Returns:
            str: Formatted bullet-list block ready for MCP tool responses or
                LLM context.
        """

    @abstractmethod
    def build_label(
        self, window_step: int, window_step_unit: Literal["minutes", "hours"]
    ) -> Self:
        """Build the label set for this model.

        Args:
            window_step (int): Step size between consecutive windows.
            window_step_unit (Literal["minutes", "hours"]): Unit for
                ``window_step``.

        Returns:
            Self: The current instance, for method chaining.
        """

    def _build_features(
        self,
        label_df: pd.DataFrame,
        output_dir: str,
        features_dir: str,
        select_tremor_columns: list[str] | None = None,
        save_tremor_matrix_per_method: bool = False,
        save_tremor_matrix_per_id: bool = False,
        overwrite: bool = False,
        n_jobs: int | None = None,
        verbose: bool | None = None,
    ) -> FeaturesBuilder:
        """Build a tremor matrix and return a configured FeaturesBuilder.

        Constructs the windowed tremor matrix aligned to label_df, then wraps
        it in a FeaturesBuilder ready for tsfresh extraction.

        Args:
            label_df (pd.DataFrame): Label DataFrame used to align tremor
                windows.
            output_dir (str): Base output directory for tremor matrix files.
            features_dir (str): Output directory for extracted features.
            select_tremor_columns (list[str] | None, optional): Tremor column
                names to include. All columns are used when ``None``. Defaults
                to ``None``.
            save_tremor_matrix_per_method (bool, optional): Save per-column
                tremor matrices under ``output_dir/tremor/per_method/`` when
                ``True``. Defaults to ``False``.
            save_tremor_matrix_per_id (bool, optional): Save per-ID tremor
                matrices when ``True``. Defaults to ``False``.
            overwrite (bool, optional): Overwrite existing matrix files when
                ``True``. Defaults to ``False``.
            n_jobs (int | None, optional): Number of parallel workers for
                ``FeaturesBuilder``. Falls back to ``self.n_jobs`` when
                ``None``. Defaults to ``None``.
            verbose (bool | None, optional): Emit verbose log messages. Falls
                back to ``self.verbose`` when ``None``. Defaults to ``None``.

        Returns:
            FeaturesBuilder: Configured builder ready to run feature
                extraction.
        """
        tremor_matrix_df = (
            TremorMatrixBuilder(
                tremor_df=self.tremor_data.df,
                label_df=label_df,
                output_dir=os.path.join(output_dir, "tremor"),
                window_size=self.window_size,
                overwrite=overwrite or self.overwrite,
                verbose=verbose if verbose else self.verbose,
            )
            .build(
                select_tremor_columns=select_tremor_columns,
                save_tremor_matrix_per_method=save_tremor_matrix_per_method,
                save_tremor_matrix_per_id=save_tremor_matrix_per_id,
            )
            .df
        )

        features_builder = FeaturesBuilder(
            tremor_matrix_df=tremor_matrix_df,
            label_df=None,
            output_dir=features_dir,
            overwrite=overwrite or self.overwrite,
            n_jobs=n_jobs if n_jobs is not None else self.n_jobs,
            verbose=self.verbose,
        )

        return features_builder

    @abstractmethod
    def extract_features(self) -> Self:
        """Extract features from the tremor data.

        Returns:
            Self: The current instance, for method chaining.
        """
