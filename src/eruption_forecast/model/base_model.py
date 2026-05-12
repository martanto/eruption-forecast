import os
import multiprocessing
from abc import ABC, abstractmethod
from typing import Self, Literal
from datetime import datetime

from eruption_forecast.utils import validate_date_ranges
from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import resolve_output_dir
from eruption_forecast.utils.date_utils import sort_dates, to_datetime


class BaseModel(ABC):
    def __init__(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        window_size: int = 2,
        eruption_dates: list[str] | None = None,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """Initialize the base model class."""

        # Set properties
        output_dir = resolve_output_dir(
            output_dir,
            root_dir,
            os.path.join("output"),
        )

        # Set default properties
        self.start_date = to_datetime(start_date)
        self.end_date = to_datetime(end_date)
        self.window_size = window_size
        self.eruption_dates = sort_dates(eruption_dates) if eruption_dates else None
        self.output_dir = output_dir
        self.root_dir = root_dir
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Set additional properties
        self.start_date_str = self.start_date.strftime("%Y-%m-%d")
        self.end_date_str = self.end_date.strftime("%Y-%m-%d")
        self.n_days = (self.end_date - self.start_date).days + 1
        self.total_cpu = multiprocessing.cpu_count()

        self._validate()

    def _validate(self) -> Self:
        """Validate the base model parameters."""
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

    @abstractmethod
    def create_directories(self, **kwargs) -> None:
        """Create the directories needed for the model."""

    @abstractmethod
    def validate(self) -> Self:
        """Validate the model parameters."""

    @abstractmethod
    def describe(self) -> str:
        """Return a natural, human-readable prose description of this class."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Return a structured dictionary of all configuration and derived fields."""

    @abstractmethod
    def to_prompt(self) -> str:
        """Return a structured labeled text block for LLM and MCP prompt input.

        Builds a deterministic, template-driven bullet-list block from ``to_dict()``
        that is stable across calls and suitable for direct inclusion in MCP tool
        responses or LLM context. Paths are reduced to filenames only to avoid
        leaking absolute system paths into prompts.

        For a human-readable prose description use ``describe()``. For raw data
        use ``to_dict()``.
        """

    @abstractmethod
    def build_label(
        self, window_step: int, window_step_unit: Literal["minutes", "hours"]
    ) -> Self:
        """Build a label for this model.

        Args:
            window_step (int): Step size between consecutive windows.
            window_step_unit (Literal["minutes", "hours"]): Step size between consecutive windows.

        Returns:
            Self: Model class
        """

    @abstractmethod
    def extract_features(self) -> Self:
        """Extract features from tremor."""
