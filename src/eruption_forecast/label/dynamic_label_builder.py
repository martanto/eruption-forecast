import os
from typing import Self, Literal
from datetime import datetime, timedelta

import pandas as pd

from eruption_forecast import LabelBuilder
from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import resolve_output_dir
from eruption_forecast.utils.date_utils import sort_dates, to_datetime


class DynamicLabelBuilder(LabelBuilder):
    """Build per-eruption label windows for volcanic eruption forecasting.

    Extends LabelBuilder to generate one label window per eruption event.
    Each window spans ``days_before_eruption`` days before the eruption date
    rather than covering a single global start/end range. Windows from all
    eruptions are concatenated into one DataFrame with unique IDs.

    Attributes:
        days_before_eruption (int): Number of days before each eruption to start
            the window.

    Examples:
        >>> builder = DynamicLabelBuilder(
        ...     days_before_eruption=7,
        ...     window_step=12,
        ...     window_step_unit="hours",
        ...     day_to_forecast=2,
        ...     eruption_dates=["2025-03-20"],
        ...     volcano_id="OJN",
        ... ).build()
        >>> print(len(builder.df_eruption))
    """

    def __init__(
        self,
        days_before_eruption: int,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        day_to_forecast: int,
        eruption_dates: list[str],
        volcano_id: str,
        output_dir: str | None = None,
        root_dir: str | None = None,
        prefix_filename: str | None = None,
        overwrite: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize DynamicLabelBuilder.

        Computes the overall date range from eruption dates and
        ``days_before_eruption``, then delegates to ``LabelBuilder.__init__``.

        Args:
            days_before_eruption (int): Days before each eruption to start its window.
            window_step (int): Step size between consecutive windows.
            window_step_unit (Literal["minutes", "hours"]): Unit for window_step.
            day_to_forecast (int): Days before eruption to start positive labeling.
            eruption_dates (list[str]): Eruption dates in YYYY-MM-DD format.
            volcano_id (str): Volcano identifier used in filenames.
            output_dir (str | None, optional): Output directory path. Defaults to None.
            root_dir (str | None, optional): Anchor directory for resolving
                relative output_dir paths. Defaults to None.
            prefix_filename (str | None, optional): Custom prefix for the output
                filename. Defaults to ``"label"``.
            overwrite (bool, optional): Overwrite existing output files. Defaults to False.
            verbose (bool, optional): Enable informational logging. Defaults to False.
            debug (bool, optional): Enable debug-level logging. Defaults to False.

        Raises:
            ValueError: If any LabelBuilder validation fails (date range too short,
                invalid window_step_unit, etc.).
        """
        eruption_dates = sort_dates(eruption_dates)
        prefix_filename = prefix_filename or "label"

        # Compute overall data range from eruption dates + look-back window.
        overall_start: datetime = to_datetime(eruption_dates[0]) - timedelta(
            days=days_before_eruption
        )
        overall_end: datetime = to_datetime(eruption_dates[-1])

        output_dir = resolve_output_dir(output_dir, root_dir, "output")
        label_dir = os.path.join(output_dir, "labels")

        overall_start_str = overall_start.strftime("%Y-%m-%d")
        overall_end_str = overall_end.strftime("%Y-%m-%d")

        # Call super first so all base attributes are properly initialised.
        super().__init__(
            overall_start,
            overall_end,
            window_step,
            window_step_unit,
            day_to_forecast,
            eruption_dates,
            volcano_id,
            output_dir,
            root_dir,
            verbose,
            debug,
        )

        # Set DynamicLabelBuilder-specific attributes after super.
        self.days_before_eruption: int = days_before_eruption

        # Override filename/csv with custom prefix if provided.
        self.filename = (
            f"{prefix_filename}_{overall_start_str}_{overall_end_str}"
            f"_step-{window_step}-{window_step_unit}"
            f"_dtf-{day_to_forecast}.csv"
        )
        self.csv = os.path.join(label_dir, self.filename)
        self.overwrite = overwrite

    @property
    def _eruption_dates(self) -> list[datetime]:
        """Convert eruption_dates strings to datetime objects.

        Returns:
            list[datetime]: Sorted list of eruption dates as datetime objects.
        """
        eruption_dates: list[datetime] = []
        for eruption_date in self.eruption_dates:
            eruption_dates.append(to_datetime(eruption_date))

        return eruption_dates

    @property
    def _label_dates(self) -> list[tuple[datetime, datetime]]:
        """Compute per-eruption (start, end) date pairs.

        Each tuple spans from ``eruption_date - days_before_eruption`` to
        ``eruption_date``.

        Returns:
            list[tuple[datetime, datetime]]: List of (start_date, end_date) tuples,
                one per eruption.
        """
        label_dates: list[tuple[datetime, datetime]] = []
        for eruption_date in self._eruption_dates:
            start_date = eruption_date - timedelta(days=self.days_before_eruption)
            end_date = eruption_date
            label_dates.append((start_date, end_date))

        return label_dates

    @logger.catch
    def build(self) -> Self:
        """Build per-eruption label windows and concatenate into one DataFrame.

        For each eruption, creates a windowed DataFrame spanning
        ``days_before_eruption`` days before the eruption and marks windows
        within the ``day_to_forecast`` lead time as positive (is_erupted=1).
        All per-eruption DataFrames are concatenated with globally unique IDs.

        If the label CSV already exists on disk it is loaded instead of
        recomputed.

        Returns:
            Self: Instance with populated ``df``, ``df_eruption``, and
                ``df_eruptions`` properties.

        Raises:
            ValueError: If the built DataFrame contains no erupted windows.

        Examples:
            >>> builder = DynamicLabelBuilder(
            ...     days_before_eruption=7,
            ...     window_step=12,
            ...     window_step_unit="hours",
            ...     day_to_forecast=2,
            ...     eruption_dates=["2025-03-20"],
            ...     volcano_id="OJN",
            ... ).build()
            >>> print(len(builder.df_eruption) > 0)
            True
        """
        if self.verbose:
            logger.info("Building using dynamic labels")

        file_exists = os.path.isfile(self.csv)

        if file_exists and not self.overwrite:
            if self.verbose:
                logger.info(f"Loading existing labels from {self.csv}")
            _df = self.from_csv(self.csv)
            self.df = _df
            df_eruption = _df[_df["is_erupted"] > 0]
            self.df_eruption = df_eruption
            return self

        dfs: list[pd.DataFrame] = []
        df_eruptions_dict: dict[str, pd.DataFrame] = {}

        for index, label_date in enumerate(self._label_dates):
            start_date, end_eruption = label_date

            # Initialise per-eruption window DataFrame.
            # Initiate ``is_erupted`` column value to 0 (no eruption).
            df = self.initiate_label(start_date=start_date, end_date=end_eruption)

            start_eruption = end_eruption - timedelta(days=self.day_to_forecast)
            start_eruption = start_eruption.replace(hour=0, minute=0, second=0)
            end_eruption_ts = end_eruption.replace(hour=23, minute=59, second=59)

            logger.info(
                f"Start label: {start_date}. End label: {end_eruption_ts}. "
                f"Total labels: {len(df)}"
            )

            df.loc[start_eruption:end_eruption_ts, "is_erupted"] = 1

            length_df = len(df)
            df["id"] = range(index * length_df, (index + 1) * length_df)
            df = df[["id", "is_erupted"]].astype({"id": int, "is_erupted": int})

            eruption_key = end_eruption.strftime("%Y-%m-%d")
            df_eruptions_dict[eruption_key] = df.loc[start_eruption:end_eruption_ts]

            dfs.append(df)

        _df = pd.concat(dfs)

        # Set the setter once after loop to avoid redundant validation.
        self.df_eruptions = df_eruptions_dict

        self.df = _df

        df_eruption = _df[_df["is_erupted"] > 0]
        self.df_eruption = df_eruption

        if self.verbose:
            erupted_count = len(df_eruption)
            total_count = len(_df)
            logger.info(
                f"Label building complete: {erupted_count} erupted windows out of "
                f"{total_count} total ({erupted_count / total_count * 100:.2f}%)"
            )

        self.save()

        return self
