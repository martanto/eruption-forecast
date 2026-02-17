"""Label building module for volcanic eruption forecasting.

This module provides tools for generating binary labels for supervised learning
by creating time windows from seismic data and marking them as erupted (1) or
not erupted (0) based on known eruption dates.

Classes:
    LabelBuilder: Creates labeled datasets with sliding time windows
    LabelData: Loads and parses pre-built label CSV files

Constants:
    See constants.py for label filename format prefixes, validation thresholds,
    and default parameter values.

Examples:
    >>> from eruption_forecast.label import LabelBuilder
    >>> builder = LabelBuilder(
    ...     start_date="2020-01-01",
    ...     end_date="2020-12-31",
    ...     window_step=12,
    ...     window_step_unit="hours",
    ...     day_to_forecast=2,
    ...     eruption_dates=["2020-06-15"],
    ...     volcano_id="VOLCANO_001"
    ... )
    >>> builder.build().save()
"""
