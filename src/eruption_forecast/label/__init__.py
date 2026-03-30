"""Label builder.

Generates binary classification labels by constructing sliding time
windows over a date range and marking each window as "erupted" (1) or "not erupted"
(0) based on known eruption dates and a configurable lead-time parameter
(``day_to_forecast``). Labels are saved as standardised CSV files whose filenames
encode all window configuration parameters for reproducibility.

Key classes:
    - ``LabelBuilder``: Builds a single label set over a global ``start_date`` to
      ``end_date`` range using fixed ``window_size`` and ``window_step`` parameters.
    - ``DynamicLabelBuilder``: Subclass of ``LabelBuilder`` that generates one
      per-eruption window spanning ``days_before_eruption`` days before each eruption,
      then concatenates all windows into one DataFrame with unique IDs.
    - ``LabelData``: Loads an existing label CSV and parses all window-configuration
      parameters from the standardised filename (dates, step, unit, day_to_forecast).

Label filename format::

    label_{start_date}_{end_date}_ws-{window_size}_step-{window_step}-{unit}_dtf-{day_to_forecast}.csv
"""
