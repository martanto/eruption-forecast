Label-building domain. Converts a date range + known eruption dates into a binary `is_erupted` label per sliding window, ready to join with the feature matrix.

Files:
- label_builder.py — `LabelBuilder`: global sliding-window labelling. A window is positive iff its end lies within `[eruption_date - day_to_forecast, eruption_date]`.
- dynamic_label_builder.py — `DynamicLabelBuilder` (extends `LabelBuilder`): three-phase per-eruption builder (zero frames → concat+dedupe → mark positives per eruption) — preferred when eruption windows would overlap.
- label_data.py — `LabelData`: thin CSV wrapper that parses window-size / step / `day_to_forecast` back out of the canonical filename.
- label_plots.py — `plot_label_distribution` rendering helper.
- constants.py — shared validation constants (`VALID_WINDOW_STEP_UNITS`, `MIN_DATE_RANGE_DAYS`).
