"""Text formatting utilities.

This module provides utilities for converting class names to slugified format
for use in filenames and directory names.
"""

import re

from eruption_forecast.config.constants import DEFAULT_FREQUENCY_BANDS


_PARAM_VALUE_RE = re.compile(r'_("[^"]*"|-?\d+(?:\.\d+)?|True|False)$')


_DEFAULT_FREQ_BANDS: dict[str, tuple[float, float]] = {
    f"f{i}": edges for i, edges in enumerate(DEFAULT_FREQUENCY_BANDS)
}


def _fmt_hz(value: float) -> str:
    """Render a Hz edge as a compact decimal string (``2`` not ``2.0``)."""
    return f"{value:g}"


def _humanize_column(
    column: str,
    freq_bands: dict[str, tuple[float, float]],
) -> str:
    """Render a tremor column name as a plain-English phrase.

    Recognises three shapes:

    - ``rsam_f<N>`` → ``"RSAM {low}–{high} Hz"`` using the edges of band
      ``f<N>`` from ``freq_bands``.
    - ``dsar_f<A>-f<B>`` → ``"DSAR ratio {low_A}/{low_B} Hz"`` using the
      lower edges of bands ``f<A>`` and ``f<B>``.
    - ``entropy`` → ``"Shannon entropy"``.

    Unknown shapes — or shapes that reference a band token missing from
    ``freq_bands`` — fall back to ``column.replace("_", " ")`` so the
    output stays readable.
    """
    if column == "entropy":
        return "Shannon entropy"

    prefix, sep, band_spec = column.partition("_")
    if not sep:
        return column.replace("_", " ")

    if prefix == "rsam" and band_spec in freq_bands:
        low, high = freq_bands[band_spec]
        return f"RSAM {_fmt_hz(low)}–{_fmt_hz(high)} Hz"

    if prefix == "dsar" and "-" in band_spec:
        left, right = band_spec.split("-", 1)
        if left in freq_bands and right in freq_bands:
            low_left = freq_bands[left][0]
            low_right = freq_bands[right][0]
            return f"DSAR ratio {_fmt_hz(low_left)}/{_fmt_hz(low_right)} Hz"

    return column.replace("_", " ")


_CALCULATOR_HUMANIZATIONS: dict[str, str] = {
    "abs_energy": "absolute energy",
    "absolute_maximum": "absolute maximum",
    "absolute_sum_of_changes": "absolute sum of changes",
    "agg_autocorrelation": "aggregated autocorrelation",
    "agg_linear_trend": "aggregated linear trend",
    "approximate_entropy": "approximate entropy",
    "ar_coefficient": "autoregressive coefficient",
    "augmented_dickey_fuller": "Augmented Dickey–Fuller test",
    "autocorrelation": "autocorrelation",
    "benford_correlation": "Benford correlation",
    "binned_entropy": "binned entropy",
    "c3": "C3 nonlinearity statistic",
    "change_quantiles": "change within quantiles",
    "cid_ce": "complexity estimate",
    "count_above": "count above threshold",
    "count_above_mean": "count above mean",
    "count_below": "count below threshold",
    "count_below_mean": "count below mean",
    "cwt_coefficients": "continuous wavelet coefficient",
    "energy_ratio_by_chunks": "energy ratio by chunk",
    "fft_aggregated": "FFT aggregated statistic",
    "fft_coefficient": "Fourier coefficient",
    "first_location_of_maximum": "first location of maximum",
    "first_location_of_minimum": "first location of minimum",
    "fourier_entropy": "Fourier entropy",
    "friedrich_coefficients": "Friedrich coefficient",
    "has_duplicate": "has duplicate",
    "has_duplicate_max": "has duplicate maximum",
    "has_duplicate_min": "has duplicate minimum",
    "index_mass_quantile": "index of mass quantile",
    "kurtosis": "kurtosis",
    "large_standard_deviation": "large standard deviation flag",
    "last_location_of_maximum": "last location of maximum",
    "last_location_of_minimum": "last location of minimum",
    "length": "length",
    "linear_trend": "linear trend",
    "linear_trend_timewise": "linear trend (timewise)",
    "longest_strike_above_mean": "longest strike above mean",
    "longest_strike_below_mean": "longest strike below mean",
    "matrix_profile": "matrix profile",
    "max_langevin_fixed_point": "maximum Langevin fixed point",
    "maximum": "maximum",
    "mean": "mean",
    "mean_abs_change": "mean absolute change",
    "mean_change": "mean change",
    "mean_n_absolute_max": "mean of n largest absolute values",
    "mean_second_derivative_central": "mean central second derivative",
    "median": "median",
    "minimum": "minimum",
    "number_crossing_m": "number of crossings",
    "number_cwt_peaks": "number of CWT peaks",
    "number_peaks": "number of peaks",
    "partial_autocorrelation": "partial autocorrelation",
    "percentage_of_reoccurring_datapoints_to_all_datapoints": "percentage of reoccurring data points",
    "percentage_of_reoccurring_values_to_all_values": "percentage of reoccurring values",
    "permutation_entropy": "permutation entropy",
    "quantile": "quantile",
    "query_similarity_count": "query similarity count",
    "range_count": "range count",
    "ratio_beyond_r_sigma": "ratio of values beyond r-sigma",
    "ratio_value_number_to_time_series_length": "ratio of unique values to length",
    "root_mean_square": "root mean square",
    "sample_entropy": "sample entropy",
    "skewness": "skewness",
    "spkt_welch_density": "Welch spectral density",
    "standard_deviation": "standard deviation",
    "sum_of_reoccurring_data_points": "sum of reoccurring data points",
    "sum_of_reoccurring_values": "sum of reoccurring values",
    "sum_values": "sum",
    "symmetry_looking": "symmetry looking test",
    "time_reversal_asymmetry_statistic": "time-reversal asymmetry statistic",
    "value_count": "value count",
    "variance": "variance",
    "variance_larger_than_standard_deviation": "variance larger than standard deviation flag",
    "variation_coefficient": "coefficient of variation",
}


def slugify_class_name(class_name: str) -> str:
    """Convert a class name to a slug for use in filenames.

    Converts CamelCase class names to lowercase hyphen-separated slugs.
    Handles consecutive uppercase letters (e.g., HTTP, XML) correctly.
    Used for creating classifier-specific directory names.

    Args:
        class_name (str): Class name in CamelCase format.

    Returns:
        str: Slugified class name in lowercase with hyphens.

    Examples:
        >>> slugify_class_name("MyClassName")
        'my-class-name'
        >>> slugify_class_name("HTTPSConnection")
        'https-connection'
        >>> slugify_class_name("XMLParser")
        'xml-parser'
        >>> slugify_class_name("XGBClassifier")
        'xgb-classifier'
    """
    # Insert hyphens before uppercase letters (except at start)
    s = re.sub("([a-z0-9])([A-Z])", r"\1-\2", class_name)
    # Handle consecutive uppercase letters (e.g., HTTP)
    s = re.sub("([A-Z]+)([A-Z][a-z])", r"\1-\2", s)

    return s.lower()


def slugify(text: str, hyphen: str = "-") -> str:
    """Convert arbitrary text into a safe filename slug.

    Lowercases the input, replaces whitespace and underscores with the chosen
    separator, strips non-alphanumeric characters (except the separator), and
    collapses consecutive separators into one.

    Args:
        text (str): Text to slugify.
        hyphen (str): Separator character to use. Defaults to ``"-"``.

    Returns:
        str: Slugified filename-safe string.

    Examples:
        >>> slugify("Hello World")
        'hello-world'
        >>> slugify("Hello World", hyphen="_")
        'hello_world'
        >>> slugify("  Multiple   Spaces  ")
        'multiple-spaces'
    """
    s = text.lower()
    s = re.sub(r"[\s_]+", hyphen, s)
    escaped = re.escape(hyphen)
    s = re.sub(rf"[^a-z0-9{escaped}]", "", s)
    s = re.sub(rf"{escaped}+", hyphen, s)
    return s.strip(hyphen)


def shorten_feature_name(name: str) -> str:
    """Render a tsfresh feature name in a compact form for plot labels.

    Splits the canonical ``column__calculator[__key_value]*`` form, swaps
    the calculator for a short alias when one is known, drops the param
    keys (the calculator implies them), and joins the remaining values in
    parentheses. Names that do not match the tsfresh shape are returned
    unchanged.

    Args:
        name (str): Canonical tsfresh feature name.

    Returns:
        str: A compact label suitable for axis ticks.

    Examples:
        >>> shorten_feature_name('dsar_f3-f4__fft_coefficient__attr_"abs"__coeff_91')
        'dsar_f3-f4 | fft_coef(abs, 91)'
        >>> shorten_feature_name("entropy__variance")
        'entropy | variance'
    """
    calc_abbrev: dict[str, str] = {
        "fft_coefficient": "fft_coef",
        "fft_aggregated": "fft_agg",
        "change_quantiles": "chg_q",
        "agg_autocorrelation": "agg_autocorr",
        "partial_autocorrelation": "partial_autocorr",
        "autocorrelation": "autocorr",
        "agg_linear_trend": "agg_lin_trend",
        "linear_trend": "lin_trend",
        "linear_trend_timewise": "lin_trend_tw",
        "index_mass_quantile": "idx_mass_q",
        "time_reversal_asymmetry_statistic": "time_rev_asym",
        "friedrich_coefficients": "friedrich_coef",
        "ar_coefficient": "ar_coef",
        "cwt_coefficients": "cwt_coef",
        "binned_entropy": "bin_ent",
        "permutation_entropy": "perm_ent",
        "sample_entropy": "samp_ent",
        "approximate_entropy": "approx_ent",
        "absolute_sum_of_changes": "abs_sum_chg",
        "large_standard_deviation": "large_std",
        "longest_strike_above_mean": "lng_strike_above_mean",
        "longest_strike_below_mean": "lng_strike_below_mean",
        "mean_abs_change": "mean_abs_chg",
        "mean_change": "mean_chg",
        "mean_second_derivative_central": "mean_2nd_deriv",
        "number_crossing_m": "n_crossings",
        "number_peaks": "n_peaks",
        "number_cwt_peaks": "n_cwt_peaks",
        "percentage_of_reoccurring_datapoints_to_all_datapoints": "pct_reoccur_dp",
        "percentage_of_reoccurring_values_to_all_values": "pct_reoccur_vals",
        "ratio_beyond_r_sigma": "ratio_beyond",
        "ratio_value_number_to_time_series_length": "ratio_n_vals_to_len",
        "sum_of_reoccurring_data_points": "sum_reoccur_dp",
        "sum_of_reoccurring_values": "sum_reoccur_vals",
        "symmetry_looking": "sym_looking",
        "variance_larger_than_standard_deviation": "var_gt_std",
        "variation_coefficient": "var_coef",
        "augmented_dickey_fuller": "adf",
        "spkt_welch_density": "spkt_welch",
        "standard_deviation": "std",
        "root_mean_square": "rms",
        "quantile": "q",
        "skewness": "skew",
        "kurtosis": "kurt",
        "maximum": "max",
        "minimum": "min",
        "median": "med",
        "length": "len",
        "sum_values": "sum",
        "value_count": "val_cnt",
        "range_count": "range_cnt",
        "count_above": "cnt_above",
        "count_above_mean": "cnt_above_mean",
        "count_below": "cnt_below",
        "count_below_mean": "cnt_below_mean",
        "first_location_of_maximum": "first_loc_max",
        "first_location_of_minimum": "first_loc_min",
        "last_location_of_maximum": "last_loc_max",
        "last_location_of_minimum": "last_loc_min",
    }

    parts = name.split("__")
    if len(parts) < 2:
        return name
    column, calculator, *params = parts
    short_calc = calc_abbrev.get(calculator, calculator)

    values: list[str] = []
    for param in params:
        match = _PARAM_VALUE_RE.search(param)
        values.append(match.group(1).strip('"') if match else param)

    if values:
        return f"{column} | {short_calc}({', '.join(values)})"
    return f"{column} | {short_calc}"


def humanize_feature_name(
    name: str,
    freq_bands: dict[str, tuple[float, float]] | None = None,
) -> str:
    """Render a tsfresh feature name as a plain-English phrase.

    Splits the canonical ``column__calculator[__key_value]*`` form, humanises
    the column and calculator, and formats any trailing parameter tokens as
    ``key=value`` pairs. The output shape is
    ``"{Humanized calculator} ({key=value, ...}) of {humanized column}"``;
    the parenthesized clause is dropped when the calculator has no parameters.
    Names that do not match the ``column__calculator`` shape are returned
    unchanged.

    The column half is resolved by :func:`_humanize_column` against
    ``freq_bands`` — when ``None`` (default), the built-in
    :data:`_DEFAULT_FREQ_BANDS` table (matching ``CalculateTremor``'s
    defaults) is used. Pass an explicit mapping if the tremor was
    calculated with non-default bands so the Hz values in the output
    reflect the actual edges. The calculator half is resolved against
    :data:`_CALCULATOR_HUMANIZATIONS`; unknown calculators fall back to
    ``name.replace("_", " ")``.

    Args:
        name (str): Canonical tsfresh feature name.
        freq_bands (dict[str, tuple[float, float]] | None, optional):
            Override of the default frequency-band edges keyed by band
            token (``"f0"``, ``"f1"``, …). When ``None`` (default), the
            built-in :data:`_DEFAULT_FREQ_BANDS` table is used.

    Returns:
        str: A plain-English phrase suitable for a description column.

    Examples:
        >>> humanize_feature_name('dsar_f3-f4__fft_coefficient__attr_"abs"__coeff_91')
        'Fourier coefficient (attr=abs, coeff=91) of DSAR ratio 4.5/8 Hz'
        >>> humanize_feature_name("rsam_f2__mean")
        'Mean of RSAM 2–5 Hz'
        >>> humanize_feature_name("entropy__autocorrelation__lag_1")
        'Autocorrelation (lag=1) of Shannon entropy'
        >>> custom = {"f0": (0.05, 0.5), "f1": (0.5, 3.0), "f2": (3.0, 10.0)}
        >>> humanize_feature_name("dsar_f0-f1__mean", freq_bands=custom)
        'Mean of DSAR ratio 0.05/0.5 Hz'
        >>> humanize_feature_name("rsam_f2__mean", freq_bands=custom)
        'Mean of RSAM 3–10 Hz'
    """
    parts = name.split("__")
    if len(parts) < 2:
        return name
    column, calculator, *params = parts

    bands = freq_bands if freq_bands is not None else _DEFAULT_FREQ_BANDS
    humanized_column = _humanize_column(column, bands)
    humanized_calc = _CALCULATOR_HUMANIZATIONS.get(
        calculator, calculator.replace("_", " ")
    )

    formatted_params: list[str] = []
    for param in params:
        match = _PARAM_VALUE_RE.search(param)
        if match is None:
            formatted_params.append(param)
            continue
        key = param[: match.start()]
        value = match.group(1).strip('"')
        formatted_params.append(f"{key}={value}" if key else value)

    prefix = humanized_calc[:1].upper() + humanized_calc[1:]
    if formatted_params:
        return f"{prefix} ({', '.join(formatted_params)}) of {humanized_column}"
    return f"{prefix} of {humanized_column}"


def get_classifier_label(classifier_name: str) -> str:
    """Return a human-readable label for a classifier given its scikit-learn class name.

    Looks up ``classifier_name`` in a fixed mapping of class names to display labels.
    If the name is not found (e.g., an unrecognised or custom classifier), the input
    string is returned unchanged.

    Args:
        classifier_name (str): Scikit-learn class name of the classifier, e.g.
            ``"RandomForestClassifier"`` or ``"XGBClassifier"``.

    Returns:
        str: Human-readable display label, e.g. ``"Random Forest"`` or ``"XGBoost"``.
            Returns ``classifier_name`` unchanged if not found in the mapping.
    """
    classifier_slugs = {
        "SVC": "svm",
        "KNeighborsClassifier": "KNN",
        "DecisionTreeClassifier": "Decision Tree",
        "RandomForestClassifier": "Random Forest",
        "LiteRandomForestClassifier": "(lite) Random Forest",
        "GradientBoostingClassifier": "Gradient Boosting",
        "XGBClassifier": "XGBoost",
        "MLPClassifier": "Neural Network",
        "GaussianNB": "Naive Bayes",
        "LogisticRegression": "Logistic Regression",
        "VotingClassifier": "Voting Classifier",
    }

    if classifier_name not in classifier_slugs:
        return classifier_name

    return classifier_slugs[classifier_name]
