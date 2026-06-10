"""Text formatting utilities.

This module provides utilities for converting class names to slugified format
for use in filenames and directory names.
"""

import os
import re
from importlib.metadata import metadata


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
        'dsar_f3-f4 · fft_coef(abs, 91)'
        >>> shorten_feature_name("entropy__variance")
        'entropy · variance'
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

    #  Trailing-value token inside a tsfresh param segment (``attr_"abs"`` /
    #  ``coeff_91`` / ``isabs_True``). Anchored at end-of-string so multi-
    #  underscore keys like ``f_agg_"var"`` resolve to ``"var"`` rather than
    #  ``agg_"var"``.
    param_value_re = re.compile(r'_("[^"]*"|-?\d+(?:\.\d+)?|True|False)$')

    parts = name.split("__")
    if len(parts) < 2:
        return name
    column, calculator, *params = parts
    short_calc = calc_abbrev.get(calculator, calculator)

    values: list[str] = []
    for param in params:
        match = param_value_re.search(param)
        values.append(match.group(1).strip('"') if match else param)

    if values:
        return f"{column} · {short_calc}({', '.join(values)})"
    return f"{column} · {short_calc}"


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


def pdf_metadata(title: str | None = None) -> dict[str, str]:
    """Build PDF metadata dict from package metadata and environment.

    Reads package version and homepage URL from ``importlib.metadata``.
    The ``Author`` field is resolved from the environment in priority order:
    ``GIT_AUTHOR_NAME`` → ``USERNAME`` (Windows) → ``USER`` (Unix) →
    ``"eruption-forecast"`` as a last-resort fallback.

    Returns:
        dict[str, str]: Metadata dict suitable for passing to
        ``matplotlib``'s ``savefig(metadata=...)``.  Keys: ``Title``,
        ``Author``, ``Subject``, ``Keywords``, ``Creator``.
    """
    package_metadata = metadata("eruption-forecast")
    version: str = package_metadata["Version"]

    _pdf_metadata = {
        "Title": title or "Eruption probability forecast",
        "Author": (
            os.environ.get("GIT_AUTHOR_NAME")
            or os.environ.get("USERNAME")
            or os.environ.get("USER")
            or "eruption-forecast"
        ),
        "Subject": "Eruption probability forecast",
        "Keywords": "eruption, forecast, seismic, tremor",
        "Creator": f"eruption-forecast v{version}",
    }

    return _pdf_metadata
