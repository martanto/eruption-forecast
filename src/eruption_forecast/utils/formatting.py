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

    pdf_metadata = {
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

    return pdf_metadata
