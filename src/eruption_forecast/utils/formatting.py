"""Text formatting utilities.

This module provides utilities for converting class names to slugified format
for use in filenames and directory names.
"""


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
    import re

    # Insert hyphens before uppercase letters (except at start)
    s = re.sub("([a-z0-9])([A-Z])", r"\1-\2", class_name)
    # Handle consecutive uppercase letters (e.g., HTTP)
    s = re.sub("([A-Z]+)([A-Z][a-z])", r"\1-\2", s)

    return s.lower()
