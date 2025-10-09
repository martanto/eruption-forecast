#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Third party imports
from pkg_resources import get_distribution

# Project imports
from eruption_forecast.calculate import Calculate

__version__ = get_distribution("eruption-forecast").version
__author__ = "Martanto"
__author_email__ = "martanto@live.com"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2025, Martanto"
__url__ = "https://github.com/martanto/eruption-forecast"

__all__ = [
    "__version__",
    "__author__",
    "__author_email__",
    "__license__",
    "__copyright__",
    "Calculate",
]
