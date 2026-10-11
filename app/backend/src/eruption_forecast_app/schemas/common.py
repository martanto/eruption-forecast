"""Shared validation for package parameters."""

from typing import Any
from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator


class Parameters(BaseModel):
    """Reject unknown package options and invalid scientific ranges."""

    model_config = ConfigDict(extra="forbid", validate_default=False)

    @field_validator("*", mode="after")
    @classmethod
    def _validate_parameter(cls, value: Any, info: Any) -> Any:
        name = info.field_name
        if value is None or value == "":
            return value
        if name.endswith("date") and isinstance(value, str):
            datetime.fromisoformat(value)
        if name == "eruption_dates":
            for date in value:
                datetime.fromisoformat(date)
        if (
            name
            in {
                "n_jobs",
                "n_grids",
                "seeds",
                "cv_splits",
                "top_n_features",
                "window_size",
                "window_step",
                "max_display",
                "dpi",
                "plot_interval",
            }
            and value < 1
        ):
            raise ValueError("must be at least 1")
        if (
            name
            in {
                "minimum_completion",
                "minimum_completion_ratio",
                "minority_threshold",
                "plot_threshold",
                "threshold",
            }
            and not 0 <= value <= 1
        ):
            raise ValueError("must be between 0 and 1")
        if name == "classifiers":
            choices = [value] if isinstance(value, str) else value
            if not choices or set(choices) - {
                "rf",
                "lite-rf",
                "gb",
                "xgb",
                "svm",
                "lr",
                "nn",
                "dt",
                "knn",
                "nb",
                "voting",
            }:
                raise ValueError("select supported classifiers")
        if name == "methods" and (
            not value or set(value) - {"rsam", "dsar", "entropy"}
        ):
            raise ValueError("select rsam, dsar, or entropy")
        if name in {"figsize", "legend_position"} and len(value) != 2:
            raise ValueError("requires two values")
        return value
