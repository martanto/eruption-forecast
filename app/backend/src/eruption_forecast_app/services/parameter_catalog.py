"""Form metadata backed by explicit Pydantic package contracts."""

from eruption_forecast_app.schemas.stages import STAGES
from eruption_forecast_app.schemas.scenario import Configuration, resolve


def catalog() -> dict:
    """Expose schema definitions and defaults to the parameter editor.

    Example:
        >>> 'training' in catalog()['stages']
        True
    """
    effective = resolve(Configuration())
    return {
        "schema": Configuration.model_json_schema(),
        "stages": {name: effective[name] for name in STAGES},
        "defaults": Configuration().model_dump(mode="json"),
        "docs": {
            "calculation": "API-Reference",
            "training": "Training-Workflow",
            "prediction": "Prediction-Workflow",
            "evaluation": "Evaluation-Workflow",
            "explanation": "Explanation-Workflow",
        },
    }
