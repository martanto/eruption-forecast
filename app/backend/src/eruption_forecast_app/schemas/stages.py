"""Complete stage contracts, including public method options."""

from typing import Literal

from pydantic import Field

from eruption_forecast_app.schemas.common import Parameters
from eruption_forecast_app.schemas.training import (
    TrainingFit,
    TrainingLabel,
    TrainingFeatures,
    TrainingConstructor,
    TrainingLoadFeatures,
)
from eruption_forecast_app.schemas.evaluation import (
    EvaluationLabel,
    EvaluationCompare,
    EvaluationExecute,
    EvaluationConstructor,
)
from eruption_forecast_app.schemas.prediction import (
    PredictionLabel,
    PredictionFeatures,
    PredictionForecast,
    PredictionConstructor,
    PredictionLoadFeatures,
)
from eruption_forecast_app.schemas.calculation import (
    CalculationRun,
    CalculationConstructor,
)
from eruption_forecast_app.schemas.explanation import (
    ExplanationPlot,
    ExplanationExecute,
    ExplanationConstructor,
)
from eruption_forecast_app.schemas.forecast_plot import ForecastPlot


class Calculation(Parameters):
    constructor: CalculationConstructor = Field(default_factory=CalculationConstructor)
    run: CalculationRun = Field(default_factory=CalculationRun)
    source: Literal["sds", "fdsn"] = "sds"
    sds_dir: str | None = None
    client_url: str | None = None
    freq_bands: list[tuple[float, float]] = Field(
        default_factory=lambda: [(0.01, 0.1), (0.1, 2), (2, 5), (4.5, 8), (8, 16)]
    )


class Training(Parameters):
    constructor: TrainingConstructor = Field(default_factory=TrainingConstructor)
    label: TrainingLabel = Field(default_factory=TrainingLabel)
    features: TrainingFeatures = Field(default_factory=TrainingFeatures)
    fit: TrainingFit = Field(default_factory=TrainingFit)
    load_features: TrainingLoadFeatures = Field(default_factory=TrainingLoadFeatures)
    use_cache: bool = True


class Prediction(Parameters):
    constructor: PredictionConstructor = Field(default_factory=PredictionConstructor)
    label: PredictionLabel = Field(default_factory=PredictionLabel)
    features: PredictionFeatures = Field(default_factory=PredictionFeatures)
    forecast: PredictionForecast = Field(default_factory=PredictionForecast)
    plot: ForecastPlot = Field(default_factory=ForecastPlot)
    load_features: PredictionLoadFeatures = Field(
        default_factory=PredictionLoadFeatures
    )
    use_features_from: Literal["all", "files", "training"] = "all"
    enable_segments_plot: bool = False
    use_cache: bool = True


class Evaluation(Parameters):
    constructor: EvaluationConstructor = Field(default_factory=EvaluationConstructor)
    label: EvaluationLabel = Field(default_factory=EvaluationLabel)
    execute: EvaluationExecute = Field(default_factory=EvaluationExecute)
    compare: EvaluationCompare = Field(default_factory=EvaluationCompare)
    model_source: Literal["training", "prediction"] = "prediction"


class Explanation(Parameters):
    constructor: ExplanationConstructor = Field(default_factory=ExplanationConstructor)
    execute: ExplanationExecute = Field(default_factory=ExplanationExecute)
    plot: ExplanationPlot = Field(default_factory=ExplanationPlot)
    model_source: Literal["training", "prediction"] = "prediction"


STAGES = {
    "calculation": Calculation,
    "training": Training,
    "prediction": Prediction,
    "evaluation": Evaluation,
    "explanation": Explanation,
}
