"""Generate explicit Pydantic contracts from package source signatures.

Example:
    uv run python -m eruption_forecast_app.generate_parameters
"""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SPECS = {
    "calculation": (
        "tremor/calculate_tremor.py",
        {"__init__": "CalculationConstructor", "run": "CalculationRun"},
    ),
    "training": (
        "model/training_model.py",
        {
            "__init__": "TrainingConstructor",
            "build_label": "TrainingLabel",
            "extract_features": "TrainingFeatures",
            "fit": "TrainingFit",
            "load_features": "TrainingLoadFeatures",
        },
    ),
    "prediction": (
        "model/prediction_model.py",
        {
            "__init__": "PredictionConstructor",
            "build_label": "PredictionLabel",
            "extract_features": "PredictionFeatures",
            "forecast": "PredictionForecast",
            "load_features": "PredictionLoadFeatures",
        },
    ),
    "evaluation": (
        "model/evaluation_model.py",
        {
            "__init__": "EvaluationConstructor",
            "build_label": "EvaluationLabel",
            "evaluate": "EvaluationExecute",
            "compare": "EvaluationCompare",
        },
    ),
    "explanation": (
        "model/explanation_model.py",
        {
            "__init__": "ExplanationConstructor",
            "explain": "ExplanationExecute",
            "plot": "ExplanationPlot",
        },
    ),
    "forecast_plot": ("plots/forecast_plots.py", {"plot_forecast": "ForecastPlot"}),
}


def _generate() -> None:
    for stage, (source, methods) in SPECS.items():
        tree = ast.parse(
            (ROOT / "src/eruption_forecast" / source).read_text(encoding="utf-8")
        )
        lines = [
            '"""Generated package parameters; regenerate with eruption_forecast_app.generate_parameters."""',
            "",
            "from typing import Literal",
            "",
            "from pydantic import Field",
            "",
            "from eruption_forecast_app.schemas.common import Parameters",
            "",
        ]
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name not in methods:
                continue
            lines += [
                "",
                f"class {methods[node.name]}(Parameters):",
                f'    """Parameters for ``{source}:{node.name}``."""',
                "",
            ]
            args = node.args.args
            defaults = [None] * (
                len(args) - len(node.args.defaults)
            ) + node.args.defaults
            for arg, default in zip(args, defaults, strict=True):
                if arg.arg in {"self", "df", "label_df"}:
                    continue
                annotation = ast.unparse(arg.annotation) if arg.annotation else "str"
                for symbol in [
                    "pd.DataFrame",
                    "ClassifierEnsemble",
                    "SeedEnsemble",
                    "TrainingModel",
                    "PredictionModel",
                    "datetime",
                ]:
                    annotation = annotation.replace(symbol, "str")
                annotation = " | ".join(dict.fromkeys(annotation.split(" | ")))
                if (
                    arg.arg in {"n_jobs", "overwrite", "verbose"}
                    and "None" not in annotation
                ):
                    annotation += " | None"
                if default is None:
                    value = "None"
                    annotation += " | None" if "None" not in annotation else ""
                else:
                    value = ast.unparse(default)
                    if value == "DEFAULT_MINIMUM_COMPLETION_RATIO":
                        value = "0.3"
                description = (
                    arg.arg.replace("_", " ").capitalize()
                    + f". Package option: {node.name}."
                )
                lines.append(
                    f"    {arg.arg}: {annotation} = Field(default={value}, description={description!r})"
                )
        (Path(__file__).parent / f"schemas/{stage}.py").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    _generate()
