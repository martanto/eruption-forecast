"""End-to-end research workflow for volcanic eruption forecasting.

This script covers all pipeline stages in a single place, with boolean
flags to skip already-completed stages. Edit the configuration block at
the top, then run::

    python workflow.py

Stage flags default to True. Set any flag to False to skip that stage.
"""

import os
import warnings
from typing import Any

from eruption_forecast import ForecastModel, notify
from eruption_forecast.decorators import timer
from eruption_forecast.model.classifier_comparator import ClassifierComparator
from eruption_forecast.model.multi_model_evaluator import MultiModelEvaluator


warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Stage flags — set to False to skip a stage
# ---------------------------------------------------------------------------

RUN_CALCULATE = True
RUN_BUILD_LABEL = True
RUN_EXTRACT_FEATURES = True
RUN_TRAIN = True            # with_evaluation=True → 80/20 split + metrics
RUN_FORECAST = True         # predict_proba on future window
RUN_EVALUATE_PER_MODEL = True   # MultiModelEvaluator aggregate plots per classifier
RUN_COMPARE_MODELS = True       # ClassifierComparator cross-classifier plots
SAVE_CONFIG = True              # save pipeline parameters to YAML after all stages

# ---------------------------------------------------------------------------
# Common configuration
# ---------------------------------------------------------------------------

SDS_DIR: str = r"D:\Data\OJN"

PARAMS: dict[str, Any] = {
    "root_dir": r"D:\Projects\eruption-forecast",
    "station": "OJN",
    "channel": "EHZ",
    "start_date": "2025-01-01",
    "end_date": "2025-08-24",
    "window_size": 2,
    "volcano_id": "Lewotobi Laki-laki",
    "verbose": True,
    "debug": False,
}

ERUPTIONS: list[str] = [
    "2025-03-20",
    "2025-04-22",
    "2025-05-18",
    "2025-06-17",
    "2025-07-07",
    "2025-08-01",
    "2025-08-17",
]

# ---------------------------------------------------------------------------
# Per-stage kwargs
# ---------------------------------------------------------------------------

CALCULATE_KWARGS: dict[str, Any] = {
    "source": "sds",
    "sds_dir": SDS_DIR,
    "plot_daily": True,
    "save_plot": True,
    "remove_outlier_method": "maximum",
}

# Used when RUN_TRAIN with with_evaluation=False — shorter end_date leaves
# room for the forecast window defined in FORECAST_KWARGS.
BUILD_LABEL_TRAIN_KWARGS: dict[str, Any] = {
    "start_date": "2025-01-01",
    "end_date": "2025-07-27",
    "day_to_forecast": 2,
    "window_step": 6,
    "window_step_unit": "hours",
    "eruption_dates": ERUPTIONS,
    "verbose": True,
}

# Used when RUN_TRAIN with with_evaluation=True — full date range covers all
# eruption events so the 80/20 test split includes every labelled period.
BUILD_LABEL_EVALUATE_KWARGS: dict[str, Any] = {
    "start_date": "2025-01-01",
    "end_date": "2025-08-24",
    "day_to_forecast": 2,
    "window_step": 6,
    "window_step_unit": "hours",
    "eruption_dates": ERUPTIONS,
    "verbose": True,
}

EXTRACT_FEATURES_KWARGS: dict[str, Any] = {
    "select_tremor_columns": ["rsam_f2", "rsam_f3", "rsam_f4", "dsar_f3-f4"],
    "save_tremor_matrix_per_method": True,
    "save_tremor_matrix_per_id": False,
    "exclude_features": [
        "agg_linear_trend",
        "linear_trend_timewise",
        "length",
        "has_duplicate_max",
        "has_duplicate_min",
        "has_duplicate",
    ],
    "use_relevant_features": True,
    "overwrite": False,
}

TRAIN_KWARGS: dict[str, Any] = {
    "with_evaluation": True,
    "classifier": ["lite-rf", "rf"],
    "cv_strategy": "stratified",
    "random_state": 0,
    "total_seed": 100,
    "number_of_significant_features": 20,
    "sampling_strategy": 0.75,
    "save_all_features": True,
    "plot_significant_features": True,
    "n_jobs": 4,
    "grid_search_n_jobs": 2,
    "overwrite": False,
    "verbose": True,
}

FORECAST_KWARGS: dict[str, Any] = {
    "start_date": "2025-07-28",
    "end_date": "2025-08-20",
    "window_step": 10,
    "window_step_unit": "minutes",
}


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


@timer("Workflow")
@notify("Laptop - Workflow")
def main() -> None:
    """Run the full eruption-forecast research workflow.

    Executes up to seven pipeline stages in order, each guarded by a boolean
    stage flag defined at module level. Stages that are disabled are logged
    and skipped without affecting downstream stages that do not depend on them.

    The stages are:
        1. Calculate tremor (CalculateTremor via ForecastModel.calculate)
        2. Build labels — uses BUILD_LABEL_EVALUATE_KWARGS when with_evaluation=True,
           BUILD_LABEL_TRAIN_KWARGS otherwise (LabelBuilder via ForecastModel.build_label)
        3. Extract features (FeaturesBuilder via ForecastModel.extract_features)
        4. Train with evaluation (ModelTrainer 80/20 split + metrics JSON per seed)
        5. Forecast (ModelPredictor.predict_proba on future window)
        6. Evaluate per model (MultiModelEvaluator aggregate plots per classifier)
        7. Compare models (ClassifierComparator cross-classifier plots, ≥2 classifiers)
        8. Save config (ForecastModel.save_config → YAML at station_dir/config_workflow.yaml)
    """
    fm = ForecastModel(
        overwrite=False,
        n_jobs=4,
        **PARAMS,
    )

    # -----------------------------------------------------------------------
    # Stage 1: Calculate tremor
    # -----------------------------------------------------------------------
    if RUN_CALCULATE:
        fm.calculate(**CALCULATE_KWARGS)
    else:
        print("[workflow] Skipping Stage 1: calculate tremor")

    # -----------------------------------------------------------------------
    # Stage 2: Build labels
    # Use the full date range when training with evaluation (80/20 split),
    # or the shorter range when training without evaluation to leave headroom
    # for the forecast window.
    # -----------------------------------------------------------------------
    if RUN_BUILD_LABEL:
        with_evaluation: bool = bool(TRAIN_KWARGS.get("with_evaluation", False))
        build_label_kwargs = (
            BUILD_LABEL_EVALUATE_KWARGS if with_evaluation else BUILD_LABEL_TRAIN_KWARGS
        )
        fm.build_label(**build_label_kwargs)
    else:
        print("[workflow] Skipping Stage 2: build label")

    # -----------------------------------------------------------------------
    # Stage 3: Extract features
    # -----------------------------------------------------------------------
    if RUN_EXTRACT_FEATURES:
        fm.extract_features(**EXTRACT_FEATURES_KWARGS)
    else:
        print("[workflow] Skipping Stage 3: extract features")

    # -----------------------------------------------------------------------
    # Stage 4: Train (with evaluation — 80/20 split + per-seed metrics JSON)
    # -----------------------------------------------------------------------
    if RUN_TRAIN:
        fm.train(**TRAIN_KWARGS)
    else:
        print("[workflow] Skipping Stage 4: train")

    # -----------------------------------------------------------------------
    # Stage 5: Forecast (predict_proba on future window)
    # -----------------------------------------------------------------------
    if RUN_FORECAST:
        fm.forecast(**FORECAST_KWARGS)
    else:
        print("[workflow] Skipping Stage 5: forecast")

    # -----------------------------------------------------------------------
    # Stage 6: Evaluate per model (MultiModelEvaluator)
    # -----------------------------------------------------------------------
    if RUN_EVALUATE_PER_MODEL:
        if not fm.trained_models:
            print(
                "[workflow] Skipping Stage 6: no trained_models available. "
                "Run Stage 4 first or load a ForecastModel with trained_models set."
            )
        else:
            for name, csv_path in fm.trained_models.items():
                print(f"[workflow] Stage 6: evaluating '{name}'")
                csv_dir = os.path.dirname(os.path.abspath(csv_path))
                metrics_dir: str | None = os.path.join(csv_dir, "metrics")
                if not os.path.isdir(metrics_dir):
                    metrics_dir = None

                evaluator = MultiModelEvaluator(
                    trained_model_csv=csv_path,
                    metrics_dir=metrics_dir,
                )
                if metrics_dir is not None:
                    evaluator.get_aggregate_metrics()
                evaluator.plot_all()
    else:
        print("[workflow] Skipping Stage 6: evaluate per model")

    # -----------------------------------------------------------------------
    # Stage 7: Compare models (ClassifierComparator)
    # -----------------------------------------------------------------------
    if RUN_COMPARE_MODELS:
        if len(fm.trained_models) < 2:
            print(
                "[workflow] Skipping Stage 7: comparison requires at least 2 classifiers "
                f"(found {len(fm.trained_models)})."
            )
        else:
            print("[workflow] Stage 7: comparing classifiers")
            comparator = ClassifierComparator(
                classifiers=fm.trained_models,
                output_dir=fm.station_dir,
            )
            comparator.plot_all()
    else:
        print("[workflow] Skipping Stage 7: compare models")

    # -----------------------------------------------------------------------
    # Stage 8: Save config (pipeline parameters → YAML)
    # -----------------------------------------------------------------------
    if SAVE_CONFIG:
        config_path = os.path.join(fm.station_dir, "config_workflow.yaml")
        saved = fm.save_config(path=config_path, fmt="yaml")
        print(f"[workflow] Config saved: {saved}")
    else:
        print("[workflow] Skipping Stage 8: save config")


if __name__ == "__main__":
    main()
