Cross-cutting utilities. Pure-Python helpers reused by every domain layer; never imports from `model/` to avoid cycles.

Files:
- array.py — outlier/anomaly detection, seed-probability aggregation, `predict_proba_from_estimator`.
- window.py — `construct_windows` (the shared window grid for label + features), per-window metric helpers.
- date_utils.py — date parsing, `parse_label_filename`, datetime-index helpers, `label_id_to_datetime`.
- ml.py — sklearn glue: `random_under_sampler`, `grid_search_cv`, `merge_seed_models`, `merge_all_classifiers`, `get_classifier_models`, `compute_g_mean`.
- validation.py — guards for random states, date ranges, window steps, column lists, sampling consistency.
- pathutils.py — `resolve_output_dir`, `setup_nslc_directories`, `ensure_dir`, `save_figure`, `save_data`.
- dataframe.py — label/CSV loaders and shape helpers (`load_label_csv`, `concat_features`, `remove_anomalies`).
- formatting.py — human-readable durations / sizes, `slugify`, `slugify_class_name`, PDF metadata builder.
