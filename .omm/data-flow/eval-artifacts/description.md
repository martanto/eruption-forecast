Under `evaluation/{training|prediction}/`:
- `classifiers/{clf}/predictions/{y_proba,y_pred}.csv` — `(n_samples, n_seeds)` matrices written by `MetricsEnsemble.compute()`.
- `classifiers/{clf}/figures/aggregate/{plot_name}.{png,csv}` — aggregate plots.
- `classifiers/{clf}/figures/{plot_name}/{seed:05d}.png` — per-seed plots.
- `comparison/` — `ClassifierComparator` rankings and cross-classifier figures.
- `labels/y_true.csv` (prediction reuse only).
- `MetricsEnsemble.pkl` (optional, via `MetricsEnsemble.save()`).
