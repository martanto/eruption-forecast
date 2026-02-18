# Code Review — eruption-forecast

_Reviewed: 2026-02-19_

All source files under `src/eruption_forecast/` were read and analyzed. Findings are grouped by severity.

---

## 🔴 Bugs

Incorrect behavior, data corruption, or runtime failures.

---

### `utils/ml.py:432` — `compute_model_probabilities()` early-exit compares seed value, not count

```python
if number_of_seeds is not None and random_state_int > number_of_seeds: break
```

`random_state_int` is the **seed value** (e.g. 42, 100, 500), not a loop counter. If any seed exceeds `number_of_seeds`, iteration stops early and all remaining models are skipped. With a 500-seed run using seeds starting at 100, the loop exits after the first iteration. Fix: use an explicit counter.

---

### `tremor/dsar.py:133` and `tremor/calculate_tremor.py:891` — `value_multiplier` silently ignored for values ≤ 1

```python
if value_multiplier > 1:
    series = series.apply(lambda x: x * value_multiplier)
```

A multiplier of e.g. `0.5` (scale down) is never applied. The condition should be `if value_multiplier != 1.0:`.

---

### `model/model_trainer.py:408` — `classifier_dir` built from unresolved `output_dir`

```python
# output_dir resolved two lines above into self.output_dir
self.classifier_dir = os.path.join(
    output_dir,  # ← wrong: should be self.output_dir
    self.classifier_slug_name,
    self.classifier_slug_cv_name,
)
```

When the caller passes a relative path or `None`, `self.classifier_dir` is built from the raw unresolved value while `self.output_dir` holds the correct anchored path.

---

### `model/model_predictor.py:300` — `model_name` shadowed inside inner loop

```python
for model_name, df_models in self.trained_models.items():
    for random_state, model_path in df_models.items():
        model_name = f"{model_name}_seed_{random_state:05d}"  # overwrites outer variable
```

The outer `model_name` (classifier key) is overwritten on every inner iteration. Any subsequent use of `model_name` in the outer loop body — including `predict_log_metrics_summary` — receives the seed-specific string instead of the classifier name. Rename the inner variable to `seed_model_name`.

---

### `features/tremor_matrix_builder.py:467` — `self.csv` never set on cache hit

```python
self.df = pd.read_csv(tremor_matrix_csv)
return self   # self.csv remains None
```

After loading from the CSV cache, `self.csv` is never assigned. Downstream code that reads `self.csv` after a cache hit will get `None`.

---

### `label/label_data.py:306,320` — filename parsed with `split(".")` instead of `os.path.splitext`

```python
def basename(self) -> str:
    return self.filename.split(".")[0]   # wrong if path contains dots

def filetype(self) -> str:
    return self.filename.split(".")[1]   # fails for multi-part extensions
```

`os.path.splitext` handles all edge cases correctly. The current approach truncates at the first dot and gives the wrong answer for paths like `label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv.bak`.

---

### `features/features_builder.py:392-394` — `.index` mutated on a column slice

```python
y_index = y[ID_COLUMN]
y = y[ERUPTED_COLUMN]
y.index = y_index   # SettingWithCopyWarning / silently fails in pandas ≥ 3.0
```

Assigning to `.index` on a slice is unsafe in pandas ≥ 3.0. Replace with:

```python
y = pd.Series(y[ERUPTED_COLUMN].values, index=y[ID_COLUMN].values)
```

---

## 🟠 Logic Issues

Silent wrong results or misleading behavior.

---

### `utils/window.py:197-212` — `value_multiplier` ignored when `remove_outlier_method=None`

```python
elif remove_outlier_method is None:
    metric_value = metric_function(window_data)   # multiplier never applied

elif remove_outlier_method:
    ...
    if value_multiplier != 1.0 and not np.isnan(metric_value):
        metric_value *= value_multiplier           # only applied here
```

The multiplier is only applied in the outlier-removal branch. Any caller that passes `remove_outlier_method=None` with a non-unity `value_multiplier` gets silently incorrect values.

---

### `model/classifier_model.py` — `set_random_state()` does not invalidate cached `_model`

```python
@property
def model(self):
    if self._model is None:
        self._model = self._create_model()
    return self._model
```

If `.model` is accessed before `set_random_state()` is called, `_model` is cached with the old seed. The subsequent `set_random_state()` updates `self.random_state` but leaves `_model` pointing to the stale instance. Any training run after this silently uses the wrong seed.

---

### `tremor/calculate_tremor.py` — `create_daily_dir()` calls `shutil.rmtree` before directory exists

When `cleanup_daily_dir=True` and the daily directory has not yet been created, `shutil.rmtree` raises `FileNotFoundError`. Should guard with `if os.path.exists(daily_dir):` before the `rmtree` call.

---

### `model/forecast_model.py:786` — `source="sds"` with `sds_dir=None` silently does nothing

```python
if source == "sds" and sds_dir:    # falsy sds_dir short-circuits the branch
    ...
elif source == "fdsn":
    ...
```

If `source="sds"` and `sds_dir=None`, neither branch executes. `self.tremor_data` is never set and the pipeline continues in a broken state without any error.

---

### `utils/ml.py:435` — `np.stack` shape comment is transposed

```python
# (n_seeds, n_windows)
probabilities_eruption_matrix = np.stack(seed_eruption_probabilities, axis=1)
```

`np.stack(..., axis=1)` produces shape `(n_windows, n_seeds)`, not `(n_seeds, n_windows)`.

---

### `label/label_builder.py` — eruption labels re-applied unconditionally on cache hit

`build()` calls `update_df_eruptions(df)` even when loading from an existing CSV file. This silently re-labels eruption windows every time, which is only correct if the eruption configuration hasn't changed between runs.

---

### `dataframe.py:130` and `tremor_matrix_builder.py:483` — sampling period inferred from first two rows only

```python
sampling_rate = (df.index[1] - df.index[0]).seconds
```

This uses only the first timestamp pair. A gap or anomaly at the start of the data gives a wrong sampling rate with no warning.

---

## 🔵 Code Quality / Dead Code

---

### `model/forecast_model.py:1120` — `sleep(3)` wastes ~25 minutes per 500-seed run

```python
sleep(3)
```

Inserted only to stagger verbose console messages between seeds. Adds 25 minutes of wall time for a 500-seed run. Remove.

---

### `model/model_predictor.py:471-479` — `if labels_df is None` check is unreachable

```python
if isinstance(labels_df, pd.Series):
    labels_df = labels_df.to_frame()
...
if labels_df is None:   # can never be True at this point
```

After the Series → DataFrame conversion, `labels_df` cannot be `None`. Move the `None` check before the type conversion.

---

### `model/model_predictor.py:993-1016` — large commented-out block in `_plot_forecast()`

Dead code should be removed, not left commented out.

---

### `model/model_trainer.py:796` — same value logged three times

```python
logger.debug(
    f"_run_train_and_evaluate: seed={random_state}, random_state={random_state}, state={random_state}"
)
```

All three format fields are the same variable.

---

### `model/classifier_model.py` — NaiveBayes hyperparameter grid is a no-op

```python
{"var_smoothing": [1.0]}
```

`GridSearchCV` over a single value is equivalent to a fixed hyperparameter. Add multiple values or remove the grid.

---

### `label/label_builder.py` and `utils/date_utils.py` — `# noqa: B904` suppresses exception chaining

Several `except` blocks raise a new `ValueError` without chaining the original:

```python
except ValueError:
    raise ValueError(f"...")  # noqa: B904
```

This discards the original traceback. Use `raise ValueError(...) from e` instead and remove the `noqa` suppression.

---

### `plots/feature_plots.py` and `plots/tremor_plots.py` — `except Exception` swallows all errors in batch workers

`_process_single_file()` and `_process_single_tremor_file()` catch all exceptions and return `"failed"`. While intentional for batch robustness, programming errors (e.g. `AttributeError`, `TypeError`) are silently demoted to log messages, making them hard to diagnose.

---

### `utils/pathutils.py` — `build_model_directories()` creates directories as a side effect

A function named "build…directories" that both computes paths *and* calls `os.makedirs` on all of them couples path resolution with filesystem side effects. Callers that only want the paths cannot opt out of directory creation.

---

## 🟡 Documentation / Style Violations

---

### `tremor/shanon_entropy.py` + `utils/window.py` — consistent misspelling of "Shannon"

`ShanonEntropy`, `shanon_entropy()`, and all import/usage sites spell "Shannon" with one 'n'. This is pervasive and would be a breaking rename to fix, but should be corrected before the next public release.

---

### `sources/sds.py:281-286` — `get_trace()` docstring documents a return value that does not exist

```
Returns:
    Trace: Trace object, or empty Trace if unavailable.
    np.ndarray: Array of trace data, or empty np.ndarray if unavailable.  ← never returned
```

The function signature is `-> Trace | None`. The `np.ndarray` line is a leftover from a previous implementation.

---

### `model/metrics_computer.py` — all docstrings missing blank line after summary sentence

All docstrings in this file run the summary and description together without the required blank line, violating the project's Google style requirement.

---

### `model/model_evaluator.py:188` — `_save_plot()` docstring not Google style

No blank line after the one-sentence summary; summary is not imperative mood.

---

### `utils/pathutils.py` — `build_model_directories()` docstring not Google style

`Args:` section uses bare `name:` format without types. Should follow:

```
Args:
    root_dir (str): Base output directory path.
```

---

### `model/model_evaluator.py:379,418` and `main.py` — incorrect type-ignore comment syntax

```python
# ty:ignore[invalid-argument-type]   ← missing space, wrong prefix
```

The correct syntax for the `ty` type checker is `# ty: ignore[...]` (with a space). Without the space the comment is not recognized and the type error is not suppressed.

---

### `logger.py` — comment/configuration mismatch

Comments in several places say "FILE LOGS EVERYTHING INCLUDING DEBUG" and reference "30 days" retention, but the actual configuration sets `level="INFO"` and `retention="3 days"`. Comments should reflect the actual settings.

---

### `tremor/rsam.py` — example in `Returns:` section of `apply_filter()` docstring

```python
Returns:
    Self: The RSAM instance for method chaining.

Examples:
    >>> rsam = RSAM(stream)
    >>> rsam.apply_filter(0.1, 2.0)
    >>> print(rsam.is_filtered)  # True   ← this belongs in Examples:, not here
```

The example line is inside the `Returns:` block. Move it to the `Examples:` section.

---

## Summary Table

| File | Finding | Severity |
|---|---|---|
| `utils/ml.py:432` | Early exit compares seed value vs. count | 🔴 Bug |
| `tremor/dsar.py:133` | `value_multiplier > 1` skips 0 < x ≤ 1 | 🔴 Bug |
| `tremor/calculate_tremor.py:891` | Same `value_multiplier > 1` issue | 🔴 Bug |
| `model/model_trainer.py:408` | `output_dir` vs `self.output_dir` in `classifier_dir` | 🔴 Bug |
| `model/model_predictor.py:300` | `model_name` shadowed in inner loop | 🔴 Bug |
| `features/tremor_matrix_builder.py:467` | `self.csv` not set on cache hit | 🔴 Bug |
| `label/label_data.py:306,320` | `split(".")` instead of `os.path.splitext` | 🔴 Bug |
| `features/features_builder.py:392-394` | `.index` mutation on slice unsafe in pandas ≥ 3.0 | 🔴 Bug |
| `utils/window.py:197-212` | `value_multiplier` ignored when `remove_outlier_method=None` | 🟠 Logic |
| `model/classifier_model.py` | `set_random_state()` does not invalidate cached `_model` | 🟠 Logic |
| `tremor/calculate_tremor.py` | `shutil.rmtree` before directory exists | 🟠 Logic |
| `model/forecast_model.py:786` | `source="sds"`, `sds_dir=None` silently skips calculation | 🟠 Logic |
| `utils/ml.py:435` | Shape comment `(n_seeds, n_windows)` is transposed | 🟠 Logic |
| `label/label_builder.py` | Eruption labels re-applied on cache hit | 🟠 Logic |
| `dataframe.py:130` + `tremor_matrix_builder.py:483` | Sampling rate from first two rows only | 🟠 Logic |
| `model/forecast_model.py:1120` | `sleep(3)` wastes ~25 min per 500-seed run | 🔵 Quality |
| `model/model_predictor.py:471-479` | Dead `if labels_df is None` check | 🔵 Quality |
| `model/model_predictor.py:993-1016` | Large commented-out block | 🔵 Quality |
| `model/model_trainer.py:796` | Same value logged three times | 🔵 Quality |
| `model/classifier_model.py` | NB grid `{"var_smoothing": [1.0]}` is no-op | 🔵 Quality |
| `label/label_builder.py` + `utils/date_utils.py` | `noqa: B904` suppresses exception chaining | 🔵 Quality |
| `plots/feature_plots.py` + `plots/tremor_plots.py` | `except Exception` hides programming errors | 🔵 Quality |
| `utils/pathutils.py` | `build_model_directories()` creates dirs as side effect | 🔵 Quality |
| `tremor/shanon_entropy.py` + `utils/window.py` | Misspelling: `shanon` → `shannon` | 🟡 Docs |
| `sources/sds.py:281-286` | `get_trace()` documents `np.ndarray` return that doesn't exist | 🟡 Docs |
| `model/metrics_computer.py` | All docstrings missing blank line after summary | 🟡 Docs |
| `model/model_evaluator.py:188` | `_save_plot()` docstring not Google style | 🟡 Docs |
| `utils/pathutils.py` | `build_model_directories()` Args missing types | 🟡 Docs |
| `model/model_evaluator.py:379,418` + `main.py` | `# ty:ignore` missing space — comment not recognized | 🟡 Docs |
| `logger.py` | Comment says DEBUG/30 days; code sets INFO/3 days | 🟡 Docs |
| `tremor/rsam.py` | Example placed inside `Returns:` block | 🟡 Docs |
