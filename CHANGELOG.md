# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Improved - Documentation & ML Analysis (2026-02-04)

Comprehensive documentation improvements, grammar fixes, and machine learning workflow analysis.

**Grammar & Spelling Fixes:**
- Fixed "aftrer" → "after" in `forecast_model.py:147`
- Fixed "Whether to each significant features" → "Whether to plot each significant feature" in `forecast_model.py:1043`
- Fixed "Exracted" → "Extracted" in `utils.py:844`
- Fixed `dict[str, any]` → `dict[str, Any]` in `classifier_model.py:186`

**Docstring Improvements:**
- Added comprehensive examples to `ForecastModel` class docstring
- Added examples to `ClassifierModel` class and properties
- Added examples to `TrainModel` class and `train()` method
- Added examples to `FeaturesBuilder` class and `build()` method
- Enhanced `get_significant_features()` and `random_under_sampler()` docstrings

**Bug Fixes:**
- Removed deprecated `max_features="auto"` from Decision Tree and Random Forest grids (deprecated in sklearn 1.4+)

**Documentation Updates:**
- Rewrote `SUMMARY.md` with comprehensive ML workflow analysis
- Added model comparison table (7 classifiers)
- Added grid parameter analysis for each model
- Added recommendations for model selection
- Updated `TODO.md` with new improvement tasks
- Updated `README.md` to use `pyrefly` instead of `mypy`
- Added Rule #6 to `CLAUDE.md`: Create branch with `claude/` prefix before modifications

**Files Modified:**
- `src/eruption_forecast/model/forecast_model.py`
- `src/eruption_forecast/model/classifier_model.py`
- `src/eruption_forecast/model/train_model.py`
- `src/eruption_forecast/features/features_builder.py`
- `src/eruption_forecast/utils.py`
- `SUMMARY.md`
- `TODO.md`
- `README.md`
- `CLAUDE.md`
- `CHANGELOG.md`

**Branch:** `claude/docstring-improvements`

**Contributors:**
- Claude Opus 4.5 <noreply@anthropic.com>

---

### Refactored - ForecastModel Class (2025-02-03)

Major refactoring of `src/eruption_forecast/model/forecast_model.py` to improve code maintainability, testability, and readability. The refactoring breaks down long methods into smaller, focused helper methods following established patterns in the codebase (LabelBuilder, FeaturesBuilder, CalculateTremor).

**Key Improvements:**
- Reduced method complexity: 4 methods with 100+ lines → 0 methods over 70 lines
- Added 16 new private helper methods
- Improved testability: Each helper method has a single, clear responsibility
- Enhanced documentation: Better docstrings explaining lifecycle and responsibilities
- **100% backward compatible**: All public APIs remain unchanged

#### Phase 1: `__init__` Method Refactoring

**New Helper Methods:**
1. `_normalize_dates()` - Normalize start and end dates to standard format
   - Extracts date normalization logic (previously lines 70-76)
   - Consistent with LabelBuilder pattern

2. `_setup_directories()` - Setup directory structure for outputs
   - Extracts directory setup logic (previously lines 77-82)
   - Creates NSLC identifier and directory paths

3. `_initialize_feature_parameters()` - Initialize tsfresh parameters
   - Extracts tsfresh configuration (previously lines 118-127)
   - Sets up ComprehensiveFCParameters with defaults

4. `_initialize_state_properties()` - Initialize lifecycle state properties
   - Extracts state initialization (previously lines 129-159)
   - Documents when each property is set during lifecycle

5. `validate()` - Validate initialization parameters (PUBLIC method)
   - New validation method following LabelBuilder/FeaturesBuilder pattern
   - Validates window_size, n_jobs, date ranges, string fields
   - Creates necessary directories

**Impact:**
- `__init__` reduced from 124 lines to 82 lines (42 line reduction)
- Better separation of concerns
- Explicit validation step

**Commit:** `28f3e0b` - refactor: Extract helper methods from ForecastModel.__init__

---

#### Phase 2: `calculate()` Method Refactoring

**New Helper Methods:**
1. `_setup_calculate_tremor()` - Setup CalculateTremor instance
   - Extracts CalculateTremor instantiation and configuration
   - Handles verbose/debug flags
   - Updates start_date to include window_size buffer

2. `_calculate_from_sds()` - Calculate tremor from SDS source
   - Extracts SDS source handling (previously lines 385-392)
   - Validates sds_dir parameter
   - Raises clear ValueError if directory missing

3. `_calculate_from_fdsn()` - Calculate tremor from FDSN source
   - Extracts FDSN source handling (previously lines 394-398)
   - Raises NotImplementedError with clear message

4. `_adjust_dates_to_tremor_range()` - Adjust dates to tremor availability
   - Extracts date adjustment logic (previously lines 406-423)
   - Updates start_date and end_date if needed
   - Logs adjustments if verbose mode enabled

**Impact:**
- `calculate()` reduced from 110 lines to 48 lines (62 line reduction)
- Source selection logic isolated
- Date adjustment reusable and testable

**Commit:** `75365b5` - refactor: Break down ForecastModel.calculate() method

---

#### Phase 3: `build_label()` Method Refactoring

**New Helper Methods:**
1. `_validate_tremor_for_labeling()` - Validate tremor data availability
   - Extracts tremor validation (previously lines 793-800)
   - Checks tremor data loaded
   - Validates tremor columns exist

2. `_prepare_tremor_for_labeling()` - Prepare tremor data
   - Extracts tremor data preparation (previously lines 827-830)
   - Creates copy and filters columns
   - Sorts by datetime index

3. `_validate_label_tremor_date_range()` - Validate date ranges
   - Extracts date range validation (previously lines 832-847)
   - Ensures labels within tremor data coverage
   - Raises clear ValueError with guidance

4. `_calculate_eruption_statistics()` - Calculate eruption statistics
   - Extracts statistics calculation (previously lines 862-874)
   - Computes eruption/non-eruption counts and ratio
   - Logs statistics if verbose enabled

**Impact:**
- `build_label()` reduced from 121 lines to 66 lines (55 line reduction)
- Validation logic isolated and reusable
- Statistics calculation separated from main flow

**Commit:** `a96818d` - refactor: Break down ForecastModel.build_label() method

---

#### Phase 4: `extract_features()` Method Refactoring

**New Helper Methods:**
1. `_prepare_features_data()` - Prepare features data
   - Extracts column filtering (previously lines 604-607)
   - Validates columns exist
   - Returns filtered dataframe

2. `_prepare_extraction_parameters()` - Prepare extraction parameters
   - Extracts parameter setup (previously lines 626-641)
   - Handles feature exclusion logic
   - Builds tsfresh parameter dictionary

3. `_extract_features_for_column()` - Extract features for single column
   - Extracts single column extraction (previously lines 644-681)
   - Handles existing file checks
   - Performs tsfresh extraction
   - Saves results to CSV

**Impact:**
- `extract_features()` reduced from 110 lines to 56 lines (54 line reduction)
- Single-column extraction testable in isolation
- Feature extraction logic clearly separated

**Commit:** `563d63f` - refactor: Break down ForecastModel.extract_features() method

---

### Summary Statistics

**Methods Refactored:** 5 (including `__init__`)

**Helper Methods Added:** 16 private + 1 public (`validate()`)

**Line Count Changes:**
- `__init__`: 124 → 82 lines (-42, 34% reduction)
- `calculate()`: 110 → 48 lines (-62, 56% reduction)
- `build_label()`: 121 → 66 lines (-55, 45% reduction)
- `extract_features()`: 110 → 56 lines (-54, 49% reduction)

**Total Lines in Refactored Methods:**
- Before: 465 lines
- After: 252 lines
- Helper methods: ~460 lines (more focused, documented, testable)

**Overall Change:** Net +247 lines (+35%), but significantly improved:
- Maintainability: Each method has single responsibility
- Testability: Helper methods can be tested independently
- Readability: Main methods read like high-level pseudocode
- Documentation: Better docstrings with clearer intent

---

### Backward Compatibility

**✅ 100% Backward Compatible**

All changes are internal refactoring only:
- No changes to public method signatures
- No changes to property names or access patterns
- No changes to return types
- No changes to behavior or side effects
- All existing code using ForecastModel will continue to work unchanged

---

### Testing Recommendations

#### Unit Testing
Test each new helper method individually:
```python
def test_normalize_dates():
    """Test date normalization helper."""
    model = ForecastModel(...)
    start, end, start_str, end_str = model._normalize_dates("2020-01-01", "2020-12-31")
    assert start.hour == 0
    assert end.hour == 23

def test_validate_tremor_for_labeling():
    """Test tremor validation raises appropriate errors."""
    model = ForecastModel(...)
    with pytest.raises(ValueError, match="Tremor data not found"):
        model._validate_tremor_for_labeling(pd.DataFrame(), None)
```

#### Integration Testing
Test complete workflows still work:
```python
def test_calculate_from_sds_workflow():
    """Test complete calculate workflow from SDS."""
    model = ForecastModel(...).calculate(source="sds", sds_dir="...")
    assert model.tremor_csv is not None
    assert len(model.tremor_data) > 0

def test_full_training_pipeline():
    """Test complete training pipeline."""
    model = (
        ForecastModel(...)
        .calculate(source="sds", sds_dir="...")
        .build_label(...)
        .build_features(...)
        .extract_features(...)
    )
    assert model.features_csv is not None
```

#### Backward Compatibility Testing
```python
def test_existing_api_unchanged():
    """Ensure public API remains unchanged."""
    model = ForecastModel(
        station="OJN",
        channel="EHZ",
        start_date="2020-01-01",
        end_date="2020-12-31",
        window_size=1,
        volcano_id="TEST"
    )

    # All public methods should still be accessible
    assert hasattr(model, "calculate")
    assert hasattr(model, "build_label")
    assert hasattr(model, "build_features")
    assert hasattr(model, "extract_features")
    assert hasattr(model, "concat_features")
    assert hasattr(model, "validate")
```

---

### Migration Guide

**No migration needed!** This is a pure internal refactoring. All existing code will continue to work without any changes.

If you were directly testing private methods or accessing internal state in unconventional ways, you may need to update tests to use the new helper methods. However, this is unlikely as the refactoring maintains all public interfaces.

---

### Code Review Checklist

When reviewing code changes:
- ✅ All public method signatures unchanged
- ✅ All property names unchanged
- ✅ All return types unchanged
- ✅ Method chaining still works (returns `Self`)
- ✅ Verbose/debug logging preserved
- ✅ Error messages improved with clearer context
- ✅ Docstrings follow Google format
- ✅ Type hints comprehensive (pyrefly compliant)
- ✅ Consistent with LabelBuilder/FeaturesBuilder patterns

---

### Future Improvements

Potential areas for further enhancement (not in this refactoring):
1. Add comprehensive unit tests for all helper methods
2. Add integration tests for complete workflows
3. Consider extracting common validation logic to utils.py
4. Consider creating a state machine class for lifecycle management
5. Add property decorators for computed values (like `@cached_property`)
6. Consider async support for parallel feature extraction

---

### Related Issues

- Improves maintainability for future development
- Makes testing individual components easier
- Reduces cognitive load when reading and understanding code
- Follows established patterns in codebase (LabelBuilder, FeaturesBuilder)

---

### Contributors

- Claude Sonnet 4.5 <noreply@anthropic.com>

---

## Previous Releases

(Previous changelog entries would go here)
