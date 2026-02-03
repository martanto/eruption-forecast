# Tests Directory

This directory contains test scripts for the eruption-forecast package.

## Test Files

### `test_tremor_calculation.py`
Comprehensive test for tremor calculation module with real SDS data.

**Features tested:**
- RSAM calculation (Real Seismic Amplitude Measurement)
- DSAR calculation (Displacement Seismic Amplitude Ratio)
- SDS data loading
- Daily tremor plotting
- Output validation

**Usage:**
```bash
# Run from project root
uv run python tests/test_tremor_calculation.py

# Or with standard Python
python tests/test_tremor_calculation.py
```

**Test Configuration:**
- Data Source: D:\Data\OJN (SDS format)
- Station: VG.OJN.00.EHZ
- Date Range: 2025-01-01 to 2025-01-03 (3 days)
- Output: output_test/ directory

**Expected Results:**
- 432 time windows (144 per day × 3 days)
- 9 tremor metrics (5 RSAM + 4 DSAR)
- 1 combined plot + 3 daily plots
- ~78 KB CSV output file

### `verify_dsar.py`
Legacy DSAR calculation verification script.

## Adding New Tests

When adding new test files:

1. Follow naming convention: `test_*.py`
2. Add comprehensive docstrings
3. Include validation steps
4. Update this README
5. Run tests before committing

## Test Output

All tests use the `output_test/` directory to avoid interfering with production data.

Directory structure:
```
output_test/
└── {NSLC}/
    ├── tremor/
    │   ├── tmp/          # Daily CSV files
    │   └── *.csv         # Merged tremor data
    └── figures/
        ├── *.png         # Combined plots
        └── daily/        # Daily plots
            └── *.png
```

## CI/CD Integration

To integrate tests into CI/CD pipelines:

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest tests/

# Or run specific test
uv run python tests/test_tremor_calculation.py
```

## Notes

- Tests require real SDS data in D:\Data\OJN
- Tests are designed to be run on Windows (see path handling)
- Test output is gitignored to keep repository clean
