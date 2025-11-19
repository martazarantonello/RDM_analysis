# RDM Analysis - Test Guide

## Quick Start

```bash
pytest              # Run all tests
pytest -v           # Verbose output
pytest -k "delay"   # Run tests matching "delay"
```

---

## Test Files

**Tests organized by module:**

```
tests/
├── conftest.py                    # Shared test fixtures
├── test_preprocessor_utils.py     # Utility function tests (55 tests)
├── test_preprocessor.py           # Main preprocessing tests (34 tests)
├── baseline_comparison.py         # Baseline comparison utility
└── analyze_save_func.ipynb        # Optimization analysis notebook
```

| File | What it tests | Test Count |
|------|---------------|------------|
| `test_preprocessor_utils.py` | Day codes, schedule parsing, delay extraction, data loading, delay matching, schedule processing | 55 tests |
| `test_preprocessor.py` | Weekday extraction, station loading, save functions, optimization validation | 34 tests |
| `baseline_comparison.py` | Compare outputs before/after optimization | **Integration testing** ⭐ |
| `analyze_save_func.ipynb` | Optimization opportunities analysis | **Documentation** |

---

## Test Coverage

**Overall: 100% of core functions tested - 89 tests total**

| Module | Total Functions | Tested | Coverage |
|--------|----------------|--------|----------|
| `preprocess/utils.py` | 12 | 12 | 100% ✅ |
| `preprocess/preprocessor.py` | 6 | 6 | 100% ✅ |

### Test Organization

**`test_preprocessor_utils.py`** - Comprehensive utility testing:
- **Basic utilities** (37 tests): Day code mapping, schedule parsing, delay extraction
- **Advanced functions** (18 tests): Schedule loading, delay matching, batch operations

**`test_preprocessor.py`** - Main workflow testing:
- **Core functions** (27 tests): Weekday extraction, station loading, data processing workflow
- **Save functions** (7 tests): Optimization validation, deduplication, file I/O
  - `save_processed_data_by_weekday_to_dataframe()` validation
  - `save_stations_by_category()` validation
  - Baseline comparison placeholders for optimization testing

---

## Optimization Testing Workflow

### ⚠️ IMPORTANT: Run BEFORE implementing optimizations!

The save functions take ~24 hours to run. Before optimizing them, you must:

### Step 1: Run Unit Tests (Current Implementation)
```bash
# Run the new save function tests
pytest tests/test_save_functions.py -v

# Should pass - establishes that current implementation is correct
```

### Step 2: Create Baseline Output (on small sample)
```bash
# Save baseline output for a few stations (e.g., 5 stations)
python tests/baseline_comparison.py save --stations 12345,67890,11111,22222,33333

# This saves the output to tests/baselines/original/
# Processing time per station will be recorded
```

### Step 3: Implement Optimizations
- Follow recommendations in `analyze_save_func.ipynb`
- Implement parallel processing, pandas optimizations, etc.

### Step 4: Run Unit Tests (Optimized Implementation)
```bash
# Run tests again to ensure basic behavior is unchanged
pytest tests/test_save_functions.py -v

# Should still pass
```

### Step 5: Create Optimized Baseline Output
```bash
# Save baseline with optimized code
python tests/baseline_comparison.py save --stations 12345,67890,11111,22222,33333 --optimized

# This saves to tests/baselines/optimized/
```

### Step 6: Compare Baselines
```bash
# Compare original vs optimized outputs
python tests/baseline_comparison.py compare --stations 12345,67890,11111,22222,33333

# This will:
# - Compare data outputs (should be identical)
# - Compare processing times (should be faster)
# - Report speedup achieved
```

### Expected Output:
```
Comparing baselines...
============================================================

Comparing station: 12345
  Processing time:
    Original:  245.32s
    Optimized: 41.18s
    Speedup:   5.96x
  ✓ MO: Data matches
  ✓ TU: Data matches
  ✓ WE: Data matches
  ...

Summary:
  ✓ All data matches between original and optimized!
  
  Average speedup: 5.96x
```

---

## Baseline Comparison Utility

**Purpose**: Ensure optimizations don't change the output

### Commands:

```bash
# Save baseline (before optimization)
python tests/baseline_comparison.py save --stations 12345,67890

# Save optimized baseline (after optimization)
python tests/baseline_comparison.py save --stations 12345,67890 --optimized

# Compare outputs and measure speedup
python tests/baseline_comparison.py compare --stations 12345,67890
```

### What it does:
1. Processes specified stations
2. Saves output as both `.parquet` and `.csv`
3. Records processing time and metadata
4. Compares data byte-by-byte
5. Reports speedup achieved

### Output structure:
```
tests/baselines/
├── original/
│   ├── 12345/
│   │   ├── MO.parquet
│   │   ├── MO.csv
│   │   ├── TU.parquet
│   │   ├── metadata.json
│   │   └── ...
│   └── summary.json
├── optimized/
│   └── (same structure)
└── comparison_results.json
```

---

## Optimization Analysis

**See: `tests/analyze_save_func.ipynb`**

This notebook contains detailed analysis of optimization opportunities:
- Current bottlenecks identified
- Expected speedup calculations
- Implementation roadmap
- Code examples for each optimization

**Key findings:**
- 🔴 **Critical**: Parallel processing → 6-8x speedup
- 🟡 **High**: Pandas-based operations → 3-5x additional speedup
- **Overall**: Expected reduction from 24 hours → 1-2 hours

---

## Test Fixtures

Available in all tests (defined in `conftest.py`):

- `sample_schedule_entry` - Mock train schedule with complete structure
- `sample_delay_entry` - Mock delay record
- `sample_stanox_ref` - Mock station reference data
- `sample_tiploc_to_stanox` - TIPLOC→STANOX mapping
- `tmp_path` - Temporary directory (pytest built-in)
- `mock_incident_files` - Mock CSV files

---

## Common Commands

```bash
# Run all tests
pytest

# Run specific file
pytest tests/test_preprocessor_utils.py -v

# Run specific test class
pytest tests/test_preprocessor_utils.py::TestDayCodeMapping -v

# Stop at first failure
pytest -x

# Run last failed tests only
pytest --lf

# Show detailed output
pytest -v --tb=short
```

---

## Configuration

**`.vscode/settings.json`**
- Enables pytest in VS Code
- Configures test discovery

**`pytest.ini`** (project root)
- Test discovery patterns
- Python path settings

---