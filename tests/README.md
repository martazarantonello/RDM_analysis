# RDM Analysis - Test Guide

## Quick Start

```bash
pytest              # Run all tests
pytest -v           # Verbose output
pytest -k "delay"   # Run tests matching "delay"
```

---

## Test Files

**82 tests organized in 3 files:**

```
tests/
├── conftest.py                          # Shared test fixtures
├── test_preprocessor_utils.py           # Simple utility functions (37 tests)
├── test_preprocessor_utils_advanced.py  # Complex processing functions (18 tests)
└── test_preprocessor_main.py            # Main workflow functions (27 tests)
```

| File | What it tests | Complexity |
|------|---------------|------------|
| `test_preprocessor_utils.py` | Day codes, schedule parsing, delay extraction | Simple, pure functions |
| `test_preprocessor_utils_advanced.py` | Schedule loading, delay matching, batch operations | Complex, requires mocking |
| `test_preprocessor_main.py` | Weekday extraction, station loading | Workflow functions |

---

## Test Coverage

**Overall: 79% of all functions tested**

| Module | Total Functions | Tested | Coverage |
|--------|----------------|--------|----------|
| `preprocessor/utils.py` | 12 | 12 | 100% ✅ |
| `preprocessor/preprocessor.py` | 6 | 3 | 50% |

### What's NOT Tested

The following 3 functions in `preprocessor/preprocessor.py` are **intentionally not tested**:

1. **`save_processed_data_by_weekday_to_dataframe()`**
2. **`save_stations_by_category()`**  
3. **`save_all_category_a_stations()`**

**Why?** These are high-level orchestration/workflow functions that:
- Primarily coordinate other functions (which ARE tested)
- Perform extensive file I/O operations (saving parquet files)
- Would require complex file system mocking
- Are better validated through integration testing or manual verification

Testing these would add complexity without significantly improving reliability, since the core logic they use is already thoroughly tested.

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

## VS Code Integration

1. Click **Testing** icon (beaker) in sidebar
2. Tests should auto-discover
3. **If discovery fails:** `Ctrl+Shift+P` → "Reload Window"
4. **Alternative:** Always use terminal commands