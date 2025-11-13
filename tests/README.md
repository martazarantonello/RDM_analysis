# RDM Analysis - Test Suite

Complete testing guide for the preprocessor module. **82 tests total - all passing ✅**

---

## 🚀 Quick Start

```bash
# Run all tests
pytest

# Run with details
pytest -v

# Run specific file
pytest tests/test_preprocessor_utils.py -v
```

---

## 📁 Test Structure

```
tests/
├── conftest.py                          # Shared test fixtures
├── test_preprocessor_utils.py           # Simple utility functions (37 tests)
├── test_preprocessor_utils_advanced.py  # Complex processing functions (18 tests)
└── test_preprocessor_main.py            # Main preprocessor functions (27 tests)
```

### File Organization Explained

**`test_preprocessor_utils.py`** - Simple, Pure Functions (37 tests)
- **What:** Small helper functions with simple inputs/outputs
- **Why separate:** Easy to test, no mocking needed
- **Tests:**
  - `TestDayCodeMapping` - Convert day indices to codes (MO, TU, etc.)
  - `TestScheduleDaysExtraction` - Parse binary schedule strings
  - `TestDelayDayExtraction` - Extract day from datetime strings
  - `TestScheduleRunsOnDay` - Check if schedule runs on a day
  - `TestProcessDelays` - Basic delay filtering (uses temp files)

**`test_preprocessor_utils_advanced.py`** - Complex Functions (18 tests)
- **What:** Large functions that load/process data, require extensive mocking
- **Why separate:** Complex setup, heavy use of mocks, test data pipelines
- **Tests:**
  - `TestLoadScheduleData` - Load schedule from pickle files
  - `TestProcessSchedule` - Full schedule processing pipeline
  - `TestAdjustScheduleTimeline` - Match delays with schedules
  - `TestLoadScheduleDataOnce` - Optimized batch loading
  - `TestLoadIncidentDataOnce` - Pre-load all incident data
  - `TestProcessDelaysOptimized` - Optimized delay processing

**`test_preprocessor_main.py`** - Main Entry Points (27 tests)
- **What:** Functions from `preprocessor.py` (not utils)
- **Why separate:** Different module, different concerns
- **Tests:**
  - `TestGetWeekdayFromScheduleEntry` - Weekday extraction logic
  - `TestLoadStations` - Load stations by DFT category
  - `TestSaveProcessedDataByWeekday` - Data organization workflow

### Quick Comparison

| Aspect | test_preprocessor_utils.py | test_preprocessor_utils_advanced.py |
|--------|---------------------------|-------------------------------------|
| **Complexity** | Simple, pure functions | Complex, multi-step pipelines |
| **Mocking** | Minimal or none | Heavy use of @patch decorators |
| **Data** | Simple strings/dicts | Large DataFrames, file I/O |
| **Function size** | 5-20 lines | 50-200+ lines |
| **Examples** | `get_day_code_mapping()` | `process_schedule()` |
| | `extract_day_of_week()` | `adjust_schedule_timeline()` |
| | `schedule_runs_on_day()` | `load_schedule_data_once()` |

### Why Split Into Two Files?

1. **Organization** - Easier to find tests for specific functions
2. **Clarity** - Simple vs complex functions clearly separated
3. **Maintainability** - Advanced tests won't clutter basic test file
4. **Speed** - Can run just simple tests quickly: `pytest tests/test_preprocessor_utils.py`

---

## 📊 Coverage Summary

**82 tests covering 79% of all functions**

| Module | Functions | Tested | Coverage |
|--------|-----------|--------|----------|
| `preprocessor/utils.py` | 13 | 12 | 92% ✅ |
| `preprocessor/preprocessor.py` | 6 | 3 | 50% |

### What's Tested

**Basic utilities (37 tests):**
- Day code mapping (MO, TU, WE, etc.)
- Schedule day extraction
- Delay date/time handling
- Schedule matching by day

**Advanced functions (18 tests):**
- Schedule data loading
- Schedule processing pipeline
- Timeline adjustment with delays
- Optimized batch operations

**Main functions (27 tests):**
- Station loading by category
- Data organization by weekday
- Multi-day schedule expansion

---

## 🎯 Common Commands

### Run Tests

```bash
pytest                          # All tests
pytest -v                       # Verbose output
pytest -k "delay"               # Tests matching "delay"
pytest tests/test_preprocessor_utils.py::TestDayCodeMapping  # Specific class
pytest -x                       # Stop at first failure
pytest --lf                     # Run last failed tests
```

### VS Code Integration

1. Click **Testing** icon (beaker) in sidebar
2. **If discovery fails:** Reload window (`Ctrl+Shift+P` → "Reload Window")
3. Click refresh button in Testing panel
4. **Alternative:** Use terminal (always works!)

---

## 🧪 Test Fixtures (in conftest.py)

Shared test data available to all tests:

- `sample_schedule_entry` - Mock train schedule
- `sample_delay_entry` - Mock delay record
- `sample_stanox_ref` - Mock station reference
- `sample_tiploc_to_stanox` - TIPLOC→STANOX mapping
- `tmp_path` - Temporary directory (pytest built-in)
- `mock_incident_files` - Mock CSV files

**Usage:**
```python
def test_example(sample_schedule_entry):
    result = process_schedule(sample_schedule_entry)
    assert len(result) > 0
```

---

## ✍️ Writing New Tests

### Basic Structure

```python
class TestMyFunction:
    """Test cases for my_function."""
    
    def test_basic_case(self):
        """Test basic functionality."""
        result = my_function(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test with None input."""
        result = my_function(None)
        assert result is None
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double(input, expected):
    assert double(input) == expected
```

### Using Mocks

```python
from unittest.mock import patch

@patch('pandas.read_csv')
def test_with_mock(mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame([...])
    result = my_function()
    assert result is not None
```

---

## 🔍 Configuration Files

**pytest.ini** (project root)
- Sets test discovery patterns
- Configures output format
- Defines pythonpath

**.vscode/settings.json**
- VS Code test integration
- Pytest arguments
- Test discovery settings

---

## 📈 Viewing Results

```bash
# Short summary
pytest -q

# Detailed output
pytest -v

# With traceback on failures
pytest -v --tb=short

# Show all outcomes summary
pytest -ra
```