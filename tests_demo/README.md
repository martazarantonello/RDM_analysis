# RDM Analysis - Demo Tests

Comprehensive test suite for functions used in the `demos/` notebooks. **One test file per demo notebook** for perfect clarity and continuity.

---

## Test Structure

```
demos/                           tests_demo/
├── aggregate_view.ipynb    →   ├── test_aggregate_view.py
├── incident_view.ipynb     →   ├── test_incident_view.py
├── station_view.ipynb      →   ├── test_station_view.py
├── time_view.ipynb         →   ├── test_time_view.py
└── train_view.ipynb        →   └── test_train_view.py
```

---

##To run the tests

```bash
# Run all tests
pytest tests_demo/ -v

# Run tests for specific notebook
pytest tests_demo/test_station_view.py -v

# Run with coverage
pytest tests_demo/ --cov=outputs.utils --cov-report=html

# Run specific test class
pytest tests_demo/test_station_view.py::TestPlotVariableRelationshipsNormalized -v
```
