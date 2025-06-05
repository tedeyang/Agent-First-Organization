# Test Suite Documentation

## Running Tests

To run tests locally with mocked LLM responses:

```bash
export ARKLEX_TEST_ENV=local
python3 -m pytest tests/test_resources.py -v
```

To run tests with real LLM responses (default):

```bash
python3 -m pytest tests/test_resources.py -v
```

## Known Warnings

The test suite may show several warnings that are safe to ignore:

### NumPy Deprecation Warning

```
DeprecationWarning: numpy.core._multiarray_umath is deprecated and has been renamed to numpy._core._multiarray_umath
```

This warning comes from the FAISS library's internal usage of NumPy. It's a deprecation warning about NumPy's internal API usage and doesn't affect functionality. This will be fixed in future versions of FAISS.

### SWIG Type Warnings

```
DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute
DeprecationWarning: builtin type SwigPyObject has no __module__ attribute
DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

These warnings are related to SWIG (Simplified Wrapper and Interface Generator) types used by FAISS. They don't affect functionality and are internal implementation details of the library.

## Test Structure

The test suite is organized as follows:

- `tests/test_resources.py`: Main test file that runs test cases for different orchestrator types
- `tests/data/`: Directory containing test configuration and test case files
- `tests/utils/`: Directory containing test utilities and mock implementations

## Adding New Tests

To add new tests:

1. Create a new orchestrator class in `tests/utils/` if needed
2. Add test configuration files in `tests/data/`
3. Add test case files in `tests/data/`
4. Add the new test case to `TEST_CASES` in `tests/test_resources.py`
