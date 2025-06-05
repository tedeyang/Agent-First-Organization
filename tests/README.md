# Test Suite Documentation

## Overview

This test suite validates the behavior of different orchestrator types in the Arklex framework. It supports both local testing with mocked LLM responses and integration testing with real LLM responses.

## Running Tests

### Local Testing (with Mocked LLM)

To run tests locally with mocked LLM responses:

```bash
export ARKLEX_TEST_ENV=local
python3 -m pytest tests/test_resources.py -v
```

This will use predefined responses for LLM interactions, making tests:

- Faster (no API calls)
- More predictable
- Independent of external services

### Integration Testing (with Real LLM)

To run tests with real LLM responses (default):

```bash
python3 -m pytest tests/test_resources.py -v
```

This will make actual API calls to the LLM service, useful for:

- Validating real-world behavior
- Testing API integration
- Catching service-specific issues

## Test Structure

The test suite is organized as follows:

```
tests/
├── README.md                 # This documentation
├── test_resources.py        # Main test runner
├── data/                    # Test data directory
│   ├── mc_worker_taskgraph.json
│   ├── mc_worker_testcases.json
│   ├── message_worker_taskgraph.json
│   ├── message_worker_testcases.json
│   ├── shopify_tool_taskgraph.json
│   └── shopify_tool_testcases.json
└── utils/                   # Test utilities
    ├── utils.py            # Core test utilities
    ├── utils_workers.py    # Worker-specific test utilities
    └── utils_tools.py      # Tool-specific test utilities
```

## Adding New Tests

### 1. Create Test Configuration

Create a new taskgraph configuration file in `tests/data/`:

```json
{
    "nodes": [
        ["0", {"type": "start", "attribute": {"value": "Hello! How can I help you?"}}],
        ["1", {"type": "message", "attribute": {"value": "What would you like to know?"}}]
    ],
    "edges": [
        ["0", "1"]
    ]
}
```

### 2. Create Test Cases

Create a test cases file in `tests/data/`:

```json
[
    {
        "user_utterance": ["Hello", "What products do you have?"],
        "expected_taskgraph_path": ["0", "1"]
    }
]
```

### 3. Add Test Case to Runner

Add your test case to `TEST_CASES` in `tests/test_resources.py`:

```python
TEST_CASES: List[TestCase] = [
    # ... existing test cases ...
    TestCase(
        YourNewOrchestrator,
        "your_taskgraph.json",
        "your_testcases.json"
    ),
]
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

## Troubleshooting

### Common Issues

1. **Test Case Loading Fails**
   - Check that JSON files are valid
   - Verify file paths in test configuration
   - Ensure test case structure matches expected format

2. **LLM Response Issues**
   - For local testing: Check mock responses in `utils.py`
   - For integration testing: Verify API credentials and connectivity

3. **Path-related Errors**
   - Ensure all paths in test configuration are relative to `tests/data/`
   - Check file permissions in the test directory

### Getting Help

If you encounter issues not covered here:

1. Check the test logs for detailed error messages
2. Review the test case configuration
3. Verify the orchestrator implementation
4. Contact the development team for assistance
