# Integration Tests

This directory contains comprehensive integration tests for the Arklex AI platform, covering HITL (Human-in-the-Loop) functionality, Shopify tools, HubSpot tools, and other external service integrations.

## Test Structure

### Shared Configuration (`conftest.py`)

The `conftest.py` file provides shared fixtures and configuration for all integration tests:

- **Environment Setup**: Common environment variables and test configuration
- **Mock Fixtures**: Pre-configured mock objects for external services
- **Test Data**: Sample data for various test scenarios
- **Utility Fixtures**: Common test utilities and helper functions

### Test Files

- `test_hitl_server.py` - HITL (Human-in-the-Loop) server integration tests
- `test_shopify_tools.py` - Shopify tools integration tests
- `test_hubspot_tools.py` - HubSpot tools integration tests
- `run_hitl_tests.py` - Test runner script for HITL tests

### Shared Utilities (`test_utils.py`)

The `test_utils.py` file provides common utilities and helper classes:

- `TestDataProvider` - Sample test data for various scenarios
- `MockFactory` - Factory for creating common mock objects
- `AssertionHelper` - Common assertion utilities
- `TestEnvironmentHelper` - Environment management utilities

## Available Fixtures

### Environment Fixtures

- `mock_environment_variables` - Sets up test environment variables
- `clean_test_state` - Ensures clean state between tests
- `test_data_dir` - Provides test data directory path
- `hitl_taskgraph_path` - Path to HITL taskgraph configuration
- `load_hitl_config` - Loads HITL configuration with test settings

### Shopify Fixtures

- `mock_shopify_session` - Mock Shopify session for testing
- `mock_shopify_graphql` - Mock Shopify GraphQL client
- `sample_shopify_product_data` - Sample Shopify product data

### HubSpot Fixtures

- `mock_hubspot_client` - Mock HubSpot client with contact search
- `mock_hubspot_ticket_client` - Mock HubSpot client for ticket creation
- `mock_hubspot_meeting_client` - Mock HubSpot client for meeting creation

### HITL Fixtures

- `mock_message_state` - Mock MessageState for HITL testing
- `mock_llm_response` - Mock LLM response
- `mock_embeddings_response` - Mock embeddings response
- `sample_conversation_history` - Sample conversation history
- `sample_user_parameters` - Sample user parameters

## Running Tests

### Using pytest directly

```bash
# Run all integration tests
pytest tests/integration/

# Run specific test file
pytest tests/integration/test_hitl_server.py

# Run tests with specific markers
pytest tests/integration/ -m "hitl"
pytest tests/integration/ -m "shopify"
pytest tests/integration/ -m "hubspot"

# Run tests in verbose mode
pytest tests/integration/ -v
```

### Using the test runner

```bash
# Run HITL tests
python tests/integration/run_hitl_tests.py

# Run with verbose output
python tests/integration/run_hitl_tests.py -v

# Run specific test file
python tests/integration/run_hitl_tests.py --test-file test_hitl_server.py

# Run tests with specific markers
python tests/integration/run_hitl_tests.py -m "hitl and not slow"

# Check environment only
python tests/integration/run_hitl_tests.py --check-only

# List available tests
python tests/integration/run_hitl_tests.py --list-tests
```

## Test Markers

The following pytest markers are available:

- `@pytest.mark.integration` - Marks tests as integration tests
- `@pytest.mark.hitl` - Marks tests as HITL tests
- `@pytest.mark.shopify` - Marks tests as Shopify integration tests
- `@pytest.mark.hubspot` - Marks tests as HubSpot integration tests
- `@pytest.mark.slow` - Marks tests as slow-running tests

## Writing New Tests

### Using Shared Fixtures

```python
import pytest
from tests.integration.test_utils import TestDataProvider, MockFactory

class TestMyIntegration:
    def test_my_function(
        self,
        mock_shopify_session,  # Use shared fixture
        sample_shopify_product_data,  # Use shared test data
    ):
        # Your test logic here
        pass
```

### Using Shared Utilities

```python
from tests.integration.test_utils import (
    TestDataProvider,
    MockFactory,
    AssertionHelper,
    create_mock_message_state,
)

def test_with_utilities():
    # Use shared test data
    product_data = TestDataProvider.get_sample_shopify_product()
    
    # Use shared mock factory
    mock_response = MockFactory.create_mock_llm_response("Test response")
    
    # Use shared assertion helper
    AssertionHelper.assert_json_response_structure(response, ["key1", "key2"])
```

### Creating Custom Fixtures

```python
@pytest.fixture
def my_custom_fixture(mock_shopify_session):
    """Custom fixture that builds on shared fixtures."""
    # Your custom setup logic
    return custom_data
```

## Environment Requirements

### Required Environment Variables

The following environment variables are automatically set for testing:

- `OPENAI_API_KEY` - Test API key for OpenAI
- `MYSQL_USERNAME` - Test MySQL username
- `MYSQL_PASSWORD` - Test MySQL password
- `MYSQL_HOSTNAME` - Test MySQL hostname
- `MYSQL_PORT` - Test MySQL port
- `MYSQL_DB_NAME` - Test MySQL database name
- `DATA_DIR` - Test data directory
- `ARKLEX_TEST_ENV` - Test environment identifier
- `TESTING` - Testing mode flag
- `LOG_LEVEL` - Log level for tests

### Required Dependencies

- `pytest` - Testing framework
- `openai` - OpenAI API client
- `langchain` - LangChain framework
- `langchain_community` - LangChain community components
- `langchain_openai` - LangChain OpenAI integration

## Best Practices

1. **Use Shared Fixtures**: Always use the shared fixtures from `conftest.py` instead of creating your own
2. **Leverage Test Utilities**: Use the utilities in `test_utils.py` for common patterns
3. **Mock External Services**: Always mock external API calls to avoid dependencies
4. **Use Descriptive Test Names**: Test names should clearly describe what is being tested
5. **Group Related Tests**: Use test classes to group related functionality
6. **Use Appropriate Markers**: Mark tests with appropriate pytest markers for filtering

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the project root is in the Python path
2. **Environment Variables**: Check that all required environment variables are set
3. **Mock Issues**: Verify that external services are properly mocked
4. **Test Data**: Ensure test data files exist in the expected locations

### Debug Mode

Run tests with verbose output and full tracebacks:

```bash
pytest tests/integration/ -v -s --tb=long
```

### Isolated Test Runs

Run a single test in isolation:

```bash
pytest tests/integration/test_hitl_server.py::TestHITLServerIntegration::test_hitl_chat_flag_activation -v -s
```
