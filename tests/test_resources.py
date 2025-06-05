"""Test resources for the Arklex framework.

This module provides test resources for testing the Arklex framework.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, cast

import pytest

from tests.utils.utils_workers import MCWorkerOrchestrator, MsgWorkerOrchestrator
from tests.utils.utils_tools import ShopifyToolOrchestrator

# Test case configuration for different orchestrator types
TEST_CASES: List[Tuple[Type[Any], str, str]] = [
    (
        MCWorkerOrchestrator,
        "mc_worker_taskgraph.json",
        "mc_worker_testcases.json",
    ),
    (
        MsgWorkerOrchestrator,
        "message_worker_taskgraph.json",
        "message_worker_testcases.json",
    ),
    (
        ShopifyToolOrchestrator,
        "shopify_tool_taskgraph.json",
        "shopify_tool_testcases.json",
    ),
]


@pytest.mark.parametrize(
    "orchestrator_cls, config_file_name, test_cases_file_name",
    TEST_CASES,
)
def test_resources(
    orchestrator_cls: Type[Any],
    config_file_name: str,
    test_cases_file_name: str,
) -> None:
    """Run test cases for a specific orchestrator class.

    This function loads test cases from a file and runs them using the specified
    orchestrator class. It handles test failures and provides detailed error
    messages.

    Args:
        orchestrator_cls (Type[Any]): The orchestrator class to test.
        config_file_name (str): Name of the configuration file.
        test_cases_file_name (str): Name of the test cases file.

    Raises:
        pytest.fail: If any test case fails, with a detailed error message
            including the test case number and orchestrator class name.
    """
    data_dir = Path(__file__).parent / "data"
    config_path = data_dir / config_file_name
    test_cases_path = data_dir / test_cases_file_name

    test_resources_instance = orchestrator_cls(str(config_path))

    try:
        with open(test_cases_path, "r") as f:
            test_cases: List[Dict[str, Any]] = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        pytest.fail(
            f"Failed to load test cases from {test_cases_file_name}: {str(e)}",
            pytrace=True,
        )

    for i, test_case in enumerate(test_cases):
        try:
            test_resources_instance.run_single_test(test_case)
        except Exception as e:
            pytest.fail(
                f"Test case {i} failed for {orchestrator_cls.__name__} from {test_cases_file_name}: {str(e)}",
                pytrace=True,
            )
