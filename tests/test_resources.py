"""Test resources for the Arklex framework.

This module provides test resources for testing the Arklex framework.
"""

import logging
import json
import os
from typing import Any, Dict, List, Tuple, Type, Optional

import pytest

from arklex.env.env import Environment
from tests.utils.utils_workers import MCWorkerOrchestrator, MsgWorkerOrchestrator
from tests.utils.utils_tools import ShopifyToolOrchestrator

logger = logging.getLogger(__name__)


class TestResources:
    """Test resources for the Arklex framework.

    This class provides test resources for testing the Arklex framework.
    """

    def __init__(self) -> None:
        """Initialize the test resources."""
        self.env = Environment(tools=[], workers=[], slot_fill_api="")


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
    test_resources_instance = orchestrator_cls(
        os.path.join(os.path.dirname(__file__), "data", config_file_name)
    )
    with open(
        os.path.join(os.path.dirname(__file__), "data", test_cases_file_name), "r"
    ) as f:
        test_cases: List[Dict[str, Any]] = json.load(f)
    for i, test_case in enumerate(test_cases):
        try:
            test_resources_instance.run_single_test(test_case)
        except Exception as _:
            pytest.fail(
                f"Test case {i} failed for {orchestrator_cls.__name__} from {test_cases_file_name}",
                pytrace=True,
            )
