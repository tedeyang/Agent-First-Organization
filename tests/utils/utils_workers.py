"""Worker test utilities for the Arklex framework.

This module provides test utilities for validating worker behavior in the Arklex
framework. It includes test orchestrators for multiple choice workers and message
workers, with validation methods to ensure correct task graph paths and response
content.
"""

import logging
from typing import Any, Dict, List

from tests.utils.utils import MockOrchestrator, MockResourceInitializer

logger = logging.getLogger(__name__)


class MCWorkerOrchestrator(MockOrchestrator):
    def __init__(self, config_file_path: str) -> None:
        """Initialize the multiple choice worker orchestrator.

        Args:
            config_file_path (str): Path to the configuration file.
        """
        super().__init__(config_file_path)
        self.resource_initializer = MockResourceInitializer()

    def _validate_result(
        self,
        test_case: Dict[str, Any],
        history: List[Dict[str, str]],
        params: Dict[str, Any],
    ) -> None:
        """Validate the test results for multiple choice workers.

        This function validates that the task graph path and response content
        match the expected values from the test case.

        Args:
            test_case (Dict[str, Any]): Test case containing expected values.
            history (List[Dict[str, str]]): Conversation history.
            params (Dict[str, Any]): Parameters containing task graph information.

        Raises:
            AssertionError: If the task graph path or response content does not
                match the expected values.
        """
        # Check taskgraph path
        node_path: List[str] = [
            i["node_id"] for i in params.get("taskgraph", {}).get("path", {})
        ]
        # Ignore initial '0' node if present
        if node_path and node_path[0] == "0":
            node_path = node_path[1:]
        print(f"DEBUG: node_path = {node_path}")
        print(
            f"DEBUG: expected_taskgraph_path = {test_case['expected_taskgraph_path']}"
        )
        assert node_path == test_case["expected_taskgraph_path"]

        # Get assistant responses from history
        assistant_records: List[Dict[str, str]] = [
            message for message in history if message["role"] == "assistant"
        ]
        expected_records: List[Dict[str, str]] = [
            message
            for message in test_case["expected_conversation"]
            if message["role"] == "assistant"
        ]

        # Check that each assistant response matches the expected response
        for i, (actual, expected) in enumerate(
            zip(assistant_records, expected_records)
        ):
            assert actual["content"] == expected["content"], (
                f"Response {i} mismatch:\n"
                f"Expected: {expected['content']}\n"
                f"Actual: {actual['content']}"
            )


class MsgWorkerOrchestrator(MockOrchestrator):
    def __init__(self, config_file_path: str) -> None:
        """Initialize the message worker orchestrator.

        Args:
            config_file_path (str): Path to the configuration file.
        """
        super().__init__(config_file_path)
        self.resource_initializer = MockResourceInitializer()

    def _validate_result(
        self,
        test_case: Dict[str, Any],
        history: List[Dict[str, str]],
        params: Dict[str, Any],
    ) -> None:
        """Validate the test results for message workers.

        This function validates that the task graph path is correct and that
        the assistant's response is non-empty.

        Args:
            test_case (Dict[str, Any]): Test case containing expected values.
            history (List[Dict[str, str]]): Conversation history.
            params (Dict[str, Any]): Parameters containing task graph information.

        Raises:
            AssertionError: If the task graph path is incorrect or if the
                assistant's response is empty.
        """
        # Check taskgraph path
        node_path: List[str] = [
            i["node_id"] for i in params.get("taskgraph", {}).get("path", {})
        ]
        # Ignore initial '0' node if present
        if node_path and node_path[0] == "0":
            node_path = node_path[1:]
        print(f"DEBUG: node_path = {node_path}")
        print(
            f"DEBUG: expected_taskgraph_path = {test_case['expected_taskgraph_path']}"
        )
        assert node_path == test_case["expected_taskgraph_path"]
        # Message response should be non-empty
        assistant_records: Dict[str, str] = [
            message for message in history if message["role"] == "assistant"
        ][0]
        assert assistant_records["content"] != ""
