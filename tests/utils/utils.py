"""Core test utilities for the Arklex framework.

This module provides core test utilities for the Arklex framework, including
the MockOrchestrator base class for testing orchestrator behavior. It includes
methods for initializing tests, executing conversations, and validating results,
with support for custom validation through abstract methods.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import contextlib
from unittest.mock import patch

from arklex.env.env import Environment
from arklex.orchestrator.orchestrator import AgentOrg
from arklex.utils.graph_state import MessageState


class MockTool:
    """Mock tool for testing purposes.

    This class provides a mock implementation of a tool that can be used in tests.
    It simulates tool execution by returning predefined responses.
    """

    def __init__(self, name: str, description: str) -> None:
        """Initialize the mock tool.

        Args:
            name (str): Name of the tool
            description (str): Description of the tool's functionality
        """
        self.name = name
        self.description = description
        self.info = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        self.output = []

    def execute(self) -> "MockTool":
        """Return self to simulate tool execution.

        Returns:
            MockTool: The mock tool instance
        """
        return self


class MockResourceInitializer:
    """Mock resource initializer for testing purposes.

    This class provides a mock implementation of resource initialization that
    creates mock tools and workers for testing.
    """

    @staticmethod
    def init_tools(tools: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Initialize mock tools from configuration.

        Args:
            tools (List[Dict[str, Any]]): List of tool configurations

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping tool IDs to their configurations
        """
        print("\n=== Debug: MockResourceInitializer.init_tools ===")
        print(f"Input tools: {json.dumps(tools, indent=2)}")

        tools_map = {}
        if not tools:
            print("No tools provided, creating dummy tool")
            dummy_tool = MockTool("dummy_tool", "A dummy tool for testing.")
            tools_map["dummy_tool"] = dummy_tool
            print(f"Created dummy tool map: {json.dumps(tools_map, indent=2)}")
            return tools_map

        for tool in tools:
            name = tool.get("name", tool.get("id", "unnamed_tool"))
            description = tool.get("description", "No description provided.")
            print(f"\nProcessing tool: {name}")
            print(f"Tool config: {json.dumps(tool, indent=2)}")

            mock_tool = MockTool(name, description)
            tool_id = tool.get("id", name)
            tools_map[tool_id] = mock_tool
            print(f"Added tool to map with ID: {tool_id}")
            print(f"Tool entry: {json.dumps(tools_map[tool_id].__dict__, indent=2)}")

        print("\nFinal tools map:")
        print(json.dumps({k: v.__dict__ for k, v in tools_map.items()}, indent=2))
        return tools_map

    @staticmethod
    def init_workers(workers: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Initialize mock workers from configuration.

        Args:
            workers (List[Dict[str, Any]]): List of worker configurations

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping worker IDs to their configurations
        """
        print("\n=== Debug: MockResourceInitializer.init_workers ===")
        print(f"Input workers: {json.dumps(workers, indent=2)}")

        workers_map = {}
        if not workers:
            print("No workers provided, returning empty map")
            return workers_map

        for worker in workers:
            name = worker.get("name", worker.get("id", "unnamed_worker"))
            description = worker.get("description", "No description provided.")
            print(f"\nProcessing worker: {name}")
            print(f"Worker config: {json.dumps(worker, indent=2)}")

            mock_tool = MockTool(name, description)
            worker_id = worker.get("id", name)
            workers_map[worker_id] = mock_tool
            print(f"Added worker to map with ID: {worker_id}")
            print(
                f"Worker entry: {json.dumps(workers_map[worker_id].__dict__, indent=2)}"
            )

        print("\nFinal workers map:")
        print(json.dumps({k: v.__dict__ for k, v in workers_map.items()}, indent=2))
        return workers_map


@contextlib.contextmanager
def mock_llm_invoke():
    class DummyAIMessage:
        def __init__(self, content):
            self.content = content

    def dummy_invoke(*args, **kwargs):
        # Return a plausible planner response for all tests
        # This should look like a real planner output
        return DummyAIMessage(
            '{"name": "respond", "arguments": {"content": "Here are the products: A, B, and C. Which one do you want to know more about?"}}'
        )

    with patch("arklex.env.planner.react_planner.ChatOpenAI.invoke", new=dummy_invoke):
        yield


class MockOrchestrator(ABC):
    def __init__(self, config_file_path: str, fixed_args: Dict[str, Any] = {}) -> None:
        """Initialize the mock orchestrator.

        Args:
            config_file_path (str): Path to the configuration file.
            fixed_args (Dict[str, Any], optional): Fixed arguments to update
                tool configurations. Defaults to empty dict.
        """
        self.user_prefix: str = "user"
        self.assistant_prefix: str = "assistant"
        config: Dict[str, Any] = json.load(open(config_file_path))
        if fixed_args:
            for tool in config["tools"]:
                tool["fixed_args"].update(fixed_args)
        self.config: Dict[str, Any] = config

    def _get_test_response(
        self, user_text: str, history: List[Dict[str, str]], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get a test response from the orchestrator.

        This function simulates a conversation by sending user text to the
        orchestrator and getting a response.

        Args:
            user_text (str): The user's input text.
            history (List[Dict[str, str]]): Conversation history.
            params (Dict[str, Any]): Parameters for the conversation.

        Returns:
            Dict[str, Any]: The orchestrator's response containing the answer
                and updated parameters.
        """
        data: Dict[str, Any] = {
            "text": user_text,
            "chat_history": history,
            "parameters": params,
        }
        from tests.utils.utils import MockResourceInitializer

        env_kwargs = dict(
            tools=self.config["tools"],
            workers=self.config["workers"],
            slot_fill_api=self.config["slotfillapi"],
            planner_enabled=True,
            resource_initializer=MockResourceInitializer(),
        )
        orchestrator = AgentOrg(
            config=self.config,
            env=Environment(**env_kwargs),
        )
        return orchestrator.get_response(data)

    def _initialize_test(self) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Initialize a test conversation.

        This function sets up the initial conversation state by creating an
        empty history and parameters, and adding the start message if one
        exists in the configuration.

        Returns:
            Tuple[List[Dict[str, str]], Dict[str, Any]]: A tuple containing
                the initial conversation history and parameters.
        """
        history: List[Dict[str, str]] = []
        params: Dict[str, Any] = {}
        start_message: Optional[str] = None
        for node in self.config["nodes"]:
            if node[1].get("type", "") == "start":
                start_message = node[1]["attribute"]["value"]
                break
        if start_message:
            history.append({"role": self.assistant_prefix, "content": start_message})
        return history, params

    def _execute_conversation(
        self,
        test_case: Dict[str, Any],
        history: List[Dict[str, str]],
        params: Dict[str, Any],
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Execute a test conversation.

        This function simulates a conversation by processing each user utterance
        in the test case, getting responses from the orchestrator, and updating
        the conversation history and parameters.

        Args:
            test_case (Dict[str, Any]): Test case containing user utterances.
            history (List[Dict[str, str]]): Initial conversation history.
            params (Dict[str, Any]): Initial parameters.

        Returns:
            Tuple[List[Dict[str, str]], Dict[str, Any]]: A tuple containing
                the updated conversation history and parameters.
        """
        for user_text in test_case["user_utterance"]:
            result: Dict[str, Any] = self._get_test_response(user_text, history, params)
            answer: str = result["answer"]
            params = result["parameters"]
            history.append({"role": self.user_prefix, "content": user_text})
            history.append({"role": self.assistant_prefix, "content": answer})
        return history, params

    @abstractmethod
    def _validate_result(
        self,
        test_case: Dict[str, Any],
        history: List[Dict[str, str]],
        params: Dict[str, Any],
    ) -> None:
        """Validate the test results.

        This abstract method should be implemented by subclasses to validate
        the results of a test case. The implementation should check that the
        conversation history and parameters match the expected values.

        Args:
            test_case (Dict[str, Any]): Test case containing expected values.
            history (List[Dict[str, str]]): Conversation history to validate.
            params (Dict[str, Any]): Parameters to validate.
        """
        # NOTE: change the assert to check the result
        pass

    def run_single_test(self, test_case: Dict[str, Any]) -> None:
        """Run a single test case.

        This function executes a test case by initializing the conversation,
        executing the conversation, and validating the results.

        Args:
            test_case (Dict[str, Any]): Test case to run.

        Raises:
            AssertionError: If the test validation fails.
        """
        with mock_llm_invoke():
            history, params = self._initialize_test()
            history, params = self._execute_conversation(test_case, history, params)
            self._validate_result(test_case, history, params)
