from unittest.mock import MagicMock, Mock, patch

import pytest

from arklex.env.agents.openai_agent import OpenAIAgent
from arklex.utils.graph_state import (
    BotConfig,
    LLMConfig,
    MessageState,
    OrchestratorMessage,
    ResourceRecord,
    StatusEnum,
)


@pytest.fixture
def mock_tools() -> dict[str, MagicMock]:
    # Simulate a tool registry with a mock tool
    mock_tool = {
        "execute": lambda: Mock(
            to_openai_tool_def_v2=Mock(
                return_value={
                    "type": "function",
                    "function": {
                        "name": "mock_tool",
                        "description": "desc",
                        "parameters": {},
                    },
                }
            ),
            func=Mock(return_value=None),
        ),
        "fixed_args": {},
    }
    return {"mock_tool_id": mock_tool}


@pytest.fixture
def mock_successors_predecessors() -> list[MagicMock]:
    # Simulate nodes with resource_id attributes
    class Node:
        def __init__(self) -> None:
            self.resource_id = "mock_tool_id"

    node = Node()
    return [node]


@pytest.fixture
def mock_state() -> MessageState:
    # Create proper Pydantic models instead of Mock objects
    llm_config = LLMConfig(model_type_or_path="gpt-3.5-turbo", llm_provider="openai")
    bot_config = BotConfig(
        bot_id="test_bot",
        version="1.0",
        language="EN",
        bot_type="test",
        llm_config=llm_config,
    )
    orchestrator_message = OrchestratorMessage(message="Hello", attribute={})

    # Create a proper trajectory structure with ResourceRecord
    resource_record = ResourceRecord(
        info={"test": "info"},
        intent="test_intent",
        input=[],
        output="",
        steps=[],
        personalized_intent="",
    )
    trajectory = [[resource_record]]  # List of lists of ResourceRecord objects

    state = MessageState(
        status=StatusEnum.INCOMPLETE,
        function_calling_trajectory=[],
        sys_instruct="Test instruction",
        orchestrator_message=orchestrator_message,
        bot_config=bot_config,
        trajectory=trajectory,
    )
    return state


def test_openai_agent_initialization(
    mock_successors_predecessors: list, mock_tools: list, mock_state: MessageState
) -> None:
    agent = OpenAIAgent(
        successors=mock_successors_predecessors,
        predecessors=[],
        tools=mock_tools,
        state=mock_state,
    )
    assert hasattr(agent, "action_graph")
    assert isinstance(agent.tool_defs, list)
    assert "mock_tool" in agent.tool_map


@patch(
    "arklex.env.agents.openai_agent.PROVIDER_MAP",
    {
        "openai": Mock(
            return_value=Mock(
                bind_tools=Mock(
                    return_value=Mock(
                        invoke=Mock(
                            return_value=Mock(content="response", tool_calls=None)
                        )
                    )
                )
            )
        )
    },
)
@patch(
    "arklex.env.agents.openai_agent.ChatOpenAI",
    Mock(
        return_value=Mock(
            bind_tools=Mock(
                return_value=Mock(
                    invoke=Mock(return_value=Mock(content="response", tool_calls=None))
                )
            )
        )
    ),
)
def test_openai_agent_execute(
    mock_successors_predecessors: list, mock_tools: list, mock_state: MessageState
) -> None:
    agent = OpenAIAgent(
        successors=mock_successors_predecessors,
        predecessors=[],
        tools=mock_tools,
        state=mock_state,
    )
    result = agent._execute(mock_state)
    assert "response" in result or isinstance(result, dict)


@patch(
    "arklex.env.agents.openai_agent.PROVIDER_MAP",
    {
        "openai": Mock(
            return_value=Mock(
                invoke=Mock(
                    return_value=Mock(content="Thank you for using Arklex. Goodbye!")
                )
            )
        )
    },
)
def test_openai_agent_with_no_tools(mock_state: MessageState) -> None:
    """Test OpenAIAgent initialization with no tools."""
    agent = OpenAIAgent(
        successors=[],
        predecessors=[],
        tools=[],
        state=mock_state,
    )

    # Should still have the end_conversation tool
    assert "end_conversation" in agent.tool_map
    assert len(agent.tool_defs) == 1  # Only end_conversation tool
