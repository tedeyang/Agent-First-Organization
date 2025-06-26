from unittest.mock import MagicMock, Mock, patch

import pytest

from arklex.env.agents.openai_agent import OpenAIAgent
from arklex.utils.graph_state import MessageState, StatusEnum


@pytest.fixture
def mock_tools():
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
def mock_successors_predecessors():
    # Simulate nodes with resource_id attributes
    Node = type("Node", (), {})
    node = Node()
    node.resource_id = "mock_tool_id"
    return [node]


@pytest.fixture
def mock_state():
    state = MessageState()
    state.status = StatusEnum.INCOMPLETE
    state.function_calling_trajectory = []
    state.sys_instruct = "Test instruction"
    state.orchestrator_message = Mock(message="Hello")
    state.bot_config = Mock()
    return state


def test_openai_agent_initialization(
    mock_successors_predecessors, mock_tools, mock_state
):
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
def test_openai_agent_execute(mock_successors_predecessors, mock_tools, mock_state):
    agent = OpenAIAgent(
        successors=mock_successors_predecessors,
        predecessors=[],
        tools=mock_tools,
        state=mock_state,
    )
    # Patch bot_config.llm_config
    mock_state.bot_config.llm_config = Mock(
        llm_provider="openai", model_type_or_path="gpt-3.5-turbo"
    )
    result = agent._execute(mock_state)
    assert "response" in result or isinstance(result, dict)


def make_tool_call_ai_message(tool_name, tool_args=None, tool_id="tool-call-id"):
    # Simulate an AIMessage with a tool call
    tool_call = {"name": tool_name, "args": tool_args or {}, "id": tool_id}
    return Mock(content="", tool_calls=[tool_call])


def make_final_ai_message(content):
    # Simulate an AIMessage with no tool calls (final response)
    return Mock(content=content, tool_calls=None)


@patch(
    "arklex.env.agents.openai_agent.load_prompts",
    return_value={"function_calling_agent_prompt": "Prompt: {message}"},
)
@patch("arklex.env.agents.openai_agent.PromptTemplate")
def test_openai_agent_tool_calling(
    mock_prompt_template,
    mock_load_prompts,
    mock_successors_predecessors,
    mock_tools,
    mock_state,
):
    # Setup tool mock to return a specific value
    tool_func = Mock(return_value={"result": "tool output"})
    tool_obj = Mock(
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
        func=tool_func,
    )
    mock_tools["mock_tool_id"]["execute"] = lambda: tool_obj

    # Patch prompt template to return a mock with .invoke().text
    mock_prompt = Mock()
    mock_prompt.invoke.return_value.text = "Prompt: Hello"
    mock_prompt_template.from_template.return_value = mock_prompt

    # Patch the LLM to simulate tool call and then a final response
    ai_message_tool_call = make_tool_call_ai_message("mock_tool", {"foo": "bar"})
    ai_message_final = make_final_ai_message("Final agent response")
    llm_mock = Mock()
    llm_mock.invoke.side_effect = [ai_message_tool_call, ai_message_final]

    with patch(
        "arklex.env.agents.openai_agent.PROVIDER_MAP",
        {"openai": Mock(return_value=Mock(bind_tools=Mock(return_value=llm_mock)))},
        create=True,
    ):
        agent = OpenAIAgent(
            successors=mock_successors_predecessors,
            predecessors=[],
            tools=mock_tools,
            state=mock_state,
        )
        mock_state.bot_config.llm_config = Mock(
            llm_provider="openai", model_type_or_path="gpt-3.5-turbo"
        )
        result = agent._execute(mock_state)

    # The tool should have been called with the correct arguments
    tool_func.assert_called_once()
    called_kwargs = tool_func.call_args.kwargs
    assert called_kwargs["state"] == mock_state
    assert called_kwargs["foo"] == "bar"
    # The agent's response should be updated to the final message
    assert mock_state.response == "Final agent response"
    # The function_calling_trajectory should include the tool call and tool response
    tool_call_msgs = [
        m
        for m in mock_state.function_calling_trajectory
        if isinstance(m, dict) and m.get("role") == "tool"
    ]
    assert tool_call_msgs, "Tool call message should be present in trajectory"
