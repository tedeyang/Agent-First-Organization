from typing import Any
from unittest.mock import Mock, patch

import pytest

from arklex.env.agents.agent import BaseAgent, register_agent
from arklex.utils.graph_state import MessageState, StatusEnum


class TestRegisterAgent:
    """Test class for register_agent decorator."""

    def test_register_agent_sets_name(self) -> None:
        """Test that register_agent decorator sets the name attribute."""

        @register_agent
        class TestAgent:
            pass

        assert TestAgent.name == "TestAgent"

    def test_register_agent_returns_class(self) -> None:
        """Test that register_agent decorator returns the original class."""

        @register_agent
        class TestAgent:
            pass

        assert TestAgent.__name__ == "TestAgent"

    def test_register_agent_preserves_existing_attributes(self) -> None:
        """Test that register_agent preserves existing class attributes."""

        @register_agent
        class TestAgent:
            description = "Test description"
            custom_attr = "custom value"

        assert TestAgent.name == "TestAgent"
        assert TestAgent.description == "Test description"
        assert TestAgent.custom_attr == "custom value"


class ConcreteAgent(BaseAgent):  # noqa: D101
    """Concrete implementation of BaseAgent for testing."""

    def _execute(self, msg_state: MessageState, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Mock implementation of _execute method."""
        return {
            "status": StatusEnum.COMPLETE,
            "response": "Test response",
            "message_flow": "",
            "function_calling_trajectory": [],
            "trajectory": msg_state.trajectory,
        }


class FailingAgent(BaseAgent):  # noqa: D101
    """Agent that raises exceptions for testing error handling."""

    def _execute(self, msg_state: MessageState, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Implementation that raises an exception."""
        raise ValueError("Test error")


@pytest.fixture
def mock_state() -> MessageState:
    """Create a mock MessageState for testing."""
    state = Mock(spec=MessageState)
    state.status = StatusEnum.INCOMPLETE
    state.response = ""
    state.message_flow = ""
    state.trajectory = [[Mock()]]
    state.trajectory[0][0].output = None
    return state


@pytest.fixture
def complete_mock_state() -> MessageState:
    """Create a mock MessageState with COMPLETE status."""
    state = Mock(spec=MessageState)
    state.status = StatusEnum.COMPLETE
    state.response = "Existing response"
    state.message_flow = ""
    state.trajectory = [[Mock()]]
    state.trajectory[0][0].output = None
    return state


class TestBaseAgent:
    """Test class for BaseAgent."""

    def test_str_representation(self) -> None:
        """Test __str__ method returns class name."""
        agent = ConcreteAgent()
        assert str(agent) == "ConcreteAgent"

    def test_repr_representation(self) -> None:
        """Test __repr__ method returns class name."""
        agent = ConcreteAgent()
        assert repr(agent) == "ConcreteAgent"

    def test_description_default_none(self) -> None:
        """Test that description defaults to None."""
        agent = ConcreteAgent()
        assert agent.description is None

    def test_description_can_be_set(self) -> None:
        """Test that description can be set on subclasses."""

        class TestAgent(BaseAgent):
            description = "Test agent description"

            def _execute(
                self,
                msg_state: MessageState,
                **kwargs: Any,  # noqa: ANN401
            ) -> dict[str, Any]:  # noqa: ANN401
                return {}

        agent = TestAgent()
        assert agent.description == "Test agent description"

    @patch("arklex.env.agents.agent.MessageState.model_validate")
    def test_execute_success(
        self, mock_validate: Mock, mock_state: MessageState
    ) -> None:
        """Test successful execution of agent."""
        # Setup mock return value
        mock_response_state = Mock(spec=MessageState)
        mock_response_state.response = "Test response"
        mock_response_state.message_flow = ""
        mock_response_state.trajectory = [[Mock()]]
        mock_response_state.trajectory[0][0].output = None
        mock_validate.return_value = mock_response_state

        agent = ConcreteAgent()
        result = agent.execute(mock_state)

        # Verify the result
        assert result == mock_response_state
        assert mock_response_state.trajectory[0][0].output == "Test response"
        mock_validate.assert_called_once()

    @patch("arklex.env.agents.agent.MessageState.model_validate")
    def test_execute_with_message_flow_fallback(
        self, mock_validate: Mock, mock_state: MessageState
    ) -> None:  # noqa: E501
        """Test execution when response is empty but message_flow has content."""
        # Setup mock return value with empty response but message_flow content
        mock_response_state = Mock(spec=MessageState)
        mock_response_state.response = ""
        mock_response_state.message_flow = "Message flow content"
        mock_response_state.trajectory = [[Mock()]]
        mock_response_state.trajectory[0][0].output = None
        mock_validate.return_value = mock_response_state

        agent = ConcreteAgent()
        result = agent.execute(mock_state)

        assert result == mock_response_state
        assert mock_response_state.trajectory[0][0].output == "Message flow content"

    @patch("arklex.env.agents.agent.MessageState.model_validate")
    def test_execute_with_none_response_and_message_flow(
        self, mock_validate: Mock, mock_state: MessageState
    ) -> None:
        """Test execution when both response and message_flow are None."""
        # Setup mock return value with None values
        mock_response_state = Mock(spec=MessageState)
        mock_response_state.response = None
        mock_response_state.message_flow = None
        mock_response_state.trajectory = [[Mock()]]
        mock_response_state.trajectory[0][0].output = None
        mock_validate.return_value = mock_response_state

        agent = ConcreteAgent()
        result = agent.execute(mock_state)

        assert result == mock_response_state
        assert mock_response_state.trajectory[0][0].output is None

    @patch("arklex.env.agents.agent.log_context.error")
    @patch("arklex.env.agents.agent.traceback.format_exc")
    def test_execute_handles_exception(
        self, mock_format_exc: Mock, mock_log_error: Mock, mock_state: MessageState
    ) -> None:
        """Test that execute handles exceptions gracefully."""
        mock_format_exc.return_value = "Mock traceback"

        agent = FailingAgent()
        result = agent.execute(mock_state)

        # Should return the original state when exception occurs
        assert result == mock_state
        mock_log_error.assert_called_once_with("Mock traceback")

    @patch("arklex.env.agents.agent.log_context.info")
    def test_complete_state_incomplete_to_complete(
        self, mock_log_info: Mock, mock_state: MessageState
    ) -> None:
        """Test complete_state changes INCOMPLETE to COMPLETE."""
        agent = ConcreteAgent()
        agent.name = "TestAgent"

        result = agent.complete_state(mock_state)

        assert result.status == StatusEnum.COMPLETE
        mock_log_info.assert_called_once_with(
            f"Ending agent {agent.name} with status {StatusEnum.COMPLETE}"
        )

    @patch("arklex.env.agents.agent.log_context.info")
    def test_complete_state_already_complete(
        self, mock_log_info: Mock, complete_mock_state: MessageState
    ) -> None:
        """Test complete_state with already COMPLETE status."""
        agent = ConcreteAgent()
        agent.name = "TestAgent"

        result = agent.complete_state(complete_mock_state)

        assert result.status == StatusEnum.COMPLETE
        mock_log_info.assert_called_once_with(
            f"Ending agent {agent.name} with status {StatusEnum.COMPLETE}"
        )

    def test_execute_with_kwargs(self, mock_state: MessageState) -> None:
        """Test that execute passes kwargs to _execute."""

        class KwargsTestAgent(BaseAgent):
            def _execute(
                self,
                msg_state: MessageState,
                **kwargs: Any,  # noqa: ANN401
            ) -> dict[str, Any]:  # noqa: ANN401
                return {
                    "status": StatusEnum.COMPLETE,
                    "response": f"Received: {kwargs.get('test_param', 'None')}",
                    "message_flow": "",
                    "function_calling_trajectory": [],
                    "trajectory": msg_state.trajectory,
                }

        agent = KwargsTestAgent()

        with patch(
            "arklex.env.agents.agent.MessageState.model_validate"
        ) as mock_validate:
            mock_response_state = Mock(spec=MessageState)
            mock_response_state.response = "Received: test_value"
            mock_response_state.message_flow = ""
            mock_response_state.trajectory = [[Mock()]]
            mock_response_state.trajectory[0][0].output = None
            mock_validate.return_value = mock_response_state

            agent.execute(mock_state, test_param="test_value")

            assert mock_response_state.trajectory[0][0].output == "Received: test_value"

    def test_complete_state_with_kwargs(self, mock_state: MessageState) -> None:
        """Test that complete_state accepts kwargs."""
        agent = ConcreteAgent()
        agent.name = "TestAgent"

        # Should not raise an exception even with extra kwargs
        result = agent.complete_state(mock_state, extra_param="value")

        assert result.status == StatusEnum.COMPLETE

    def test_abstract_execute_method(self) -> None:
        """Test that BaseAgent cannot be instantiated due to abstract method."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseAgent()


class TestAgentIntegration:
    """Integration tests for agent functionality."""

    def test_registered_agent_with_execution(self, mock_state: MessageState) -> None:
        """Test a registered agent can be executed successfully."""

        @register_agent
        class IntegrationTestAgent(BaseAgent):
            description = "Integration test agent"

            def _execute(
                self,
                msg_state: MessageState,
                **kwargs: Any,  # noqa: ANN401
            ) -> dict[str, Any]:  # noqa: ANN401, E501
                return {
                    "status": StatusEnum.COMPLETE,
                    "response": "Integration test complete",
                    "message_flow": "",
                    "function_calling_trajectory": [],
                    "trajectory": msg_state.trajectory,
                }

        agent = IntegrationTestAgent()

        # Test registration
        assert agent.name == "IntegrationTestAgent"
        assert agent.description == "Integration test agent"

        # Test execution
        with patch(
            "arklex.env.agents.agent.MessageState.model_validate"
        ) as mock_validate:
            mock_response_state = Mock(spec=MessageState)
            mock_response_state.response = "Integration test complete"
            mock_response_state.message_flow = ""
            mock_response_state.trajectory = [[Mock()]]
            mock_response_state.trajectory[0][0].output = None
            mock_validate.return_value = mock_response_state

            result = agent.execute(mock_state)
            assert (
                mock_response_state.trajectory[0][0].output
                == "Integration test complete"
            )

        # Test completion
        result = agent.complete_state(mock_state)
        assert result.status == StatusEnum.COMPLETE
