import logging
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar

from arklex.utils.graph_state import MessageState, StatusEnum

logger = logging.getLogger(__name__)

T = TypeVar("T")


def register_agent(cls: Type[T]) -> Type[T]:
    """Register an agent class with the Arklex framework.

    This decorator registers an agent class and automatically sets its name
    to the class name. It is used to mark classes as agents in the system.

    Args:
        cls (Type[T]): The agent class to register.

    Returns:
        Type[T]: The registered agent class.
    """
    cls.name = cls.__name__  # Automatically set name to the class name
    return cls


class BaseAgent(ABC):
    """Base abstract class for agents in the Arklex framework.

    This class defines the interface for agent execution and provides common
    functionality for all agents. It includes methods for string representation,
    execution handling, and error management.

    Attributes:
        description (Optional[str]): Description of the agent's functionality.
    """

    description: Optional[str] = None

    def __str__(self) -> str:
        """Get a string representation of the agent.

        Returns:
            str: The name of the agent class.
        """
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        """Get a detailed string representation of the agent.

        Returns:
            str: The name of the agent class.
        """
        return f"{self.__class__.__name__}"

    @abstractmethod
    def _execute(self, msg_state: MessageState, **kwargs: Any) -> Dict[str, Any]:
        """Execute the agent's core functionality.

        This abstract method must be implemented by concrete agent classes to
        define their specific execution logic.

        Args:
            msg_state (MessageState): The current message state.
            **kwargs (Any): Additional keyword arguments for the execution.

        Returns:
            Dict[str, Any]: The execution results as a dictionary.
        """
        pass

    def execute(self, msg_state: MessageState, **kwargs: Any) -> MessageState:
        """Execute the agent with error handling and state management.

        This method wraps the agent's execution with error handling and state
        management. It processes the execution results and updates the message state.

        Args:
            msg_state (MessageState): The current message state.
            **kwargs (Any): Additional keyword arguments for the execution.

        Returns:
            MessageState: The updated message state after execution.
        """
        try:
            response_return: Dict[str, Any] = self._execute(msg_state, **kwargs)
            response_state: MessageState = MessageState.model_validate(response_return)
            response_state.trajectory[-1][-1].output = (
                response_state.response
                if response_state.response
                else response_state.message_flow
            )
            if response_state.status == StatusEnum.INCOMPLETE:
                response_state.status = StatusEnum.COMPLETE
            return response_state
        except Exception as e:
            logger.error(traceback.format_exc())
            msg_state.status = StatusEnum.INCOMPLETE
            return msg_state

    def complete_state(self, msg_state: MessageState, **kwargs: Any) -> MessageState:
        """Clean up resources or perform any final actions when the agent is no longer needed.

        This method can be overridden by subclasses to implement specific cleanup logic.
        """
        try:
            if msg_state.status == StatusEnum.INCOMPLETE:
                msg_state.status = StatusEnum.COMPLETE
            logger.info(f"Ending agent {self.name} with status {msg_state.status}")
        except Exception as e:
            logger.error(f"Error when ending agent : {traceback.format_exc()}")
            msg_state.status = StatusEnum.INCOMPLETE
            msg_state.response = str(e)
            logger.error(f"Agent {self.name} ended with error: {e}")
        return msg_state
