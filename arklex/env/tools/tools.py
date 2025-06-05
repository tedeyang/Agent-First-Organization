"""Tool management for the Arklex framework.

This module provides functionality for managing tools, including
initialization, execution, and slot filling integration.
"""

import os
import logging
import json
import uuid
import inspect
import traceback
from typing import Any, Callable, Dict, List, Optional

from arklex.utils.graph_state import MessageState, StatusEnum
from arklex.utils.slot import Slot
from arklex.orchestrator.NLU.core.slot import SlotFiller
from arklex.utils.utils import format_chat_history
from arklex.exceptions import ToolExecutionError, AuthenticationError

logger = logging.getLogger(__name__)


def register_tool(
    desc: str,
    slots: List[Dict[str, Any]] = [],
    outputs: List[str] = [],
    isResponse: bool = False,
) -> Callable:
    """Register a tool with the Arklex framework.

    This decorator registers a function as a tool with the specified description, slots,
    outputs, and response flag. It handles path normalization and tool initialization.

    Args:
        desc (str): Description of the tool's functionality.
        slots (List[Dict[str, Any]], optional): List of slot definitions. Defaults to [].
        outputs (List[str], optional): List of output field names. Defaults to [].
        isResponse (bool, optional): Whether the tool is a response tool. Defaults to False.

    Returns:
        Callable: A function that creates and returns a Tool instance.
    """
    current_file_dir: str = os.path.dirname(__file__)

    def inner(func: Callable) -> Callable:
        file_path: str = inspect.getfile(func)
        relative_path: str = os.path.relpath(file_path, current_file_dir)
        # reformat the relative path to replace / and \\ with -, and remove .py, because the function calling in openai only allow the function name match the patter the pattern '^[a-zA-Z0-9_-]+$'
        # different file paths format in Windows and linux systems
        relative_path = (
            relative_path.replace("/", "-").replace("\\", "-").replace(".py", "")
        )
        key: str = f"{relative_path}-{func.__name__}"

        def tool() -> "Tool":
            return Tool(func, key, desc, slots, outputs, isResponse)

        return tool

    return inner


class BaseTool:
    """Base class for tools.

    This class provides the base functionality for tools, including
    initialization, execution, and slot filling integration.

    Attributes:
        slot_fill_api (Optional[SlotFiller]): Slot filling API instance
        // ... rest of attributes ...
    """

    def __init__(self, name: str, description: str) -> None:
        """Initialize the tool.

        Args:
            name: Name of the tool
            description: Description of the tool's functionality
        """
        self.name = name
        self.description = description
        self.slot_fill_api: Optional[SlotFiller] = None

    def init_slot_filler(self, slot_fill_api: SlotFiller) -> None:
        """Initialize the slot filling API.

        Args:
            slot_fill_api: Slot filling API instance
        """
        self.slot_fill_api = slot_fill_api

    def execute(self, message_state: MessageState) -> Any:
        """Execute the tool.

        Args:
            message_state: Current message state

        Returns:
            Result of tool execution
        """
        # do slot filling
        if self.slot_fill_api:
            slots = self._get_required_slots()
            filled_slots = self.slot_fill_api.execute(slots, message_state.text)
            message_state.slots = filled_slots

        return self._execute_impl(message_state)

    def _execute_impl(self, message_state: MessageState) -> Any:
        """Implementation of tool execution.

        Args:
            message_state: Current message state

        Returns:
            Result of tool execution
        """
        raise NotImplementedError

    def _get_required_slots(self) -> List[Slot]:
        """Get required slots for the tool.

        Returns:
            List of required slots
        """
        raise NotImplementedError


class Tool:
    """Base class for tools in the Arklex framework.

    This class provides the core functionality for tool execution, slot management,
    and state handling. It supports slot filling, parameter validation, and error
    handling during tool execution.

    Attributes:
        func (Callable): The function implementing the tool's functionality.
        name (str): The name of the tool.
        description (str): Description of the tool's functionality.
        output (List[str]): List of output field names.
        slotfillapi (Optional[SlotFiller]): Slot filling API instance.
        info (Dict[str, Any]): Tool information including parameters and requirements.
        slots (List[Slot]): List of slot instances.
        isResponse (bool): Whether the tool is a response tool.
        properties (Dict[str, Dict[str, Any]]): Tool properties.
        llm_config (Dict[str, Any]): Language model configuration.
    """

    def __init__(
        self,
        func: Callable,
        name: str,
        description: str,
        slots: List[Dict[str, Any]],
        outputs: List[str],
        isResponse: bool,
    ):
        """Initialize a new Tool instance.

        Args:
            func (Callable): The function implementing the tool's functionality.
            name (str): The name of the tool.
            description (str): Description of the tool's functionality.
            slots (List[Dict[str, Any]]): List of slot definitions.
            outputs (List[str]): List of output field names.
            isResponse (bool): Whether the tool is a response tool.
        """
        self.func: Callable = func
        self.name: str = name
        self.description: str = description
        self.output: List[str] = outputs
        self.slotfillapi: Optional[SlotFiller] = None
        self.info: Dict[str, Any] = self.get_info(slots)
        self.slots: List[Slot] = [Slot.model_validate(slot) for slot in slots]
        self.isResponse: bool = isResponse
        self.properties: Dict[str, Dict[str, Any]] = {}
        self.llm_config: Dict[str, Any] = {}

    def get_info(self, slots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get tool information including parameters and requirements.

        This method processes the slot definitions to create a structured
        representation of the tool's parameters and requirements.

        Args:
            slots (List[Dict[str, Any]]): List of slot definitions.

        Returns:
            Dict[str, Any]: Tool information including parameters and requirements.
        """
        self.properties = {}
        for slot in slots:
            self.properties[slot["name"]] = {
                k: v
                for k, v in slot.items()
                if k in ["type", "description", "prompt", "items"]
            }
        required: List[str] = [
            slot["name"] for slot in slots if slot.get("required", False)
        ]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": required,
                },
            },
        }

    def init_slotfilling(self, slotfillapi: SlotFiller) -> None:
        """Initialize the slot filling API.

        Args:
            slotfillapi (SlotFiller): The slot filling API instance to use.
        """
        self.slotfillapi = slotfillapi

    def _init_slots(self, state: MessageState) -> None:
        """Initialize slots with default values from the message state.

        This method processes default slots from the message state and updates
        the tool's slots with their values.

        Args:
            state (MessageState): The current message state.
        """
        default_slots: List[Slot] = state.slots.get("default_slots", [])
        logger.info(f"Default slots are: {default_slots}")
        if not default_slots:
            return
        response: Dict[str, Any] = {}
        for default_slot in default_slots:
            response[default_slot.name] = default_slot.value
            for slot in self.slots:
                if slot.name == default_slot.name and default_slot.value:
                    slot.value = default_slot.value
                    slot.verified = True
        state.function_calling_trajectory.append(
            {
                "role": "tool",
                "tool_call_id": str(uuid.uuid4()),
                "name": "default_slots",
                "content": json.dumps(response),
            }
        )

        logger.info(f"Slots after initialization are: {self.slots}")

    def _execute(self, state: MessageState, **fixed_args: Any) -> MessageState:
        """Execute the tool with the current state and fixed arguments.

        This method handles slot filling, parameter validation, and tool execution.
        It manages the execution flow, error handling, and state updates.

        Args:
            state (MessageState): The current message state.
            **fixed_args (Any): Additional fixed arguments for the tool.

        Returns:
            MessageState: The updated message state after tool execution.
        """
        slot_verification: bool = False
        reason: str = ""
        # if this tool has been called before, then load the previous slots status
        if state.slots.get(self.name):
            self.slots = state.slots[self.name]
        else:
            state.slots[self.name] = self.slots
        # init slot values saved in default slots
        self._init_slots(state)
        # do slotfilling
        chat_history_str: str = format_chat_history(state.function_calling_trajectory)
        slots: List[Slot] = self.slotfillapi.execute(
            self.slots, chat_history_str, self.llm_config
        )
        logger.info(f"{slots=}")
        if not all([slot.value and slot.verified for slot in slots if slot.required]):
            for slot in slots:
                # if there is extracted slots values but haven't been verified
                if slot.value and not slot.verified:
                    # check whether it verified or not
                    verification_needed: bool
                    thought: str
                    verification_needed, thought = self.slotfillapi.verify_needed(
                        slot, chat_history_str, self.llm_config
                    )
                    if verification_needed:
                        response: str = slot.prompt + "The reason is: " + thought
                        slot_verification = True
                        reason = thought
                        break
                    else:
                        slot.verified = True
                # if there is no extracted slots values, then should prompt the user to fill the slot
                if not slot.value:
                    response = slot.prompt
                    break

            state.status = StatusEnum.INCOMPLETE

        # if slot.value is not empty for all slots, and all the slots has been verified, then execute the function
        tool_success: bool = False
        if all([slot.value and slot.verified for slot in slots if slot.required]):
            logger.info("all slots filled")
            kwargs: Dict[str, Any] = {slot.name: slot.value for slot in slots}
            combined_kwargs: Dict[str, Any] = {
                **kwargs,
                **fixed_args,
                **self.llm_config,
            }
            try:
                response = self.func(**combined_kwargs)
                tool_success = True
            except ToolExecutionError as tee:
                logger.error(traceback.format_exc())
                response = tee.extra_message
            except AuthenticationError as ae:
                logger.error(traceback.format_exc())
                response = str(ae)
            except Exception as e:
                logger.error(traceback.format_exc())
                response = str(e)
            logger.info(f"Tool {self.name} response: {response}")
            call_id: str = str(uuid.uuid4())
            state.function_calling_trajectory.append(
                {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": json.dumps(kwargs),
                                "name": self.name,
                            },
                            "id": call_id,
                            "type": "function",
                        }
                    ],
                    "function_call": None,
                }
            )
            state.function_calling_trajectory.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": self.name,
                    "content": response,
                }
            )
            state.status = (
                StatusEnum.COMPLETE if tool_success else StatusEnum.INCOMPLETE
            )

        state.trajectory[-1][-1].input = slots
        state.trajectory[-1][-1].output = response

        if tool_success:
            # Tool execution success
            if self.isResponse:
                logger.info(
                    "Tool exeuction COMPLETE, and the output is stored in response"
                )
                state.response = response
            else:
                logger.info(
                    "Tool execution COMPLETE, and the output is stored in message flow"
                )
                state.message_flow = (
                    state.message_flow
                    + f"Context from {self.name} tool execution: {response}\n"
                )
        else:
            # Tool execution failed
            if slot_verification:
                logger.info("Tool execution INCOMPLETE due to slot verification")
                state.message_flow = f"Context from {self.name} tool execution: {response}\n Focus on the '{reason}' to generate the verification request in response please and make sure the request appear in the response."
            else:
                logger.info("Tool execution INCOMPLETE due to tool execution failure")
                state.message_flow = (
                    state.message_flow
                    + f"Context from {self.name} tool execution: {response}\n"
                )
        state.slots[self.name] = slots
        return state

    def execute(self, state: MessageState, **fixed_args: Any) -> MessageState:
        """Execute the tool with the current state and fixed arguments.

        This method is a wrapper around _execute that handles the execution flow
        and state management.

        Args:
            state (MessageState): The current message state.
            **fixed_args (Any): Additional fixed arguments for the tool.

        Returns:
            MessageState: The updated message state after tool execution.
        """
        self.llm_config = state.bot_config.llm_config.model_dump()
        state = self._execute(state, **fixed_args)
        return state

    def __str__(self) -> str:
        """Get a string representation of the tool.

        Returns:
            str: A string representation of the tool.
        """
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        """Get a detailed string representation of the tool.

        Returns:
            str: A detailed string representation of the tool.
        """
        return f"{self.__class__.__name__}"
