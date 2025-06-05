"""Environment management for the Arklex framework.

This module provides functionality for managing the environment, including
worker initialization, tool management, and slot filling integration.
"""

import importlib
import logging
import os
import uuid
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, cast

from arklex.env.planner.react_planner import DefaultPlanner, ReactPlanner
from arklex.env.tools.tools import Tool
from arklex.env.workers.worker import BaseWorker
from arklex.orchestrator.NLU.core.slot import SlotFiller
from arklex.utils.graph_state import MessageState, NodeInfo, Params

logger = logging.getLogger(__name__)


class BaseResourceInitializer:
    """Abstract base class for resource initialization.

    This class defines the interface for initializing tools and workers in the environment.
    Concrete implementations must provide methods for tool and worker initialization.
    """

    @staticmethod
    def init_tools(tools: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Initialize tools from configuration.

        Args:
            tools: List of tool configurations

        Returns:
            Dictionary mapping tool IDs to their configurations

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    @staticmethod
    def init_workers(workers: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Initialize workers from configuration.

        Args:
            workers: List of worker configurations

        Returns:
            Dictionary mapping worker IDs to their configurations

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError


class DefaultResourceInitializer(BaseResourceInitializer):
    """Default implementation of resource initialization.

    This class provides a default implementation for initializing tools and workers
    in the environment.
    """

    @staticmethod
    def init_tools(tools: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Initialize tools from configuration.

        Args:
            tools: List of tool configurations

        Returns:
            Dictionary mapping tool IDs to their configurations
        """
        return {tool["id"]: tool for tool in tools}

    @staticmethod
    def init_workers(workers: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Initialize workers from configuration.

        Args:
            workers: List of worker configurations

        Returns:
            Dictionary mapping worker IDs to their configurations
        """
        return {worker["id"]: worker for worker in workers}


class Environment:
    """Environment management for workers and tools.

    This class manages the environment for workers and tools, including
    initialization, state management, and slot filling integration.

    Attributes:
        slot_fill_api (SlotFiller): Slot filling API instance
        // ... rest of attributes ...
    """

    def __init__(
        self,
        tools: List[Dict[str, Any]],
        workers: List[Dict[str, Any]],
        slot_fill_api: str = "",
        resource_initializer: Optional[BaseResourceInitializer] = None,
        planner_enabled: bool = False,
    ) -> None:
        """Initialize the environment.

        Args:
            tools: List of tools to initialize
            workers: List of workers to initialize
            slot_fill_api: API endpoint for slot filling
            resource_initializer: Resource initializer instance
            planner_enabled: Whether planning is enabled
        """
        self.tools = tools
        self.workers = workers
        self.slot_fill_api = self.initialize_slot_fill_api(slot_fill_api)
        self.resource_initializer = resource_initializer or DefaultResourceInitializer()
        self.planner_enabled = planner_enabled
        if planner_enabled:
            tools_map = DefaultResourceInitializer.init_tools(self.tools)
            workers_map = DefaultResourceInitializer.init_workers(self.workers)
            name2id = {tool["name"]: tool["id"] for tool in self.tools}
            name2id.update({worker["name"]: worker["id"] for worker in self.workers})
            self.planner = ReactPlanner(tools_map, workers_map, name2id)
        else:
            self.planner = None

    def initialize_slot_fill_api(self, slot_fill_api: str) -> SlotFiller:
        """Initialize the slot filling API.

        Args:
            slot_fill_api: API endpoint for slot filling

        Returns:
            SlotFiller: Initialized slot filling API instance
        """
        return SlotFiller(slot_fill_api)

    def _init_tools(self) -> None:
        """Initialize tools with slot filling API."""
        for tool in self.tools:
            if hasattr(tool, "init_slot_filler"):
                tool.init_slot_filler(self.slot_fill_api)

    def _init_workers(self) -> None:
        """Initialize workers with slot filling API."""
        for worker in self.workers:
            if hasattr(worker, "init_slot_filler"):
                worker.init_slot_filler(self.slot_fill_api)

    def step(
        self, id: str, message_state: MessageState, params: Params, node_info: NodeInfo
    ) -> Tuple[MessageState, Params]:
        """Execute a step in the environment.

        This method handles the execution of tools, workers, or planner actions based on
        the provided ID. It manages state updates and parameter modifications.

        Args:
            id: Resource ID to execute
            message_state: Current message state
            params: Current parameters
            node_info: Information about the current node

        Returns:
            Tuple containing updated message state and parameters
        """
        response_state: MessageState
        if id in self.tools:
            logger.info(f"{self.tools[id]['name']} tool selected")
            tool: Tool = self.tools[id]["execute"]()
            tool.init_slotfilling(self.slot_fill_api)
            combined_args: Dict[str, Any] = {
                **self.tools[id]["fixed_args"],
                **(node_info.additional_args or {}),
            }
            response_state = tool.execute(message_state, **combined_args)
            params.memory.function_calling_trajectory = (
                response_state.function_calling_trajectory
            )
            params.taskgraph.dialog_states = response_state.slots
            params.taskgraph.node_status[params.taskgraph.curr_node] = (
                response_state.status
            )

        elif id in self.workers:
            logger.info(f"{self.workers[id]['name']} worker selected")
            worker: BaseWorker = self.workers[id]["execute"]()
            if hasattr(worker, "init_slotfilling"):
                worker.init_slotfilling(self.slot_fill_api)
            response_state = worker.execute(message_state, **node_info.additional_args)
            call_id: str = str(uuid.uuid4())
            params.memory.function_calling_trajectory.append(
                {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {"arguments": "{}", "name": self.id2name[id]},
                            "id": call_id,
                            "type": "function",
                        }
                    ],
                    "function_call": None,
                }
            )
            params.memory.function_calling_trajectory.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": self.id2name[id],
                    "content": (
                        response_state.response
                        if response_state.response
                        else response_state.message_flow
                    ),
                }
            )
            params.taskgraph.node_status[params.taskgraph.curr_node] = (
                response_state.status
            )
        else:
            logger.info("planner selected")
            action: str
            response_state: MessageState
            msg_history: List[Dict[str, Any]]
            action, response_state, msg_history = self.planner.execute(
                message_state, params.memory.function_calling_trajectory
            )

        logger.info(f"Response state from {id}: {response_state}")
        return response_state, params
