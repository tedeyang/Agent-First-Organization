"""Environment management for the Arklex framework.

This module provides functionality for managing the environment, including
worker initialization, tool management, and slot filling integration.
"""

import uuid
from typing import Any

from arklex.env.agents.agent import BaseAgent
from arklex.env.entities import NodeResponse
from arklex.env.resource_map import RESOURCE_MAP
from arklex.env.tools.tools import Tool
from arklex.env.workers.base.base_worker import BaseWorker
from arklex.orchestrator.entities.orchestrator_state_entities import OrchestratorState
from arklex.orchestrator.entities.taskgraph_entities import NodeInfo, StatusEnum
from arklex.orchestrator.NLU.core.slot import Slot, SlotFiller
from arklex.orchestrator.NLU.services.api_service import APIClientService
from arklex.orchestrator.NLU.services.model_service import (
    DummyModelService,
    ModelService,
)
from arklex.types.resource_types import ToolItem, WorkerItem
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class BaseResourceInitializer:
    """Abstract base class for resource initialization.

    This class defines the interface for initializing tools and workers in the environment.
    Concrete implementations must provide methods for tool and worker initialization.
    """

    @staticmethod
    def init_tools(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Initialize tools from configuration.

        Args:
            tools: list of tool configurations

        Returns:
            dictionary mapping tool IDs to their configurations

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    @staticmethod
    def init_workers(workers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Initialize workers from configuration.

        Args:
            workers: list of worker configurations

        Returns:
            dictionary mapping worker IDs to their configurations

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
    def init_tools(
        tools: list[dict[str, Any]], nodes: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Initialize tools from configuration.

        Args:
            tools: list of tool configurations
            attributes_list: optional list of attributes for the tools

        Returns:
            dictionary mapping tool IDs to their configurations
        """
        tool_registry: dict[str, dict[str, Any]] = {}
        for tool in tools:
            tool_id: str = tool["id"]
            try:
                if tool_id == ToolItem.HTTP_TOOL:
                    http_tool_collection = {}
                    for node in nodes:
                        node_info = node[1]
                        node_data = node_info.get("data", {})
                        if (
                            node_info.get("resource", {}).get("id") != tool_id
                            or not node_data
                        ):
                            continue
                        # Create a new tool instance for each node to avoid sharing state
                        base_tool: Tool = RESOURCE_MAP[tool_id]["item_cls"]
                        tool_instance: Tool = Tool(
                            func=base_tool.func,
                            name=base_tool.name,
                            description=base_tool.description,
                            slots=[],
                        )
                        tool_instance.auth.update(tool.get("auth", {}))
                        tool_instance.node_specific_data = node_data
                        # --- Begin slot group merge logic ---
                        slots = node_data.get("slots", [])
                        slot_groups = node_data.get("slot_groups", [])
                        group_slots = []
                        for group in slot_groups:
                            # Generate prompt/description for the group
                            required_fields = [
                                s["name"]
                                for s in group.get("schema", [])
                                if s.get("required", False)
                            ]
                            prompt = (
                                f"Please provide at least one set of the following fields: {', '.join(required_fields)}."
                                if required_fields
                                else f"Please provide a set of values for group '{group['name']}'."
                            )
                            description = f"Slot group '{group['name']}' with schema: {[s['name'] for s in group.get('schema', [])]}"
                            group_slots.append(
                                {
                                    "name": group["name"],
                                    "type": "group",
                                    "schema": group.get("schema", []),
                                    "required": group.get("required", False),
                                    "repeatable": group.get("repeatable", True),
                                    "prompt": prompt,
                                    "description": description,
                                }
                            )
                        all_slots = slots + group_slots
                        tool_instance.load_slots(all_slots)
                        tool_instance.name = node_data.get("name", "")
                        tool_instance.description = node_data.get("task", "")
                        http_tool_collection[tool_instance.name] = {
                            "tool_instance": tool_instance,
                        }
                    tool_registry[tool_id] = http_tool_collection
                else:
                    print(f"tool_id: {tool_id}")
                    print(f"RESOURCE_MAP: {RESOURCE_MAP}")
                    tool_instance: Tool = RESOURCE_MAP[tool_id]["item_cls"]
                    print(f"tool_instance: {tool_instance}")
                    tool_instance.auth.update(tool.get("auth", {}))
                    tool_instance.node_specific_data = {}
                    tool_registry[tool_id] = {
                        "tool_instance": tool_instance,
                    }
            except Exception as e:
                log_context.exception(e)
                log_context.error(f"Tool {tool_id} is not registered, error: {e}")

        return tool_registry

    @staticmethod
    def init_workers(workers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Initialize workers from configuration.

        Args:
            workers: list of worker configurations

        Returns:
            dictionary mapping worker IDs to their configurations
        """
        worker_registry: dict[str, dict[str, Any]] = {}
        for worker in workers:
            worker_id: str = worker["id"]
            try:
                worker_registry[worker_id] = {
                    "item_cls": RESOURCE_MAP[worker["id"]]["item_cls"],
                }
            except Exception as e:
                log_context.error(f"Worker {worker_id} is not registered, error: {e}")
        return worker_registry

    @staticmethod
    def init_agents(agents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Initialize agents from configuration.

        Args:
            agents: list of agent configurations

        Returns:
            dictionary mapping agent IDs to their configurations
        """
        agent_registry: dict[str, dict[str, Any]] = {}
        for agent in agents:
            agent_id: str = agent["id"]
            try:
                agent_instance: BaseAgent = RESOURCE_MAP[agent_id]["item_cls"]
                agent_registry[agent_id] = {
                    "agent_instance": agent_instance,
                }
            except Exception as e:
                log_context.error(f"Agent {agent_id} is not registered, error: {e}")
                continue

        return agent_registry


class Environment:
    """Environment management for workers and tools.

    This class manages the environment for workers and tools, including
    initialization, state management, and slot filling integration.
    """

    def __init__(
        self,
        tools: list[dict[str, Any]],
        workers: list[dict[str, Any]],
        agents: list[dict[str, Any]],
        nodes: list[dict[str, Any]],
        slotsfillapi: str = "",
        resource_initializer: BaseResourceInitializer | None = None,
        model_service: ModelService | None = None,
        **kwargs: str | int | float | bool | None,
    ) -> None:
        """Initialize the environment.

        Args:
            tools: list of tools to initialize
            workers: list of workers to initialize
            slotsfillapi: API endpoint for slot filling
            resource_initializer: Resource initializer instance
            planner_enabled: Whether planning is enabled
            model_service: Model service for intent detection and slot filling
        """
        # Accept slot_fill_api as an alias for slotsfillapi for compatibility with tests
        if "slot_fill_api" in kwargs and not slotsfillapi:
            slotsfillapi = kwargs["slot_fill_api"]
        resource_initializer = DefaultResourceInitializer()
        self.tools: dict[str, dict[str, Any]] = resource_initializer.init_tools(
            tools, nodes
        )
        self.workers: dict[str, dict[str, Any]] = resource_initializer.init_workers(
            workers
        )
        self.agents: dict[str, dict[str, Any]] = resource_initializer.init_agents(
            agents
        )
        self.model_service = model_service or DummyModelService(
            {
                "model_name": "dummy",
                "api_key": "dummy",
                "endpoint": "http://dummy",
                "model_type_or_path": "dummy-path",
                "llm_provider": "dummy",
            }
        )
        self.slotfillapi: SlotFiller = self.initialize_slotfillapi(slotsfillapi)

    def initialize_slotfillapi(self, slotsfillapi: str) -> SlotFiller:
        """Initialize the slot filling API.

        Args:
            slotsfillapi: API endpoint for slot filling. If not a string or empty,
                         falls back to local model-based slot filling.

        Returns:
            SlotFiller: Initialized slot filler instance, either API-based or local model-based.
        """
        if isinstance(slotsfillapi, str) and slotsfillapi:
            api_service = APIClientService(base_url=slotsfillapi)
            return SlotFiller(model_service=self.model_service, api_service=api_service)
        else:
            return SlotFiller(model_service=self.model_service)

    def step(
        self,
        id: str,
        orch_state: OrchestratorState,
        node_info: NodeInfo,
        dialog_states: dict[str, list[Slot]],
    ) -> tuple[OrchestratorState, NodeResponse]:
        """Execute a step in the environment.

        Args:
            id: Resource ID to execute
            message_state: Current message state
            params: Current parameters
            node_info: Information about the current node

        Returns:
            Tuple containing updated message state and parameters
        """
        node_response: NodeResponse
        if id in self.tools:
            if id == ToolItem.HTTP_TOOL:
                log_context.info(f"HTTP tool {node_info.data.get('name', '')} selected")
                tool: Tool = self.tools[id][node_info.data.get("name", "")][
                    "tool_instance"
                ]
            else:
                log_context.info(f"{id} tool selected")
                tool: Tool = self.tools[id]["tool_instance"]
            tool.init_slotfiller(self.slotfillapi)
            log_context.info(f"orch state: {orch_state}")
            response_state, tool_output = tool.execute(
                orch_state, all_slots=dialog_states, auth=tool.auth
            )
            log_context.info(f"response state: {response_state}")
            response_state.message_flow = tool_output.message_flow
            if id == ToolItem.SHOPIFY_SEARCH_PRODUCTS:
                node_response = NodeResponse(
                    status=tool_output.status,
                    response=tool_output.response,
                    slots=tool_output.slots,
                )
            else:
                node_response = NodeResponse(
                    status=tool_output.status,
                    slots=tool_output.slots,
                )

        elif id in self.workers:
            log_context.info(f"{id} worker selected")
            worker: BaseWorker = self.workers[id]["item_cls"]()
            orch_state, worker_output = worker.execute(
                orch_state, node_specific_data=node_info.data
            )
            content = ""
            if id == WorkerItem.MULTIPLE_CHOICE_WORKER:
                node_response = NodeResponse(
                    status=worker_output.status,
                    response=worker_output.response,
                    choice_list=worker_output.choice_list,
                )
                content = (
                    worker_output.response + "\n" + "\n".join(worker_output.choice_list)
                )
            elif id == WorkerItem.HUMAN_IN_THE_LOOP_WORKER:
                node_response = NodeResponse(
                    status=worker_output.status,
                )
                content = orch_state.message_flow
            else:
                node_response = NodeResponse(
                    status=worker_output.status,
                    response=worker_output.response,
                )
                content = worker_output.response
            call_id: str = str(uuid.uuid4())
            orch_state.function_calling_trajectory.append(
                {
                    "type": "function_call",
                    "id": "fc_" + call_id,
                    "call_id": "call_" + call_id,
                    "name": id,
                    "arguments": "{}",
                }
            )
            orch_state.function_calling_trajectory.append(
                {
                    "type": "function_call_output",
                    "call_id": "call_" + call_id,
                    "output": content,
                }
            )

        elif id in self.agents:
            log_context.info(f"{self.agents[id]} agent selected")
            agent: BaseAgent = self.agents[id]["agent_instance"](
                successors=node_info.successors,
                predecessors=node_info.predecessors,
                tools=self.tools,
            )
            orch_state, agent_output = agent.execute(
                orch_state,
                node_specific_data=node_info.data,
                # log_context.info(f"{self.agents[id]['name']} agent selected")
                # agent_config = self.agents[id].get("config", {})
                # successors = node_info.additional_args.get("successors", [])
                # # Resolve sub-agents if needed
                # if not agent_config and id == "multi_agent":
                #     agent_config = node_info.attributes.copy()
                #     agent_config["sub_agents"] = self.resolve_sub_agents(successors)
                #     self.agents[id]["config"] = agent_config
                # agent: BaseAgent = self.agents[id]["execute"](
                #     successors=node_info.additional_args.get("successors", []),
                #     predecessors=node_info.additional_args.get("predecessors", []),
                #     tools=self.tools,
                #     state=message_state,
                #     **({"multi_agent_config": agent_config} if agent_config else {}),
                # )
                # if agent.is_async():
                #     response_state = asyncio.run(
                #         agent.async_execute(message_state, **node_info.additional_args)
                #     )
                # else:
                #     response_state = agent.execute(
                #         message_state, **node_info.additional_args
                #     )
                # call_id: str = str(uuid.uuid4())
                # params.memory.function_calling_trajectory = (
                #     response_state.function_calling_trajectory
            )
            node_response = NodeResponse(
                status=agent_output.status,
                response=agent_output.response,
            )

        else:
            # Resource not found in any registry, use planner as fallback
            log_context.info(
                f"Resource {id} not found in registries, return orch_state directly"
            )
            node_response = NodeResponse(
                status=StatusEnum.COMPLETE,
            )

        log_context.info(f"Response state from {id}: {orch_state}")
        return orch_state, node_response

    def resolve_sub_agents(self, successors: list[NodeInfo]) -> list[dict[str, Any]]:
        resolved = []

        for node in successors:
            # make sure it is a sub_agent
            if node.type != "agent" and node.attributes.get("type") != "agent":
                continue
            attributes = node.attributes or {}
            name = node.additional_args.get("name", []) or attributes.get(
                "node_specific_data", {}
            ).get("name", "")
            task = attributes.get("task")
            # Find tools for this agent by collecting tools from its predecessors
            tools = []
            for pred in node.additional_args.get("predecessors", []):
                tool_info = self.tools.get(pred.resource_id)
                if tool_info:
                    tool_name = tool_info.get("name", "")
                    tool_id = (
                        tool_name.split(".py-")[-1]
                        if ".py-" in tool_name
                        else tool_name
                    )

                    # Merge tool_info fixed_args with node-specific ones
                    node_specific_args = (
                        pred.attributes.get("node_specific_data", {}) or {}
                    )
                    merged_args = {
                        **(tool_info.get("fixed_args") or {}),
                        **node_specific_args,
                    }

                    tools.append(
                        {
                            "id": tool_id,
                            "path": None
                            if tool_info.get("agent_sdk_tool")
                            else f"{pred.resource_id}.py",
                            "description": tool_info.get("description"),
                            "fixed_args": merged_args,
                        }
                    )

            resolved.append(
                {
                    "name": name,
                    "task": task,
                    "tools": tools,
                }
            )
        return resolved
