"""Task graph formatting component for the Arklex framework.

This module provides the TaskGraphFormatter class that handles formatting and
structuring of task graphs. It ensures consistent graph representation and
proper visualization of task relationships.

Key Features:
- Graph structure formatting
- Node formatting
- Edge formatting
- Error handling
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union

from arklex.orchestrator.generator.formatting.edge_formatter import EdgeFormatter
from arklex.orchestrator.generator.formatting.node_formatter import NodeFormatter
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class TaskGraphFormatter:
    """Formats task graphs.

    This class handles formatting of task graphs, ensuring they are properly
    structured and contain all required components.

    Attributes:
        _role (str): Role of the assistant
        _user_objective (str): User's objective
        _builder_objective (str): Builder's objective
        _domain (str): Domain of the task
        _intro (str): Introduction text
        _task_docs (List[Dict[str, Any]]): Task documentation
        _rag_docs (List[Dict[str, Any]]): RAG documentation
        _workers (List[Dict[str, Any]]): List of workers
    """

    def __init__(
        self,
        role: str = "",
        user_objective: str = "",
        builder_objective: str = "",
        domain: str = "",
        intro: str = "",
        task_docs: Optional[List[Dict[str, Any]]] = None,
        rag_docs: Optional[List[Dict[str, Any]]] = None,
        workers: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        nluapi: str = "",
        slotfillapi: str = "",
        default_intent: str = "depends_on",
        default_weight: int = 1,
        nodes: Optional[List[Any]] = None,
        edges: Optional[List[Any]] = None,
    ) -> None:
        """Initialize the TaskGraphFormatter.

        Args:
            role (str): Role of the assistant
            user_objective (str): User's objective
            builder_objective (str): Builder's objective
            domain (str): Domain of the task
            intro (str): Introduction text
            task_docs (Optional[List[Dict[str, Any]]]): Task documentation
            rag_docs (Optional[List[Dict[str, Any]]]): RAG documentation
            workers (Optional[List[Dict[str, Any]]]): List of workers
            tools (Optional[List[Dict[str, Any]]]): List of tools
            nluapi (str): NLU API
            slotfillapi (str): Slotfill API
            default_intent (str): Default intent for edges
            default_weight (int): Default weight for edges
            nodes (Optional[List[Any]]): List of nodes
            edges (Optional[List[Any]]): List of edges
        """
        self._role = role
        self._user_objective = user_objective
        self._builder_objective = builder_objective
        self._domain = domain
        self._intro = intro
        self._task_docs = task_docs or []
        self._rag_docs = rag_docs or []
        self._workers = workers or []
        self._tools = tools or []
        self._nluapi = nluapi
        self._slotfillapi = slotfillapi
        self._default_intent = default_intent
        self._default_weight = default_weight
        self._nodes = nodes
        self._edges = edges

    def format_task_graph(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format a task graph using config nodes/edges if present, otherwise generate from tasks."""
        if self._nodes is not None and self._edges is not None:
            # Use nodes and edges from config directly
            graph = {
                "nodes": self._nodes,
                "edges": self._edges,
                "role": self._role,
                "user_objective": self._user_objective,
                "builder_objective": self._builder_objective,
                "domain": self._domain,
                "intro": self._intro,
                "task_docs": self._task_docs,
                "rag_docs": self._rag_docs,
                "tasks": tasks,
                "workers": self._workers,
                "tools": self._tools,
                "nluapi": self._nluapi,
                "slotfillapi": self._slotfillapi,
            }
            return graph
        # Otherwise, fall back to generating from tasks
        nodes = []
        edges = []
        node_id_counter = 0
        node_lookup = {}  # task_id/step_id -> node_id (str)
        step_parent_lookup = {}  # step node_id -> parent task node_id
        step_nodes = []

        # Add start node if there are tasks
        if tasks:
            start_node = {
                "resource": {
                    "id": "start_message_worker",
                    "name": "MessageWorker",
                },
                "attribute": {
                    "value": "Hello! I'm here to assist you with any customer service inquiries you may have. Whether you need information about our products, services, or policies, or if you need help resolving an issue or completing a transaction, feel free to ask. How can I assist you today?",
                    "task": "start message",
                    "directed": False,
                },
                "limit": 1,
                "type": "start",
            }
            nodes.append([str(node_id_counter), start_node])
            start_node_id = str(node_id_counter)
            node_id_counter += 1
        else:
            start_node_id = None

        # First pass: create nodes for tasks
        for task in tasks:
            resource = task.get("resource", {})
            node = {
                "resource": {
                    "id": task.get("task_id", str(node_id_counter)),
                    "name": resource.get("name", task.get("name", "")),
                },
                "attribute": {
                    "value": task.get("description", ""),
                    "task": task.get("name", ""),
                    "directed": False,
                },
            }
            if "limit" in task:
                node["limit"] = task["limit"]
            if "type" in task:
                node["type"] = task["type"]
            nodes.append([str(node_id_counter), node])
            node_lookup[task.get("task_id", str(node_id_counter))] = str(
                node_id_counter
            )
            task["_node_id"] = str(node_id_counter)  # for step parent lookup
            node_id_counter += 1

        # Second pass: create nodes for steps
        for task in tasks:
            steps = task.get("steps", [])
            for idx, step in enumerate(steps):
                step_id = f"{task.get('task_id', 'task')}_step{idx}"
                resource = step.get("resource", {})
                step_node = {
                    "resource": {
                        "id": resource.get("id", "step_message_worker"),
                        "name": resource.get("name", "MessageWorker"),
                    },
                    "attribute": {
                        "value": step.get("description", step.get("value", "")),
                        "task": step.get("name", step.get("task", "")),
                        "directed": False,
                    },
                }
                nodes.append([str(node_id_counter), step_node])
                node_lookup[step_id] = str(node_id_counter)
                step_parent_lookup[str(node_id_counter)] = task["_node_id"]
                step_nodes.append((str(node_id_counter), step_id, task["_node_id"]))
                node_id_counter += 1

        # Edges: dependencies (task-to-task)
        for task in tasks:
            this_node_id = task["_node_id"]
            dependencies = task.get("dependencies", [])
            if dependencies:
                for dep in dependencies:
                    if dep in node_lookup:
                        edge_data = {
                            "intent": "depends_on",
                            "attribute": {
                                "weight": 1,
                                "pred": True,
                                "definition": f"{task.get('name', '')} depends on {dep}",
                                "sample_utterances": [],
                            },
                        }
                        edges.append([node_lookup[dep], this_node_id, edge_data])
            elif start_node_id is not None:
                # No dependencies: connect from start node
                edge_data = {
                    "intent": f"User inquires about {task.get('name', '').lower()}",
                    "attribute": {
                        "weight": 1,
                        "pred": True,
                        "definition": "",
                        "sample_utterances": [],
                    },
                }
                edges.append([start_node_id, this_node_id, edge_data])

        # Edges: task-to-step
        for task in tasks:
            steps = task.get("steps", [])
            if not steps:
                continue

            # Connect first step to parent task
            first_step_node_id = node_lookup[f"{task.get('task_id', 'task')}_step0"]
            edge_data = {
                "intent": "step",
                "attribute": {
                    "weight": 1,
                    "pred": False,
                    "definition": "",
                    "sample_utterances": [],
                },
            }
            edges.append([task["_node_id"], first_step_node_id, edge_data])

            # Connect subsequent steps sequentially
            for i in range(len(steps) - 1):
                current_step_id = f"{task.get('task_id', 'task')}_step{i}"
                next_step_id = f"{task.get('task_id', 'task')}_step{i + 1}"
                current_step_node_id = node_lookup[current_step_id]
                next_step_node_id = node_lookup[next_step_id]

                edge_data = {
                    "intent": "next_step",
                    "attribute": {
                        "weight": 1,
                        "pred": False,
                        "definition": "",
                        "sample_utterances": [],
                    },
                }
                edges.append([current_step_node_id, next_step_node_id, edge_data])

        graph = {
            "nodes": nodes,
            "edges": edges,
            "role": self._role,
            "user_objective": self._user_objective,
            "builder_objective": self._builder_objective,
            "domain": self._domain,
            "intro": self._intro,
            "task_docs": self._task_docs,
            "rag_docs": self._rag_docs,
            "tasks": tasks,
            "workers": self._workers,
            "tools": self._tools,
            "nluapi": self._nluapi,
            "slotfillapi": self._slotfillapi,
        }
        return graph
