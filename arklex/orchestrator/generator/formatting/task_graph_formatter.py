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

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


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

    def format_task_graph(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format a task graph (alias for format_graph for backward compatibility).

        Args:
            tasks (List[Dict[str, Any]]): List of tasks to format

        Returns:
            Dict[str, Any]: Formatted task graph
        """
        nodes = []
        edges = []
        for i, task in enumerate(tasks):
            node_data = {
                "resource": {
                    "id": task.get("task_id", str(i)),
                    "name": task.get("name", ""),
                },
                "attribute": {
                    "value": task.get("description", ""),
                    "task": task.get("name", ""),
                    "directed": True,
                },
            }
            if "limit" in task:
                node_data["limit"] = task["limit"]
            if "type" in task:
                node_data["type"] = task["type"]
            nodes.append([str(i), node_data])
            for dep in task.get("dependencies", []):
                dep_idx = next(
                    (j for j, t in enumerate(tasks) if t.get("task_id") == dep), None
                )
                if dep_idx is not None:
                    source_name = tasks[dep_idx].get("name", "")
                    target_name = task.get("name", "")
                    edge_data = {
                        "intent": self._default_intent,
                        "attribute": {
                            "weight": self._default_weight,
                            "pred": self._default_intent,
                            "definition": f"{target_name} depends on {source_name}",
                            "sample_utterances": [
                                f"I need to complete {source_name} before {target_name}",
                                f"{target_name} requires {source_name} to be done first",
                            ],
                        },
                    }
                    edges.append([str(dep_idx), str(i), edge_data])
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
            "tasks": [],
            "workers": self._workers,
            "tools": self._tools,
            "nluapi": self._nluapi,
            "slotfillapi": self._slotfillapi,
        }
        return graph
