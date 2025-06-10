"""Task graph formatting component for the Arklex framework.

This module provides the TaskGraphFormatter class that handles formatting and
structuring of task graphs. It ensures consistent output format and proper
visualization of task relationships.

Key Features:
- Task graph structure formatting
- Node and edge formatting
- Graph validation and error handling
- Support for hierarchical task organization
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from datetime import datetime

from arklex.orchestrator.generator.formatting.node_formatter import NodeFormatter
from arklex.orchestrator.generator.formatting.edge_formatter import EdgeFormatter
from arklex.orchestrator.generator.formatting.graph_validator import GraphValidator

logger = logging.getLogger(__name__)


class TaskGraphFormatter:
    """Formats and validates task graphs.

    This class handles the formatting and validation of task graphs using
    node, edge, and graph formatters.

    Attributes:
        _node_formatter (NodeFormatter): Formatter for task nodes
        _edge_formatter (EdgeFormatter): Formatter for graph edges
        _graph_validator (GraphValidator): Validator for task graphs
    """

    def __init__(self) -> None:
        """Initialize the TaskGraphFormatter."""
        self._node_formatter = NodeFormatter()
        self._edge_formatter = EdgeFormatter()
        self._graph_validator = GraphValidator()

    def format_graph(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format a task graph.

        Args:
            tasks (List[Dict[str, Any]]): Tasks to format into a graph

        Returns:
            Dict[str, Any]: Formatted task graph

        Raises:
            ValueError: If any task is invalid
        """
        # Format nodes
        nodes = self.format_nodes(tasks)

        # Format edges
        edges = self.format_edges(tasks)

        # Create graph
        graph = {
            "nodes": nodes,
            "edges": edges,
            "metadata": self._format_metadata(nodes, edges),
        }

        # Validate graph
        if not self.validate_graph(graph):
            raise ValueError("Invalid task graph")

        return graph

    def format_task_graph(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format a task graph.

        Args:
            tasks (List[Dict[str, Any]]): List of tasks to format

        Returns:
            Dict[str, Any]: Formatted task graph
        """
        nodes = self.format_nodes(tasks)
        edges = self.format_edges(tasks)
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": self._format_metadata(nodes, edges),
        }

    def format_nodes(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format task nodes.

        Args:
            tasks (List[Dict[str, Any]]): List of tasks to format

        Returns:
            List[Dict[str, Any]]: Formatted nodes
        """
        return [self._node_formatter.format_node(task) for task in tasks]

    def format_edges(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format task edges.

        Args:
            tasks (List[Dict[str, Any]]): List of tasks to format

        Returns:
            List[Dict[str, Any]]: Formatted edges
        """
        edges = []
        for task in tasks:
            for dependency in task.get("dependencies", []):
                # Find the dependency task
                dep_task = next((t for t in tasks if t["task_id"] == dependency), None)
                if dep_task:
                    edge = self._edge_formatter.format_edge(
                        source=dep_task, target=task, type="depends_on"
                    )
                    edges.append(edge)
        return edges

    def validate_graph(self, graph: Dict[str, Any]) -> bool:
        """Validate a task graph.

        Args:
            graph (Dict[str, Any]): Task graph to validate

        Returns:
            bool: True if graph is valid
        """
        return self._graph_validator.validate_graph(graph)

    def get_error_messages(self) -> List[str]:
        """Get validation error messages.

        Returns:
            List[str]: Validation error messages
        """
        return self._graph_validator.get_error_messages()

    def format_and_validate(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format and validate a task graph.

        Args:
            tasks (List[Dict[str, Any]]): Tasks to format into a graph

        Returns:
            Dict[str, Any]: Formatted and validated task graph

        Raises:
            ValueError: If graph is invalid
        """
        graph = self.format_graph(tasks)
        if not self.validate_graph(graph):
            raise ValueError("Invalid task graph")
        return graph

    def _format_metadata(
        self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Format graph metadata.

        Args:
            nodes (List[Dict[str, Any]]): Graph nodes
            edges (List[Dict[str, Any]]): Graph edges

        Returns:
            Dict[str, Any]: Formatted metadata
        """
        return {
            "created_at": datetime.now().isoformat(),
            "modified_at": datetime.now().isoformat(),
            "version": "1.0",
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "max_depth": self._calculate_max_depth(nodes, edges),
        }

    def _calculate_max_depth(
        self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> int:
        """Calculate maximum depth of the task graph.

        Args:
            nodes (List[Dict[str, Any]]): Graph nodes
            edges (List[Dict[str, Any]]): Graph edges

        Returns:
            int: Maximum depth
        """
        # Create adjacency list
        adj_list = {node["id"]: [] for node in nodes}
        for edge in edges:
            adj_list[edge["source"]].append(edge["target"])

        # Calculate depth for each node
        depths = {}
        for node in nodes:
            if node["id"] not in depths:
                self._calculate_node_depth(node["id"], adj_list, depths)

        return max(depths.values()) if depths else 0

    def _calculate_node_depth(
        self, node_id: str, adj_list: Dict[str, List[str]], depths: Dict[str, int]
    ) -> int:
        """Calculate depth of a node.

        Args:
            node_id (str): Node ID
            adj_list (Dict[str, List[str]]): Adjacency list
            depths (Dict[str, int]): Node depths

        Returns:
            int: Node depth
        """
        if node_id in depths:
            return depths[node_id]

        # Calculate depth of dependencies
        max_dep_depth = 0
        for dep in adj_list[node_id]:
            dep_depth = self._calculate_node_depth(dep, adj_list, depths)
            max_dep_depth = max(max_dep_depth, dep_depth)

        # Set node depth
        depths[node_id] = max_dep_depth + 1
        return depths[node_id]

    def build_hierarchy(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build task hierarchy.

        Args:
            tasks (List[Dict[str, Any]]): List of tasks to build hierarchy from

        Returns:
            Dict[str, Any]: Task hierarchy
        """
        # Create adjacency list
        adj_list = {task["task_id"]: [] for task in tasks}
        for task in tasks:
            for dep in task.get("dependencies", []):
                adj_list[dep].append(task["task_id"])

        # Calculate depths
        depths = {}
        for task in tasks:
            if task["task_id"] not in depths:
                self._calculate_node_depth(task["task_id"], adj_list, depths)

        # Group tasks by depth
        levels = {}
        for task_id, depth in depths.items():
            if depth not in levels:
                levels[depth] = []
            levels[depth].append(task_id)

        return {"levels": levels, "max_depth": max(depths.values()) if depths else 0}
