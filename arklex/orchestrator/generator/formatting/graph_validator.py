"""Graph validation component for the Arklex framework.

This module provides the GraphValidator class that handles the validation
of task graph structure, connectivity, and consistency.
"""

from typing import Dict, Any, List, Set, Optional
from .node_formatter import NodeFormatter
from .edge_formatter import EdgeFormatter


class GraphValidator:
    """Validator for task graph structure and connectivity."""

    def __init__(self) -> None:
        """Initialize the graph validator."""
        self._node_formatter = NodeFormatter()
        self._edge_formatter = EdgeFormatter()
        self._errors: List[str] = []

    def validate_graph(self, graph: Dict[str, Any]) -> bool:
        """Validate a task graph.

        Args:
            graph (Dict[str, Any]): Graph to validate

        Returns:
            bool: True if graph is valid
        """
        self._errors = []
        is_valid = True

        # Validate nodes
        if not self.validate_nodes(graph.get("nodes", [])):
            is_valid = False
            self._errors.append("Invalid nodes")

        # Validate edges
        if not self.validate_edges(graph.get("edges", [])):
            is_valid = False
            self._errors.append("Invalid edges")

        # Validate connectivity
        if not self.validate_connectivity(graph):
            is_valid = False
            self._errors.append("Invalid graph connectivity")

        # Check for invalid edge references
        node_ids = {node["id"] for node in graph.get("nodes", [])}
        for edge in graph.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if source not in node_ids or target not in node_ids:
                is_valid = False
                self._errors.append("Invalid edge references")
                break

        return is_valid

    def validate_nodes(self, nodes: List[Dict[str, Any]]) -> bool:
        """Validate graph nodes.

        Args:
            nodes (List[Dict[str, Any]]): Nodes to validate

        Returns:
            bool: True if nodes are valid
        """
        if not nodes:
            return False

        # Check for duplicate node IDs
        node_ids = set()
        for node in nodes:
            if not isinstance(node, dict):
                return False
            if "id" not in node:
                return False
            if node["id"] in node_ids:
                return False
            node_ids.add(node["id"])

            # Validate node structure
            if not self._node_formatter.validate_node(node):
                return False

        return True

    def validate_edges(self, edges: List[Dict[str, Any]]) -> bool:
        """Validate graph edges.

        Args:
            edges (List[Dict[str, Any]]): Edges to validate

        Returns:
            bool: True if edges are valid
        """
        if not edges:
            return False

        # Check for duplicate edge IDs
        edge_ids = set()
        for edge in edges:
            if not isinstance(edge, dict):
                return False
            if "id" not in edge:
                return False
            if edge["id"] in edge_ids:
                return False
            edge_ids.add(edge["id"])

            # Validate edge structure
            if not self._edge_formatter.validate_edge(
                source=edge.get("source"),
                target=edge.get("target"),
                type=edge.get("type", "dependency"),
            ):
                return False

        return True

    def validate_connectivity(self, graph: Dict[str, Any]) -> bool:
        """Validate graph connectivity.

        Args:
            graph (Dict[str, Any]): Graph to validate

        Returns:
            bool: True if graph is valid
        """
        if not isinstance(graph, dict):
            return False

        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        if not nodes:
            return False
        if len(nodes) == 1:
            return True
        if not edges:
            return False

        # Create set of valid node IDs
        node_ids = {node["id"] for node in nodes}

        # Validate all edge references
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if not source or not target:
                return False
            if source not in node_ids or target not in node_ids:
                return False

        # Check if graph is connected
        if len(nodes) > 1 and not edges:
            return False

        return True

    def get_validation_errors(self, graph: Dict[str, Any]) -> List[str]:
        """Get validation errors for a graph.

        Args:
            graph (Dict[str, Any]): Graph to validate

        Returns:
            List[str]: List of validation errors
        """
        errors = []
        if not self.validate_nodes(graph.get("nodes", [])):
            errors.append("Invalid nodes")
        if not self.validate_edges(graph.get("edges", [])):
            errors.append("Invalid edges")
        if not self.validate_connectivity(graph):
            errors.append("Invalid graph connectivity")
        # Check for invalid edge references
        node_ids = {node["id"] for node in graph.get("nodes", [])}
        for edge in graph.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if source not in node_ids or target not in node_ids:
                errors.append("Invalid edge references")
                break
        return errors

    def get_error_messages(self, graph: Dict[str, Any]) -> List[str]:
        """Get validation error messages.

        Args:
            graph (Dict[str, Any]): The graph to validate

        Returns:
            List[str]: List of error messages
        """
        errors = []

        # Check structure
        if not self._validate_structure(graph):
            errors.append("Invalid graph structure")

        # Check nodes
        node_errors = self._get_node_errors(graph.get("nodes", []))
        errors.extend(node_errors)

        # Check edges
        edge_errors = self._get_edge_errors(graph.get("edges", []))
        errors.extend(edge_errors)

        # Check connectivity
        if not self._validate_connectivity(graph):
            errors.append("Graph connectivity issues")

        return errors

    def _validate_structure(self, graph: Dict[str, Any]) -> bool:
        """Validate graph structure.

        Args:
            graph (Dict[str, Any]): The graph to validate

        Returns:
            bool: True if structure is valid
        """
        # Check required fields
        if "nodes" not in graph or "edges" not in graph:
            return False

        # Check types
        if not isinstance(graph["nodes"], list):
            return False
        if not isinstance(graph["edges"], list):
            return False

        return True

    def _validate_nodes(self, nodes: List[Dict[str, Any]]) -> bool:
        """Validate graph nodes.

        Args:
            nodes (List[Dict[str, Any]]): The nodes to validate

        Returns:
            bool: True if nodes are valid
        """
        # Check each node
        for node in nodes:
            if not self._node_formatter.validate_node(node):
                return False

        # Check for duplicate node IDs
        node_ids = [node.get("id", "") for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            return False

        return True

    def _validate_edges(self, edges: List[Dict[str, Any]]) -> bool:
        """Validate graph edges.

        Args:
            edges (List[Dict[str, Any]]): The edges to validate

        Returns:
            bool: True if edges are valid
        """
        # Check each edge
        for edge in edges:
            if not self._edge_formatter.validate_edge(edge):
                return False

        # Check for duplicate edges
        edge_keys = [(edge.get("source", ""), edge.get("target", "")) for edge in edges]
        if len(edge_keys) != len(set(edge_keys)):
            return False

        return True

    def _validate_connectivity(self, graph: Dict[str, Any]) -> bool:
        """Validate graph connectivity.

        Args:
            graph (Dict[str, Any]): The graph to validate

        Returns:
            bool: True if graph is connected
        """
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        # Create node set
        node_ids = {node.get("id", "") for node in nodes}

        # Create adjacency list
        adjacency = {node_id: set() for node_id in node_ids}
        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            if source in node_ids and target in node_ids:
                adjacency[source].add(target)

        # Check for isolated nodes
        for node_id in node_ids:
            if not adjacency[node_id] and not any(
                node_id in adj for adj in adjacency.values()
            ):
                return False

        # Check for cycles
        visited = set()
        path = set()

        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            path.add(node_id)

            for neighbor in adjacency[node_id]:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in path:
                    return True

            path.remove(node_id)
            return False

        for node_id in node_ids:
            if node_id not in visited:
                if has_cycle(node_id):
                    return False

        return True

    def _get_node_errors(self, nodes: List[Dict[str, Any]]) -> List[str]:
        """Get node validation errors.

        Args:
            nodes (List[Dict[str, Any]]): The nodes to validate

        Returns:
            List[str]: List of error messages
        """
        errors = []

        # Check each node
        for i, node in enumerate(nodes):
            node_errors = self._node_formatter.get_error_messages(node)
            if node_errors:
                errors.extend([f"Node {i}: {error}" for error in node_errors])

        # Check for duplicate node IDs
        node_ids = [node.get("id", "") for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            errors.append("Duplicate node IDs found")

        return errors

    def _get_edge_errors(self, edges: List[Dict[str, Any]]) -> List[str]:
        """Get edge validation errors.

        Args:
            edges (List[Dict[str, Any]]): The edges to validate

        Returns:
            List[str]: List of error messages
        """
        errors = []

        # Check each edge
        for i, edge in enumerate(edges):
            edge_errors = self._edge_formatter.get_error_messages(edge)
            if edge_errors:
                errors.extend([f"Edge {i}: {error}" for error in edge_errors])

        # Check for duplicate edges
        edge_keys = [(edge.get("source", ""), edge.get("target", "")) for edge in edges]
        if len(edge_keys) != len(set(edge_keys)):
            errors.append("Duplicate edges found")

        return errors
