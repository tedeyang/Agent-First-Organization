"""Test suite for the Arklex task graph formatting components.

This module contains comprehensive tests for the task graph formatting,
node formatting, edge formatting, and graph validation components of the
Arklex framework. It includes unit tests for individual components and
integration tests for the complete formatting pipeline.
"""

import pytest
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from arklex.orchestrator.generator.formatting.task_graph_formatter import (
    TaskGraphFormatter,
)
from arklex.orchestrator.generator.formatting.node_formatter import NodeFormatter
from arklex.orchestrator.generator.formatting.edge_formatter import EdgeFormatter
from arklex.orchestrator.generator.formatting.graph_validator import GraphValidator

# Sample test data
SAMPLE_TASKS = [
    {
        "task_id": "task1",
        "name": "Gather product details",
        "description": "Collect all required product information",
        "steps": [{"task": "Get product name"}, {"task": "Get product description"}],
        "dependencies": [],
        "required_resources": ["Product form"],
        "estimated_duration": "30 minutes",
        "priority": 1,
        "level": 0,
    },
    {
        "task_id": "task2",
        "name": "Set product pricing",
        "description": "Determine product pricing strategy",
        "steps": [{"task": "Research market prices"}, {"task": "Set final price"}],
        "dependencies": ["task1"],
        "required_resources": ["Pricing guide"],
        "estimated_duration": "45 minutes",
        "priority": 2,
        "level": 1,
    },
]

SAMPLE_NODE = {
    "id": "task1",
    "type": "task",
    "data": {
        "task_id": "task1",
        "name": "Gather product details",
        "description": "Collect all required product information",
        "steps": [{"task": "Get product name"}, {"task": "Get product description"}],
        "dependencies": [],
        "required_resources": ["Product form"],
        "estimated_duration": "30 minutes",
        "priority": 1,
        "level": 0,
    },
}

SAMPLE_EDGE = {
    "id": "edge1",
    "source": "task1",
    "target": "task2",
    "type": "dependency",
    "data": {"type": "dependency", "description": "Task 2 depends on Task 1"},
}

SAMPLE_GRAPH = {
    "nodes": [SAMPLE_NODE, {"id": "task2", "type": "task", "data": {}}],
    "edges": [SAMPLE_EDGE],
    "metadata": {"version": "1.0", "last_updated": "2024-03-20"},
}

# Additional test data for edge cases
COMPLEX_TASKS = [
    {
        "task_id": "task1",
        "name": "Task 1",
        "description": "Description 1",
        "steps": [],
        "dependencies": [],
        "priority": "high",
    },
    {
        "task_id": "task2",
        "name": "Task 2",
        "description": "Description 2",
        "steps": [],
        "dependencies": ["task1"],
        "priority": "medium",
    },
    {
        "task_id": "task3",
        "name": "Task 3",
        "description": "Description 3",
        "steps": [],
        "dependencies": ["task1", "task2"],
        "priority": "low",
    },
]

INVALID_TASKS = [
    {
        "task_id": "task1",
        "name": "Task 1",
        "description": "Description 1",
        "dependencies": ["nonexistent"],
    },
    {
        "task_id": "task2",
        "name": "Task 2",
        "description": "Description 2",
        "dependencies": ["task1", "task1"],  # Duplicate dependency
    },
]

EMPTY_TASKS = []


@pytest.fixture
def task_graph_formatter():
    """Create a TaskGraphFormatter instance for testing."""
    return TaskGraphFormatter()


@pytest.fixture
def node_formatter():
    """Create a NodeFormatter instance for testing."""
    return NodeFormatter()


@pytest.fixture
def edge_formatter():
    """Create an EdgeFormatter instance for testing."""
    return EdgeFormatter()


@pytest.fixture
def graph_validator():
    """Create a GraphValidator instance for testing."""
    return GraphValidator()


class TestTaskGraphFormatter:
    """Test suite for the TaskGraphFormatter class."""

    def test_format_task_graph(self, task_graph_formatter):
        """Test task graph formatting."""
        formatted_graph = task_graph_formatter.format_task_graph(SAMPLE_TASKS)
        assert isinstance(formatted_graph, dict)
        assert "nodes" in formatted_graph
        assert "edges" in formatted_graph
        assert len(formatted_graph["nodes"]) == len(SAMPLE_TASKS)
        assert len(formatted_graph["edges"]) > 0

    def test_format_task_graph_with_complex_tasks(self, task_graph_formatter):
        """Test task graph formatting with complex task dependencies."""
        formatted_graph = task_graph_formatter.format_task_graph(COMPLEX_TASKS)
        assert len(formatted_graph["nodes"]) == len(COMPLEX_TASKS)
        assert (
            len(formatted_graph["edges"]) == 3
        )  # task1->task2, task1->task3, task2->task3

    def test_format_task_graph_with_empty_tasks(self, task_graph_formatter):
        """Test task graph formatting with empty task list."""
        formatted_graph = task_graph_formatter.format_task_graph(EMPTY_TASKS)
        assert len(formatted_graph["nodes"]) == 0
        assert len(formatted_graph["edges"]) == 0

    def test_format_task_graph_with_invalid_tasks(self, task_graph_formatter):
        """Test task graph formatting with invalid task dependencies."""
        formatted_graph = task_graph_formatter.format_task_graph(INVALID_TASKS)
        assert len(formatted_graph["nodes"]) == len(INVALID_TASKS)
        assert (
            len(formatted_graph["edges"]) == 2
        )  # Both dependencies should be included

    def test_format_nodes(self, task_graph_formatter):
        """Test node formatting."""
        formatted_nodes = task_graph_formatter.format_nodes(SAMPLE_TASKS)
        assert isinstance(formatted_nodes, list)
        assert len(formatted_nodes) == len(SAMPLE_TASKS)
        assert all("id" in node for node in formatted_nodes)
        assert all("type" in node for node in formatted_nodes)
        assert all("data" in node for node in formatted_nodes)

    def test_format_nodes_with_priority(self, task_graph_formatter):
        """Test node formatting with different priorities."""
        formatted_nodes = task_graph_formatter.format_nodes(COMPLEX_TASKS)
        priorities = {node["data"]["priority"] for node in formatted_nodes}
        assert priorities == {"high", "medium", "low"}

    def test_format_edges(self, task_graph_formatter):
        """Test edge formatting."""
        formatted_edges = task_graph_formatter.format_edges(SAMPLE_TASKS)
        assert isinstance(formatted_edges, list)
        assert len(formatted_edges) > 0
        assert all("id" in edge for edge in formatted_edges)
        assert all("source" in edge for edge in formatted_edges)
        assert all("target" in edge for edge in formatted_edges)

    def test_format_edges_with_multiple_dependencies(self, task_graph_formatter):
        """Test edge formatting with multiple dependencies."""
        formatted_edges = task_graph_formatter.format_edges(COMPLEX_TASKS)
        assert len(formatted_edges) == 3
        sources = {edge["source"] for edge in formatted_edges}
        targets = {edge["target"] for edge in formatted_edges}
        assert "task1" in sources
        assert "task2" in sources
        assert "task2" in targets
        assert "task3" in targets

    def test_build_hierarchy(self, task_graph_formatter):
        """Test hierarchy building."""
        hierarchy = task_graph_formatter.build_hierarchy(SAMPLE_TASKS)
        assert isinstance(hierarchy, dict)
        assert "levels" in hierarchy
        assert len(hierarchy["levels"]) > 0

    def test_build_hierarchy_with_complex_tasks(self, task_graph_formatter):
        """Test hierarchy building with complex task dependencies."""
        hierarchy = task_graph_formatter.build_hierarchy(COMPLEX_TASKS)
        assert len(hierarchy["levels"]) == 4
        assert "task1" in hierarchy["levels"][4]
        assert "task2" in hierarchy["levels"][3]
        assert "task3" in hierarchy["levels"][2]


class TestNodeFormatter:
    """Test suite for the NodeFormatter class."""

    def test_format_node(self, node_formatter):
        """Test node formatting."""
        formatted_node = node_formatter.format_node(SAMPLE_TASKS[0])
        assert isinstance(formatted_node, dict)
        assert "id" in formatted_node
        assert "type" in formatted_node
        assert "data" in formatted_node
        assert formatted_node["id"] == SAMPLE_TASKS[0]["task_id"]

    def test_format_node_data(self, node_formatter):
        """Test node data formatting."""
        formatted_data = node_formatter.format_node_data(SAMPLE_TASKS[0])
        assert isinstance(formatted_data, dict)
        assert "task_id" in formatted_data
        assert "name" in formatted_data
        assert "description" in formatted_data
        assert "steps" in formatted_data

    def test_format_node_style(self, node_formatter):
        """Test node style formatting."""
        style = node_formatter.format_node_style(SAMPLE_TASKS[0])
        assert isinstance(style, dict)
        assert "color" in style
        assert "border" in style
        assert "padding" in style

    def test_format_node_with_missing_fields(self, node_formatter):
        """Test node formatting with missing fields."""
        incomplete_task = {"task_id": "t1"}  # Missing name, description, etc.
        node = node_formatter.format_node(incomplete_task)
        assert "id" in node and node["id"] == "t1"
        assert "data" in node

    def test_format_node_data_with_extra_fields(self, node_formatter):
        """Test node data formatting with extra fields."""
        extra_task = {
            "task_id": "t2",
            "name": "n",
            "description": "d",
            "steps": [],
            "extra": 123,
        }
        data = node_formatter.format_node_data(extra_task)
        assert "task_id" in data and data["task_id"] == "t2"
        assert "extra" not in data  # Should not include unexpected fields

    def test_format_node_style_with_different_priorities(self, node_formatter):
        """Test node style formatting with different priorities."""
        for priority in ["high", "medium", "low"]:
            task = {**SAMPLE_TASKS[0], "priority": priority}
            style = node_formatter.format_node_style(task)
            assert "color" in style
            assert style["color"] != "#808080"  # Should not be default gray


class TestEdgeFormatter:
    """Test suite for the EdgeFormatter class."""

    def test_format_edge(self, edge_formatter):
        """Test edge formatting."""
        formatted_edge = edge_formatter.format_edge(
            source=SAMPLE_TASKS[0], target=SAMPLE_TASKS[1]
        )
        assert isinstance(formatted_edge, dict)
        assert "id" in formatted_edge
        assert "source" in formatted_edge
        assert "target" in formatted_edge
        assert "type" in formatted_edge
        assert "data" in formatted_edge

    def test_format_edge_data(self, edge_formatter):
        """Test edge data formatting."""
        formatted_data = edge_formatter.format_edge_data(
            source=SAMPLE_TASKS[0], target=SAMPLE_TASKS[1]
        )
        assert isinstance(formatted_data, dict)
        assert "type" in formatted_data
        assert "description" in formatted_data

    def test_format_edge_style(self, edge_formatter):
        """Test edge style formatting."""
        style = edge_formatter.format_edge_style(
            source=SAMPLE_TASKS[0], target=SAMPLE_TASKS[1]
        )
        assert isinstance(style, dict)
        assert "color" in style
        assert "width" in style
        assert "style" in style

    def test_format_edge_with_custom_type(self, edge_formatter):
        """Test edge formatting with custom type."""
        edge = edge_formatter.format_edge(
            source=SAMPLE_TASKS[0],
            target=SAMPLE_TASKS[1],
            type="custom_type",
            weight=2.0,
            label="Custom Label",
        )
        assert edge["type"] == "custom_type"
        assert edge["weight"] == 2.0
        assert edge["label"] == "Custom Label"

    def test_format_edge_with_metadata(self, edge_formatter):
        """Test edge formatting with metadata."""
        metadata = {"custom_field": "value"}
        edge = edge_formatter.format_edge(
            source=SAMPLE_TASKS[0],
            target=SAMPLE_TASKS[1],
            metadata=metadata,
        )
        assert edge["metadata"] == metadata


class TestGraphValidator:
    """Test suite for the GraphValidator class."""

    def test_validate_graph(self, graph_validator):
        """Test graph validation."""
        assert graph_validator.validate_graph(SAMPLE_GRAPH)
        assert not graph_validator.get_validation_errors(SAMPLE_GRAPH)

    def test_validate_nodes(self, graph_validator):
        """Test node validation."""
        assert graph_validator.validate_nodes(SAMPLE_GRAPH["nodes"])
        assert not graph_validator.get_validation_errors(SAMPLE_GRAPH)

    def test_validate_edges(self, graph_validator):
        """Test edge validation."""
        assert graph_validator.validate_edges(SAMPLE_GRAPH["edges"])
        assert not graph_validator.get_validation_errors(SAMPLE_GRAPH)

    def test_validate_connectivity(self, graph_validator):
        """Test graph connectivity validation."""
        assert graph_validator.validate_connectivity(SAMPLE_GRAPH)

    def test_validate_graph_with_missing_nodes(self, graph_validator):
        """Test graph validation with missing nodes."""
        invalid_graph = {"edges": SAMPLE_GRAPH["edges"]}
        assert not graph_validator.validate_graph(invalid_graph)
        assert "Invalid nodes" in graph_validator.get_validation_errors(invalid_graph)

    def test_validate_graph_with_missing_edges(self, graph_validator):
        """Test graph validation with missing edges."""
        graph = {"nodes": SAMPLE_GRAPH["nodes"]}
        assert not graph_validator.validate_graph(graph)
        assert "Invalid edges" in graph_validator.get_validation_errors(graph)

    def test_validate_graph_with_duplicate_node_ids(self, graph_validator):
        """Test graph validation with duplicate node IDs."""
        invalid_graph = {
            "nodes": [
                {"id": "node1", "type": "task", "data": {}},
                {"id": "node1", "type": "task", "data": {}},
            ],
            "edges": [],
        }
        assert not graph_validator.validate_graph(invalid_graph)
        assert "Invalid nodes" in graph_validator.get_validation_errors(invalid_graph)

    def test_validate_graph_with_duplicate_edge_ids(self, graph_validator):
        """Test graph validation with duplicate edge IDs."""
        invalid_graph = {
            "nodes": [
                {"id": "node1", "type": "task", "data": {}},
                {"id": "node2", "type": "task", "data": {}},
            ],
            "edges": [
                {
                    "id": "edge1",
                    "source": "node1",
                    "target": "node2",
                    "type": "dependency",
                },
                {
                    "id": "edge1",
                    "source": "node1",
                    "target": "node2",
                    "type": "dependency",
                },
            ],
        }
        assert not graph_validator.validate_graph(invalid_graph)
        assert "Invalid edges" in graph_validator.get_validation_errors(invalid_graph)

    def test_validate_graph_with_invalid_edge_references(self, graph_validator):
        """Test graph validation with invalid edge references."""
        invalid_graph = {
            "nodes": [{"id": "node1", "type": "task", "data": {}}],
            "edges": [
                {
                    "id": "edge1",
                    "source": "node1",
                    "target": "nonexistent",
                    "type": "dependency",
                }
            ],
        }
        assert not graph_validator.validate_graph(invalid_graph)
        assert "Invalid edge references" in graph_validator.get_validation_errors(
            invalid_graph
        )


def test_integration_formatting_pipeline():
    """Test the complete task graph formatting pipeline integration."""
    # Initialize components
    task_graph_formatter = TaskGraphFormatter()
    node_formatter = NodeFormatter()
    edge_formatter = EdgeFormatter()
    graph_validator = GraphValidator()

    # Format task graph
    formatted_graph = task_graph_formatter.format_task_graph(SAMPLE_TASKS)
    assert isinstance(formatted_graph, dict)
    assert "nodes" in formatted_graph
    assert "edges" in formatted_graph

    # Format nodes
    formatted_nodes = task_graph_formatter.format_nodes(SAMPLE_TASKS)
    assert isinstance(formatted_nodes, list)
    assert len(formatted_nodes) == len(SAMPLE_TASKS)

    # Format edges
    formatted_edges = task_graph_formatter.format_edges(SAMPLE_TASKS)
    assert isinstance(formatted_edges, list)
    assert len(formatted_edges) > 0

    # Validate graph
    is_valid = graph_validator.validate_graph(formatted_graph)
    assert is_valid

    # Verify integration
    assert all("id" in node for node in formatted_nodes)
    assert all("source" in edge for edge in formatted_edges)
    assert all("target" in edge for edge in formatted_edges)
    assert graph_validator.validate_connectivity(formatted_graph)
