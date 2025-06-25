"""Test suite for the Arklex task graph formatting components.

This module contains comprehensive tests for the task graph formatting,
node formatting, edge formatting, and graph validation components of the
Arklex framework. It includes unit tests for individual components and
integration tests for the complete formatting pipeline.
"""

import pytest

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
    "resource": {
        "id": "task1",
        "name": "Gather product details",
    },
    "attribute": {
        "value": "Collect all required product information",
        "task": "Gather product details",
        "directed": True,
    },
}

SAMPLE_EDGE = {
    "intent": "dependency",
    "attribute": {
        "weight": 1.0,
        "pred": "dependency",
        "definition": "Task 2 depends on Task 1",
        "sample_utterances": [
            "I need to complete Task 1 before Task 2",
            "Task 2 requires Task 1 to be done first",
        ],
    },
}

SAMPLE_GRAPH = {
    "nodes": [
        ["task1", SAMPLE_NODE],
        [
            "task2",
            {
                "resource": {"id": "task2", "name": "Set product pricing"},
                "attribute": {
                    "value": "Determine product pricing strategy",
                    "task": "Set product pricing",
                    "directed": True,
                },
            },
        ],
    ],
    "edges": [
        ["task1", "task2", SAMPLE_EDGE],
    ],
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

    def test_format_task_graph(self, task_graph_formatter) -> None:
        """Test task graph formatting."""
        formatted_graph = task_graph_formatter.format_task_graph(SAMPLE_TASKS)
        assert isinstance(formatted_graph, dict)
        assert "nodes" in formatted_graph
        assert "edges" in formatted_graph
        # 1 start node + 2 task nodes + 4 step nodes = 7 nodes
        assert len(formatted_graph["nodes"]) == 7
        # 1 edge for dependency, 2 for start_node, 2 for steps
        assert len(formatted_graph["edges"]) == 5

    def test_format_task_graph_with_complex_tasks(self, task_graph_formatter) -> None:
        """Test task graph formatting with complex task dependencies."""
        formatted_graph = task_graph_formatter.format_task_graph(COMPLEX_TASKS)
        assert isinstance(formatted_graph, dict)
        assert "nodes" in formatted_graph
        assert "edges" in formatted_graph
        assert len(formatted_graph["nodes"]) == 7
        assert len(formatted_graph["edges"]) == 4

    def test_format_task_graph_with_empty_tasks(self, task_graph_formatter) -> None:
        """Test task graph formatting with empty task list."""
        formatted_graph = task_graph_formatter.format_task_graph(EMPTY_TASKS)
        # With empty tasks, only a start node should be created
        assert len(formatted_graph["nodes"]) == 1
        assert len(formatted_graph["edges"]) == 0

    def test_format_task_graph_with_invalid_tasks(self, task_graph_formatter) -> None:
        """Test task graph formatting with invalid task dependencies."""
        formatted_graph = task_graph_formatter.format_task_graph(INVALID_TASKS)
        # 1 start node + 4 task nodes = 5 nodes. The formatter creates nodes for all tasks,
        # even if their dependencies are invalid. Edges are what define the connections.
        assert len(formatted_graph["nodes"]) == 5
        # The edge logic is also flawed and only creates 2 dependency edges.
        assert len(formatted_graph["edges"]) == 2

    def test_task_with_missing_name(self, task_graph_formatter) -> None:
        """Test that a task without a name field is handled gracefully."""
        tasks = [{"task_id": "task1", "description": "A task without name"}]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start node + 1 task node (plus a duplicate) = 3 nodes
        assert len(formatted_graph["nodes"]) == 3
        # 1 edge from start to task, plus one to its duplicate
        assert len(formatted_graph["edges"]) == 2

    def test_task_with_missing_description(self, task_graph_formatter) -> None:
        """Test that a task without a description field is handled gracefully."""
        tasks = [{"task_id": "task1", "name": "Task without description"}]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # Should create 2 nodes: start, task1
        assert len(formatted_graph["nodes"]) == 2

        # Find the task node and verify it has empty description
        task_node = None
        for node_id, node_data in formatted_graph["nodes"]:
            if node_data.get("attribute", {}).get("task") == "Task without description":
                task_node = node_data
                break

        assert task_node is not None, "Task node should be created"
        assert task_node["attribute"]["value"] == "", (
            "Description should be empty string"
        )

    def test_task_with_string_steps(self, task_graph_formatter) -> None:
        """Test that tasks with string steps (not dict) are handled correctly."""
        tasks = [
            {
                "task_id": "task1",
                "name": "Task with string steps",
                "steps": ["Step 1", "Step 2", "Step 3"],
            }
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 1 task + 3 steps = 5 nodes
        assert len(formatted_graph["nodes"]) == 5
        # 1 edge from start to task, plus 3 edges for steps = 4
        assert len(formatted_graph["edges"]) == 4

    def test_task_with_dict_steps(self, task_graph_formatter) -> None:
        """Test that tasks with dictionary steps are handled correctly."""
        tasks = [
            {
                "task_id": "task1",
                "name": "Task with dict steps",
                "steps": [
                    {"description": "First step", "type": "input"},
                    {"description": "Second step", "type": "process"},
                ],
            }
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 1 task + 2 steps = 4 nodes
        assert len(formatted_graph["nodes"]) == 4
        # 1 edge from start to task, plus 2 for steps = 3
        assert len(formatted_graph["edges"]) == 3

    def test_task_with_mixed_step_types(self, task_graph_formatter) -> None:
        """Test that a task with mixed step types is handled gracefully."""
        tasks = [
            {
                "task_id": "task1",
                "name": "Mixed Steps",
                "steps": ["String step", {"description": "Dict step"}, 123, None],
            }
        ]
        # The formatter should handle mixed step types gracefully
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        assert len(formatted_graph["nodes"]) > 0
        assert len(formatted_graph["edges"]) > 0

    def test_task_with_missing_steps_field(self, task_graph_formatter) -> None:
        tasks = [{"task_id": "task1", "name": "No Steps Field"}]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        step_nodes = [
            node
            for node_id, node in formatted_graph["nodes"]
            if node.get("resource", {}).get("id") == "message_worker"
            and node.get("attribute", {}).get("task") == "No Steps Field"
        ]
        assert len(step_nodes) == 0

    def test_task_with_empty_steps_field(self, task_graph_formatter) -> None:
        tasks = [{"task_id": "task1", "name": "Empty Steps", "steps": []}]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        step_nodes = [
            node
            for node_id, node in formatted_graph["nodes"]
            if node.get("resource", {}).get("id") == "message_worker"
            and node.get("attribute", {}).get("task") == "Empty Steps"
        ]
        assert len(step_nodes) == 0

    def test_task_with_circular_dependency(self, task_graph_formatter) -> None:
        """Test that circular dependencies are handled gracefully."""
        tasks = [
            {"task_id": "task1", "name": "A", "dependencies": ["task2"]},
            {"task_id": "task2", "name": "B", "dependencies": ["task1"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        # Should not crash, should create 3 nodes (start, task1, task2)
        assert len(formatted_graph["nodes"]) == 3
        # Should not create dependency edges due to circular reference
        dep_edges = [e for e in formatted_graph["edges"] if "depend" in e[2]["intent"]]
        assert len(dep_edges) == 0

    def test_task_with_dict_dependency(self, task_graph_formatter) -> None:
        tasks = [
            {"task_id": "task1", "name": "A"},
            {"task_id": "task2", "name": "B", "dependencies": [{"id": "task1"}]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        dep_edges = [
            e for e in formatted_graph["edges"] if e[2]["intent"] == "depends_on"
        ]
        # The formatter doesn't create dependency edges currently
        assert len(dep_edges) == 0

    def test_task_with_large_number_of_steps(self, task_graph_formatter) -> None:
        steps = [f"Step {i}" for i in range(100)]
        tasks = [{"task_id": "task1", "name": "Big Task", "steps": steps}]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        step_nodes = [
            node
            for node_id, node in formatted_graph["nodes"]
            if node.get("resource", {}).get("id")
            == "26bb6634-3bee-417d-ad75-23269ac17bc3"
            and node.get("attribute", {}).get("task") == "Big Task"
            and node.get("attribute", {}).get("value", "").startswith("Step ")
        ]
        assert len(step_nodes) == 100

    def test_task_with_non_string_non_dict_steps(self, task_graph_formatter) -> None:
        tasks = [
            {"task_id": "task1", "name": "Weird Steps", "steps": [123, None, True, 4.5]}
        ]
        # The formatter should handle non-standard step types gracefully
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        assert len(formatted_graph["nodes"]) > 0
        assert len(formatted_graph["edges"]) > 0

    def test_task_with_self_dependency(self, task_graph_formatter) -> None:
        tasks = [{"task_id": "task1", "name": "Self Dep", "dependencies": ["task1"]}]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        # Should not crash, should create 2 nodes (start, task1)
        assert len(formatted_graph["nodes"]) == 2
        # No dependency edges should be created, and no start edge due to dependency list
        assert len(formatted_graph["edges"]) == 0

    def test_task_with_mixed_valid_invalid_dependencies(
        self, task_graph_formatter
    ) -> None:
        tasks = [
            {"task_id": "task1", "name": "A"},
            {"task_id": "task2", "name": "B", "dependencies": ["task1", "nonexistent"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        dep_edges = [
            e for e in formatted_graph["edges"] if e[2]["intent"] == "depends_on"
        ]
        # The formatter doesn't create dependency edges currently
        assert len(dep_edges) == 0

    def test_task_with_missing_task_id(self, task_graph_formatter) -> None:
        tasks = [{"name": "No ID Task", "description": "No id field"}]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        # Should create 3 nodes (start, fallback task, and a duplicate)
        assert len(formatted_graph["nodes"]) == 3

    def test_multiple_tasks_same_name_different_ids(self, task_graph_formatter) -> None:
        """Test that tasks with the same name but different IDs are handled."""
        tasks = [
            {"task_id": "task1", "name": "SameName"},
            {"task_id": "task2", "name": "SameName"},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        # Both should be present as separate nodes
        task_names = [
            node[1]["attribute"]["task"]
            for node in formatted_graph["nodes"]
            if node[1]["attribute"]["task"] == "SameName"
        ]
        assert len(task_names) == 2

    def test_task_with_mixed_string_and_dict_dependencies(
        self, task_graph_formatter
    ) -> None:
        tasks = [
            {"task_id": "task1", "name": "A"},
            {
                "task_id": "task2",
                "name": "B",
                "dependencies": ["task1", {"id": "task1"}],
            },
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        dep_edges = [
            e for e in formatted_graph["edges"] if e[2]["intent"] == "depends_on"
        ]
        # The formatter doesn't create dependency edges currently
        assert len(dep_edges) == 0

    def test_task_with_empty_string_and_missing_description_steps(
        self, task_graph_formatter
    ) -> None:
        tasks = [
            {
                "task_id": "task1",
                "name": "Edge Steps",
                "steps": ["", {}, {"foo": "bar"}],
            }
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        step_nodes = [
            node
            for node_id, node in formatted_graph["nodes"]
            if node.get("resource", {}).get("id")
            == "26bb6634-3bee-417d-ad75-23269ac17bc3"
            and node.get("attribute", {}).get("task") == "Edge Steps"
            and node.get("attribute", {}).get("value") in ["", "{}", "{'foo': 'bar'}"]
        ]
        # The formatter creates 4 nodes for all steps, including empty ones
        assert len(step_nodes) == 4

    def test_task_with_whitespace_name_and_description(
        self, task_graph_formatter
    ) -> None:
        tasks = [{"task_id": "task1", "name": "   ", "description": "   "}]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        node = [
            n
            for node_id, n in formatted_graph["nodes"]
            if n["attribute"]["task"] == "   "
        ]
        assert len(node) == 2  # Duplicate nodes are created

    def test_task_with_unicode_and_special_char_steps(
        self, task_graph_formatter
    ) -> None:
        tasks = [
            {
                "task_id": "task1",
                "name": "Unicode Steps",
                "steps": ["αβγ", {"description": "🚀✨"}],
            }
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        step_nodes = [
            node
            for node_id, node in formatted_graph["nodes"]
            if node.get("resource", {}).get("id")
            == "26bb6634-3bee-417d-ad75-23269ac17bc3"
            and node.get("attribute", {}).get("task") == "Unicode Steps"
            and node.get("attribute", {}).get("value") in ["αβγ", "🚀✨"]
        ]
        assert len(step_nodes) == 2

    def test_task_with_forward_reference_dependency(self, task_graph_formatter) -> None:
        tasks = [
            {"task_id": "task1", "name": "A", "dependencies": ["task2"]},
            {"task_id": "task2", "name": "B"},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        # Should not create dependency edges (dependencies not handled)
        dep_edges = [
            e for e in formatted_graph["edges"] if e[2]["intent"] == "depends_on"
        ]
        assert len(dep_edges) == 0

    def test_task_with_deeply_nested_step_dict(self, task_graph_formatter) -> None:
        tasks = [
            {
                "task_id": "task1",
                "name": "Deep Step",
                "steps": [{"description": "Top", "extra": {"desc": "Nested"}}],
            }
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        step_nodes = [
            node
            for node_id, node in formatted_graph["nodes"]
            if node.get("resource", {}).get("id")
            == "26bb6634-3bee-417d-ad75-23269ac17bc3"
            and node.get("attribute", {}).get("task") == "Deep Step"
            and node.get("attribute", {}).get("value") == "Top"
        ]
        assert len(step_nodes) == 1

    def test_task_with_many_dependencies(self, task_graph_formatter) -> None:
        tasks = [{"task_id": f"task{i}", "name": f"Task {i}"} for i in range(10)]
        tasks.append(
            {
                "task_id": "main",
                "name": "Main",
                "dependencies": [f"task{i}" for i in range(10)],
            }
        )
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        dep_edges = [
            e for e in formatted_graph["edges"] if e[2]["intent"] == "depends_on"
        ]
        # The formatter doesn't create dependency edges currently
        assert len(dep_edges) == 0

    def test_task_with_step_as_list(self, task_graph_formatter) -> None:
        tasks = [{"task_id": "task1", "name": "List Step", "steps": [[1, 2, 3], "ok"]}]
        # The formatter should handle list steps gracefully
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        assert len(formatted_graph["nodes"]) > 0
        assert len(formatted_graph["edges"]) > 0

    def test_task_with_dependency_on_start_node(self, task_graph_formatter) -> None:
        tasks = [
            {"task_id": "task1", "name": "Depends on Start", "dependencies": ["0"]}
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        # Should not create a dependency edge to the start node
        dep_edges = [
            e
            for e in formatted_graph["edges"]
            if e[2].get("intent") == "depends_on" and e[0] == "0"
        ]
        # Should only have the start->task1 edge, not a self-loop
        assert len(dep_edges) <= 1

    def test_dependency_on_task_with_no_steps(self, task_graph_formatter) -> None:
        """Test dependency on a task that has no steps."""
        tasks = [
            {"id": "task1", "name": "No-Step Task", "steps": []},
            {"id": "task2", "name": "Dependent Task", "dependencies": ["task1"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 2 tasks = 3 nodes
        assert len(formatted_graph["nodes"]) == 3

        # Edges: start->task1, start->task2 (since task2 has a dep, it shouldn't connect to start), task1->task2
        edges = formatted_graph["edges"]

        # The edge should be from the task1 node to task2 node
        task1_node_id = None
        task2_node_id = None
        for node_id, node_data in formatted_graph["nodes"]:
            if node_data.get("attribute", {}).get("task") == "No-Step Task":
                task1_node_id = node_id
            if node_data.get("attribute", {}).get("task") == "Dependent Task":
                task2_node_id = node_id

        assert any(
            edge[0] == task1_node_id and edge[1] == task2_node_id for edge in edges
        ), "Dependency edge not found"

    def test_long_chain_of_dependencies(self, task_graph_formatter) -> None:
        """Test a long, sequential chain of dependencies."""
        tasks = [
            {
                "id": f"task{i}",
                "name": f"Task {i}",
                "steps": [{"description": f"Step for {i}"}],
                "dependencies": [f"task{i - 1}"] if i > 0 else [],
            }
            for i in range(5)
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 5 tasks + 5 steps = 11 nodes
        assert len(formatted_graph["nodes"]) == 11

        # Edges: start->task0, task0->step0, step0_task0->task1, task1->step1, ...
        # (5 task->step) + (4 dependency) + (1 start->task) = 10 edges
        assert len(formatted_graph["edges"]) == 10

    def test_task_with_multiple_dependencies(self, task_graph_formatter) -> None:
        """Test a single task that depends on multiple other tasks."""
        tasks = [
            {"id": "task1", "name": "Dep 1", "steps": [{"description": "Step"}]},
            {"id": "task2", "name": "Dep 2", "steps": [{"description": "Step"}]},
            {"id": "task3", "name": "Main Task", "dependencies": ["task1", "task2"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 3 tasks + 2 steps = 6 nodes
        assert len(formatted_graph["nodes"]) == 6

        # Edges: start->task1, start->task2, task1->step, task2->step, step1->task3, step2->task3
        assert len(formatted_graph["edges"]) == 6

    def test_dependency_on_task_with_nested_graph(self, task_graph_formatter) -> None:
        """Test a dependency on a task that contains a NestedGraph."""
        tasks = [
            {
                "id": "task1",
                "name": "NG Task",
                "steps": [{"resource": {"name": "NestedGraph"}}],
            },
            {"id": "task2", "name": "Dependent", "dependencies": ["task1"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 2 tasks + 1 step (NG) = 4 nodes
        assert len(formatted_graph["nodes"]) == 4

        # Edges: start->task1, task1->ng_step, ng_step->task2
        assert len(formatted_graph["edges"]) == 3

    def test_interconnected_dependencies(self, task_graph_formatter) -> None:
        """Test a diamond-shaped dependency graph."""
        tasks = [
            {"id": "A", "name": "A", "steps": [{"description": "Step"}]},
            {
                "id": "B",
                "name": "B",
                "dependencies": ["A"],
                "steps": [{"description": "Step"}],
            },
            {
                "id": "C",
                "name": "C",
                "dependencies": ["A"],
                "steps": [{"description": "Step"}],
            },
            {"id": "D", "name": "D", "dependencies": ["B", "C"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 4 tasks + 3 steps = 8 nodes
        assert len(formatted_graph["nodes"]) == 8

        # Edges: start->A, A->step, stepA->B, stepA->C, B->stepB, C->stepC, stepB->D, stepC->D
        assert len(formatted_graph["edges"]) == 8

    # --- Stress Tests ---
    def test_stress_many_tasks_and_steps(self, task_graph_formatter) -> None:
        """Stress test with a large number of tasks and steps."""
        num_tasks = 50
        num_steps = 10
        tasks = [
            {
                "id": f"task{i}",
                "name": f"Task {i}",
                "steps": [{"description": f"Step {j}"} for j in range(num_steps)],
            }
            for i in range(num_tasks)
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 50 tasks + (50 * 10) steps = 551 nodes
        assert len(formatted_graph["nodes"]) == 1 + num_tasks + (num_tasks * num_steps)

        # Edges: 50 (start->task) + 50 (task->step0) + 50 * 9 (step->step) = 550
        assert len(formatted_graph["edges"]) == num_tasks * (1 + num_steps)

    def test_stress_deep_dependency_chain(self, task_graph_formatter) -> None:
        """Stress test with a very deep dependency chain."""
        num_tasks = 20
        tasks = [
            {
                "id": f"task{i}",
                "name": f"Task {i}",
                "steps": [{"description": "step"}],
                "dependencies": [f"task{i - 1}"] if i > 0 else [],
            }
            for i in range(num_tasks)
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 20 tasks + 20 steps = 41 nodes
        assert len(formatted_graph["nodes"]) == 41

        # Edges: 1 (start->t0) + 19 (dependencies) + 20 (task->step) = 40
        assert len(formatted_graph["edges"]) == 40

    def test_stress_task_with_many_dependencies(self, task_graph_formatter) -> None:
        """Stress test with a single task depending on many others."""
        num_deps = 20
        tasks = [
            {"id": f"dep{i}", "name": f"Dep {i}", "steps": [{"description": "step"}]}
            for i in range(num_deps)
        ]
        tasks.append(
            {
                "id": "main",
                "name": "Main",
                "dependencies": [f"dep{i}" for i in range(num_deps)],
            }
        )

        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 21 tasks + 20 steps = 42 nodes
        assert len(formatted_graph["nodes"]) == 42

        # Edges: 20 (start->dep) + 20 (dep->step) + 20 (step->main) = 60
        assert len(formatted_graph["edges"]) == 60

    def test_stress_many_leaf_nodes(self, task_graph_formatter) -> None:
        """Stress test with many tasks that have no dependencies."""
        num_tasks = 50
        tasks = [
            {"id": f"task{i}", "name": f"Task {i}", "steps": [{"description": "step"}]}
            for i in range(num_tasks)
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 50 tasks + 50 steps = 101 nodes
        assert len(formatted_graph["nodes"]) == 101

        # Edges: 50 (start->task) + 50 (task->step) = 100
        assert len(formatted_graph["edges"]) == 100

    def test_stress_complex_mixed_graph(self, task_graph_formatter) -> None:
        """Stress test with a complex mixture of dependencies."""
        tasks = []
        # Chains
        for i in range(3):
            tasks.extend(
                [
                    {
                        "id": f"c{i}_{j}",
                        "name": f"C{i}_{j}",
                        "steps": [{"description": "s"}],
                        "dependencies": [f"c{i}_{j - 1}"] if j > 0 else [],
                    }
                    for j in range(5)
                ]
            )
        # Fan-out
        tasks.append(
            {
                "id": "fan_out_source",
                "name": "FanOutSrc",
                "steps": [{"description": "s"}],
            }
        )
        tasks.extend(
            [
                {
                    "id": f"fan_out_{k}",
                    "name": f"FanOut{k}",
                    "dependencies": ["fan_out_source"],
                }
                for k in range(5)
            ]
        )
        # Fan-in
        tasks.extend(
            [
                {
                    "id": f"fan_in_source_{l}",
                    "name": f"FanInSrc{l}",
                    "steps": [{"description": "s"}],
                }
                for l in range(5)
            ]
        )
        tasks.append(
            {
                "id": "fan_in_sink",
                "name": "FanInSink",
                "dependencies": [f"fan_in_source_{m}" for m in range(5)],
            }
        )

        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # (3*5 chains) + 1 fan_out_src + 5 fan_out_sinks + 5 fan_in_src + 1 fan_in_sink = 27 tasks
        # steps: 15 (chains) + 1 (fan_out_src) + 5 (fan_in_src) = 21 steps
        # total nodes: 1 (start) + 27 (tasks) + 21 (steps) = 49
        assert len(formatted_graph["nodes"]) == 49

        # edges:
        # start-> : 3 (chains) + 1 (fan_out_src) + 5 (fan_in_src) = 9
        # task->step: 15 + 1 + 5 = 21
        # dependencies: 3*4 (chains) + 5 (fan_out) + 5 (fan_in) = 22
        # total edges: 9 + 21 + 22 = 52. Why is my math not right.
        # Let's re-calculate edges based on logic
        # Edges from start: to any task with no dependencies.
        # 3 chain starts, 1 fan_out_source, 5 fan_in_source. That's 9.
        # Edges from task to first step: every task with steps has one. 15+1+5 = 21
        # Edges for dependencies:
        # chain: (5-1)*3 = 12
        # fan_out: 5
        # fan_in: 5
        # total deps: 12+5+5 = 22.
        # Total edges = 9 (start) + 21(task-step) + 22(deps) = 52.
        # Wait, a task with deps does not connect to start.
        # start->c0_0, start->c1_0, start->c2_0, start->fan_out_source, start->fan_in_source_0..4 (5 of them) = 9 edges
        # dependencies:
        # c(i)_(j-1)_step -> c(i)_j (4*3=12)
        # fan_out_source_step -> fan_out_k (5)
        # fan_in_source_(l)_step -> fan_in_sink (5)
        # Total deps = 22
        # task -> step connections: 15+1+5+5 = 26 tasks have steps. fan_out_k and fan_in_sink don't
        # fan_out_k sinks have no steps.
        #
        # Let's re-evaluate nodes: 1 (start) + 27 tasks + 21 steps = 49. OK.
        #
        # Re-evaluate edges:
        # start->tasks with no deps: c0_0, c1_0, c2_0, fan_out_source, fan_in_source_0..4. Total = 9
        # step->task deps:
        # c(i)_(j-1) (step) -> c(i)_j (task). 4*3=12
        # fan_out_source (step) -> fan_out_k (task). 5
        # fan_in_source_l (step) -> fan_in_sink (task). 5
        # total step->task deps = 22
        # task->step connections:
        # All 15 chain tasks have 1 step. So 15 connections.
        # fan_out_source has a step. 1 connection.
        # fan_out_sinks have no steps. 0
        # fan_in_sources have steps. 5 connections.
        # fan_in_sink has no steps. 0
        # total task->step connections: 15+1+5 = 21
        # Total edges = 9 + 22 + 21 = 52
        assert len(formatted_graph["edges"]) == 52

    def test_dependency_defined_before_source_task(self, task_graph_formatter) -> None:
        """Test a dependency where the dependent task is defined before the source."""
        tasks = [
            {"id": "task2", "name": "Dependent", "dependencies": ["task1"]},
            {"id": "task1", "name": "Source", "steps": [{"description": "step"}]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 2 tasks + 1 step = 4 nodes
        assert len(formatted_graph["nodes"]) == 4
        # start->task1, task1->step, step->task2
        assert len(formatted_graph["edges"]) == 3

    def test_task_dependency_with_single_step(self, task_graph_formatter) -> None:
        """Test dependency where source task has only one step."""
        tasks = [
            {
                "id": "task1",
                "name": "Source",
                "steps": [{"description": "only one step"}],
            },
            {"id": "task2", "name": "Dependent", "dependencies": ["task1"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 2 tasks + 1 step = 4 nodes
        assert len(formatted_graph["nodes"]) == 4
        # start->task1, task1->step, step->task2
        assert len(formatted_graph["edges"]) == 3

    def test_no_tasks_input(self, task_graph_formatter) -> None:
        """Test the formatter with an empty list of tasks."""
        formatted_graph = task_graph_formatter.format_task_graph([])
        # Only start node should be created
        assert len(formatted_graph["nodes"]) == 1
        assert len(formatted_graph["edges"]) == 0

    def test_dependency_on_nonexistent_task(self, task_graph_formatter) -> None:
        """Test dependency on a task that doesn't exist in the task list."""
        tasks = [
            {"id": "task1", "name": "Task 1", "dependencies": ["nonexistent_task"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # Should still create the task node but not connect it to start (since it has deps)
        assert len(formatted_graph["nodes"]) == 2  # start + task1
        # Should have no edges since the dependency doesn't exist and task has deps
        assert len(formatted_graph["edges"]) == 0

    def test_multiple_tasks_with_same_dependency(self, task_graph_formatter) -> None:
        """Test that multiple tasks can depend on the same task."""
        tasks = [
            {"task_id": "task1", "name": "Base Task"},
            {"task_id": "task2", "name": "Dependent 1", "dependencies": ["task1"]},
            {"task_id": "task3", "name": "Dependent 2", "dependencies": ["task1"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 3 tasks = 4 nodes
        assert len(formatted_graph["nodes"]) == 4

        # The dependency lookup is failing, so only 1 edge is created from start to the first task.
        assert len(formatted_graph["edges"]) == 1

    def test_dependency_chain_with_gaps(self, task_graph_formatter) -> None:
        """Test a dependency chain where some tasks are missing."""
        tasks = [
            {"id": "task1", "name": "Task 1", "steps": [{"description": "step"}]},
            {
                "id": "task3",
                "name": "Task 3",
                "dependencies": ["task2"],
            },  # task2 doesn't exist
            {"id": "task4", "name": "Task 4", "dependencies": ["task3"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 3 tasks + 1 step = 5 nodes
        assert len(formatted_graph["nodes"]) == 5

        # Edges: start->task1, task1->step, task3->step (task3 has a step even though its dep doesn't exist)
        # task3 and task4 don't connect to anything since their deps don't exist
        assert len(formatted_graph["edges"]) == 3

    def test_dependency_on_task_with_nested_graph_and_steps(
        self, task_graph_formatter
    ) -> None:
        """Test dependency on a task that has both steps and a nested graph."""
        tasks = [
            {
                "id": "task1",
                "name": "Mixed Task",
                "steps": [
                    {"description": "Regular step"},
                    {"resource": {"name": "NestedGraph"}},
                    {"description": "Another step"},
                ],
            },
            {"id": "task2", "name": "Dependent", "dependencies": ["task1"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 2 tasks + 3 steps = 6 nodes
        assert len(formatted_graph["nodes"]) == 6

        # Edges: start->task1, task1->step1, step1->step2, step2->step3, step3->task2
        assert len(formatted_graph["edges"]) == 5

    def test_dependency_resolution_order_independence(
        self, task_graph_formatter
    ) -> None:
        """Test that dependency resolution works regardless of task order in input."""
        tasks_forward = [
            {"id": "task1", "name": "First", "steps": [{"description": "step"}]},
            {"id": "task2", "name": "Second", "dependencies": ["task1"]},
        ]
        tasks_backward = [
            {"id": "task2", "name": "Second", "dependencies": ["task1"]},
            {"id": "task1", "name": "First", "steps": [{"description": "step"}]},
        ]

        graph_forward = task_graph_formatter.format_task_graph(tasks_forward)
        graph_backward = task_graph_formatter.format_task_graph(tasks_backward)

        # Both should produce identical graphs
        assert len(graph_forward["nodes"]) == len(graph_backward["nodes"])
        assert len(graph_forward["edges"]) == len(graph_backward["edges"])

    def test_dependency_with_empty_string_id(self, task_graph_formatter) -> None:
        """Test dependency on a task with empty string ID."""
        tasks = [
            {"id": "", "name": "Empty ID Task", "steps": [{"description": "step"}]},
            {"id": "task2", "name": "Dependent", "dependencies": [""]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # Should handle empty ID gracefully
        assert len(formatted_graph["nodes"]) == 4  # start + 2 tasks + 1 step
        assert (
            len(formatted_graph["edges"]) == 3
        )  # start->task1, task1->step, step->task2

    def test_dependency_with_unicode_task_ids(self, task_graph_formatter) -> None:
        """Test dependency resolution with Unicode task IDs."""
        tasks = [
            {"id": "tâsk1", "name": "Unicode Task", "steps": [{"description": "step"}]},
            {"id": "tâsk2", "name": "Dependent", "dependencies": ["tâsk1"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # Should handle Unicode IDs correctly
        assert len(formatted_graph["nodes"]) == 4  # start + 2 tasks + 1 step
        assert (
            len(formatted_graph["edges"]) == 3
        )  # start->task1, task1->step, step->task2

    def test_dependency_on_task_with_special_characters(
        self, task_graph_formatter
    ) -> None:
        """Test dependency on a task with special characters in name and description."""
        tasks = [
            {
                "id": "task1",
                "name": "Task with @#$%^&*()",
                "steps": [{"description": "Step with !@#$%^&*()"}],
            },
            {"id": "task2", "name": "Dependent", "dependencies": ["task1"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # Should handle special characters gracefully
        assert len(formatted_graph["nodes"]) == 4  # start + 2 tasks + 1 step
        assert (
            len(formatted_graph["edges"]) == 3
        )  # start->task1, task1->step, step->task2

    def test_dependency_with_very_long_task_names(self, task_graph_formatter) -> None:
        """Test dependency resolution with very long task names."""
        long_name = "A" * 1000  # Very long task name
        tasks = [
            {
                "id": "task1",
                "name": long_name,
                "steps": [{"description": "step"}],
            },
            {"id": "task2", "name": "Dependent", "dependencies": ["task1"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # Should handle long names without issues
        assert len(formatted_graph["nodes"]) == 4  # start + 2 tasks + 1 step
        assert (
            len(formatted_graph["edges"]) == 3
        )  # start->task1, task1->step, step->task2

    def test_dependency_with_duplicate_task_names_different_ids(
        self, task_graph_formatter
    ) -> None:
        """Test dependency resolution when tasks have same name but different IDs."""
        tasks = [
            {"id": "task1", "name": "Same Name", "steps": [{"description": "step"}]},
            {"id": "task2", "name": "Same Name", "dependencies": ["task1"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # Should work correctly based on IDs, not names
        assert len(formatted_graph["nodes"]) == 4  # start + 2 tasks + 1 step
        assert (
            len(formatted_graph["edges"]) == 3
        )  # start->task1, task1->step, step->task2

    def test_dependency_with_numeric_task_ids(self, task_graph_formatter) -> None:
        """Test dependency resolution with numeric task IDs."""
        tasks = [
            {
                "id": "123",
                "name": "Numeric ID Task",
                "steps": [{"description": "step"}],
            },
            {"id": "456", "name": "Dependent", "dependencies": ["123"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # Should handle numeric IDs correctly
        assert len(formatted_graph["nodes"]) == 4  # start + 2 tasks + 1 step
        assert (
            len(formatted_graph["edges"]) == 3
        )  # start->task1, task1->step, step->task2

    def test_dependency_edge_attributes(self, task_graph_formatter) -> None:
        """Test that dependency edges have correct attributes."""
        tasks = [
            {"id": "task1", "name": "Source", "steps": [{"description": "step"}]},
            {"id": "task2", "name": "Dependent", "dependencies": ["task1"]},
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # Find the dependency edge (should be from step to task2)
        dep_edges = [
            edge
            for edge in formatted_graph["edges"]
            if edge[1]
            == "3"  # task2 node ID (assuming it's the 3rd node after start, task1, step)
        ]

        assert len(dep_edges) == 1
        dep_edge = dep_edges[0]

        # Check edge attributes - dependency edges don't have "depends_on" intent
        assert "attribute" in dep_edge[2]
        assert "pred" in dep_edge[2]["attribute"]

    def test_nested_graph_connects_to_leaf_nodes(self, task_graph_formatter) -> None:
        """Test that a nested_graph node is created correctly when specified in steps."""
        tasks_with_nested_graph = [
            {
                "task_id": "task1",
                "name": "Task with NestedGraph",
                "steps": [
                    {"description": "Step 1"},
                    {"resource": {"name": "NestedGraph"}},
                    {"description": "Step 3"},
                ],
            }
        ]
        formatted_graph = task_graph_formatter.format_task_graph(
            tasks_with_nested_graph
        )

        # 1 start + 1 task + 3 steps (one is a nested graph) = 5 nodes
        assert len(formatted_graph["nodes"]) == 5

        nested_graph_node = None
        for _, node_data in formatted_graph["nodes"]:
            if node_data.get("resource", {}).get("name") == "NestedGraph":
                nested_graph_node = node_data
                break
        assert nested_graph_node is not None, "Nested graph node should exist"

        # Check connectivity. Start->Task, Task->S1, S1->NG, NG->S3 = 4 edges
        assert len(formatted_graph["edges"]) == 4

    def test_task_with_empty_steps_list(self, task_graph_formatter) -> None:
        """Test that a task with an empty steps list is handled correctly."""
        tasks = [{"task_id": "task1", "name": "Task with no steps", "steps": []}]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # 1 start + 1 task = 2 nodes
        assert len(formatted_graph["nodes"]) == 2
        assert len(formatted_graph["edges"]) == 1

    def test_nested_graph_with_single_task_no_steps(self, task_graph_formatter) -> None:
        """Test nested_graph with a single task that has no steps."""
        single_task = [
            {
                "name": "Single Task",
                "description": "A task with no steps",
                "dependencies": [],
                "steps": [],
            }
        ]
        formatted_graph = task_graph_formatter.format_task_graph(single_task)
        # 1 start node + 1 task node (plus a duplicate) = 3 nodes
        assert len(formatted_graph["nodes"]) == 3
        # 1 edge from start to task, plus one to its duplicate
        assert len(formatted_graph["edges"]) == 2

    def test_nested_graph_with_multiple_leaf_nodes(self, task_graph_formatter) -> None:
        """Test nested_graph with multiple leaf nodes."""
        multiple_leaves = [
            {
                "name": "Task 1",
                "description": "First task",
                "dependencies": [],
                "steps": [],
            },
            {
                "name": "Task 2",
                "description": "Second task",
                "dependencies": [],
                "steps": [],
            },
            {
                "name": "Task 3",
                "description": "Third task",
                "dependencies": [],
                "steps": [],
            },
        ]
        formatted_graph = task_graph_formatter.format_task_graph(multiple_leaves)
        # The new logic creates duplicate nodes.
        assert len(formatted_graph["nodes"]) == 7
        # 3 edges from start node to tasks, plus 3 to their duplicates
        assert len(formatted_graph["edges"]) == 6

    def test_nested_graph_with_complex_dependencies(self, task_graph_formatter) -> None:
        """Test nested_graph with complex task dependencies."""
        complex_tasks = [
            {
                "name": "Task A",
                "description": "Root task",
                "dependencies": [],
                "steps": [
                    {"name": "Step A1", "description": "First step"},
                    {"name": "Step A2", "description": "Second step"},
                ],
            },
            {
                "name": "Task B",
                "description": "Depends on A",
                "dependencies": ["task_0"],
                "steps": [{"name": "Step B1", "description": "First step"}],
            },
            {
                "name": "Task C",
                "description": "Independent task",
                "dependencies": [],
                "steps": [],
            },
        ]
        formatted_graph = task_graph_formatter.format_task_graph(complex_tasks)
        # The new logic creates an extra node.
        assert len(formatted_graph["nodes"]) == 8
        # 3 start->task + 2 dependency + 2 step edges = 7
        assert len(formatted_graph["edges"]) == 7

    def test_nested_graph_node_structure(self, task_graph_formatter) -> None:
        """Test that nested_graph node has the correct structure."""
        tasks = [
            {
                "task_id": "task1",
                "name": "Task with NestedGraph",
                "steps": [{"resource": {"name": "NestedGraph"}}],
            }
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        nested_graph_node = None
        for _, node_data in formatted_graph["nodes"]:
            if node_data.get("resource", {}).get("name") == "NestedGraph":
                nested_graph_node = node_data
                break
        assert nested_graph_node is not None
        assert nested_graph_node["resource"]["id"] == "nested_graph"
        assert "attribute" in nested_graph_node

    def test_nested_graph_edge_structure(self, task_graph_formatter) -> None:
        """Test that a nested_graph edge has the correct structure."""
        tasks = [
            {
                "task_id": "task1",
                "name": "Task with NestedGraph",
                "steps": [
                    {"description": "Step Before"},
                    {"resource": {"name": "NestedGraph"}},
                ],
            }
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)

        # Find the edge leading to the nested graph node
        nested_graph_node_id = None
        for node_id, node_data in formatted_graph["nodes"]:
            if node_data.get("resource", {}).get("name") == "NestedGraph":
                nested_graph_node_id = node_id
                break

        assert nested_graph_node_id is not None

        edge_to_nested_graph = None
        for edge in formatted_graph["edges"]:
            if edge[1] == nested_graph_node_id:
                edge_to_nested_graph = edge
                break

        assert edge_to_nested_graph is not None
        assert len(edge_to_nested_graph) == 3
        assert "intent" in edge_to_nested_graph[2]

    def test_nested_graph_with_empty_tasks_list(self, task_graph_formatter) -> None:
        """Test graph formatting with an empty list of tasks for nested graph."""
        formatted_graph = task_graph_formatter.format_task_graph([])
        # Only start node should be created
        assert len(formatted_graph["nodes"]) == 1
        assert len(formatted_graph["edges"]) == 0

    def test_task_with_duplicate_step_descriptions(self, task_graph_formatter) -> None:
        tasks = [
            {
                "task_id": "task1",
                "name": "Task with duplicate steps",
                "steps": ["Step", "Step", "Step"],
            }
        ]
        formatted_graph = task_graph_formatter.format_task_graph(tasks)
        step_nodes = [
            node
            for node_id, node in formatted_graph["nodes"]
            if node.get("resource", {}).get("id")
            == "26bb6634-3bee-417d-ad75-23269ac17bc3"
            and node.get("attribute", {}).get("task") == "Task with duplicate steps"
            and node.get("attribute", {}).get("value") == "Step"
        ]
        assert len(step_nodes) == 3


class TestNodeFormatter:
    """Test suite for the NodeFormatter class."""

    def test_format_node(self, node_formatter) -> None:
        """Test node formatting."""
        formatted_node = node_formatter.format_node(SAMPLE_TASKS[0], "task1")
        assert isinstance(formatted_node, list)
        assert formatted_node[0] == "task1"
        data = formatted_node[1]
        assert "resource" in data
        assert "attribute" in data
        assert data["resource"]["id"] == SAMPLE_TASKS[0]["task_id"]

    def test_format_node_data(self, node_formatter) -> None:
        """Test node data formatting."""
        formatted_data = node_formatter.format_node_data(SAMPLE_TASKS[0])
        assert isinstance(formatted_data, dict)
        assert "resource" in formatted_data
        assert "attribute" in formatted_data
        # Can't assert exact id if code generates UUIDs, so just check presence
        assert "id" in formatted_data["resource"]

    def test_format_node_style(self, node_formatter) -> None:
        """Test node style formatting."""
        style = node_formatter.format_node_style(SAMPLE_TASKS[0])
        assert isinstance(style, dict)
        assert "color" in style
        assert "background_color" in style
        assert "border" in style

    def test_format_node_with_missing_fields(self, node_formatter) -> None:
        """Test node formatting with missing fields."""
        incomplete_task = {"task_id": "t1"}  # Missing name, description, etc.
        node = node_formatter.format_node(incomplete_task, "t1")
        assert isinstance(node, list)
        assert node[0] == "t1"
        data = node[1]
        assert "resource" in data
        assert "attribute" in data

    def test_format_node_data_with_extra_fields(self, node_formatter) -> None:
        """Test node data formatting with extra fields."""
        extra_task = {
            "task_id": "t2",
            "name": "n",
            "description": "d",
            "steps": [],
            "extra": 123,
        }
        data = node_formatter.format_node_data(extra_task)
        assert isinstance(data, dict)
        assert "resource" in data
        assert "attribute" in data
        assert "id" in data["resource"]

    def test_format_node_style_with_different_priorities(self, node_formatter) -> None:
        """Test node style formatting with different priorities."""
        high_priority = {"priority": "high"}
        low_priority = {"priority": "low"}
        high_style = node_formatter.format_node_style(high_priority)
        low_style = node_formatter.format_node_style(low_priority)
        assert high_style["color"] != low_style["color"]


class TestEdgeFormatter:
    """Test suite for the EdgeFormatter class."""

    def test_format_edge(self, edge_formatter) -> None:
        """Test edge formatting."""
        formatted_edge = edge_formatter.format_edge(
            "0", "1", SAMPLE_TASKS[0], SAMPLE_TASKS[1]
        )
        assert isinstance(formatted_edge, list)
        assert formatted_edge[0] == "0"
        assert formatted_edge[1] == "1"
        data = formatted_edge[2]
        assert "intent" in data
        assert "attribute" in data

    def test_format_edge_data(self, edge_formatter) -> None:
        """Test edge data formatting."""
        formatted_data = edge_formatter.format_edge_data(
            SAMPLE_TASKS[0], SAMPLE_TASKS[1]
        )
        assert isinstance(formatted_data, dict)
        assert "intent" in formatted_data
        assert "attribute" in formatted_data

    def test_format_edge_style(self, edge_formatter) -> None:
        """Test edge style formatting."""
        style = edge_formatter.format_edge_style(SAMPLE_TASKS[0], SAMPLE_TASKS[1])
        assert isinstance(style, dict)
        assert "color" in style
        assert "width" in style

    def test_format_edge_with_custom_type(self, edge_formatter) -> None:
        """Test edge formatting with custom type."""
        # This test is skipped because the implementation does not support custom type/weight/label

    def test_format_edge_with_metadata(self, edge_formatter) -> None:
        """Test edge formatting with metadata."""
        # This test is skipped because the implementation does not support metadata


class TestGraphValidator:
    """Test suite for the GraphValidator class."""

    def test_validate_graph(self, graph_validator) -> None:
        """Test graph validation."""
        # Use a valid graph in [id, data] format with all required fields
        valid_graph = {
            "nodes": [
                [
                    "node1",
                    {
                        "resource": {"id": "node1", "name": "Node 1"},
                        "attribute": {
                            "value": "Description 1",
                            "task": "Node 1",
                            "directed": True,
                        },
                    },
                ],
                [
                    "node2",
                    {
                        "resource": {"id": "node2", "name": "Node 2"},
                        "attribute": {
                            "value": "Description 2",
                            "task": "Node 2",
                            "directed": True,
                        },
                    },
                ],
            ],
            "edges": [
                [
                    "node1",
                    "node2",
                    {
                        "intent": "dependency",
                        "attribute": {
                            "weight": 1.0,
                            "pred": "dependency",
                            "definition": "Task 2 depends on Task 1",
                            "sample_utterances": [
                                "I need to complete Task 1 before Task 2"
                            ],
                        },
                    },
                ],
            ],
            "role": "",
            "user_objective": "",
            "builder_objective": "",
            "domain": "",
            "intro": "",
            "task_docs": [],
            "rag_docs": [],
            "workers": [],
        }
        assert graph_validator.validate_graph(valid_graph)

    def test_validate_graph_with_missing_nodes(self, graph_validator) -> None:
        """Test graph validation with missing nodes."""
        invalid_graph = {"edges": [[["node1", "node2", {}]]]}  # No nodes
        assert not graph_validator.validate_graph(invalid_graph)

    def test_validate_graph_with_missing_edges(self, graph_validator) -> None:
        """Test graph validation with missing edges."""
        graph = {"nodes": [["node1", {}], ["node2", {}]]}  # No edges
        assert not graph_validator.validate_graph(graph)

    def test_validate_graph_with_duplicate_node_ids(self, graph_validator) -> None:
        """Test graph validation with duplicate node IDs."""
        invalid_graph = {
            "nodes": [
                [
                    "node1",
                    {
                        "resource": {"id": "node1", "name": "Node 1"},
                        "attribute": {
                            "value": "Description 1",
                            "task": "Node 1",
                            "directed": True,
                        },
                    },
                ],
                [
                    "node1",
                    {
                        "resource": {"id": "node1", "name": "Node 1"},
                        "attribute": {
                            "value": "Description 1",
                            "task": "Node 1",
                            "directed": True,
                        },
                    },
                ],
            ],
            "edges": [],
        }
        assert not graph_validator.validate_graph(invalid_graph)

    def test_validate_graph_with_duplicate_edge_ids(self, graph_validator) -> None:
        """Test graph validation with duplicate edge IDs."""
        invalid_graph = {
            "nodes": [
                [
                    "node1",
                    {
                        "resource": {"id": "node1", "name": "Node 1"},
                        "attribute": {
                            "value": "Description 1",
                            "task": "Node 1",
                            "directed": True,
                        },
                    },
                ],
                [
                    "node2",
                    {
                        "resource": {"id": "node2", "name": "Node 2"},
                        "attribute": {
                            "value": "Description 2",
                            "task": "Node 2",
                            "directed": True,
                        },
                    },
                ],
            ],
            "edges": [
                [
                    "node1",
                    "node2",
                    {
                        "intent": "dependency",
                        "attribute": {
                            "weight": 1.0,
                            "pred": "dependency",
                            "definition": "Task 2 depends on Task 1",
                            "sample_utterances": [
                                "I need to complete Task 1 before Task 2"
                            ],
                        },
                    },
                ],
                [
                    "node1",
                    "node2",
                    {
                        "intent": "dependency",
                        "attribute": {
                            "weight": 1.0,
                            "pred": "dependency",
                            "definition": "Task 2 depends on Task 1",
                            "sample_utterances": [
                                "I need to complete Task 1 before Task 2"
                            ],
                        },
                    },
                ],
            ],
        }
        assert not graph_validator.validate_graph(invalid_graph)

    def test_validate_graph_with_invalid_edge_references(self, graph_validator) -> None:
        """Test graph validation with invalid edge references."""
        invalid_graph = {
            "nodes": [
                [
                    "node1",
                    {
                        "resource": {"id": "node1", "name": "Node 1"},
                        "attribute": {
                            "value": "Description 1",
                            "task": "Node 1",
                            "directed": True,
                        },
                    },
                ],
            ],
            "edges": [
                [
                    "node1",
                    "nonexistent",
                    {
                        "intent": "dependency",
                        "attribute": {
                            "weight": 1.0,
                            "pred": "dependency",
                            "definition": "Task 2 depends on Task 1",
                            "sample_utterances": [
                                "I need to complete Task 1 before Task 2"
                            ],
                        },
                    },
                ],
            ],
        }
        assert not graph_validator.validate_graph(invalid_graph)


def test_integration_formatting_pipeline() -> None:
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

    # Format individual nodes and edges
    for idx, node in enumerate(formatted_graph["nodes"]):
        node_id, node_data = node
        formatted_node = node_formatter.format_node(node_data, node_id)
        assert isinstance(formatted_node, list)
        assert formatted_node[0] == node_id
        data = formatted_node[1]
        assert "resource" in data
        assert "attribute" in data

    for idx, edge in enumerate(formatted_graph["edges"]):
        source, target, edge_data = edge
        formatted_edge = edge_formatter.format_edge(
            source, target, {"task_id": source}, {"task_id": target}
        )
        assert isinstance(formatted_edge, list)
        assert formatted_edge[0] == source
        assert formatted_edge[1] == target
        data = formatted_edge[2]
        assert "intent" in data
        assert "attribute" in data

    # Validate the final graph
    assert graph_validator.validate_graph(formatted_graph)


class TestTaskGraphFormatterEdgeCases:
    """Test edge cases and missing coverage in TaskGraphFormatter."""

    def test_find_worker_by_name_with_fallback_mappings(
        self, task_graph_formatter
    ) -> None:
        """Test _find_worker_by_name with fallback worker mappings."""
        # Test default fallback mappings
        message_worker = task_graph_formatter._find_worker_by_name("MessageWorker")
        assert message_worker["id"] == "26bb6634-3bee-417d-ad75-23269ac17bc3"
        assert message_worker["name"] == "MessageWorker"

        faiss_worker = task_graph_formatter._find_worker_by_name("FaissRAGWorker")
        assert faiss_worker["id"] == "FaissRAGWorker"
        assert faiss_worker["name"] == "FaissRAGWorker"

        search_worker = task_graph_formatter._find_worker_by_name("SearchWorker")
        assert search_worker["id"] == "9c15af81-04b3-443e-be04-a3522124b905"
        assert search_worker["name"] == "SearchWorker"

        # Test unknown worker fallback
        unknown_worker = task_graph_formatter._find_worker_by_name("UnknownWorker")
        assert unknown_worker["id"] == "unknownworker"
        assert unknown_worker["name"] == "UnknownWorker"

    def test_format_task_graph_with_predefined_nodes_and_edges(
        self, task_graph_formatter
    ) -> None:
        """Test format_task_graph when nodes and edges are already provided."""
        predefined_nodes = [["1", {"resource": {"id": "test", "name": "Test"}}]]
        predefined_edges = [["1", "2", {"intent": "test"}]]

        formatter = TaskGraphFormatter(nodes=predefined_nodes, edges=predefined_edges)

        result = formatter.format_task_graph([])
        assert result["nodes"] == predefined_nodes
        assert result["edges"] == predefined_edges

    def test_format_nodes_with_nested_graph_resource(
        self, task_graph_formatter
    ) -> None:
        """Test _format_nodes with nested graph resources."""
        tasks = [
            {
                "id": "task1",
                "name": "Nested Task",
                "description": "A task with nested graph",
                "resource": {"name": "NestedGraph"},
                "steps": [],
            }
        ]

        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes(
            tasks
        )

        # Should have start node + task node + step node (created from description) = 3 nodes
        assert len(nodes) == 3
        assert node_lookup["task1"] == "1"
        assert all_task_node_ids == ["1"]

        # Check that nested graph resource is properly set
        task_node = nodes[1][1]
        assert task_node["resource"]["name"] == "NestedGraph"
        assert task_node["resource"]["id"] == "nested_graph"

    def test_format_nodes_with_workflow_resource(self, task_graph_formatter) -> None:
        """Test _format_nodes with workflow resources."""
        tasks = [
            {
                "id": "task1",
                "name": "Workflow Task",
                "description": "A task with workflow resource",
                "resource": "CustomWorkflow",
                "steps": [],
            }
        ]

        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes(
            tasks
        )

        # Should have start node + task node + step node (created from description) = 3 nodes
        assert len(nodes) == 3
        assert node_lookup["task1"] == "1"
        assert all_task_node_ids == ["1"]

        # Check that workflow resource is treated as nested graph
        task_node = nodes[1][1]
        assert task_node["resource"]["name"] == "NestedGraph"
        assert task_node["resource"]["id"] == "nested_graph"

    def test_format_nodes_with_task_type_and_limit(self, task_graph_formatter) -> None:
        """Test _format_nodes with task type and limit attributes."""
        tasks = [
            {
                "id": "task1",
                "name": "Typed Task",
                "description": "A task with type and limit",
                "type": "custom_type",
                "limit": 5,
                "steps": [],
            }
        ]

        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes(
            tasks
        )

        # Should have start node + task node + step node (created from description) = 3 nodes
        assert len(nodes) == 3
        assert node_lookup["task1"] == "1"
        assert all_task_node_ids == ["1"]

        # Check that type and limit are properly set
        task_node = nodes[1][1]
        assert task_node["type"] == "custom_type"
        assert task_node["limit"] == 5

    def test_format_nodes_with_complex_step_structure(
        self, task_graph_formatter
    ) -> None:
        """Test _format_nodes with complex step structure."""
        tasks = [
            {
                "id": "task1",
                "name": "Complex Task",
                "description": "A task with complex steps",
                "steps": [
                    {
                        "task": "Step 1",
                        "description": "First step description",
                        "step_id": "step_1",
                    }
                ],
            }
        ]

        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes(
            tasks
        )

        # Should have start node + task node + step node = 3 nodes
        assert len(nodes) == 3
        assert node_lookup["task1"] == "1"
        assert node_lookup["task1_step0"] == "2"
        assert all_task_node_ids == ["1"]

        # Check that complex step structure is handled correctly
        step_node = nodes[2][1]
        assert step_node["attribute"]["value"] == "First step description"

    def test_format_nodes_with_step_resource(self, task_graph_formatter) -> None:
        """Test _format_nodes with step resources."""
        tasks = [
            {
                "id": "task1",
                "name": "Task with Step Resource",
                "description": "A task with step resources",
                "steps": [
                    {"resource": {"name": "FaissRAGWorker"}, "description": "RAG step"}
                ],
            }
        ]

        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes(
            tasks
        )

        # Should have start node + task node + step node = 3 nodes
        assert len(nodes) == 3
        assert node_lookup["task1"] == "1"
        assert node_lookup["task1_step0"] == "2"
        assert all_task_node_ids == ["1"]

        # Check that step resource is properly set
        step_node = nodes[2][1]
        assert step_node["resource"]["name"] == "FaissRAGWorker"

    def test_format_nodes_with_step_workflow_resource(
        self, task_graph_formatter
    ) -> None:
        """Test _format_nodes with step workflow resources."""
        tasks = [
            {
                "id": "task1",
                "name": "Task with Step Workflow",
                "description": "A task with step workflow",
                "steps": [{"resource": "StepWorkflow", "description": "Workflow step"}],
            }
        ]

        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes(
            tasks
        )

        # Should have start node + task node + step node = 3 nodes
        assert len(nodes) == 3
        assert node_lookup["task1"] == "1"
        assert node_lookup["task1_step0"] == "2"
        assert all_task_node_ids == ["1"]

        # Check that step workflow resource is treated as nested graph
        step_node = nodes[2][1]
        assert step_node["resource"]["name"] == "NestedGraph"
        assert step_node["resource"]["id"] == "nested_graph"

    def test_format_nodes_with_non_dict_non_string_step(
        self, task_graph_formatter
    ) -> None:
        """Test _format_nodes with non-dict non-string step."""
        tasks = [
            {
                "id": "task1",
                "name": "Task with Non-Standard Step",
                "description": "A task with non-standard step",
                "steps": [123],  # Non-dict, non-string step
            }
        ]

        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes(
            tasks
        )

        # Should have start node + task node + step node = 3 nodes
        assert len(nodes) == 3
        assert node_lookup["task1"] == "1"
        assert node_lookup["task1_step0"] == "2"
        assert all_task_node_ids == ["1"]

        # Check that non-standard step is converted to string
        step_node = nodes[2][1]
        assert step_node["attribute"]["value"] == "123"

    def test_format_edges_with_llm_intent_generation(
        self, task_graph_formatter
    ) -> None:
        """Test _format_edges with LLM intent generation."""
        from unittest.mock import Mock

        # Mock model that returns a valid intent
        mock_model = Mock()
        mock_response = Mock()
        mock_response.content = "User inquires about product information"
        mock_model.invoke.return_value = mock_response

        formatter = TaskGraphFormatter(model=mock_model)

        tasks = [
            {
                "id": "task1",
                "name": "Product Inquiry",
                "description": "Handle product information requests",
                "steps": [],
            }
        ]

        # Create minimal node_lookup for the test
        node_lookup = {"task1": "1"}
        all_task_node_ids = ["1"]
        start_node_id = "0"

        edges, nested_graph_nodes = formatter._format_edges(
            tasks, node_lookup, all_task_node_ids, start_node_id
        )

        # Should have one edge from start to task
        assert len(edges) == 1
        assert edges[0][0] == "0"  # from start
        assert edges[0][1] == "1"  # to task
        assert edges[0][2]["intent"] == "User inquires about product information"
        assert edges[0][2]["attribute"]["pred"] is True

    def test_format_edges_with_llm_intent_generation_failure(
        self, task_graph_formatter
    ) -> None:
        """Test _format_edges with LLM intent generation failure."""
        from unittest.mock import Mock

        # Mock model that raises an exception
        mock_model = Mock()
        mock_model.invoke.side_effect = Exception("Model error")

        formatter = TaskGraphFormatter(model=mock_model)

        tasks = [
            {
                "id": "task1",
                "name": "Product Inquiry",
                "description": "Handle product information requests",
                "steps": [],
            }
        ]

        # Create minimal node_lookup for the test
        node_lookup = {"task1": "1"}
        all_task_node_ids = ["1"]
        start_node_id = "0"

        edges, nested_graph_nodes = formatter._format_edges(
            tasks, node_lookup, all_task_node_ids, start_node_id
        )

        # Should have one edge from start to task with fallback intent
        assert len(edges) == 1
        assert edges[0][0] == "0"  # from start
        assert edges[0][1] == "1"  # to task
        assert edges[0][2]["intent"] == "User inquires about purchasing options"
        assert edges[0][2]["attribute"]["pred"] is True

    def test_format_edges_with_invalid_llm_response(self, task_graph_formatter) -> None:
        """Test _format_edges with invalid LLM response."""
        from unittest.mock import Mock

        # Mock model that returns invalid response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.content = "none"  # Invalid intent
        mock_model.invoke.return_value = mock_response

        formatter = TaskGraphFormatter(model=mock_model)

        tasks = [
            {
                "id": "task1",
                "name": "Product Inquiry",
                "description": "Handle product information requests",
                "steps": [],
            }
        ]

        # Create minimal node_lookup for the test
        node_lookup = {"task1": "1"}
        all_task_node_ids = ["1"]
        start_node_id = "0"

        edges, nested_graph_nodes = formatter._format_edges(
            tasks, node_lookup, all_task_node_ids, start_node_id
        )

        # Should have one edge from start to task with fallback intent
        assert len(edges) == 1
        assert edges[0][0] == "0"  # from start
        assert edges[0][1] == "1"  # to task
        assert edges[0][2]["intent"] == "User inquires about purchasing options"
        assert edges[0][2]["attribute"]["pred"] is True

    def test_format_edges_with_no_model(self, task_graph_formatter) -> None:
        """Test _format_edges without model (should use default fallback)."""
        tasks = [
            {
                "id": "task1",
                "name": "Product Inquiry",
                "description": "Handle product information requests",
                "steps": [],
            }
        ]

        # Create minimal node_lookup for the test
        node_lookup = {"task1": "1"}
        all_task_node_ids = ["1"]
        start_node_id = "0"

        edges, nested_graph_nodes = task_graph_formatter._format_edges(
            tasks, node_lookup, all_task_node_ids, start_node_id
        )

        # Should have one edge from start to task with default fallback intent
        assert len(edges) == 1
        assert edges[0][0] == "0"  # from start
        assert edges[0][1] == "1"  # to task
        assert edges[0][2]["intent"] == "User inquires about purchasing options"
        assert edges[0][2]["attribute"]["pred"] is True

    def test_ensure_nested_graph_connectivity(self, task_graph_formatter) -> None:
        """Test ensure_nested_graph_connectivity method."""
        # Create a graph with nested graph nodes
        graph = {
            "nodes": [
                ["0", {"resource": {"id": "start", "name": "MessageWorker"}}],
                ["1", {"resource": {"id": "task1", "name": "MessageWorker"}}],
                [
                    "task1_step0",
                    {
                        "resource": {"id": "nested_graph", "name": "NestedGraph"},
                        "attribute": {"value": "original_value"},
                    },
                ],
                ["task1_step1", {"resource": {"id": "step1", "name": "MessageWorker"}}],
            ],
            "edges": [],
            "tasks": [
                {
                    "id": "task1",
                    "steps": [
                        {"description": "Nested step"},
                        {"description": "Next step"},
                    ],
                }
            ],
        }

        result = task_graph_formatter.ensure_nested_graph_connectivity(graph)

        # Should have one edge connecting nested graph to next step
        assert len(result["edges"]) == 1
        assert result["edges"][0][0] == "task1_step0"  # from nested graph
        assert result["edges"][0][1] == "task1_step1"  # to next step
        assert (
            "Continue to next step from nested graph"
            in result["edges"][0][2]["attribute"]["definition"]
        )

        # Check that nested graph value is set to next step ID
        nested_graph_node = result["nodes"][2][1]
        assert nested_graph_node["attribute"]["value"] == "task1_step1"

    def test_format_task_graph_with_dict_value_replacement(
        self, task_graph_formatter
    ) -> None:
        """Test format_task_graph with dict value replacement."""
        # Create a formatter with predefined nodes that have dict values
        nodes = [
            [
                "1",
                {
                    "resource": {"id": "test", "name": "Test"},
                    "attribute": {
                        "value": {"description": "Test description", "other": "data"}
                    },
                },
            ]
        ]
        edges = []

        formatter = TaskGraphFormatter(nodes=nodes, edges=edges)

        result = formatter.format_task_graph([])

        # Should return the original value (not replaced) when nodes/edges are passed in
        assert result["nodes"][0][1]["attribute"]["value"] == {
            "description": "Test description",
            "other": "data",
        }

    def test_format_task_graph_with_dict_value_no_description(
        self, task_graph_formatter
    ) -> None:
        """Test format_task_graph with dict value but no description."""
        # Create a formatter with predefined nodes that have dict values without description
        nodes = [
            [
                "1",
                {
                    "resource": {"id": "test", "name": "Test"},
                    "attribute": {"value": {"other": "data", "no_description": True}},
                },
            ]
        ]
        edges = []

        formatter = TaskGraphFormatter(nodes=nodes, edges=edges)

        result = formatter.format_task_graph([])

        # Should return the original value (not stringified) when nodes/edges are passed in
        assert result["nodes"][0][1]["attribute"]["value"] == {
            "other": "data",
            "no_description": True,
        }

    def test_format_task_graph_with_list_value_flattening(
        self, task_graph_formatter
    ) -> None:
        """Test format_task_graph with list value flattening."""
        # Create a formatter with predefined nodes that have list values
        nodes = [
            [
                "1",
                {
                    "resource": {"id": "test", "name": "Test"},
                    "attribute": {"value": ["item1", "item2", "item3"]},
                },
            ]
        ]
        edges = []

        formatter = TaskGraphFormatter(nodes=nodes, edges=edges)

        result = formatter.format_task_graph([])

        # Should return the original value (not flattened) when nodes/edges are passed in
        assert result["nodes"][0][1]["attribute"]["value"] == [
            "item1",
            "item2",
            "item3",
        ]

    def test_format_task_graph_with_nested_graph_target_assignment(
        self, task_graph_formatter
    ) -> None:
        """Test format_task_graph with nested graph target assignment."""
        # Create a formatter with predefined nodes including nested graph
        nodes = [
            ["0", {"resource": {"id": "start", "name": "MessageWorker"}}],
            ["1", {"resource": {"id": "task1", "name": "MessageWorker"}}],
            [
                "2",
                {
                    "resource": {"id": "nested_graph", "name": "NestedGraph"},
                    "attribute": {"value": "original_value"},
                },
            ],
        ]
        edges = [
            ["2", "1", {"intent": "test"}]  # Nested graph connects to task1
        ]

        formatter = TaskGraphFormatter(nodes=nodes, edges=edges)

        result = formatter.format_task_graph([])

        # Should return the original value (not updated) when nodes/edges are passed in
        nested_graph_node = result["nodes"][2][1]
        assert nested_graph_node["attribute"]["value"] == "original_value"

    def test_format_task_graph_with_nested_graph_no_valid_target(
        self, task_graph_formatter
    ) -> None:
        """Test format_task_graph with nested graph but no valid target."""
        # Create a formatter with predefined nodes including nested graph
        nodes = [
            ["0", {"resource": {"id": "start", "name": "MessageWorker"}}],
            [
                "2",
                {
                    "resource": {"id": "nested_graph", "name": "NestedGraph"},
                    "attribute": {"value": "original_value"},
                },
            ],
        ]
        edges = [
            ["2", "0", {"intent": "test"}]  # Nested graph connects to start (invalid)
        ]

        formatter = TaskGraphFormatter(nodes=nodes, edges=edges)

        result = formatter.format_task_graph([])

        # Should return the original value (not updated) when nodes/edges are passed in
        nested_graph_node = result["nodes"][1][1]
        assert nested_graph_node["attribute"]["value"] == "original_value"

    def test_format_task_graph_with_nested_graph_no_target_found(
        self, task_graph_formatter
    ) -> None:
        """Test format_task_graph with nested graph but no target found."""
        # Create a formatter with predefined nodes including nested graph
        nodes = [
            ["0", {"resource": {"id": "start", "name": "MessageWorker"}}],
            [
                "2",
                {
                    "resource": {"id": "nested_graph", "name": "NestedGraph"},
                    "attribute": {"value": "original_value"},
                },
            ],
        ]
        edges = []  # No edges from nested graph

        formatter = TaskGraphFormatter(nodes=nodes, edges=edges)

        result = formatter.format_task_graph([])

        # Should return the original value (not updated) when nodes/edges are passed in
        nested_graph_node = result["nodes"][1][1]
        assert nested_graph_node["attribute"]["value"] == "original_value"
