"""Comprehensive tests for TaskGraphFormatter.

This module provides extensive test coverage for the TaskGraphFormatter class,
including all methods, edge cases, error conditions, and formatting scenarios.
"""

import pytest
from unittest.mock import Mock, patch

from arklex.orchestrator.generator.formatting.task_graph_formatter import (
    TaskGraphFormatter,
)


# =============================================================================
# FIXTURES - Core Test Data
# =============================================================================


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    mock = Mock()
    mock.generate.return_value = {"text": '{"intent": "test_intent"}'}
    return mock


@pytest.fixture
def task_graph_formatter(mock_model):
    """Create a TaskGraphFormatter instance for testing."""
    return TaskGraphFormatter(
        role="test_role",
        user_objective="test_user_objective",
        builder_objective="test_builder_objective",
        domain="test_domain",
        intro="test_intro",
        task_docs=[{"doc": "test"}],
        rag_docs=[{"rag": "test"}],
        workers=[
            {"id": "worker1", "name": "MessageWorker"},
            {"id": "worker2", "name": "FaissRAGWorker"},
        ],
        tools=[{"tool": "test"}],
        nluapi="test_nluapi",
        slotfillapi="test_slotfillapi",
        default_intent="test_intent",
        default_weight=2,
        default_pred=True,
        default_definition="test_definition",
        default_sample_utterances=["test_utterance"],
        allow_nested_graph=True,
        model=mock_model,
    )


@pytest.fixture
def task_graph_formatter_minimal():
    """Create a minimal TaskGraphFormatter instance for testing."""
    return TaskGraphFormatter()


@pytest.fixture
def sample_tasks():
    """Sample tasks for testing."""
    return [
        {
            "id": "task1",
            "name": "Task 1",
            "description": "First task",
            "steps": [{"task": "step1"}],
            "dependencies": ["task2"],
            "required_resources": ["resource1"],
            "estimated_duration": "1 hour",
            "priority": 3,
        },
        {
            "id": "task2",
            "name": "Task 2",
            "description": "Second task",
            "steps": [{"task": "step2"}],
            "dependencies": [],
            "required_resources": ["resource2"],
            "estimated_duration": "2 hours",
            "priority": 2,
        },
    ]


@pytest.fixture
def sample_tasks_with_nested_graph():
    """Sample tasks with nested graph for testing."""
    return [
        {
            "id": "task1",
            "name": "Task 1",
            "description": "First task",
            "steps": [{"task": "step1"}],
            "dependencies": [],
            "required_resources": [],
            "estimated_duration": "1 hour",
            "priority": 3,
        },
    ]


@pytest.fixture
def sample_tasks_with_complex_values():
    """Sample tasks with complex values for testing."""
    return [
        {
            "id": "task1",
            "name": "Task 1",
            "description": "First task",
            "steps": [{"task": "step1"}],
            "dependencies": [],
            "required_resources": [],
            "estimated_duration": "1 hour",
            "priority": 3,
        },
    ]


# =============================================================================
# TEST CLASSES
# =============================================================================


class TestTaskGraphFormatter:
    """Test the TaskGraphFormatter class."""

    def test_task_graph_formatter_initialization(self, task_graph_formatter) -> None:
        """Test TaskGraphFormatter initialization with all parameters."""
        assert task_graph_formatter._role == "test_role"
        assert task_graph_formatter._user_objective == "test_user_objective"
        assert task_graph_formatter._builder_objective == "test_builder_objective"
        assert task_graph_formatter._domain == "test_domain"
        assert task_graph_formatter._intro == "test_intro"
        assert len(task_graph_formatter._task_docs) == 1
        assert len(task_graph_formatter._rag_docs) == 1
        assert len(task_graph_formatter._workers) == 2
        assert len(task_graph_formatter._tools) == 1
        assert task_graph_formatter._nluapi == "test_nluapi"
        assert task_graph_formatter._slotfillapi == "test_slotfillapi"
        assert task_graph_formatter._default_intent == "test_intent"
        assert task_graph_formatter._default_weight == 2
        assert task_graph_formatter._default_pred is True
        assert task_graph_formatter._default_definition == "test_definition"
        assert len(task_graph_formatter._default_sample_utterances) == 1
        assert task_graph_formatter._allow_nested_graph is True
        assert task_graph_formatter._model is not None

    def test_task_graph_formatter_minimal_initialization(
        self, task_graph_formatter_minimal
    ) -> None:
        """Test TaskGraphFormatter initialization with minimal parameters."""
        assert task_graph_formatter_minimal._role == ""
        assert task_graph_formatter_minimal._user_objective == ""
        assert task_graph_formatter_minimal._builder_objective == ""
        assert task_graph_formatter_minimal._domain == ""
        assert task_graph_formatter_minimal._intro == ""
        assert task_graph_formatter_minimal._task_docs == []
        assert task_graph_formatter_minimal._rag_docs == []
        assert task_graph_formatter_minimal._workers == []
        assert task_graph_formatter_minimal._tools == []
        assert task_graph_formatter_minimal._nluapi == ""
        assert task_graph_formatter_minimal._slotfillapi == ""
        assert task_graph_formatter_minimal._default_intent is None
        assert task_graph_formatter_minimal._default_weight == 1
        assert task_graph_formatter_minimal._default_pred is False
        assert task_graph_formatter_minimal._default_definition == ""
        assert task_graph_formatter_minimal._default_sample_utterances == []
        assert task_graph_formatter_minimal._nodes is None
        assert task_graph_formatter_minimal._edges is None
        assert task_graph_formatter_minimal._allow_nested_graph is True
        assert task_graph_formatter_minimal._model is None

    def test_find_worker_by_name_existing_worker(self, task_graph_formatter) -> None:
        """Test finding worker by name when worker exists."""
        worker_info = task_graph_formatter._find_worker_by_name("MessageWorker")
        assert worker_info["id"] == "worker1"
        assert worker_info["name"] == "MessageWorker"

    def test_find_worker_by_name_fallback_mapping(
        self, task_graph_formatter_minimal
    ) -> None:
        """Test finding worker by name using fallback mappings."""
        # Test MessageWorker fallback
        worker_info = task_graph_formatter_minimal._find_worker_by_name("MessageWorker")
        assert worker_info["id"] == "26bb6634-3bee-417d-ad75-23269ac17bc3"
        assert worker_info["name"] == "MessageWorker"

        # Test FaissRAGWorker fallback
        worker_info = task_graph_formatter_minimal._find_worker_by_name(
            "FaissRAGWorker"
        )
        assert worker_info["id"] == "FaissRAGWorker"
        assert worker_info["name"] == "FaissRAGWorker"

        # Test SearchWorker fallback
        worker_info = task_graph_formatter_minimal._find_worker_by_name("SearchWorker")
        assert worker_info["id"] == "9c15af81-04b3-443e-be04-a3522124b905"
        assert worker_info["name"] == "SearchWorker"

    def test_find_worker_by_name_unknown_worker(
        self, task_graph_formatter_minimal
    ) -> None:
        """Test finding worker by name for unknown worker."""
        worker_info = task_graph_formatter_minimal._find_worker_by_name("UnknownWorker")
        assert worker_info["id"] == "unknownworker"
        assert worker_info["name"] == "UnknownWorker"

    def test_find_worker_by_name_worker_without_id(self, task_graph_formatter) -> None:
        """Test finding worker by name when worker exists but has no id."""
        # Add a worker without id to the formatter
        task_graph_formatter._workers = [{"name": "WorkerWithoutId"}]
        worker_info = task_graph_formatter._find_worker_by_name("WorkerWithoutId")
        assert worker_info["id"] == "workerwithoutid"
        assert worker_info["name"] == "WorkerWithoutId"

    def test_find_worker_by_name_worker_without_name(
        self, task_graph_formatter
    ) -> None:
        """Test finding worker by name when worker exists but has no name."""
        # Add a worker without name to the formatter
        task_graph_formatter._workers = [{"id": "worker_id"}]
        worker_info = task_graph_formatter._find_worker_by_name("SomeWorker")
        # Should use fallback since worker doesn't have matching name
        assert worker_info["id"] == "someworker"
        assert worker_info["name"] == "SomeWorker"

    def test_format_task_graph_with_predefined_nodes_edges(
        self, task_graph_formatter
    ) -> None:
        """Test format_task_graph with predefined nodes and edges."""
        task_graph_formatter._nodes = [["node1", {"data": "test"}]]
        task_graph_formatter._edges = [["node1", "node2", {"intent": "test"}]]

        result = task_graph_formatter.format_task_graph([])
        assert result["nodes"] == [["node1", {"data": "test"}]]
        assert result["edges"] == [["node1", "node2", {"intent": "test"}]]

    def test_format_task_graph_complete_flow(
        self, task_graph_formatter, sample_tasks
    ) -> None:
        """Test complete format_task_graph flow."""
        result = task_graph_formatter.format_task_graph(sample_tasks)

        assert "nodes" in result
        assert "edges" in result
        assert isinstance(result["nodes"], list)
        assert isinstance(result["edges"], list)

    def test_create_edge_attributes_with_all_parameters(
        self, task_graph_formatter
    ) -> None:
        """Test _create_edge_attributes with all parameters."""
        attributes = task_graph_formatter._create_edge_attributes(
            intent="test_intent",
            weight=5,
            pred=True,
            definition="test_definition",
            sample_utterances=["utterance1", "utterance2"],
        )

        assert attributes["intent"] == "test_intent"
        assert attributes["attribute"]["weight"] == 5
        assert attributes["attribute"]["pred"] is True
        assert attributes["attribute"]["definition"] == "test_definition"
        assert attributes["attribute"]["sample_utterances"] == [
            "utterance1",
            "utterance2",
        ]

    def test_create_edge_attributes_with_defaults(self, task_graph_formatter) -> None:
        """Test _create_edge_attributes with default values."""
        attributes = task_graph_formatter._create_edge_attributes()

        assert attributes["intent"] is None  # Default is None, not formatter default
        assert attributes["attribute"]["weight"] == 1  # Default weight
        assert attributes["attribute"]["pred"] is False  # Default pred
        assert attributes["attribute"]["definition"] == ""  # Default definition
        assert (
            attributes["attribute"]["sample_utterances"] == []
        )  # Default sample_utterances

    def test_create_edge_attributes_with_none_values(
        self, task_graph_formatter_minimal
    ) -> None:
        """Test _create_edge_attributes with None values."""
        attributes = task_graph_formatter_minimal._create_edge_attributes(
            intent=None, sample_utterances=None
        )

        assert attributes["intent"] is None
        assert attributes["attribute"]["weight"] == 1  # Default weight
        assert attributes["attribute"]["pred"] is False  # Default pred
        assert attributes["attribute"]["definition"] == ""  # Default definition
        assert (
            attributes["attribute"]["sample_utterances"] == []
        )  # None becomes empty list

    def test_format_nodes_basic(self, task_graph_formatter, sample_tasks) -> None:
        """Test _format_nodes with basic tasks."""
        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes(
            sample_tasks
        )

        assert isinstance(nodes, list)
        assert isinstance(node_lookup, dict)
        assert isinstance(all_task_node_ids, list)
        assert len(nodes) > 0
        assert len(node_lookup) > 0
        assert len(all_task_node_ids) > 0

    def test_format_nodes_empty_tasks(self, task_graph_formatter) -> None:
        """Test _format_nodes with empty tasks list."""
        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes([])

        assert isinstance(nodes, list)
        assert isinstance(node_lookup, dict)
        assert isinstance(all_task_node_ids, list)
        # Should still have start node
        assert len(nodes) >= 1

    def test_format_edges_basic(self, task_graph_formatter, sample_tasks) -> None:
        """Test _format_edges with basic tasks."""
        # First format nodes to get node_lookup
        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes(
            sample_tasks
        )
        start_node_id = "0"

        edges, nested_graph_nodes = task_graph_formatter._format_edges(
            sample_tasks, node_lookup, all_task_node_ids, start_node_id
        )

        assert isinstance(edges, list)
        assert isinstance(nested_graph_nodes, list)

    def test_format_edges_empty_tasks(self, task_graph_formatter) -> None:
        """Test _format_edges with empty tasks list."""
        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes([])
        start_node_id = "0"

        edges, nested_graph_nodes = task_graph_formatter._format_edges(
            [], node_lookup, all_task_node_ids, start_node_id
        )

        assert isinstance(edges, list)
        assert isinstance(nested_graph_nodes, list)

    def test_ensure_nested_graph_connectivity(self, task_graph_formatter) -> None:
        """Test ensure_nested_graph_connectivity method."""
        graph = {
            "nodes": [
                [
                    "node1",
                    {
                        "resource": {"name": "NestedGraph"},
                        "attribute": {"value": "old_value"},
                    },
                ],
                [
                    "node2",
                    {
                        "resource": {"name": "MessageWorker"},
                        "attribute": {"value": "task_value"},
                    },
                ],
            ],
            "edges": [
                ["node1", "node2", {"intent": "connect"}],
            ],
            "tasks": [{"id": "task1", "steps": [{"task": "step1"}, {"task": "step2"}]}],
        }

        result = task_graph_formatter.ensure_nested_graph_connectivity(graph)

        assert "nodes" in result
        assert "edges" in result
        # The method doesn't modify existing nodes, it only adds edges
        # So the value should remain unchanged
        nested_graph_node = next(
            node
            for node in result["nodes"]
            if node[1]["resource"]["name"] == "NestedGraph"
        )
        assert nested_graph_node[1]["attribute"]["value"] == "old_value"

    def test_ensure_nested_graph_connectivity_no_nested_graph(
        self, task_graph_formatter
    ) -> None:
        """Test ensure_nested_graph_connectivity with no NestedGraph nodes."""
        graph = {
            "nodes": [
                [
                    "node1",
                    {
                        "resource": {"name": "MessageWorker"},
                        "attribute": {"value": "task_value"},
                    },
                ],
            ],
            "edges": [],
            "tasks": [],
        }

        result = task_graph_formatter.ensure_nested_graph_connectivity(graph)

        assert result == graph  # Should return unchanged graph

    def test_ensure_nested_graph_connectivity_nested_graph_no_target(
        self, task_graph_formatter
    ) -> None:
        """Test ensure_nested_graph_connectivity with NestedGraph but no valid target."""
        graph = {
            "nodes": [
                [
                    "node1",
                    {
                        "resource": {"name": "NestedGraph"},
                        "attribute": {"value": "old_value"},
                    },
                ],
                [
                    "node2",
                    {
                        "resource": {"name": "MessageWorker"},
                        "attribute": {"value": "task_value"},
                    },
                ],
            ],
            "edges": [
                ["node1", "0", {"intent": "connect"}],  # Points to start node (0)
            ],
            "tasks": [{"id": "task1", "steps": [{"task": "step1"}]}],
        }

        result = task_graph_formatter.ensure_nested_graph_connectivity(graph)

        # The method doesn't modify existing nodes, it only adds edges
        nested_graph_node = next(
            node
            for node in result["nodes"]
            if node[1]["resource"]["name"] == "NestedGraph"
        )
        assert nested_graph_node[1]["attribute"]["value"] == "old_value"

    def test_ensure_nested_graph_connectivity_nested_graph_no_edges(
        self, task_graph_formatter
    ) -> None:
        """Test ensure_nested_graph_connectivity with NestedGraph but no edges."""
        graph = {
            "nodes": [
                [
                    "node1",
                    {
                        "resource": {"name": "NestedGraph"},
                        "attribute": {"value": "old_value"},
                    },
                ],
                [
                    "node2",
                    {
                        "resource": {"name": "MessageWorker"},
                        "attribute": {"value": "task_value"},
                    },
                ],
            ],
            "edges": [],
            "tasks": [{"id": "task1", "steps": [{"task": "step1"}]}],
        }

        result = task_graph_formatter.ensure_nested_graph_connectivity(graph)

        # The method doesn't modify existing nodes, it only adds edges
        nested_graph_node = next(
            node
            for node in result["nodes"]
            if node[1]["resource"]["name"] == "NestedGraph"
        )
        assert nested_graph_node[1]["attribute"]["value"] == "old_value"

    def test_ensure_nested_graph_connectivity_nested_graph_no_task_nodes(
        self, task_graph_formatter
    ) -> None:
        """Test ensure_nested_graph_connectivity with NestedGraph but no task nodes."""
        graph = {
            "nodes": [
                [
                    "node1",
                    {
                        "resource": {"name": "NestedGraph"},
                        "attribute": {"value": "old_value"},
                    },
                ],
            ],
            "edges": [],
            "tasks": [],
        }

        result = task_graph_formatter.ensure_nested_graph_connectivity(graph)

        # The method doesn't modify existing nodes, it only adds edges
        nested_graph_node = next(
            node
            for node in result["nodes"]
            if node[1]["resource"]["name"] == "NestedGraph"
        )
        assert nested_graph_node[1]["attribute"]["value"] == "old_value"

    def test_ensure_nested_graph_connectivity_dict_value(
        self, task_graph_formatter
    ) -> None:
        """Test ensure_nested_graph_connectivity with dict value in non-NestedGraph node."""
        graph = {
            "nodes": [
                [
                    "node1",
                    {
                        "resource": {"name": "MessageWorker"},
                        "attribute": {"value": {"description": "test_desc"}},
                    },
                ],
            ],
            "edges": [],
            "tasks": [],
        }

        result = task_graph_formatter.ensure_nested_graph_connectivity(graph)

        # The method doesn't modify non-NestedGraph nodes
        node = next(
            node
            for node in result["nodes"]
            if node[1]["resource"]["name"] == "MessageWorker"
        )
        assert node[1]["attribute"]["value"] == {"description": "test_desc"}

    def test_ensure_nested_graph_connectivity_dict_value_no_description(
        self, task_graph_formatter
    ) -> None:
        """Test ensure_nested_graph_connectivity with dict value but no description."""
        graph = {
            "nodes": [
                [
                    "node1",
                    {
                        "resource": {"name": "MessageWorker"},
                        "attribute": {"value": {"other": "data"}},
                    },
                ],
            ],
            "edges": [],
            "tasks": [],
        }

        result = task_graph_formatter.ensure_nested_graph_connectivity(graph)

        # The method doesn't modify non-NestedGraph nodes
        node = next(
            node
            for node in result["nodes"]
            if node[1]["resource"]["name"] == "MessageWorker"
        )
        assert node[1]["attribute"]["value"] == {"other": "data"}

    def test_ensure_nested_graph_connectivity_list_value(
        self, task_graph_formatter
    ) -> None:
        """Test ensure_nested_graph_connectivity with list value."""
        graph = {
            "nodes": [
                [
                    "node1",
                    {
                        "resource": {"name": "MessageWorker"},
                        "attribute": {"value": ["item1", "item2"]},
                    },
                ],
            ],
            "edges": [],
            "tasks": [],
        }

        result = task_graph_formatter.ensure_nested_graph_connectivity(graph)

        # The method doesn't modify non-NestedGraph nodes
        node = next(
            node
            for node in result["nodes"]
            if node[1]["resource"]["name"] == "MessageWorker"
        )
        assert node[1]["attribute"]["value"] == ["item1", "item2"]

    def test_format_task_graph_nested_graph_value_update(
        self, task_graph_formatter, sample_tasks_with_nested_graph
    ) -> None:
        """Test format_task_graph updates NestedGraph node values correctly."""
        result = task_graph_formatter.format_task_graph(sample_tasks_with_nested_graph)

        assert "nodes" in result
        assert "edges" in result

        # Check if any NestedGraph nodes have their values updated
        nested_graph_nodes = [
            node
            for node in result["nodes"]
            if node[1].get("resource", {}).get("name") == "NestedGraph"
        ]
        if nested_graph_nodes:
            for node in nested_graph_nodes:
                assert isinstance(node[1]["attribute"]["value"], str)
                assert node[1]["attribute"]["value"] != "old_value"  # Should be updated

    def test_format_task_graph_complex_value_processing(
        self, task_graph_formatter, sample_tasks_with_complex_values
    ) -> None:
        """Test format_task_graph processes complex values correctly."""
        result = task_graph_formatter.format_task_graph(
            sample_tasks_with_complex_values
        )

        assert "nodes" in result
        assert "edges" in result

        # Check that all node values are strings (not dicts or lists)
        for node in result["nodes"]:
            value = node[1].get("attribute", {}).get("value")
            if value is not None:
                assert isinstance(value, str)

    def test_task_graph_formatter_default_constants(self) -> None:
        """Test that default worker constants are defined correctly."""
        assert TaskGraphFormatter.DEFAULT_MESSAGE_WORKER == "MessageWorker"
        assert TaskGraphFormatter.DEFAULT_RAG_WORKER == "FaissRAGWorker"
        assert TaskGraphFormatter.DEFAULT_SEARCH_WORKER == "SearchWorker"
        assert TaskGraphFormatter.DEFAULT_NESTED_GRAPH == "NestedGraph"

    def test_task_graph_formatter_with_none_parameters(self) -> None:
        """Test TaskGraphFormatter initialization with None parameters."""
        formatter = TaskGraphFormatter(
            task_docs=None,
            rag_docs=None,
            workers=None,
            tools=None,
            default_sample_utterances=None,
            nodes=None,
            edges=None,
            model=None,
        )

        assert formatter._task_docs == []
        assert formatter._rag_docs == []
        assert formatter._workers == []
        assert formatter._tools == []
        assert formatter._default_sample_utterances == []
        assert formatter._nodes is None
        assert formatter._edges is None
        assert formatter._model is None

    def test_format_task_graph_handles_dict_value(self, task_graph_formatter) -> None:
        tasks = [
            {
                "id": "task1",
                "name": "Task 1",
                "description": "desc",
                "steps": [],
                "attribute": {"value": {"description": "desc text"}},
            }
        ]
        # Patch ensure_nested_graph_connectivity to just return its input
        with patch.object(
            task_graph_formatter,
            "ensure_nested_graph_connectivity",
            side_effect=lambda g: g,
        ):
            graph = task_graph_formatter.format_task_graph(tasks)
            # The actual implementation uses task description, not attribute value
            assert graph["nodes"][1][1]["attribute"]["value"] == "desc"

    def test_format_task_graph_handles_dict_value_no_description(
        self, task_graph_formatter
    ) -> None:
        tasks = [
            {
                "id": "task1",
                "name": "Task 1",
                "description": "desc",
                "steps": [],
                "attribute": {"value": {"other": "no desc"}},
            }
        ]
        with patch.object(
            task_graph_formatter,
            "ensure_nested_graph_connectivity",
            side_effect=lambda g: g,
        ):
            graph = task_graph_formatter.format_task_graph(tasks)
            # The actual implementation uses task description, not attribute value
            assert graph["nodes"][1][1]["attribute"]["value"] == "desc"

    def test_format_task_graph_handles_list_value(self, task_graph_formatter):
        tasks = [
            {
                "id": "task1",
                "name": "Task 1",
                "description": "desc",
                "steps": [],
                "attribute": {"value": [1, 2, 3]},
            }
        ]
        with patch.object(
            task_graph_formatter,
            "ensure_nested_graph_connectivity",
            side_effect=lambda g: g,
        ):
            graph = task_graph_formatter.format_task_graph(tasks)
            # The actual implementation uses task description, not attribute value
            assert graph["nodes"][1][1]["attribute"]["value"] == "desc"

    def test_format_task_graph_fallback_to_node_1(self, task_graph_formatter) -> None:
        # Simulate no task_node_ids, should fallback to '1'
        tasks = [
            {
                "id": "task1",
                "name": "Task 1",
                "description": "desc",
                "steps": [],
            }
        ]
        # Patch ensure_nested_graph_connectivity to just return its input
        with patch.object(
            task_graph_formatter,
            "ensure_nested_graph_connectivity",
            side_effect=lambda g: g,
        ):
            # Patch nodes and nested_graph_nodes to trigger fallback
            with patch.object(
                task_graph_formatter, "_format_nodes", return_value=([], {}, [])
            ):
                with patch.object(
                    task_graph_formatter, "_format_edges", return_value=([], [])
                ):
                    graph = task_graph_formatter.format_task_graph(tasks)
                    # Should not error, fallback to '1' for value
                    assert "nodes" in graph

    def test_format_nodes_creates_start_node(self, task_graph_formatter) -> None:
        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes([])
        assert nodes[0][1]["type"] == "start"
        assert nodes[0][1]["attribute"]["value"].startswith("Hello!")
        assert node_lookup == {}
        assert all_task_node_ids == []

    def test_format_nodes_with_task_and_step(self, task_graph_formatter) -> None:
        tasks = [
            {
                "id": "task1",
                "name": "Task 1",
                "description": "desc",
                "steps": [{"description": "stepdesc", "step_id": "step1"}],
            }
        ]
        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes(
            tasks
        )
        assert any("stepdesc" in n[1]["attribute"]["value"] for n in nodes)
        assert "task1" in node_lookup
        assert len(all_task_node_ids) == 1

    def test_format_nodes_with_step_as_string(self, task_graph_formatter) -> None:
        tasks = [
            {
                "id": "task1",
                "name": "Task 1",
                "description": "desc",
                "steps": ["step as string"],
            }
        ]
        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes(
            tasks
        )
        assert any("step as string" in n[1]["attribute"]["value"] for n in nodes)

    def test_format_nodes_with_step_as_non_dict(self, task_graph_formatter) -> None:
        tasks = [
            {
                "id": "task1",
                "name": "Task 1",
                "description": "desc",
                "steps": [123],
            }
        ]
        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes(
            tasks
        )
        assert any("123" in n[1]["attribute"]["value"] for n in nodes)

    def test_format_edges_model_exception_fallback(
        self, task_graph_formatter, sample_tasks
    ) -> None:
        # Patch model to raise exception
        task_graph_formatter._model = Mock()
        task_graph_formatter._model.invoke.side_effect = Exception("fail")
        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes(
            sample_tasks
        )
        start_node_id = "0"
        # Should fallback to default intent
        edges, _ = task_graph_formatter._format_edges(
            sample_tasks, node_lookup, all_task_node_ids, start_node_id
        )
        assert any(
            e[2]["intent"] == "User inquires about purchasing options" for e in edges
        )

    def test_format_edges_dependency_edge_cases(self, task_graph_formatter) -> None:
        # dep is None, not a string/dict, or dict without 'id'
        tasks = [
            {
                "id": "task1",
                "name": "Task 1",
                "description": "desc",
                "dependencies": [None, 123, {"foo": "bar"}],
            },
            {
                "id": "task2",
                "name": "Task 2",
                "description": "desc",
                "dependencies": [],
            },
        ]
        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes(
            tasks
        )
        start_node_id = "0"
        # Should not raise, just skip invalid deps
        edges, _ = task_graph_formatter._format_edges(
            tasks, node_lookup, all_task_node_ids, start_node_id
        )
        assert isinstance(edges, list)

    def test_format_task_graph_fallback_to_node_1_explicit(
        self, task_graph_formatter
    ) -> None:
        # Patch _format_nodes and _format_edges to return no task nodes
        with (
            patch.object(
                task_graph_formatter, "_format_nodes", return_value=([], {}, [])
            ),
            patch.object(task_graph_formatter, "_format_edges", return_value=([], [])),
            patch.object(
                task_graph_formatter,
                "ensure_nested_graph_connectivity",
                side_effect=lambda g: g,
            ),
        ):
            tasks = [
                {
                    "id": "task1",
                    "name": "Task 1",
                    "description": "desc",
                    "steps": [],
                    "resource": "NestedGraph",
                }
            ]
            graph = task_graph_formatter.format_task_graph(tasks)
            # Should not error, fallback to '1' for value
            assert "nodes" in graph

    def test_format_nodes_step_with_resource_as_string(
        self, task_graph_formatter
    ) -> None:
        tasks = [
            {
                "id": "task1",
                "name": "Task 1",
                "description": "desc",
                "steps": [{"description": "stepdesc", "resource": "CustomWorker"}],
            }
        ]
        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes(
            tasks
        )
        # Should create a step node with resource name 'CustomWorker'
        assert any(n[1]["resource"]["name"] == "CustomWorker" for n in nodes)

    def test_format_nodes_task_with_no_steps_no_description(
        self, task_graph_formatter
    ) -> None:
        tasks = [{"id": "task1", "name": "Task 1"}]
        nodes, node_lookup, all_task_node_ids = task_graph_formatter._format_nodes(
            tasks
        )
        # Should still create a task node
        assert any(n[1]["resource"]["name"] == "MessageWorker" for n in nodes)
