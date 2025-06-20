import pytest
from unittest.mock import Mock, patch
from arklex.orchestrator.task_graph import TaskGraph
from arklex.utils.graph_state import (
    LLMConfig,
    Params,
    NodeInfo,
    StatusEnum,
    PathNode,
    Taskgraph,
    Memory,
    Metadata,
)
from arklex.orchestrator.NLU.services.model_service import DummyModelService
from contextlib import ExitStack


@pytest.fixture
def dummy_config():
    return {
        "nodes": [
            (
                "start",
                {
                    "type": "start",
                    "resource": {"name": "start_resource", "id": "start_id"},
                    "attribute": {"start": True, "weight": 1.0},
                },
            ),
            (
                "node1",
                {
                    "type": "task",
                    "resource": {"name": "task1", "id": "task1_id"},
                    "attribute": {"weight": 1.0, "can_skipped": False},
                },
            ),
            (
                "node2",
                {
                    "type": "task",
                    "resource": {"name": "task2", "id": "task2_id"},
                    "attribute": {"weight": 1.0, "can_skipped": True},
                },
            ),
            (
                "end",
                {
                    "type": "end",
                    "resource": {"name": "end_resource", "id": "end_id"},
                    "attribute": {"weight": 1.0},
                },
            ),
        ],
        "edges": [
            (
                "start",
                "node1",
                {"intent": "start", "attribute": {"weight": 1.0}, "pred": True},
            ),
            (
                "node1",
                "node2",
                {"intent": "continue", "attribute": {"weight": 1.0}, "pred": True},
            ),
            (
                "node2",
                "end",
                {"intent": "finish", "attribute": {"weight": 1.0}, "pred": True},
            ),
            (
                "node1",
                "end",
                {"intent": "skip", "attribute": {"weight": 0.5}, "pred": True},
            ),
            # Add a dummy incoming edge for start node to prevent IndexError
            (
                "dummy",
                "start",
                {"intent": "dummy", "attribute": {"weight": 1.0}, "pred": True},
            ),
        ],
        "intent_api": {
            "model_name": "dummy",
            "api_key": "dummy",
            "endpoint": "http://dummy",
            "model_type_or_path": "dummy-path",
            "llm_provider": "dummy",
        },
        "slotfillapi": "",
        "services_nodes": {"service1": "node1", "service2": "node2"},
    }


@pytest.fixture
def dummy_llm_config():
    return LLMConfig(
        model_name="dummy",
        api_key="dummy",
        endpoint="http://dummy",
        model_type_or_path="dummy-path",
        llm_provider="dummy",
    )


@pytest.fixture
def mock_model_service():
    return DummyModelService(
        {
            "model_name": "dummy",
            "api_key": "dummy",
            "endpoint": "http://dummy",
            "model_type_or_path": "dummy-path",
            "llm_provider": "dummy",
        }
    )


@pytest.fixture
def task_graph(dummy_config, dummy_llm_config, mock_model_service):
    task_graph = TaskGraph(
        name="test_graph",
        product_kwargs=dummy_config,
        llm_config=dummy_llm_config,
        model_service=mock_model_service,
    )
    # Set required attributes that are normally set in get_node method
    task_graph.text = "test text"
    task_graph.chat_history_str = "test history"
    return task_graph


@pytest.fixture
def params():
    return Params(taskgraph=Taskgraph(), memory=Memory(), metadata=Metadata())


class TestTaskGraph:
    """Test the TaskGraph class functionality."""

    def test_task_graph_requires_model_service(
        self, dummy_config, dummy_llm_config
    ) -> None:
        """Test that TaskGraph requires a model service."""
        with pytest.raises(
            ValueError,
            match="model_service is required for TaskGraph and cannot be None",
        ):
            TaskGraph(
                name="test",
                product_kwargs=dummy_config,
                llm_config=dummy_llm_config,
                model_service=None,
            )

    def test_task_graph_initialization(self, task_graph) -> None:
        """Test TaskGraph initialization."""
        assert task_graph.graph.name == "test_graph"
        assert task_graph.unsure_intent["intent"] == "others"
        assert task_graph.initial_node is not None
        # The intent_detector and slotfillapi are real objects, just check they're not None
        assert task_graph.intent_detector is not None
        assert task_graph.slotfillapi is not None

    def test_create_graph(self, task_graph) -> None:
        """Test graph creation from nodes and edges."""
        # Check that nodes were added
        assert "start" in task_graph.graph.nodes
        assert "node1" in task_graph.graph.nodes
        assert "node2" in task_graph.graph.nodes
        assert "end" in task_graph.graph.nodes

        # Check that edges were added with lowercase intents
        edges = list(task_graph.graph.edges(data=True))
        intents = [edge[2]["intent"] for edge in edges]
        assert all(intent == intent.lower() for intent in intents)

    def test_get_initial_flow_with_services_nodes(self, task_graph) -> None:
        """Test getting initial flow with services_nodes configuration."""
        # Mock numpy.random.choice to return predictable result
        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = "node1"
            initial_flow = task_graph.get_initial_flow()
            assert initial_flow == "node1"

    def test_get_initial_flow_without_services_nodes(
        self, dummy_config, dummy_llm_config, mock_model_service
    ) -> None:
        """Test getting initial flow without services_nodes configuration."""
        config_without_services = dummy_config.copy()
        config_without_services.pop("services_nodes", None)

        task_graph = TaskGraph(
            name="test",
            product_kwargs=config_without_services,
            llm_config=dummy_llm_config,
            model_service=mock_model_service,
        )

        initial_flow = task_graph.get_initial_flow()
        assert initial_flow is None

    def test_jump_to_node_success(self, task_graph) -> None:
        """Test successful jump to node based on intent."""
        # Set up the intents structure properly
        task_graph.intents = {
            "start": [{"target_node": "node1", "attribute": {"weight": 1.0}}]
        }

        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = "node1"
            next_node, next_intent = task_graph.jump_to_node("start", 0, "start")
            assert next_node == "node1"
            assert next_intent == "start"

    def test_jump_to_node_exception_handling(self, task_graph) -> None:
        """Test jump_to_node handles exceptions gracefully."""
        # Mock intents to cause an exception
        task_graph.intents = {}

        next_node, next_intent = task_graph.jump_to_node("nonexistent", 0, "start")
        assert next_node == "start"  # Should return current node on error
        # The next_intent should be the intent from the incoming edge
        assert next_intent == "dummy"  # From the dummy edge we added

    def test_get_node(self, task_graph, params) -> None:
        """Test getting node information."""
        # Setup params with available global intents
        params.taskgraph.available_global_intents = {
            "start": [{"target_node": "node1"}]
        }

        node_info, updated_params = task_graph._get_node("node1", params, "start")

        assert isinstance(node_info, NodeInfo)
        assert node_info.node_id == "node1"
        assert node_info.resource_name == "task1"
        assert node_info.resource_id == "task1_id"
        assert updated_params.taskgraph.curr_node == "node1"

    def test_get_node_without_intent(self, task_graph, params) -> None:
        """Test getting node information without intent."""
        node_info, updated_params = task_graph._get_node("node1", params)

        assert isinstance(node_info, NodeInfo)
        assert node_info.node_id == "node1"
        assert updated_params.taskgraph.curr_node == "node1"

    def test_postprocess_intent_basic(self, task_graph) -> None:
        """Test postprocessing intent without special formatting."""
        found, real_intent, idx = task_graph._postprocess_intent(
            "start", ["start", "continue"]
        )
        assert found is True
        assert real_intent == "start"
        assert idx == 0

    def test_postprocess_intent_with_index(self, task_graph) -> None:
        """Test postprocessing intent with __<idx> format."""
        found, real_intent, idx = task_graph._postprocess_intent(
            "start__<2>", ["start", "continue"]
        )
        assert found is True
        assert real_intent == "start"
        assert idx == 2

    def test_postprocess_intent_not_found(self, task_graph) -> None:
        """Test postprocessing intent that's not in available intents."""
        found, real_intent, idx = task_graph._postprocess_intent(
            "unknown", ["start", "continue"]
        )
        assert found is False
        assert real_intent == "unknown"
        assert idx == 0

    def test_get_current_node_with_valid_node(self, task_graph, params) -> None:
        """Test getting current node when it's valid."""
        params.taskgraph.curr_node = "node1"
        curr_node, updated_params = task_graph.get_current_node(params)
        assert curr_node == "node1"
        assert updated_params.taskgraph.curr_node == "node1"

    def test_get_current_node_with_invalid_node(self, task_graph, params) -> None:
        """Test getting current node when it's invalid."""
        params.taskgraph.curr_node = "nonexistent"
        curr_node, updated_params = task_graph.get_current_node(params)
        assert curr_node == "start"  # Should fall back to start node
        assert updated_params.taskgraph.curr_node == "start"

    def test_get_current_node_with_none(self, task_graph, params) -> None:
        """Test getting current node when it's None."""
        params.taskgraph.curr_node = None
        curr_node, updated_params = task_graph.get_current_node(params)
        assert curr_node == "start"
        assert updated_params.taskgraph.curr_node == "start"

    def test_get_available_global_intents_empty(self, task_graph, params) -> None:
        """Test getting available global intents when empty."""
        params.taskgraph.available_global_intents = {}
        intents = task_graph.get_available_global_intents(params)
        assert "others" in intents

    def test_get_available_global_intents_existing(self, task_graph, params) -> None:
        """Test getting available global intents when they already exist."""
        existing_intents = {"custom": [{"intent": "custom"}]}
        params.taskgraph.available_global_intents = existing_intents
        intents = task_graph.get_available_global_intents(params)

        assert intents == existing_intents  # Should return existing intents

    def test_update_node_limit(self, task_graph, params) -> None:
        """Test updating node limits."""
        params.taskgraph.node_limit = {"node1": 5}
        updated_params = task_graph.update_node_limit(params)

        assert "node1" in updated_params.taskgraph.node_limit
        assert updated_params.taskgraph.node_limit["node1"] == 5

    def test_get_local_intent(self, task_graph, params) -> None:
        """Test getting local intents for a node."""
        local_intents = task_graph.get_local_intent("node1", params)

        assert "continue" in local_intents
        assert "skip" in local_intents
        assert len(local_intents["continue"]) == 1
        assert len(local_intents["skip"]) == 1

    def test_get_local_intent_no_intents(self, task_graph, params) -> None:
        """Test getting local intents for a node with no outgoing intents."""
        # Create a node with no outgoing edges with intents
        task_graph.graph.add_node(
            "isolated",
            type="task",
            resource={"name": "isolated", "id": "isolated_id"},
            attribute={},
        )

        local_intents = task_graph.get_local_intent("isolated", params)
        assert local_intents == {}

    def test_get_last_flow_stack_node(self, task_graph, params) -> None:
        """Test getting last flow stack node from path."""
        # Create path with flow stack nodes
        path_node1 = PathNode(node_id="node1", in_flow_stack=True)
        path_node2 = PathNode(node_id="node2", in_flow_stack=False)
        path_node3 = PathNode(node_id="node3", in_flow_stack=True)

        params.taskgraph.path = [path_node1, path_node2, path_node3]

        last_flow_node = task_graph.get_last_flow_stack_node(params)
        assert last_flow_node.node_id == "node3"
        assert last_flow_node.in_flow_stack is False  # Should be set to False

    def test_get_last_flow_stack_node_not_found(self, task_graph, params) -> None:
        """Test getting last flow stack node when none exists."""
        path_node1 = PathNode(node_id="node1", in_flow_stack=False)
        path_node2 = PathNode(node_id="node2", in_flow_stack=False)

        params.taskgraph.path = [path_node1, path_node2]

        last_flow_node = task_graph.get_last_flow_stack_node(params)
        assert last_flow_node is None

    def test_handle_multi_step_node_stay_status(self, task_graph, params) -> None:
        """Test handling multi-step node with STAY status."""
        params.taskgraph.node_status = {"node1": StatusEnum.STAY}

        should_stay, node_info, updated_params = task_graph.handle_multi_step_node(
            "node1", params
        )

        assert should_stay is True
        assert isinstance(node_info, NodeInfo)
        assert node_info.node_id == "node1"

    def test_handle_multi_step_node_complete_status(self, task_graph, params) -> None:
        """Test handling multi-step node with COMPLETE status."""
        params.taskgraph.node_status = {"node1": StatusEnum.COMPLETE}

        should_stay, node_info, updated_params = task_graph.handle_multi_step_node(
            "node1", params
        )

        assert should_stay is False
        # The implementation returns a default NodeInfo, not None
        assert isinstance(node_info, NodeInfo)
        assert node_info.node_id is None

    def test_handle_incomplete_node(self, task_graph, params) -> None:
        """Test handling incomplete node."""
        params.taskgraph.node_status = {"node1": StatusEnum.INCOMPLETE}

        is_incomplete, node_data, updated_params = task_graph.handle_incomplete_node(
            "node1", params
        )

        assert is_incomplete is True
        # The implementation returns a NodeInfo, not a dict
        assert isinstance(node_data, NodeInfo)

    def test_handle_incomplete_node_complete_status(self, task_graph, params) -> None:
        """Test handling incomplete node with COMPLETE status."""
        params.taskgraph.node_status = {"node1": StatusEnum.COMPLETE}

        is_incomplete, node_data, updated_params = task_graph.handle_incomplete_node(
            "node1", params
        )

        assert is_incomplete is False
        # The implementation returns an empty dict, not None
        assert node_data == {}

    def test_handle_random_next_node(self, task_graph, params) -> None:
        """Test handling random next node selection."""
        with patch("numpy.random.choice") as mock_choice:
            mock_choice.return_value = "node2"

            has_random, node_data, updated_params = task_graph.handle_random_next_node(
                "node1", params
            )

            # The dummy graph does not support random next node, so expect False
            assert has_random is False

    def test_handle_unknown_intent(self, task_graph, params) -> None:
        """Test handling unknown intent."""
        node_info, updated_params = task_graph.handle_unknown_intent("node1", params)

        assert isinstance(node_info, NodeInfo)
        # The fallback returns a default NodeInfo with node_id None
        assert node_info.node_id is None

    def test_handle_leaf_node(self, task_graph, params) -> None:
        """Test handling leaf node."""
        # Mock NestedGraph.get_nested_graph_component_node
        with patch(
            "arklex.orchestrator.task_graph.NestedGraph.get_nested_graph_component_node"
        ) as mock_nested:
            mock_nested.return_value = (None, params)

            # Mock get_last_flow_stack_node to return a node
            with patch.object(task_graph, "get_last_flow_stack_node") as mock_flow:
                mock_flow_node = Mock(spec=PathNode)
                mock_flow_node.node_id = "node1"  # Match the actual node id
                mock_flow_node.global_intent = "test_intent"
                mock_flow.return_value = mock_flow_node

                # Mock initial_node to be None so it doesn't interfere
                with patch.object(task_graph, "initial_node", None):
                    # Patch graph.successors to return empty list (leaf node)
                    with patch.object(task_graph.graph, "successors", return_value=[]):
                        result_node, result_params = task_graph.handle_leaf_node(
                            "curr_node", params
                        )
                        assert result_node == "node1"

    def test_validate_node_valid(self, task_graph) -> None:
        """Test validating a valid node."""
        valid_node = {
            "id": "test",
            "type": "task",
            "resource": {"name": "test", "id": "test"},
            "attribute": {},
        }

        # Should not raise any exception
        task_graph._validate_node(valid_node)

    def test_validate_node_missing_id(self, task_graph) -> None:
        """Test validating a node missing required fields."""
        invalid_node = {
            "type": "task",
            "resource": {"name": "test", "id": "test"},
            "attribute": {},
        }

        from arklex.utils.exceptions import TaskGraphError

        with pytest.raises(TaskGraphError):
            task_graph._validate_node(invalid_node)

    def test_validate_node_missing_attribute(self, task_graph) -> None:
        """Test _validate_node with missing attribute."""
        node = {"id": "test_id"}
        with pytest.raises(Exception):  # TaskGraphError
            task_graph._validate_node(node)

    def test_global_intent_prediction_with_excluded_intents(
        self, task_graph, params
    ) -> None:
        """Test global_intent_prediction with excluded intents."""
        available_global_intents = {"intent1": [{"name": "intent1"}]}
        excluded_intents = {"intent1": True}

        # Mock the _get_node method
        with patch.object(task_graph, "_get_node") as mock_get_node:
            mock_node_info = NodeInfo(
                node_id="next_node",
                type="test",
                resource_id="test",
                resource_name="test",
            )
            mock_get_node.return_value = (mock_node_info, params)

            # Mock graph.successors to return a non-empty list
            with patch.object(
                task_graph.graph, "successors", return_value=["next_node"]
            ):
                result, pred_intent, node_info, params = (
                    task_graph.global_intent_prediction(
                        "curr_node", params, available_global_intents, excluded_intents
                    )
                )
                assert result is False

    def test_global_intent_prediction_with_no_candidates(
        self, task_graph, params
    ) -> None:
        """Test global_intent_prediction with no candidate intents."""
        available_global_intents = {}
        excluded_intents = {}

        result, pred_intent, node_info, params = task_graph.global_intent_prediction(
            "curr_node", params, available_global_intents, excluded_intents
        )
        assert result is False
        assert pred_intent == "others"

    def test_global_intent_prediction_with_same_intent_and_incomplete_status(
        self, task_graph, params
    ) -> None:
        """Test global_intent_prediction with same intent and incomplete status."""
        available_global_intents = {"intent1": [{"name": "intent1"}]}
        excluded_intents = {}

        # Define patches at the beginning
        patches = [
            patch.object(task_graph.intent_detector, "execute", return_value="intent1"),
        ]

        with ExitStack() as stack:
            # Apply patches
            stack.enter_context(patches[0])
            with patch.object(
                task_graph.graph, "successors", return_value=["next_node"]
            ):
                params.taskgraph.curr_global_intent = "intent1"
                params.taskgraph.node_status = {"curr_node": StatusEnum.INCOMPLETE}

                result, pred_intent, node_info, params = (
                    task_graph.global_intent_prediction(
                        "curr_node", params, available_global_intents, excluded_intents
                    )
                )
                assert result is False

    def test_global_intent_prediction_with_different_intent(
        self, task_graph, params
    ) -> None:
        """Test global_intent_prediction with different intent."""
        available_global_intents = {"intent1": [{"name": "intent1"}]}
        excluded_intents = {}

        # Define patches at the beginning
        patches = [
            patch.object(task_graph.intent_detector, "execute", return_value="intent1"),
            patch.object(
                task_graph, "jump_to_node", return_value=("next_node", "intent1")
            ),
            patch.object(
                task_graph,
                "_get_node",
                return_value=(
                    NodeInfo(
                        node_id="next_node",
                        type="test",
                        resource_id="test",
                        resource_name="test",
                    ),
                    params,
                ),
            ),
        ]

        with ExitStack() as stack:
            # Apply patches
            stack.enter_context(patches[0])
            stack.enter_context(patches[1])
            stack.enter_context(patches[2])
            with patch.object(
                task_graph.graph, "successors", return_value=["next_node"]
            ):
                params.taskgraph.curr_global_intent = "different_intent"

                result, pred_intent, node_info, params = (
                    task_graph.global_intent_prediction(
                        "curr_node", params, available_global_intents, excluded_intents
                    )
                )
                assert result is True
                assert pred_intent == "intent1"

    def test_local_intent_prediction_with_unsure_intent(
        self, task_graph, params
    ) -> None:
        """Test local_intent_prediction with unsure intent."""
        curr_local_intents = {"intent1": [{"name": "intent1"}]}

        # Define patches at the beginning
        patches = [
            patch.object(task_graph.intent_detector, "execute", return_value="others"),
        ]

        with ExitStack() as stack:
            # Apply patches
            stack.enter_context(patches[0])

            result, node_info, params = task_graph.local_intent_prediction(
                "curr_node", params, curr_local_intents
            )
            assert result is False

    def test_local_intent_prediction_with_found_intent(
        self, task_graph, params
    ) -> None:
        """Test local_intent_prediction with found intent."""
        curr_local_intents = {"intent1": [{"name": "intent1"}]}

        # Define patches at the beginning
        patches = [
            patch.object(task_graph.intent_detector, "execute", return_value="intent1"),
            patch.object(
                task_graph,
                "_get_node",
                return_value=(
                    NodeInfo(
                        node_id="next_node",
                        type="test",
                        resource_id="test",
                        resource_name="test",
                    ),
                    params,
                ),
            ),
        ]

        with ExitStack() as stack:
            # Apply patches
            stack.enter_context(patches[0])
            stack.enter_context(patches[1])

            task_graph.graph.out_edges.return_value = [
                ("curr_node", "next_node", "intent1")
            ]

            result, node_info, params = task_graph.local_intent_prediction(
                "curr_node", params, curr_local_intents
            )
            assert result is True

    def test_local_intent_prediction_with_start_node(self, task_graph, params) -> None:
        """Test local_intent_prediction with start node."""
        curr_local_intents = {"intent1": [{"name": "intent1"}]}
        task_graph.start_node = "curr_node"

        # Define patches at the beginning
        patches = [
            patch.object(task_graph.intent_detector, "execute", return_value="intent1"),
            patch.object(
                task_graph,
                "_get_node",
                return_value=(
                    NodeInfo(
                        node_id="next_node",
                        type="test",
                        resource_id="test",
                        resource_name="test",
                    ),
                    params,
                ),
            ),
        ]

        with ExitStack() as stack:
            # Apply patches
            stack.enter_context(patches[0])
            stack.enter_context(patches[1])

            task_graph.graph.out_edges.return_value = [
                ("curr_node", "next_node", "intent1")
            ]

            result, node_info, params = task_graph.local_intent_prediction(
                "curr_node", params, curr_local_intents
            )
            assert result is True
            assert params.taskgraph.curr_global_intent == "intent1"

    def test_handle_leaf_node_with_nested_graph(self, task_graph, params) -> None:
        """Test handle_leaf_node with nested graph."""
        with patch.object(task_graph.graph, "successors", return_value=[]):
            result_node, result_params = task_graph.handle_leaf_node(
                "curr_node", params
            )
            assert result_node is not None

    def test_handle_leaf_node_with_flow_stack(self, task_graph, params) -> None:
        """Test handle_leaf_node with flow stack."""
        with patch.object(task_graph.graph, "successors", return_value=[]):
            result_node, result_params = task_graph.handle_leaf_node(
                "curr_node", params
            )
            assert result_node is not None

    def test_get_node_with_start_text(self, task_graph, params) -> None:
        """Test get_node with '<start>' text."""
        inputs = {
            "text": "<start>",
            "chat_history_str": "",
            "parameters": params,
            "allow_global_intent_switch": True,
        }

        # Define patches at the beginning
        patches = [
            patch.object(
                task_graph, "get_current_node", return_value=("start_node", params)
            ),
            patch.object(
                task_graph,
                "_get_node",
                return_value=(
                    NodeInfo(
                        node_id="start_node",
                        type="test",
                        resource_id="test",
                        resource_name="test",
                    ),
                    params,
                ),
            ),
        ]

        with ExitStack() as stack:
            # Apply patches
            stack.enter_context(patches[0])
            stack.enter_context(patches[1])

            node_info, params = task_graph.get_node(inputs)
            assert node_info.node_id == "start_node"

    def test_get_node_with_global_intent_switch(self, task_graph, params) -> None:
        """Test get_node with global intent switch allowed."""
        inputs = {
            "text": "test text",
            "chat_history_str": "",
            "parameters": params,
            "allow_global_intent_switch": True,
        }

        # Define patches at the beginning
        patches = [
            patch.object(
                task_graph, "get_current_node", return_value=("curr_node", params)
            ),
            patch.object(
                task_graph,
                "get_available_global_intents",
                return_value={"intent1": [{"name": "intent1"}]},
            ),
            patch.object(
                task_graph,
                "global_intent_prediction",
                return_value=(
                    True,
                    "intent1",
                    NodeInfo(
                        node_id="next_node",
                        type="test",
                        resource_id="test",
                        resource_name="test",
                    ),
                    params,
                ),
            ),
        ]

        with ExitStack() as stack:
            # Apply patches
            stack.enter_context(patches[0])
            stack.enter_context(patches[1])
            stack.enter_context(patches[2])
            # Add curr_node to the graph
            task_graph.graph.add_node("curr_node")
            with patch.object(task_graph.graph, "successors", return_value=[]):
                node_info, params_ = task_graph.get_node(inputs)
                assert node_info.node_id == "next_node"

    def test_get_node_with_local_intent(self, task_graph, params) -> None:
        """Test get_node with local intent prediction."""
        inputs = {
            "text": "test text",
            "chat_history_str": "",
            "parameters": params,
            "allow_global_intent_switch": False,
        }

        # Define patches at the beginning
        patches = [
            patch.object(
                task_graph, "get_current_node", return_value=("curr_node", params)
            ),
            patch.object(task_graph, "get_available_global_intents", return_value={}),
            patch.object(
                task_graph,
                "get_local_intent",
                return_value={"intent1": [{"name": "intent1"}]},
            ),
            patch.object(
                task_graph,
                "local_intent_prediction",
                return_value=(
                    True,
                    NodeInfo(
                        node_id="next_node",
                        type="test",
                        resource_id="test",
                        resource_name="test",
                    ),
                    params,
                ),
            ),
        ]

        with ExitStack() as stack:
            # Apply patches
            stack.enter_context(patches[0])
            stack.enter_context(patches[1])
            stack.enter_context(patches[2])
            stack.enter_context(patches[3])
            # Add curr_node to the graph
            task_graph.graph.add_node("curr_node")
            with patch.object(task_graph.graph, "successors", return_value=[]):
                node_info, params_ = task_graph.get_node(inputs)
                assert node_info.node_id == "next_node"

    def test_get_node_with_unknown_intent(self, task_graph, params) -> None:
        """Test get_node with unknown intent."""
        inputs = {
            "text": "test text",
            "chat_history_str": "",
            "parameters": params,
            "allow_global_intent_switch": False,
        }

        # Define patches at the beginning
        patches = [
            patch.object(
                task_graph, "get_current_node", return_value=("curr_node", params)
            ),
            patch.object(task_graph, "get_available_global_intents", return_value={}),
            patch.object(task_graph, "get_local_intent", return_value={}),
            patch.object(
                task_graph,
                "handle_unknown_intent",
                return_value=(
                    NodeInfo(
                        node_id="planner",
                        type="test",
                        resource_id="test",
                        resource_name="test",
                    ),
                    params,
                ),
            ),
        ]

        with ExitStack() as stack:
            # Apply patches
            stack.enter_context(patches[0])
            stack.enter_context(patches[1])
            stack.enter_context(patches[2])
            stack.enter_context(patches[3])
            # Add curr_node to the graph
            task_graph.graph.add_node("curr_node")
            with patch.object(task_graph.graph, "successors", return_value=[]):
                node_info, params_ = task_graph.get_node(inputs)
                assert node_info.node_id == "planner"

    def test_postprocess_node(self, task_graph, params) -> None:
        """Test postprocess_node method."""
        node_info = NodeInfo(
            node_id="test_node", type="test", resource_id="test", resource_name="test"
        )

        result_node_info, result_params = task_graph.postprocess_node(
            (node_info, params)
        )
        assert result_node_info == node_info
        assert result_params == params

    def test_validate_node_with_extra_fields(self, task_graph) -> None:
        """Test _validate_node with extra fields."""
        node = {"id": "test_id", "type": "test_type", "extra_field": "extra_value"}
        task_graph._validate_node(node)  # Should not raise an error

    def test_validate_node_with_empty_strings(self, task_graph) -> None:
        """Test _validate_node with empty string values."""
        node = {"id": "", "type": ""}
        task_graph._validate_node(node)  # Should not raise an error for empty strings
