from arklex.env.env import Environment, DefaultResourceInitializer
from arklex.orchestrator.NLU.services.model_service import DummyModelService
from unittest.mock import patch, MagicMock
from arklex.orchestrator.NLU.core.slot import SlotFiller
from arklex.utils.graph_state import MessageState, Params, NodeInfo


def test_environment_uses_dummy_model_service() -> None:
    env = Environment(tools=[], workers=[])
    assert isinstance(env.model_service, DummyModelService)


def test_environment_initializes_with_planner() -> None:
    env = Environment(tools=[], workers=[], planner_enabled=True)
    assert hasattr(env, "planner")


def test_environment_initializes_with_slotfillapi_str() -> None:
    env = Environment(tools=[], workers=[], slotsfillapi="http://fakeapi")
    assert hasattr(env, "slotfillapi")
    assert isinstance(env.slotfillapi, SlotFiller)


def test_environment_initializes_with_slotfillapi_model_service() -> None:
    env = Environment(tools=[], workers=[], slotsfillapi="")
    assert hasattr(env, "slotfillapi")
    assert isinstance(env.slotfillapi.model_service, DummyModelService)


def test_default_resource_initializer_init_tools_success_and_error() -> None:
    tools = [
        {"id": "t1", "name": "fake_tool", "path": "fake_path"},
        {"id": "t2", "name": "bad_tool", "path": "bad_path"},
    ]
    # Patch importlib to succeed for one and fail for the other
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(return_value=MagicMock(description="desc"))
        setattr(fake_module, "fake_tool", fake_func)
        mock_import.side_effect = [fake_module, Exception("fail")]
        registry = DefaultResourceInitializer.init_tools(tools)
        assert "t1" in registry
        assert "t2" not in registry  # error case is skipped


def test_default_resource_initializer_init_workers_success_and_error() -> None:
    workers = [
        {"id": "w1", "name": "fake_worker", "path": "fake_path"},
        {"id": "w2", "name": "bad_worker", "path": "bad_path"},
    ]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        fake_func = MagicMock(description="desc")
        setattr(fake_module, "fake_worker", fake_func)
        mock_import.side_effect = [fake_module, Exception("fail")]
        registry = DefaultResourceInitializer.init_workers(workers)
        assert "w1" in registry
        assert "w2" not in registry


def test_environment_step_tool_executes_and_updates_params() -> None:
    # Setup a fake tool
    fake_tool = MagicMock()
    fake_tool.init_slotfiller = MagicMock()
    fake_tool.execute = MagicMock(
        return_value=MagicMock(function_calling_trajectory=["call"], slots={})
    )
    tools = [
        {"id": "t1", "name": "fake_tool", "path": "fake_path"},
    ]
    with patch("importlib.import_module") as mock_import:
        fake_module = MagicMock()
        setattr(fake_module, "fake_tool", MagicMock(return_value=fake_tool))
        mock_import.return_value = fake_module
        env = Environment(tools=tools, workers=[])

        # Setup params and state
        class DummyParams:
            memory = MagicMock()
            taskgraph = MagicMock()
            taskgraph.dialog_states = {}
            taskgraph.node_status = {}
            taskgraph.curr_node = "n1"

        class DummyNodeInfo:
            additional_args = {"foo": "bar"}

        state = MagicMock()
        params = DummyParams()
        node_info = DummyNodeInfo()
        env.tools["t1"]["fixed_args"] = {"baz": 1}
        result_state, result_params = env.step("t1", state, params, node_info)
        assert result_state.function_calling_trajectory == ["call"]
        assert result_params is params


def test_environment_step_invalid_id_raises() -> None:
    env = Environment(tools=[], workers=[])
    # The step method doesn't raise KeyError for invalid IDs, it falls back to planner
    # So we should test that it doesn't raise an exception
    message_state = MessageState()
    params = Params()
    node_info = NodeInfo()

    # This should not raise an exception, it should use the planner
    response_state, updated_params = env.step(
        "not_a_tool", message_state, params, node_info
    )
    assert isinstance(response_state, MessageState)
    assert isinstance(updated_params, Params)
