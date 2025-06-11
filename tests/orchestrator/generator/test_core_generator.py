import pytest
from unittest.mock import Mock, patch
from arklex.orchestrator.generator.core.generator import Generator


@pytest.fixture
def mock_model():
    model = Mock()
    return model


@pytest.fixture
def minimal_config():
    return {
        "role": "test_role",
        "user_objective": "test_objective",
        "builder_objective": "test_builder_objective",
        "intro": "test_intro",
        "instruction_docs": [],
        "task_docs": [],
        "user_tasks": [],
        "product_kwargs": {"workers": [], "tools": []},
    }


def test_generator_initialization(minimal_config, mock_model):
    gen = Generator(config=minimal_config, model=mock_model)
    assert gen.role == "test_role"
    assert gen.u_objective == "test_objective"
    assert gen.b_objective == "test_builder_objective"
    assert gen.intro == "test_intro"
    assert gen.model == mock_model


def test_generator_generate_calls_components(minimal_config, mock_model):
    gen = Generator(config=minimal_config, model=mock_model)
    # Patch component initializers to return mocks and UI_AVAILABLE to False
    with (
        patch.object(gen, "_initialize_document_loader") as doc_loader_patch,
        patch.object(gen, "_initialize_task_generator") as task_gen_patch,
        patch.object(gen, "_initialize_best_practice_manager") as bpm_patch,
        patch.object(gen, "_initialize_reusable_task_manager") as rtm_patch,
        patch.object(gen, "_initialize_task_graph_formatter") as tgf_patch,
        patch("arklex.orchestrator.generator.core.generator.UI_AVAILABLE", False),
    ):
        doc_loader = Mock()
        doc_loader.load_task_document.return_value = "docs"
        doc_loader.load_instructions.return_value = "instructions"
        doc_loader_patch.return_value = doc_loader
        task_gen = Mock()
        task_gen.add_provided_tasks.return_value = []
        task_gen.generate_tasks.return_value = []
        task_gen_patch.return_value = task_gen
        bpm = Mock()
        bpm.generate_best_practices.return_value = []
        bpm_patch.return_value = bpm
        rtm = Mock()
        rtm.generate_reusable_tasks.return_value = {}
        rtm_patch.return_value = rtm
        tgf = Mock()
        tgf.format_task_graph.return_value = {"nodes": [], "edges": []}
        tgf_patch.return_value = tgf
        result = gen.generate()
        assert isinstance(result, dict)
        assert "nodes" in result and "edges" in result


def test_generator_save_task_graph(tmp_path, minimal_config, mock_model):
    gen = Generator(config=minimal_config, model=mock_model, output_dir=str(tmp_path))
    task_graph = {"nodes": [1], "edges": [2]}
    path = gen.save_task_graph(task_graph)
    assert path.endswith(".json")
    import os

    assert os.path.exists(path)


def test_generator_with_invalid_resource_initializer(minimal_config, mock_model):
    # Should fallback to DefaultResourceInitializer if None is provided
    gen = Generator(config=minimal_config, model=mock_model, resource_inizializer=None)
    assert gen.workers is not None
    assert gen.tools is not None
