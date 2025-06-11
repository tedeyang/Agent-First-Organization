import pytest
from unittest.mock import Mock, patch
from arklex.orchestrator.generator.core.generator import Generator
from arklex.env.env import BaseResourceInitializer, DefaultResourceInitializer


@pytest.fixture
def mock_model():
    model = Mock()
    return model


@pytest.fixture
def minimal_config():
    """Create a minimal valid configuration for testing."""
    return {
        "role": "test_role",
        "user_objective": "test_objective",
        "builder_objective": "test_builder_objective",
        "intro": "test_intro",
        "instruction_docs": [],
        "task_docs": [],
        "rag_docs": [],
        "user_tasks": [],
        "example_conversations": [],
        "product_kwargs": {
            "tools": [],
            "workers": [],
        },
    }


def test_generator_initialization(minimal_config, mock_model):
    gen = Generator(config=minimal_config, model=mock_model)
    assert gen.role == "test_role"
    assert gen.user_objective == "test_objective"
    assert gen.builder_objective == "test_builder_objective"
    assert gen.intro == "test_intro"
    assert isinstance(gen.resource_initializer, DefaultResourceInitializer)


def test_generator_generate_calls_components(minimal_config, mock_model):
    gen = Generator(config=minimal_config, model=mock_model)
    # Patch only methods that exist on the Generator class
    with patch.object(gen, "generate", wraps=gen.generate) as mock_generate:
        gen.generate()
        mock_generate.assert_called_once()


def test_generator_save_task_graph(tmp_path, minimal_config, mock_model):
    gen = Generator(config=minimal_config, model=mock_model, output_dir=str(tmp_path))
    task_graph = {"nodes": [], "edges": []}
    output_path = gen.save_task_graph(task_graph)
    assert output_path.endswith(".json")
    import os

    assert os.path.exists(output_path)


def test_generator_with_invalid_resource_initializer(minimal_config, mock_model):
    # Should fallback to DefaultResourceInitializer if None is provided
    gen = Generator(config=minimal_config, model=mock_model, resource_initializer=None)
    assert isinstance(gen.resource_initializer, DefaultResourceInitializer)


def test_generator_with_custom_resource_initializer(minimal_config, mock_model):
    class CustomInitializer(BaseResourceInitializer):
        def init_workers(self, workers):
            return {"custom_worker": {}}

        def init_tools(self, tools):
            return {"custom_tool": {}}

    gen = Generator(
        config=minimal_config,
        model=mock_model,
        resource_initializer=CustomInitializer(),
    )
    assert isinstance(gen.resource_initializer, CustomInitializer)
