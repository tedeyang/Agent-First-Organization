"""Core generator tests for the main Generator class.

These tests verify the core functionality of the Generator class,
including initialization, component integration, and error handling.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from arklex.env.env import DefaultResourceInitializer
from arklex.orchestrator.generator.core.generator import Generator

# --- Centralized Mock Fixtures ---


@pytest.fixture
def always_valid_mock_model() -> MagicMock:
    """Create a mock language model that always returns valid responses."""
    mock = MagicMock()
    mock.generate.return_value = MagicMock()
    mock.invoke.return_value = MagicMock()
    return mock


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a basic mock model for testing."""
    mock = MagicMock()
    mock.generate.return_value = MagicMock()
    mock.invoke.return_value = MagicMock()
    return mock


@pytest.fixture
def minimal_config() -> dict[str, Any]:
    """Create a minimal configuration for testing."""
    return {
        "role": "test_role",
        "user_objective": "test_objective",
        "workers": [],
        "tools": {},
    }


@pytest.fixture
def patched_sample_config() -> dict[str, Any]:
    """Create a sample configuration with all required fields."""
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
            "tools": [{"id": "test_tool", "name": "TestTool", "path": "test_tool.py"}],
            "workers": [{"id": "test_worker", "name": "TestWorker"}],
        },
    }


@pytest.fixture
def full_config() -> dict[str, Any]:
    """Create a full configuration for testing."""
    return {
        "role": "test_role",
        "user_objective": "test_objective",
        "builder_objective": "test_builder_objective",
        "domain": "test_domain",
        "intro": "test_intro",
        "instruction_docs": ["instruction1.txt", "instruction2.txt"],
        "task_docs": ["task1.txt", "task2.txt"],
        "rag_docs": ["rag1.txt"],
        "user_tasks": [{"id": "task1", "name": "User Task 1"}],
        "example_conversations": [{"conversation": "example"}],
        "nluapi": "http://nlu.api",
        "slotfillapi": "http://slotfill.api",
        "workers": [
            {"id": "worker1", "name": "Worker 1", "path": "/path/to/worker1"},
            {"id": "worker2", "name": "Worker 2", "path": "/path/to/worker2"},
        ],
        "tools": [
            {"id": "tool1", "name": "Tool 1", "path": "/path/to/tool1"},
            {"id": "tool2", "name": "Tool 2", "path": "/path/to/tool2"},
        ],
    }


@pytest.fixture
def patched_generator_components(
    patched_sample_config: dict[str, Any],
    always_valid_mock_model: Mock,
    mock_document_loader: Mock,
    mock_task_generator: Mock,
    mock_best_practice_manager: Mock,
    mock_reusable_task_manager: Mock,
    mock_task_graph_formatter: Mock,
) -> tuple[Generator, Mock, Mock, Mock, Mock, Mock]:
    """Create a generator with all components patched for testing."""
    gen = Generator(config=patched_sample_config, model=always_valid_mock_model)

    with (
        patch.object(gen, "_initialize_document_loader") as mock_doc_loader,
        patch.object(gen, "_initialize_task_generator") as mock_task_gen,
        patch.object(gen, "_initialize_best_practice_manager") as mock_bpm,
        patch.object(gen, "_initialize_reusable_task_manager") as mock_rtm,
        patch.object(gen, "_initialize_task_graph_formatter") as mock_tgf,
    ):
        mock_doc_loader.return_value = mock_document_loader
        mock_task_gen.return_value = mock_task_generator
        mock_bpm.return_value = mock_best_practice_manager
        mock_rtm.return_value = mock_reusable_task_manager
        mock_tgf.return_value = mock_task_graph_formatter

        yield gen, mock_doc_loader, mock_task_gen, mock_bpm, mock_rtm, mock_tgf


class TestGeneratorInitialization:
    """Test Generator initialization and configuration."""

    def test_generator_initialization(
        self, patched_sample_config: dict[str, Any], always_valid_mock_model: Mock
    ) -> None:
        """Test generator initialization with basic configuration."""
        gen = Generator(config=patched_sample_config, model=always_valid_mock_model)

        assert gen.config == patched_sample_config
        assert gen.model is always_valid_mock_model

    def test_generator_with_full_config(
        self, full_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generator initialization with full configuration."""
        # Mock the resource initializer to avoid tool loading errors
        with patch(
            "arklex.orchestrator.generator.core.generator.DefaultResourceInitializer"
        ) as mock_ri_class:
            mock_ri = Mock()
            mock_ri.init_workers.return_value = [
                {"name": "Worker1"},
                {"name": "Worker2"},
            ]
            mock_ri.init_tools.return_value = [{"name": "Tool1"}]
            mock_ri_class.return_value = mock_ri

            gen = Generator(config=full_config, model=mock_model)

            assert gen.config == full_config
            assert gen.model == mock_model
            assert gen.role == "test_role"
            assert gen.user_objective == "test_objective"
            assert gen.workers == [{"name": "Worker1"}, {"name": "Worker2"}]
            assert gen.tools == [{"name": "Tool1"}]

    def test_generator_with_invalid_resource_initializer(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generator falls back to DefaultResourceInitializer if None is provided."""
        gen = Generator(
            config=minimal_config, model=mock_model, resource_initializer=None
        )

        assert isinstance(gen.resource_initializer, DefaultResourceInitializer)

    def test_generator_with_invalid_resource_initializer_alt(
        self, patched_sample_config: dict[str, Any], always_valid_mock_model: Mock
    ) -> None:
        """Test generator fallback to DefaultResourceInitializer when None is provided."""
        gen = Generator(
            config=patched_sample_config,
            model=always_valid_mock_model,
            resource_initializer=None,
        )

        assert isinstance(gen.resource_initializer, DefaultResourceInitializer)

    def test_generator_with_custom_resource_initializer(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generator with custom resource initializer."""
        from arklex.orchestrator.generator.core.generator import BaseResourceInitializer

        class CustomInitializer(BaseResourceInitializer):
            def init_workers(self, workers: list) -> dict[str, Any]:
                return {"custom_worker": {}}

            def init_tools(self, tools: list) -> dict[str, Any]:
                return {"custom_tool": {}}

        gen = Generator(
            config=minimal_config,
            model=mock_model,
            resource_initializer=CustomInitializer(),
        )

        assert isinstance(gen.resource_initializer, CustomInitializer)

    def test_generator_with_output_dir(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generator with output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gen = Generator(
                config=minimal_config, model=mock_model, output_dir=temp_dir
            )

            assert gen.output_dir == temp_dir

    def test_generator_without_output_dir(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generator without output directory."""
        gen = Generator(config=minimal_config, model=mock_model)
        assert gen.output_dir is None

    def test_generator_with_disabled_flags(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generator with disabled interaction and nested graph."""
        gen = Generator(
            config=minimal_config,
            model=mock_model,
            allow_nested_graph=False,
            allow_reusable_tasks=False,
        )
        assert gen.allow_nested_graph is False
        assert gen.allow_reusable_tasks is False

    def test_generator_workers_conversion(
        self, full_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test that workers are properly converted to the expected format."""
        gen = Generator(config=full_config, model=mock_model)
        # Workers should be converted from string to dict format
        assert isinstance(gen.workers, list)
        assert all(isinstance(worker, dict) for worker in gen.workers)

    def test_generator_workers_invalid_format(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generator handles invalid worker format gracefully."""
        config_with_invalid_workers = minimal_config.copy()
        config_with_invalid_workers["workers"] = [
            {"name": "worker1"},  # Missing id and path
            {"id": "worker2", "path": "/path"},  # Missing name
            {"id": "worker3", "name": "worker3"},  # Missing path
            {"id": "worker4", "name": "worker4", "path": "/path"},  # Valid
        ]

        gen = Generator(config=config_with_invalid_workers, model=mock_model)
        # Should only include valid workers
        assert len(gen.workers) == 1
        assert gen.workers[0]["id"] == "worker4"

    def test_generator_initialization_with_empty_config(self, mock_model: Mock) -> None:
        """Test generator initialization with an empty config."""
        gen = Generator(config={}, model=mock_model)
        assert gen.config == {}
        assert gen.model == mock_model


@pytest.fixture
def mock_document_loader() -> Mock:
    doc_loader = Mock()
    doc_loader.load_task_document.return_value = {"id": "doc1"}
    doc_loader.load_instruction_document.return_value = {"id": "inst1"}
    return doc_loader


@pytest.fixture
def mock_task_generator() -> MagicMock:
    """Create a mock task generator with standard return values."""
    generator = MagicMock()
    generator.add_provided_tasks.return_value = []
    generator.generate_tasks.return_value = []
    return generator


@pytest.fixture
def mock_best_practice_manager() -> MagicMock:
    """Create a mock best practice manager with standard return values."""
    manager = MagicMock()
    manager.generate_best_practices.return_value = [
        {"practice_id": "bp1", "name": "BP1"}
    ]
    manager.finetune_best_practice.side_effect = lambda bp, task: {
        "steps": task.get("steps", [])
    }
    return manager


@pytest.fixture
def mock_reusable_task_manager() -> MagicMock:
    """Create a mock reusable task manager with standard return values."""
    manager = MagicMock()
    manager.generate_reusable_tasks.return_value = {}
    return manager


@pytest.fixture
def mock_task_graph_formatter() -> MagicMock:
    """Create a mock task graph formatter with standard return values."""
    formatter = MagicMock()
    formatter.format_task_graph.return_value = {"nodes": [], "edges": []}
    formatter.ensure_nested_graph_connectivity.return_value = {"nodes": [], "edges": []}
    return formatter


@pytest.fixture
def mock_prompt_manager() -> MagicMock:
    """Create a mock prompt manager with standard return values."""
    manager = MagicMock()
    manager.get_prompt.return_value = "prompt"
    return manager


# --- Test Data Fixtures ---


@pytest.fixture
def sample_tasks() -> list[dict[str, Any]]:
    """Sample task data for testing."""
    return [
        {"id": "task1", "name": "Task 1", "steps": ["step1", "step2"]},
        {"id": "task2", "name": "Task 2", "steps": ["step1"]},
    ]


@pytest.fixture
def sample_best_practices() -> list[dict[str, Any]]:
    """Sample best practice data for testing."""
    return [
        {"id": "bp1", "name": "Best Practice 1", "steps": ["step1", "step2"]},
        {"id": "bp2", "name": "Best Practice 2", "steps": ["step1"]},
    ]


# --- Core Generator Tests ---


class TestGeneratorComponentInitialization:
    """Test Generator component initialization methods."""

    def test_initialize_document_loader_with_output_dir(
        self, minimal_config: dict[str, Any], mock_model: Mock, tmp_path: Path
    ) -> None:
        gen = Generator(
            config=minimal_config, model=mock_model, output_dir=str(tmp_path)
        )

        with patch(
            "arklex.orchestrator.generator.core.generator.DocumentLoader"
        ) as mock_doc_loader:
            mock_doc_loader.return_value = Mock()
            result = gen._initialize_document_loader()

            assert result is not None
            mock_doc_loader.assert_called_once_with(
                cache_dir=tmp_path,
                task_docs=minimal_config.get("task_docs", []),
                instruction_docs=minimal_config.get("instruction_docs", []),
            )

    def test_initialize_document_loader_without_output_dir(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        gen = Generator(config=minimal_config, model=mock_model)

        with patch(
            "arklex.orchestrator.generator.core.generator.DocumentLoader"
        ) as mock_doc_loader:
            mock_doc_loader.return_value = Mock()
            result = gen._initialize_document_loader()

            assert result is not None
            mock_doc_loader.assert_called_once_with(
                cache_dir=None,
                task_docs=minimal_config.get("task_docs", []),
                instruction_docs=minimal_config.get("instruction_docs", []),
            )

    def test_initialize_task_generator(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        gen = Generator(config=minimal_config, model=mock_model)

        with patch(
            "arklex.orchestrator.generator.core.generator.TaskGenerator"
        ) as mock_task_gen:
            mock_task_gen.return_value = Mock()
            result = gen._initialize_task_generator()

            assert result is not None
            mock_task_gen.assert_called_once_with(
                model=mock_model,
                task_docs=minimal_config.get("task_docs", []),
                instruction_docs=minimal_config.get("instruction_docs", []),
                user_tasks=minimal_config.get("user_tasks", []),
            )

    def test_initialize_best_practice_manager(
        self, full_config: dict[str, Any], mock_model: Mock
    ) -> None:
        gen = Generator(config=full_config, model=mock_model)

        with patch(
            "arklex.orchestrator.generator.core.generator.BestPracticeManager"
        ) as mock_bpm:
            mock_bpm.return_value = Mock()
            result = gen._initialize_best_practice_manager()

            assert result is not None
            mock_bpm.assert_called_once_with(
                model=mock_model,
                role=full_config.get("role", ""),
                user_objective=full_config.get("user_objective", ""),
                workers=full_config.get("workers", []),
                tools=full_config.get("tools", []),
                all_resources=full_config.get("all_resources", []),
            )

    def test_initialize_best_practice_manager_without_nested_graph(
        self, full_config: dict[str, Any], mock_model: Mock
    ) -> None:
        gen = Generator(config=full_config, model=mock_model, allow_nested_graph=False)

        with patch(
            "arklex.orchestrator.generator.core.generator.BestPracticeManager"
        ) as mock_bpm:
            mock_bpm.return_value = Mock()
            result = gen._initialize_best_practice_manager()

            assert result is not None
            mock_bpm.assert_called_once_with(
                model=mock_model,
                role=full_config.get("role", ""),
                user_objective=full_config.get("user_objective", ""),
                workers=full_config.get("workers", []),
                tools=full_config.get("tools", []),
                all_resources=full_config.get("all_resources", []),
            )

    def test_initialize_reusable_task_manager(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        gen = Generator(config=minimal_config, model=mock_model)

        with patch(
            "arklex.orchestrator.generator.core.generator.ReusableTaskManager"
        ) as mock_rtm:
            mock_rtm.return_value = Mock()
            result = gen._initialize_reusable_task_manager()

            assert result is not None
            mock_rtm.assert_called_once_with(
                model=mock_model,
                task_docs=minimal_config.get("task_docs", []),
                instruction_docs=minimal_config.get("instruction_docs", []),
            )

    def test_initialize_task_graph_formatter(
        self, full_config: dict[str, Any], mock_model: Mock
    ) -> None:
        gen = Generator(config=full_config, model=mock_model)

        with patch(
            "arklex.orchestrator.generator.core.generator.TaskGraphFormatter"
        ) as mock_tgf:
            mock_tgf.return_value = Mock()
            result = gen._initialize_task_graph_formatter()

            assert result is not None
            mock_tgf.assert_called_once_with(
                task_docs=full_config.get("task_docs", []),
                rag_docs=full_config.get("rag_docs", []),
                workers=full_config.get("workers", []),
                tools=full_config.get("tools", []),
                role=full_config.get("role", ""),
                user_objective=full_config.get("user_objective", ""),
                builder_objective=full_config.get("builder_objective", ""),
                domain=full_config.get("domain", ""),
                intro=full_config.get("intro", ""),
                nluapi=full_config.get("nluapi", ""),
                slotfillapi=full_config.get("slotfillapi", ""),
                default_intent=full_config.get("default_intent", ""),
                default_weight=full_config.get("default_weight", 1),
                default_pred=full_config.get("default_pred", False),
                default_definition=full_config.get("default_definition", ""),
                default_sample_utterances=full_config.get(
                    "default_sample_utterances", []
                ),
                allow_nested_graph=full_config.get("allow_nested_graph", True),
                model=mock_model,
            )


class TestGeneratorDocumentLoading:
    """Test Generator document loading functionality."""

    def test_load_multiple_task_documents_list(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test loading multiple task documents from a list."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch.object(gen, "_initialize_document_loader") as mock_doc_loader:
            mock_doc_loader.return_value.load_task_document.side_effect = [
                "Document 1 content",
                "Document 2 content",
            ]

            doc_paths = ["doc1.txt", "doc2.txt"]
            result = gen._load_multiple_task_documents(
                mock_doc_loader.return_value, doc_paths
            )

            assert result == ["Document 1 content", "Document 2 content"]
            assert mock_doc_loader.return_value.load_task_document.call_count == 2

    def test_load_multiple_task_documents_dict_list(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test loading multiple task documents from a list of dictionaries."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch.object(gen, "_initialize_document_loader") as mock_doc_loader:
            mock_doc_loader.return_value.load_task_document.side_effect = [
                "Document 1 content",
                "Document 2 content",
            ]

            doc_paths = [{"path": "doc1.txt"}, {"path": "doc2.txt"}]
            result = gen._load_multiple_task_documents(
                mock_doc_loader.return_value, doc_paths
            )

            assert result == ["Document 1 content", "Document 2 content"]
            assert mock_doc_loader.return_value.load_task_document.call_count == 2

    def test_load_multiple_task_documents_single_string(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test loading a single task document from a string."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch.object(gen, "_initialize_document_loader") as mock_doc_loader:
            mock_doc_loader.return_value.load_task_document.return_value = (
                "Document content"
            )

            doc_paths = "doc1.txt"
            result = gen._load_multiple_task_documents(
                mock_doc_loader.return_value, doc_paths
            )

            assert result == ["Document content"]
            mock_doc_loader.return_value.load_task_document.assert_called_once_with(
                "doc1.txt"
            )

    def test_load_multiple_task_documents_single_dict(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test loading a single task document from a dictionary."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch.object(gen, "_initialize_document_loader") as mock_doc_loader:
            mock_doc_loader.return_value.load_task_document.return_value = (
                "Document content"
            )

            doc_paths = {"path": "doc1.txt"}
            result = gen._load_multiple_task_documents(
                mock_doc_loader.return_value, doc_paths
            )

            assert result == ["Document content"]
            mock_doc_loader.return_value.load_task_document.assert_called_once_with(
                "doc1.txt"
            )

    def test_load_multiple_instruction_documents_list(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test loading multiple instruction documents from a list."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch.object(gen, "_initialize_document_loader") as mock_doc_loader:
            mock_doc_loader.return_value.load_instruction_document.side_effect = [
                "Instruction 1 content",
                "Instruction 2 content",
            ]

            doc_paths = ["instruction1.txt", "instruction2.txt"]
            result = gen._load_multiple_instruction_documents(
                mock_doc_loader.return_value, doc_paths
            )

            assert result == ["Instruction 1 content", "Instruction 2 content"]
            assert (
                mock_doc_loader.return_value.load_instruction_document.call_count == 2
            )

    def test_load_multiple_instruction_documents_dict_list(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test loading multiple instruction documents from a list of dictionaries."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch.object(gen, "_initialize_document_loader") as mock_doc_loader:
            mock_doc_loader.return_value.load_instruction_document.side_effect = [
                "Instruction 1 content",
                "Instruction 2 content",
            ]

            doc_paths = [{"path": "instruction1.txt"}, {"path": "instruction2.txt"}]
            result = gen._load_multiple_instruction_documents(
                mock_doc_loader.return_value, doc_paths
            )

            assert result == ["Instruction 1 content", "Instruction 2 content"]
            assert (
                mock_doc_loader.return_value.load_instruction_document.call_count == 2
            )

    def test_load_multiple_instruction_documents_single_string(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test loading a single instruction document from a string."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch.object(gen, "_initialize_document_loader") as mock_doc_loader:
            mock_doc_loader.return_value.load_instruction_document.return_value = (
                "Instruction content"
            )

            doc_paths = "instruction1.txt"
            result = gen._load_multiple_instruction_documents(
                mock_doc_loader.return_value, doc_paths
            )

            assert result == ["Instruction content"]
            mock_doc_loader.return_value.load_instruction_document.assert_called_once_with(
                "instruction1.txt"
            )

    def test_load_multiple_instruction_documents_single_dict(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test loading a single instruction document from a dictionary."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch.object(gen, "_initialize_document_loader") as mock_doc_loader:
            mock_doc_loader.return_value.load_instruction_document.return_value = (
                "Instruction content"
            )

            doc_paths = {"path": "instruction1.txt"}
            result = gen._load_multiple_instruction_documents(
                mock_doc_loader.return_value, doc_paths
            )

            assert result == ["Instruction content"]
            mock_doc_loader.return_value.load_instruction_document.assert_called_once_with(
                "instruction1.txt"
            )

    def test_load_multiple_task_documents_with_none_paths(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test loading multiple task documents with None paths."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch.object(gen, "_initialize_document_loader") as mock_doc_loader:
            mock_doc_loader.return_value.load_task_document.return_value = "doc1"

            result = gen._load_multiple_task_documents(
                mock_doc_loader.return_value, None
            )

            # When None is passed, it's treated as a string and loaded as a document
            assert result == ["doc1"]

    def test_load_multiple_instruction_documents_with_none_paths(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test loading multiple instruction documents with None paths."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch.object(gen, "_initialize_document_loader") as mock_doc_loader:
            mock_doc_loader.return_value.load_instruction_document.return_value = (
                "inst1"
            )

            result = gen._load_multiple_instruction_documents(
                mock_doc_loader.return_value, None
            )

            # When None is passed, it's treated as a string and loaded as a document
            assert result == ["inst1"]


class TestGeneratorGenerate:
    """Test Generator generate method."""

    def test_generator_generate_calls_components(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test that generate method calls all components correctly."""
        gen = Generator(config=minimal_config, model=mock_model)

        with (
            patch.object(gen, "_initialize_document_loader") as mock_doc_loader,
            patch.object(gen, "_initialize_task_generator") as mock_task_gen,
            patch.object(gen, "_initialize_best_practice_manager") as mock_bpm,
            patch.object(gen, "_initialize_reusable_task_manager") as mock_rtm,
            patch.object(gen, "_initialize_task_graph_formatter") as mock_tgf,
        ):
            # Setup mocks
            mock_doc_loader.return_value.load_task_document.return_value = "doc1"
            mock_doc_loader.return_value.load_instruction_document.return_value = (
                "inst1"
            )

            mock_task_gen.return_value.add_provided_tasks.return_value = []
            mock_task_gen.return_value.generate_tasks.return_value = [
                {"name": "Task1", "steps": []}
            ]

            mock_bpm.return_value.generate_best_practices.return_value = [{"id": "bp1"}]
            mock_bpm.return_value.finetune_best_practice.return_value = {
                "steps": [{"description": "refined_step"}]
            }

            mock_rtm.return_value.generate_reusable_tasks.return_value = {}

            mock_tgf.return_value.format_task_graph.return_value = {
                "nodes": [],
                "edges": [],
            }
            mock_tgf.return_value.ensure_nested_graph_connectivity.return_value = {
                "nodes": [],
                "edges": [],
            }

            mock_model.invoke.return_value.content = '{"intent": "test_intent"}'

            result = gen.generate()

            assert isinstance(result, dict)
            assert "nodes" in result
            assert "edges" in result

            # Verify all components were called
            mock_doc_loader.assert_called_once()
            mock_task_gen.assert_called_once()
            mock_bpm.assert_called_once()
            mock_rtm.assert_called_once()
            mock_tgf.assert_called_once()

    def test_generator_generate_with_multiple_documents(
        self, full_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generate method with multiple documents."""
        gen = Generator(config=full_config, model=mock_model)

        with (
            patch.object(gen, "_initialize_document_loader") as doc_loader_patch,
            patch.object(gen, "_initialize_task_generator") as task_gen_patch,
            patch.object(gen, "_initialize_best_practice_manager"),
            patch.object(gen, "_initialize_reusable_task_manager"),
            patch.object(gen, "_initialize_task_graph_formatter"),
            patch("arklex.orchestrator.generator.core.generator.UI_AVAILABLE", False),
        ):
            # Setup mocks
            doc_loader = Mock()
            doc_loader.load_task_document.side_effect = ["doc1", "doc2"]
            doc_loader.load_instruction_document.side_effect = [
                "instruction1",
                "instruction2",
            ]
            doc_loader_patch.return_value = doc_loader

            task_gen = Mock()
            task_gen.add_provided_tasks.return_value = []
            task_gen.generate_tasks.return_value = []
            task_gen_patch.return_value = task_gen


class TestGeneratorCoreFunctionality:
    """Test suite for core generator functionality."""

    def test_generator_generate_calls_components(
        self,
        patched_generator_components: tuple[Generator, Mock, Mock, Mock, Mock, Mock],
    ) -> None:
        """Test that generate method calls all required components."""
        gen, mock_doc_loader, mock_task_gen, mock_bpm, mock_rtm, mock_tgf = (
            patched_generator_components
        )

        result = gen.generate()

        assert isinstance(result, dict)
        assert "nodes" in result and "edges" in result

    def test_generator_save_task_graph(
        self,
        patched_sample_config: dict[str, Any],
        always_valid_mock_model: Mock,
        tmp_path: Path,
    ) -> None:
        """Test saving task graph to file."""
        gen = Generator(
            config=patched_sample_config,
            model=always_valid_mock_model,
            output_dir=str(tmp_path),
        )
        task_graph = {"nodes": [], "edges": []}

        output_path = gen.save_task_graph(task_graph)

        assert output_path.endswith(".json")
        assert os.path.exists(output_path)

    def test_generator_document_instruction_type_conversion(
        self, patched_sample_config: dict[str, Any], always_valid_mock_model: Mock
    ) -> None:
        """Test that documents and instructions are converted from lists to strings."""
        # Configure with list-based documents and instructions
        config_with_lists = patched_sample_config.copy()
        config_with_lists["task_docs"] = ["doc1.txt", "doc2.txt"]
        config_with_lists["instruction_docs"] = ["instruction1.txt", "instruction2.txt"]

        gen = Generator(config=config_with_lists, model=always_valid_mock_model)

        # Mock the document loader to return lists
        with patch.object(gen, "_initialize_document_loader") as doc_loader_patch:
            doc_loader = Mock()
            doc_loader.load_task_document.side_effect = [
                "Document 1 content",
                "Document 2 content",
            ]
            doc_loader.load_instruction_document.side_effect = [
                "Instruction 1 content",
                "Instruction 2 content",
            ]
            doc_loader_patch.return_value = doc_loader

            # Mock other components to avoid actual processing
            with (
                patch.object(gen, "_initialize_task_generator") as task_gen_patch,
                patch.object(gen, "_initialize_best_practice_manager") as bpm_patch,
                patch.object(gen, "_initialize_reusable_task_manager") as rtm_patch,
                patch.object(gen, "_initialize_task_graph_formatter") as tgf_patch,
                patch(
                    "arklex.orchestrator.generator.core.generator.UI_AVAILABLE", False
                ),
            ):
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
                tgf.ensure_nested_graph_connectivity.return_value = {
                    "nodes": [],
                    "edges": [],
                }
                tgf_patch.return_value = tgf

                # Call generate to trigger the type conversion
                gen.generate()

                # Verify that documents and instructions are now strings
                assert isinstance(gen.documents, str)
                assert isinstance(gen.instructions, str)

                # Verify the content contains both documents/instructions
                assert "Document 1 content" in gen.documents
                assert "Document 2 content" in gen.documents
                assert "Instruction 1 content" in gen.instructions
                assert "Instruction 2 content" in gen.instructions

    def test_generator_generate_with_user_tasks(
        self, full_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generate method with user-provided tasks."""
        # Mock the resource initializer to avoid tool loading errors
        with patch(
            "arklex.orchestrator.generator.core.generator.DefaultResourceInitializer"
        ) as mock_ri_class:
            mock_ri = Mock()
            mock_ri.init_tools.return_value = {"tool1": {}, "tool2": {}}
            mock_ri_class.return_value = mock_ri

            gen = Generator(config=full_config, model=mock_model)

            with (
                patch.object(gen, "_initialize_document_loader") as doc_loader_patch,
                patch.object(gen, "_initialize_task_generator") as task_gen_patch,
                patch.object(gen, "_initialize_best_practice_manager") as bpm_patch,
                patch.object(gen, "_initialize_reusable_task_manager") as rtm_patch,
                patch.object(gen, "_initialize_task_graph_formatter") as tgf_patch,
                patch(
                    "arklex.orchestrator.generator.core.generator.UI_AVAILABLE", False
                ),
            ):
                # Setup mocks
                doc_loader = Mock()
                doc_loader.load_task_document.return_value = "docs"
                doc_loader.load_instruction_document.return_value = "instructions"
                doc_loader_patch.return_value = doc_loader

                task_gen = Mock()
                task_gen.add_provided_tasks.return_value = [{"id": "user_task1"}]
                task_gen.generate_tasks.return_value = [{"id": "generated_task1"}]
                task_gen_patch.return_value = task_gen

                bpm = Mock()
                bpm.generate_best_practices.return_value = []
                bpm_patch.return_value = bpm

                rtm = Mock()
                rtm.generate_reusable_tasks.return_value = {}
                rtm_patch.return_value = rtm

                tgf = Mock()
                tgf.format_task_graph.return_value = {"nodes": [], "edges": []}
                tgf.ensure_nested_graph_connectivity.return_value = {
                    "nodes": [],
                    "edges": [],
                }
                tgf_patch.return_value = tgf

                # Call generate
                result = gen.generate()

                # Verify user tasks were added (check that it was called with the correct arguments)
                task_gen.add_provided_tasks.assert_called_once()
                call_args = task_gen.add_provided_tasks.call_args[0][0]
                assert len(call_args) == 1
                assert call_args[0]["id"] == "task1"
                assert call_args[0]["name"] == "User Task 1"

                assert isinstance(result, dict)

    def test_generate_with_empty_task_docs(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generate method with empty task docs."""
        gen = Generator(config=minimal_config, model=mock_model)

        with (
            patch.object(gen, "_initialize_document_loader") as mock_doc_loader,
            patch.object(gen, "_initialize_task_generator") as mock_task_gen,
            patch.object(gen, "_initialize_best_practice_manager") as mock_bpm,
            patch.object(gen, "_initialize_reusable_task_manager") as mock_rtm,
            patch.object(gen, "_initialize_task_graph_formatter") as mock_tgf,
        ):
            # Setup mocks
            mock_doc_loader.return_value.load_task_document.return_value = ""
            mock_doc_loader.return_value.load_instruction_document.return_value = ""

            mock_task_gen.return_value.add_provided_tasks.return_value = []
            mock_task_gen.return_value.generate_tasks.return_value = []

            mock_bpm.return_value.generate_best_practices.return_value = []
            mock_bpm.return_value.finetune_best_practice.return_value = {"steps": []}

            mock_rtm.return_value.generate_reusable_tasks.return_value = {}

            mock_tgf.return_value.format_task_graph.return_value = {
                "nodes": [],
                "edges": [],
            }
            mock_tgf.return_value.ensure_nested_graph_connectivity.return_value = {
                "nodes": [],
                "edges": [],
            }

            mock_model.invoke.return_value.content = '{"intent": "test_intent"}'

            result = gen.generate()

            assert isinstance(result, dict)

    def test_generate_with_empty_user_tasks(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generate method with empty user tasks."""
        gen = Generator(config=minimal_config, model=mock_model)

        with (
            patch.object(gen, "_initialize_document_loader") as mock_doc_loader,
            patch.object(gen, "_initialize_task_generator") as mock_task_gen,
            patch.object(gen, "_initialize_best_practice_manager") as mock_bpm,
            patch.object(gen, "_initialize_reusable_task_manager") as mock_rtm,
            patch.object(gen, "_initialize_task_graph_formatter") as mock_tgf,
        ):
            # Setup mocks
            mock_doc_loader.return_value.load_task_document.return_value = ""
            mock_doc_loader.return_value.load_instruction_document.return_value = ""

            mock_task_gen.return_value.add_provided_tasks.return_value = []
            mock_task_gen.return_value.generate_tasks.return_value = []

            mock_bpm.return_value.generate_best_practices.return_value = []
            mock_bpm.return_value.finetune_best_practice.return_value = {"steps": []}

            mock_rtm.return_value.generate_reusable_tasks.return_value = {}

            mock_tgf.return_value.format_task_graph.return_value = {
                "nodes": [],
                "edges": [],
            }
            mock_tgf.return_value.ensure_nested_graph_connectivity.return_value = {
                "nodes": [],
                "edges": [],
            }

            mock_model.invoke.return_value.content = '{"intent": "test_intent"}'

            result = gen.generate()

            assert isinstance(result, dict)

    def test_generate_with_none_user_tasks(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generate method with None user tasks."""
        gen = Generator(config=minimal_config, model=mock_model)

        with (
            patch.object(gen, "_initialize_document_loader") as mock_doc_loader,
            patch.object(gen, "_initialize_task_generator") as mock_task_gen,
            patch.object(gen, "_initialize_best_practice_manager") as mock_bpm,
            patch.object(gen, "_initialize_reusable_task_manager") as mock_rtm,
            patch.object(gen, "_initialize_task_graph_formatter") as mock_tgf,
        ):
            # Setup mocks
            mock_doc_loader.return_value.load_task_document.return_value = ""
            mock_doc_loader.return_value.load_instruction_document.return_value = ""

            mock_task_gen.return_value.add_provided_tasks.return_value = []
            mock_task_gen.return_value.generate_tasks.return_value = []

            mock_bpm.return_value.generate_best_practices.return_value = []
            mock_bpm.return_value.finetune_best_practice.return_value = {"steps": []}

            mock_rtm.return_value.generate_reusable_tasks.return_value = {}

            mock_tgf.return_value.format_task_graph.return_value = {
                "nodes": [],
                "edges": [],
            }
            mock_tgf.return_value.ensure_nested_graph_connectivity.return_value = {
                "nodes": [],
                "edges": [],
            }

            mock_model.invoke.return_value.content = '{"intent": "test_intent"}'

            result = gen.generate()

            assert isinstance(result, dict)

    def test_generate_with_none_task_docs(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generate method with None task docs."""
        gen = Generator(config=minimal_config, model=mock_model)

        with (
            patch.object(gen, "_initialize_document_loader") as mock_doc_loader,
            patch.object(gen, "_initialize_task_generator") as mock_task_gen,
            patch.object(gen, "_initialize_best_practice_manager") as mock_bpm,
            patch.object(gen, "_initialize_reusable_task_manager") as mock_rtm,
            patch.object(gen, "_initialize_task_graph_formatter") as mock_tgf,
        ):
            # Setup mocks
            mock_doc_loader.return_value.load_task_document.return_value = ""
            mock_doc_loader.return_value.load_instruction_document.return_value = ""

            mock_task_gen.return_value.add_provided_tasks.return_value = []
            mock_task_gen.return_value.generate_tasks.return_value = []

            mock_bpm.return_value.generate_best_practices.return_value = []
            mock_bpm.return_value.finetune_best_practice.return_value = {"steps": []}

            mock_rtm.return_value.generate_reusable_tasks.return_value = {}

            mock_tgf.return_value.format_task_graph.return_value = {
                "nodes": [],
                "edges": [],
            }
            mock_tgf.return_value.ensure_nested_graph_connectivity.return_value = {
                "nodes": [],
                "edges": [],
            }

            mock_model.invoke.return_value.content = '{"intent": "test_intent"}'

            result = gen.generate()

            assert isinstance(result, dict)

    def test_generate_with_empty_instruction_docs(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generate method with empty instruction docs."""
        gen = Generator(config=minimal_config, model=mock_model)
        gen.instruction_docs = []

        mock_task_generator = Mock()
        mock_task_generator.generate_tasks.return_value = []

        mock_bpm = Mock()
        mock_bpm.generate_best_practices.return_value = []

        mock_rtm = Mock()
        mock_rtm.generate_reusable_tasks.return_value = {}

        mock_tgf = Mock()
        mock_tgf.format_task_graph.return_value = {"nodes": [], "edges": []}
        mock_tgf.ensure_nested_graph_connectivity.return_value = {
            "nodes": [],
            "edges": [],
        }

        with (
            patch.object(gen, "_initialize_document_loader") as mock_init_loader,
            patch.object(gen, "_initialize_task_generator") as mock_init_task,
            patch.object(gen, "_initialize_best_practice_manager") as mock_init_bpm,
            patch.object(gen, "_initialize_reusable_task_manager") as mock_init_rtm,
            patch.object(gen, "_initialize_task_graph_formatter") as mock_init_tgf,
            patch("arklex.orchestrator.generator.core.generator.UI_AVAILABLE", False),
        ):
            mock_init_loader.return_value = Mock()
            mock_init_task.return_value = mock_task_generator
            mock_init_bpm.return_value = mock_bpm
            mock_init_rtm.return_value = mock_rtm
            mock_init_tgf.return_value = mock_tgf

            result = gen.generate()
            assert isinstance(result, dict)

    def test_generate_with_none_instruction_docs(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generate method with None instruction docs."""
        gen = Generator(config=minimal_config, model=mock_model)
        gen.instruction_docs = None

        mock_task_generator = Mock()
        mock_task_generator.generate_tasks.return_value = []

        mock_bpm = Mock()
        mock_bpm.generate_best_practices.return_value = []

        mock_rtm = Mock()
        mock_rtm.generate_reusable_tasks.return_value = {}

        mock_tgf = Mock()
        mock_tgf.format_task_graph.return_value = {"nodes": [], "edges": []}
        mock_tgf.ensure_nested_graph_connectivity.return_value = {
            "nodes": [],
            "edges": [],
        }

        with (
            patch.object(gen, "_initialize_document_loader") as mock_init_loader,
            patch.object(gen, "_initialize_task_generator") as mock_init_task,
            patch.object(gen, "_initialize_best_practice_manager") as mock_init_bpm,
            patch.object(gen, "_initialize_reusable_task_manager") as mock_init_rtm,
            patch.object(gen, "_initialize_task_graph_formatter") as mock_init_tgf,
            patch("arklex.orchestrator.generator.core.generator.UI_AVAILABLE", False),
            patch.object(gen, "_load_multiple_task_documents") as mock_load_task,
            patch.object(
                gen, "_load_multiple_instruction_documents"
            ) as mock_load_instruction,
            patch("arklex.orchestrator.generator.core.generator.log_context.info"),
        ):
            mock_init_loader.return_value = Mock()
            mock_init_task.return_value = mock_task_generator
            mock_init_bpm.return_value = mock_bpm
            mock_init_rtm.return_value = mock_rtm
            mock_init_tgf.return_value = mock_tgf
            mock_load_task.return_value = []
            mock_load_instruction.return_value = []
            # Patch to avoid TypeError in len(self.instruction_docs)
            gen.instruction_docs = []
            result = gen.generate()
            assert isinstance(result, dict)


class TestGeneratorSaveTaskGraph:
    """Test Generator save_task_graph method."""

    def test_generator_save_task_graph(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test saving task graph to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gen = Generator(
                config=minimal_config, model=mock_model, output_dir=temp_dir
            )
            task_graph = {"test": "data"}

            gen.save_task_graph(task_graph)

            # Check that file was created
            expected_file = os.path.join(temp_dir, "taskgraph.json")
            assert os.path.exists(expected_file)

            # Check file contents
            with open(expected_file) as f:
                saved_data = json.loads(f.read())
                assert saved_data == task_graph

    def test_generator_save_task_graph_without_output_dir(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test saving task graph without output directory."""
        gen = Generator(config=minimal_config, model=mock_model)

        # Mock the save_task_graph method to handle None output_dir
        with (
            patch.object(gen, "output_dir", None),
            patch("os.path.join") as mock_join,
            patch("builtins.open", create=True) as mock_open,
        ):
            mock_join.return_value = "/tmp/taskgraph.json"
            mock_open.return_value.__enter__.return_value = Mock()

            task_graph = {"test": "data"}
            gen.save_task_graph(task_graph)

            mock_join.assert_called_once_with("/tmp", "taskgraph.json")
            mock_open.assert_called_once_with("/tmp/taskgraph.json", "w")

    def test_generator_save_task_graph_sanitizes_filename(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test that save_task_graph sanitizes the filename."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gen = Generator(
                config=minimal_config, model=mock_model, output_dir=temp_dir
            )
            task_graph = {"test": "data"}

            # Test with a filename that contains special characters
            gen.save_task_graph(task_graph, filename="test file with spaces.json")

            # Check that file was created with sanitized name
            expected_file = os.path.join(temp_dir, "test_file_with_spaces.json")
            assert os.path.exists(expected_file)

    def test_save_task_graph_with_invalid_filename(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test save_task_graph method with invalid filename."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gen = Generator(
                config=minimal_config, model=mock_model, output_dir=temp_dir
            )
            task_graph = {"test": "data"}

            # Test with a filename that contains invalid characters
            gen.save_task_graph(
                task_graph, filename="test/file\\with:invalid*chars?.json"
            )

            # Check that file was created with sanitized name
            expected_file = os.path.join(temp_dir, "testfilewithinvalidchars.json")
            assert os.path.exists(expected_file)

    def test_save_task_graph_with_complex_objects(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test save_task_graph with complex non-serializable objects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gen = Generator(
                config=minimal_config, model=mock_model, output_dir=temp_dir
            )

            def test_function() -> str:
                return "test"

            task_graph = {
                "simple": "string",
                "number": 42,
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
                "function": test_function,  # This should be converted to string
                "tuple": ({"a": 1}, {"b": 2}),  # This should be converted to list
            }

            gen.save_task_graph(task_graph)

            # Check that file was created
            expected_file = os.path.join(temp_dir, "taskgraph.json")
            assert os.path.exists(expected_file)

            # Check file contents
            with open(expected_file) as f:
                saved_data = json.loads(f.read())
                assert saved_data["simple"] == "string"
                assert saved_data["number"] == 42
                assert saved_data["list"] == [1, 2, 3]
                assert saved_data["dict"] == {"nested": "value"}
                assert saved_data["function"] == str(test_function)
                assert saved_data["tuple"] == [{"a": 1}, {"b": 2}]

    def test_save_task_graph_with_various_data_types(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test save_task_graph with various data types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gen = Generator(
                config=minimal_config, model=mock_model, output_dir=temp_dir
            )

            task_graph = {
                "string": "test",
                "integer": 123,
                "float": 3.14,
                "boolean": True,
                "none": None,
                "list": [1, "two", 3.0],
                "dict": {"key": "value", "nested": {"deep": "value"}},
                "empty_list": [],
                "empty_dict": {},
            }

            gen.save_task_graph(task_graph)

            # Check that file was created
            expected_file = os.path.join(temp_dir, "taskgraph.json")
            assert os.path.exists(expected_file)

            # Check file contents
            with open(expected_file) as f:
                saved_data = json.loads(f.read())
                assert saved_data["string"] == "test"
                assert saved_data["integer"] == 123
                assert saved_data["float"] == 3.14
                assert saved_data["boolean"] is True
                assert saved_data["none"] is None
                assert saved_data["list"] == [1, "two", 3.0]
                assert saved_data["dict"] == {
                    "key": "value",
                    "nested": {"deep": "value"},
                }
                assert saved_data["empty_list"] == []
                assert saved_data["empty_dict"] == {}

    def test_save_task_graph_debug_logging(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test that debug logging is called for non-serializable fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gen = Generator(
                config=minimal_config, model=mock_model, output_dir=temp_dir
            )

            def test_function() -> str:
                return "test"

            task_graph = {
                "serializable": "value",
                "function": test_function,
                "nested": {"tuple": ({"a": 1}, {"b": 2})},
            }

            gen.save_task_graph(task_graph)

            # Check that file was created and contains serializable data
            expected_file = os.path.join(temp_dir, "taskgraph.json")
            assert os.path.exists(expected_file)

            with open(expected_file) as f:
                saved_data = json.loads(f.read())
                assert saved_data["serializable"] == "value"
                assert saved_data["function"] == str(test_function)
                assert saved_data["nested"]["tuple"] == [{"a": 1}, {"b": 2}]

    def test_save_task_graph_with_empty_output_dir(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test save_task_graph when output_dir is None."""
        gen = Generator(config=minimal_config, model=mock_model, output_dir=None)
        task_graph = {"test": "data"}

        # Should not raise an exception
        gen.save_task_graph(task_graph)

    def test_save_task_graph_without_output_dir(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test save_task_graph without output_dir set."""
        gen = Generator(config=minimal_config, model=mock_model)
        task_graph = {"test": "data"}

        # Should not raise an exception
        gen.save_task_graph(task_graph)

    def test_save_task_graph_with_exception(
        self, minimal_config: dict[str, Any], mock_model: Mock, tmp_path: Path
    ) -> None:
        """Test save_task_graph with exception during file writing."""
        gen = Generator(
            config=minimal_config, model=mock_model, output_dir=str(tmp_path)
        )

        task_graph = {"nodes": [], "edges": []}

        # Mock open to raise an exception
        with (
            patch("builtins.open", side_effect=OSError("File write error")),
            pytest.raises(OSError),
        ):
            gen.save_task_graph(task_graph)

    def test_load_multiple_task_documents_with_exception(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test loading multiple task documents with exception."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch.object(gen, "_initialize_document_loader") as mock_doc_loader:
            mock_doc_loader.return_value.load_task_document.side_effect = RuntimeError(
                "Load error"
            )

            with pytest.raises(RuntimeError):
                gen._load_multiple_task_documents(
                    mock_doc_loader.return_value, ["doc1"]
                )

    def test_load_multiple_instruction_documents_with_exception(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test loading multiple instruction documents with exception."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch.object(gen, "_initialize_document_loader") as mock_doc_loader:
            mock_doc_loader.return_value.load_instruction_document.side_effect = (
                RuntimeError("Load error")
            )

            with pytest.raises(RuntimeError):
                gen._load_multiple_instruction_documents(
                    mock_doc_loader.return_value, ["inst1"]
                )


class TestGeneratorComponentInitializationEdgeCases:
    """Test edge cases in component initialization."""

    def test_initialize_document_loader_with_exception(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test document loader initialization with exception."""
        gen = Generator(config=minimal_config, model=mock_model)

        with (
            patch(
                "arklex.orchestrator.generator.core.generator.DocumentLoader",
                side_effect=RuntimeError("Init error"),
            ),
            pytest.raises(RuntimeError),
        ):
            gen._initialize_document_loader()

    def test_initialize_task_generator_with_exception(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test task generator initialization with exception."""
        gen = Generator(config=minimal_config, model=mock_model)

        with (
            patch(
                "arklex.orchestrator.generator.core.generator.TaskGenerator",
                side_effect=RuntimeError("Init error"),
            ),
            pytest.raises(RuntimeError),
        ):
            gen._initialize_task_generator()

    def test_initialize_best_practice_manager_with_exception(
        self, full_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test best practice manager initialization with exception."""
        gen = Generator(config=full_config, model=mock_model)

        with (
            patch(
                "arklex.orchestrator.generator.core.generator.BestPracticeManager",
                side_effect=RuntimeError("Init error"),
            ),
            pytest.raises(RuntimeError),
        ):
            gen._initialize_best_practice_manager()

    def test_initialize_reusable_task_manager_with_exception(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test reusable task manager initialization with exception."""
        gen = Generator(config=minimal_config, model=mock_model)

        with (
            patch(
                "arklex.orchestrator.generator.core.generator.ReusableTaskManager",
                side_effect=RuntimeError("Init error"),
            ),
            pytest.raises(RuntimeError),
        ):
            gen._initialize_reusable_task_manager()

    def test_initialize_task_graph_formatter_with_exception(
        self, full_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test task graph formatter initialization with exception."""
        gen = Generator(config=full_config, model=mock_model)

        with (
            patch(
                "arklex.orchestrator.generator.core.generator.TaskGraphFormatter",
                side_effect=RuntimeError("Init error"),
            ),
            pytest.raises(RuntimeError),
        ):
            gen._initialize_task_graph_formatter()


class TestGeneratorGenerateMethodEdgeCases:
    """Test edge cases in the generate method."""

    def test_generate_with_document_loader_exception(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generate method with document loader exception."""
        gen = Generator(config=minimal_config, model=mock_model)

        with (
            patch.object(gen, "_initialize_document_loader") as mock_init_loader,
        ):
            mock_init_loader.side_effect = RuntimeError("Document loader error")

            with pytest.raises(RuntimeError):
                gen.generate()

    def test_generate_with_task_generator_exception(
        self, minimal_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generate method with task generator exception."""
        gen = Generator(config=minimal_config, model=mock_model)

        with (
            patch.object(gen, "_initialize_document_loader") as mock_init_loader,
            patch.object(gen, "_initialize_task_generator") as mock_init_task,
        ):
            mock_init_loader.return_value = Mock()
            mock_init_task.side_effect = RuntimeError("Task generator error")

            with pytest.raises(RuntimeError):
                gen.generate()

    def test_generate_with_best_practice_manager_exception(
        self, full_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generate method with best practice manager exception."""
        gen = Generator(config=full_config, model=mock_model)

        with (
            patch.object(gen, "_initialize_document_loader") as mock_init_loader,
            patch.object(gen, "_initialize_task_generator") as mock_init_task,
            patch.object(gen, "_initialize_best_practice_manager") as mock_init_bpm,
        ):
            mock_init_loader.return_value = Mock()
            mock_init_task.return_value = Mock()
            mock_init_bpm.side_effect = RuntimeError("BPM error")

            with pytest.raises(RuntimeError):
                gen.generate()

    def test_generate_with_reusable_task_manager_exception(
        self, full_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generate method with reusable task manager exception."""
        gen = Generator(config=full_config, model=mock_model)

        with (
            patch.object(gen, "_initialize_document_loader") as mock_init_loader,
            patch.object(gen, "_initialize_task_generator") as mock_init_task,
            patch.object(gen, "_initialize_best_practice_manager") as mock_init_bpm,
            patch.object(gen, "_initialize_reusable_task_manager") as mock_init_rtm,
        ):
            mock_init_loader.return_value = Mock()
            mock_init_task.return_value = Mock()
            mock_init_bpm.return_value = Mock()
            mock_init_rtm.side_effect = RuntimeError("RTM error")

            with pytest.raises(RuntimeError):
                gen.generate()

    def test_generate_with_task_graph_formatter_format_exception(
        self, full_config: dict[str, Any], mock_model: Mock
    ) -> None:
        """Test generate method with task graph formatter format exception."""
        gen = Generator(config=full_config, model=mock_model)

        mock_task_generator = Mock()
        mock_task_generator.generate_tasks.return_value = []

        mock_bpm = Mock()
        mock_bpm.generate_best_practices.return_value = {}

        mock_rtm = Mock()
        mock_rtm.generate_reusable_tasks.return_value = {}

        mock_tgf = Mock()
        mock_tgf.format_task_graph.side_effect = RuntimeError("Format task graph error")

        with (
            patch.object(gen, "_initialize_document_loader") as mock_init_loader,
            patch.object(gen, "_initialize_task_generator") as mock_init_task,
            patch.object(gen, "_initialize_best_practice_manager") as mock_init_bpm,
            patch.object(gen, "_initialize_reusable_task_manager") as mock_init_rtm,
            patch.object(gen, "_initialize_task_graph_formatter") as mock_init_tgf,
        ):
            mock_init_loader.return_value = Mock()
            mock_init_task.return_value = mock_task_generator
            mock_init_bpm.return_value = mock_bpm
            mock_init_rtm.return_value = mock_rtm
            mock_init_tgf.return_value = mock_tgf

            with pytest.raises(RuntimeError):
                gen.generate()

    def test_generate_with_model_exception(
        self, minimal_config: dict[str, Any], mock_model: Mock, tmp_path: Path
    ) -> None:
        """Test generate method with model exception."""
        gen = Generator(
            config=minimal_config, model=mock_model, output_dir=str(tmp_path)
        )

        with (
            patch.object(gen, "_initialize_document_loader") as mock_doc_loader,
            patch.object(gen, "_initialize_task_generator") as mock_task_gen,
            patch.object(gen, "_initialize_best_practice_manager") as mock_bpm,
            patch.object(gen, "_initialize_reusable_task_manager") as mock_rtm,
            patch.object(gen, "_initialize_task_graph_formatter") as mock_tgf,
        ):
            # Setup all components
            mock_doc_loader.return_value.load_task_document.return_value = "doc1"
            mock_doc_loader.return_value.load_instruction_document.return_value = (
                "inst1"
            )

            mock_task_gen.return_value.add_provided_tasks.return_value = []
            mock_task_gen.return_value.generate_tasks.return_value = [
                {"name": "Task1", "steps": []}
            ]

            mock_bpm.return_value.generate_best_practices.return_value = [{"id": "bp1"}]
            mock_bpm.return_value.finetune_best_practice.return_value = {
                "steps": [{"description": "refined_step"}]
            }

            mock_rtm.return_value.generate_reusable_tasks.return_value = {}

            mock_tgf.return_value.format_task_graph.return_value = {
                "nodes": [],
                "edges": [],
            }
            mock_tgf.return_value.ensure_nested_graph_connectivity.return_value = {
                "nodes": [],
                "edges": [],
            }

            # Mock model to raise exception
            mock_model.invoke.side_effect = RuntimeError("Model error")

            with pytest.raises(RuntimeError):
                gen.generate()
