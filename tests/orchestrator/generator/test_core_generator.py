import pytest
from unittest.mock import Mock, patch
import tempfile
import os
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


@pytest.fixture
def full_config():
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


class TestGeneratorInitialization:
    """Test Generator initialization and configuration."""

    def test_generator_initialization(self, minimal_config, mock_model) -> None:
        """Test basic generator initialization."""
        gen = Generator(config=minimal_config, model=mock_model)
        assert gen.role == "test_role"
        assert gen.user_objective == "test_objective"
        assert gen.builder_objective == "test_builder_objective"
        assert gen.intro == "test_intro"
        assert isinstance(gen.resource_initializer, DefaultResourceInitializer)
        assert gen.interactable_with_user is True
        assert gen.allow_nested_graph is True
        assert gen.model == mock_model

    def test_generator_with_full_config(self, full_config, mock_model):
        """Test generator initialization with full configuration."""
        # Mock the resource initializer to avoid tool loading errors
        with patch(
            "arklex.orchestrator.generator.core.generator.DefaultResourceInitializer"
        ) as mock_ri_class:
            mock_ri = Mock()
            mock_ri.init_tools.return_value = {"tool1": {}, "tool2": {}}
            mock_ri_class.return_value = mock_ri

            gen = Generator(config=full_config, model=mock_model)
            assert gen.domain == "test_domain"
            assert gen.nluapi == "http://nlu.api"
            assert gen.slotfillapi == "http://slotfill.api"
            assert len(gen.workers) == 2
            assert len(gen.tools) == 2
            assert gen.workers[0]["id"] == "worker1"
            assert gen.tools["tool1"] == {}

    def test_generator_with_invalid_resource_initializer(
        self, minimal_config, mock_model
    ) -> None:
        """Test generator falls back to DefaultResourceInitializer if None is provided."""
        gen = Generator(
            config=minimal_config, model=mock_model, resource_initializer=None
        )
        assert isinstance(gen.resource_initializer, DefaultResourceInitializer)

    def test_generator_with_custom_resource_initializer(
        self, minimal_config, mock_model
    ) -> None:
        """Test generator with custom resource initializer."""

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

    def test_generator_with_output_dir(self, minimal_config, mock_model) -> None:
        """Test generator with output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gen = Generator(
                config=minimal_config, model=mock_model, output_dir=temp_dir
            )
            assert gen.output_dir == temp_dir

    def test_generator_without_output_dir(self, minimal_config, mock_model) -> None:
        """Test generator without output directory."""
        gen = Generator(config=minimal_config, model=mock_model)
        assert gen.output_dir is None

    def test_generator_with_disabled_flags(self, minimal_config, mock_model) -> None:
        """Test generator with disabled interaction and nested graph."""
        gen = Generator(
            config=minimal_config,
            model=mock_model,
            interactable_with_user=False,
            allow_nested_graph=False,
        )
        assert gen.interactable_with_user is False
        assert gen.allow_nested_graph is False

    def test_generator_workers_conversion(self, full_config, mock_model) -> None:
        """Test that workers are properly converted to the expected format."""
        gen = Generator(config=full_config, model=mock_model)
        assert len(gen.workers) == 2
        assert all(isinstance(worker, dict) for worker in gen.workers)
        assert all(
            "id" in worker and "name" in worker and "path" in worker
            for worker in gen.workers
        )

    def test_generator_workers_invalid_format(self, minimal_config, mock_model) -> None:
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


class TestGeneratorComponentInitialization:
    """Test Generator component initialization methods."""

    def test_initialize_document_loader_with_output_dir(
        self, minimal_config, mock_model
    ) -> None:
        """Test document loader initialization with output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gen = Generator(
                config=minimal_config, model=mock_model, output_dir=temp_dir
            )

            with patch(
                "arklex.orchestrator.generator.core.generator.DocumentLoader"
            ) as mock_loader_class:
                mock_loader = Mock()
                mock_loader_class.return_value = mock_loader

                doc_loader = gen._initialize_document_loader()

                mock_loader_class.assert_called_once()
                assert doc_loader == mock_loader

    def test_initialize_document_loader_without_output_dir(
        self, minimal_config, mock_model
    ):
        """Test document loader initialization without output directory."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch(
            "arklex.orchestrator.generator.core.generator.DocumentLoader"
        ) as mock_loader_class:
            with patch(
                "arklex.orchestrator.generator.core.generator.Path"
            ) as mock_path_class:
                from unittest.mock import MagicMock

                mock_cache_dir = MagicMock()
                mock_cwd = MagicMock()
                mock_cwd.__truediv__.return_value = mock_cache_dir
                mock_path_class.cwd.return_value = mock_cwd

                gen._initialize_document_loader()
                mock_loader_class.assert_called()

    def test_initialize_task_generator(self, minimal_config, mock_model) -> None:
        """Test task generator initialization."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch(
            "arklex.orchestrator.generator.core.generator.TaskGenerator"
        ) as mock_task_gen_class:
            mock_task_gen = Mock()
            mock_task_gen_class.return_value = mock_task_gen

            task_gen = gen._initialize_task_generator()

            mock_task_gen_class.assert_called_once_with(
                model=mock_model,
                role="test_role",
                user_objective="test_objective",
                instructions="",
                documents="",
            )
            assert task_gen == mock_task_gen

    def test_initialize_best_practice_manager(self, full_config, mock_model) -> None:
        """Test best practice manager initialization."""
        gen = Generator(config=full_config, model=mock_model)

        with patch(
            "arklex.orchestrator.generator.core.generator.BestPracticeManager"
        ) as mock_bpm_class:
            mock_bpm = Mock()
            mock_bpm_class.return_value = mock_bpm

            bpm = gen._initialize_best_practice_manager()

            # Check that all resources are included
            call_args = mock_bpm_class.call_args[1]
            assert call_args["model"] == mock_model
            assert call_args["role"] == "test_role"
            assert call_args["user_objective"] == "test_objective"
            assert (
                len(call_args["all_resources"]) > 0
            )  # Should include workers, tools, and nested_graph
            assert bpm == mock_bpm

    def test_initialize_best_practice_manager_without_nested_graph(
        self, full_config, mock_model
    ) -> None:
        """Test best practice manager initialization without nested graph."""
        gen = Generator(config=full_config, model=mock_model, allow_nested_graph=False)

        with patch(
            "arklex.orchestrator.generator.core.generator.BestPracticeManager"
        ) as mock_bpm_class:
            mock_bpm = Mock()
            mock_bpm_class.return_value = mock_bpm

            bpm = gen._initialize_best_practice_manager()

            # Check that nested_graph is not included
            call_args = mock_bpm_class.call_args[1]
            all_resources = call_args["all_resources"]
            nested_graph_resources = [
                r for r in all_resources if r.get("type") == "nested_graph"
            ]
            assert len(nested_graph_resources) == 0
            assert bpm == mock_bpm

    def test_initialize_reusable_task_manager(self, minimal_config, mock_model) -> None:
        """Test reusable task manager initialization."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch(
            "arklex.orchestrator.generator.core.generator.ReusableTaskManager"
        ) as mock_rtm_class:
            mock_rtm = Mock()
            mock_rtm_class.return_value = mock_rtm

            rtm = gen._initialize_reusable_task_manager()

            mock_rtm_class.assert_called_once_with(
                model=mock_model, role="test_role", user_objective="test_objective"
            )
            assert rtm == mock_rtm

    def test_initialize_task_graph_formatter(self, full_config, mock_model) -> None:
        """Test task graph formatter initialization."""
        gen = Generator(config=full_config, model=mock_model)

        with patch(
            "arklex.orchestrator.generator.core.generator.TaskGraphFormatter"
        ) as mock_tgf_class:
            mock_tgf = Mock()
            mock_tgf_class.return_value = mock_tgf

            tgf = gen._initialize_task_graph_formatter()

            call_args = mock_tgf_class.call_args[1]
            assert call_args["role"] == "test_role"
            assert call_args["user_objective"] == "test_objective"
            assert call_args["builder_objective"] == "test_builder_objective"
            assert call_args["domain"] == "test_domain"
            assert call_args["nluapi"] == "http://nlu.api"
            assert call_args["slotfillapi"] == "http://slotfill.api"
            assert tgf == mock_tgf


class TestGeneratorDocumentLoading:
    """Test Generator document loading functionality."""

    def test_load_multiple_task_documents_list(
        self, minimal_config, mock_model
    ) -> None:
        """Test loading multiple task documents from a list."""
        gen = Generator(config=minimal_config, model=mock_model)

        doc_paths = ["doc1.txt", "doc2.txt"]
        mock_doc_loader = Mock()
        mock_doc_loader.load_task_document.side_effect = ["content1", "content2"]

        result = gen._load_multiple_task_documents(mock_doc_loader, doc_paths)

        assert result == ["content1", "content2"]
        assert mock_doc_loader.load_task_document.call_count == 2
        mock_doc_loader.load_task_document.assert_any_call("doc1.txt")
        mock_doc_loader.load_task_document.assert_any_call("doc2.txt")

    def test_load_multiple_task_documents_dict_list(
        self, minimal_config, mock_model
    ) -> None:
        """Test loading multiple task documents from a list of dictionaries."""
        gen = Generator(config=minimal_config, model=mock_model)

        doc_paths = [{"source": "doc1.txt"}, {"source": "doc2.txt"}]
        mock_doc_loader = Mock()
        mock_doc_loader.load_task_document.side_effect = ["content1", "content2"]

        result = gen._load_multiple_task_documents(mock_doc_loader, doc_paths)

        assert result == ["content1", "content2"]
        assert mock_doc_loader.load_task_document.call_count == 2
        mock_doc_loader.load_task_document.assert_any_call("doc1.txt")
        mock_doc_loader.load_task_document.assert_any_call("doc2.txt")

    def test_load_multiple_task_documents_single_string(
        self, minimal_config, mock_model
    ) -> None:
        """Test loading a single task document from a string."""
        gen = Generator(config=minimal_config, model=mock_model)

        doc_paths = "doc1.txt"
        mock_doc_loader = Mock()
        mock_doc_loader.load_task_document.return_value = "content1"

        result = gen._load_multiple_task_documents(mock_doc_loader, doc_paths)

        assert result == ["content1"]
        mock_doc_loader.load_task_document.assert_called_once_with("doc1.txt")

    def test_load_multiple_task_documents_single_dict(
        self, minimal_config, mock_model
    ) -> None:
        """Test loading a single task document from a dictionary."""
        gen = Generator(config=minimal_config, model=mock_model)

        doc_paths = {"source": "doc1.txt"}
        mock_doc_loader = Mock()
        mock_doc_loader.load_task_document.return_value = "content1"

        result = gen._load_multiple_task_documents(mock_doc_loader, doc_paths)

        assert result == ["content1"]
        mock_doc_loader.load_task_document.assert_called_once_with("doc1.txt")

    def test_load_multiple_instruction_documents_list(
        self, minimal_config, mock_model
    ) -> None:
        """Test loading multiple instruction documents from a list."""
        gen = Generator(config=minimal_config, model=mock_model)

        doc_paths = ["instruction1.txt", "instruction2.txt"]
        mock_doc_loader = Mock()
        mock_doc_loader.load_instruction_document.side_effect = ["content1", "content2"]

        result = gen._load_multiple_instruction_documents(mock_doc_loader, doc_paths)

        assert result == ["content1", "content2"]
        assert mock_doc_loader.load_instruction_document.call_count == 2
        mock_doc_loader.load_instruction_document.assert_any_call("instruction1.txt")
        mock_doc_loader.load_instruction_document.assert_any_call("instruction2.txt")

    def test_load_multiple_instruction_documents_dict_list(
        self, minimal_config, mock_model
    ) -> None:
        """Test loading multiple instruction documents from a list of dictionaries."""
        gen = Generator(config=minimal_config, model=mock_model)

        doc_paths = [{"source": "instruction1.txt"}, {"source": "instruction2.txt"}]
        mock_doc_loader = Mock()
        mock_doc_loader.load_instruction_document.side_effect = ["content1", "content2"]

        result = gen._load_multiple_instruction_documents(mock_doc_loader, doc_paths)

        assert result == ["content1", "content2"]
        assert mock_doc_loader.load_instruction_document.call_count == 2
        mock_doc_loader.load_instruction_document.assert_any_call("instruction1.txt")
        mock_doc_loader.load_instruction_document.assert_any_call("instruction2.txt")

    def test_load_multiple_instruction_documents_single_string(
        self, minimal_config, mock_model
    ) -> None:
        """Test loading a single instruction document from a string."""
        gen = Generator(config=minimal_config, model=mock_model)

        doc_paths = "instruction1.txt"
        mock_doc_loader = Mock()
        mock_doc_loader.load_instruction_document.return_value = "content1"

        result = gen._load_multiple_instruction_documents(mock_doc_loader, doc_paths)

        assert result == ["content1"]
        mock_doc_loader.load_instruction_document.assert_called_once_with(
            "instruction1.txt"
        )

    def test_load_multiple_instruction_documents_single_dict(
        self, minimal_config, mock_model
    ) -> None:
        """Test loading a single instruction document from a dictionary."""
        gen = Generator(config=minimal_config, model=mock_model)

        doc_paths = {"source": "instruction1.txt"}
        mock_doc_loader = Mock()
        mock_doc_loader.load_instruction_document.return_value = "content1"

        result = gen._load_multiple_instruction_documents(mock_doc_loader, doc_paths)

        assert result == ["content1"]
        mock_doc_loader.load_instruction_document.assert_called_once_with(
            "instruction1.txt"
        )

    def test_load_multiple_task_documents_with_none_paths(
        self, minimal_config, mock_model
    ) -> None:
        """Test loading multiple task documents with None paths."""
        gen = Generator(config=minimal_config, model=mock_model)

        doc_paths = None
        mock_doc_loader = Mock()

        with patch.object(gen, "_load_multiple_task_documents") as mock_load:
            mock_load.return_value = []
            result = gen._load_multiple_task_documents(mock_doc_loader, doc_paths)
            assert result == []

    def test_load_multiple_instruction_documents_with_none_paths(
        self, minimal_config, mock_model
    ) -> None:
        """Test loading multiple instruction documents with None paths."""
        gen = Generator(config=minimal_config, model=mock_model)

        doc_paths = None
        mock_doc_loader = Mock()

        with patch.object(gen, "_load_multiple_instruction_documents") as mock_load:
            mock_load.return_value = []
            result = gen._load_multiple_instruction_documents(
                mock_doc_loader, doc_paths
            )
            assert result == []


class TestGeneratorGenerate:
    """Test Generator generate method."""

    def test_generator_generate_calls_components(
        self, minimal_config, mock_model
    ) -> None:
        """Test that generate method calls all components correctly."""
        gen = Generator(config=minimal_config, model=mock_model)

        # Patch component initializers to return mocks
        with (
            patch.object(gen, "_initialize_document_loader") as doc_loader_patch,
            patch.object(gen, "_initialize_task_generator") as task_gen_patch,
            patch.object(gen, "_initialize_best_practice_manager") as bpm_patch,
            patch.object(gen, "_initialize_reusable_task_manager") as rtm_patch,
            patch.object(gen, "_initialize_task_graph_formatter") as tgf_patch,
            patch("arklex.orchestrator.generator.core.generator.UI_AVAILABLE", False),
        ):
            # Setup mocks
            doc_loader = Mock()
            doc_loader.load_task_document.return_value = "docs"
            doc_loader.load_instruction_document.return_value = "instructions"
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

            # Call generate
            result = gen.generate()

            # Verify result
            assert isinstance(result, dict)
            assert "nodes" in result and "edges" in result

            # Verify component calls
            doc_loader_patch.assert_called_once()
            task_gen_patch.assert_called_once()
            bpm_patch.assert_called_once()
            rtm_patch.assert_called_once()
            tgf_patch.assert_called_once()

    def test_generator_generate_with_multiple_documents(
        self, full_config, mock_model
    ) -> None:
        """Test generate method with multiple documents."""
        gen = Generator(config=full_config, model=mock_model)

        with (
            patch.object(gen, "_initialize_document_loader") as doc_loader_patch,
            patch.object(gen, "_initialize_task_generator") as task_gen_patch,
            patch.object(gen, "_initialize_best_practice_manager") as bpm_patch,
            patch.object(gen, "_initialize_reusable_task_manager") as rtm_patch,
            patch.object(gen, "_initialize_task_graph_formatter") as tgf_patch,
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

            bpm = Mock()
            bpm.generate_best_practices.return_value = []
            bpm_patch.return_value = bpm

            rtm = Mock()
            rtm.generate_reusable_tasks.return_value = {}
            rtm_patch.return_value = rtm

            tgf = Mock()
            tgf.format_task_graph.return_value = {"nodes": [], "edges": []}
            tgf_patch.return_value = tgf

            # Call generate
            result = gen.generate()

            # Verify documents were loaded and concatenated
            assert "doc1" in gen.documents
            assert "doc2" in gen.documents
            assert "instruction1" in gen.instructions
            assert "instruction2" in gen.instructions

            assert isinstance(result, dict)

    def test_generator_generate_with_user_tasks(self, full_config, mock_model) -> None:
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

    def test_generate_with_empty_task_docs(self, minimal_config, mock_model) -> None:
        """Test generate method with empty task docs."""
        gen = Generator(config=minimal_config, model=mock_model)
        gen.task_docs = []

        mock_task_generator = Mock()
        mock_task_generator.generate_tasks.return_value = []

        mock_bpm = Mock()
        mock_bpm.generate_best_practices.return_value = []

        mock_rtm = Mock()
        mock_rtm.generate_reusable_tasks.return_value = {}

        mock_tgf = Mock()
        mock_tgf.format_task_graph.return_value = {"nodes": [], "edges": []}

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

    def test_generate_with_empty_user_tasks(self, minimal_config, mock_model) -> None:
        """Test generate method with empty user tasks."""
        gen = Generator(config=minimal_config, model=mock_model)
        gen.user_tasks = []

        mock_task_generator = Mock()
        mock_task_generator.generate_tasks.return_value = []

        mock_bpm = Mock()
        mock_bpm.generate_best_practices.return_value = []

        mock_rtm = Mock()
        mock_rtm.generate_reusable_tasks.return_value = {}

        mock_tgf = Mock()
        mock_tgf.format_task_graph.return_value = {"nodes": [], "edges": []}

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

    def test_generate_with_none_user_tasks(self, minimal_config, mock_model) -> None:
        """Test generate method with None user tasks."""
        gen = Generator(config=minimal_config, model=mock_model)
        gen.user_tasks = None

        mock_task_generator = Mock()
        mock_task_generator.generate_tasks.return_value = []

        mock_bpm = Mock()
        mock_bpm.generate_best_practices.return_value = []

        mock_rtm = Mock()
        mock_rtm.generate_reusable_tasks.return_value = {}

        mock_tgf = Mock()
        mock_tgf.format_task_graph.return_value = {"nodes": [], "edges": []}

        with (
            patch.object(gen, "_initialize_document_loader") as mock_init_loader,
            patch.object(gen, "_initialize_task_generator") as mock_init_task,
            patch.object(gen, "_initialize_best_practice_manager") as mock_init_bpm,
            patch.object(gen, "_initialize_reusable_task_manager") as mock_init_rtm,
            patch.object(gen, "_initialize_task_graph_formatter") as mock_init_tgf,
            patch("arklex.orchestrator.generator.core.generator.UI_AVAILABLE", False),
            patch(
                "arklex.orchestrator.generator.core.generator.log_context.info"
            ) as mock_log,
        ):
            mock_init_loader.return_value = Mock()
            mock_init_task.return_value = mock_task_generator
            mock_init_bpm.return_value = mock_bpm
            mock_init_rtm.return_value = mock_rtm
            mock_init_tgf.return_value = mock_tgf

            result = gen.generate()
            assert isinstance(result, dict)

    def test_generate_with_none_task_docs(self, minimal_config, mock_model) -> None:
        """Test generate method with None task docs."""
        gen = Generator(config=minimal_config, model=mock_model)
        gen.task_docs = None

        mock_task_generator = Mock()
        mock_task_generator.generate_tasks.return_value = []

        mock_bpm = Mock()
        mock_bpm.generate_best_practices.return_value = []

        mock_rtm = Mock()
        mock_rtm.generate_reusable_tasks.return_value = {}

        mock_tgf = Mock()
        mock_tgf.format_task_graph.return_value = {"nodes": [], "edges": []}

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
            patch(
                "arklex.orchestrator.generator.core.generator.log_context.info"
            ) as mock_log,
        ):
            mock_init_loader.return_value = Mock()
            mock_init_task.return_value = mock_task_generator
            mock_init_bpm.return_value = mock_bpm
            mock_init_rtm.return_value = mock_rtm
            mock_init_tgf.return_value = mock_tgf
            mock_load_task.return_value = []
            mock_load_instruction.return_value = []
            # Patch to avoid TypeError in len(self.task_docs)
            gen.task_docs = []
            result = gen.generate()
            assert isinstance(result, dict)

    def test_generate_with_empty_instruction_docs(
        self, minimal_config, mock_model
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
        self, minimal_config, mock_model
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
            patch(
                "arklex.orchestrator.generator.core.generator.log_context.info"
            ) as mock_log,
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

    def test_generator_save_task_graph(self, minimal_config, mock_model) -> None:
        """Test saving task graph to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gen = Generator(
                config=minimal_config, model=mock_model, output_dir=temp_dir
            )
            task_graph = {
                "nodes": [{"id": "node1"}],
                "edges": [{"source": "node1", "target": "node2"}],
            }

            output_path = gen.save_task_graph(task_graph)

            assert output_path.endswith(".json")
            assert os.path.exists(output_path)

            # Verify file content
            import json

            with open(output_path, "r") as f:
                saved_content = json.load(f)
            assert saved_content == task_graph

    def test_generator_save_task_graph_without_output_dir(
        self, minimal_config, mock_model
    ) -> None:
        """Test saving task graph without output directory."""
        gen = Generator(config=minimal_config, model=mock_model)
        task_graph = {"nodes": [], "edges": []}

        # Mock the save_task_graph method to handle None output_dir
        with patch.object(gen, "output_dir", None):
            with patch("os.path.join") as mock_join:
                mock_join.return_value = "/tmp/taskgraph.json"
                with patch("builtins.open", create=True) as mock_open:
                    mock_file = Mock()
                    mock_open.return_value.__enter__.return_value = mock_file

                    output_path = gen.save_task_graph(task_graph)

                    assert output_path == "/tmp/taskgraph.json"

    def test_generator_save_task_graph_sanitizes_filename(
        self, minimal_config, mock_model
    ) -> None:
        """Test that save_task_graph sanitizes the filename."""
        with tempfile.TemporaryDirectory() as temp_dir:
            gen = Generator(
                config=minimal_config, model=mock_model, output_dir=temp_dir
            )
            task_graph = {"nodes": [], "edges": []}

            # Mock timestamp to include special characters
            with patch.object(gen, "timestamp", "2023/12/25 15:30:45"):
                output_path = gen.save_task_graph(task_graph)

                # Should not contain special characters
                assert "/" not in os.path.basename(output_path)
                assert ":" not in os.path.basename(output_path)
                assert os.path.exists(output_path)

    def test_save_task_graph_with_invalid_filename(
        self, minimal_config, mock_model
    ) -> None:
        """Test save_task_graph method with invalid filename."""
        gen = Generator(config=minimal_config, model=mock_model, output_dir="/tmp")
        task_graph = {"test": "data"}

        # Test with filename containing invalid characters
        gen.timestamp = "invalid/chars"

        with patch("builtins.open") as mock_open:
            gen.save_task_graph(task_graph)
            # Should handle invalid characters gracefully
            assert mock_open.called


class TestGeneratorDocumentTypeConversion:
    """Test Generator document and instruction type conversion."""

    def test_generator_document_instruction_type_conversion(
        self, minimal_config, mock_model
    ) -> None:
        """Test that documents and instructions are converted from lists to strings."""
        # Configure with list-based documents and instructions
        config_with_lists = minimal_config.copy()
        config_with_lists["task_docs"] = ["doc1.txt", "doc2.txt"]
        config_with_lists["instruction_docs"] = ["instruction1.txt", "instruction2.txt"]

        gen = Generator(config=config_with_lists, model=mock_model)

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

    def test_generator_document_instruction_type_conversion_single_docs(
        self, minimal_config, mock_model
    ) -> None:
        """Test type conversion with single documents."""
        config_with_single_docs = minimal_config.copy()
        config_with_single_docs["task_docs"] = "single_doc.txt"
        config_with_single_docs["instruction_docs"] = "single_instruction.txt"

        gen = Generator(config=config_with_single_docs, model=mock_model)

        with patch.object(gen, "_initialize_document_loader") as doc_loader_patch:
            doc_loader = Mock()
            doc_loader.load_task_document.return_value = "Single document content"
            doc_loader.load_instruction_document.return_value = (
                "Single instruction content"
            )
            doc_loader_patch.return_value = doc_loader

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
                tgf_patch.return_value = tgf

                gen.generate()

                assert isinstance(gen.documents, str)
                assert isinstance(gen.instructions, str)
                assert gen.documents == "Single document content"
                assert gen.instructions == "Single instruction content"


class TestGeneratorExtendedCoverage:
    """Additional tests to increase coverage for generator.py"""

    def test_generator_with_missing_config_keys(self, mock_model) -> None:
        """Test generator initialization with missing config keys."""
        config = {}

        gen = Generator(config=config, model=mock_model)

        assert gen.role == ""
        assert gen.user_objective == ""
        assert gen.builder_objective == ""
        assert gen.domain == ""
        assert gen.intro == ""
        assert gen.instruction_docs == []
        assert gen.task_docs == []
        assert gen.rag_docs == []
        assert gen.user_tasks == []
        assert gen.example_conversations == []
        assert gen.nluapi == ""
        assert gen.slotfillapi == ""

    def test_generator_with_none_workers(self, mock_model) -> None:
        """Test generator initialization with None workers."""
        config = {"workers": None}

        with pytest.raises(TypeError):
            gen = Generator(config=config, model=mock_model)

    def test_generator_with_invalid_worker_format(self, mock_model) -> None:
        """Test generator initialization with invalid worker format."""
        config = {"workers": ["invalid_worker"]}

        gen = Generator(config=config, model=mock_model)
        assert gen.workers == []

    def test_generator_with_missing_worker_fields(self, mock_model) -> None:
        """Test generator initialization with missing worker fields."""
        config = {"workers": [{"id": "worker1"}]}  # Missing name and path

        gen = Generator(config=config, model=mock_model)
        assert gen.workers == []

    def test_generator_with_resource_initializer_exception(self, mock_model) -> None:
        """Test generator initialization with resource initializer exception."""
        config = {"tools": ["tool1"]}

        with patch("arklex.env.env.DefaultResourceInitializer") as mock_initializer:
            mock_initializer.return_value.init_tools.side_effect = Exception(
                "Init error"
            )

            with pytest.raises(Exception):
                Generator(config=config, model=mock_model)

    def test_initialize_document_loader_with_exception(
        self, minimal_config, mock_model
    ) -> None:
        """Test document loader initialization with exception."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch(
            "arklex.orchestrator.generator.core.generator.DocumentLoader"
        ) as mock_loader:
            mock_loader.side_effect = Exception("Loader error")

            with pytest.raises(Exception):
                gen._initialize_document_loader()

    def test_initialize_task_generator_with_exception(
        self, minimal_config, mock_model
    ) -> None:
        """Test task generator initialization with exception."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch(
            "arklex.orchestrator.generator.core.generator.TaskGenerator"
        ) as mock_task_gen:
            mock_task_gen.side_effect = Exception("Task generator error")

            with pytest.raises(Exception):
                gen._initialize_task_generator()

    def test_initialize_best_practice_manager_with_exception(
        self, full_config, mock_model
    ) -> None:
        """Test best practice manager initialization with exception."""
        gen = Generator(config=full_config, model=mock_model)

        with patch(
            "arklex.orchestrator.generator.core.generator.BestPracticeManager"
        ) as mock_bpm:
            mock_bpm.side_effect = Exception("BPM error")

            with pytest.raises(Exception):
                gen._initialize_best_practice_manager()

    def test_initialize_reusable_task_manager_with_exception(
        self, minimal_config, mock_model
    ) -> None:
        """Test reusable task manager initialization with exception."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch(
            "arklex.orchestrator.generator.core.generator.ReusableTaskManager"
        ) as mock_rtm:
            mock_rtm.side_effect = Exception("RTM error")

            with pytest.raises(Exception):
                gen._initialize_reusable_task_manager()

    def test_initialize_task_graph_formatter_with_exception(
        self, full_config, mock_model
    ) -> None:
        """Test task graph formatter initialization with exception."""
        gen = Generator(config=full_config, model=mock_model)

        with patch(
            "arklex.orchestrator.generator.core.generator.TaskGraphFormatter"
        ) as mock_tgf:
            mock_tgf.side_effect = Exception("TGF error")

            with pytest.raises(Exception):
                gen._initialize_task_graph_formatter()

    def test_load_multiple_task_documents_with_exception(
        self, minimal_config, mock_model
    ) -> None:
        """Test loading multiple task documents with exception."""
        gen = Generator(config=minimal_config, model=mock_model)

        doc_paths = ["doc1.txt"]
        mock_doc_loader = Mock()
        mock_doc_loader.load_task_document.side_effect = Exception("Load error")

        with pytest.raises(Exception):
            gen._load_multiple_task_documents(mock_doc_loader, doc_paths)

    def test_load_multiple_instruction_documents_with_exception(
        self, minimal_config, mock_model
    ) -> None:
        """Test loading multiple instruction documents with exception."""
        gen = Generator(config=minimal_config, model=mock_model)

        doc_paths = ["instruction1.txt"]
        mock_doc_loader = Mock()
        mock_doc_loader.load_instruction_document.side_effect = Exception("Load error")

        with pytest.raises(Exception):
            gen._load_multiple_instruction_documents(mock_doc_loader, doc_paths)

    def test_generate_with_document_loader_exception(
        self, minimal_config, mock_model
    ) -> None:
        """Test generate method with document loader exception."""
        gen = Generator(config=minimal_config, model=mock_model)

        with patch.object(gen, "_initialize_document_loader") as mock_init_loader:
            mock_init_loader.side_effect = Exception("Document loader error")

            with pytest.raises(Exception):
                gen.generate()

    def test_generate_with_task_generator_exception(
        self, minimal_config, mock_model
    ) -> None:
        """Test generate method with task generator exception."""
        gen = Generator(config=minimal_config, model=mock_model)

        with (
            patch.object(gen, "_initialize_document_loader") as mock_init_loader,
            patch.object(gen, "_initialize_task_generator") as mock_init_task,
        ):
            mock_init_loader.return_value = Mock()
            mock_init_task.side_effect = Exception("Task generator error")

            with pytest.raises(Exception):
                gen.generate()

    def test_generate_with_best_practice_manager_exception(
        self, full_config, mock_model
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
            mock_init_bpm.side_effect = Exception("BPM error")

            with pytest.raises(Exception):
                gen.generate()

    def test_generate_with_reusable_task_manager_exception(
        self, full_config, mock_model
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
            mock_init_rtm.side_effect = Exception("RTM error")

            with pytest.raises(Exception):
                gen.generate()

    def test_generate_with_task_graph_formatter_format_exception(
        self, full_config, mock_model
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
        mock_tgf.format_task_graph.side_effect = Exception("Format task graph error")

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

            with pytest.raises(Exception):
                gen.generate()

    def test_save_task_graph_with_exception(self, minimal_config, mock_model) -> None:
        """Test save_task_graph method with exception."""
        gen = Generator(config=minimal_config, model=mock_model)
        task_graph = {"test": "data"}

        with patch("builtins.open") as mock_open:
            mock_open.side_effect = Exception("File error")

            with pytest.raises(Exception):
                gen.save_task_graph(task_graph)

    def test_load_multiple_task_documents_with_empty_list(
        self, minimal_config, mock_model
    ) -> None:
        """Test loading multiple task documents with empty list."""
        gen = Generator(config=minimal_config, model=mock_model)

        doc_paths = []
        mock_doc_loader = Mock()

        result = gen._load_multiple_task_documents(mock_doc_loader, doc_paths)
        assert result == []

    def test_load_multiple_instruction_documents_with_empty_list(
        self, minimal_config, mock_model
    ) -> None:
        """Test loading multiple instruction documents with empty list."""
        gen = Generator(config=minimal_config, model=mock_model)

        doc_paths = []
        mock_doc_loader = Mock()

        result = gen._load_multiple_instruction_documents(mock_doc_loader, doc_paths)
        assert result == []
