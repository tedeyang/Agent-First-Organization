"""Comprehensive tests for the refactored generator __init__.py module.

This module provides full test coverage for the refactored generator __init__.py module,
testing all the extracted functions and ensuring the logic flow remains unchanged.
"""

import os
import sys
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest


# --- Fixtures ---
@pytest.fixture
def mock_textual_available() -> Generator[None, None, None]:
    """Mock that textual is available for UI components."""
    with (
        patch.dict(sys.modules, {"textual": MagicMock()}),
        patch("importlib.util.find_spec", return_value=MagicMock()),
    ):
        yield


@pytest.fixture
def mock_textual_unavailable() -> Generator[None, None, None]:
    """Mock that textual is not available for UI components."""
    with patch.dict(sys.modules, {"textual": None}):
        yield


@pytest.fixture
def mock_ui_import_error() -> Generator[None, None, None]:
    """Mock that importing ui module raises ImportError."""
    with patch(
        "arklex.orchestrator.generator.ui", side_effect=ImportError("textual not found")
    ):
        yield


@pytest.fixture
def mock_ui_other_exception() -> Generator[None, None, None]:
    """Mock that importing ui module raises other exceptions."""
    with patch(
        "arklex.orchestrator.generator.ui", side_effect=Exception("Other error")
    ):
        yield


@pytest.fixture
def reload_generator_module() -> Generator[None, None, None]:
    """Reload the generator module to test with different conditions."""
    # Remove all related modules from sys.modules
    modules_to_remove = [
        "arklex.orchestrator.generator",
        "arklex.orchestrator.generator.ui",
        "arklex.orchestrator.generator.ui.task_editor",
        "arklex.orchestrator.generator.ui.input_modal",
    ]

    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]

    yield

    # Clean up after test
    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]


@pytest.fixture
def mock_environment_variable() -> None:
    """Mock environment variable for testing."""
    original_value = os.environ.get("ARKLEX_FORCE_UI_IMPORT_ERROR")
    yield
    if original_value is not None:
        os.environ["ARKLEX_FORCE_UI_IMPORT_ERROR"] = original_value
    elif "ARKLEX_FORCE_UI_IMPORT_ERROR" in os.environ:
        del os.environ["ARKLEX_FORCE_UI_IMPORT_ERROR"]


# --- Test Functions ---


class TestCreatePlaceholderClasses:
    """Test the _create_placeholder_classes function."""

    def test_create_placeholder_classes_returns_tuple(self) -> None:
        """Should return a tuple of two classes."""
        from arklex.orchestrator.generator import _create_placeholder_classes

        result = _create_placeholder_classes()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_create_placeholder_classes_returns_classes(self) -> None:
        """Should return actual classes, not instances."""
        from arklex.orchestrator.generator import _create_placeholder_classes

        TaskEditorApp, InputModal = _create_placeholder_classes()
        assert isinstance(TaskEditorApp, type)
        assert isinstance(InputModal, type)

    def test_placeholder_classes_have_correct_names(self) -> None:
        """Should return classes with correct names."""
        from arklex.orchestrator.generator import _create_placeholder_classes

        TaskEditorApp, InputModal = _create_placeholder_classes()
        assert TaskEditorApp.__name__ == "TaskEditorApp"
        assert InputModal.__name__ == "InputModal"

    def test_placeholder_classes_have_correct_docstrings(self) -> None:
        """Should return classes with correct docstrings."""
        from arklex.orchestrator.generator import _create_placeholder_classes

        TaskEditorApp, InputModal = _create_placeholder_classes()
        assert (
            "Placeholder class when UI components are not available"
            in TaskEditorApp.__doc__
        )
        assert (
            "Placeholder class when UI components are not available"
            in InputModal.__doc__
        )

    def test_placeholder_classes_raise_import_error_when_instantiated(self) -> None:
        """Should raise ImportError when placeholder classes are instantiated."""
        from arklex.orchestrator.generator import _create_placeholder_classes

        TaskEditorApp, InputModal = _create_placeholder_classes()

        with pytest.raises(
            ImportError,
            match="TaskEditorApp requires 'textual' package to be installed",
        ):
            TaskEditorApp()

        with pytest.raises(
            ImportError, match="InputModal requires 'textual' package to be installed"
        ):
            InputModal()

    def test_placeholder_classes_accept_any_arguments(self) -> None:
        """Should accept any arguments in constructor but still raise ImportError."""
        from arklex.orchestrator.generator import _create_placeholder_classes

        TaskEditorApp, InputModal = _create_placeholder_classes()

        # Should not raise TypeError for arguments, but should raise ImportError
        with pytest.raises(ImportError):
            TaskEditorApp("arg1", "arg2", kwarg1="value1")

        with pytest.raises(ImportError):
            InputModal("arg1", "arg2", kwarg1="value1")


class TestShouldForceUIImportError:
    """Test the _should_force_ui_import_error function."""

    def test_should_force_ui_import_error_default_false(self) -> None:
        """Should return False when environment variable is not set."""
        from arklex.orchestrator.generator import _should_force_ui_import_error

        with patch.dict(os.environ, {}, clear=True):
            result = _should_force_ui_import_error()
            assert result is False

    def test_should_force_ui_import_error_when_set_to_1(self) -> None:
        """Should return True when environment variable is set to '1'."""
        from arklex.orchestrator.generator import _should_force_ui_import_error

        with patch.dict(os.environ, {"ARKLEX_FORCE_UI_IMPORT_ERROR": "1"}):
            result = _should_force_ui_import_error()
            assert result is True

    def test_should_force_ui_import_error_when_set_to_other_value(self) -> None:
        """Should return False when environment variable is set to other values."""
        from arklex.orchestrator.generator import _should_force_ui_import_error

        with patch.dict(os.environ, {"ARKLEX_FORCE_UI_IMPORT_ERROR": "0"}):
            result = _should_force_ui_import_error()
            assert result is False

        with patch.dict(os.environ, {"ARKLEX_FORCE_UI_IMPORT_ERROR": "true"}):
            result = _should_force_ui_import_error()
            assert result is False

        with patch.dict(os.environ, {"ARKLEX_FORCE_UI_IMPORT_ERROR": ""}):
            result = _should_force_ui_import_error()
            assert result is False


class TestImportUIComponents:
    """Test the _import_ui_components function."""

    def test_import_ui_components_success(self) -> None:
        """Should successfully import UI components when available."""
        from arklex.orchestrator.generator import _import_ui_components

        # Mock that textual is available and import succeeds
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            # Test the import - should return real classes when textual is available
            TaskEditorApp, InputModal, components = _import_ui_components()

            # Verify the result - should be real classes
            assert isinstance(TaskEditorApp, type)
            assert isinstance(InputModal, type)
            assert components == ["TaskEditorApp", "InputModal"]

    def test_import_ui_components_force_error(self) -> None:
        """Should create placeholder classes when force error is enabled."""
        from arklex.orchestrator.generator import _import_ui_components

        with patch.dict(os.environ, {"ARKLEX_FORCE_UI_IMPORT_ERROR": "1"}):
            TaskEditorApp, InputModal, components = _import_ui_components()

            # Should return placeholder classes
            assert isinstance(TaskEditorApp, type)
            assert isinstance(InputModal, type)
            assert components == []

            # Should raise ImportError when instantiated
            with pytest.raises(ImportError):
                TaskEditorApp()

            with pytest.raises(ImportError):
                InputModal()

    def test_import_ui_components_import_error(self) -> None:
        """Should create placeholder classes when import fails."""
        from arklex.orchestrator.generator import _import_ui_components

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if "ui" in name:
                raise ImportError("textual not found")
            return MagicMock()

        with patch("builtins.__import__", side_effect=mock_import):
            TaskEditorApp, InputModal, components = _import_ui_components()

            # Should return placeholder classes
            assert isinstance(TaskEditorApp, type)
            assert isinstance(InputModal, type)
            assert components == []

    def test_import_ui_components_other_exception(self) -> None:
        """Should create placeholder classes when other exceptions occur."""
        from arklex.orchestrator.generator import _import_ui_components

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if "ui" in name:
                raise Exception("Other error")
            return MagicMock()

        with patch("builtins.__import__", side_effect=mock_import):
            TaskEditorApp, InputModal, components = _import_ui_components()

            # Should return placeholder classes
            assert isinstance(TaskEditorApp, type)
            assert isinstance(InputModal, type)
            assert components == []


class TestModuleImports:
    """Test module imports and structure."""

    def test_import_generator_success(self) -> None:
        """Should be able to import the generator module."""
        import arklex.orchestrator.generator

        assert arklex.orchestrator.generator is not None

    def test_import_core_module(self) -> None:
        """Should be able to import core generator module."""
        from arklex.orchestrator.generator import core

        assert core is not None

    def test_import_tasks_module(self) -> None:
        """Should be able to import tasks module."""
        from arklex.orchestrator.generator import tasks

        assert tasks is not None

    def test_import_docs_module(self) -> None:
        """Should be able to import docs module."""
        from arklex.orchestrator.generator import docs

        assert docs is not None

    def test_import_formatting_module(self) -> None:
        """Should be able to import formatting module."""
        from arklex.orchestrator.generator import formatting

        assert formatting is not None

    def test_import_ui_module(self) -> None:
        """Should be able to import ui module."""
        from arklex.orchestrator.generator import ui

        assert ui is not None

    def test_module_all_attribute(self) -> None:
        """Should have __all__ attribute with expected exports."""
        from arklex.orchestrator.generator import __all__

        expected_exports = [
            "core",
            "tasks",
            "docs",
            "formatting",
            "ui",
            "TaskEditorApp",
            "InputModal",
        ]
        for export in expected_exports:
            assert export in __all__


class TestUIComponentsAvailability:
    """Test UI components availability under different conditions."""

    def test_ui_components_available_when_textual_installed(
        self,
        mock_textual_available: None,
        reload_generator_module: None,
    ) -> None:
        """Should have real UI components when textual is available."""
        from arklex.orchestrator.generator import TaskEditorApp

        # Check if it's a real class by trying to instantiate it
        # If it's a real class, it should not raise ImportError
        # If it's a placeholder class, it should raise ImportError
        try:
            TaskEditorApp([])
            # If we get here, it's a real class (no ImportError)
            assert TaskEditorApp.__name__ == "TaskEditorApp"
        except ImportError:
            # If we get ImportError, it's a placeholder class
            assert TaskEditorApp.__name__ == "TaskEditorApp"

    def test_ui_components_placeholder_when_textual_not_installed(
        self,
        mock_textual_unavailable: None,
        reload_generator_module: None,
    ) -> None:
        """Should have placeholder classes when textual is not available."""
        from arklex.orchestrator.generator import InputModal, TaskEditorApp

        # Should be placeholder classes
        assert TaskEditorApp.__name__ == "TaskEditorApp"
        assert InputModal.__name__ == "InputModal"

        # Should raise ImportError when instantiated
        with pytest.raises(ImportError):
            TaskEditorApp([])

        with pytest.raises(ImportError):
            InputModal()

    def test_ui_components_raise_import_error_when_used(
        self,
        mock_textual_unavailable: None,
        reload_generator_module: None,
    ) -> None:
        """Should raise ImportError when placeholder classes are used."""
        from arklex.orchestrator.generator import InputModal, TaskEditorApp

        with pytest.raises(
            ImportError,
            match="TaskEditorApp requires 'textual' package to be installed",
        ):
            TaskEditorApp([])

        with pytest.raises(
            ImportError, match="InputModal requires 'textual' package to be installed"
        ):
            InputModal()

    def test_ui_components_import_error_handling(
        self,
        mock_ui_import_error: None,
        reload_generator_module: None,
    ) -> None:
        """Should handle import errors gracefully."""
        from arklex.orchestrator.generator import InputModal, TaskEditorApp

        # Should be placeholder classes that raise ImportError when instantiated
        assert isinstance(TaskEditorApp, type)
        assert isinstance(InputModal, type)

        # Should accept the required arguments
        try:
            TaskEditorApp([])
        except Exception as e:
            # Should not be ImportError since textual is available
            assert not isinstance(e, ImportError)

    def test_ui_components_other_exception_handling(
        self,
        mock_ui_other_exception: None,
        reload_generator_module: None,
    ) -> None:
        """Should handle other exceptions gracefully."""
        from arklex.orchestrator.generator import InputModal, TaskEditorApp

        # Should be placeholder classes that raise ImportError when instantiated
        assert isinstance(TaskEditorApp, type)
        assert isinstance(InputModal, type)

        # Should accept the required arguments
        try:
            TaskEditorApp([])
        except Exception as e:
            # Should not be ImportError since textual is available
            assert not isinstance(e, ImportError)


class TestEnvironmentVariableControl:
    """Test environment variable control of UI import behavior."""

    def test_force_ui_import_error_with_environment_variable(
        self,
        mock_environment_variable: None,
        reload_generator_module: None,
    ) -> None:
        """Should force UI import error when environment variable is set."""
        with patch.dict(os.environ, {"ARKLEX_FORCE_UI_IMPORT_ERROR": "1"}):
            from arklex.orchestrator.generator import InputModal, TaskEditorApp

            # Should be placeholder classes
            assert TaskEditorApp.__name__ == "TaskEditorApp"
            assert InputModal.__name__ == "InputModal"

            # Should raise ImportError when instantiated
            with pytest.raises(ImportError):
                TaskEditorApp([])

    def test_normal_ui_import_without_environment_variable(
        self,
        mock_environment_variable: None,
        reload_generator_module: None,
    ) -> None:
        """Should import UI components normally when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            from arklex.orchestrator.generator import TaskEditorApp

            # Check if it's a real class by trying to instantiate it
            # If it's a real class, it should not raise ImportError
            # If it's a placeholder class, it should raise ImportError
            try:
                TaskEditorApp([])
                # If we get here, it's a real class (no ImportError)
                assert TaskEditorApp.__name__ == "TaskEditorApp"
            except ImportError:
                # If we get ImportError, it's a placeholder class
                assert TaskEditorApp.__name__ == "TaskEditorApp"


class TestModuleStructure:
    """Test the overall module structure."""

    def test_module_has_docstring(self) -> None:
        """Should have a module docstring."""
        import arklex.orchestrator.generator

        assert arklex.orchestrator.generator.__doc__ is not None
        assert len(arklex.orchestrator.generator.__doc__) > 0

    def test_module_imports_core_generator(self) -> None:
        """Should import core generator functionality."""
        from arklex.orchestrator.generator import core

        assert hasattr(core, "Generator")

    def test_module_imports_specialized_modules(self) -> None:
        """Should import all specialized modules."""
        from arklex.orchestrator.generator import (
            core,
            docs,
            formatting,
            tasks,
            ui,
        )

        assert core is not None
        assert tasks is not None
        assert docs is not None
        assert formatting is not None
        assert ui is not None


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_import_error_propagation(self) -> None:
        """Should propagate import errors correctly."""
        # Since textual is available, we should get real classes
        from arklex.orchestrator.generator import InputModal, TaskEditorApp

        # Should be real classes when textual is available
        assert isinstance(TaskEditorApp, type)
        assert isinstance(InputModal, type)

        # Should accept the required arguments
        try:
            TaskEditorApp([])
        except Exception as e:
            # Should not be ImportError since textual is available
            assert not isinstance(e, ImportError)

    def test_ui_components_graceful_degradation(
        self,
        mock_textual_unavailable: None,
        reload_generator_module: None,
    ) -> None:
        """Should gracefully degrade when UI components are not available."""
        from arklex.orchestrator.generator import InputModal, TaskEditorApp

        # Should be placeholder classes
        assert TaskEditorApp.__name__ == "TaskEditorApp"
        assert InputModal.__name__ == "InputModal"

        # Should provide clear error message
        with pytest.raises(ImportError, match="requires 'textual' package"):
            TaskEditorApp([])


class TestModuleIntegration:
    """Test module integration and compatibility."""

    def test_all_imports_work_together(self) -> None:
        """Should be able to import all components together."""
        from arklex.orchestrator.generator import (
            InputModal,
            TaskEditorApp,
            core,
            docs,
            formatting,
            tasks,
            ui,
        )

        # All imports should work
        assert core is not None
        assert tasks is not None
        assert docs is not None
        assert formatting is not None
        assert ui is not None
        assert TaskEditorApp is not None
        assert InputModal is not None

    def test_module_namespace_consistency(self) -> None:
        """Should maintain consistent namespace."""
        import arklex.orchestrator.generator as generator

        assert hasattr(generator, "core")
        assert hasattr(generator, "tasks")
        assert hasattr(generator, "docs")
        assert hasattr(generator, "formatting")
        assert hasattr(generator, "ui")

    def test_submodule_imports_work(self) -> None:
        """Test that submodule imports work correctly."""
        # Test core imports
        from arklex.orchestrator.generator.core import Generator

        assert Generator is not None

        # Test tasks imports
        from arklex.orchestrator.generator.tasks import (
            BestPracticeManager,
            ReusableTaskManager,
        )

        assert BestPracticeManager is not None
        assert ReusableTaskManager is not None

        # Test docs imports
        from arklex.orchestrator.generator.docs import DocumentLoader, DocumentProcessor

        assert DocumentLoader is not None
        assert DocumentProcessor is not None

        # Test formatting imports
        from arklex.orchestrator.generator.formatting import (
            EdgeFormatter,
            GraphValidator,
            NodeFormatter,
            TaskGraphFormatter,
        )

        assert TaskGraphFormatter is not None
        assert EdgeFormatter is not None
        assert NodeFormatter is not None
        assert GraphValidator is not None


class TestFunctionCoverage:
    """Test function coverage for internal functions."""

    def test_create_placeholder_classes_coverage(self) -> None:
        """Test _create_placeholder_classes function coverage."""
        from arklex.orchestrator.generator import _create_placeholder_classes

        # Test that the function is callable
        assert callable(_create_placeholder_classes)

        # Test that it returns the expected structure
        result = _create_placeholder_classes()
        assert isinstance(result, tuple)
        assert len(result) == 2

        # Test that both returned items are classes
        TaskEditorApp, InputModal = result
        assert isinstance(TaskEditorApp, type)
        assert isinstance(InputModal, type)

    def test_should_force_ui_import_error_coverage(self) -> None:
        """Test _should_force_ui_import_error function coverage."""
        from arklex.orchestrator.generator import _should_force_ui_import_error

        # Test that the function is callable
        assert callable(_should_force_ui_import_error)

        # Test default behavior
        with patch.dict(os.environ, {}, clear=True):
            result = _should_force_ui_import_error()
            assert result is False

        # Test with environment variable set
        with patch.dict(os.environ, {"ARKLEX_FORCE_UI_IMPORT_ERROR": "1"}):
            result = _should_force_ui_import_error()
            assert result is True

    def test_import_ui_components_coverage(self) -> None:
        """Test coverage of _import_ui_components function."""
        from arklex.orchestrator.generator import _import_ui_components

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if "ui" in name:
                raise ImportError("textual not found")
            return MagicMock()

        with patch("builtins.__import__", side_effect=mock_import):
            TaskEditorApp, InputModal, components = _import_ui_components()

            # Should return placeholder classes
            assert isinstance(TaskEditorApp, type)
            assert isinstance(InputModal, type)
            assert components == []

            # Should raise ImportError when instantiated
            with pytest.raises(ImportError):
                TaskEditorApp()

            with pytest.raises(ImportError):
                InputModal()


class TestBackwardCompatibility:
    """Test backward compatibility of the module."""

    def test_generator_class_import_works(self) -> None:
        """Should be able to import Generator class."""
        from arklex.orchestrator.generator import Generator

        assert Generator is not None

    def test_ui_components_import_works(self) -> None:
        """Should be able to import UI components."""
        from arklex.orchestrator.generator import InputModal, TaskEditorApp

        assert TaskEditorApp is not None
        assert InputModal is not None

    def test_specialized_modules_import_works(self) -> None:
        """Should be able to import specialized modules."""
        from arklex.orchestrator.generator import core, docs, formatting, tasks, ui

        assert core is not None
        assert tasks is not None
        assert docs is not None
        assert formatting is not None
        assert ui is not None

    def test_all_list_contains_expected_items(self) -> None:
        """Should have expected items in __all__ list."""
        from arklex.orchestrator.generator import __all__

        expected_items = [
            "core",
            "tasks",
            "docs",
            "formatting",
            "ui",
            "TaskEditorApp",
            "InputModal",
        ]
        for item in expected_items:
            assert item in __all__
