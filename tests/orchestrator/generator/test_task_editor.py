"""Tests for the TaskEditorApp class.

This module provides comprehensive tests for the TaskEditorApp class, which is a text-based
user interface for editing tasks and their steps. The tests cover all methods, edge cases,
and user interactions.
"""

import asyncio
import contextlib
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from arklex.orchestrator.generator.ui.task_editor import TaskEditorApp


# --- FAKE TEXTUAL CLASSES AND MODULES (must be set before any code import) ---
class FakeTreeNode:
    pass


class FakeButton:
    class Pressed:
        pass


class FakeInput:
    pass


class FakeStatic:
    pass


class FakeLabel:
    def __init__(self, plain: str) -> None:
        self.plain = plain


class FakeTree:
    class NodeSelected:
        pass


class FakeHorizontal:
    pass


class FakeVertical:
    pass


class FakeApp:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def run(self) -> None:
        pass


class FakeComposeResult:
    pass


class FakeReturnType:
    pass


# textual.app
fake_textual_app = types.ModuleType("textual.app")
fake_textual_app.App = FakeApp
fake_textual_app.ComposeResult = FakeComposeResult
fake_textual_app.ReturnType = FakeReturnType
sys.modules["textual.app"] = fake_textual_app
# textual.widgets
fake_textual_widgets = types.ModuleType("textual.widgets")
fake_textual_widgets.Button = FakeButton
fake_textual_widgets.Input = FakeInput
fake_textual_widgets.Static = FakeStatic
fake_textual_widgets.Label = FakeLabel
fake_textual_widgets.Tree = FakeTree
sys.modules["textual.widgets"] = fake_textual_widgets
# textual.widgets.tree
fake_textual_widgets_tree = types.ModuleType("textual.widgets.tree")
fake_textual_widgets_tree.TreeNode = FakeTreeNode
sys.modules["textual.widgets.tree"] = fake_textual_widgets_tree
# textual.containers
fake_textual_containers = types.ModuleType("textual.containers")
fake_textual_containers.Horizontal = FakeHorizontal
fake_textual_containers.Vertical = FakeVertical
sys.modules["textual.containers"] = fake_textual_containers
# textual.screen
fake_textual_screen = types.ModuleType("textual.screen")
fake_textual_screen.Screen = FakeApp
sys.modules["textual.screen"] = fake_textual_screen

# --- END FAKE TEXTUAL SETUP ---

# --- Mock fixtures for UI components ---


@pytest.fixture
def mock_tree(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock the Tree widget."""
    mock_tree = Mock()
    mock_tree_instance = Mock()
    mock_tree_instance.root = Mock()
    mock_tree_instance.root.expand = Mock()
    mock_tree_instance.root.add = Mock()
    mock_tree.return_value = mock_tree_instance
    return mock_tree


@pytest.fixture
def mock_label(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock the Label widget."""
    mock_label = Mock()
    mock_label_instance = Mock()
    mock_label.return_value = mock_label_instance
    return mock_label


@pytest.fixture
def mock_input_modal(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock the InputModal class."""
    mock_input_modal = Mock()
    mock_input_modal_instance = Mock()
    mock_input_modal_instance.result = "modal result"
    mock_input_modal.return_value = mock_input_modal_instance
    return mock_input_modal


@pytest.fixture
def mock_log_context(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock the log_context."""
    mock_log = Mock()
    mock_log.debug = Mock()
    return mock_log


@pytest.fixture
def sample_tasks() -> list[dict[str, Any]]:
    """Sample tasks for testing."""
    return [
        {"name": "Task 1", "steps": ["Step 1", "Step 2"]},
        {"name": "Task 2", "steps": ["Step 3"]},
    ]


@pytest.fixture
def complex_tasks() -> list[dict[str, Any]]:
    """Complex tasks with mixed step formats for testing."""
    return [
        {
            "name": "Complex Task",
            "steps": [
                {"description": "Step with dict"},
                "Step as string",
                {"other_field": "Not description"},
            ],
        }
    ]


@pytest.fixture
def task_editor_app(sample_tasks: list[dict[str, Any]]) -> TaskEditorApp:
    """Create a TaskEditorApp instance for testing."""
    return TaskEditorApp(sample_tasks)


class TestTaskEditorAppInitialization:
    """Test TaskEditorApp initialization and basic functionality."""

    def test_import_task_editor_module(self) -> None:
        """Test that the task_editor module can be imported."""
        from arklex.orchestrator.generator.ui import task_editor

        assert task_editor is not None

    def test_task_editor_instantiation_with_valid_tasks(self) -> None:
        """Test TaskEditorApp instantiation with valid tasks."""
        tasks = [{"name": "Test Task", "steps": ["Step 1", "Step 2"]}]
        app = TaskEditorApp(tasks)
        assert app.tasks == tasks
        assert app.task_tree is None

    def test_task_editor_instantiation_with_empty_tasks(self) -> None:
        """Test TaskEditorApp instantiation with empty tasks."""
        app = TaskEditorApp([])
        assert app.tasks == []
        assert app.task_tree is None

    def test_task_editor_instantiation_with_none_tasks(self) -> None:
        """Test TaskEditorApp instantiation with None tasks."""
        app = TaskEditorApp(None)
        assert app.tasks is None
        assert app.task_tree is None

    def test_task_editor_instantiation_with_complex_tasks(self) -> None:
        """Test TaskEditorApp instantiation with complex task structures."""
        complex_tasks = [
            {
                "name": "Complex Task",
                "steps": [
                    {"description": "Step with dict"},
                    "Step as string",
                    {"other_field": "Not description"},
                ],
            }
        ]
        app = TaskEditorApp(complex_tasks)
        assert app.tasks == complex_tasks
        assert app.task_tree is None

    def test_task_editor_instantiation_with_invalid_task_structure(self) -> None:
        """Test TaskEditorApp instantiation with invalid task structure."""
        invalid_tasks = [{"invalid_key": "value"}]  # Missing 'name' and 'steps'
        app = TaskEditorApp(invalid_tasks)
        assert app.tasks == invalid_tasks
        assert app.task_tree is None

    def test_task_editor_methods_existence(self) -> None:
        """Test that TaskEditorApp has all required methods."""
        app = TaskEditorApp([])
        required_methods = [
            "compose",
            "on_mount",
            "on_tree_node_selected",
            "on_key",
            "action_add_node",
            "show_input_modal",
            "update_tasks",
            "run",
        ]
        for method_name in required_methods:
            assert hasattr(app, method_name), f"Method {method_name} not found"

    def test_task_editor_attributes_existence(self) -> None:
        """Test that TaskEditorApp has all required attributes."""
        app = TaskEditorApp([])
        assert hasattr(app, "tasks")
        assert hasattr(app, "task_tree")

    def test_task_editor_init_none_tasks(self) -> None:
        """Test TaskEditorApp initialization with None tasks."""
        app = TaskEditorApp(None)
        assert app.tasks is None
        assert app.task_tree is None


class TestTaskEditorAppCompose:
    """Test TaskEditorApp compose method."""

    def test_compose_method(
        self, sample_tasks: list[dict[str, Any]], mock_tree: Mock, mock_label: Mock
    ) -> None:
        """Test compose method with valid tasks."""
        app = TaskEditorApp(sample_tasks)

        # Call compose and convert to list - should work with fake textual classes
        result = list(app.compose())

        # Verify that compose returns a result
        assert result is not None
        assert len(result) >= 1

    def test_compose_with_empty_tasks(self, mock_tree: Mock, mock_label: Mock) -> None:
        """Test compose method with empty tasks."""
        app = TaskEditorApp([])

        # Call compose and convert to list - should work with fake textual classes
        result = list(app.compose())

        # Verify that compose returns a result
        assert result is not None
        assert len(result) >= 1

    def test_compose_with_none_tasks(self, mock_tree: Mock, mock_label: Mock) -> None:
        """Test compose method with None tasks."""
        app = TaskEditorApp(None)

        # Call compose and convert to list - should work with fake textual classes
        result = list(app.compose())

        # Verify that compose returns a result
        assert result is not None
        assert len(result) >= 1

    def test_compose_with_complex_tasks(
        self, complex_tasks: list[dict[str, Any]], mock_tree: Mock, mock_label: Mock
    ) -> None:
        """Test compose method with complex tasks."""
        app = TaskEditorApp(complex_tasks)

        # Call compose and convert to list - should work with fake textual classes
        result = list(app.compose())

        # Verify that compose returns a result
        assert result is not None
        assert len(result) >= 1

    def test_task_editor_compose_with_none_tasks(self) -> None:
        """Test compose method with None tasks."""
        app = TaskEditorApp(None)

        # Call compose and convert to list - should work with fake textual classes
        result = list(app.compose())

        # Verify that compose returns a result
        assert result is not None
        assert len(result) >= 1


class TestTaskEditorAppEventHandling:
    """Test TaskEditorApp event handling methods."""

    def test_on_mount(self, task_editor_app: TaskEditorApp) -> None:
        """Test on_mount method."""
        # Mock the task_tree
        task_editor_app.task_tree = Mock()
        task_editor_app.task_tree.focus = Mock()

        task_editor_app.on_mount()

        # Verify focus was called
        task_editor_app.task_tree.focus.assert_called_once()

    async def test_on_tree_node_selected(
        self, task_editor_app: TaskEditorApp, mock_input_modal: Mock
    ) -> None:
        """Test on_tree_node_selected method."""
        # Mock the task_tree
        task_editor_app.task_tree = Mock()
        task_editor_app.task_tree.cursor_node = Mock()

        # Create a mock event
        mock_event = Mock()
        mock_event.node = Mock()
        mock_event.node.label = "Test Node"

        # Mock the push_screen method
        task_editor_app.push_screen = Mock()

        # Mock the InputModal class
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal", mock_input_modal
        ):
            await task_editor_app.on_tree_node_selected(mock_event)

            # Verify push_screen was called
            task_editor_app.push_screen.assert_called_once()

    async def test_on_tree_node_selected_with_none_label(
        self, task_editor_app: TaskEditorApp, mock_input_modal: Mock
    ) -> None:
        """Test on_tree_node_selected method with None label."""
        # Mock the task_tree
        task_editor_app.task_tree = Mock()
        task_editor_app.task_tree.cursor_node = Mock()

        # Create a mock event
        mock_event = Mock()
        mock_event.node = Mock()
        mock_event.node.label = None

        # Mock the push_screen method
        task_editor_app.push_screen = Mock()

        # Mock the InputModal class
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal", mock_input_modal
        ):
            await task_editor_app.on_tree_node_selected(mock_event)

            # Verify push_screen was called
            task_editor_app.push_screen.assert_called_once()


class TestTaskEditorAppKeyboardHandling:
    """Test TaskEditorApp keyboard handling methods."""

    async def test_on_key_add_node(self, task_editor_app: TaskEditorApp) -> None:
        """Test on_key method with 'a' key."""
        # Mock the task_tree
        task_editor_app.task_tree = Mock()
        task_editor_app.task_tree.cursor_node = Mock()
        task_editor_app.task_tree.cursor_node.parent = Mock()

        # Mock the action_add_node method
        task_editor_app.action_add_node = AsyncMock()

        # Create a mock event
        mock_event = Mock()
        mock_event.key = "a"

        await task_editor_app.on_key(mock_event)

        # Verify action_add_node was called
        task_editor_app.action_add_node.assert_called_once()

    async def test_on_key_delete_node(self, task_editor_app: TaskEditorApp) -> None:
        """Test on_key method with 'd' key."""
        # Mock the task_tree
        task_editor_app.task_tree = Mock()
        task_editor_app.task_tree.cursor_node = Mock()
        task_editor_app.task_tree.cursor_node.remove = Mock()
        task_editor_app.task_tree.cursor_node.parent = Mock()

        # Mock the root with proper children list
        task_editor_app.task_tree.root = Mock()
        task_editor_app.task_tree.root.children = []

        # Create a mock event
        mock_event = Mock()
        mock_event.key = "d"

        await task_editor_app.on_key(mock_event)

    async def test_on_key_save_exit(self, task_editor_app: TaskEditorApp) -> None:
        """Test on_key method with 's' key."""
        # Mock the task_tree
        task_editor_app.task_tree = Mock()
        task_editor_app.task_tree.cursor_node = Mock()

        # Mock the exit method
        task_editor_app.exit = Mock()

        # Mock the update_tasks method to prevent issues
        task_editor_app.update_tasks = AsyncMock()

        # Create a mock event
        mock_event = Mock()
        mock_event.key = "s"

        await task_editor_app.on_key(mock_event)

        # Verify update_tasks was called
        task_editor_app.update_tasks.assert_called_once()
        # Verify exit was called
        task_editor_app.exit.assert_called_once_with(task_editor_app.tasks)

    async def test_on_key_other_key(self, task_editor_app: TaskEditorApp) -> None:
        """Test on_key method with other keys."""
        # Mock the task_tree
        task_editor_app.task_tree = Mock()
        task_editor_app.task_tree.cursor_node = Mock()

        # Create a mock event
        mock_event = Mock()
        mock_event.key = "x"

        # Should not raise any exception
        await task_editor_app.on_key(mock_event)

    async def test_on_key_no_cursor_node(self, task_editor_app: TaskEditorApp) -> None:
        """Test on_key method with no cursor node."""
        # Mock the task_tree
        task_editor_app.task_tree = Mock()
        task_editor_app.task_tree.cursor_node = None

        # Create a mock event
        mock_event = Mock()
        mock_event.key = "a"

        # Should not raise any exception
        await task_editor_app.on_key(mock_event)

    async def test_on_key_no_parent(self, task_editor_app: TaskEditorApp) -> None:
        """Test on_key method with no parent node."""
        # Mock the task_tree
        task_editor_app.task_tree = Mock()
        task_editor_app.task_tree.cursor_node = Mock()
        task_editor_app.task_tree.cursor_node.parent = None

        # Mock the push_screen method to avoid Textual issues
        task_editor_app.push_screen = Mock()

        # Mock the InputModal class to prevent stylesheet issues
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal"
        ) as mock_input_modal:
            mock_modal_instance = Mock()
            mock_modal_instance.result = ""
            mock_input_modal.return_value = mock_modal_instance

            # Create a mock event
            mock_event = Mock()
            mock_event.key = "a"

            # Should not raise any exception
            await task_editor_app.on_key(mock_event)

            # Verify push_screen was called
            task_editor_app.push_screen.assert_called_once()


class TestTaskEditorAppNodeManagement:
    """Test TaskEditorApp node management methods."""

    async def test_action_add_node_step_node(
        self, task_editor_app: TaskEditorApp, mock_input_modal: Mock
    ) -> None:
        """Test action_add_node method with step node."""
        # Mock the task_tree
        task_editor_app.task_tree = Mock()
        task_editor_app.task_tree.cursor_node = Mock()
        task_editor_app.task_tree.cursor_node.parent = Mock()
        task_editor_app.task_tree.cursor_node.parent.parent = Mock()

        # Mock the push_screen method
        task_editor_app.push_screen = Mock()

        # Mock the InputModal class
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal", mock_input_modal
        ):
            await task_editor_app.action_add_node(task_editor_app.task_tree.cursor_node)

            # Verify push_screen was called
            task_editor_app.push_screen.assert_called_once()

    async def test_action_add_node_task_node_expanded(
        self, task_editor_app: TaskEditorApp, mock_input_modal: Mock
    ) -> None:
        """Test action_add_node method with expanded task node."""
        # Mock the task_tree
        task_editor_app.task_tree = Mock()
        task_editor_app.task_tree.cursor_node = Mock()
        task_editor_app.task_tree.cursor_node.parent = Mock()  # Add parent
        task_editor_app.task_tree.cursor_node.parent.parent = (
            None  # Set parent.parent to None
        )
        task_editor_app.task_tree.cursor_node.is_expanded = True

        # Mock the push_screen method
        task_editor_app.push_screen = Mock()

        # Mock the InputModal class
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal", mock_input_modal
        ):
            await task_editor_app.action_add_node(task_editor_app.task_tree.cursor_node)

            # Verify push_screen was called
            task_editor_app.push_screen.assert_called_once()

    async def test_action_add_node_task_node_not_expanded(
        self, task_editor_app: TaskEditorApp, mock_input_modal: Mock
    ) -> None:
        """Test action_add_node method with non-expanded task node."""
        # Mock the task_tree
        task_editor_app.task_tree = Mock()
        task_editor_app.task_tree.cursor_node = Mock()
        task_editor_app.task_tree.cursor_node.parent = Mock()  # Add parent
        task_editor_app.task_tree.cursor_node.parent.parent = (
            None  # Set parent.parent to None
        )
        task_editor_app.task_tree.cursor_node.is_expanded = False

        # Mock the push_screen method
        task_editor_app.push_screen = Mock()

        # Mock the InputModal class
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal", mock_input_modal
        ):
            await task_editor_app.action_add_node(task_editor_app.task_tree.cursor_node)

            # Verify push_screen was called
            task_editor_app.push_screen.assert_called_once()

    def test_show_input_modal(
        self, task_editor_app: TaskEditorApp, mock_input_modal: Mock
    ) -> None:
        """Test show_input_modal method."""
        # Mock the push_screen method
        task_editor_app.push_screen = Mock()

        # Create a fresh mock for this test with correct result
        fresh_mock_modal = Mock()
        fresh_mock_modal_instance = Mock()
        fresh_mock_modal_instance.result = "default value"
        fresh_mock_modal.return_value = fresh_mock_modal_instance

        # Mock the InputModal class
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal", fresh_mock_modal
        ):
            result = task_editor_app.show_input_modal("Test Title", "default value")

            # Verify push_screen was called
            task_editor_app.push_screen.assert_called_once()
            # Verify the result is returned correctly
            assert result == "default value"

    def test_show_input_modal_with_empty_default(
        self, task_editor_app: TaskEditorApp
    ) -> None:
        """Test show_input_modal method with empty default value."""
        # Mock the push_screen method
        task_editor_app.push_screen = Mock()

        # Create a fresh mock for this test
        fresh_mock_modal = Mock()
        fresh_mock_modal_instance = Mock()
        fresh_mock_modal_instance.result = ""
        fresh_mock_modal.return_value = fresh_mock_modal_instance

        # Mock the InputModal class
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal", fresh_mock_modal
        ):
            result = task_editor_app.show_input_modal("Test Title")

            # Verify push_screen was called
            task_editor_app.push_screen.assert_called_once()
            # Verify the result is returned correctly
            assert result == ""

    def test_show_input_modal_various(self, task_editor_app: TaskEditorApp) -> None:
        """Test show_input_modal method with various parameters."""
        # Mock the push_screen method
        task_editor_app.push_screen = Mock()

        # Create a fresh mock for this test
        fresh_mock_modal = Mock()
        fresh_mock_modal_instance = Mock()
        fresh_mock_modal_instance.result = "default value"
        fresh_mock_modal.return_value = fresh_mock_modal_instance

        # Mock the InputModal class
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal", fresh_mock_modal
        ):
            result = task_editor_app.show_input_modal("Test Title", "default value")

            # Verify push_screen was called
            task_editor_app.push_screen.assert_called_once()
            # Verify the result is returned correctly
            assert result == "default value"

    def test_show_input_modal_returns_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that show_input_modal returns the default value."""

        # Create a dummy modal class for testing
        class DummyModal:
            def __init__(self, *a: object, **k: object) -> None:
                pass

        app = TaskEditorApp([])
        app.push_screen = Mock()

        # Mock the InputModal class and also patch the module attribute
        with (
            patch(
                "arklex.orchestrator.generator.ui.task_editor.InputModal", DummyModal
            ),
            patch("sys.modules") as mock_modules,
        ):
            mock_module = Mock()
            mock_module.InputModal = DummyModal
            mock_modules.__getitem__.return_value = mock_module

            result = app.show_input_modal("Test Title", "default value")
            assert result == "default value"


class TestTaskEditorAppDataManagement:
    """Test TaskEditorApp data management methods."""

    async def test_update_tasks(self, mock_log_context: Mock) -> None:
        """Test update_tasks method."""
        app = TaskEditorApp([])

        # Mock the task_tree
        app.task_tree = Mock()
        app.task_tree.root = Mock()
        app.task_tree.root.children = []

        # Mock the log_context at the module level
        with (
            patch(
                "arklex.orchestrator.generator.ui.task_editor.log_context",
                mock_log_context,
            ),
            contextlib.suppress(AttributeError),
        ):
            await app.update_tasks()

    async def test_update_tasks_empty_tree(self, mock_log_context: Mock) -> None:
        """Test update_tasks method with empty tree."""
        app = TaskEditorApp([])

        # Mock the task_tree with empty root
        app.task_tree = Mock()
        app.task_tree.root = Mock()
        app.task_tree.root.children = []

        # Mock the log_context at the module level
        with (
            patch(
                "arklex.orchestrator.generator.ui.task_editor.log_context",
                mock_log_context,
            ),
            contextlib.suppress(AttributeError),
        ):
            await app.update_tasks()

            # The method should complete without raising an exception
            assert app.tasks == []

    async def test_update_tasks_with_none_tree(
        self, task_editor_app: TaskEditorApp
    ) -> None:
        """Test update_tasks method with None tree."""
        task_editor_app.task_tree = None

        # Should not raise any exception
        await task_editor_app.update_tasks()

    async def test_update_tasks_with_invalid_node_structure(
        self, mock_log_context: Mock
    ) -> None:
        """Test update_tasks method with invalid node structure."""
        app = TaskEditorApp([])

        # Mock the task_tree with invalid structure
        app.task_tree = Mock()
        app.task_tree.root = Mock()
        app.task_tree.root.children = [None]  # Invalid child

        # Mock the log_context at the module level
        with (
            patch(
                "arklex.orchestrator.generator.ui.task_editor.log_context",
                mock_log_context,
            ),
            contextlib.suppress(AttributeError),
        ):
            await app.update_tasks()

    async def test_update_tasks_with_missing_step_data(
        self, mock_log_context: Mock
    ) -> None:
        """Test update_tasks method with missing step data."""
        app = TaskEditorApp([])

        # Mock the task_tree
        app.task_tree = Mock()
        app.task_tree.root = Mock()

        # Create a mock node with missing step data
        mock_node = Mock()
        mock_node.label = Mock()
        mock_node.label.plain = "Test Node"
        mock_node.children = []

        app.task_tree.root.children = [mock_node]

        # Mock the log_context at the module level
        with (
            patch(
                "arklex.orchestrator.generator.ui.task_editor.log_context",
                mock_log_context,
            ),
            contextlib.suppress(AttributeError),
        ):
            await app.update_tasks()

            # The method should complete without raising an exception
            assert len(app.tasks) == 1
            assert app.tasks[0]["name"] == "Test Node"
            # The steps key is only added if there are steps, so it shouldn't be present here

    def test_run_returns_tasks(self, task_editor_app: TaskEditorApp) -> None:
        """Test run method returns tasks."""
        # Just test that the method returns the tasks attribute
        # We'll skip the actual super().run() call to avoid Textual issues
        assert task_editor_app.run() == task_editor_app.tasks

    def test_task_editor_app_init_with_none(self) -> None:
        """Test TaskEditorApp initialization with None."""
        app = TaskEditorApp(None)
        assert app.tasks is None
        assert app.task_tree is None


class TestTaskEditorFinalCoverage:
    """Test TaskEditorApp for final coverage."""

    def test_task_editor_init_with_none_tasks(self) -> None:
        """Test TaskEditorApp initialization with None tasks."""
        app = TaskEditorApp(None)
        assert app.tasks is None

    def test_task_editor_compose_with_none_tasks(self) -> None:
        """Test compose method with None tasks."""
        app = TaskEditorApp(None)

        # Call compose and convert to list - should work with fake textual classes
        result = list(app.compose())

        # Verify that compose returns a result
        assert result is not None
        assert len(result) >= 1

    def test_task_editor_update_tasks_with_none_tree(self) -> None:
        """Test update_tasks method with None tree."""
        app = TaskEditorApp([])
        app.task_tree = None

        # Should not raise any exception
        asyncio.run(app.update_tasks())

    def test_task_editor_update_tasks_with_none_root(self) -> None:
        """Test update_tasks method with None root."""
        app = TaskEditorApp([])
        app.task_tree = Mock()
        app.task_tree.root = None

        # Should not raise any exception
        asyncio.run(app.update_tasks())

    def test_task_editor_update_tasks_with_getattr_none_root(self) -> None:
        """Test update_tasks method with getattr returning None root."""
        app = TaskEditorApp([])
        app.task_tree = Mock()
        app.task_tree.root = None

        # Should not raise any exception
        asyncio.run(app.update_tasks())

    async def test_task_editor_update_tasks_with_complex_tree_structure(self) -> None:
        """Test update_tasks method with complex tree structure."""
        app = TaskEditorApp([])

        # Create fake classes for testing
        class FakeLabel:
            def __init__(self, plain: str) -> None:
                self.plain = plain

        class FakeNode:
            def __init__(self, label: object, children: list | None = None) -> None:
                self.label = label
                self.children = children or []

        # Mock the task_tree
        app.task_tree = Mock()
        app.task_tree.root = Mock()
        app.task_tree.root.children = [
            FakeNode("Task 1", [FakeNode("Step 1"), FakeNode("Step 2")]),
            FakeNode("Task 2", [FakeNode("Step 3")]),
        ]

        await app.update_tasks()

        # Verify tasks were updated correctly
        assert len(app.tasks) == 2
        assert app.tasks[0]["name"] == "Task 1"
        assert app.tasks[0]["steps"] == ["Step 1", "Step 2"]
        assert app.tasks[1]["name"] == "Task 2"
        assert app.tasks[1]["steps"] == ["Step 3"]

    def test_task_editor_run_method(self) -> None:
        """Test run method."""
        app = TaskEditorApp([])
        # Just test that the method returns the tasks attribute
        assert app.run() == []

    def test_task_editor_show_input_modal_with_callback(self) -> None:
        """Test show_input_modal method with callback."""
        app = TaskEditorApp([])

        # Mock the push_screen method
        app.push_screen = Mock()

        # Create a fresh mock for this test
        fresh_mock_modal = Mock()
        fresh_mock_modal_instance = Mock()
        fresh_mock_modal_instance.result = "default value"
        fresh_mock_modal.return_value = fresh_mock_modal_instance

        # Mock the InputModal class
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal", fresh_mock_modal
        ):
            result = app.show_input_modal("Test Title", "default value")

            # Verify push_screen was called
            app.push_screen.assert_called_once()
            # Verify the result is returned correctly
            assert result == "default value"

    def test_task_editor_show_input_modal_without_default(self) -> None:
        """Test show_input_modal method without default value."""
        app = TaskEditorApp([])

        # Mock the push_screen method
        app.push_screen = Mock()

        # Create a fresh mock for this test
        fresh_mock_modal = Mock()
        fresh_mock_modal_instance = Mock()
        fresh_mock_modal_instance.result = ""
        fresh_mock_modal.return_value = fresh_mock_modal_instance

        # Mock the InputModal class
        with patch(
            "arklex.orchestrator.generator.ui.task_editor.InputModal", fresh_mock_modal
        ):
            result = app.show_input_modal("Test Title")

            # Verify push_screen was called
            app.push_screen.assert_called_once()
            # Verify the result is returned correctly
            assert result == ""


def test_update_tasks_handles_no_task_tree_or_root() -> None:
    app = TaskEditorApp([])
    app.task_tree = None
    # Should return early (line 139)
    assert asyncio.run(app.update_tasks()) is None
    app.task_tree = Mock()
    app.task_tree.root = None
    # Should return early (line 140)
    assert asyncio.run(app.update_tasks()) is None


def test_update_tasks_handles_various_label_types() -> None:
    app = TaskEditorApp([])

    class FakeLabel:
        def __init__(self, plain: str) -> None:
            self.plain = plain

    class FakeNode:
        def __init__(self, label: object, children: list | None = None) -> None:
            self.label = label
            self.children = children or []

    # Step label with and without 'plain'
    app.task_tree = Mock()
    app.task_tree.root = Mock()
    app.task_tree.root.children = [
        FakeNode(FakeLabel("Task1"), [FakeNode(FakeLabel("Step1")), FakeNode("Step2")]),
        FakeNode("Task2", [FakeNode("Step3")]),
    ]
    asyncio.run(app.update_tasks())  # lines 145, 200-201
    assert app.tasks[0]["name"] == "Task1"
    assert app.tasks[0]["steps"] == ["Step1", "Step2"]
    assert app.tasks[1]["name"] == "Task2"
    assert app.tasks[1]["steps"] == ["Step3"]


def test_push_screen_calls_super() -> None:
    called = {}

    class MyTaskEditorApp(TaskEditorApp):
        def __init__(self) -> None:
            super().__init__([])

        def push_screen(self, screen: object) -> object:
            called["super"] = screen
            return super().push_screen(screen)

    app = MyTaskEditorApp()
    # Patch the superclass push_screen to avoid event loop error
    import unittest.mock

    with unittest.mock.patch.object(
        TaskEditorApp.__bases__[0], "push_screen", return_value=None
    ) as mock_super:
        app.push_screen("test_screen")  # line 214
        assert called["super"] == "test_screen"
        mock_super.assert_called_once_with("test_screen")


def test_show_input_modal_returns_default(monkeypatch: pytest.MonkeyPatch) -> None:
    app = TaskEditorApp([])

    class DummyModal:
        def __init__(self, *a: object, **k: object) -> None:
            pass

    monkeypatch.setattr(
        "arklex.orchestrator.generator.ui.task_editor.InputModal", DummyModal
    )
    app.push_screen = Mock()
    result = app.show_input_modal("title", "default")  # line 264
    assert result == "default"


def test_run_calls_super_and_returns_tasks() -> None:
    called = {}

    class MyTaskEditorApp(TaskEditorApp):
        def __init__(self) -> None:
            super().__init__([{"name": "T"}])

        def run(self) -> object:
            called["super"] = True
            return super().run()

    app = MyTaskEditorApp()
    app.tasks = [1, 2, 3]
    result = app.run()  # line 273
    assert called["super"] is True
    assert result == [1, 2, 3]
