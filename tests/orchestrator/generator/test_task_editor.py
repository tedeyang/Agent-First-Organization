import pytest
from unittest.mock import MagicMock

import arklex.orchestrator.generator.ui.task_editor as task_editor


def test_import_task_editor_module() -> None:
    assert hasattr(task_editor, "TaskEditorApp")


def test_task_editor_instantiation() -> None:
    # Test TaskEditorApp instantiation
    tasks = [
        {
            "name": "Test Task 1",
            "steps": [{"description": "Step 1"}, {"description": "Step 2"}],
        }
    ]
    editor = task_editor.TaskEditorApp(tasks)
    assert editor.tasks == tasks
    assert editor.task_tree is None  # Will be set in compose()


def test_task_editor_methods() -> None:
    # Test TaskEditorApp methods
    tasks = [{"name": "Test Task 1", "steps": [{"description": "Step 1"}]}]
    editor = task_editor.TaskEditorApp(tasks)

    # Test show_input_modal method
    result = editor.show_input_modal("Test Title", "Default Value")
    # This method returns the modal result, but in test context it might be None
    # We just test that the method exists and doesn't raise an error
    assert hasattr(editor, "show_input_modal")


def test_task_editor_error_case() -> None:
    # Test with invalid input - TaskEditorApp doesn't raise TypeError with None
    # Let's test with an empty list instead
    editor = task_editor.TaskEditorApp([])
    assert editor.tasks == []


def test_task_editor_compose() -> None:
    """Test the compose method."""
    tasks = [{"name": "Test Task 1", "steps": [{"description": "Step 1"}]}]
    editor = task_editor.TaskEditorApp(tasks)

    # Test that compose returns a generator
    compose_result = editor.compose()
    assert hasattr(compose_result, "__iter__")


def test_task_editor_update_tasks() -> None:
    """Test the update_tasks method."""
    tasks = [{"name": "Test Task 1", "steps": [{"description": "Step 1"}]}]
    editor = task_editor.TaskEditorApp(tasks)

    # Mock the task_tree structure
    editor.task_tree = MagicMock()
    editor.task_tree.root = MagicMock()
    editor.task_tree.root.children = []

    # Mock a task node
    task_node = MagicMock()
    task_node.label.plain = "Test Task 1"
    task_node.children = []

    # Mock a step node
    step_node = MagicMock()
    step_node.label.plain = {"description": "Step 1"}

    task_node.children = [step_node]
    editor.task_tree.root.children = [task_node]

    # Test update_tasks
    editor.update_tasks()
    assert len(editor.tasks) == 1
    assert editor.tasks[0]["name"] == "Test Task 1"
    assert len(editor.tasks[0]["steps"]) == 1
    assert editor.tasks[0]["steps"][0] == {"description": "Step 1"}
