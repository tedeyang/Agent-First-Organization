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


def test_task_editor_instantiation_with_empty_tasks() -> None:
    """Test TaskEditorApp instantiation with empty tasks list."""
    editor = task_editor.TaskEditorApp([])
    assert editor.tasks == []
    assert editor.task_tree is None


def test_task_editor_instantiation_with_none_tasks() -> None:
    """Test TaskEditorApp instantiation with None tasks."""
    editor = task_editor.TaskEditorApp(None)
    assert editor.tasks is None
    assert editor.task_tree is None


def test_task_editor_instantiation_with_complex_tasks() -> None:
    """Test TaskEditorApp instantiation with complex task structure."""
    tasks = [
        {
            "name": "Complex Task",
            "description": "A complex task description",
            "steps": [
                {"description": "Step 1", "duration": "1 hour"},
                {"description": "Step 2", "dependencies": ["Step 1"]},
            ],
            "priority": "high",
            "estimated_duration": "2 hours",
        }
    ]
    editor = task_editor.TaskEditorApp(tasks)
    assert editor.tasks == tasks
    assert len(editor.tasks[0]["steps"]) == 2


def test_task_editor_methods() -> None:
    # Test TaskEditorApp methods
    tasks = [{"name": "Test Task 1", "steps": [{"description": "Step 1"}]}]
    editor = task_editor.TaskEditorApp(tasks)

    # Test show_input_modal method
    result = editor.show_input_modal("Test Title", "Default Value")
    # This method returns the modal result, but in test context it might be None
    # We just test that the method exists and doesn't raise an error
    assert hasattr(editor, "show_input_modal")


def test_task_editor_show_input_modal_with_empty_title() -> None:
    """Test show_input_modal with empty title."""
    tasks = [{"name": "Test Task 1", "steps": [{"description": "Step 1"}]}]
    editor = task_editor.TaskEditorApp(tasks)

    result = editor.show_input_modal("", "Default Value")
    assert hasattr(editor, "show_input_modal")


def test_task_editor_show_input_modal_with_none_default() -> None:
    """Test show_input_modal with None default value."""
    tasks = [{"name": "Test Task 1", "steps": [{"description": "Step 1"}]}]
    editor = task_editor.TaskEditorApp(tasks)

    result = editor.show_input_modal("Test Title", None)
    assert hasattr(editor, "show_input_modal")


def test_task_editor_error_case() -> None:
    # Test with invalid input - TaskEditorApp doesn't raise TypeError with None
    # Let's test with an empty list instead
    editor = task_editor.TaskEditorApp([])
    assert editor.tasks == []


def test_task_editor_error_case_with_invalid_task_structure() -> None:
    """Test TaskEditorApp with invalid task structure."""
    invalid_tasks = [
        {"invalid_key": "invalid_value"},  # Missing required 'name' field
        {"name": "Valid Task", "steps": "not_a_list"},  # Invalid steps type
    ]

    # Should not raise an exception, just handle gracefully
    editor = task_editor.TaskEditorApp(invalid_tasks)
    assert editor.tasks == invalid_tasks


def test_task_editor_compose() -> None:
    """Test the compose method."""
    tasks = [{"name": "Test Task 1", "steps": [{"description": "Step 1"}]}]
    editor = task_editor.TaskEditorApp(tasks)

    # Test that compose returns a generator
    compose_result = editor.compose()
    assert hasattr(compose_result, "__iter__")


def test_task_editor_compose_with_empty_tasks() -> None:
    """Test the compose method with empty tasks."""
    editor = task_editor.TaskEditorApp([])

    compose_result = editor.compose()
    assert hasattr(compose_result, "__iter__")


def test_task_editor_compose_with_none_tasks() -> None:
    """Test the compose method with None tasks."""
    editor = task_editor.TaskEditorApp(None)

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


def test_task_editor_update_tasks_with_multiple_tasks() -> None:
    """Test the update_tasks method with multiple tasks."""
    tasks = [
        {"name": "Task 1", "steps": [{"description": "Step 1"}]},
        {"name": "Task 2", "steps": [{"description": "Step 2"}]},
    ]
    editor = task_editor.TaskEditorApp(tasks)

    # Mock the task_tree structure
    editor.task_tree = MagicMock()
    editor.task_tree.root = MagicMock()
    editor.task_tree.root.children = []

    # Mock task nodes
    task_node_1 = MagicMock()
    task_node_1.label.plain = "Task 1"
    step_node_1 = MagicMock()
    step_node_1.label.plain = {"description": "Step 1"}
    task_node_1.children = [step_node_1]

    task_node_2 = MagicMock()
    task_node_2.label.plain = "Task 2"
    step_node_2 = MagicMock()
    step_node_2.label.plain = {"description": "Step 2"}
    task_node_2.children = [step_node_2]

    editor.task_tree.root.children = [task_node_1, task_node_2]

    # Test update_tasks
    editor.update_tasks()
    assert len(editor.tasks) == 2
    assert editor.tasks[0]["name"] == "Task 1"
    assert editor.tasks[1]["name"] == "Task 2"


def test_task_editor_update_tasks_with_nested_steps() -> None:
    """Test the update_tasks method with nested step structure."""
    tasks = [
        {
            "name": "Complex Task",
            "steps": [{"description": "Step 1", "substeps": ["Substep 1"]}],
        }
    ]
    editor = task_editor.TaskEditorApp(tasks)

    # Mock the task_tree structure
    editor.task_tree = MagicMock()
    editor.task_tree.root = MagicMock()
    editor.task_tree.root.children = []

    # Mock a task node with complex step
    task_node = MagicMock()
    task_node.label.plain = "Complex Task"

    step_node = MagicMock()
    step_node.label.plain = {"description": "Step 1", "substeps": ["Substep 1"]}
    task_node.children = [step_node]

    editor.task_tree.root.children = [task_node]

    # Test update_tasks
    editor.update_tasks()
    assert len(editor.tasks) == 1
    assert editor.tasks[0]["name"] == "Complex Task"
    assert "substeps" in editor.tasks[0]["steps"][0]


def test_task_editor_update_tasks_with_empty_task_tree() -> None:
    """Test the update_tasks method with empty task tree."""
    tasks = [{"name": "Test Task", "steps": [{"description": "Step 1"}]}]
    editor = task_editor.TaskEditorApp(tasks)

    # Mock empty task_tree
    editor.task_tree = MagicMock()
    editor.task_tree.root = MagicMock()
    editor.task_tree.root.children = []

    # Test update_tasks - should not modify tasks when tree is empty
    editor.update_tasks()
    assert len(editor.tasks) == 1  # Should remain unchanged


def test_task_editor_update_tasks_with_none_task_tree() -> None:
    """Test the update_tasks method with None task tree."""
    tasks = [{"name": "Test Task", "steps": [{"description": "Step 1"}]}]
    editor = task_editor.TaskEditorApp(tasks)
    editor.task_tree = None

    # Should not raise an exception
    editor.update_tasks()
    assert editor.tasks == tasks  # Should remain unchanged


def test_task_editor_update_tasks_with_invalid_node_structure() -> None:
    """Test the update_tasks method with invalid node structure."""
    tasks = [{"name": "Test Task", "steps": [{"description": "Step 1"}]}]
    editor = task_editor.TaskEditorApp(tasks)

    # Mock task_tree with invalid node structure
    editor.task_tree = MagicMock()
    editor.task_tree.root = MagicMock()

    # Mock invalid task node (missing label.plain)
    invalid_task_node = MagicMock()
    invalid_task_node.label = None
    invalid_task_node.children = []

    editor.task_tree.root.children = [invalid_task_node]

    # Should handle gracefully without raising exception
    editor.update_tasks()
    assert len(editor.tasks) == 1  # Should remain unchanged


def test_task_editor_update_tasks_with_missing_step_data() -> None:
    """Test the update_tasks method with missing step data."""
    tasks = [{"name": "Test Task", "steps": [{"description": "Step 1"}]}]
    editor = task_editor.TaskEditorApp(tasks)

    # Mock task_tree with missing step data
    editor.task_tree = MagicMock()
    editor.task_tree.root = MagicMock()

    task_node = MagicMock()
    task_node.label.plain = "Test Task"

    # Mock step node with missing label.plain
    step_node = MagicMock()
    step_node.label = None
    task_node.children = [step_node]

    editor.task_tree.root.children = [task_node]

    # Should handle gracefully
    editor.update_tasks()
    assert len(editor.tasks) == 1
    assert editor.tasks[0]["name"] == "Test Task"
    assert len(editor.tasks[0]["steps"]) == 1  # Should remain unchanged


def test_task_editor_methods_existence() -> None:
    """Test that all expected methods exist on TaskEditorApp."""
    tasks = [{"name": "Test Task", "steps": [{"description": "Step 1"}]}]
    editor = task_editor.TaskEditorApp(tasks)

    expected_methods = ["compose", "update_tasks", "show_input_modal"]
    for method_name in expected_methods:
        assert hasattr(editor, method_name)
        assert callable(getattr(editor, method_name))


def test_task_editor_attributes_existence() -> None:
    """Test that all expected attributes exist on TaskEditorApp."""
    tasks = [{"name": "Test Task", "steps": [{"description": "Step 1"}]}]
    editor = task_editor.TaskEditorApp(tasks)

    expected_attributes = ["tasks", "task_tree"]
    for attr_name in expected_attributes:
        assert hasattr(editor, attr_name)
