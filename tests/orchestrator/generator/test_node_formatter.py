import pytest

import arklex.orchestrator.generator.formatting.node_formatter as node_formatter


def test_import_node_formatter_module() -> None:
    assert hasattr(node_formatter, "NodeFormatter")


def test_format_node_basic() -> None:
    # Test the NodeFormatter class
    formatter = node_formatter.NodeFormatter()
    task = {"name": "Test Task", "description": "Test Description"}
    result = formatter.format_node(task, "n1")
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == "n1"
    assert "resource" in result[1]
    assert "attribute" in result[1]


def test_format_node_missing_fields() -> None:
    # Test with missing optional fields
    formatter = node_formatter.NodeFormatter()
    task = {"name": "Test Task"}  # Missing description
    result = formatter.format_node(task, "n2")
    assert isinstance(result, list)
    assert len(result) == 2


def test_format_node_invalid_type() -> None:
    # Test with task that has type field
    formatter = node_formatter.NodeFormatter()
    task = {"name": "Test Task", "type": "unknown"}
    result = formatter.format_node(task, "n3")
    assert isinstance(result, list)
    assert "type" in result[1]


def test_format_node_error_case() -> None:
    # Test with invalid input
    formatter = node_formatter.NodeFormatter()
    with pytest.raises(AttributeError):
        formatter.format_node(None, "n4")


def test_format_node_data() -> None:
    """Test the format_node_data method."""
    formatter = node_formatter.NodeFormatter()
    task = {"name": "Test Task", "description": "Test Description"}
    result = formatter.format_node_data(task)
    assert isinstance(result, dict)
    assert "resource" in result
    assert "attribute" in result
    assert result["resource"]["name"] == "MessageWorker"


def test_format_node_style() -> None:
    """Test the format_node_style method."""
    formatter = node_formatter.NodeFormatter()
    task = {"name": "Test Task", "priority": "high"}
    result = formatter.format_node_style(task)
    assert isinstance(result, dict)
    assert "color" in result
    assert "border" in result
    assert "padding" in result
    assert result["color"] == "#ff0000"  # high priority color


def test_validate_node() -> None:
    """Test the validate_node method."""
    formatter = node_formatter.NodeFormatter()

    # Valid node
    valid_node = {
        "id": "n1",
        "type": "task",
        "data": {
            "name": "Test Task",
            "description": "Test Description",
            "priority": "high",
        },
    }
    assert formatter.validate_node(valid_node)

    # Invalid node - missing id
    invalid_node = {"type": "task", "data": {"name": "Test"}}
    assert not formatter.validate_node(invalid_node)

    # Invalid node - missing type
    invalid_node = {"id": "n1", "data": {"name": "Test"}}
    assert not formatter.validate_node(invalid_node)

    # Invalid node - wrong id type
    invalid_node = {"id": 123, "type": "task", "data": {"name": "Test"}}
    assert not formatter.validate_node(invalid_node)

    # Invalid node - wrong type type
    invalid_node = {"id": "n1", "type": 123, "data": {"name": "Test"}}
    assert not formatter.validate_node(invalid_node)

    # Invalid node - invalid data structure
    invalid_node = {"id": "n1", "type": "task", "data": "not_a_dict"}
    assert not formatter.validate_node(invalid_node)


def test_validate_attribute() -> None:
    """Test the _validate_attribute method."""
    formatter = node_formatter.NodeFormatter()

    # Test steps validation
    valid_steps = [
        {"step_id": "s1", "description": "Step 1", "order": 1},
        {"step_id": "s2", "description": "Step 2", "order": 2},
    ]
    assert formatter._validate_attribute(valid_steps, "steps")

    # Test dependencies validation
    valid_deps = ["dep1", "dep2"]
    assert formatter._validate_attribute(valid_deps, "dependencies")

    # Test required_resources validation
    valid_resources = ["resource1", "resource2"]
    assert formatter._validate_attribute(valid_resources, "required_resources")

    # Test estimated_duration validation
    assert formatter._validate_attribute("2 hours", "estimated_duration")

    # Test priority validation
    assert formatter._validate_attribute("high", "priority")
    assert formatter._validate_attribute("medium", "priority")
    assert formatter._validate_attribute("low", "priority")

    # Test invalid priority
    assert not formatter._validate_attribute("invalid", "priority")
