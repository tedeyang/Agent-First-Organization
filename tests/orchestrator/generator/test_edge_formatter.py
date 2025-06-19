import pytest

import arklex.orchestrator.generator.formatting.edge_formatter as edge_formatter


def test_import_edge_formatter_module() -> None:
    assert hasattr(edge_formatter, "EdgeFormatter")


def test_format_edge_basic() -> None:
    # Test the EdgeFormatter class
    formatter = edge_formatter.EdgeFormatter()
    source_task = {"name": "Task A"}
    target_task = {"name": "Task B"}
    result = formatter.format_edge("n1", "n2", source_task, target_task)
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == "n1"
    assert result[1] == "n2"
    assert "intent" in result[2]
    assert "attribute" in result[2]


def test_format_edge_missing_fields() -> None:
    # Test with missing optional fields
    formatter = edge_formatter.EdgeFormatter()
    source_task = {"name": "Task A"}
    target_task = {}  # Missing name
    result = formatter.format_edge("n1", "n2", source_task, target_task)
    assert isinstance(result, list)
    assert len(result) == 3


def test_format_edge_invalid_type() -> None:
    # Test with custom edge type
    formatter = edge_formatter.EdgeFormatter(default_intent="custom_type")
    source_task = {"name": "Task A"}
    target_task = {"name": "Task B"}
    result = formatter.format_edge("n1", "n2", source_task, target_task)
    assert isinstance(result, list)
    assert result[2]["intent"] == "custom_type"


def test_format_edge_error_case() -> None:
    # Test with None as source_idx, should not raise, just return None as first element
    formatter = edge_formatter.EdgeFormatter()
    result = formatter.format_edge(None, "n2", {}, {})
    assert result[0] is None
    assert result[1] == "n2"
    assert isinstance(result[2], dict)


def test_format_edge_data() -> None:
    """Test the format_edge_data method."""
    formatter = edge_formatter.EdgeFormatter()
    source = {"id": "n1", "name": "Source"}
    target = {"id": "n2", "name": "Target"}
    result = formatter.format_edge_data(
        source, target, "depends_on", 2.0, "Custom Label"
    )
    assert isinstance(result, dict)
    assert "intent" in result
    assert "attribute" in result
    assert result["intent"] == "Custom Label"


def test_format_edge_style() -> None:
    """Test the format_edge_style method."""
    formatter = edge_formatter.EdgeFormatter()
    source = {"id": "n1"}
    target = {"id": "n2"}
    result = formatter.format_edge_style(source, target, "depends_on", 1.0, "Label")
    assert isinstance(result, dict)
    assert "color" in result
    assert "width" in result
    assert "style" in result


def test_validate_edge() -> None:
    """Test the validate_edge method."""
    formatter = edge_formatter.EdgeFormatter()

    # Valid edge
    assert formatter.validate_edge("n1", "n2", "depends_on", 1.0, "Label")

    # Invalid edge - missing source ID
    assert not formatter.validate_edge({}, "n2", "depends_on", 1.0, "Label")

    # Invalid edge - missing target ID
    assert not formatter.validate_edge("n1", {}, "depends_on", 1.0, "Label")

    # Invalid edge - wrong type
    assert not formatter.validate_edge("n1", "n2", 123, 1.0, "Label")

    # Invalid edge - wrong weight
    assert not formatter.validate_edge("n1", "n2", "depends_on", "invalid", "Label")

    # Invalid edge - wrong label
    assert not formatter.validate_edge("n1", "n2", "depends_on", 1.0, 123)
