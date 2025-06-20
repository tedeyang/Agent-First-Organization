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


def test_format_edge_data_label_fallback() -> None:
    """Test format_edge_data falls back to type if label is empty."""
    formatter = edge_formatter.EdgeFormatter()
    source = {"id": "n1", "name": "Source"}
    target = {"id": "n2", "name": "Target"}
    result = formatter.format_edge_data(source, target, "blocks", 2.0)
    assert result["intent"] == "blocks"


@pytest.mark.parametrize(
    "etype,expected_color",
    [
        ("depends_on", "#ff0000"),
        ("blocks", "#ffa500"),
        ("related_to", "#00ff00"),
        ("part_of", "#0000ff"),
        ("unknown_type", "#808080"),
    ],
)
def test_format_edge_style_types(etype, expected_color) -> None:
    formatter = edge_formatter.EdgeFormatter()
    source = {"id": "n1"}
    target = {"id": "n2"}
    result = formatter.format_edge_style(source, target, etype)
    assert result["color"] == expected_color
    # Check all expected keys
    for key in [
        "color",
        "width",
        "style",
        "arrow_size",
        "arrow_style",
        "label_color",
        "label_font_size",
        "label_font_family",
        "label_font_weight",
        "opacity",
    ]:
        assert key in result


def test_validate_edge_metadata_and_id_variants() -> None:
    formatter = edge_formatter.EdgeFormatter()
    # Valid: source/target with only task_id
    assert formatter.validate_edge({"task_id": "n1"}, {"task_id": "n2"})
    # Valid: source/target with only id
    assert formatter.validate_edge({"id": "n1"}, {"id": "n2"})
    # Invalid: metadata is not a dict
    assert not formatter.validate_edge("n1", "n2", metadata=[1, 2, 3])
    # Valid: metadata is a dict
    assert formatter.validate_edge("n1", "n2", metadata={"foo": "bar"})
