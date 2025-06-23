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
=======
"""Tests for the EdgeFormatter class.

This module provides comprehensive tests for the EdgeFormatter class,
including edge formatting, validation, and style management functionality.
"""

from typing import Dict, Any
import pytest

from arklex.orchestrator.generator.formatting.edge_formatter import EdgeFormatter


# --- Fixtures ---


@pytest.fixture
def edge_formatter() -> EdgeFormatter:
    """Create an EdgeFormatter instance for testing."""
    return EdgeFormatter(default_intent="depends_on", default_weight=2)


@pytest.fixture
def sample_source() -> Dict[str, Any]:
    """Create a sample source task for testing."""
    return {"task_id": "source_1", "id": "source_1", "name": "Source Task"}


@pytest.fixture
def sample_target() -> Dict[str, Any]:
    """Create a sample target task for testing."""
    return {"task_id": "target_1", "id": "target_1", "name": "Target Task"}


@pytest.fixture
def invalid_source() -> Dict[str, Any]:
    """Create an invalid source task (missing ID) for testing."""
    return {"name": "No ID"}


@pytest.fixture
def invalid_target() -> Dict[str, Any]:
    """Create an invalid target task (missing ID) for testing."""
    return {"name": "No ID"}


@pytest.fixture
def edge_types() -> Dict[str, str]:
    """Define edge types and their expected colors for testing."""
    return {
        "depends_on": "#ff0000",
        "blocks": "#ffa500",
        "related_to": "#00ff00",
        "part_of": "#0000ff",
        "custom_type": "#808080",  # Unknown type defaults to gray
    }


# --- Test Classes ---


class TestEdgeFormatting:
    """Test suite for edge formatting functionality."""

    def test_format_edge_style_types(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: Dict[str, Any],
        sample_target: Dict[str, Any],
        edge_types: Dict[str, str],
    ) -> None:
        """Test format_edge_style with different edge types."""
        for edge_type, expected_color in edge_types.items():
            style = edge_formatter.format_edge_style(
                sample_source, sample_target, type=edge_type
            )
            assert style["color"] == expected_color, (
                f"Failed for edge type: {edge_type}"
            )

    def test_format_edge_style_structure(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: Dict[str, Any],
        sample_target: Dict[str, Any],
    ) -> None:
        """Test the structure of format_edge_style output."""
        style = edge_formatter.format_edge_style(sample_source, sample_target)

        # Check all required style properties exist
        required_properties = [
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
        ]

        for prop in required_properties:
            assert prop in style, f"Missing property: {prop}"

        # Check data types
        assert isinstance(style["width"], int)
        assert isinstance(style["opacity"], float)


class TestEdgeValidation:
    """Test suite for edge validation functionality."""

    def test_validate_edge_valid_inputs(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: Dict[str, Any],
        sample_target: Dict[str, Any],
    ) -> None:
        """Test validate_edge with various valid inputs."""
        # Test with dict inputs
        assert (
            edge_formatter.validate_edge(
                sample_source,
                sample_target,
                type="depends_on",
                weight=1.0,
                label="",
                metadata=None,
            )
            is True
        )

        # Test with string inputs
        assert (
            edge_formatter.validate_edge(
                "source_1",
                "target_1",
                type="blocks",
                weight=2,
                label="label",
                metadata={},
            )
            is True
        )

    def test_validate_edge_invalid_source(
        self,
        edge_formatter: EdgeFormatter,
        invalid_source: Dict[str, Any],
        sample_target: Dict[str, Any],
    ) -> None:
        """Test validate_edge with invalid source input."""
        assert edge_formatter.validate_edge(invalid_source, sample_target) is False

    def test_validate_edge_invalid_target(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: Dict[str, Any],
        invalid_target: Dict[str, Any],
    ) -> None:
        """Test validate_edge with invalid target input."""
        assert edge_formatter.validate_edge(sample_source, invalid_target) is False

    def test_validate_edge_invalid_type(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: Dict[str, Any],
        sample_target: Dict[str, Any],
    ) -> None:
        """Test validate_edge with invalid type input."""
        assert (
            edge_formatter.validate_edge(sample_source, sample_target, type=123)
            is False
        )

    def test_validate_edge_invalid_weight(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: Dict[str, Any],
        sample_target: Dict[str, Any],
    ) -> None:
        """Test validate_edge with invalid weight input."""
        assert (
            edge_formatter.validate_edge(
                sample_source, sample_target, weight="not_a_number"
            )
            is False
        )

    def test_validate_edge_invalid_label(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: Dict[str, Any],
        sample_target: Dict[str, Any],
    ) -> None:
        """Test validate_edge with invalid label input."""
        assert (
            edge_formatter.validate_edge(sample_source, sample_target, label=123)
            is False
        )

    def test_validate_edge_invalid_metadata(
        self,
        edge_formatter: EdgeFormatter,
        sample_source: Dict[str, Any],
        sample_target: Dict[str, Any],
    ) -> None:
        """Test validate_edge with invalid metadata input."""
        assert (
            edge_formatter.validate_edge(
                sample_source, sample_target, metadata="not_a_dict"
            )
            is False
        )

    def test_validate_edge_with_string_ids(self, edge_formatter: EdgeFormatter) -> None:
        """Test validate_edge with string source and target IDs."""
        assert edge_formatter.validate_edge("source_id", "target_id") is True

    def test_validate_edge_with_missing_ids(
        self, edge_formatter: EdgeFormatter
    ) -> None:
        """Test validate_edge with dicts missing 'id' and 'task_id'."""
        source = {"foo": "bar"}
        target = {"baz": "qux"}
        assert edge_formatter.validate_edge(source, target) is False
