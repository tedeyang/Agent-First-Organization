"""Edge formatting component for the Arklex framework.

This module provides the EdgeFormatter class that handles formatting of task
graph edges.
"""

from typing import Dict, Any, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EdgeFormatter:
    """Formats task graph edges.

    This class handles the formatting of task graph edges, including data
    and style attributes.

    Attributes:
        _required_attributes (List[str]): Required edge attributes
        _optional_attributes (List[str]): Optional edge attributes
        _valid_types (List[str]): Valid edge types
    """

    def __init__(self) -> None:
        """Initialize the EdgeFormatter."""
        self._required_attributes = ["source", "target", "type"]
        self._optional_attributes = ["weight", "label", "metadata"]
        self._valid_types = ["depends_on", "blocks", "related_to", "part_of"]

    def format_edge(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any],
        type: str = "depends_on",
        weight: float = 1.0,
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Format a task graph edge.

        Args:
            source (Dict[str, Any]): Source node dict
            target (Dict[str, Any]): Target node dict
            type (str): Edge type
            weight (float): Edge weight
            label (str): Edge label
            metadata (Optional[Dict[str, Any]]): Additional metadata

        Returns:
            Dict[str, Any]: Formatted edge
        """
        source_id = source.get("task_id") or source.get("id")
        target_id = target.get("task_id") or target.get("id")
        edge_id = f"{source_id}_{target_id}"
        edge = {
            "id": edge_id,
            "source": source_id,
            "target": target_id,
            "type": type,
            "data": self.format_edge_data(
                source, target, type=type, weight=weight, label=label, metadata=metadata
            ),
            "style": self.format_edge_style(
                source, target, type=type, weight=weight, label=label, metadata=metadata
            ),
        }
        if weight is not None:
            edge["weight"] = weight
        if label:
            edge["label"] = label
        if metadata:
            edge["metadata"] = metadata
        return edge

    def format_edge_data(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any],
        type: str = "depends_on",
        weight: float = 1.0,
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Format edge data.

        Args:
            source (Dict[str, Any]): Source node dict
            target (Dict[str, Any]): Target node dict
            type (str): Edge type
            weight (float): Edge weight
            label (str): Edge label
            metadata (Optional[Dict[str, Any]]): Additional metadata

        Returns:
            Dict[str, Any]: Formatted edge data
        """
        source_id = source.get("task_id") or source.get("id")
        target_id = target.get("task_id") or target.get("id")
        return {
            "source": source_id,
            "target": target_id,
            "type": type,
            "weight": weight,
            "label": label,
            "description": f"{source.get('name', source_id)} {type.replace('_', ' ')} {target.get('name', target_id)}",
            "created_at": datetime.now().isoformat(),
            "modified_at": datetime.now().isoformat(),
            "version": "1.0",
            **(metadata or {}),
        }

    def format_edge_style(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any],
        type: str = "depends_on",
        weight: float = 1.0,
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Format edge style.

        Args:
            source (Dict[str, Any]): Source node dict
            target (Dict[str, Any]): Target node dict
            type (str): Edge type
            weight (float): Edge weight
            label (str): Edge label
            metadata (Optional[Dict[str, Any]]): Additional metadata

        Returns:
            Dict[str, Any]: Formatted edge style
        """
        type_colors = {
            "depends_on": "#ff0000",
            "blocks": "#ffa500",
            "related_to": "#00ff00",
            "part_of": "#0000ff",
        }
        color = type_colors.get(type, "#808080")
        return {
            "color": color,
            "width": 2,
            "style": "solid",
            "arrow_size": 10,
            "arrow_style": "triangle",
            "label_color": "#000000",
            "label_font_size": 12,
            "label_font_family": "Arial",
            "label_font_weight": "normal",
            "opacity": 1.0,
        }

    def validate_edge(
        self,
        source: Union[str, Dict[str, Any]],
        target: Union[str, Dict[str, Any]],
        type: str = "depends_on",
        weight: float = 1.0,
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Validate an edge.

        Args:
            source (Union[str, Dict[str, Any]]): Source node ID or dict
            target (Union[str, Dict[str, Any]]): Target node ID or dict
            type (str): Edge type
            weight (float): Edge weight
            label (str): Edge label
            metadata (Optional[Dict[str, Any]]): Additional metadata

        Returns:
            bool: True if edge is valid
        """
        # Handle string inputs
        if isinstance(source, str):
            source_id = source
        else:
            source_id = source.get("task_id") or source.get("id")
            if not source_id:
                return False

        if isinstance(target, str):
            target_id = target
        else:
            target_id = target.get("task_id") or target.get("id")
            if not target_id:
                return False

        # Validate edge type
        if not isinstance(type, str):
            return False

        # Validate weight
        if not isinstance(weight, (int, float)):
            return False

        # Validate label
        if not isinstance(label, str):
            return False

        # Validate metadata if provided
        if metadata is not None and not isinstance(metadata, dict):
            return False

        return True
