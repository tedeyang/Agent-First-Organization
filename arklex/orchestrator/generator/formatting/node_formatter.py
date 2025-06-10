"""Node formatting component for the Arklex framework.

This module provides the NodeFormatter class that handles formatting of task
nodes in the task graph.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class NodeFormatter:
    """Formats task nodes.

    This class handles the formatting of task nodes in the task graph,
    including data and style attributes.

    Attributes:
        _required_attributes (List[str]): Required node attributes
        _optional_attributes (List[str]): Optional node attributes
    """

    def __init__(self) -> None:
        """Initialize the NodeFormatter."""
        self._required_attributes = ["id", "name", "description"]
        self._optional_attributes = [
            "steps",
            "dependencies",
            "required_resources",
            "estimated_duration",
            "priority",
        ]

    def format_node(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Format a task node.

        Args:
            task (Dict[str, Any]): Task to format

        Returns:
            Dict[str, Any]: Formatted node
        """
        # Create base node
        node = {
            "id": task.get("id", task.get("task_id", "")),
            "name": task.get("name", ""),
            "description": task.get("description", ""),
            "type": "task",
            "data": self.format_node_data(task),
            "style": self.format_node_style(task),
        }

        # Add optional attributes
        for attr in self._optional_attributes:
            if attr in task:
                node[attr] = task[attr]

        return node

    def format_node_data(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Format node data.

        Args:
            task (Dict[str, Any]): Task to format

        Returns:
            Dict[str, Any]: Formatted node data
        """
        return {
            "task_id": task.get("task_id", ""),
            "name": task.get("name", ""),
            "description": task.get("description", ""),
            "steps": task.get("steps", []),
            "dependencies": task.get("dependencies", []),
            "required_resources": task.get("required_resources", []),
            "estimated_duration": task.get("estimated_duration", ""),
            "priority": task.get("priority", "medium"),
            "created_at": datetime.now().isoformat(),
            "modified_at": datetime.now().isoformat(),
            "version": "1.0",
        }

    def format_node_style(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Format node style.

        Args:
            task (Dict[str, Any]): Task to format

        Returns:
            Dict[str, Any]: Formatted node style
        """
        # Get priority color
        priority_colors = {"high": "#ff0000", "medium": "#ffa500", "low": "#00ff00"}
        priority = task.get("priority", "medium")
        color = priority_colors.get(priority, "#808080")

        return {
            "color": color,
            "border": {"color": "#000000", "width": 2, "style": "solid", "radius": 4},
            "padding": {"top": 10, "right": 10, "bottom": 10, "left": 10},
            "text_color": "#000000",
            "background_color": "#ffffff",
            "opacity": 1.0,
        }

    def validate_node(self, node: Dict[str, Any]) -> bool:
        """Validate a node.

        Args:
            node (Dict[str, Any]): Node to validate

        Returns:
            bool: True if node is valid
        """
        # Require 'id' and 'type' at the top level
        if "id" not in node or not isinstance(node["id"], str):
            return False
        if "type" not in node or not isinstance(node["type"], str):
            return False
        # Check 'data' for 'name' and 'description'
        if "data" in node:
            data = node["data"]
            if not isinstance(data, dict):
                return False
            if "name" in data and not isinstance(data["name"], str):
                return False
            if "description" in data and not isinstance(data["description"], str):
                return False
            if "priority" in data and not isinstance(data["priority"], (str, int)):
                return False
        return True

    def _validate_attribute(self, value: Any, attr: str) -> bool:
        """Validate a node attribute.

        Args:
            value (Any): Attribute value to validate
            attr (str): Attribute name

        Returns:
            bool: True if attribute is valid
        """
        if attr == "steps":
            if not isinstance(value, list):
                return False
            for step in value:
                if not isinstance(step, dict):
                    return False
                if "step_id" not in step:
                    return False
                if "description" not in step:
                    return False
                if "order" not in step:
                    return False
        elif attr == "dependencies":
            if not isinstance(value, list):
                return False
            for dep in value:
                if not isinstance(dep, str):
                    return False
        elif attr == "required_resources":
            if not isinstance(value, list):
                return False
            for resource in value:
                if not isinstance(resource, str):
                    return False
        elif attr == "estimated_duration":
            if not isinstance(value, str):
                return False
        elif attr == "priority":
            if not isinstance(value, str):
                return False
            if value not in ["high", "medium", "low"]:
                return False

        return True
