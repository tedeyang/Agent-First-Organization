"""Task graph generator module for the Arklex framework.

This module provides a comprehensive framework for generating task graphs and managing
task hierarchies. It includes classes for task editing, task generation, and handling
reusable tasks.

The module has been refactored into specialized components:
- Core: Main orchestration logic (Generator class)
- UI: Interactive components for task editing
- Tasks: Task generation, best practices, and reusable tasks
- Docs: Document loading and processing
- Formatting: Task graph structure formatting

Key Components:
- Generator: Main class for creating task graphs based on user objectives and documentation
- TaskEditorApp: Text-based UI for editing tasks and their steps
- InputModal: Modal dialog for editing task and step descriptions

Features:
- Natural language task generation
- Interactive task editing
- Reusable task management
- Best practice integration
- Documentation processing
- Resource initialization
- Task graph formatting
- Configuration management

Usage:
    from arklex.orchestrator.generator import Generator
    from arklex.env.env import DefaultResourceInitializer

    # Initialize generator
    config = {
        "role": "customer_service",
        "user_objective": "Handle customer inquiries",
        "builder_objective": "Create efficient response system",
        "instructions": [...],
        "tasks": [...],
        "workers": [...],
        "tools": [...]
    }

    generator = Generator(
        config=config,
        model=language_model,
        output_dir="output",
        resource_inizializer=DefaultResourceInitializer()
    )

    # Generate task graph
    task_graph = generator.generate()

    # Save task graph
    output_path = generator.save_task_graph(task_graph)
"""

import importlib.util
import os

from . import core, docs, formatting, tasks
from .core import Generator


def _create_placeholder_classes() -> tuple[type, type]:
    """Create placeholder classes when UI components are not available.

    Returns:
        Tuple[Type, Type]: Tuple of (TaskEditorApp, InputModal) placeholder classes
    """

    class TaskEditorApp:
        """Placeholder class when UI components are not available."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "TaskEditorApp requires 'textual' package to be installed"
            )

    class InputModal:
        """Placeholder class when UI components are not available."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("InputModal requires 'textual' package to be installed")

    return TaskEditorApp, InputModal


def _should_force_ui_import_error() -> bool:
    """Check if UI import error should be forced for testing purposes.

    Returns:
        bool: True if UI import error should be forced
    """
    return os.environ.get("ARKLEX_FORCE_UI_IMPORT_ERROR") == "1"


def _import_ui_components() -> tuple[type, type, list[str]]:
    """Import UI components or create placeholders if not available.

    Returns:
        Tuple[Type, Type, List[str]]: Tuple of (TaskEditorApp, InputModal, ui_components_list)
    """
    if _should_force_ui_import_error():
        # Create placeholder classes when UI dependencies are not available (test hook)
        TaskEditorApp, InputModal = _create_placeholder_classes()
        return TaskEditorApp, InputModal, []

    try:
        # Check if textual is available
        if importlib.util.find_spec("textual") is None:
            raise ImportError("textual package not found")

        from .ui import InputModal, TaskEditorApp

        return TaskEditorApp, InputModal, ["TaskEditorApp", "InputModal"]
    except (ImportError, Exception):
        # Create placeholder classes when UI dependencies are not available
        # Handle both ImportError and other exceptions (like AttributeError, etc.)
        TaskEditorApp, InputModal = _create_placeholder_classes()
        return TaskEditorApp, InputModal, []


# Import UI components or create placeholders
TaskEditorApp, InputModal, _UI_COMPONENTS = _import_ui_components()

# Determine if ui module should be included in __all__
_UI_MODULE_AVAILABLE = len(_UI_COMPONENTS) > 0

# Import ui module if available
if _UI_MODULE_AVAILABLE:
    from . import ui
else:
    # Create a placeholder ui module with proper structure
    class _PlaceholderUIModule:
        """Placeholder UI module when UI components are not available."""

        def __init__(self) -> None:
            # Create placeholder submodules that tests expect
            self.task_editor = _PlaceholderTaskEditorModule()
            self.input_modal = _PlaceholderInputModalModule()

    class _PlaceholderTaskEditorModule:
        """Placeholder task_editor submodule."""

        def __getattr__(self, name: str) -> object:
            if name == "TaskEditorApp":
                return TaskEditorApp
            elif name == "InputModal":
                return InputModal
            elif name == "log_context":
                # Return a mock logger for tests
                import logging

                return logging.getLogger("arklex.orchestrator.generator.ui.task_editor")
            elif name == "Tree":
                # Return a mock Tree class for tests
                class MockTree:
                    pass

                return MockTree
            elif name == "Label":
                # Return a mock Label class for tests
                class MockLabel:
                    pass

                return MockLabel
            elif name == "TreeNode":
                # Return a mock TreeNode class for tests
                class MockTreeNode:
                    pass

                return MockTreeNode
            else:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )

    class _PlaceholderInputModalModule:
        """Placeholder input_modal submodule."""

        def __getattr__(self, name: str) -> object:
            if name == "InputModal":
                return InputModal
            elif name == "Screen":
                # Return a mock Screen class for tests
                class MockScreen:
                    pass

                return MockScreen
            elif name == "Button":
                # Return a mock Button class for tests
                class MockButton:
                    pass

                return MockButton
            elif name == "Input":
                # Return a mock Input class for tests
                class MockInput:
                    pass

                return MockInput
            elif name == "Static":
                # Return a mock Static class for tests
                class MockStatic:
                    pass

                return MockStatic
            else:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )

    ui = _PlaceholderUIModule()

__all__ = [
    "Generator",
    *_UI_COMPONENTS,
    "core",
    "ui",
    "tasks",
    "docs",
    "formatting",
]
