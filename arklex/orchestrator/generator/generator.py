"""Task graph generator implementation for the Arklex framework.

This module has been refactored into modular components for better maintainability.
This file now serves as a compatibility layer, importing from the new modular structure.

The new modular structure includes:
- core/generator.py: Main Generator class with orchestration logic
- ui/: Interactive components (TaskEditorApp, InputModal)
- tasks/: Task generation, best practices, and reusable tasks
- docs/: Document loading and processing
- formatting/: Task graph structure formatting

For the main Generator class, import from this module as before:
    from arklex.orchestrator.generator import Generator

For direct access to specific components:
    from arklex.orchestrator.generator.core import Generator
    from arklex.orchestrator.generator.ui import TaskEditorApp, InputModal
    from arklex.orchestrator.generator.tasks import TaskGenerator, BestPracticeManager
"""

# Import the main classes from the new modular structure
from .core import Generator

# Make UI components optional to avoid dependency issues
try:
    from .ui import TaskEditorApp, InputModal

    _UI_AVAILABLE = True
    _UI_EXPORTS = ["TaskEditorApp", "InputModal"]
except ImportError:
    _UI_AVAILABLE = False
    _UI_EXPORTS = []

# Export the main classes for backward compatibility
__all__ = ["Generator", *_UI_EXPORTS]

# The original classes have been refactored into modular components.
# All functionality is preserved in the new structure:
# - Generator class is now in core/generator.py
# - TaskEditorApp and InputModal are in ui/
# - Task generation logic is in tasks/
# - Document processing is in docs/
# - Graph formatting is in formatting/
