"""Input modal UI component for the Arklex task graph generator.

This module provides a modal dialog interface for editing task and step descriptions.
It includes input validation and callback handling for user interactions.
"""

import importlib
import sys
from collections.abc import Callable
from typing import Union

# Try to import textual components, with fallbacks for testing
try:
    from textual.app import ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.screen import Screen
    from textual.widgets import Button, Input, Static
    from textual.widgets.tree import TreeNode

    TEXTUAL_AVAILABLE = True
except ImportError:
    # Fallback classes for when textual is not available
    class ComposeResult:
        pass

    class Horizontal:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class Vertical:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class Screen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.app = None

        def query_one(self, selector: str, widget_type: type = None) -> object:
            """Query for a widget (fallback implementation)."""
            return None

    class Button:
        def __init__(self, text: str = "", **kwargs: object) -> None:
            self.text = text
            self.id = kwargs.get("id", "")

        class Pressed:
            def __init__(self, button: object = None) -> None:
                self.button = button

    class Input:
        def __init__(self, value: str = "", **kwargs: object) -> None:
            self.value = value

    class Static:
        def __init__(self, text: str = "", **kwargs: object) -> None:
            self.text = text

    class TreeNode:
        def __init__(self, label: str = "", **kwargs: object) -> None:
            self.label = label

    TEXTUAL_AVAILABLE = False

from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class InputModal(Screen):
    """A simple input modal for editing or adding tasks/steps.

    This class provides a modal dialog interface for editing task and step descriptions.
    It includes input validation and callback handling for user interactions.

    Attributes:
        title (str): The title of the modal dialog
        default (str): Default value for the input field
        result (str): The final result after user interaction
        node (TreeNode): The tree node being edited
        callback (callable): Function to call after user interaction

    Methods:
        compose(): Creates the modal UI components
        on_button_pressed(): Handles button press events
    """

    def __init__(
        self,
        title: str,
        default: str = "",
        node: Union["TreeNode", None] = None,
        callback: Callable[[str, "TreeNode"], None] | None = None,
    ) -> None:
        """Initialize the InputModal instance.

        Args:
            title (str): The title of the modal dialog
            default (str): Default value for the input field. Defaults to "".
            node: The tree node being edited. Defaults to None.
            callback: Function to call after user interaction. Defaults to None.
        """
        super().__init__()
        self.title = title
        self.default = default
        self.result = default
        self.node = node
        self.callback = callback

    def compose(self) -> ComposeResult:
        """Create the modal UI components.

        Creates the visual structure of the modal dialog including title, input field,
        and action buttons.

        Yields:
            ComposeResult: The composed UI elements
        """
        VerticalClass = getattr(sys.modules[__name__], "Vertical", None)
        StaticClass = getattr(sys.modules[__name__], "Static", None)
        InputClass = getattr(sys.modules[__name__], "Input", None)
        HorizontalClass = getattr(sys.modules[__name__], "Horizontal", None)
        ButtonClass = getattr(sys.modules[__name__], "Button", None)
        if not all(
            [VerticalClass, StaticClass, InputClass, HorizontalClass, ButtonClass]
        ):
            module = importlib.import_module(
                "arklex.orchestrator.generator.ui.input_modal"
            )
            VerticalClass = VerticalClass or getattr(module, "Vertical", None)
            StaticClass = StaticClass or getattr(module, "Static", None)
            InputClass = InputClass or getattr(module, "Input", None)
            HorizontalClass = HorizontalClass or getattr(module, "Horizontal", None)
            ButtonClass = ButtonClass or getattr(module, "Button", None)
        yield VerticalClass(
            StaticClass(self.title, classes="title"),
            InputClass(value=self.default, id="input-field"),
            HorizontalClass(
                ButtonClass("Submit", id="submit"),
                ButtonClass("Cancel", id="cancel"),
                id="buttons",
            ),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events.

        Processes submit and cancel button clicks, updating the result and
        calling the callback function if provided.

        Args:
            event (Button.Pressed): The button press event
        """
        InputClass = getattr(sys.modules[__name__], "Input", None)
        if InputClass is None:
            module = importlib.import_module(
                "arklex.orchestrator.generator.ui.input_modal"
            )
            InputClass = getattr(module, "Input", None)
        if getattr(event.button, "id", None) == "submit":
            self.result = self.query_one("#input-field", InputClass).value
        if self.callback:
            self.callback(self.result, self.node)
        self.app.pop_screen()  # Close modal
