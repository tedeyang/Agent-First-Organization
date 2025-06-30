"""Input modal UI component for the Arklex task graph generator.

This module provides a modal dialog interface for editing task and step descriptions.
It includes input validation and callback handling for user interactions.
"""

from collections.abc import Callable

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Static
from textual.widgets.tree import TreeNode

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
        node (TreeNode | None): The tree node being edited
        callback (Callable[[str, TreeNode | None], None] | None): Function to call after user interaction
        app (App | None): The parent app instance

    Methods:
        compose(): Creates the modal UI components
        on_button_pressed(): Handles button press events
    """

    def __init__(
        self,
        title: str,
        default: str = "",
        node: "TreeNode" | None = None,
        callback: Callable[[str, "TreeNode" | None], None] | None = None,
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
        self._app = None

    @property
    def app(self) -> "App":
        """Get the app instance."""
        return self._app

    @app.setter
    def app(self, value: "App") -> None:
        """Set the app instance."""
        self._app = value

    def compose(self) -> ComposeResult:
        """Create the modal UI components.

        Creates the visual structure of the modal dialog including title, input field,
        and action buttons.

        Yields:
            ComposeResult: The composed UI elements
        """
        yield Vertical(
            Static(self.title, classes="title"),
            Input(value=self.default, id="input-field"),
            Horizontal(
                Button("Submit", id="submit"),
                Button("Cancel", id="cancel"),
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
        if getattr(event.button, "id", None) == "submit":
            self.result = self.query_one("#input-field", Input).value
        if self.callback:
            self.callback(self.result, self.node)
        if hasattr(self, "app") and self.app and hasattr(self.app, "pop_screen"):
            self.app.pop_screen()  # Close modal
