"""Task editor UI component for the Arklex framework.

This module provides the TaskEditor class that handles the UI for editing
task definitions.
"""

from collections.abc import Callable
from typing import Any

from textual.app import App, ComposeResult
from textual.events import Key
from textual.widgets import Label, Tree
from textual.widgets.tree import TreeNode

from arklex.utils.logging_utils import LogContext

from .input_modal import InputModal

log_context = LogContext(__name__)


class TaskEditorApp(App):
    """A Textual app to edit tasks and steps in a hierarchical structure.

    This class provides a text-based user interface for editing tasks and their steps.
    It supports adding, editing, and deleting tasks and steps in a tree structure.

    Attributes:
        tasks (list): List of task dictionaries containing task names and steps
        task_tree (Tree | None): The tree widget displaying tasks and steps, initialized as None

    Methods:
        compose(): Creates the main UI components
        on_mount(): Initializes the UI after mounting
        on_tree_node_selected(): Handles node selection events
        on_key(): Processes keyboard input
        action_add_node(): Adds new nodes to the tree
        show_input_modal(): Displays the input modal dialog
        update_tasks(): Updates the tasks list from the tree structure
        run(): Runs the app and returns the updated tasks list
    """

    def __init__(self, tasks: list[dict[str, Any]]) -> None:
        """Initialize the TaskEditorApp instance.

        Args:
            tasks (List[Dict[str, Any]]): List of task dictionaries containing task names and steps
        """
        super().__init__()
        self.tasks = tasks
        self.task_tree = None

    def compose(self) -> ComposeResult:
        """Create the main UI components.

        Creates the tree widget and populates it with tasks and steps, along with
        instruction labels for user interaction.

        Yields:
            ComposeResult: The composed UI elements
        """
        self.task_tree = Tree("Tasks")
        self.task_tree.root.expand()
        tasks = self.tasks if self.tasks is not None else []
        for task in tasks:
            task_node = self.task_tree.root.add(task["name"])
            if "steps" in task and task["steps"]:
                for step in task["steps"]:
                    if isinstance(step, dict):
                        step_text = step.get("description", str(step))
                    else:
                        step_text = str(step)
                    task_node.add_leaf(step_text)
        yield self.task_tree
        yield Label(
            "Use 'a' to add nodes, 'd' to delete, 's' to save and exit, arrow keys to navigate"
        )

    def on_mount(self) -> None:
        """Initialize the UI after mounting.

        Sets focus to the task tree widget to enable keyboard navigation.
        """
        self.task_tree.focus()

    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle node selection events.

        When a tree node is selected, shows an input modal for editing the node's label.

        Args:
            event (Tree.NodeSelected): The node selection event
        """
        selected_node = event.node

        def handle_modal_result(result: str, node: TreeNode) -> None:
            """Handle the result from the input modal.

            Args:
                result (str): The result from the modal input
                node (TreeNode): The tree node being edited
            """
            if result and result.strip():
                node.set_label(result.strip())

        # Get the current label value
        if hasattr(selected_node.label, "plain"):
            current_label = selected_node.label.plain
        else:
            current_label = str(selected_node.label)

        self.show_input_modal(
            "Edit node", current_label, selected_node, handle_modal_result
        )

    async def on_key(self, event: Key) -> None:
        """Process keyboard input.

        Handles keyboard shortcuts for adding nodes ('a'), deleting nodes ('d'),
        and saving and exiting ('s').

        Args:
            event (Key): The keyboard event
        """
        if event.key == "a":
            await self.action_add_node(self.task_tree.cursor_node)
        elif event.key == "d":
            if self.task_tree.cursor_node and self.task_tree.cursor_node.parent:
                self.task_tree.cursor_node.remove()
        elif event.key == "s":
            await self.update_tasks()
            self.exit(self.tasks)

    async def action_add_node(self, node: TreeNode) -> None:
        """Add a new node to the tree.

        Shows an input modal to get the new node's label and adds it as a child
        of the currently selected node.

        Args:
            node (TreeNode): The parent node to add the new node to
        """
        if not node:
            return

        def handle_modal_result(result: str, parent_node: TreeNode) -> None:
            """Handle the result from the input modal.

            Args:
                result (str): The result from the modal input
                parent_node (TreeNode): The parent node for the new node
            """
            if result and result.strip():
                parent_node.add(result.strip())

        self.show_input_modal("Add new node", "", node, handle_modal_result)

    def push_screen(self, screen: "InputModal") -> None:
        """Push a screen to the app.

        Args:
            screen (InputModal): The screen to push
        """
        super().push_screen(screen)

    def show_input_modal(
        self,
        title: str,
        default: str = "",
        node: TreeNode | None = None,
        callback: Callable[[str, TreeNode | None], None] | None = None,
    ) -> str:
        """Show an input modal dialog.

        Creates and displays an input modal with the given title and default value.
        The actual result will be handled by the callback function when the user
        submits or cancels the modal.

        Args:
            title (str): The title of the modal
            default (str): The default value for the input field
            node (TreeNode | None): The tree node being edited
            callback (Callable[[str, TreeNode | None], None] | None): Callback function

        Returns:
            str: The default value (the actual result is handled by the callback)
        """
        modal = InputModal(title, default, node, callback)
        self.push_screen(modal)
        # Return the default value as the immediate result
        # The actual result will be handled by the callback
        return default

    async def update_tasks(self) -> None:
        """Update the tasks list from the tree structure.

        Traverses the tree structure and updates the tasks list to reflect
        the current state of the tree.
        """
        if not self.task_tree or not self.task_tree.root:
            return

        updated_tasks = []
        for task_node in self.task_tree.root.children:
            # Handle different label formats
            if hasattr(task_node.label, "plain"):
                task_name = task_node.label.plain
            else:
                task_name = str(task_node.label)

            task = {"name": task_name}
            steps = []
            for step_node in task_node.children:
                # Handle different label formats for steps too
                if hasattr(step_node.label, "plain"):
                    step_name = step_node.label.plain
                else:
                    step_name = str(step_node.label)
                steps.append(step_name)
            if steps:
                task["steps"] = steps
            updated_tasks.append(task)

        self.tasks = updated_tasks

    def run(self) -> list[dict[str, Any]]:
        """Run the task editor app.

        Starts the app and returns the updated tasks list.

        Returns:
            List[Dict[str, Any]]: The updated tasks list
        """
        super().run()
        return self.tasks
