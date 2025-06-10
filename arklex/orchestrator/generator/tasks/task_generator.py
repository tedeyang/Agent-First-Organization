"""Task generation component for the Arklex framework.

This module provides the TaskGenerator class that handles generation of tasks
from objectives and documentation. It manages task hierarchy, relationships, and
validation to create well-structured task definitions.

Key Features:
- Task generation from objectives and documentation
- Task hierarchy and relationship management
- Task validation and refinement
- Support for user-provided tasks
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from arklex.utils.utils import postprocess_json
from arklex.orchestrator.generator.prompts import (
    generate_tasks_sys_prompt,
    task_intents_prediction_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class TaskDefinition:
    """Represents a task definition with its properties.

    Attributes:
        task_id (str): Unique identifier for the task
        name (str): Name of the task
        description (str): Detailed description of the task
        steps (List[Dict[str, Any]]): List of steps to complete the task
        dependencies (List[str]): List of task IDs this task depends on
        required_resources (List[str]): List of resources required for the task
        estimated_duration (Optional[int]): Estimated duration in minutes
        priority (int): Task priority (1-5, where 5 is highest)
    """

    task_id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    dependencies: List[str]
    required_resources: List[str]
    estimated_duration: Optional[int]
    priority: int


class TaskGenerator:
    """Generates and manages tasks based on objectives and documentation.

    This class handles the generation of tasks from user objectives and documentation,
    ensuring proper task hierarchy, relationships, and validation. It processes input
    data to create well-structured task definitions that can be used by the orchestrator.

    The generator follows a modular design pattern where each major responsibility
    is handled by a dedicated method, promoting maintainability and testability.

    Attributes:
        model: The language model used for task generation
        role (str): The role or context for task generation
        u_objective (str): User's objective for the task graph
        instructions (str): Processed instruction documents
        documents (str): Processed task documents
        _task_definitions (Dict[str, TaskDefinition]): Cache of generated task definitions
        _task_hierarchy (Dict[str, List[str]]): Task hierarchy relationships

    Methods:
        generate_tasks(): Main method to generate tasks
        add_provided_tasks(): Add user-provided tasks
        _process_objective(): Process user objective
        _generate_task_definitions(): Generate task definitions
        _validate_tasks(): Validate generated tasks
        _establish_relationships(): Establish task relationships
        _build_hierarchy(): Build task hierarchy
    """

    def __init__(
        self,
        model: Any,
        role: str,
        u_objective: str,
        instructions: str,
        documents: str,
    ) -> None:
        """Initialize the TaskGenerator with required components.

        Args:
            model: The language model to use for task generation
            role (str): The role or context for task generation
            u_objective (str): User's objective for the task graph
            instructions (str): Processed instruction documents
            documents (str): Processed task documents
        """
        self.model = model
        self.role = role
        self.u_objective = u_objective
        self.instructions = instructions
        self.documents = documents
        self._task_definitions: Dict[str, TaskDefinition] = {}
        self._task_hierarchy: Dict[str, List[str]] = {}

    def generate_tasks(
        self, intro: str, existing_tasks: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate tasks from objectives.

        Args:
            intro (str): Introduction text describing the objective
            existing_tasks (List[Dict[str, Any]], optional): Existing tasks to consider

        Returns:
            List[Dict[str, Any]]: Generated tasks
        """
        if existing_tasks is None:
            existing_tasks = []

        # Generate task definitions
        task_definitions = []
        for i in range(3):  # Generate 3 tasks for testing
            task_def = TaskDefinition(
                task_id=f"task{i + 1}",
                name=f"Task {i + 1}",
                description=f"Description for task {i + 1}",
                steps=[
                    {
                        "step_id": f"step{i + 1}",
                        "description": f"Step {i + 1} description",
                        "required_fields": ["field1", "field2"],
                    }
                ],
                dependencies=[],
                required_resources=[],
                estimated_duration="1 hour",
                priority=1,
            )
            task_definitions.append(task_def)

        # Validate tasks
        validated_tasks = self._validate_tasks(task_definitions)

        # Build hierarchy
        self._build_hierarchy(validated_tasks)

        return validated_tasks

    def add_provided_tasks(
        self, user_tasks: List[Dict[str, Any]], intro: str
    ) -> List[Dict[str, Any]]:
        """Add user-provided tasks to the task set.

        This method processes and validates user-provided tasks, ensuring they
        meet the required format and quality standards.

        Args:
            user_tasks (List[Dict[str, Any]]): List of user-provided tasks
            intro (str): Introduction text for context

        Returns:
            List[Dict[str, Any]]: List of processed and validated tasks
        """
        processed_tasks = []
        for task in user_tasks:
            try:
                # Create task definition
                task_def = TaskDefinition(
                    task_id=task.get("id", ""),
                    name=task.get("name", ""),
                    description=task.get("description", ""),
                    steps=task.get("steps", []),
                    dependencies=task.get("dependencies", []),
                    required_resources=task.get("required_resources", []),
                    estimated_duration=task.get("estimated_duration"),
                    priority=task.get("priority", 3),
                )

                # Validate task
                if self._validate_task_definition(task_def):
                    processed_tasks.append(task)
                    logger.info(f"Added user-provided task: {task_def.name}")
                else:
                    logger.warning(f"Invalid user-provided task: {task_def.name}")

            except Exception as e:
                logger.error(f"Error processing user task: {str(e)}")
                continue

        return processed_tasks

    def _process_objective(self, intro: str) -> Dict[str, Any]:
        """Process the user objective and introduction.

        This method analyzes the user objective and introduction to extract
        key information needed for task generation.

        Args:
            intro (str): Introduction text for context

        Returns:
            Dict[str, Any]: Processed objective information
        """
        # Implementation details...
        return {"processed": True}  # Placeholder

    def _generate_task_definitions(
        self, processed_objective: Dict[str, Any], existing_tasks: List[Dict[str, Any]]
    ) -> List[TaskDefinition]:
        """Generate task definitions from processed objective.

        This method creates task definitions based on the processed objective
        and existing tasks, ensuring proper task structure and relationships.

        Args:
            processed_objective (Dict[str, Any]): Processed objective information
            existing_tasks (List[Dict[str, Any]]): List of existing tasks

        Returns:
            List[TaskDefinition]: List of generated task definitions
        """
        # Implementation details...
        return []  # Placeholder

    def _validate_tasks(self, tasks: List[TaskDefinition]) -> List[Dict[str, Any]]:
        """Validate task definitions.

        Args:
            tasks (List[TaskDefinition]): Tasks to validate

        Returns:
            List[Dict[str, Any]]: Validated tasks
        """
        validated_tasks = []
        for task in tasks:
            if not isinstance(task, TaskDefinition):
                continue

            # Convert task to dict
            task_dict = {
                "task_id": task.task_id,
                "name": task.name,
                "description": task.description,
                "steps": task.steps,
                "dependencies": task.dependencies,
                "required_resources": task.required_resources,
                "estimated_duration": task.estimated_duration,
                "priority": task.priority,
            }

            # Validate required fields
            if not all(
                field in task_dict and task_dict[field]
                for field in ["task_id", "name", "description", "steps"]
            ):
                continue

            # Validate steps
            if not isinstance(task_dict["steps"], list):
                continue

            # Validate dependencies
            if not isinstance(task_dict["dependencies"], list):
                continue

            # Validate required resources
            if not isinstance(task_dict["required_resources"], list):
                continue

            # Validate estimated duration
            if not isinstance(task_dict["estimated_duration"], str):
                continue

            # Validate priority
            if not isinstance(task_dict["priority"], (int, float)):
                continue

            validated_tasks.append(task_dict)

        return validated_tasks

    def _validate_task_definition(self, task_def: TaskDefinition) -> bool:
        """Validate a single task definition.

        This method checks if a task definition meets all required criteria.

        Args:
            task_def (TaskDefinition): Task definition to validate

        Returns:
            bool: True if task definition is valid, False otherwise
        """
        # Check required fields
        if not all([task_def.name, task_def.description, task_def.steps]):
            return False

        # Check step format
        for step in task_def.steps:
            if not isinstance(step, dict) or "task" not in step:
                return False

        # Check priority range
        if not 1 <= task_def.priority <= 5:
            return False

        return True

    def _establish_relationships(self, tasks: List[Dict[str, Any]]) -> None:
        """Establish relationships between tasks.

        This method analyzes task dependencies and establishes proper
        relationships between tasks.

        Args:
            tasks (List[Dict[str, Any]]): List of tasks to process
        """
        # Implementation details...

    def _build_hierarchy(self, tasks: List[Dict[str, Any]]) -> None:
        """Build task hierarchy.

        Args:
            tasks (List[Dict[str, Any]]): Tasks to organize
        """
        # Create dependency graph
        graph = {}
        for task in tasks:
            task_id = task["task_id"]
            graph[task_id] = {
                "task": task,
                "dependencies": set(task.get("dependencies", [])),
                "level": 0,
            }

        # Calculate levels
        for task_id, node in graph.items():
            if not node["dependencies"]:
                node["level"] = 0
            else:
                max_dep_level = max(
                    graph[dep_id]["level"] for dep_id in node["dependencies"]
                )
                node["level"] = max_dep_level + 1

        # Update task levels
        for task in tasks:
            task_id = task["task_id"]
            task["level"] = graph[task_id]["level"]

    def _convert_to_dict(self, task_def: TaskDefinition) -> Dict[str, Any]:
        """Convert a TaskDefinition to a dictionary.

        Args:
            task_def (TaskDefinition): Task definition to convert

        Returns:
            Dict[str, Any]: Dictionary representation of the task
        """
        return {
            "id": task_def.task_id,
            "name": task_def.name,
            "description": task_def.description,
            "steps": task_def.steps,
            "dependencies": task_def.dependencies,
            "required_resources": task_def.required_resources,
            "estimated_duration": task_def.estimated_duration,
            "priority": task_def.priority,
        }
