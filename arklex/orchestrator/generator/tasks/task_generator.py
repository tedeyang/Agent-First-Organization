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
import json

from arklex.utils.utils import postprocess_json
from arklex.orchestrator.generator.prompts import (
    generate_tasks_sys_prompt,
    task_intents_prediction_prompt,
)
from arklex.orchestrator.generator.prompts import PromptManager

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
        estimated_duration (Optional[str]): Estimated duration as a string (e.g., "1 hour")
        priority (int): Task priority (1-5, where 5 is highest)
    """

    task_id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    dependencies: List[str]
    required_resources: List[str]
    estimated_duration: Optional[str]
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
        user_objective (str): User's objective for the task graph
        instructions (str): Processed instruction documents
        documents (str): Processed task documents
        _task_definitions (Dict[str, TaskDefinition]): Cache of generated task definitions
        _task_hierarchy (Dict[str, List[str]]): Task hierarchy relationships
        prompt_manager: The prompt manager for generating prompts

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
        user_objective: str,
        instructions: str,
        documents: str,
    ) -> None:
        """Initialize the TaskGenerator with required components.

        Args:
            model: The language model to use for task generation
            role (str): The role or context for task generation
            user_objective (str): User's objective for the task graph
            instructions (str): Processed instruction documents
            documents (str): Processed task documents
        """
        self.model = model
        self.role = role
        self.user_objective = user_objective
        self.instructions = instructions
        self.documents = documents
        self._task_definitions: Dict[str, TaskDefinition] = {}
        self._task_hierarchy: Dict[str, List[str]] = {}
        self.prompt_manager = PromptManager()  # Initialize prompt manager

    def generate_tasks(
        self, intro: str, existing_tasks: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate tasks from the introduction and existing tasks."""
        if existing_tasks is None:
            existing_tasks = []

        # Extract objective, intro, and docs from the intro string or config
        # For now, treat intro as both intro and docs, and use self.objective if available
        objective = getattr(self, "objective", intro)
        docs = intro

        processed_objective = self._process_objective(
            objective, intro, docs, existing_tasks
        )
        task_definitions = self._generate_task_definitions(
            processed_objective, existing_tasks
        )

        # Convert to dict and add 'id' field
        tasks = []
        for i, task_def in enumerate(task_definitions):
            task_dict = self._convert_to_dict(task_def)
            task_dict["id"] = f"task_{i + 1}"
            tasks.append(task_dict)

        validated_tasks = self._validate_tasks(tasks)
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

    def _process_objective(
        self,
        objective: str,
        intro: str,
        docs: str,
        existing_tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Process objective and documentation to generate tasks.

        Args:
            objective: User objective
            intro: Introduction text
            docs: Documentation text
            existing_tasks: Optional list of existing tasks

        Returns:
            Dictionary containing processed objective and tasks
        """
        # Format existing tasks as JSON if provided
        existing_tasks_json = ""
        if existing_tasks:
            existing_tasks_json = json.dumps(existing_tasks, indent=2)

        # Generate tasks using the model
        prompt = self.prompt_manager.generate_tasks_sys_prompt.format(
            role=self.role,
            u_objective=objective,
            intro=intro,
            docs=docs,
            instructions=self.instructions,
            existing_tasks=existing_tasks_json,
        )

        try:
            from langchain_core.messages import HumanMessage

            messages = [HumanMessage(content=prompt)]
        except ImportError:
            messages = [{"role": "user", "content": prompt}]

        response = self.model.generate([messages])
        # Extract text content from LLMResult
        if hasattr(response.generations[0][0], "text"):
            response_text = response.generations[0][0].text
        elif hasattr(response.generations[0][0], "message") and hasattr(
            response.generations[0][0].message, "content"
        ):
            response_text = response.generations[0][0].message.content
        else:
            response_text = str(response)

        # Extract tasks from response
        try:
            # Find JSON array in response
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                tasks_json = response_text[json_start:json_end]
                tasks = json.loads(tasks_json)
                logger.info(f"Generated {len(tasks)} tasks from documentation")
                return {"processed": True, "tasks": tasks}
            else:
                logger.error("No valid JSON array found in response")
                return {"processed": False, "tasks": []}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tasks from response: {e}")
            return {"processed": False, "tasks": []}

    def _generate_task_definitions(
        self,
        processed_objective: Dict[str, Any],
        existing_tasks: List[Dict[str, Any]] = None,
    ) -> List[TaskDefinition]:
        """Generate task definitions from documentation.

        Args:
            processed_objective: The processed objective containing task data
            existing_tasks: Optional list of existing tasks to avoid duplication

        Returns:
            List of TaskDefinition objects
        """
        tasks_data = processed_objective.get("tasks", [])

        # If no tasks found in processed_objective, generate from documentation
        if not tasks_data:
            # Format existing tasks for the prompt
            existing_tasks_json = "[]"
            if existing_tasks:
                existing_tasks_json = json.dumps(
                    [
                        {"intent": task.get("intent", ""), "task": task.get("name", "")}
                        for task in existing_tasks
                    ],
                    indent=4,
                )

            # Generate tasks from documentation
            prompt = self.prompt_manager.generate_tasks_sys_prompt.format(
                role=processed_objective.get("role", ""),
                u_objective=processed_objective.get("objective", ""),
                intro=processed_objective.get("intro", ""),
                docs=processed_objective.get("docs", ""),
                instructions=processed_objective.get("instructions", ""),
                existing_tasks=existing_tasks_json,
            )

            response = self.model.invoke(prompt)
            try:
                # Extract content from AIMessage if needed
                if hasattr(response, "content"):
                    response_text = response.content
                else:
                    response_text = str(response)

                # Find JSON array in the response
                json_start = response_text.find("[")
                json_end = response_text.rfind("]") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    tasks_data = json.loads(json_str)
                    logger.info(f"Generated {len(tasks_data)} tasks from documentation")
                else:
                    logger.error("No valid JSON array found in response")
                    tasks_data = []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse generated tasks: {e}")
                tasks_data = []

        # Convert task data to TaskDefinition objects
        task_definitions = []
        for i, task_data in enumerate(tasks_data):
            # Extract steps from task data
            steps = task_data.get("steps", [])
            if not steps and "task" in task_data:
                # Create a default step if none provided
                steps = [{"task": f"Execute {task_data['task']}"}]
            elif isinstance(steps, list) and steps and isinstance(steps[0], str):
                # Convert string steps to dictionaries
                steps = [{"task": step} for step in steps]

            task_def = TaskDefinition(
                task_id=f"task_{i + 1}",  # Generate unique task ID
                name=task_data.get("task", ""),
                description=task_data.get("intent", ""),  # Use intent as description
                steps=steps,
                dependencies=task_data.get("dependencies", []),
                required_resources=task_data.get("required_resources", []),
                estimated_duration=task_data.get("estimated_duration", "1 hour"),
                priority=task_data.get("priority", 3),  # Use integer priority
            )
            task_definitions.append(task_def)

        return task_definitions

    def _validate_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate task definitions.

        Args:
            tasks: List of tasks to validate

        Returns:
            List of validated tasks
        """
        validated_tasks = []
        for task in tasks:
            # Validate required fields
            if not all(field in task for field in ["name", "description", "steps"]):
                logger.warning(f"Task missing required fields: {task}")
                continue

            # Validate steps format
            if not isinstance(task["steps"], list):
                logger.warning(f"Task steps must be a list: {task}")
                continue

            # Convert steps to proper format if needed
            if task["steps"] and isinstance(task["steps"][0], str):
                task["steps"] = [{"task": step} for step in task["steps"]]

            # Validate step format
            if not all(
                isinstance(step, dict) and "task" in step for step in task["steps"]
            ):
                logger.warning(
                    f"Task steps must be dictionaries with 'task' key: {task}"
                )
                continue

            # Validate dependencies
            if "dependencies" not in task:
                task["dependencies"] = []
            elif not isinstance(task["dependencies"], list):
                task["dependencies"] = []

            # Validate required resources
            if "required_resources" not in task:
                task["required_resources"] = []
            elif not isinstance(task["required_resources"], list):
                task["required_resources"] = []

            # Validate estimated duration
            if "estimated_duration" not in task:
                task["estimated_duration"] = "1 hour"
            elif not isinstance(task["estimated_duration"], str):
                task["estimated_duration"] = "1 hour"

            # Validate priority
            if "priority" not in task:
                task["priority"] = 3
            elif not isinstance(task["priority"], (int, float)):
                task["priority"] = 3
            elif not 1 <= task["priority"] <= 5:
                task["priority"] = 3

            validated_tasks.append(task)

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
            tasks: List of tasks to organize
        """
        # Create dependency graph
        graph = {}
        for task in tasks:
            task_id = task["id"]
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
                # Find maximum level of dependencies
                max_dep_level = 0
                for dep_id in node["dependencies"]:
                    if dep_id in graph:
                        max_dep_level = max(max_dep_level, graph[dep_id]["level"])
                node["level"] = max_dep_level + 1

        # Update task levels
        for task in tasks:
            task_id = task["id"]
            if task_id in graph:
                task["level"] = graph[task_id]["level"]
            else:
                task["level"] = 0

        # Sort tasks by level
        tasks.sort(key=lambda x: x["level"])

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

    def _convert_to_task_dict(
        self, task_definitions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Convert task definitions to task dictionary format."""
        tasks = {}
        for task_def in task_definitions:
            task_id = f"task_{len(tasks) + 1}"
            tasks[task_id] = {
                "name": task_def["task"],
                "description": task_def["intent"],
                "steps": task_def.get("steps", []),
                "dependencies": [],
                "required_resources": [],
                "estimated_duration": "1 hour",
                "priority": "high",
            }
        return tasks
