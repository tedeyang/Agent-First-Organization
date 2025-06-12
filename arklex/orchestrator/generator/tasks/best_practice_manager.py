"""Best practice management component for the Arklex framework.

This module provides the BestPracticeManager class that handles generation and
management of best practices for task execution. It ensures quality and consistency
in task execution through practice refinement and optimization.

Key Features:
- Best practice generation from tasks
- Practice refinement through feedback
- Practice categorization and optimization
- Quality assurance and validation
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BestPractice:
    """Represents a best practice with its properties.

    Attributes:
        practice_id (str): Unique identifier for the practice
        name (str): Name of the practice
        description (str): Detailed description of the practice
        steps (List[Dict[str, Any]]): List of steps to follow
        rationale (str): Explanation of why this practice is important
        examples (List[str]): Example applications of the practice
        priority (int): Practice priority (1-5, where 5 is highest)
        category (str): Category of the practice (e.g., 'efficiency', 'quality')
    """

    practice_id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    rationale: str
    examples: List[str]
    priority: int
    category: str


class BestPracticeManager:
    """Manages the generation and refinement of best practices.

    This class handles the creation and management of best practices for task
    execution, ensuring high-quality and consistent task performance. It
    processes task information to create well-structured practice guidelines
    that can be used by the orchestrator.

    The manager follows a modular design pattern where each major responsibility
    is handled by a dedicated method, promoting maintainability and testability.

    Attributes:
        model: The language model used for practice generation
        role (str): The role or context for practice generation
        user_objective (str): User's objective for the task graph
        _practices (Dict[str, BestPractice]): Cache of generated practices
        _practice_categories (Dict[str, List[str]]): Practice categorization

    Methods:
        generate_best_practices(): Main method to generate practices
        finetune_best_practice(): Refine a practice based on feedback
        _generate_practice_definitions(): Generate practice definitions
        _validate_practices(): Validate generated practices
        _categorize_practices(): Categorize practices by type
        _optimize_practices(): Optimize practices for efficiency
    """

    def __init__(
        self,
        model: Any,
        role: str,
        user_objective: str,
    ) -> None:
        """Initialize the BestPracticeManager with required components.

        Args:
            model: The language model to use for practice generation
            role (str): The role or context for practice generation
            user_objective (str): User's objective for the task graph
        """
        self.model = model
        self.role = role
        self.user_objective = user_objective
        self._practices: Dict[str, BestPractice] = {}
        self._practice_categories: Dict[str, List[str]] = {}

    def generate_best_practices(
        self, tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate best practices for the given tasks.

        This method orchestrates the practice generation process by:
        1. Generating practice definitions
        2. Validating practices
        3. Categorizing practices
        4. Optimizing practices

        Args:
            tasks (List[Dict[str, Any]]): List of tasks to generate practices for

        Returns:
            List[Dict[str, Any]]: List of generated best practices
        """
        # Generate practice definitions
        practice_definitions = self._generate_practice_definitions(tasks)
        logger.info(f"Generated {len(practice_definitions)} practice definitions")

        # Validate practices
        validated_practices = self._validate_practices(practice_definitions)
        logger.info(f"Validated {len(validated_practices)} practices")

        # Categorize practices
        self._categorize_practices(validated_practices)
        logger.info("Categorized practices")

        # Optimize practices
        optimized_practices = self._optimize_practices(validated_practices)
        logger.info("Optimized practices")

        return optimized_practices

    def finetune_best_practice(
        self, practice: Dict[str, Any], task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Refine a best practice based on feedback and task information.

        This method takes a practice and a task, and refines the practice based on
        the task's requirements and feedback.

        Args:
            practice (Dict[str, Any]): The practice to refine
            task (Dict[str, Any]): The task information to use for refinement

        Returns:
            Dict[str, Any]: Refined practice
        """
        try:
            # Create practice definition
            practice_def = BestPractice(
                practice_id=practice.get("practice_id", ""),
                name=practice.get("name", "Refined Practice"),
                description=practice.get("description", ""),
                steps=task.get("steps", []),
                rationale=practice.get("rationale", ""),
                examples=practice.get("examples", []),
                priority=practice.get("priority", 3),
                category=practice.get("category", "refined"),
            )

            # Validate practice
            if not self._validate_practice_definition(practice_def):
                logger.warning("Invalid practice definition")
                return practice

            # Optimize steps
            optimized_steps = self._optimize_steps(practice_def.steps)
            logger.info("Optimized practice steps")

            # Update practice with optimized steps
            practice["steps"] = optimized_steps
            return practice

        except Exception as e:
            logger.error(f"Error refining practice: {str(e)}")
            return practice

    def _generate_practice_definitions(
        self, tasks: List[Dict[str, Any]]
    ) -> List[BestPractice]:
        """Generate practice definitions from tasks.

        This method creates practice definitions based on the tasks,
        ensuring proper structure and quality.

        Args:
            tasks (List[Dict[str, Any]]): List of tasks to process

        Returns:
            List[BestPractice]: List of generated practice definitions
        """
        practice_definitions = []
        for i, task in enumerate(tasks):
            practice_def = BestPractice(
                practice_id=f"practice_{i + 1}",
                name=f"Best Practice for {task['name']}",
                description=f"Best practices for executing {task['name']}",
                steps=task.get("steps", []),
                rationale="Ensures consistent and high-quality task execution",
                examples=[],
                priority=task.get("priority", 3),
                category=task.get("category", "general"),
            )
            practice_definitions.append(practice_def)
        return practice_definitions

    def _validate_practices(
        self, practice_definitions: List[BestPractice]
    ) -> List[Dict[str, Any]]:
        """Validate generated practice definitions.

        This method ensures that all practice definitions meet the required
        format and quality standards.

        Args:
            practice_definitions (List[BestPractice]): List of practice definitions

        Returns:
            List[Dict[str, Any]]: List of validated practices with the following structure:
                {
                    "practice_id": str,
                    "name": str,
                    "description": str,
                    "steps": List[Dict[str, Any]],
                    "rationale": str,
                    "examples": List[str],
                    "priority": int,
                    "category": str
                }
        """
        validated_practices = []
        for practice_def in practice_definitions:
            if self._validate_practice_definition(practice_def):
                validated_practices.append(self._convert_to_dict(practice_def))
        return validated_practices

    def _validate_practice_definition(self, practice_def: BestPractice) -> bool:
        """Validate a practice definition.

        Args:
            practice_def (BestPractice): Practice definition to validate

        Returns:
            bool: True if practice definition is valid
        """
        # Check required fields
        if not practice_def.practice_id:
            return False
        if not practice_def.name:
            return False
        if not practice_def.description:
            return False
        if not practice_def.steps:
            return False
        if not isinstance(practice_def.steps, list):
            return False
        if not practice_def.rationale:
            return False
        if not isinstance(practice_def.examples, list):
            return False
        if not isinstance(practice_def.priority, int):
            return False
        if not practice_def.category:
            return False

        return True

    def _categorize_practices(self, practices: List[Dict[str, Any]]) -> None:
        """Categorize practices by type.

        Args:
            practices (List[Dict[str, Any]]): List of practices to categorize
        """
        self._practice_categories = {}
        for practice in practices:
            category = practice.get("category", "general")
            if category not in self._practice_categories:
                self._practice_categories[category] = []
            self._practice_categories[category].append(practice["practice_id"])

    def _optimize_practices(
        self, practices: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Optimize practices for efficiency and effectiveness.

        This method analyzes and optimizes the practices to ensure they are
        efficient and effective in achieving their goals.

        Args:
            practices (List[Dict[str, Any]]): List of practices to optimize

        Returns:
            List[Dict[str, Any]]: List of optimized practices with the same structure
                as the input practices
        """
        optimized_practices = []
        for practice in practices:
            # Optimize steps
            practice["steps"] = self._optimize_steps(practice["steps"])
            optimized_practices.append(practice)
        return optimized_practices

    def _optimize_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize individual steps within a practice.

        This method analyzes and optimizes the steps to ensure they are
        clear, concise, and effective.

        Args:
            steps (List[Dict[str, Any]]): List of steps to optimize

        Returns:
            List[Dict[str, Any]]: List of optimized steps with the same structure
                as the input steps
        """
        optimized_steps = []
        for step in steps:
            # Ensure step has required fields
            if "step_id" not in step:
                step["step_id"] = f"step_{len(optimized_steps) + 1}"
            if "description" not in step:
                step["description"] = "Step description"
            if "required_fields" not in step:
                step["required_fields"] = []
            optimized_steps.append(step)
        return optimized_steps

    def _convert_to_dict(self, practice_def: BestPractice) -> Dict[str, Any]:
        """Convert a BestPractice object to a dictionary.

        Args:
            practice_def (BestPractice): Practice definition to convert

        Returns:
            Dict[str, Any]: Dictionary representation of the practice
        """
        return {
            "practice_id": practice_def.practice_id,
            "name": practice_def.name,
            "description": practice_def.description,
            "steps": practice_def.steps,
            "rationale": practice_def.rationale,
            "examples": practice_def.examples,
            "priority": practice_def.priority,
            "category": practice_def.category,
        }
