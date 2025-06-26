"""Tests for the generator components.

This module provides comprehensive tests for the task generation, best practices,
and reusable task components of the Arklex framework.
"""

import dataclasses
from typing import Dict, List

import pytest
from unittest.mock import Mock
from unittest.mock import MagicMock

from arklex.orchestrator.generator.tasks.task_generator import (
    TaskGenerator,
    TaskDefinition,
)
from arklex.orchestrator.generator.tasks.best_practice_manager import (
    BestPracticeManager,
    BestPractice,
)
from arklex.orchestrator.generator.tasks.reusable_task_manager import (
    ReusableTaskManager,
    ReusableTask,
)


# --- Mock Classes ---


class FakeResponse:
    """Mock response class for testing."""

    content = '[{"task": "Test", "intent": "Test intent", "description": "Test description", "steps": [{"task": "Step 1", "description": "Test step description"}]}]'
    text = content


class FakeGeneration:
    """Mock generation class for testing."""

    def __init__(self):
        self.text = '[{"task": "Test", "intent": "Test intent", "description": "Test description", "steps": [{"task": "Step 1", "description": "Test step description"}]}]'


class FakeGenerationResponse:
    """Mock generation response class for testing."""

    def __init__(self):
        self.generations = [[FakeGeneration()]]


class FakeResourceResponse:
    """Mock resource response class for testing."""

    content = '[{"step": 1, "task": "Step 1", "resource": "MessageWorker", "example_response": "Test response", "description": "Test step description"}]'
    text = content


# --- Fixtures ---


@pytest.fixture
def always_valid_mock_model() -> Mock:
    """Create a mock language model that always returns valid responses."""
    mock_model = Mock()
    mock_model.generate.return_value = FakeGenerationResponse()
    mock_model.invoke.return_value = FakeResourceResponse()
    return mock_model


@pytest.fixture
def sample_objective() -> Dict:
    """Create a sample objective for testing."""
    return {
        "description": "Create a new product in the store",
        "requirements": ["Product details", "Pricing information"],
        "constraints": ["Must follow store guidelines"],
    }


@pytest.fixture
def sample_tasks() -> List[Dict]:
    """Create sample tasks for testing."""
    return [
        {
            "task_id": "task1",
            "name": "Gather product details",
            "description": "Collect all required product information",
            "steps": [
                {
                    "task": "Get product name",
                    "description": "Get the name of the product.",
                },
                {
                    "task": "Get product description",
                    "description": "Get the description of the product.",
                },
            ],
            "dependencies": [],
            "required_resources": ["Product form"],
            "estimated_duration": "30 minutes",
            "priority": 1,
        },
        {
            "task_id": "task2",
            "name": "Set product pricing",
            "description": "Determine product pricing strategy",
            "steps": [
                {
                    "task": "Research market prices",
                    "description": "Research market prices to determine the best pricing strategy.",
                },
                {
                    "task": "Set final price",
                    "description": "Set the final price based on the research.",
                },
            ],
            "dependencies": ["task1"],
            "required_resources": ["Pricing guide"],
            "estimated_duration": "45 minutes",
            "priority": 2,
        },
    ]


@pytest.fixture
def sample_practices() -> List[BestPractice]:
    """Create sample best practices for testing."""
    return [
        BestPractice(
            practice_id="practice1",
            name="Product validation",
            description="Validate product information before submission",
            steps=[
                {
                    "task": "Check required fields",
                    "description": "Check all required fields are filled out.",
                },
                {
                    "task": "Verify data format",
                    "description": "Verify the data format is correct.",
                },
            ],
            rationale="Ensure data quality",
            examples=["Product name validation", "Price format check"],
            priority=1,
            category="validation",
        )
    ]


@pytest.fixture
def task_generator(always_valid_mock_model: Mock) -> TaskGenerator:
    """Create a TaskGenerator instance for testing."""
    return TaskGenerator(
        model=always_valid_mock_model,
        role="product_manager",
        user_objective="Create a new product",
        instructions=[],
        documents=[],
    )


@pytest.fixture
def best_practice_manager(always_valid_mock_model: Mock) -> BestPracticeManager:
    """Create a BestPracticeManager instance for testing."""
    return BestPracticeManager(
        model=always_valid_mock_model,
        role="product_manager",
        user_objective="Create a new product",
    )


@pytest.fixture
def reusable_task_manager(always_valid_mock_model: Mock) -> ReusableTaskManager:
    """Create a ReusableTaskManager instance for testing."""
    return ReusableTaskManager(
        model=always_valid_mock_model,
        role="product_manager",
        user_objective="Create a new product",
    )


@pytest.fixture
def reusable_template() -> ReusableTask:
    """Create a reusable task template for testing."""
    return ReusableTask(
        template_id="tmpl1",
        name="Reusable Template",
        description="Reusable template description",
        parameters={"param1": "string", "param2": "string"},
        steps=[
            {"task": "Step 1", "description": "Step 1 desc"},
            {"task": "Step 2", "description": "Step 2 desc"},
        ],
        examples=[],
        version="1.0",
        category="test",
    )


@pytest.fixture
def sample_tasks_dict() -> List[Dict]:
    """Sample tasks as dicts for reusable task manager tests."""
    return [
        {
            "task_id": "task1",
            "name": "Gather product details",
            "description": "Collect all required product information",
            "steps": [
                {
                    "task": "Get product name",
                    "description": "Get the name of the product.",
                },
                {
                    "task": "Get product description",
                    "description": "Get the description of the product.",
                },
            ],
            "dependencies": [],
            "required_resources": ["Product form"],
            "estimated_duration": "30 minutes",
            "priority": 1,
        },
        {
            "task_id": "task2",
            "name": "Set product pricing",
            "description": "Determine product pricing strategy",
            "steps": [
                {
                    "task": "Research market prices",
                    "description": "Research market prices to determine the best pricing strategy.",
                },
                {
                    "task": "Set final price",
                    "description": "Set the final price based on the research.",
                },
            ],
            "dependencies": ["task1"],
            "required_resources": ["Pricing guide"],
            "estimated_duration": "45 minutes",
            "priority": 2,
        },
    ]


@pytest.fixture
def sample_practices_dict() -> List[Dict]:
    """Sample best practices as dicts for best practice manager tests."""
    return [
        {
            "practice_id": "practice1",
            "name": "Product validation",
            "description": "Validate product information before submission",
            "steps": [
                {
                    "task": "Check required fields",
                    "description": "Check all required fields are filled out.",
                },
                {
                    "task": "Verify data format",
                    "description": "Verify the data format is correct.",
                },
            ],
            "rationale": "Ensure data quality",
            "examples": ["Product name validation", "Price format check"],
            "priority": 1,
            "category": "validation",
        }
    ]


# --- Test Classes ---


class TestTaskGenerator:
    """Test TaskGenerator core logic and validation."""

    def test_generate_tasks(self, task_generator: TaskGenerator) -> None:
        """Should generate a list of tasks from objectives."""
        tasks = task_generator.generate_tasks(
            intro="Create a new product", existing_tasks=[]
        )
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        assert all(isinstance(task, dict) for task in tasks)
        assert all("id" in task for task in tasks)

    def test_add_provided_tasks(self, task_generator: TaskGenerator) -> None:
        """Should add user-provided tasks to the task list."""
        user_tasks = [
            {
                "name": "Custom task",
                "description": "User-defined task",
                "steps": [
                    {"task": "Custom step", "description": "Custom step description"}
                ],
            }
        ]
        tasks = task_generator.add_provided_tasks(
            user_tasks=user_tasks, intro="Add custom task"
        )
        assert isinstance(tasks, list)
        assert len(tasks) == 1
        assert tasks[0]["name"] == "Custom task"

    def test_validate_tasks(self, task_generator: TaskGenerator) -> None:
        """Should validate and convert TaskDefinition objects to dicts."""
        task_definitions = [
            TaskDefinition(
                task_id="test1",
                name="Test task",
                description="Test description",
                steps=[{"task": "Test step", "description": "Test step description"}],
                dependencies=[],
                required_resources=[],
                estimated_duration="1 hour",
                priority=1,
            )
        ]
        validated_tasks = task_generator._validate_tasks(task_definitions)
        assert isinstance(validated_tasks, list)
        assert len(validated_tasks) == 1
        assert validated_tasks[0]["id"] == "test1"

    def test_establish_relationships(
        self, task_generator: TaskGenerator, sample_tasks_dict: List[Dict]
    ) -> None:
        """Should establish dependencies between tasks."""
        tasks = [dict(t) for t in sample_tasks_dict]
        task_generator._establish_relationships(tasks)
        assert tasks[1]["dependencies"] == ["task1"]

    def test_build_hierarchy(
        self, task_generator: TaskGenerator, sample_tasks_dict: List[Dict]
    ) -> None:
        """Should build hierarchy levels for tasks."""
        tasks = [dict(t) for t in sample_tasks_dict]
        task_generator._build_hierarchy(tasks)
        assert all("level" in task for task in tasks)

    def test_convert_to_task_definitions(self) -> None:
        """Test _convert_to_task_definitions method."""
        generator = TaskGenerator(
            model=Mock(),
            role="test_role",
            user_objective="test_objective",
            instructions="test_instructions",
            documents="test_documents",
        )

        tasks_with_steps = [
            {
                "task": "Task 1",
                "intent": "intent_1",
                "steps": [
                    {"task": "Step 1", "description": "First step"},
                    {"task": "Step 2", "description": "Second step"},
                ],
                "dependencies": ["dep1"],
                "required_resources": ["resource1"],
                "estimated_duration": "1 hour",
                "priority": 5,
            },
            {
                "task": "Task 2",
                "intent": "intent_2",
                "steps": [{"task": "Step 3", "description": "Third step"}],
                "dependencies": [],
                "required_resources": [],
                "estimated_duration": "30 minutes",
                "priority": 3,
            },
        ]

        task_definitions = generator._convert_to_task_definitions(tasks_with_steps)

        assert len(task_definitions) == 2
        assert task_definitions[0].task_id == "task_1"
        assert task_definitions[0].name == "Task 1"
        assert task_definitions[0].description == "intent_1"
        assert len(task_definitions[0].steps) == 2
        assert task_definitions[0].dependencies == ["dep1"]
        assert task_definitions[0].required_resources == ["resource1"]
        assert task_definitions[0].estimated_duration == "1 hour"
        assert task_definitions[0].priority == 5

        assert task_definitions[1].task_id == "task_2"
        assert task_definitions[1].name == "Task 2"
        assert task_definitions[1].description == "intent_2"
        assert len(task_definitions[1].steps) == 1
        assert task_definitions[1].dependencies == []
        assert task_definitions[1].required_resources == []
        assert task_definitions[1].estimated_duration == "30 minutes"
        assert task_definitions[1].priority == 3

    def test_process_objective_with_response_parsing(self) -> None:
        """Test _process_objective with response parsing logic."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.generations = [[Mock()]]
        mock_response.generations[0][
            0
        ].text = '[{"task": "Task 1", "intent": "intent_1"}]'
        mock_model.generate.return_value = mock_response

        generator = TaskGenerator(
            model=mock_model,
            role="test_role",
            user_objective="test_objective",
            instructions="test_instructions",
            documents="test_documents",
        )

        result = generator._process_objective(
            objective="test_objective", intro="test_intro", docs="test_docs"
        )

        assert "tasks" in result
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["task"] == "Task 1"
        assert result["tasks"][0]["intent"] == "intent_1"

    def test_process_objective_with_message_content(self) -> None:
        """Test _process_objective with message.content response format."""
        mock_model = Mock()
        mock_response = Mock()

        class Message:
            def __init__(self, content):
                self.content = content

        class Generation:
            def __init__(self, message):
                self.message = message

        content_string = '[{"task": "Task 2", "intent": "intent_2"}]'
        message = Message(content_string)
        generation = Generation(message)
        mock_response.generations = [[generation]]
        mock_model.generate.return_value = mock_response

        generator = TaskGenerator(
            model=mock_model,
            role="test_role",
            user_objective="test_objective",
            instructions="test_instructions",
            documents="test_documents",
        )

        result = generator._process_objective(
            objective="test_objective", intro="test_intro", docs="test_docs"
        )

        assert "tasks" in result
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["task"] == "Task 2"
        assert result["tasks"][0]["intent"] == "intent_2"

    def test_process_objective_with_dict_response(self) -> None:
        """Test _process_objective with dict response format."""
        mock_model = Mock()
        mock_response = {"text": '[{"task": "Task 3", "intent": "intent_3"}]'}
        mock_model.generate.return_value = mock_response

        generator = TaskGenerator(
            model=mock_model,
            role="test_role",
            user_objective="test_objective",
            instructions="test_instructions",
            documents="test_documents",
        )

        result = generator._process_objective(
            objective="test_objective", intro="test_intro", docs="test_docs"
        )

        assert "tasks" in result
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["task"] == "Task 3"
        assert result["tasks"][0]["intent"] == "intent_3"

    def test_process_objective_with_content_dict_response(self) -> None:
        """Test _process_objective with content dict response format."""
        mock_model = Mock()
        mock_response = {"content": '[{"task": "Task 4", "intent": "intent_4"}]'}
        mock_model.generate.return_value = mock_response

        generator = TaskGenerator(
            model=mock_model,
            role="test_role",
            user_objective="test_objective",
            instructions="test_instructions",
            documents="test_documents",
        )

        result = generator._process_objective(
            objective="test_objective", intro="test_intro", docs="test_docs"
        )

        assert "tasks" in result
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["task"] == "Task 4"
        assert result["tasks"][0]["intent"] == "intent_4"

    def test_process_objective_with_str_response(self) -> None:
        """Test _process_objective with string response format."""
        mock_model = Mock()
        mock_response = '[{"task": "Task 5", "intent": "intent_5"}]'
        mock_model.generate.return_value = mock_response

        generator = TaskGenerator(
            model=mock_model,
            role="test_role",
            user_objective="test_objective",
            instructions="test_instructions",
            documents="test_documents",
        )

        result = generator._process_objective(
            objective="test_objective", intro="test_intro", docs="test_docs"
        )

        assert "tasks" in result
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["task"] == "Task 5"
        assert result["tasks"][0]["intent"] == "intent_5"


class TestBestPracticeManager:
    """Test BestPracticeManager core logic and validation."""

    def test_generate_best_practices(
        self, best_practice_manager: BestPracticeManager, sample_tasks_dict: List[Dict]
    ) -> None:
        """Should generate best practices from tasks."""
        practices = best_practice_manager.generate_best_practices(sample_tasks_dict)
        assert isinstance(practices, list)
        assert len(practices) > 0
        assert all(isinstance(practice, dict) for practice in practices)
        assert all("practice_id" in practice for practice in practices)

    def test_finetune_best_practice(
        self, best_practice_manager: BestPracticeManager
    ) -> None:
        """Should refine a best practice with a new task."""
        practice = {
            "practice_id": "test1",
            "name": "Test Practice",
            "description": "Test Description",
            "steps": [
                {"task": "Original step", "description": "Original step description"}
            ],
            "rationale": "Test Rationale",
            "examples": [],
            "priority": 3,
            "category": "test",
        }
        task = {
            "name": "Test Task",
            "steps": [{"task": "New step", "description": "New step description"}],
        }
        refined_practice = best_practice_manager.finetune_best_practice(practice, task)
        assert isinstance(refined_practice, dict)
        assert "steps" in refined_practice
        assert len(refined_practice["steps"]) > 0

    def test_validate_practices(self, best_practice_manager, sample_practices) -> None:
        """Should validate and convert BestPractice objects to dicts."""
        validated = best_practice_manager._validate_practices(sample_practices)
        assert isinstance(validated, list)
        assert len(validated) == 1
        assert validated[0]["practice_id"] == "practice1"

    def test_categorize_practices(
        self, best_practice_manager, sample_practices_dict
    ) -> None:
        """Should categorize best practices by category."""
        dict_practices = [dict(p) for p in sample_practices_dict]
        best_practice_manager._categorize_practices(dict_practices)
        assert isinstance(best_practice_manager._practice_categories, dict)
        assert "validation" in best_practice_manager._practice_categories

    def test_optimize_practices(
        self, best_practice_manager, sample_practices_dict
    ) -> None:
        """Should optimize best practices."""
        dict_practices = [dict(p) for p in sample_practices_dict]
        optimized = best_practice_manager._optimize_practices(dict_practices)
        assert isinstance(optimized, list)
        assert len(optimized) == 1

    def test_generate_practice_definitions(self) -> None:
        """Test _generate_practice_definitions method."""
        mock_model = MagicMock()
        manager = BestPracticeManager(
            model=mock_model, role="test_role", user_objective="test_objective"
        )

        tasks = [
            {
                "name": "Task 1",
                "steps": [{"description": "Step 1"}],
                "priority": 5,
                "category": "efficiency",
            },
            {
                "name": "Task 2",
                "steps": [{"description": "Step 2"}],
                "priority": 3,
                "category": "quality",
            },
        ]

        practice_definitions = manager._generate_practice_definitions(tasks)

        assert len(practice_definitions) == 2
        assert practice_definitions[0].name == "Best Practice for Task 1"
        assert practice_definitions[0].priority == 5
        assert practice_definitions[0].category == "efficiency"
        assert practice_definitions[1].name == "Best Practice for Task 2"
        assert practice_definitions[1].priority == 3
        assert practice_definitions[1].category == "quality"

    def test_optimize_steps_with_invalid_description(self) -> None:
        """Test _optimize_steps with invalid description."""
        mock_model = MagicMock()
        manager = BestPracticeManager(
            model=mock_model, role="test_role", user_objective="test_objective"
        )

        steps = [
            {"step_id": "step1", "description": None},
            {"step_id": "step2", "description": 123},
            {"step_id": "step3", "description": ""},
            {"step_id": "step4", "description": "Valid description"},
        ]

        optimized_steps = manager._optimize_steps(steps)

        assert len(optimized_steps) == 4
        assert optimized_steps[0]["description"] == "Step 1"  # Default for None
        assert optimized_steps[1]["description"] == "123"  # String conversion
        assert optimized_steps[2]["description"] == "Step 3"  # Default for empty
        assert optimized_steps[3]["description"] == "Valid description"  # Valid
        assert optimized_steps[0]["step_id"] == "step1"
        assert optimized_steps[1]["step_id"] == "step2"
        assert optimized_steps[2]["step_id"] == "step3"
        assert optimized_steps[3]["step_id"] == "step4"

    def test_optimize_steps_with_missing_step_id(self) -> None:
        """Test _optimize_steps with missing step_id."""
        mock_model = MagicMock()
        manager = BestPracticeManager(
            model=mock_model, role="test_role", user_objective="test_objective"
        )

        steps = [
            {"description": "Step 1"},  # Missing step_id
            {"step_id": "step2", "description": "Step 2"},
        ]

        optimized_steps = manager._optimize_steps(steps)

        assert len(optimized_steps) == 2
        assert optimized_steps[0]["step_id"] == "step_1"  # Generated step_id
        assert optimized_steps[1]["step_id"] == "step2"  # Existing step_id
        assert optimized_steps[0]["description"] == "Step 1"
        assert optimized_steps[1]["description"] == "Step 2"

    def test_optimize_steps_with_missing_required_fields(self) -> None:
        """Test _optimize_steps with missing required_fields."""
        mock_model = MagicMock()
        manager = BestPracticeManager(
            model=mock_model, role="test_role", user_objective="test_objective"
        )

        steps = [
            {"step_id": "step1", "description": "Step 1"}  # Missing required_fields
        ]

        optimized_steps = manager._optimize_steps(steps)

        assert len(optimized_steps) == 1
        assert optimized_steps[0]["step_id"] == "step1"
        assert optimized_steps[0]["description"] == "Step 1"
        assert optimized_steps[0]["required_fields"] == []  # Default empty list


class TestReusableTaskManager:
    """Test ReusableTaskManager core logic and validation."""

    def test_generate_reusable_tasks(
        self, reusable_task_manager, sample_tasks_dict
    ) -> None:
        """Should generate reusable task templates from tasks."""
        templates = reusable_task_manager.generate_reusable_tasks(sample_tasks_dict)
        assert isinstance(templates, dict)
        assert len(templates) > 0
        assert all(isinstance(template, dict) for template in templates.values())

    def test_instantiate_template(
        self, reusable_task_manager, reusable_template
    ) -> None:
        """Should instantiate a reusable task template."""
        reusable_task_manager._templates[reusable_template.template_id] = (
            reusable_template
        )
        params = {"param1": "value1", "param2": "value2"}
        instantiated = reusable_task_manager.instantiate_template(
            reusable_template.template_id, params
        )
        assert isinstance(instantiated, dict)
        assert instantiated["template_id"] == "tmpl1"

    def test_validate_templates(self, reusable_task_manager, reusable_template) -> None:
        """Should validate reusable task templates."""
        templates = {reusable_template.template_id: reusable_template}
        validated = reusable_task_manager._validate_templates(templates)
        assert isinstance(validated, dict)
        assert "tmpl1" in validated

    def test_validate_parameters(
        self, reusable_task_manager, reusable_template
    ) -> None:
        """Should validate parameters for reusable task templates."""
        params = {"param1": "value1", "param2": "value2"}
        valid = reusable_task_manager._validate_parameters(reusable_template, params)
        assert valid is True

    def test_categorize_templates(
        self, reusable_task_manager, reusable_template
    ) -> None:
        """Should categorize reusable task templates by category."""
        templates = {
            reusable_template.template_id: dataclasses.asdict(reusable_template)
        }
        reusable_task_manager._categorize_templates(templates)
        assert isinstance(reusable_task_manager._template_categories, dict)
        assert "test" in reusable_task_manager._template_categories


def test_integration_generation_pipeline(always_valid_mock_model: Mock) -> None:
    """Integration test for the full generation pipeline."""
    task_gen = TaskGenerator(
        model=always_valid_mock_model,
        role="product_manager",
        user_objective="Create a new product",
        instructions=[],
        documents=[],
    )
    tasks = task_gen.generate_tasks(intro="Create a new product", existing_tasks=[])

    best_mgr = BestPracticeManager(
        model=always_valid_mock_model,
        role="product_manager",
        user_objective="Create a new product",
    )
    practices = best_mgr.generate_best_practices(tasks)

    reusable_mgr = ReusableTaskManager(
        model=always_valid_mock_model,
        role="product_manager",
        user_objective="Create a new product",
    )
    reusable = reusable_mgr.generate_reusable_tasks(tasks)

    assert isinstance(tasks, list)
    assert isinstance(practices, list)
    assert isinstance(reusable, dict)
