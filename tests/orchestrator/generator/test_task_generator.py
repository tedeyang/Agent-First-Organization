import pytest
from unittest.mock import Mock
from arklex.orchestrator.generator.tasks.task_generator import (
    TaskGenerator,
    TaskDefinition,
)


@pytest.fixture
def mock_model():
    return Mock()


@pytest.fixture
def task_generator(mock_model):
    return TaskGenerator(
        model=mock_model,
        role="test_role",
        user_objective="test objective",
        instructions="test instructions",
        documents="test documents",
    )


@pytest.fixture
def sample_tasks():
    return [
        {"task": "Task 1", "intent": "intent 1", "steps": [{"task": "Step 1"}]},
        {"task": "Task 2", "intent": "intent 2", "steps": [{"task": "Step 2"}]},
    ]


@pytest.fixture
def sample_task_definition():
    return TaskDefinition(
        task_id="task_1",
        name="Test Task",
        description="Test description",
        steps=[{"task": "Test step"}],
        dependencies=[],
        required_resources=[],
        estimated_duration="1 hour",
        priority=3,
    )


class TestTaskDefinition:
    """Test the TaskDefinition dataclass."""

    def test_task_definition_creation(self) -> None:
        """Test TaskDefinition creation with all fields."""
        task_def = TaskDefinition(
            task_id="test_id",
            name="Test Task",
            description="Test description",
            steps=[{"task": "step1"}],
            dependencies=["dep1"],
            required_resources=["resource1"],
            estimated_duration="2 hours",
            priority=5,
        )

        assert task_def.task_id == "test_id"
        assert task_def.name == "Test Task"
        assert task_def.description == "Test description"
        assert task_def.steps == [{"task": "step1"}]
        assert task_def.dependencies == ["dep1"]
        assert task_def.required_resources == ["resource1"]
        assert task_def.estimated_duration == "2 hours"
        assert task_def.priority == 5

    def test_task_definition_optional_fields(self) -> None:
        """Test TaskDefinition creation with optional fields."""
        task_def = TaskDefinition(
            task_id="test_id",
            name="Test Task",
            description="Test description",
            steps=[],
            dependencies=[],
            required_resources=[],
            estimated_duration=None,
            priority=1,
        )

        assert task_def.estimated_duration is None


class TestTaskGenerator:
    """Test the TaskGenerator class."""

    def test_initialization(self, mock_model) -> None:
        """Test TaskGenerator initialization."""
        generator = TaskGenerator(
            model=mock_model,
            role="test_role",
            user_objective="test objective",
            instructions="test instructions",
            documents="test documents",
        )

        assert generator.model == mock_model
        assert generator.role == "test_role"
        assert generator.user_objective == "test objective"
        assert generator.instructions == "test instructions"
        assert generator.documents == "test documents"
        assert generator._task_definitions == {}
        assert generator._task_hierarchy == {}
        assert generator.prompt_manager is not None

    def test_generate_tasks_basic(self, task_generator, sample_tasks) -> None:
        """Test basic task generation."""
        task_generator._generate_high_level_tasks = Mock(return_value=sample_tasks)
        task_generator._check_task_breakdown_original = Mock(return_value=False)
        task_generator._convert_to_task_definitions = Mock(return_value=[])
        task_generator._validate_tasks = Mock(return_value=sample_tasks)
        task_generator._build_hierarchy = Mock()

        result = task_generator.generate_tasks("test intro")

        assert result == sample_tasks
        task_generator._generate_high_level_tasks.assert_called_once_with(
            "test intro", []
        )
        task_generator._validate_tasks.assert_called_once()
        task_generator._build_hierarchy.assert_called_once()

    def test_generate_tasks_with_existing_tasks(
        self, task_generator, sample_tasks
    ) -> None:
        """Test task generation with existing tasks."""
        existing_tasks = [{"task": "existing task"}]
        task_generator._generate_high_level_tasks = Mock(return_value=sample_tasks)
        task_generator._check_task_breakdown_original = Mock(return_value=False)
        task_generator._convert_to_task_definitions = Mock(return_value=[])
        task_generator._validate_tasks = Mock(return_value=sample_tasks)
        task_generator._build_hierarchy = Mock()

        result = task_generator.generate_tasks("test intro", existing_tasks)

        assert result == sample_tasks
        task_generator._generate_high_level_tasks.assert_called_once_with(
            "test intro", existing_tasks
        )

    def test_generate_tasks_with_breakdown_needed(self, task_generator) -> None:
        """Test task generation when breakdown is needed."""
        high_level_tasks = [{"task": "Complex Task", "intent": "complex intent"}]
        steps = [{"task": "Step 1"}, {"task": "Step 2"}]

        task_generator._generate_high_level_tasks = Mock(return_value=high_level_tasks)
        task_generator._check_task_breakdown_original = Mock(return_value=True)
        task_generator._generate_task_steps_original = Mock(return_value=steps)
        task_generator._convert_to_task_definitions = Mock(return_value=[])
        task_generator._validate_tasks = Mock(return_value=high_level_tasks)
        task_generator._build_hierarchy = Mock()

        result = task_generator.generate_tasks("test intro")

        assert result == high_level_tasks
        task_generator._check_task_breakdown_original.assert_called_once_with(
            "Complex Task", "complex intent"
        )
        task_generator._generate_task_steps_original.assert_called_once_with(
            "Complex Task", "complex intent"
        )

    def test_add_provided_tasks(self, task_generator):
        """Test adding user-provided tasks."""
        user_tasks = [
            {
                "task": "User Task 1",
                "intent": "user intent 1",
                "steps": [{"task": "Step 1"}],
            },
            {
                "task": "User Task 2",
                "intent": "user intent 2",
                "steps": [{"task": "Step 2"}],
            },
        ]

        # Mock _validate_task_definition to return True for valid tasks
        task_generator._validate_task_definition = Mock(return_value=True)
        task_generator._process_objective = Mock(return_value={"processed": "data"})
        task_generator._validate_tasks = Mock(return_value=user_tasks)
        task_generator._build_hierarchy = Mock()

        result = task_generator.add_provided_tasks(user_tasks, "test intro")

        assert result == user_tasks

    def test_add_provided_tasks_with_invalid_task(self, task_generator):
        """Test adding user-provided tasks with invalid task."""
        user_tasks = [
            {
                "task": "Valid Task",
                "intent": "valid intent",
                "steps": [{"task": "Step 1"}],
                "name": "Valid Task",
                "description": "A valid task",
            },
            {"invalid": "task"},  # Invalid task
        ]

        # Mock _validate_task_definition to return True for valid tasks, False for invalid
        def mock_validate(task_def):
            return task_def.name == "Valid Task"

        task_generator._validate_task_definition = Mock(side_effect=mock_validate)
        task_generator._process_objective = Mock(return_value={"processed": "data"})
        task_generator._validate_tasks = Mock(
            return_value=[user_tasks[0]]
        )  # Only valid task
        task_generator._build_hierarchy = Mock()

        result = task_generator.add_provided_tasks(user_tasks, "test intro")

        assert len(result) == 1

    def test_process_objective(self, task_generator) -> None:
        """Test _process_objective method."""
        objective = "test objective"
        intro = "test intro"
        docs = "test docs"
        existing_tasks = [{"task": "existing"}]

        # Mock the model response structure
        mock_response = Mock()
        mock_response.generations = [[Mock()]]
        mock_response.generations[0][0].text = "processed objective"
        task_generator.model.generate.return_value = mock_response

        result = task_generator._process_objective(
            objective, intro, docs, existing_tasks
        )

        assert result is not None
        task_generator.model.generate.assert_called_once()

    def test_generate_high_level_tasks(self, task_generator) -> None:
        """Test _generate_high_level_tasks method."""
        intro = "test intro"
        existing_tasks = [{"task": "existing"}]

        mock_response = '{"tasks": [{"task": "Task 1", "intent": "intent 1"}]}'
        task_generator.model.invoke.return_value = mock_response

        result = task_generator._generate_high_level_tasks(intro, existing_tasks)

        assert len(result) == 1
        assert result[0]["task"] == "Task 1"
        assert result[0]["intent"] == "intent 1"

    def test_check_task_breakdown_original_true(self, task_generator) -> None:
        """Test _check_task_breakdown_original returns True."""
        task_name = "Complex Task"
        task_intent = "complex intent"

        # Return a string with extra text before and after the JSON, with 'answer': 'yes'
        response_str = 'Some text before {"answer": "yes"} some text after'
        task_generator.model.invoke.return_value = response_str

        result = task_generator._check_task_breakdown_original(task_name, task_intent)
        assert result is True

    def test_check_task_breakdown_original_false(self, task_generator) -> None:
        """Test _check_task_breakdown_original returns False."""
        task_name = "Simple Task"
        task_intent = "simple intent"

        mock_response = '{"needs_breakdown": false}'
        task_generator.model.invoke.return_value = mock_response

        result = task_generator._check_task_breakdown_original(task_name, task_intent)

        assert result is False

    def test_generate_task_steps_original(self, task_generator) -> None:
        """Test _generate_task_steps_original method."""
        task_name = "Test Task"
        task_intent = "test intent"

        mock_response = '{"steps": [{"task": "Step 1"}, {"task": "Step 2"}]}'
        task_generator.model.invoke.return_value = mock_response

        result = task_generator._generate_task_steps_original(task_name, task_intent)

        assert len(result) == 2
        assert result[0]["task"] == "Step 1"
        assert result[1]["task"] == "Step 2"

    def test_convert_to_task_definitions(self, task_generator, sample_tasks) -> None:
        """Test _convert_to_task_definitions method."""
        result = task_generator._convert_to_task_definitions(sample_tasks)

        assert len(result) == 2
        assert all(isinstance(task_def, TaskDefinition) for task_def in result)
        assert result[0].name == "Task 1"
        assert result[1].name == "Task 2"

    def test_validate_tasks(self, task_generator, sample_tasks) -> None:
        """Test _validate_tasks method."""
        # Update sample_tasks to include required fields
        tasks = [
            {
                "task": "Task 1",
                "intent": "intent 1",
                "steps": [{"task": "Step 1"}],
                "name": "Task 1",
                "description": "intent 1",
            },
            {
                "task": "Task 2",
                "intent": "intent 2",
                "steps": [{"task": "Step 2"}],
                "name": "Task 2",
                "description": "intent 2",
            },
        ]
        task_generator._validate_task_definition = Mock(return_value=True)

        result = task_generator._validate_tasks(tasks)

        assert result == tasks

    def test_validate_task_definition_valid(
        self, task_generator, sample_task_definition
    ) -> None:
        """Test _validate_task_definition with valid task."""
        result = task_generator._validate_task_definition(sample_task_definition)

        assert result is True

    def test_validate_task_definition_invalid(self, task_generator) -> None:
        """Test _validate_task_definition with invalid task."""
        invalid_task = TaskDefinition(
            task_id="",
            name="",
            description="",
            steps=[],
            dependencies=[],
            required_resources=[],
            estimated_duration=None,
            priority=0,
        )

        result = task_generator._validate_task_definition(invalid_task)

        assert result is False

    def test_establish_relationships(self, task_generator) -> None:
        """Test _establish_relationships method."""
        tasks = [
            {"id": "task_1", "dependencies": ["task_2"]},
            {"id": "task_2", "dependencies": []},
        ]

        task_generator._establish_relationships(tasks)

        # This method doesn't return anything, just verify it doesn't raise an exception
        assert True

    def test_build_hierarchy(self, task_generator) -> None:
        """Test _build_hierarchy method."""
        tasks = [
            {"id": "task_1", "dependencies": ["task_2"]},
            {"id": "task_2", "dependencies": []},
        ]

        task_generator._build_hierarchy(tasks)

        # This method doesn't return anything, just verify it doesn't raise an exception
        assert True

    def test_convert_to_dict(self, task_generator, sample_task_definition) -> None:
        """Test _convert_to_dict method."""
        result = task_generator._convert_to_dict(sample_task_definition)

        assert isinstance(result, dict)
        assert result["task_id"] == "task_1"
        assert result["name"] == "Test Task"
        assert result["description"] == "Test description"
        assert result["steps"] == [{"task": "Test step"}]
        assert result["dependencies"] == []
        assert result["required_resources"] == []
        assert result["estimated_duration"] == "1 hour"
        assert result["priority"] == 3

    def test_convert_to_task_dict(self, task_generator) -> None:
        """Test _convert_to_task_dict method."""
        task_definitions = [
            {"task": "Task 1", "intent": "Intent 1", "steps": []},
            {"task": "Task 2", "intent": "Intent 2", "steps": []},
        ]

        result = task_generator._convert_to_task_dict(task_definitions)

        assert isinstance(result, dict)
        assert "task_1" in result
        assert "task_2" in result

    def test_generate_tasks_with_model_error(self, task_generator) -> None:
        """Test generate_tasks handles model errors gracefully."""
        task_generator._generate_high_level_tasks = Mock(
            side_effect=Exception("Model error")
        )

        with pytest.raises(Exception):
            task_generator.generate_tasks("test intro")

    def test_generate_tasks_with_json_error(self, task_generator) -> None:
        """Test generate_tasks handles JSON parsing errors."""
        task_generator._generate_high_level_tasks = Mock(
            return_value=[{"invalid": "json"}]
        )
        task_generator._check_task_breakdown_original = Mock(return_value=False)
        task_generator._convert_to_task_definitions = Mock(return_value=[])
        task_generator._validate_tasks = Mock(return_value=[])
        task_generator._build_hierarchy = Mock()

        result = task_generator.generate_tasks("test intro")

        assert result == []

    def test_add_provided_tasks_with_processing_error(self, task_generator) -> None:
        """Test add_provided_tasks handles processing errors."""
        user_tasks = [{"task": "Valid Task", "intent": "valid intent", "steps": []}]

        # Mock _validate_task_definition to return False, which should cause an exception
        task_generator._validate_task_definition = Mock(return_value=False)
        task_generator._process_objective = Mock(
            side_effect=Exception("Processing error")
        )

        # The method should handle the exception and continue processing
        result = task_generator.add_provided_tasks(user_tasks, "test intro")

        # Should return empty list since validation fails
        assert result == []
