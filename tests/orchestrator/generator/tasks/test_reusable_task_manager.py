"""Comprehensive tests for ReusableTaskManager."""

import pytest
from unittest.mock import Mock, patch
from arklex.orchestrator.generator.tasks.reusable_task_manager import (
    ReusableTask,
    ReusableTaskManager,
)


@pytest.fixture
def mock_model():
    model = Mock()
    return model


@pytest.fixture
def reusable_task_manager(mock_model):
    return ReusableTaskManager(
        model=mock_model, role="test_role", user_objective="test objective"
    )


@pytest.fixture
def sample_tasks():
    return [
        {
            "name": "Task 1",
            "description": "Description 1",
            "steps": [
                {"task": "Step 1", "description": "Step 1 desc"},
                {"task": "Step 2", "description": "Step 2 desc"},
            ],
        },
        {
            "name": "Task 2",
            "description": "Description 2",
            "steps": [
                {"task": "Step 1", "description": "Step 1 desc"},
            ],
        },
    ]


class TestReusableTaskManager:
    def test_generate_reusable_tasks(self, reusable_task_manager, sample_tasks) -> None:
        templates = reusable_task_manager.generate_reusable_tasks(sample_tasks)
        assert isinstance(templates, dict)
        assert len(templates) > 0
        for t in templates.values():
            assert isinstance(t, dict)
            assert "name" in t
            assert "description" in t
            assert "steps" in t

    def test_generate_reusable_tasks_empty(self, reusable_task_manager) -> None:
        templates = reusable_task_manager.generate_reusable_tasks([])
        assert isinstance(templates, dict)
        assert len(templates) == 0

    def test_instantiate_template_success(self, reusable_task_manager) -> None:
        template = ReusableTask(
            template_id="tid1",
            name="Template 1",
            description="Desc",
            steps=[{"task": "Step", "description": "Desc"}],
            parameters={"param1": "string"},
            examples=[],
            version="1.0",
            category="cat",
        )
        reusable_task_manager._templates["tid1"] = template
        params = {"param1": "value"}
        instance = reusable_task_manager.instantiate_template("tid1", params)
        assert isinstance(instance, dict)
        assert "steps" in instance

    def test_instantiate_template_not_found(self, reusable_task_manager) -> None:
        with pytest.raises(ValueError, match="Template not found"):
            reusable_task_manager.instantiate_template("not_exist", {})

    def test_instantiate_template_invalid_params(self, reusable_task_manager) -> None:
        template = ReusableTask(
            template_id="tid2",
            name="Template 2",
            description="Desc",
            steps=[{"task": "Step", "description": "Desc"}],
            parameters={"param1": "string"},
            examples=[],
            version="1.0",
            category="cat",
        )
        reusable_task_manager._templates["tid2"] = template
        with pytest.raises(ValueError, match="Invalid parameters"):
            reusable_task_manager.instantiate_template("tid2", {"wrong": 1})

    def test_instantiate_template_exception(self, reusable_task_manager) -> None:
        with patch.object(
            reusable_task_manager, "_validate_parameters", side_effect=Exception("fail")
        ):
            template = ReusableTask(
                template_id="tid3",
                name="Template 3",
                description="Desc",
                steps=[{"task": "Step", "description": "Desc"}],
                parameters={"param1": "string"},
                examples=[],
                version="1.0",
                category="cat",
            )
            reusable_task_manager._templates["tid3"] = template
            with pytest.raises(Exception, match="fail"):
                reusable_task_manager.instantiate_template("tid3", {"param1": "v"})

    def test_identify_patterns(self, reusable_task_manager, sample_tasks) -> None:
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        assert isinstance(patterns, list)
        assert len(patterns) == len(sample_tasks)
        for p in patterns:
            assert "name" in p
            assert "description" in p
            assert "steps" in p
            assert "parameters" in p

    def test_extract_components(self, reusable_task_manager, sample_tasks) -> None:
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        components = reusable_task_manager._extract_components(patterns)
        assert isinstance(components, list)
        assert len(components) == len(patterns)

    def test_create_templates(self, reusable_task_manager, sample_tasks) -> None:
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        components = reusable_task_manager._extract_components(patterns)
        templates = reusable_task_manager._create_templates(components)
        assert isinstance(templates, dict)
        assert len(templates) == len(components)
        for t in templates.values():
            assert isinstance(t, ReusableTask)

    def test_validate_templates(self, reusable_task_manager, sample_tasks) -> None:
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        components = reusable_task_manager._extract_components(patterns)
        templates = reusable_task_manager._create_templates(components)
        validated = reusable_task_manager._validate_templates(templates)
        assert isinstance(validated, dict)
        assert len(validated) == len(templates)
        for v in validated.values():
            assert isinstance(v, dict)

    def test_validate_template(self, reusable_task_manager, sample_tasks) -> None:
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        components = reusable_task_manager._extract_components(patterns)
        templates = reusable_task_manager._create_templates(components)
        for t in templates.values():
            assert reusable_task_manager._validate_template(t)
        # Test with missing fields
        bad = ReusableTask(
            template_id="",
            name="",
            description="",
            steps=[],
            parameters={},
            examples=[],
            version="",
            category="",
        )
        assert not reusable_task_manager._validate_template(bad)

    def test_validate_parameters(self, reusable_task_manager) -> None:
        template = ReusableTask(
            template_id="tid4",
            name="Template 4",
            description="Desc",
            steps=[{"task": "Step", "description": "Desc"}],
            parameters={"param1": "string", "param2": "int"},
            examples=[],
            version="1.0",
            category="cat",
        )
        assert reusable_task_manager._validate_parameters(
            template, {"param1": "v", "param2": 1}
        )
        assert not reusable_task_manager._validate_parameters(template, {"param1": "v"})
        assert not reusable_task_manager._validate_parameters(template, {"param2": 1})
        assert not reusable_task_manager._validate_parameters(template, {})

    def test_create_instance(self, reusable_task_manager) -> None:
        template = ReusableTask(
            template_id="tid5",
            name="Template 5",
            description="Desc",
            steps=[{"task": "Step", "description": "Desc"}],
            parameters={"param1": "string"},
            examples=[],
            version="1.0",
            category="cat",
        )
        params = {"param1": "value"}
        instance = reusable_task_manager._create_instance(template, params)
        assert isinstance(instance, dict)
        assert "steps" in instance

    def test_categorize_templates(self, reusable_task_manager, sample_tasks) -> None:
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        components = reusable_task_manager._extract_components(patterns)
        templates = reusable_task_manager._create_templates(components)
        validated = reusable_task_manager._validate_templates(templates)
        reusable_task_manager._categorize_templates(validated)
        assert isinstance(reusable_task_manager._template_categories, dict)
        for cat, ids in reusable_task_manager._template_categories.items():
            assert isinstance(cat, str)
            assert isinstance(ids, list)

    def test_convert_to_dict(self, reusable_task_manager, sample_tasks) -> None:
        patterns = reusable_task_manager._identify_patterns(sample_tasks)
        components = reusable_task_manager._extract_components(patterns)
        templates = reusable_task_manager._create_templates(components)
        for t in templates.values():
            d = reusable_task_manager._convert_to_dict(t)
            assert isinstance(d, dict)
            assert "template_id" in d
            assert "name" in d
            assert "description" in d
            assert "steps" in d
            assert "parameters" in d
            assert "examples" in d
            assert "version" in d
            assert "category" in d
