"""Tests for the ModelService class.

This module contains tests for the ModelService class, focusing on
slot filling functionality and response processing.
"""

import pytest
from typing import Dict, Any, List
from arklex.orchestrator.NLU.services.model_service import ModelService
from arklex.utils.exceptions import ModelError, ValidationError
from arklex.orchestrator.NLU.core.base import (
    IntentResponse,
    SlotResponse,
    VerificationResponse,
)
from unittest.mock import Mock, patch


@pytest.fixture
def mock_api_service():
    """Create a mock API service."""
    mock = Mock()
    mock.get_model_response.return_value = Mock(
        content='{"intent": "test", "confidence": 0.9}'
    )
    return mock


@pytest.fixture
def model_service(mock_api_service):
    """Create a ModelService instance with mocked dependencies."""
    with patch(
        "arklex.orchestrator.NLU.services.model_service.APIService",
        return_value=mock_api_service,
    ):
        return ModelService()


@pytest.fixture
def sample_slots():
    """Create sample slot definitions for testing."""
    return [
        {
            "name": "location",
            "type": "string",
            "description": "The city or location where the event will take place",
            "required": True,
        },
        {
            "name": "date",
            "type": "string",
            "description": "The date of the event",
            "required": True,
        },
        {
            "name": "time",
            "type": "string",
            "description": "The time of the event",
            "required": False,
        },
    ]


@pytest.fixture
def sample_context():
    """Create sample context for testing."""
    return (
        "I want to schedule a meeting in New York on March 15th "
        "at 2:00 PM in the afternoon."
    )


def test_format_slot_input(model_service, sample_slots, sample_context):
    """Test the format_slot_input method."""
    user_prompt, system_prompt = model_service.format_slot_input(
        sample_slots, sample_context
    )

    # Check that both prompts are strings
    assert isinstance(user_prompt, str)
    assert isinstance(system_prompt, str)

    # Check that the prompts contain the necessary information
    assert "Context:" in user_prompt
    assert sample_context in user_prompt
    assert "Slot definitions:" in user_prompt
    assert "location" in user_prompt
    assert "date" in user_prompt
    assert "time" in user_prompt
    assert "required" in user_prompt
    assert "optional" in user_prompt

    # Check system prompt
    assert "slot filling assistant" in system_prompt.lower()
    assert "extract" in system_prompt.lower()
    assert "json format" in system_prompt.lower()


def test_process_slot_response(model_service, sample_slots):
    """Test the process_slot_response method."""
    # Sample response from the model
    response = '{"location": "New York", "date": "March 15th", "time": "2:00 PM"}'

    # Process the response
    updated_slots = model_service.process_slot_response(response, sample_slots)

    # Check that all slots were processed
    assert len(updated_slots) == len(sample_slots)

    # Check that values were correctly assigned
    for slot in updated_slots:
        if slot["name"] == "location":
            assert slot["value"] == "New York"
        elif slot["name"] == "date":
            assert slot["value"] == "March 15th"
        elif slot["name"] == "time":
            assert slot["value"] == "2:00 PM"


def test_process_slot_response_with_missing_values(model_service, sample_slots):
    """Test the process_slot_response method with missing values."""
    # Sample response with missing values
    response = '{"location": "New York", "date": "March 15th"}'

    # Process the response
    updated_slots = model_service.process_slot_response(response, sample_slots)

    # Check that all slots were processed
    assert len(updated_slots) == len(sample_slots)

    # Check that values were correctly assigned
    for slot in updated_slots:
        if slot["name"] == "location":
            assert slot["value"] == "New York"
        elif slot["name"] == "date":
            assert slot["value"] == "March 15th"
        elif slot["name"] == "time":
            assert slot["value"] is None


def test_process_slot_response_with_invalid_json(model_service, sample_slots):
    """Test the process_slot_response method with invalid JSON."""
    # Invalid JSON response
    response = "invalid json"

    # Check that it raises ValueError
    with pytest.raises(ValueError):
        model_service.process_slot_response(response, sample_slots)


def test_format_slot_input_with_enum_values(model_service):
    """Test the format_slot_input method with enum values."""
    slots = [
        {
            "name": "category",
            "type": "string",
            "description": "The category of the item",
            "required": True,
            "items": {"enum": ["electronics", "clothing", "books", "food"]},
        }
    ]
    context = "I want to buy some electronics."

    user_prompt, system_prompt = model_service.format_slot_input(slots, context)

    # Check that enum values are included in the prompt
    assert "Possible values:" in user_prompt
    assert "electronics" in user_prompt
    assert "clothing" in user_prompt
    assert "books" in user_prompt
    assert "food" in user_prompt


def test_predict_intent_success(model_service, mock_api_service):
    """Test successful intent prediction."""
    response = model_service.predict_intent("test input")
    assert isinstance(response, IntentResponse)
    assert response.intent == "test"
    assert response.confidence == 0.9


def test_predict_intent_validation_error(model_service):
    """Test intent prediction with invalid input."""
    with pytest.raises(ValidationError):
        model_service.predict_intent("")


def test_predict_intent_model_error(model_service, mock_api_service):
    """Test intent prediction with model error."""
    mock_api_service.get_model_response.side_effect = Exception("Model error")
    with pytest.raises(ModelError):
        model_service.predict_intent("test input")


def test_fill_slots_success(model_service, mock_api_service):
    """Test successful slot filling."""
    mock_api_service.get_model_response.return_value = Mock(
        content='{"slots": {"slot1": "value1"}}'
    )
    response = model_service.fill_slots("test input", "test_intent")
    assert isinstance(response, SlotResponse)
    assert response.slots == {"slot1": "value1"}


def test_fill_slots_validation_error(model_service):
    """Test slot filling with invalid input."""
    with pytest.raises(ValidationError):
        model_service.fill_slots("", "test_intent")
    with pytest.raises(ValidationError):
        model_service.fill_slots("test input", "")


def test_fill_slots_model_error(model_service, mock_api_service):
    """Test slot filling with model error."""
    mock_api_service.get_model_response.side_effect = Exception("Model error")
    with pytest.raises(ModelError):
        model_service.fill_slots("test input", "test_intent")


def test_verify_slots_success(model_service, mock_api_service):
    """Test successful slot verification."""
    mock_api_service.get_model_response.return_value = Mock(
        content='{"verified": true, "slots": {"slot1": true}}'
    )
    response = model_service.verify_slots("test input", {"slot1": "value1"})
    assert isinstance(response, VerificationResponse)
    assert response.verified is True
    assert response.slots == {"slot1": True}


def test_verify_slots_validation_error(model_service):
    """Test slot verification with invalid input."""
    with pytest.raises(ValidationError):
        model_service.verify_slots("", {"slot1": "value1"})
    with pytest.raises(ValidationError):
        model_service.verify_slots("test input", {})


def test_verify_slots_model_error(model_service, mock_api_service):
    """Test slot verification with model error."""
    mock_api_service.get_model_response.side_effect = Exception("Model error")
    with pytest.raises(ModelError):
        model_service.verify_slots("test input", {"slot1": "value1"})


def test_invalid_json_response(model_service, mock_api_service):
    """Test handling of invalid JSON responses."""
    mock_api_service.get_model_response.return_value = Mock(content="invalid json")
    with pytest.raises(ModelError):
        model_service.predict_intent("test input")


def test_empty_model_response(model_service, mock_api_service):
    """Test handling of empty model responses."""
    mock_api_service.get_model_response.return_value = Mock(content="")
    with pytest.raises(ModelError):
        model_service.predict_intent("test input")
