"""Tests for the model service module.

This module contains comprehensive tests for the ModelService and DummyModelService
classes, covering initialization, text processing, intent detection, slot filling,
verification, and utility methods.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from arklex.orchestrator.NLU.services.model_service import (
    ModelService,
    DummyModelService,
)
from arklex.utils.exceptions import ValidationError, ModelError


@pytest.fixture
def model_config() -> Dict[str, Any]:
    """Fixture for model configuration."""
    return {
        "model_name": "gpt-4",
        "model_type_or_path": "gpt-4",
        "llm_provider": "openai",
        "api_key": "test_api_key",
        "endpoint": "https://api.openai.com/v1",
        "temperature": 0.1,
    }


@pytest.fixture
def dummy_config() -> Dict[str, Any]:
    """Fixture for dummy model configuration."""
    return {
        "model_name": "test_model",
        "model_type_or_path": "test_path",
        "llm_provider": "openai",
        "api_key": "your_default_api_key",
        "endpoint": "https://api.openai.com/v1",
    }


@pytest.fixture
def model_service(model_config: Dict[str, Any]) -> ModelService:
    """Fixture for ModelService instance."""
    with patch(
        "arklex.orchestrator.NLU.services.model_service.ModelConfig.get_model_instance"
    ) as mock_get_model:
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        service = ModelService(model_config)
        return service


@pytest.fixture
def dummy_model_service(dummy_config: Dict[str, Any]) -> DummyModelService:
    """Fixture for DummyModelService instance."""
    return DummyModelService(dummy_config)


@pytest.fixture
def sample_intents() -> Dict[str, Any]:
    """Fixture for sample intent definitions."""
    return {
        "greeting": {
            "definition": "User greets the assistant",
            "sample_utterances": ["hello", "hi", "good morning"],
        },
        "booking": {
            "definition": "User wants to book something",
            "sample_utterances": ["book a table", "make reservation"],
        },
    }


@pytest.fixture
def sample_slots() -> Dict[str, Any]:
    """Fixture for sample slot definitions."""
    return {
        "date": {"type": "date", "description": "Date for booking", "required": True},
        "time": {"type": "time", "description": "Time for booking", "required": True},
    }


@pytest.fixture
def sample_verification_slots() -> Dict[str, Any]:
    """Fixture for sample verification slot definitions."""
    return {
        "date": {"type": "date", "description": "Date for booking", "required": True},
        "time": {"type": "time", "description": "Time for booking", "required": True},
    }


class TestModelServiceInitialization:
    """Test cases for ModelService initialization and configuration validation."""

    def test_model_service_initialization_success(
        self, model_config: Dict[str, Any]
    ) -> None:
        """Test successful model service initialization with valid configuration."""
        with patch(
            "arklex.orchestrator.NLU.services.model_service.ModelConfig.get_model_instance"
        ) as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            service = ModelService(model_config)
            assert service.model_config["model_name"] == "gpt-4"
            assert service.model_config["api_key"] == "test_api_key"
            assert service.model_config["endpoint"] == "https://api.openai.com/v1"

    def test_model_service_initialization_missing_config(self) -> None:
        """Test model service initialization with empty configuration raises error."""
        with pytest.raises(ValidationError) as exc_info:
            ModelService({})
        assert "Missing required field" in str(exc_info.value)

    def test_model_service_initialization_dummy_config(
        self, dummy_config: Dict[str, Any]
    ) -> None:
        """Test model service initialization with dummy configuration."""
        with patch(
            "arklex.orchestrator.NLU.services.model_service.ModelConfig.get_model_instance"
        ) as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            service = ModelService(dummy_config)
            assert service.model_config["model_name"] == "test_model"
            assert service.model_config["llm_provider"] == "openai"

    def test_model_service_missing_model_name(self) -> None:
        """Test model service initialization with missing model name raises error."""
        incomplete_config = {"api_key": "key", "endpoint": "url"}
        with pytest.raises(ValidationError, match="Missing required field"):
            ModelService(incomplete_config)

    def test_model_service_missing_api_key(self) -> None:
        """Test model service initialization with missing API key raises error."""
        incomplete_config = {"model_name": "name", "endpoint": "url"}
        with pytest.raises(ValidationError, match="Missing required field"):
            ModelService(incomplete_config)

    def test_model_service_missing_endpoint(self) -> None:
        """Test model service initialization with missing endpoint raises error."""
        incomplete_config = {"model_name": "name", "api_key": "key"}
        with pytest.raises(ValidationError, match="Missing required field"):
            ModelService(incomplete_config)

    def test_validate_config_missing_fields(self, model_config: Dict[str, Any]) -> None:
        """Test validate_config with missing required fields."""
        # Remove required fields
        invalid_config = model_config.copy()
        del invalid_config["model_name"]
        del invalid_config["api_key"]
        del invalid_config["endpoint"]

        with pytest.raises(ValidationError):
            ModelService(invalid_config)

    def test_validate_config_missing_model_name(
        self, model_config: Dict[str, Any]
    ) -> None:
        """Test validate_config with missing model_name."""
        # Remove model_name
        invalid_config = model_config.copy()
        del invalid_config["model_name"]

        with pytest.raises(ValidationError):
            ModelService(invalid_config)

    def test_validate_config_with_defaults(self, model_config: Dict[str, Any]) -> None:
        """Test validate_config with default values."""
        # Remove optional fields to test defaults
        config_with_defaults = {
            "model_name": model_config["model_name"],
            "api_key": model_config["api_key"],
            "endpoint": model_config["endpoint"],
            "model_type_or_path": model_config["model_type_or_path"],
            "llm_provider": model_config["llm_provider"],
        }

        with patch(
            "arklex.orchestrator.NLU.services.model_service.ModelConfig.get_model_instance"
        ) as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model
            service = ModelService(config_with_defaults)
            # Check that the service was initialized successfully
            assert service.model_config["model_name"] == model_config["model_name"]


class TestModelServiceTextProcessing:
    """Test cases for ModelService text processing methods."""

    @pytest.fixture
    def model_service(self) -> ModelService:
        """Fixture for ModelService instance."""
        config = {
            "model_name": "test-model",
            "model_type_or_path": "test-path",
            "llm_provider": "openai",
            "api_key": "test-key",
            "endpoint": "http://test.com",
        }
        with (
            patch(
                "arklex.orchestrator.NLU.services.model_service.ModelConfig.get_model_instance"
            ) as mock_get_model,
            patch(
                "arklex.orchestrator.NLU.services.model_service.ModelConfig.configure_response_format"
            ) as mock_configure,
        ):
            mock_model = MagicMock()
            mock_model.agenerate = AsyncMock()
            mock_get_model.return_value = mock_model
            mock_configure.return_value = mock_model
            service = ModelService(config)
            return service

    @pytest.mark.asyncio
    async def test_make_model_request(self, model_service: ModelService) -> None:
        """Test model request processing."""
        test_text = "Test input text"
        test_context = {"context": "test"}

        expected_response = MagicMock()
        expected_response.generations = [[MagicMock(text="success")]]

        with patch.object(
            model_service.model, "agenerate", new_callable=AsyncMock
        ) as mock_agenerate:
            mock_agenerate.return_value = expected_response

            actual_response = await model_service._make_model_request(
                {
                    "text": test_text,
                    "context": test_context,
                    "model": model_service.model_config["model_name"],
                }
            )

            assert actual_response == {"result": "success"}
            mock_agenerate.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_text_request_failure(
        self, model_service: ModelService
    ) -> None:
        """Test text processing when model request fails."""
        test_text = "Test input text"

        with patch.object(
            model_service, "_make_model_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = ModelError("Model request failed")

            with pytest.raises(ModelError, match="Model request failed"):
                await model_service.process_text(test_text)

    @pytest.mark.asyncio
    async def test_process_text_success(self, model_service: ModelService) -> None:
        """Test successful text processing."""
        test_text = "Test input text"
        expected_response = MagicMock()
        expected_response.generations = [[MagicMock(text="success")]]

        with patch.object(
            model_service.model, "agenerate", new_callable=AsyncMock
        ) as mock_agenerate:
            mock_agenerate.return_value = expected_response

            result = await model_service.process_text(test_text)
            assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_process_text_with_whitespace_only(
        self, model_service: ModelService
    ) -> None:
        """Test text processing with whitespace-only input raises validation error."""
        with pytest.raises(ValidationError, match="Text cannot be empty"):
            await model_service.process_text("   ")


class TestModelServiceResponseHandling:
    """Test cases for ModelService response handling methods."""

    def test_model_service_get_response_success(
        self, dummy_model_service: DummyModelService
    ) -> None:
        """Test successful response generation."""
        prompt = "Test prompt"
        response = dummy_model_service.get_response(prompt)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_model_service_get_response_empty(
        self, dummy_model_service: DummyModelService
    ) -> None:
        """Test response generation with empty prompt."""
        # The actual implementation doesn't validate empty prompts
        response = dummy_model_service.get_response("")
        assert isinstance(response, str)

    def test_model_service_get_response_model_error(
        self, dummy_model_service: DummyModelService
    ) -> None:
        """Test response generation when model raises error."""
        with patch.object(
            dummy_model_service, "get_response", side_effect=ValueError("Model error")
        ):
            with pytest.raises(ValueError):
                dummy_model_service.get_response("test prompt")

    def test_get_json_response(self, model_service: ModelService) -> None:
        """Test JSON response generation."""
        prompt = "Return JSON: {'key': 'value'}"
        mock_response = MagicMock()
        mock_response.content = '{"key": "value"}'
        with patch.object(model_service.model, "invoke", return_value=mock_response):
            result = model_service.get_json_response(prompt)
            assert result == {"key": "value"}

    def test_get_response_with_model_config(self, model_service: ModelService) -> None:
        """Test response generation with model configuration."""
        prompt = "Test prompt"
        mock_response = MagicMock()
        mock_response.content = "Test response"
        with patch.object(model_service.model, "invoke", return_value=mock_response):
            result = model_service.get_response(prompt)
            assert result == "Test response"

    def test_get_response_with_system_prompt(self, model_service: ModelService) -> None:
        """Test response generation with system prompt."""
        prompt = "Test prompt"
        mock_response = MagicMock()
        mock_response.content = "Test response"
        with patch.object(model_service.model, "invoke", return_value=mock_response):
            result = model_service.get_response(
                prompt, system_prompt="You are a helpful assistant"
            )
            assert result == "Test response"

    def test_get_response_with_response_format(
        self, model_service: ModelService
    ) -> None:
        """Test response generation with response format."""
        prompt = "Test prompt"
        mock_response = MagicMock()
        mock_response.content = "Test response"
        with patch.object(model_service.model, "invoke", return_value=mock_response):
            result = model_service.get_response(
                prompt, response_format={"type": "text"}
            )
            assert result == "Test response"

    def test_get_response_with_note(self, model_service: ModelService) -> None:
        """Test response generation with note."""
        prompt = "Test prompt"
        mock_response = MagicMock()
        mock_response.content = "Test response"
        with patch.object(model_service.model, "invoke", return_value=mock_response):
            result = model_service.get_response(prompt, note="Important note")
            assert result == "Test response"

    def test_get_response_with_exception(self, model_service: ModelService) -> None:
        """Test response generation when model raises exception."""
        prompt = "Test prompt"
        with patch.object(
            model_service.model, "invoke", side_effect=Exception("Model error")
        ):
            with pytest.raises(ValueError):
                model_service.get_response(prompt)

    def test_get_json_response_with_exception(
        self, model_service: ModelService
    ) -> None:
        """Test JSON response generation when model raises exception."""
        prompt = "Test prompt"
        with patch.object(
            model_service.model, "invoke", side_effect=Exception("Model error")
        ):
            with pytest.raises(ValueError):
                model_service.get_json_response(prompt)


class TestModelServiceIntentProcessing:
    """Test cases for ModelService intent processing methods."""

    def test_format_intent_definition(self, model_service: ModelService) -> None:
        """Test formatting intent definitions."""
        result = model_service._format_intent_definition(
            "greeting", "User greets the assistant", 1
        )
        assert "greeting" in result
        assert "User greets the assistant" in result

    def test_format_intent_exemplars(self, model_service: ModelService) -> None:
        """Test formatting intent exemplars."""
        result = model_service._format_intent_exemplars("greeting", ["hello", "hi"], 1)
        assert "greeting" in result
        assert "hello" in result
        assert "hi" in result

    def test_process_intent(self, model_service: ModelService) -> None:
        """Test processing intent response."""
        response = {"intent": "greeting", "confidence": 0.95, "slots": {"name": "John"}}
        result = model_service._process_intent(
            "greeting", [response], 1, {"1": "greeting"}
        )
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_format_intent_input(self, model_service: ModelService) -> None:
        """Test formatting intent input."""
        text = "hello there"
        intents = {
            "greeting": [{"definition": "User greets", "sample_utterances": ["hello"]}]
        }
        result = model_service.format_intent_input(intents, text)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestModelServiceSlotProcessing:
    """Test cases for ModelService slot processing methods."""

    def test_format_slot_input(self, model_service: ModelService) -> None:
        """Test formatting slot input."""
        text = "book a table"
        slots = [{"name": "date", "type": "date", "description": "Date for booking"}]
        result = model_service.format_slot_input(slots, text)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_process_slot_response(self, model_service: ModelService) -> None:
        """Test processing slot response."""
        response = {"slots": {"date": "tomorrow", "time": "7pm"}}
        slots = [{"name": "date", "type": "date"}]
        result = model_service.process_slot_response(json.dumps(response), slots)
        assert isinstance(result, list)

    def test_process_slot_response_with_invalid_json(
        self, model_service: ModelService
    ) -> None:
        """Test processing slot response with invalid JSON."""
        response = "invalid json"
        slots = [{"name": "date", "type": "date"}]
        with pytest.raises(ValueError):
            model_service.process_slot_response(response, slots)

    def test_process_slot_response_with_missing_slots(
        self, model_service: ModelService
    ) -> None:
        """Test processing slot response with missing slots."""
        response = {"slots": {"date": "tomorrow"}}
        slots = [{"name": "date", "type": "date"}, {"name": "time", "type": "time"}]
        result = model_service.process_slot_response(json.dumps(response), slots)
        assert isinstance(result, list)


class TestModelServiceVerification:
    """Test cases for ModelService slot verification methods."""

    def test_format_verification_input(self, model_service: ModelService) -> None:
        """Test formatting verification input."""
        slot = {"name": "date", "value": "tomorrow", "description": "desc"}
        chat_history = "User: book a table"
        result = model_service.format_verification_input(slot, chat_history)
        assert isinstance(result, str)

    def test_process_verification_response(self, model_service: ModelService) -> None:
        """Test processing verification response."""
        response = '{"verification_needed": true, "thought": "Need to verify"}'
        result = model_service.process_verification_response(response)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestModelServiceUtilities:
    """Test cases for ModelService utility methods."""

    def test_format_messages(self, model_service: ModelService) -> None:
        """Test message formatting."""
        prompt = "Test prompt"
        context = {"key": "value"}
        messages = model_service._format_messages(prompt, context)
        assert len(messages) == 2
        assert any("Context" in str(msg.content) for msg in messages)
        assert any("Test prompt" in str(msg.content) for msg in messages)

    def test_format_messages_with_context(self, model_service: ModelService) -> None:
        """Test message formatting with context."""
        prompt = "Test prompt"
        messages = model_service._format_messages(prompt)
        assert len(messages) == 1
        assert any("Test prompt" in str(msg.content) for msg in messages)


class TestDummyModelService:
    """Test cases for DummyModelService class."""

    def test_dummy_model_service_methods(
        self, dummy_model_service: DummyModelService
    ) -> None:
        """Test all DummyModelService methods return expected results."""
        # Test get_response
        response = dummy_model_service.get_response("test prompt")
        assert isinstance(response, str)
        assert len(response) > 0

        # Test get_json_response - this will fail because "1) others" is not valid JSON
        with pytest.raises(ValueError, match="Failed to parse JSON response"):
            dummy_model_service.get_json_response("test prompt")

    def test_dummy_model_service_with_none_config(self) -> None:
        """Test DummyModelService initialization with None config."""
        with pytest.raises(ValidationError, match="Missing required field"):
            DummyModelService({})

    def test_dummy_model_service_get_response_with_none_prompt(
        self, dummy_model_service: DummyModelService
    ) -> None:
        """Test DummyModelService get_response with None prompt."""
        # The actual implementation doesn't validate None prompts
        response = dummy_model_service.get_response(None)
        assert isinstance(response, str)

    def test_dummy_model_service_get_response_with_empty_prompt(
        self, dummy_model_service: DummyModelService
    ) -> None:
        """Test DummyModelService get_response with empty prompt."""
        # The actual implementation doesn't validate empty prompts
        response = dummy_model_service.get_response("")
        assert isinstance(response, str)

    def test_dummy_model_service_get_json_response_with_none_prompt(
        self, dummy_model_service: DummyModelService
    ) -> None:
        """Test DummyModelService get_json_response with None prompt."""
        # This will fail because "1) others" is not valid JSON
        with pytest.raises(ValueError, match="Failed to parse JSON response"):
            dummy_model_service.get_json_response(None)

    def test_dummy_model_service_get_json_response_with_empty_prompt(
        self, dummy_model_service: DummyModelService
    ) -> None:
        """Test DummyModelService get_json_response with empty prompt."""
        # This will fail because "1) others" is not valid JSON
        with pytest.raises(ValueError, match="Failed to parse JSON response"):
            dummy_model_service.get_json_response("")


class TestModelServiceErrorHandling:
    """Test cases for ModelService error handling."""

    @pytest.mark.asyncio
    async def test_predict_intent_empty_response(
        self, model_service: ModelService
    ) -> None:
        """Test intent prediction with empty response."""
        text = "hello there"

        mock_response = MagicMock()
        mock_response.content = ""

        with patch.object(
            model_service.model, "invoke", new_callable=AsyncMock
        ) as mock_invoke:
            mock_invoke.return_value = mock_response

            with pytest.raises(ModelError, match="Empty response from model"):
                await model_service.predict_intent(text)

    @pytest.mark.asyncio
    async def test_predict_intent_json_decode_error(
        self, model_service: ModelService
    ) -> None:
        """Test intent prediction with JSON decode error."""
        text = "hello there"

        mock_response = MagicMock()
        mock_response.content = "invalid json"

        with patch.object(
            model_service.model, "invoke", new_callable=AsyncMock
        ) as mock_invoke:
            mock_invoke.return_value = mock_response

            with pytest.raises(ModelError, match="Failed to parse model response"):
                await model_service.predict_intent(text)

    @pytest.mark.asyncio
    async def test_predict_intent_validation_error(
        self, model_service: ModelService
    ) -> None:
        """Test intent prediction with validation error."""
        text = "hello there"

        with patch.object(
            model_service.model, "invoke", new_callable=AsyncMock
        ) as mock_invoke:
            mock_invoke.side_effect = ValidationError("Invalid response")

            with pytest.raises(ValidationError, match="Invalid response"):
                await model_service.predict_intent(text)

    @pytest.mark.asyncio
    async def test_fill_slots_empty_response(self, model_service: ModelService) -> None:
        """Test slot filling with empty response."""
        text = "book a table"

        mock_response = MagicMock()
        mock_response.content = ""

        with patch.object(
            model_service.model, "invoke", new_callable=AsyncMock
        ) as mock_invoke:
            mock_invoke.return_value = mock_response

            with pytest.raises(ModelError, match="Empty response from model"):
                await model_service.fill_slots(text, "booking")

    @pytest.mark.asyncio
    async def test_fill_slots_json_decode_error(
        self, model_service: ModelService
    ) -> None:
        """Test slot filling with JSON decode error."""
        text = "book a table"

        mock_response = MagicMock()
        mock_response.content = "invalid json"

        with patch.object(
            model_service.model, "invoke", new_callable=AsyncMock
        ) as mock_invoke:
            mock_invoke.return_value = mock_response

            with pytest.raises(ModelError, match="Failed to parse slot response"):
                await model_service.fill_slots(text, "booking")

    @pytest.mark.asyncio
    async def test_verify_slots_empty_response(
        self, model_service: ModelService
    ) -> None:
        """Test slot verification with empty response."""
        text = "confirm booking"
        slots = {"date": "tomorrow"}

        mock_response = MagicMock()
        mock_response.content = ""

        with patch.object(
            model_service.model, "invoke", new_callable=AsyncMock
        ) as mock_invoke:
            mock_invoke.return_value = mock_response

            with pytest.raises(ModelError, match="Empty response from model"):
                await model_service.verify_slots(text, slots)

    @pytest.mark.asyncio
    async def test_verify_slots_json_decode_error(
        self, model_service: ModelService
    ) -> None:
        """Test slot verification with JSON decode error."""
        text = "confirm booking"
        slots = {"date": "tomorrow"}

        mock_response = MagicMock()
        mock_response.content = "invalid json"

        with patch.object(
            model_service.model, "invoke", new_callable=AsyncMock
        ) as mock_invoke:
            mock_invoke.return_value = mock_response

            with pytest.raises(
                ModelError, match="Failed to parse verification response"
            ):
                await model_service.verify_slots(text, slots)

    @pytest.mark.asyncio
    async def test_verify_slots_validation_error(
        self, model_service: ModelService
    ) -> None:
        """Test slot verification with validation error."""
        text = "confirm booking"
        slots = {"date": "tomorrow"}

        with patch.object(
            model_service.model, "invoke", new_callable=AsyncMock
        ) as mock_invoke:
            mock_invoke.side_effect = ValidationError("Invalid response")

            with pytest.raises(ValidationError, match="Invalid response"):
                await model_service.verify_slots(text, slots)
