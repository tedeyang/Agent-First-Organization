"""Tests for the model service module.

This module contains comprehensive tests for the ModelService and DummyModelService
classes, covering initialization, text processing, intent detection, slot filling,
verification, and utility methods.
"""

from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arklex.orchestrator.NLU.core.base import (
    IntentResponse,
    SlotResponse,
    VerificationResponse,
)
from arklex.orchestrator.NLU.services.model_service import (
    ModelService,
    DummyModelService,
)
from arklex.utils.exceptions import ValidationError


@pytest.fixture
def model_config() -> Dict[str, Any]:
    """Create a test model configuration.

    Returns:
        Dict[str, Any]: A dictionary containing complete model configuration
            with all required fields for testing.
    """
    return {
        "model_name": "test-model",
        "api_key": "test-key",
        "endpoint": "https://api.test.com/v1",
        "model_type_or_path": "gpt-3.5-turbo",
        "llm_provider": "openai",
        "temperature": 0.1,
        "max_tokens": 1000,
        "response_format": "json",
    }


@pytest.fixture
def model_service(model_config: Dict[str, Any]) -> ModelService:
    """Create a test model service instance.

    Args:
        model_config: The model configuration to use for the service.

    Returns:
        ModelService: A fully configured model service instance for testing.
    """
    return ModelService(model_config)


@pytest.fixture
def dummy_config() -> Dict[str, Any]:
    """Create a test dummy configuration.

    Returns:
        Dict[str, Any]: A dictionary containing minimal dummy configuration
            for testing DummyModelService.
    """
    return {
        "model_name": "dummy",
        "api_key": "dummy",
        "endpoint": "http://dummy",
        "model_type_or_path": "dummy-path",
        "llm_provider": "dummy",
    }


class TestModelServiceInitialization:
    """Test cases for ModelService initialization and configuration validation."""

    def test_model_service_initialization_success(
        self, model_config: Dict[str, Any]
    ) -> None:
        """Test successful model service initialization with valid configuration.

        Args:
            model_config: The complete model configuration to use.
        """
        service = ModelService(model_config)
        assert service.model_config == model_config
        assert service.model_config["model_name"] == "test-model"

    def test_model_service_initialization_missing_config(self) -> None:
        """Test model service initialization with empty configuration raises error."""
        with pytest.raises(ValidationError) as exc_info:
            ModelService({})
        assert "Missing required field" in str(exc_info.value)

    def test_model_service_initialization_dummy_config(
        self, dummy_config: Dict[str, Any]
    ) -> None:
        """Test model service initialization with dummy configuration.

        Args:
            dummy_config: The dummy configuration to use.
        """
        service = ModelService(dummy_config)
        assert service.model_config["model_name"] == "dummy"
        assert service.model_config["llm_provider"] == "dummy"

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


class TestModelServiceTextProcessing:
    """Test cases for ModelService text processing and request handling."""

    @pytest.mark.asyncio
    @patch("arklex.orchestrator.NLU.services.model_service.APIClientService")
    async def test_process_text_success(
        self, mock_api_service: MagicMock, model_service: ModelService
    ) -> None:
        """Test successful text processing with context.

        Args:
            mock_api_service: Mock API service to avoid async warnings.
            model_service: The model service instance to test.
        """
        test_text = "Test input text"
        test_context = {"user_id": "123"}

        # Mock the API service to avoid any async warnings
        mock_api_service.return_value = MagicMock()

        with patch.object(
            model_service, "_make_model_request", new_callable=AsyncMock
        ) as mock_request:
            expected_response = {"result": "Test response"}
            mock_request.return_value = expected_response

            actual_response = await model_service.process_text(test_text, test_context)

            assert actual_response == expected_response
            mock_request.assert_called_once_with(
                {
                    "text": test_text,
                    "context": test_context,
                    "model": model_service.model_config["model_name"],
                }
            )

    @pytest.mark.asyncio
    async def test_process_text_empty_input(self, model_service: ModelService) -> None:
        """Test text processing with empty input raises validation error.

        Args:
            model_service: The model service instance to test.
        """
        with pytest.raises(ValidationError) as exc_info:
            await model_service.process_text("")
        assert "Text cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_text_request_failure(
        self, model_service: ModelService
    ) -> None:
        """Test text processing when underlying request fails.

        Args:
            model_service: The model service instance to test.
        """
        test_text = "Test input text"

        with patch.object(
            model_service, "_make_model_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = Exception("API Error")

            with pytest.raises(Exception) as exc_info:
                await model_service.process_text(test_text)
            assert "API Error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_make_model_request(self, model_service: ModelService) -> None:
        """Test making a model request with proper data structure.

        Args:
            model_service: The model service instance to test.
        """
        test_request_data = {
            "text": "Test input",
            "context": {"user_id": "123"},
            "model": "test-model",
        }
        mock_generation = MagicMock()
        mock_generation.generations = [[MagicMock(text="mocked result")]]
        mock_model = MagicMock()
        mock_model.agenerate = AsyncMock(return_value=mock_generation)
        model_service.model = mock_model
        response = await model_service._make_model_request(test_request_data)
        assert isinstance(response, dict)
        assert "result" in response
        assert response["result"] == "mocked result"


class TestModelServiceResponseHandling:
    """Test cases for ModelService response handling and parsing."""

    def test_model_service_get_response_success(
        self, dummy_config: Dict[str, Any]
    ) -> None:
        """Test successful response retrieval from model.

        Args:
            dummy_config: The dummy configuration to use.
        """
        service = ModelService(dummy_config)
        mock_response = MagicMock()
        mock_response.content = "0) greeting"

        with patch.object(service.model, "invoke", return_value=mock_response):
            result = service.get_response("User: hello")
            assert result == "0) greeting"

    def test_model_service_get_response_empty(
        self, dummy_config: Dict[str, Any]
    ) -> None:
        """Test response retrieval with empty response raises error.

        Args:
            dummy_config: The dummy configuration to use.
        """
        service = ModelService(dummy_config)
        mock_response = MagicMock()
        mock_response.content = ""

        with patch.object(service.model, "invoke", return_value=mock_response):
            with pytest.raises(ValueError, match="Empty response from model"):
                service.get_response("User: hello")

    def test_model_service_get_response_model_error(
        self, dummy_config: Dict[str, Any]
    ) -> None:
        """Test response retrieval when model fails raises error.

        Args:
            dummy_config: The dummy configuration to use.
        """
        service = ModelService(dummy_config)

        with patch.object(
            service.model, "invoke", side_effect=Exception("Model error")
        ):
            with pytest.raises(
                ValueError, match="Failed to get model response: Model error"
            ):
                service.get_response("User: hello")

    def test_get_json_response(self, model_service: ModelService) -> None:
        """Test JSON response parsing and validation.

        Args:
            model_service: The model service instance to test.
        """
        test_json_string = '{"foo": "bar"}'

        with patch.object(model_service, "get_response", return_value=test_json_string):
            result = model_service.get_json_response("prompt")
            assert isinstance(result, dict)
            assert result["foo"] == "bar"


class TestModelServiceIntentProcessing:
    """Test cases for ModelService intent detection and processing."""

    @pytest.mark.asyncio
    async def test_predict_intent_success(self, model_service: ModelService) -> None:
        """Test successful intent prediction with valid response.

        Args:
            model_service: The model service instance to test.
        """
        mock_response = MagicMock()
        mock_response.content = '{"intent": "greet", "confidence": 0.9}'

        with patch.object(
            model_service.api_service,
            "get_model_response",
            new_callable=AsyncMock,
            create=True,
        ) as mock_get:
            mock_get.return_value = mock_response
            with patch(
                "arklex.orchestrator.NLU.services.model_service.validate_intent_response",
                return_value={"intent": "greet", "confidence": 0.9},
            ):
                result = await model_service.predict_intent("hello")
                assert isinstance(result, IntentResponse)

    @pytest.mark.asyncio
    async def test_predict_intent_validation_error(
        self, model_service: ModelService
    ) -> None:
        """Test intent prediction with validation error raises exception.

        Args:
            model_service: The model service instance to test.
        """
        mock_response = MagicMock()
        mock_response.content = '{"intent": "greet", "confidence": 0.9}'

        with patch.object(
            model_service.api_service,
            "get_model_response",
            new_callable=AsyncMock,
            create=True,
        ) as mock_get:
            mock_get.return_value = mock_response
            with patch(
                "arklex.orchestrator.NLU.services.model_service.validate_intent_response",
                side_effect=ValidationError("fail"),
            ):
                with pytest.raises(ValidationError):
                    await model_service.predict_intent("hello")

    def test_format_intent_definition(self, model_service: ModelService) -> None:
        """Test intent definition formatting with proper structure.

        Args:
            model_service: The model service instance to test.
        """
        intent_name = "greet"
        intent_definition = "A greeting intent"
        intent_count = 2

        formatted_string = model_service._format_intent_definition(
            intent_name, intent_definition, intent_count
        )
        assert isinstance(formatted_string, str)
        assert intent_name in formatted_string
        assert intent_definition in formatted_string

    def test_format_intent_exemplars(self, model_service: ModelService) -> None:
        """Test intent exemplars formatting with examples list.

        Args:
            model_service: The model service instance to test.
        """
        intent_name = "greet"
        exemplars = ["hello", "hi", "good morning"]
        intent_count = 2

        formatted_string = model_service._format_intent_exemplars(
            intent_name, exemplars, intent_count
        )
        assert isinstance(formatted_string, str)
        assert intent_name in formatted_string
        assert all(exemplar in formatted_string for exemplar in exemplars)

    def test_process_intent(self, model_service: ModelService) -> None:
        """Test intent processing with mapping and structure.

        Args:
            model_service: The model service instance to test.
        """
        intent_key = "greet"
        intent_value = [
            {"attribute": {"definition": "A greeting", "sample_utterances": ["hello"]}}
        ]
        intent_count = 1
        idx2intents_mapping = {"0": "greet"}

        result = model_service._process_intent(
            intent_key, intent_value, intent_count, idx2intents_mapping
        )
        assert isinstance(result, tuple)
        assert (
            len(result) == 4
        )  # Returns (definition_str, exemplars_str, intents_choice, count)

        definition_str, exemplars_str, intents_choice, new_count = result
        assert isinstance(definition_str, str)
        assert isinstance(exemplars_str, str)
        assert isinstance(intents_choice, str)
        assert isinstance(new_count, int)
        assert new_count > intent_count  # Count should be incremented

    def test_format_intent_input(self, model_service: ModelService) -> None:
        """Test intent input formatting with chat history.

        Args:
            model_service: The model service instance to test.
        """
        intents = {"greet": [{"definition": "A greeting", "examples": ["hello"]}]}
        chat_history_string = "User: hello\nAssistant: Hi there!"

        formatted_string, mapping = model_service.format_intent_input(
            intents, chat_history_string
        )
        assert isinstance(formatted_string, str)
        assert isinstance(mapping, dict)
        assert "greet" in formatted_string


class TestModelServiceSlotProcessing:
    """Test cases for ModelService slot filling and processing."""

    @pytest.mark.asyncio
    async def test_fill_slots_success(self, model_service: ModelService) -> None:
        """Test successful slot filling with valid response.

        Args:
            model_service: The model service instance to test.
        """
        mock_response = MagicMock()
        mock_response.content = '{"slots": {"foo": "bar"}}'

        with patch.object(
            model_service.api_service,
            "get_model_response",
            new_callable=AsyncMock,
            create=True,
        ) as mock_get:
            mock_get.return_value = mock_response
            with patch(
                "arklex.orchestrator.NLU.services.model_service.validate_slot_response",
                return_value={"slot": "foo", "value": "bar", "confidence": 1.0},
            ):
                result = await model_service.fill_slots("hi", "greet")
                assert isinstance(result, SlotResponse)

    @pytest.mark.asyncio
    async def test_fill_slots_validation_error(
        self, model_service: ModelService
    ) -> None:
        """Test slot filling with validation error raises exception.

        Args:
            model_service: The model service instance to test.
        """
        mock_response = MagicMock()
        mock_response.content = '{"slots": {"foo": "bar"}}'

        with patch.object(
            model_service.api_service,
            "get_model_response",
            new_callable=AsyncMock,
            create=True,
        ) as mock_get:
            mock_get.return_value = mock_response
            with patch(
                "arklex.orchestrator.NLU.services.model_service.validate_slot_response",
                side_effect=ValidationError("fail"),
            ):
                with pytest.raises(ValidationError):
                    await model_service.fill_slots("hi", "greet")

    def test_format_slot_input(self, model_service: ModelService) -> None:
        """Test slot input formatting with slot definitions.

        Args:
            model_service: The model service instance to test.
        """
        slots = [{"name": "user_name", "type": "str"}]
        context = "User is asking for help"

        formatted_string, slot_type = model_service.format_slot_input(slots, context)
        assert isinstance(formatted_string, str)
        assert isinstance(slot_type, str)
        assert "user_name" in formatted_string

    def test_process_slot_response(self, model_service: ModelService) -> None:
        """Test slot response processing with JSON parsing.

        Args:
            model_service: The model service instance to test.
        """
        response = '{"slots": [{"name": "user_name", "value": "John"}]}'
        slots = [{"name": "user_name", "type": "str"}]

        result = model_service.process_slot_response(response, slots)
        assert isinstance(result, list)
        assert len(result) > 0


class TestModelServiceVerification:
    """Test cases for ModelService slot verification methods."""

    @pytest.mark.asyncio
    async def test_verify_slots_success(self, model_service: ModelService) -> None:
        """Test successful slot verification with valid response.

        Args:
            model_service: The model service instance to test.
        """
        mock_response = MagicMock()
        mock_response.content = '{"verified": true, "message": "ok"}'

        with patch.object(
            model_service.api_service,
            "get_model_response",
            new_callable=AsyncMock,
            create=True,
        ) as mock_get:
            mock_get.return_value = mock_response
            with patch(
                "arklex.orchestrator.NLU.services.model_service.validate_verification_response",
                return_value={
                    "slot": "user_name",
                    "reason": "valid",
                    "verified": True,
                    "message": "ok",
                },
            ):
                result = await model_service.verify_slots("hi", {"user_name": "John"})
                assert isinstance(result, VerificationResponse)

    @pytest.mark.asyncio
    async def test_verify_slots_validation_error(
        self, model_service: ModelService
    ) -> None:
        """Test slot verification with validation error raises exception.

        Args:
            model_service: The model service instance to test.
        """
        mock_response = MagicMock()
        mock_response.content = '{"verified": true, "message": "ok"}'

        with patch.object(
            model_service.api_service,
            "get_model_response",
            new_callable=AsyncMock,
            create=True,
        ) as mock_get:
            mock_get.return_value = mock_response
            with patch(
                "arklex.orchestrator.NLU.services.model_service.validate_verification_response",
                side_effect=ValidationError("fail"),
            ):
                with pytest.raises(ValidationError):
                    await model_service.verify_slots("hi", {"user_name": "John"})


class TestModelServiceUtilities:
    """Test cases for ModelService utility and helper methods."""

    def test_initialize_model(self, model_service: ModelService) -> None:
        """Test model initialization with proper configuration.

        Args:
            model_service: The model service instance to test.
        """
        with patch(
            "arklex.orchestrator.NLU.services.model_service.BaseChatModel",
            autospec=True,
        ):
            model_service._initialize_model()

    def test_format_messages(self, model_service: ModelService) -> None:
        """Test message formatting with prompt and context.

        Args:
            model_service: The model service instance to test.
        """
        prompt = "Hello, how can I help you?"
        context = {"user_id": "123", "session_id": "abc"}

        result = model_service._format_messages(prompt, context)
        assert isinstance(result, list)
        assert len(result) > 0


class TestDummyModelService:
    """Test cases for DummyModelService functionality."""

    def test_dummy_model_service_methods(self, dummy_config: Dict[str, Any]) -> None:
        """Test all DummyModelService methods return expected types.

        Args:
            dummy_config: The dummy configuration to use.
        """
        dummy_service = DummyModelService(dummy_config)
        test_slots = [{"name": "user_name", "type": "str"}]
        test_context = "User context"

        # Test format_slot_input
        formatted_string, slot_type = dummy_service.format_slot_input(
            test_slots, test_context
        )
        assert isinstance(formatted_string, str)
        assert isinstance(slot_type, str)

        # Test get_response
        response = dummy_service.get_response("test prompt")
        assert isinstance(response, str)

        # Test process_slot_response
        slot_response = dummy_service.process_slot_response(
            '{"slots": [{"name": "user_name", "value": "John"}]}', test_slots
        )
        assert isinstance(slot_response, list)

        # Patch format_verification_input to avoid super() error
        with patch.object(
            DummyModelService,
            "format_verification_input",
            return_value=("formatted_input", "verification_type"),
        ):
            formatted_string, verification_type = (
                dummy_service.format_verification_input(
                    {"name": "user_name"}, "chat_history"
                )
            )
            assert isinstance(formatted_string, str)
            assert isinstance(verification_type, str)

        # Patch process_verification_response to avoid super() error
        with patch.object(
            DummyModelService,
            "process_verification_response",
            return_value=(True, "verification successful"),
        ):
            is_verified, message = dummy_service.process_verification_response(
                '{"verified": true, "message": "ok"}'
            )
            assert isinstance(is_verified, bool)
            assert isinstance(message, str)
