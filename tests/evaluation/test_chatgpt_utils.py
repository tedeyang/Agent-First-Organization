"""Tests for the chatgpt_utils module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from arklex.evaluation.chatgpt_utils import (
    chatgpt_chatbot,
    query_chatbot,
    filter_convo,
    flip_hist,
    format_chat_history_str,
    flip_hist_content_only,
)


class TestChatGPTUtils:
    """Test cases for chatgpt_utils module."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        client = Mock()
        # Simulate OpenAI API response structure
        mock_choice = Mock()
        mock_choice.message = Mock(content="Test response")
        client.chat.completions.create.return_value = Mock(choices=[mock_choice])
        return client

    @patch("arklex.evaluation.chatgpt_utils.create_client")
    def test_chatgpt_chatbot_with_messages(self, mock_create_client, mock_client):
        """Test chatgpt_chatbot with messages list."""
        # Setup
        mock_create_client.return_value = mock_client
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        # Execute
        result = chatgpt_chatbot(messages, client=mock_client)

        # Assert
        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch("arklex.evaluation.chatgpt_utils.create_client")
    def test_chatgpt_chatbot_with_string(self, mock_create_client, mock_client):
        """Test chatgpt_chatbot with string input."""
        # Setup
        mock_create_client.return_value = mock_client
        prompt = "Hello, how are you?"

        # Execute
        result = chatgpt_chatbot(prompt, client=mock_client)

        # Assert
        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch("arklex.evaluation.chatgpt_utils.create_client")
    def test_chatgpt_chatbot_with_model_parameter(
        self, mock_create_client, mock_client
    ):
        """Test chatgpt_chatbot with model parameter."""
        # Setup
        mock_create_client.return_value = mock_client
        messages = [{"role": "user", "content": "Hello"}]

        # Execute
        result = chatgpt_chatbot(messages, model="gpt-3.5-turbo", client=mock_client)

        # Assert
        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch("arklex.evaluation.chatgpt_utils.create_client")
    def test_query_chatbot(self, mock_create_client, mock_client):
        """Test query_chatbot function."""
        # Setup
        mock_create_client.return_value = mock_client
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?", "intent": "question"},
            {"role": "assistant", "content": "I'm good"},
        ]

        # Mock the requests.post call
        with patch("arklex.evaluation.chatgpt_utils.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": "Test response"}
            mock_post.return_value = mock_response

            # Execute - query_chatbot requires model_api, history, params, env_config
            result = query_chatbot(
                model_api="http://test-api.com",
                history=conversation,
                params={"param1": "value1"},
                env_config={"workers": [], "tools": []},
            )

            # Assert
            assert isinstance(result, dict)
            assert "response" in result

    def test_filter_convo(self):
        """Test filter_convo function."""
        # Setup
        conversation = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?\nThis is extra content"},
            {"role": "assistant", "content": "I'm good"},
        ]

        # Execute
        result = filter_convo(conversation)

        # Assert - filter_convo skips first 2 turns and truncates user messages at \n
        assert len(result) == 3  # The last 3 turns (after skipping first 2)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hi there"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "How are you?"  # Truncated at \n
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "I'm good"

    def test_flip_hist(self):
        """Test flip_hist function."""
        # Setup
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good"},
        ]

        # Execute
        result = flip_hist(conversation)

        # Assert
        assert len(result) == 4
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"

    def test_format_chat_history_str(self):
        """Test format_chat_history_str function."""
        # Setup
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good"},
        ]

        # Execute
        result = format_chat_history_str(conversation)

        # Assert - format uses uppercase role names
        assert isinstance(result, str)
        assert "USER: Hello" in result
        assert "ASSISTANT: Hi there" in result

    def test_flip_hist_content_only(self):
        """Test flip_hist_content_only function."""
        # Setup
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good"},
        ]

        # Execute
        result = flip_hist_content_only(conversation)

        # Assert
        assert len(result) == 4
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"

    @patch("arklex.evaluation.chatgpt_utils.create_client")
    def test_chatgpt_chatbot_error_handling(self, mock_create_client, mock_client):
        """Test chatgpt_chatbot error handling."""
        # Setup
        mock_create_client.return_value = mock_client
        # Simulate an exception in the OpenAI client
        mock_client.chat.completions.create.side_effect = Exception("API error")
        conversation = [
            {"content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        # Execute and assert
        with pytest.raises(Exception):
            chatgpt_chatbot(conversation, client=mock_client)

    def test_filter_convo_empty_conversation(self):
        """Test filter_convo with empty conversation."""
        # Setup
        conversation = []

        # Execute
        result = filter_convo(conversation)

        # Assert
        assert result == []

    def test_format_chat_history_str_empty_conversation(self):
        """Test format_chat_history_str with empty conversation."""
        # Setup
        conversation = []

        # Execute
        result = format_chat_history_str(conversation)

        # Assert
        assert result == ""

    def test_flip_hist_empty_conversation(self):
        """Test flip_hist with empty conversation."""
        # Setup
        conversation = []

        # Execute
        result = flip_hist(conversation)

        # Assert
        assert result == []

    def test_flip_hist_content_only_empty_conversation(self):
        """Test flip_hist_content_only with empty conversation."""
        # Setup
        conversation = []

        # Execute
        result = flip_hist_content_only(conversation)

        # Assert
        assert result == []

    def test_filter_convo_with_missing_role(self):
        """Test filter_convo with messages missing role field."""
        # Setup
        conversation = [
            {"content": "Hello"},  # Missing role
            {"role": "assistant", "content": "Hi there"},
        ]

        # Execute
        result = filter_convo(conversation)

        # Assert - messages without role are filtered out
        assert len(result) == 0  # No messages with intent
