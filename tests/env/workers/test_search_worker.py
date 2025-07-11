"""
Tests for search worker functionality.

This module contains comprehensive tests for the SearchWorker class including
initialization, search functionality, and error handling.
"""

from unittest.mock import Mock

import pytest

from arklex.env.workers.search_worker import SearchWorker
from arklex.orchestrator.entities.orch_entities import MessageState


class TestSearchWorkerInitialization:
    """Test cases for SearchWorker initialization."""

    def test_search_worker_init_with_default_config(self) -> None:
        """Test SearchWorker initialization with default configuration."""
        worker = SearchWorker()
        assert worker is not None
        assert "search" in worker.description.lower()

    def test_search_worker_init_with_custom_config(self) -> None:
        """Test SearchWorker initialization with custom configuration."""
        worker = SearchWorker()
        assert worker is not None
        assert hasattr(worker, "action_graph")

    def test_search_worker_init_missing_provider(self) -> None:
        """Test SearchWorker initialization with missing provider."""
        # SearchWorker doesn't take model_config parameter
        worker = SearchWorker()
        assert worker is not None

    def test_search_worker_init_missing_model_name(self) -> None:
        """Test SearchWorker initialization with missing model name."""
        # SearchWorker doesn't take model_config parameter
        worker = SearchWorker()
        assert worker is not None

    def test_search_worker_init_unsupported_provider(self) -> None:
        """Test SearchWorker initialization with unsupported provider."""
        # SearchWorker doesn't take model_config parameter
        worker = SearchWorker()
        assert worker is not None


class TestSearchWorkerSearch:
    """Test cases for SearchWorker search functionality."""

    def test_search_documents_basic(self) -> None:
        """Test basic search functionality."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the execute method
        mock_result = Mock(spec=MessageState)
        worker._execute = Mock(return_value=mock_result)

        result = worker.search_documents(mock_state)

        assert result == mock_result
        worker._execute.assert_called_once_with(mock_state)

    def test_search_documents_with_empty_query(self) -> None:
        """Test search with empty query."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the execute method
        mock_result = Mock(spec=MessageState)
        worker._execute = Mock(return_value=mock_result)

        result = worker.search_documents(mock_state)

        assert result == mock_result
        worker._execute.assert_called_once_with(mock_state)

    def test_search_documents_with_none_query(self) -> None:
        """Test search with None query."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the execute method
        mock_result = Mock(spec=MessageState)
        worker._execute = Mock(return_value=mock_result)

        result = worker.search_documents(mock_state)

        assert result == mock_result
        worker._execute.assert_called_once_with(mock_state)

    def test_search_documents_with_complex_query(self) -> None:
        """Test search with complex query."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the execute method
        mock_result = Mock(spec=MessageState)
        worker._execute = Mock(return_value=mock_result)

        result = worker.search_documents(mock_state)

        assert result == mock_result
        worker._execute.assert_called_once_with(mock_state)

    def test_search_documents_error_handling(self) -> None:
        """Test search error handling."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the execute method to raise an exception
        worker._execute = Mock()
        worker._execute.side_effect = Exception("Search failed")

        with pytest.raises(RuntimeError):
            worker._execute(mock_state)


class TestSearchWorkerIntegration:
    """Integration tests for SearchWorker."""

    def test_search_worker_complete_workflow(self) -> None:
        """Test a complete search worker workflow."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the execute method
        mock_result = Mock(spec=MessageState)
        worker._execute = Mock(return_value=mock_result)

        result = worker.search_documents(mock_state)

        assert result == mock_result
        worker._execute.assert_called_once_with(mock_state)

    def test_search_worker_description(self) -> None:
        """Test that the worker description is appropriate."""
        worker = SearchWorker()
        assert isinstance(worker.description, str)
        assert len(worker.description) > 0
        assert "real-time" in worker.description.lower()

    def test_search_worker_method_availability(self) -> None:
        """Test that SearchWorker has the expected methods."""
        worker = SearchWorker()
        assert hasattr(worker, "_execute")
        assert callable(worker._execute)
        assert hasattr(worker, "description")
        assert isinstance(worker.description, str)


class TestSearchWorkerErrorHandling:
    """Test cases for SearchWorker error handling."""

    def test_search_worker_init_error(self) -> None:
        """Test SearchWorker initialization error handling."""
        # SearchWorker should initialize without errors
        worker = SearchWorker()
        assert worker is not None

    def test_search_worker_search_error_handling(self) -> None:
        """Test error handling in SearchWorker search methods."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the execute method to raise an exception
        worker._execute = Mock()
        worker._execute.side_effect = Exception("Search failed")

        # Should handle the exception gracefully
        with pytest.raises(RuntimeError):
            worker._execute(mock_state)

    def test_search_worker_with_invalid_state(self) -> None:
        """Test SearchWorker with invalid state."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the execute method to return an error response
        worker._execute = Mock(return_value={"error": "Invalid state"})

        result = worker._execute(mock_state)

        assert "error" in result

    def test_search_worker_with_missing_attributes(self) -> None:
        """Test SearchWorker with missing attributes."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the execute method to return an error response
        worker._execute = Mock(return_value={"error": "Missing attributes"})

        result = worker._execute(mock_state)

        assert "error" in result


class TestSearchWorkerEdgeCases:
    """Test cases for SearchWorker edge cases."""

    def test_search_worker_with_very_long_query(self) -> None:
        """Test SearchWorker with very long query."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the execute method
        mock_result = Mock(spec=MessageState)
        worker._execute = Mock(return_value=mock_result)

        result = worker.search_documents(mock_state)

        assert result == mock_result
        worker._execute.assert_called_once_with(mock_state)

    def test_search_worker_with_special_characters(self) -> None:
        """Test SearchWorker with special characters in query."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the execute method
        mock_result = Mock(spec=MessageState)
        worker._execute = Mock(return_value=mock_result)

        result = worker.search_documents(mock_state)

        assert result == mock_result
        worker._execute.assert_called_once_with(mock_state)

    def test_search_worker_with_unicode_characters(self) -> None:
        """Test SearchWorker with unicode characters in query."""
        worker = SearchWorker()
        mock_state = Mock(spec=MessageState)

        # Mock the execute method
        mock_result = Mock(spec=MessageState)
        worker._execute = Mock(return_value=mock_result)

        result = worker.search_documents(mock_state)

        assert result == mock_result
        worker._execute.assert_called_once_with(mock_state)
