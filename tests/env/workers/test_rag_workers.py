"""
Tests for RAG workers functionality.

This module contains comprehensive tests for all RAG worker classes including
FaissRAGWorker, MilvusRAGWorker, and RAGMessageWorker.
"""

from unittest.mock import Mock

from arklex.env.workers.faiss_rag_worker import FaissRAGWorker
from arklex.env.workers.milvus_rag_worker import MilvusRAGWorker
from arklex.env.workers.rag_message_worker import RAGMessageWorker
from arklex.orchestrator.entities.orch_entities import MessageState


class TestFaissRAGWorker:
    """Test cases for FaissRAGWorker."""

    def test_faiss_rag_worker_init(self) -> None:
        """Test FaissRAGWorker initialization."""
        worker = FaissRAGWorker()
        assert worker.stream_response is True
        assert hasattr(worker, "action_graph")

    def test_faiss_rag_worker_with_custom_config(self) -> None:
        """Test FaissRAGWorker initialization with custom stream configuration."""
        worker = FaissRAGWorker(stream_response=False)
        assert worker.stream_response is False
        assert hasattr(worker, "action_graph")

    def test_faiss_rag_worker_choose_tool_generator(self) -> None:
        """Test FaissRAGWorker choose_tool_generator method."""
        worker = FaissRAGWorker()
        mock_state = Mock(spec=MessageState)

        # Test with streaming enabled and state is stream
        mock_state.is_stream = True
        result = worker.choose_tool_generator(mock_state)
        assert result == "stream_tool_generator"

        # Test with streaming enabled but state is not stream
        mock_state.is_stream = False
        result = worker.choose_tool_generator(mock_state)
        assert result == "tool_generator"

        # Test with streaming disabled
        worker.stream_response = False
        result = worker.choose_tool_generator(mock_state)
        assert result == "tool_generator"


class TestMilvusRAGWorker:
    """Test cases for MilvusRAGWorker."""

    def test_milvus_rag_worker_init(self) -> None:
        """Test MilvusRAGWorker initialization."""
        worker = MilvusRAGWorker()
        assert worker.stream_response is True
        assert worker.tags == {}
        assert hasattr(worker, "action_graph")

    def test_milvus_rag_worker_with_custom_config(self) -> None:
        """Test MilvusRAGWorker initialization with custom configuration."""
        worker = MilvusRAGWorker(stream_response=False)
        assert worker.stream_response is False
        assert worker.tags == {}

    def test_milvus_rag_worker_choose_tool_generator(self) -> None:
        """Test MilvusRAGWorker choose_tool_generator method."""
        worker = MilvusRAGWorker()
        mock_state = Mock(spec=MessageState)

        # Test with streaming enabled and state is stream
        mock_state.is_stream = True
        result = worker.choose_tool_generator(mock_state)
        assert result == "stream_tool_generator"

        # Test with streaming enabled but state is not stream
        mock_state.is_stream = False
        result = worker.choose_tool_generator(mock_state)
        assert result == "tool_generator"

        # Test with streaming disabled
        worker.stream_response = False
        result = worker.choose_tool_generator(mock_state)
        assert result == "tool_generator"


class TestRAGMessageWorker:
    """Test cases for RAGMessageWorker."""

    def test_rag_message_worker_init(self) -> None:
        """Test RAGMessageWorker initialization."""
        worker = RAGMessageWorker()
        assert hasattr(worker, "model_config")
        assert hasattr(worker, "llm")

    def test_rag_message_worker_with_custom_config(self) -> None:
        """Test RAGMessageWorker initialization with custom configuration."""
        custom_config = {
            "llm_provider": "openai",
            "model_type_or_path": "gpt-3.5-turbo",
            "api_key": "test-api-key",
        }
        worker = RAGMessageWorker(model_config=custom_config)
        assert worker.model_config == custom_config

    def test_rag_message_worker_search_documents(self) -> None:
        """Test RAGMessageWorker search_documents method."""
        worker = RAGMessageWorker()
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        mock_orchestrator_message.attribute = {"query": "test query"}
        mock_state.orchestrator_message = mock_orchestrator_message

        result = worker.search_documents(mock_state)

        assert "test query" in result

    def test_rag_message_worker_search_documents_no_query(self) -> None:
        """Test RAGMessageWorker search_documents method with no query."""
        worker = RAGMessageWorker()
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        mock_orchestrator_message.attribute = {}
        mock_state.orchestrator_message = mock_orchestrator_message

        result = worker.search_documents(mock_state)

        assert "No query provided" in result

    def test_rag_message_worker_generate_response(self) -> None:
        """Test RAGMessageWorker generate_response method."""
        worker = RAGMessageWorker()
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        mock_orchestrator_message.attribute = {"query": "test query"}
        mock_state.orchestrator_message = mock_orchestrator_message

        result = worker.generate_response(mock_state)

        assert "test query" in result

    def test_rag_message_worker_generate_response_no_query(self) -> None:
        """Test RAGMessageWorker generate_response method with no query."""
        worker = RAGMessageWorker()
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        mock_orchestrator_message.attribute = {}
        mock_state.orchestrator_message = mock_orchestrator_message

        result = worker.generate_response(mock_state)

        assert "No query provided" in result


class TestRAGWorkersIntegration:
    """Integration tests for RAG workers."""

    def test_rag_workers_description_comparison(self) -> None:
        """Test that RAG workers have appropriate descriptions."""
        # Create workers
        faiss_worker = FaissRAGWorker()
        milvus_worker = MilvusRAGWorker()
        rag_message_worker = RAGMessageWorker()

        # Test that all workers have descriptions
        assert isinstance(faiss_worker.description, str)
        assert isinstance(milvus_worker.description, str)
        assert isinstance(rag_message_worker.description, str)

        # Test that descriptions are not empty
        assert len(faiss_worker.description) > 0
        assert len(milvus_worker.description) > 0
        assert len(rag_message_worker.description) > 0

        # Test that descriptions contain relevant keywords
        assert "internal" in faiss_worker.description.lower()
        assert "internal" in milvus_worker.description.lower()
        assert "rag" in rag_message_worker.description.lower()

    def test_rag_workers_initialization_consistency(self) -> None:
        """Test that all RAG workers initialize consistently."""
        # Create workers
        faiss_worker = FaissRAGWorker()
        milvus_worker = MilvusRAGWorker()
        rag_message_worker = RAGMessageWorker()

        # Test that all workers have action_graph
        assert hasattr(faiss_worker, "action_graph")
        assert hasattr(milvus_worker, "action_graph")
        # RAGMessageWorker doesn't have action_graph by default

        # Test that all workers have descriptions
        assert hasattr(faiss_worker, "description")
        assert hasattr(milvus_worker, "description")
        assert hasattr(rag_message_worker, "description")

        # Test that descriptions are not empty
        assert len(faiss_worker.description) > 0
        assert len(milvus_worker.description) > 0
        assert len(rag_message_worker.description) > 0

    def test_rag_workers_method_availability(self) -> None:
        """Test that RAG workers have the expected methods."""
        # Create workers
        faiss_worker = FaissRAGWorker()
        milvus_worker = MilvusRAGWorker()
        rag_message_worker = RAGMessageWorker()

        # Test that all workers have _execute method
        assert hasattr(faiss_worker, "_execute")
        assert hasattr(milvus_worker, "_execute")
        assert hasattr(rag_message_worker, "_execute")

        # Test that RAGMessageWorker has search_documents and generate_response methods
        assert hasattr(rag_message_worker, "search_documents")
        assert hasattr(rag_message_worker, "generate_response")

        # Test that FaissRAGWorker and MilvusRAGWorker have choose_tool_generator method
        assert hasattr(faiss_worker, "choose_tool_generator")
        assert hasattr(milvus_worker, "choose_tool_generator")


class TestRAGWorkersErrorHandling:
    """Test error handling for RAG workers."""

    def test_faiss_rag_worker_init_error(self) -> None:
        """Test FaissRAGWorker initialization error handling."""
        # This should not raise any errors
        worker = FaissRAGWorker()
        assert worker is not None

    def test_milvus_rag_worker_init_error(self) -> None:
        """Test MilvusRAGWorker initialization error handling."""
        # This should not raise any errors
        worker = MilvusRAGWorker()
        assert worker is not None

    def test_rag_message_worker_init_error(self) -> None:
        """Test RAGMessageWorker initialization error handling."""
        # This should not raise any errors
        worker = RAGMessageWorker()
        assert worker is not None

    def test_rag_workers_search_error_handling(self) -> None:
        """Test RAG workers search error handling."""
        # Test that workers handle missing orchestrator_message gracefully
        rag_message_worker = RAGMessageWorker()
        mock_state = Mock(spec=MessageState)
        mock_state.orchestrator_message = None

        # This should not raise an error
        result = rag_message_worker.search_documents(mock_state)
        assert "No query provided" in result

        # Test with orchestrator_message but no attribute
        mock_state.orchestrator_message = Mock()
        mock_state.orchestrator_message.attribute = None

        result = rag_message_worker.search_documents(mock_state)
        assert "No query provided" in result

        # Test with orchestrator_message and empty attribute dict
        mock_state.orchestrator_message.attribute = {}

        result = rag_message_worker.search_documents(mock_state)
        assert "No query provided" in result
