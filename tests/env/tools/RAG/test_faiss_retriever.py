from unittest.mock import Mock, mock_open, patch

import pytest
from langchain_core.documents import Document

from arklex.env.tools.RAG.retrievers.faiss_retriever import (
    FaissRetrieverExecutor,
    RetrieveEngine,
)
from arklex.utils.graph_state import LLMConfig, MessageState


@pytest.fixture
def mock_documents() -> list[Document]:
    return [
        Document(
            page_content="Test document 1", metadata={"source": "test1.txt", "page": 1}
        ),
        Document(
            page_content="Test document 2", metadata={"source": "test2.txt", "page": 2}
        ),
    ]


@pytest.fixture
def mock_llm_config() -> Mock:
    config = Mock(spec=LLMConfig)
    config.llm_provider = "openai"
    config.model_type_or_path = "text-embedding-ada-002"
    return config


@pytest.fixture
def faiss_retriever(
    mock_documents: list[Document], mock_llm_config: Mock
) -> FaissRetrieverExecutor:
    with (
        patch("arklex.env.tools.RAG.retrievers.faiss_retriever.OpenAIEmbeddings"),
        patch("arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS"),
    ):
        retriever = FaissRetrieverExecutor(
            documents=mock_documents,
            llm_config=mock_llm_config,
            database_path="/tmp/test_db",
        )
        return retriever


class TestFaissRetrieverExecutor:
    """Test the FaissRetrieverExecutor class."""

    def test_initialization(
        self, mock_documents: list[Document], mock_llm_config: Mock
    ) -> None:
        """Test FaissRetrieverExecutor initialization."""
        with (
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.OpenAIEmbeddings"
            ) as mock_embeddings,
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS"
            ) as mock_faiss,
        ):
            mock_embeddings_instance = Mock()
            mock_embeddings.return_value = mock_embeddings_instance

            mock_faiss_instance = Mock()
            mock_faiss.from_documents.return_value = mock_faiss_instance
            mock_faiss_instance.as_retriever.return_value = Mock()

            retriever = FaissRetrieverExecutor(
                documents=mock_documents,
                llm_config=mock_llm_config,
                database_path="/tmp/test_db",
            )

            assert retriever.documents == mock_documents
            assert retriever.llm_config == mock_llm_config
            assert retriever.database_path == "/tmp/test_db"
            assert retriever.embedding_model is not None
            assert retriever.retriever is not None

    def test_initialization_anthropic_provider(
        self, mock_documents: list[Document]
    ) -> None:
        """Test initialization with Anthropic provider."""
        config = Mock(spec=LLMConfig)
        config.llm_provider = "anthropic"
        config.model_type_or_path = "claude-3-sonnet-20240229"

        with (
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.AnthropicEmbeddings"
            ) as mock_embeddings,
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS"
            ) as mock_faiss,
        ):
            mock_embeddings_instance = Mock()
            mock_embeddings.return_value = mock_embeddings_instance

            mock_faiss_instance = Mock()
            mock_faiss.from_documents.return_value = mock_faiss_instance
            mock_faiss_instance.as_retriever.return_value = Mock()

            retriever = FaissRetrieverExecutor(
                documents=mock_documents,
                llm_config=config,
                database_path="/tmp/test_db",
            )

            assert retriever.embedding_model is not None
            assert retriever.retriever is not None

    def test_init_retriever(self, faiss_retriever: FaissRetrieverExecutor) -> None:
        """Test _init_retriever method."""
        with patch(
            "arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS.from_documents"
        ) as mock_from_documents:
            mock_faiss_instance = Mock()
            mock_from_documents.return_value = mock_faiss_instance
            mock_faiss_instance.as_retriever.return_value = Mock()

            faiss_retriever._init_retriever()

            mock_from_documents.assert_called_once()
            mock_faiss_instance.as_retriever.assert_called_once_with(k=5)

    def test_retrieve_w_score_default_k(
        self, faiss_retriever: FaissRetrieverExecutor
    ) -> None:
        """Test retrieve_w_score with default k value."""
        mock_docs_and_scores = [
            (Document(page_content="doc1"), 0.8),
            (Document(page_content="doc2"), 0.6),
        ]
        faiss_retriever.retriever.similarity_search_with_score = Mock(
            return_value=mock_docs_and_scores
        )

        result = faiss_retriever.retrieve_w_score("test query")

        assert result == mock_docs_and_scores
        faiss_retriever.retriever.similarity_search_with_score.assert_called_once_with(
            "test query", k=5
        )

    def test_retrieve_w_score_custom_k(
        self, faiss_retriever: FaissRetrieverExecutor
    ) -> None:
        """Test retrieve_w_score with custom k value."""
        mock_docs_and_scores = [
            (Document(page_content="doc1"), 0.8),
            (Document(page_content="doc2"), 0.6),
        ]
        faiss_retriever.retriever.similarity_search_with_score = Mock(
            return_value=mock_docs_and_scores
        )

        result = faiss_retriever.retrieve_w_score("test query", k=10)

        assert result == mock_docs_and_scores
        faiss_retriever.retriever.similarity_search_with_score.assert_called_once_with(
            "test query", k=10
        )

    def test_search(self, faiss_retriever: FaissRetrieverExecutor) -> None:
        """Test search method."""
        mock_docs_and_scores = [
            (Document(page_content="doc1", metadata={"source": "test1.txt"}), 0.8),
            (Document(page_content="doc2", metadata={"source": "test2.txt"}), 0.6),
        ]
        faiss_retriever.retrieve_w_score = Mock(return_value=mock_docs_and_scores)

        result = faiss_retriever.search("test query")

        assert len(result) == 2
        assert result[0]["content"] == "doc1"
        assert result[0]["source"] == "test1.txt"
        assert result[0]["confidence"] == 0.8
        assert result[1]["content"] == "doc2"
        assert result[1]["source"] == "test2.txt"
        assert result[1]["confidence"] == 0.6

    def test_search_empty_results(
        self, faiss_retriever: FaissRetrieverExecutor
    ) -> None:
        """Test search method with empty results."""
        faiss_retriever.retrieve_w_score = Mock(return_value=[])

        result = faiss_retriever.search("test query")

        assert result == []

    @staticmethod
    def test_load_docs_success(mock_llm_config: Mock) -> None:
        """Test load_docs static method success."""
        mock_documents = [
            Document(page_content="doc1", metadata={"source": "test1.txt"}),
            Document(page_content="doc2", metadata={"source": "test2.txt"}),
        ]

        with (
            patch(
                "arklex.env.tools.RAG.retrievers.faiss_retriever.pickle.load"
            ) as mock_load,
            patch("builtins.open", mock_open(read_data="test data")),
        ):
            mock_load.return_value = mock_documents

            result = FaissRetrieverExecutor.load_docs(
                database_path="/tmp/test_db", llm_config=mock_llm_config
            )

            assert result == mock_documents

    @staticmethod
    def test_faiss_retriever_file_not_found() -> None:  # noqa: N802
        """Test load_docs with file not found."""
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            pytest.raises(FileNotFoundError),
        ):
            FaissRetrieverExecutor.load_docs(database_path="/tmp", llm_config=None)


class TestRetrieveEngine:
    """Test the RetrieveEngine class."""

    def test_faiss_retrieve(self) -> None:
        """Test faiss_retrieve static method."""
        mock_state = Mock(spec=MessageState)
        mock_state.user_message.message = "test query"

        mock_retriever = Mock()
        mock_retriever.search.return_value = [
            {"content": "doc1", "source": "test1.txt", "confidence": 0.8}
        ]

        with patch(
            "arklex.env.tools.RAG.retrievers.faiss_retriever.FaissRetrieverExecutor.load_docs"
        ) as mock_load_docs:
            mock_load_docs.return_value = [Document(page_content="doc1")]

            result = RetrieveEngine.faiss_retrieve(mock_state, "/tmp/test_db")

            assert result == [
                {"content": "doc1", "source": "test1.txt", "confidence": 0.8}
            ]
