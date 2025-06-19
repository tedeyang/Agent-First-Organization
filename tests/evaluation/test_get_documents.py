"""Tests for the get_documents module.

This module contains comprehensive test cases for document loading and processing
functionality, including domain information extraction, document loading with caching,
and handling different document types (web, file, text).
"""

import os
import pickle
from typing import Dict, Any, List
from unittest.mock import Mock, patch, mock_open

import pytest

from arklex.evaluation.get_documents import get_domain_info, load_docs
from arklex.utils.loader import CrawledObject, SourceType


class TestGetDocuments:
    """Test cases for get_documents module.

    This class contains comprehensive tests for document loading and processing,
    including domain information extraction and document loading from various sources.
    """

    def test_get_domain_info_with_summary(self) -> None:
        """Test get_domain_info function with summary document present."""
        # Setup
        documents: List[Dict[str, str]] = [
            {"URL": "http://example.com", "content": "Example content"},
            {"URL": "summary", "content": "Summary content"},
            {"URL": "http://test.com", "content": "Test content"},
        ]

        # Execute
        result = get_domain_info(documents)

        # Assert
        assert result == "Summary content"

    def test_get_domain_info_without_summary(self) -> None:
        """Test get_domain_info function without summary document."""
        # Setup
        documents: List[Dict[str, str]] = [
            {"URL": "http://example.com", "content": "Example content"},
            {"URL": "http://test.com", "content": "Test content"},
        ]

        # Execute
        result = get_domain_info(documents)

        # Assert
        assert result is None

    def test_get_domain_info_empty_list(self) -> None:
        """Test get_domain_info function with empty document list."""
        # Setup
        documents: List[Dict[str, str]] = []

        # Execute
        result = get_domain_info(documents)

        # Assert
        assert result is None

    @patch("pickle.load")
    @patch("arklex.evaluation.get_documents.os.path.exists")
    @patch("arklex.evaluation.get_documents.Loader")
    def test_load_docs_with_existing_cache(
        self, mock_loader_class: Mock, mock_exists: Mock, mock_pickle_load: Mock
    ) -> None:
        """Test load_docs function with existing cache file."""
        # Setup
        document_dir = "/test/dir"
        doc_config = {"rag_docs": [{"source": "test", "type": "url", "num": 5}]}
        limit = 10

        mock_exists.return_value = True
        mock_crawled_obj = Mock(spec=CrawledObject)
        mock_crawled_obj.source_type = SourceType.WEB
        mock_crawled_obj.to_dict.return_value = {"URL": "test.com", "content": "test"}
        mock_pickle_load.return_value = [mock_crawled_obj]

        mock_loader = Mock()
        mock_loader.get_candidates_websites.return_value = [mock_crawled_obj]
        mock_loader_class.return_value = mock_loader

        # Execute
        result = load_docs(document_dir, doc_config, limit)

        # Assert
        assert isinstance(result, list)

    @patch("pickle.load")
    @patch("arklex.evaluation.get_documents.os.path.exists")
    @patch("arklex.evaluation.get_documents.Loader")
    @patch("arklex.evaluation.get_documents.os.listdir")
    def test_load_docs_without_cache_url_type(
        self,
        mock_listdir: Mock,
        mock_loader_class: Mock,
        mock_exists: Mock,
        mock_pickle_load: Mock,
    ) -> None:
        """Test load_docs function without cache file, URL type documents."""
        # Setup
        document_dir = "/test/dir"
        doc_config = {
            "rag_docs": [{"source": "http://test.com", "type": "url", "num": 2}]
        }
        limit = 10

        mock_exists.return_value = False
        mock_listdir.return_value = ["file1.txt", "file2.txt"]

        mock_loader = Mock()
        mock_loader.get_all_urls.return_value = ["http://test.com", "http://test2.com"]
        mock_loader.to_crawled_url_objs.return_value = [Mock(spec=CrawledObject)]
        mock_loader.get_candidates_websites.return_value = [Mock(spec=CrawledObject)]
        mock_loader.save = Mock()
        mock_loader_class.return_value = mock_loader

        # Execute
        result = load_docs(document_dir, doc_config, limit)

        # Assert
        assert isinstance(result, list)

    @patch("pickle.load")
    @patch("arklex.evaluation.get_documents.os.path.exists")
    @patch("arklex.evaluation.get_documents.Loader")
    @patch("arklex.evaluation.get_documents.os.listdir")
    def test_load_docs_file_type(
        self,
        mock_listdir: Mock,
        mock_loader_class: Mock,
        mock_exists: Mock,
        mock_pickle_load: Mock,
    ) -> None:
        """Test load_docs function with file type documents."""
        # Setup
        document_dir = "/test/dir"
        doc_config = {"rag_docs": [{"source": "/test/files", "type": "file"}]}
        limit = 10

        mock_exists.return_value = False
        mock_listdir.return_value = ["file1.txt", "file2.txt"]

        mock_loader = Mock()
        mock_loader.to_crawled_local_objs.return_value = [Mock(spec=CrawledObject)]
        mock_loader.save = Mock()
        mock_loader_class.return_value = mock_loader

        # Execute
        result = load_docs(document_dir, doc_config, limit)

        # Assert
        assert isinstance(result, list)

    @patch("pickle.load")
    @patch("arklex.evaluation.get_documents.os.path.exists")
    @patch("arklex.evaluation.get_documents.Loader")
    def test_load_docs_text_type(
        self, mock_loader_class: Mock, mock_exists: Mock, mock_pickle_load: Mock
    ) -> None:
        """Test load_docs function with text type documents."""
        # Setup
        document_dir = "/test/dir"
        doc_config = {"rag_docs": [{"source": "Sample text content", "type": "text"}]}
        limit = 10

        mock_exists.return_value = False

        mock_loader = Mock()
        mock_loader.to_crawled_text.return_value = [Mock(spec=CrawledObject)]
        mock_loader.save = Mock()
        mock_loader_class.return_value = mock_loader

        # Execute
        result = load_docs(document_dir, doc_config, limit)

        # Assert
        assert isinstance(result, list)

    @patch("arklex.evaluation.get_documents.os.path.exists")
    def test_load_docs_invalid_type(self, mock_exists: Mock) -> None:
        """Test load_docs function with invalid document type."""
        # Setup
        document_dir = "/test/dir"
        doc_config = {"rag_docs": [{"source": "test", "type": "invalid"}]}
        limit = 10

        mock_exists.return_value = False

        # Execute
        result = load_docs(document_dir, doc_config, limit)

        # Assert
        assert result == []

    @patch("arklex.evaluation.get_documents.os.path.exists")
    def test_load_docs_missing_type(self, mock_exists: Mock) -> None:
        """Test load_docs function with missing document type."""
        # Setup
        document_dir = "/test/dir"
        doc_config = {"rag_docs": [{"source": "test"}]}  # Missing 'type' key
        limit = 10

        mock_exists.return_value = False

        # Execute
        result = load_docs(document_dir, doc_config, limit)

        # Assert
        assert result == []

    @patch("arklex.evaluation.get_documents.os.path.exists")
    def test_load_docs_missing_rag_and_task_docs(self, mock_exists: Mock) -> None:
        """Test load_docs function with missing rag_docs and task_docs keys."""
        # Setup
        document_dir = "/test/dir"
        doc_config = {}  # Missing both rag_docs and task_docs
        limit = 10

        mock_exists.return_value = False

        # Execute
        result = load_docs(document_dir, doc_config, limit)

        # Assert
        assert result == []

    def test_load_docs_none_directory(self) -> None:
        """Test load_docs function with None directory."""
        # Setup
        document_dir = None
        doc_config = {"rag_docs": [{"source": "test", "type": "url", "num": 5}]}
        limit = 10

        # Execute
        result = load_docs(document_dir, doc_config, limit)

        # Assert
        assert result == []

    @patch("arklex.evaluation.get_documents.os.path.exists")
    def test_load_docs_with_exception(self, mock_exists: Mock) -> None:
        """Test load_docs function when an exception occurs during loading."""
        # Setup
        document_dir = "/test/dir"
        doc_config = {"rag_docs": [{"source": "test", "type": "url", "num": 5}]}
        limit = 10

        mock_exists.side_effect = Exception("File system error")

        # Execute
        result = load_docs(document_dir, doc_config, limit)

        # Assert
        assert result == []

    @patch("pickle.load")
    @patch("arklex.evaluation.get_documents.os.path.exists")
    @patch("arklex.evaluation.get_documents.Loader")
    def test_load_docs_high_total_docs(
        self, mock_loader_class: Mock, mock_exists: Mock, mock_pickle_load: Mock
    ) -> None:
        """Test load_docs function with high total document count."""
        # Setup
        document_dir = "/test/dir"
        doc_config = {"rag_docs": [{"source": "test", "type": "url", "num": 100}]}
        limit = 10  # Lower limit than requested documents

        mock_exists.return_value = False

        mock_loader = Mock()
        mock_loader.get_all_urls.return_value = ["http://test.com"] * 100
        mock_loader.to_crawled_url_objs.return_value = [Mock(spec=CrawledObject)] * 100
        mock_loader.get_candidates_websites.return_value = [
            Mock(spec=CrawledObject)
        ] * 10
        mock_loader.save = Mock()
        mock_loader_class.return_value = mock_loader

        # Execute
        result = load_docs(document_dir, doc_config, limit)

        # Assert
        assert isinstance(result, list)
        assert len(result) <= limit

    @patch("pickle.load")
    @patch("arklex.evaluation.get_documents.os.path.exists")
    @patch("arklex.evaluation.get_documents.Loader")
    def test_load_docs_mixed_source_types(
        self, mock_loader_class: Mock, mock_exists: Mock, mock_pickle_load: Mock
    ) -> None:
        """Test load_docs function with mixed source types."""
        # Setup
        document_dir = "/test/dir"
        doc_config = {
            "rag_docs": [
                {"source": "http://test.com", "type": "url", "num": 2},
                {"source": "/test/files", "type": "file"},
                {"source": "Sample text", "type": "text"},
            ]
        }
        limit = 20

        mock_exists.return_value = False

        mock_loader = Mock()
        mock_loader.get_all_urls.return_value = ["http://test.com", "http://test2.com"]
        mock_loader.to_crawled_url_objs.return_value = [Mock(spec=CrawledObject)] * 2
        mock_loader.to_crawled_local_objs.return_value = [Mock(spec=CrawledObject)] * 3
        mock_loader.to_crawled_text.return_value = [Mock(spec=CrawledObject)] * 1
        mock_loader.get_candidates_websites.return_value = [
            Mock(spec=CrawledObject)
        ] * 6
        mock_loader.save = Mock()
        mock_loader_class.return_value = mock_loader

        # Execute
        result = load_docs(document_dir, doc_config, limit)

        # Assert
        assert isinstance(result, list)

    @patch("pickle.load")
    @patch("arklex.evaluation.get_documents.os.path.exists")
    @patch("arklex.evaluation.get_documents.Loader")
    def test_load_docs_invalid_crawled_objects(
        self, mock_loader_class: Mock, mock_exists: Mock, mock_pickle_load: Mock
    ) -> None:
        """Test load_docs function with invalid crawled objects."""
        # Setup
        document_dir = "/test/dir"
        doc_config = {"rag_docs": [{"source": "test", "type": "url", "num": 5}]}
        limit = 10

        mock_exists.return_value = False

        # Create invalid crawled object (missing required methods)
        invalid_obj = Mock()
        del invalid_obj.to_dict  # Remove required method

        mock_loader = Mock()
        mock_loader.get_all_urls.return_value = ["http://test.com"]
        mock_loader.to_crawled_url_objs.return_value = [invalid_obj]
        mock_loader.get_candidates_websites.return_value = [invalid_obj]
        mock_loader.save = Mock()
        mock_loader_class.return_value = mock_loader

        # Execute
        result = load_docs(document_dir, doc_config, limit)

        # Assert
        assert isinstance(result, list)
