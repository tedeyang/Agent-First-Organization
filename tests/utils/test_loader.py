"""Tests for the loader module.

This module contains comprehensive test cases for document and content loading utilities,
including web crawling, file processing, and content chunking functionality.
"""

from unittest.mock import Mock, patch, mock_open
import pytest
import tempfile
import os

from arklex.utils.loader import (
    encode_image,
    SourceType,
    DocObject,
    CrawledObject,
    Loader,
)


class TestEncodeImage:
    """Test cases for encode_image function."""

    @patch("arklex.utils.loader.log_context")
    def test_encode_image_success(self, mock_log_context: Mock) -> None:
        """Test encode_image with valid image file."""
        # Setup
        test_content = b"fake image content"

        with patch("builtins.open", mock_open(read_data=test_content)):
            # Execute
            result = encode_image("test_image.jpg")

        # Assert
        import base64

        expected = base64.b64encode(test_content).decode("utf-8")
        assert result == expected

    @patch("arklex.utils.loader.log_context")
    def test_encode_image_file_not_found(self, mock_log_context: Mock) -> None:
        """Test encode_image with non-existent file."""
        # Setup
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            # Execute
            result = encode_image("nonexistent.jpg")

        # Assert
        assert result is None
        mock_log_context.error.assert_called_once()

    @patch("arklex.utils.loader.log_context")
    def test_encode_image_general_exception(self, mock_log_context: Mock) -> None:
        """Test encode_image with general exception."""
        # Setup
        with patch("builtins.open", side_effect=Exception("General error")):
            # Execute
            result = encode_image("test_image.jpg")

        # Assert
        assert result is None
        mock_log_context.error.assert_called_once()

    def test_encode_image_nonexistent(self) -> None:
        """Test encode_image with non-existent file returns None."""
        result = encode_image("nonexistent.png")
        assert result is None

    def test_encode_image_with_directory(self) -> None:
        """Test encode_image with directory path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = encode_image(temp_dir)
            assert result is None

    def test_encode_image_with_non_image_file(self) -> None:
        """Test encode_image with non-image file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"This is not an image")
            temp_file.flush()
            try:
                result = encode_image(temp_file.name)
                # The function should return the base64 encoded content, not None
                assert result is not None
                assert isinstance(result, str)
            finally:
                os.unlink(temp_file.name)


class TestSourceType:
    """Test cases for SourceType enum."""

    def test_source_type_values(self) -> None:
        """Test SourceType enum values."""
        # Assert
        assert SourceType.WEB.value == 1
        assert SourceType.FILE.value == 2
        assert SourceType.TEXT.value == 3

    def test_source_type_unknown(self) -> None:
        """Test SourceType with unknown value."""
        with pytest.raises(ValueError):
            SourceType(999)


class TestDocObject:
    """Test cases for DocObject class."""

    def test_doc_object_initialization(self) -> None:
        """Test DocObject initialization."""
        # Setup
        doc_id = "test_id"
        source = "test_source"

        # Execute
        doc_obj = DocObject(doc_id, source)

        # Assert
        assert doc_obj.id == doc_id
        assert doc_obj.source == source

    def test_docobject_missing_fields(self) -> None:
        """Test DocObject with missing fields raises TypeError."""
        with pytest.raises(TypeError):
            DocObject()  # Missing required fields


class TestCrawledObject:
    """Test cases for CrawledObject class."""

    def test_crawled_object_initialization(self) -> None:
        """Test CrawledObject initialization with all parameters."""
        # Setup
        doc_id = "test_id"
        source = "test_source"
        content = "test_content"
        metadata = {"key": "value"}

        # Execute
        crawled_obj = CrawledObject(
            id=doc_id,
            source=source,
            content=content,
            metadata=metadata,
            is_chunk=True,
            is_error=False,
            error_message=None,
            source_type=SourceType.WEB,
        )

        # Assert
        assert crawled_obj.id == doc_id
        assert crawled_obj.source == source
        assert crawled_obj.content == content
        assert crawled_obj.metadata == metadata
        assert crawled_obj.is_chunk is True
        assert crawled_obj.is_error is False
        assert crawled_obj.error_message is None
        assert crawled_obj.source_type == SourceType.WEB

    def test_crawled_object_default_values(self) -> None:
        """Test CrawledObject initialization with default values."""
        # Execute
        crawled_obj = CrawledObject("test_id", "test_source", "test_content")

        # Assert
        assert crawled_obj.metadata == {}
        assert crawled_obj.is_chunk is False
        assert crawled_obj.is_error is False
        assert crawled_obj.error_message is None
        assert crawled_obj.source_type == SourceType.WEB

    def test_crawled_object_to_dict(self) -> None:
        """Test CrawledObject to_dict method."""
        # Setup
        crawled_obj = CrawledObject(
            id="test_id",
            source="test_source",
            content="test_content",
            metadata={"key": "value"},
            is_chunk=True,
            is_error=False,
            error_message=None,
            source_type=SourceType.FILE,
        )

        # Execute
        result = crawled_obj.to_dict()

        # Assert
        expected = {
            "id": "test_id",
            "source": "test_source",
            "content": "test_content",
            "metadata": {"key": "value"},
            "is_chunk": True,
            "is_error": False,
            "error_message": None,
            "source_type": SourceType.FILE,
        }
        assert result == expected

    def test_crawled_object_from_dict(self) -> None:
        """Test CrawledObject from_dict class method."""
        # Setup
        data = {
            "id": "test_id",
            "source": "test_source",
            "content": "test_content",
            "metadata": {"key": "value"},
            "is_chunk": True,
            "is_error": False,
            "error_message": "test error",
            "source_type": SourceType.TEXT,
        }

        # Execute
        result = CrawledObject.from_dict(data)

        # Assert
        assert result.id == "test_id"
        assert result.source == "test_source"
        assert result.content == "test_content"
        assert result.metadata == {"key": "value"}
        assert result.is_chunk is True
        assert result.is_error is False
        assert result.error_message == "test error"
        assert result.source_type == SourceType.TEXT


class TestLoader:
    """Test cases for Loader class."""

    def test_loader_initialization(self) -> None:
        """Test Loader initialization."""
        # Execute
        loader = Loader()

        # Assert
        assert loader is not None

    @patch("arklex.utils.loader.uuid.uuid4")
    def test_to_crawled_url_objs(self, mock_uuid: Mock) -> None:
        """Test to_crawled_url_objs method."""
        # Setup
        mock_uuid.return_value = "test-uuid"
        loader = Loader()
        url_list = ["http://example.com", "http://test.com"]

        with patch.object(loader, "crawl_urls") as mock_crawl:
            mock_crawl.return_value = [
                CrawledObject("test-uuid", "http://example.com", "content1"),
                CrawledObject("test-uuid", "http://test.com", "content2"),
            ]

            # Execute
            result = loader.to_crawled_url_objs(url_list)

        # Assert
        assert len(result) == 2
        mock_crawl.assert_called_once()

    @patch("arklex.utils.loader.log_context")
    def test_crawl_urls_selenium_success(self, mock_log_context: Mock) -> None:
        """Test crawl_urls with successful Selenium crawling."""
        # Setup
        loader = Loader()
        url_objects = [DocObject("id1", "http://example.com")]

        with patch.object(loader, "_crawl_with_selenium") as mock_selenium:
            mock_selenium.return_value = [
                CrawledObject("id1", "http://example.com", "content", is_error=False)
            ]

            # Execute
            result = loader.crawl_urls(url_objects)

        # Assert
        assert len(result) == 1
        assert not result[0].is_error
        mock_selenium.assert_called_once_with(url_objects)

    @patch("arklex.utils.loader.log_context")
    def test_crawl_urls_selenium_failure_requests_success(
        self, mock_log_context: Mock
    ) -> None:
        """Test crawl_urls with Selenium failure but requests success."""
        # Setup
        loader = Loader()
        url_objects = [DocObject("id1", "http://example.com")]

        with patch.object(loader, "_crawl_with_selenium") as mock_selenium:
            with patch.object(loader, "_crawl_with_requests") as mock_requests:
                mock_selenium.return_value = [
                    CrawledObject("id1", "http://example.com", "", is_error=True)
                ]
                mock_requests.return_value = [
                    CrawledObject(
                        "id1", "http://example.com", "content", is_error=False
                    )
                ]

                # Execute
                result = loader.crawl_urls(url_objects)

        # Assert
        assert len(result) == 1
        assert not result[0].is_error
        mock_selenium.assert_called_once()
        mock_requests.assert_called_once()

    @patch("arklex.utils.loader.log_context")
    def test_crawl_urls_all_failure_mock_content(self, mock_log_context: Mock) -> None:
        """Test crawl_urls with all crawling methods failing."""
        # Setup
        loader = Loader()
        url_objects = [DocObject("id1", "http://example.com")]

        with patch.object(loader, "_crawl_with_selenium") as mock_selenium:
            with patch.object(loader, "_crawl_with_requests") as mock_requests:
                with patch.object(
                    loader, "_create_mock_content_from_urls"
                ) as mock_mock:
                    mock_selenium.return_value = [
                        CrawledObject("id1", "http://example.com", "", is_error=True)
                    ]
                    mock_requests.return_value = [
                        CrawledObject("id1", "http://example.com", "", is_error=True)
                    ]
                    mock_mock.return_value = [
                        CrawledObject(
                            "id1", "http://example.com", "mock content", is_error=False
                        )
                    ]

                    # Execute
                    result = loader.crawl_urls(url_objects)

        # Assert
        assert len(result) == 1
        assert not result[0].is_error
        mock_mock.assert_called_once()

    @patch("arklex.utils.loader.webdriver.ChromeOptions")
    @patch("arklex.utils.loader.webdriver.Chrome")
    @patch("arklex.utils.loader.Service")
    @patch("arklex.utils.loader.log_context")
    @patch("arklex.utils.loader.time.time")
    def test_crawl_with_selenium_timeout(
        self,
        mock_time: Mock,
        mock_log_context: Mock,
        mock_service: Mock,
        mock_chrome: Mock,
        mock_options: Mock,
    ) -> None:
        """Test _crawl_with_selenium with timeout."""
        # Setup
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver

        # Use a function that returns time values
        time_values = [
            0,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
            35,
        ]
        mock_time.side_effect = lambda: time_values.pop(0) if time_values else 35

        # Execute
        result = loader._crawl_with_selenium(url_objects)

        # Assert
        assert len(result) == 1
        assert result[0].is_error is True

    @patch("arklex.utils.loader.webdriver.ChromeOptions")
    @patch("arklex.utils.loader.webdriver.Chrome")
    @patch("arklex.utils.loader.log_context")
    def test_crawl_with_selenium_retry_success(
        self, mock_log_context: Mock, mock_chrome: Mock, mock_options: Mock
    ) -> None:
        """Test _crawl_with_selenium with retry success."""
        # Setup
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]
        mock_driver = Mock()
        mock_chrome.side_effect = [Exception("First attempt"), mock_driver]
        mock_driver.page_source = "<html><title>Test</title><body>Content</body></html>"

        # Execute
        result = loader._crawl_with_selenium(url_objects)

        # Assert
        assert len(result) == 1
        assert result[0].content is not None

    @patch("arklex.utils.loader.webdriver.ChromeOptions")
    @patch("arklex.utils.loader.webdriver.Chrome")
    @patch("arklex.utils.loader.log_context")
    def test_crawl_with_selenium_expected_error_filtering(
        self, mock_log_context: Mock, mock_chrome: Mock, mock_options: Mock
    ) -> None:
        """Test _crawl_with_selenium with expected error filtering."""
        # Setup
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]
        mock_chrome.side_effect = Exception("cannot determine loading status")

        # Execute
        result = loader._crawl_with_selenium(url_objects)

        # Assert
        assert len(result) == 1
        assert result[0].is_error is True
        mock_log_context.debug.assert_called()

    @patch("arklex.utils.loader.requests.get")
    @patch("arklex.utils.loader.log_context")
    def test_crawl_with_requests_http_error(
        self, mock_log_context: Mock, mock_get: Mock
    ) -> None:
        """Test _crawl_with_requests with HTTP error."""
        # Setup
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_get.return_value = mock_response

        # Execute
        result = loader._crawl_with_requests(url_objects)

        # Assert
        assert len(result) == 1
        assert result[0].is_error is True

    def test_create_mock_content_from_urls_various_urls(self) -> None:
        """Test _create_mock_content_from_urls with various URL types."""
        # Setup
        loader = Loader()
        url_objects = [
            DocObject("1", "http://example.com/contact"),
            DocObject("2", "http://example.com/privacy"),
            DocObject("3", "http://example.com/terms"),
            DocObject("4", "http://example.com/resources"),
            DocObject("5", "http://example.com/solutions"),
            DocObject("6", "http://example.com/clouffee"),
            DocObject("7", "http://example.com/headquarters"),
            DocObject("8", "http://example.com/award"),
            DocObject("9", "http://example.com/cleaning"),
            DocObject("10", "http://example.com/delivery"),
            DocObject("11", "http://example.com/production"),
        ]

        # Execute
        result = loader._create_mock_content_from_urls(url_objects)

        # Assert
        assert len(result) == 11
        assert any("contact" in doc.content.lower() for doc in result)
        assert any("privacy" in doc.content.lower() for doc in result)
        assert any("terms" in doc.content.lower() for doc in result)
        assert any("resources" in doc.content.lower() for doc in result)
        assert any("solutions" in doc.content.lower() for doc in result)
        assert any("cloutea" in doc.content.lower() for doc in result)
        assert any("headquarters" in doc.content.lower() for doc in result)
        assert any("award" in doc.content.lower() for doc in result)
        assert any("cleaning" in doc.content.lower() for doc in result)
        assert any("delivery" in doc.content.lower() for doc in result)
        assert any("production" in doc.content.lower() for doc in result)

    @patch("arklex.utils.loader.requests.get")
    @patch("arklex.utils.loader.BeautifulSoup")
    @patch("arklex.utils.loader.log_context")
    @patch("arklex.utils.loader.time.time")
    def test_get_all_urls_timeout(
        self, mock_time: Mock, mock_log_context: Mock, mock_bs4: Mock, mock_get: Mock
    ) -> None:
        """Test get_all_urls with timeout."""
        # Setup
        loader = Loader()
        # Use a function that returns time values
        time_values = [
            0,
            70,
            70,
            70,
            70,
            70,
            70,
            70,
            70,
            70,
            70,
            70,
            70,
            70,
            70,
            70,
            70,
            70,
            70,
            70,
        ]
        mock_time.side_effect = lambda: time_values.pop(0) if time_values else 70
        mock_response = Mock()
        mock_response.text = '<html><a href="http://example.com/page1">Link</a></html>'
        mock_get.return_value = mock_response

        # Execute
        result = loader.get_all_urls("http://example.com", 5)

        # Assert
        # If timeout occurs before any URLs are visited, result may be empty
        assert len(result) in (0, 1)
        mock_log_context.warning.assert_called()

    @patch("arklex.utils.loader.requests.get")
    @patch("arklex.utils.loader.BeautifulSoup")
    @patch("arklex.utils.loader.log_context")
    def test_get_all_urls_exception_handling(
        self, mock_log_context: Mock, mock_bs4: Mock, mock_get: Mock
    ) -> None:
        """Test get_all_urls with exception handling."""
        # Setup
        loader = Loader()
        mock_get.side_effect = Exception("Network error")

        # Execute
        result = loader.get_all_urls("http://example.com", 5)

        # Assert
        assert len(result) == 1  # Only base URL
        mock_log_context.error.assert_called()

    def test_check_url_edge_cases(self) -> None:
        """Test _check_url with edge cases."""
        # Setup
        loader = Loader()

        # Test with invalid URLs
        assert loader._check_url("invalid-url", "http://example.com") is False
        assert loader._check_url("", "http://example.com") is False
        # The actual implementation returns True for valid URLs even with empty base_url
        assert loader._check_url("http://example.com", "") is True

    def test_get_candidates_websites_edge_cases(self) -> None:
        """Test get_candidates_websites with edge cases."""
        # Setup
        loader = Loader()

        # Test with empty list
        result = loader.get_candidates_websites([], 5)
        assert len(result) == 0

        # Test with top_k larger than available
        urls = [CrawledObject("1", "http://example.com", "content")]
        result = loader.get_candidates_websites(urls, 10)
        assert len(result) == 1

    @patch("arklex.utils.loader.MISTRAL_API_KEY", "test_key")
    @patch("arklex.utils.loader.TextLoader")
    @patch("arklex.utils.loader.log_context")
    @patch("builtins.open", new_callable=mock_open, read_data="test content")
    def test_crawl_file_with_mistral_api(
        self, mock_file: Mock, mock_log_context: Mock, mock_text_loader: Mock
    ) -> None:
        """Test crawl_file with MISTRAL_API_KEY set."""
        # Setup
        loader = Loader()
        local_obj = DocObject("1", "test.txt")
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [
            Mock(to_json=lambda: {"kwargs": {"page_content": "test content"}})
        ]
        mock_text_loader.return_value = mock_loader_instance

        # Execute
        result = loader.crawl_file(local_obj)

        # Assert
        assert result.content is not None

    @patch("arklex.utils.loader.RecursiveCharacterTextSplitter")
    def test_chunk_with_empty_docs(self, mock_splitter: Mock) -> None:
        """Test chunk method with empty document list."""
        # Setup
        doc_objs = []
        mock_splitter_instance = Mock()
        mock_splitter_instance.split_text.return_value = []
        mock_splitter.return_value = mock_splitter_instance

        # Execute
        result = Loader.chunk(doc_objs)

        # Assert
        assert len(result) == 0

    @patch("arklex.utils.loader.RecursiveCharacterTextSplitter")
    def test_chunk_with_exception(self, mock_splitter: Mock) -> None:
        """Test chunk method with exception."""
        # Setup
        doc_objs = [CrawledObject("1", "test", "content")]
        mock_splitter.side_effect = Exception("Splitter error")

        # Execute
        result = Loader.chunk(doc_objs)

        # Assert
        # The actual implementation returns empty list on exception
        assert len(result) == 0

    def test_loader_load_nonexistent_file(self) -> None:
        loader = Loader()
        with pytest.raises(Exception):
            loader.load("nonexistent_file.txt")

    def test_loader_load_unsupported_extension(self) -> None:
        loader = Loader()
        with pytest.raises(Exception):
            loader.load("file.unsupported")

    def test_loader_crawl_urls_empty(self) -> None:
        loader = Loader()
        result = loader.crawl_urls([])
        assert result == []

    def test_loader_chunk_empty_docs(self) -> None:
        loader = Loader()
        result = loader.chunk([])
        assert result == []

    def test_source_type_unknown(self) -> None:
        with pytest.raises(ValueError):
            SourceType("unknown")

    @patch("arklex.utils.loader.webdriver.Chrome")
    @patch("arklex.utils.loader.Service")
    def test_crawl_with_selenium_chromedriver_installation_failure(
        self, mock_service, mock_driver
    ) -> None:
        """Test selenium crawling with ChromeDriver installation failure."""
        mock_service.side_effect = Exception("ChromeDriver not found")

        loader = Loader()
        url_objects = [DocObject(id="1", source="http://example.com")]

        result = loader._crawl_with_selenium(url_objects)

        assert len(result) == 1
        assert result[0].is_error is True
        assert "ChromeDriver installation failed" in result[0].error_message

    @patch("arklex.utils.loader.webdriver.Chrome")
    @patch("arklex.utils.loader.Service")
    def test_crawl_with_selenium_webdriver_exception(
        self, mock_service, mock_driver
    ) -> None:
        """Test selenium crawling with WebDriver exception."""
        mock_service.return_value = Mock()
        mock_driver_instance = Mock()
        mock_driver_instance.get.side_effect = Exception("Driver error")
        mock_driver.return_value = mock_driver_instance

        loader = Loader()
        url_objects = [DocObject(id="1", source="http://example.com")]

        result = loader._crawl_with_selenium(url_objects)

        assert len(result) == 1
        assert result[0].is_error is True
        assert "Driver error" in result[0].error_message

    @patch("arklex.utils.loader.requests.get")
    def test_crawl_with_requests_timeout(self, mock_get) -> None:
        """Test requests crawling with timeout."""
        mock_get.side_effect = Exception("Request timeout")

        loader = Loader()
        url_objects = [DocObject(id="1", source="http://example.com")]

        result = loader._crawl_with_requests(url_objects)

        assert len(result) == 1
        assert result[0].is_error is True
        assert "Request timeout" in result[0].error_message

    @patch("builtins.open", create=True)
    def test_crawl_file_permission_error(self, mock_open) -> None:
        """Test crawl_file with permission error."""
        mock_open.side_effect = PermissionError("Permission denied")

        loader = Loader()
        local_obj = DocObject(id="1", source="/protected/file.txt")

        result = loader.crawl_file(local_obj)

        assert result.id == "1"
        assert result.source == "/protected/file.txt"
        assert result.is_error is True
        assert result.error_message == "Error loading /protected/file.txt"

    @patch("builtins.open", create=True)
    def test_crawl_file_unicode_error(self, mock_open) -> None:
        """Test crawl_file with unicode error."""
        mock_file = Mock()
        mock_file.read.side_effect = UnicodeDecodeError(
            "utf-8", b"", 0, 1, "invalid utf-8"
        )
        mock_open.return_value.__enter__.return_value = mock_file

        loader = Loader()
        local_obj = DocObject(id="1", source="/path/to/file.txt")

        result = loader.crawl_file(local_obj)

        assert result.id == "1"
        assert result.source == "/path/to/file.txt"
        assert result.is_error is True
        assert result.error_message == "Error loading /path/to/file.txt"

    @patch("builtins.open", create=True)
    def test_crawl_file_general_exception(self, mock_open) -> None:
        """Test crawl_file with general exception."""
        mock_open.side_effect = Exception("General error")

        loader = Loader()
        local_obj = DocObject(id="1", source="/path/to/file.txt")

        result = loader.crawl_file(local_obj)

        assert result.id == "1"
        assert result.source == "/path/to/file.txt"
        assert result.is_error is True
        assert result.error_message == "Error loading /path/to/file.txt"

    @patch("builtins.open", create=True)
    def test_crawl_file_with_mistral_api_error(self, mock_open) -> None:
        """Test crawl_file with Mistral API error."""
        mock_file = Mock()
        mock_file.read.return_value = "Test file content"
        mock_open.return_value.__enter__.return_value = mock_file

        # Mock the Mistral API call with error
        with patch("arklex.utils.loader.requests.post") as mock_post:
            mock_post.side_effect = Exception("API error")

            loader = Loader()
            local_obj = DocObject(id="1", source="/path/to/file.txt")

            result = loader.crawl_file(local_obj)

            assert result.id == "1"
            assert result.source == "/path/to/file.txt"
            assert result.is_error is False

    def test_save_method(self) -> None:
        """Test save static method."""
        docs = [
            CrawledObject(id="1", source="http://example1.com", content="content1"),
            CrawledObject(id="2", source="http://example2.com", content="content2"),
        ]

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pkl", delete=False
        ) as temp_file:
            temp_path = temp_file.name

        try:
            Loader.save(temp_path, docs)

            # Verify file was created and contains data
            assert os.path.exists(temp_path)
            with open(temp_path, "rb") as f:
                content = f.read()
                assert len(content) > 0  # Should contain pickle data

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_chunk_method_with_exception(self) -> None:
        """Test chunk method with exception during processing."""
        docs = [Mock()]  # Mock object that will cause issues

        result = Loader.chunk(docs)

        assert result == []
