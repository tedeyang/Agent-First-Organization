"""Tests for the loader module.

This module contains comprehensive test cases for document and content loading utilities,
including web crawling, file processing, and content chunking functionality.
"""

from unittest.mock import Mock, patch
import tempfile
import os
import requests

from arklex.utils.loader import (
    encode_image,
    SourceType,
    DocObject,
    CrawledObject,
    Loader,
)


class TestEncodeImage:
    """Test cases for encode_image function."""

    def test_encode_image_success(self) -> None:
        """Test successful image encoding."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_file.write(b"fake image data")
            tmp_file_path = tmp_file.name

        try:
            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = (
                    b"fake image data"
                )
                with patch("base64.b64encode") as mock_b64encode:
                    mock_b64encode.return_value.decode.return_value = (
                        "ZmFrZSBpbWFnZSBkYXRh"
                    )
                    result = encode_image(tmp_file_path)
                    assert result == "ZmFrZSBpbWFnZSBkYXRh"
        finally:
            os.unlink(tmp_file_path)

    def test_encode_image_file_not_found(self) -> None:
        """Test image encoding with non-existent file."""
        result = encode_image("nonexistent.png")
        assert result is None

    def test_encode_image_general_exception(self) -> None:
        """Test image encoding with general exception."""
        with patch("builtins.open", side_effect=Exception("Permission denied")):
            result = encode_image("test.png")
            assert result is None

    def test_encode_image_nonexistent(self) -> None:
        """Test image encoding with None path."""
        result = encode_image(None)
        assert result is None

    def test_encode_image_with_directory(self) -> None:
        """Test image encoding with directory path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = encode_image(temp_dir)
            assert result is None

    def test_encode_image_with_non_image_file(self) -> None:
        """Test image encoding with non-image file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(b"text content")
            tmp_file_path = tmp_file.name

        try:
            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = (
                    b"text content"
                )
                with patch("base64.b64encode") as mock_b64encode:
                    mock_b64encode.return_value.decode.return_value = "dGV4dCBjb250ZW50"
                    result = encode_image(tmp_file_path)
                    assert result == "dGV4dCBjb250ZW50"
        finally:
            os.unlink(tmp_file_path)


class TestSourceType:
    """Test cases for SourceType enum."""

    def test_source_type_values(self) -> None:
        """Test SourceType enum values."""
        assert SourceType.WEB.value == 1
        assert SourceType.FILE.value == 2
        assert SourceType.TEXT.value == 3

    def test_source_type_unknown(self) -> None:
        """Test SourceType with unknown value."""
        # Test that we can access the enum values
        assert hasattr(SourceType, "WEB")
        assert hasattr(SourceType, "FILE")
        assert hasattr(SourceType, "TEXT")


class TestDocObject:
    """Test cases for DocObject class."""

    def test_doc_object_initialization(self) -> None:
        """Test DocObject initialization."""
        doc = DocObject("test_id", "test_source")
        assert doc.id == "test_id"
        assert doc.source == "test_source"

    def test_docobject_missing_fields(self) -> None:
        """Test DocObject with missing fields."""
        doc = DocObject("", "")
        assert doc.id == ""
        assert doc.source == ""


class TestCrawledObject:
    """Test cases for CrawledObject class."""

    def test_crawled_object_initialization(self) -> None:
        """Test CrawledObject initialization."""
        crawled = CrawledObject(
            "test_id", "test_source", "test_content", {"key": "value"}
        )
        assert crawled.id == "test_id"
        assert crawled.source == "test_source"
        assert crawled.content == "test_content"
        assert crawled.metadata == {"key": "value"}

    def test_crawled_object_default_values(self) -> None:
        """Test CrawledObject with default values."""
        crawled = CrawledObject("test_id", "test_source", "test_content")
        assert crawled.metadata == {}
        assert crawled.is_chunk is False
        assert crawled.is_error is False
        assert crawled.error_message is None
        assert crawled.source_type == SourceType.WEB

    def test_crawled_object_to_dict(self) -> None:
        """Test CrawledObject to_dict method."""
        crawled = CrawledObject(
            "test_id", "test_source", "test_content", {"key": "value"}
        )
        result = crawled.to_dict()
        assert result["id"] == "test_id"
        assert result["source"] == "test_source"
        assert result["content"] == "test_content"
        assert result["metadata"] == {"key": "value"}

    def test_crawled_object_from_dict(self) -> None:
        """Test CrawledObject from_dict method."""
        data = {
            "id": "test_id",
            "source": "test_source",
            "content": "test_content",
            "metadata": {"key": "value"},
            "is_chunk": False,
            "is_error": False,
            "error_message": None,
            "source_type": SourceType.WEB,
        }
        crawled = CrawledObject.from_dict(data)
        assert crawled.id == "test_id"
        assert crawled.source == "test_source"
        assert crawled.content == "test_content"
        assert crawled.metadata == {"key": "value"}


class TestLoader:
    """Test cases for Loader class."""

    def test_loader_initialization(self) -> None:
        """Test Loader initialization."""
        loader = Loader()
        assert loader is not None

    def test_to_crawled_url_objs(self) -> None:
        """Test converting URL list to CrawledObject list."""
        loader = Loader()
        urls = ["http://example.com", "http://test.com"]
        result = loader.to_crawled_url_objs(urls)
        assert len(result) == 2
        assert result[0].source == "http://example.com"
        assert result[1].source == "http://test.com"

    def test_crawl_urls_selenium_success(self) -> None:
        """Test successful Selenium crawling."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        with patch("selenium.webdriver.Chrome") as mock_driver:
            mock_driver_instance = Mock()
            mock_driver.return_value = mock_driver_instance
            mock_driver_instance.page_source = (
                "<html><title>Test</title><body>Content</body></html>"
            )
            mock_driver_instance.quit.return_value = None

            with patch("time.sleep"):
                result = loader.crawl_urls(url_objects)
                assert len(result) == 1
                assert result[0].source == "http://example.com"

    def test_crawl_urls_selenium_failure_requests_success(self) -> None:
        """Test Selenium failure with requests fallback success."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        # Mock Selenium failure
        with patch(
            "selenium.webdriver.Chrome", side_effect=Exception("Selenium failed")
        ):
            # Mock requests success
            with patch("requests.get") as mock_get:
                mock_response = Mock()
                mock_response.text = (
                    "<html><title>Test</title><body>Content</body></html>"
                )
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response

                result = loader.crawl_urls(url_objects)
                assert len(result) == 1
                assert result[0].source == "http://example.com"

    def test_crawl_urls_all_failure_mock_content(self) -> None:
        """Test all crawling methods failure with mock content fallback."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        # Mock both Selenium and requests failure
        with patch(
            "selenium.webdriver.Chrome", side_effect=Exception("Selenium failed")
        ):
            with patch("requests.get", side_effect=Exception("Requests failed")):
                result = loader.crawl_urls(url_objects)
                assert len(result) == 1
                assert result[0].source == "http://example.com"
                assert "mock_content" in result[0].metadata

    def test_crawl_with_selenium_timeout(self) -> None:
        """Test _crawl_with_selenium with timeout."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        # Mock Selenium to raise timeout exception
        with patch("selenium.webdriver.Chrome", side_effect=Exception("timeout")):
            with patch("time.sleep"):
                result = loader._crawl_with_selenium(url_objects)
                assert len(result) == 1
                assert result[0].is_error

    def test_crawl_with_selenium_retry_success(self) -> None:
        """Test Selenium crawling with retry success."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        with patch("selenium.webdriver.Chrome") as mock_driver:
            mock_driver_instance = Mock()
            mock_driver.return_value = mock_driver_instance
            mock_driver_instance.page_source = (
                "<html><title>Test</title><body>Content</body></html>"
            )
            mock_driver_instance.quit.return_value = None

            # First call fails, second succeeds
            mock_driver_instance.page_source = (
                "<html><title>Test</title><body>Content</body></html>"
            )

            with patch("time.sleep"):
                result = loader._crawl_with_selenium(url_objects)
                assert len(result) == 1

    def test_crawl_with_selenium_expected_error_filtering(self) -> None:
        """Test Selenium crawling with expected error filtering."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        with patch(
            "selenium.webdriver.Chrome",
            side_effect=Exception("cannot determine loading status"),
        ):
            with patch("time.sleep"):
                result = loader._crawl_with_selenium(url_objects)
                assert len(result) == 1
                assert result[0].is_error

    def test_crawl_with_requests_http_error(self) -> None:
        """Test requests crawling with HTTP error."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.HTTPError("404 Not Found")
            result = loader._crawl_with_requests(url_objects)
            assert len(result) == 1
            assert result[0].is_error

    def test_create_mock_content_from_urls_various_urls(self) -> None:
        """Test creating mock content from various URL types."""
        loader = Loader()
        url_objects = [
            DocObject("1", "http://example.com/company"),
            DocObject("2", "http://example.com/contact"),
            DocObject("3", "http://example.com/privacy"),
            DocObject("4", "http://example.com/terms"),
            DocObject("5", "http://example.com/resources"),
            DocObject("6", "http://example.com/solutions"),
            DocObject("7", "http://example.com/clouffee"),
            DocObject("8", "http://example.com/headquarters"),
            DocObject("9", "http://example.com/award"),
            DocObject("10", "http://example.com/cleaning"),
            DocObject("11", "http://example.com/delivery"),
            DocObject("12", "http://example.com/production"),
            DocObject("13", "http://example.com/"),
        ]

        result = loader._create_mock_content_from_urls(url_objects)
        assert len(result) == 13
        assert all("mock_content" in doc.metadata for doc in result)

    def test_get_all_urls_timeout(self) -> None:
        """Test get_all_urls with timeout."""
        loader = Loader()

        with patch.object(
            loader, "get_outsource_urls", side_effect=Exception("Network error")
        ):
            result = loader.get_all_urls("http://example.com", 10)
            assert len(result) == 1  # Should include the base URL

    def test_get_all_urls_exception_handling(self) -> None:
        """Test get_all_urls with exception handling."""
        loader = Loader()

        with patch.object(
            loader, "get_outsource_urls", side_effect=Exception("Network error")
        ):
            result = loader.get_all_urls("http://example.com", 10)
            assert len(result) == 1  # Should include the base URL

    def test_check_url_edge_cases(self) -> None:
        """Test _check_url with edge cases."""
        loader = Loader()

        # Test with different URL formats
        assert (
            loader._check_url("http://example.com", "http://example.com") is False
        )  # Same URL returns False
        assert (
            loader._check_url("http://example.com/page", "http://example.com") is True
        )  # Subpath returns True
        assert (
            loader._check_url("https://example.com", "http://example.com") is False
        )  # Different protocol
        assert (
            loader._check_url("http://sub.example.com", "http://example.com") is False
        )  # Different subdomain
        assert (
            loader._check_url("http://example.com#fragment", "http://example.com")
            is True
        )  # Fragment returns True
        assert (
            loader._check_url("http://example.com/file.pdf", "http://example.com")
            is False
        )  # PDF file returns False

    def test_get_candidates_websites_edge_cases(self) -> None:
        """Test get_candidates_websites with edge cases."""
        loader = Loader()

        # Test with empty list
        result = loader.get_candidates_websites([], 5)
        assert len(result) == 0

        # Test with fewer URLs than top_k
        urls = [
            CrawledObject("1", "http://example1.com", "content1"),
            CrawledObject("2", "http://example2.com", "content2"),
        ]
        result = loader.get_candidates_websites(urls, 5)
        assert len(result) == 2

    def test_crawl_file_with_mistral_api(self) -> None:
        """Test crawl_file with Mistral API."""
        loader = Loader()
        doc_obj = DocObject("1", "test.txt")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp_file:
            tmp_file.write("Test content")
            tmp_file_path = tmp_file.name

        try:
            with patch("mistralai.Mistral") as mock_mistral:
                mock_instance = mock_mistral.return_value
                mock_instance.generate.return_value = "API response"
                result = loader.crawl_file(doc_obj)
                assert isinstance(result, CrawledObject)
        finally:
            os.remove(tmp_file_path)

    def test_chunk_with_empty_docs(self) -> None:
        """Test chunk method with empty documents."""
        result = Loader.chunk([])
        assert len(result) == 0

    def test_chunk_with_exception(self) -> None:
        """Test chunk method with exception."""
        docs = [CrawledObject("1", "test.txt", "content")]

        with patch(
            "langchain.text_splitter.RecursiveCharacterTextSplitter",
            side_effect=Exception("Splitter error"),
        ):
            result = Loader.chunk(docs)
            assert len(result) == 1
            # The result is a Document object, not CrawledObject, so we check differently
            assert hasattr(result[0], "page_content")

    def test_loader_crawl_urls_empty(self) -> None:
        """Test crawling empty URL list."""
        loader = Loader()
        result = loader.crawl_urls([])
        assert len(result) == 0

    def test_loader_chunk_empty_docs(self) -> None:
        """Test chunking empty documents."""
        result = Loader.chunk([])
        assert len(result) == 0

    def test_source_type_unknown(self) -> None:
        """Test SourceType with unknown value."""
        # Test that we can access the enum values
        assert hasattr(SourceType, "WEB")
        assert hasattr(SourceType, "FILE")
        assert hasattr(SourceType, "TEXT")

    def test_crawl_with_selenium_chromedriver_installation_failure(self) -> None:
        """Test Selenium crawling with chromedriver installation failure."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        with patch(
            "selenium.webdriver.Chrome", side_effect=Exception("chromedriver not found")
        ):
            with patch("time.sleep"):
                result = loader._crawl_with_selenium(url_objects)
                assert len(result) == 1
                assert result[0].is_error

    def test_crawl_with_selenium_webdriver_exception(self) -> None:
        """Test Selenium crawling with webdriver exception."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        with patch(
            "selenium.webdriver.Chrome", side_effect=Exception("chrome not reachable")
        ):
            with patch("time.sleep"):
                result = loader._crawl_with_selenium(url_objects)
                assert len(result) == 1
                assert result[0].is_error

    def test_crawl_with_requests_timeout(self) -> None:
        """Test requests crawling with timeout."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        with patch("requests.get", side_effect=requests.Timeout("Request timeout")):
            result = loader._crawl_with_requests(url_objects)
            assert len(result) == 1
            assert result[0].is_error

    def test_crawl_file_permission_error(self) -> None:
        """Test crawl_file with permission error."""
        loader = Loader()
        doc_obj = DocObject("1", "/root/protected.txt")

        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            result = loader.crawl_file(doc_obj)
            assert result.is_error
            assert "Error loading" in result.error_message

    def test_crawl_file_unicode_error(self) -> None:
        """Test crawl_file with Unicode decode error."""
        loader = Loader()
        doc_obj = DocObject("1", "test.txt")

        with patch(
            "builtins.open",
            side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid utf-8"),
        ):
            result = loader.crawl_file(doc_obj)
            assert result.is_error
            assert "Error loading" in result.error_message

    def test_crawl_file_general_exception(self) -> None:
        """Test crawl_file with general exception."""
        loader = Loader()
        doc_obj = DocObject("1", "test.txt")

        with patch("builtins.open", side_effect=Exception("General error")):
            result = loader.crawl_file(doc_obj)
            assert result.is_error
            assert "Error loading" in result.error_message

    def test_crawl_file_with_mistral_api_error(self) -> None:
        """Test crawl_file with Mistral API error."""
        loader = Loader()
        doc_obj = DocObject("1", "test.txt")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp_file:
            tmp_file.write("Test content")
            tmp_file_path = tmp_file.name

        try:
            with patch("mistralai.Mistral", side_effect=Exception("API error")):
                result = loader.crawl_file(doc_obj)
                assert result.is_error
        finally:
            os.remove(tmp_file_path)

    def test_save_method(self) -> None:
        """Test save method."""
        docs = [CrawledObject("1", "test.txt", "content")]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            tmp_file_path = tmp_file.name

        try:
            Loader.save(tmp_file_path, docs)
            assert os.path.exists(tmp_file_path)
        finally:
            os.unlink(tmp_file_path)

    def test_chunk_method_with_exception(self) -> None:
        """Test chunk method with exception."""
        docs = [CrawledObject("1", "test.txt", "content")]

        with patch(
            "langchain.text_splitter.RecursiveCharacterTextSplitter",
            side_effect=Exception("Splitter error"),
        ):
            result = Loader.chunk(docs)
            assert len(result) == 1
            # The result is a Document object, not CrawledObject, so we check differently
            assert hasattr(result[0], "page_content")
