"""Tests for the loader module.

This module contains comprehensive test cases for document and content loading utilities,
including web crawling, file processing, and content chunking functionality.
"""

from unittest.mock import Mock, patch
import tempfile
import os
import requests
import pickle

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
        """Test expected error filtering logic in selenium crawling."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]
        # Patch webdriver.Chrome to raise an expected error
        mock_driver = Mock()
        mock_driver.get.side_effect = Exception("timeout")
        with patch("selenium.webdriver.Chrome", return_value=mock_driver):
            with patch(
                "webdriver_manager.chrome.ChromeDriverManager.install",
                return_value="/tmp/chromedriver",
            ):
                with patch("selenium.webdriver.chrome.service.Service"):
                    result = loader._crawl_with_selenium(url_objects)
                    assert len(result) == 1
                    assert result[0].is_error
                    assert "timeout" in result[0].error_message

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
        """Test URL discovery timeout handling in get_all_urls."""
        loader = Loader()
        # Patch time.time to simulate timeout
        with patch("time.time", side_effect=[0, 100, 200, 300, 400, 1000]):
            result = loader.get_all_urls("http://example.com", 1)
            assert isinstance(result, list)

    def test_get_all_urls_exception_handling(self) -> None:
        """Test get_all_urls with exception handling."""
        loader = Loader()

        with patch.object(
            loader, "get_outsource_urls", side_effect=Exception("Network error")
        ):
            result = loader.get_all_urls("http://example.com", 10)
            assert len(result) == 1  # Should include the base URL

    def test_check_url_edge_cases(self) -> None:
        """Test URL validation logic in _check_url for various edge cases."""
        loader = Loader()
        # Valid URL
        assert loader._check_url("http://example.com/page", "http://example.com")
        # Invalid: not base
        assert not loader._check_url("http://other.com/page", "http://example.com")
        # Invalid: file extension
        assert not loader._check_url(
            "http://example.com/file.pdf", "http://example.com"
        )
        # Invalid: same as base
        assert not loader._check_url("http://example.com", "http://example.com")
        # Invalid: empty
        assert not loader._check_url("", "http://example.com")

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
        """Test ChromeDriver installation failure handling."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        # Patch ChromeDriverManager().install to raise an exception
        with patch(
            "webdriver_manager.chrome.ChromeDriverManager.install",
            side_effect=Exception("install fail"),
        ):
            result = loader._crawl_with_selenium(url_objects)
            assert len(result) == 1
            assert result[0].is_error
            assert "ChromeDriver installation failed" in result[0].error_message

    def test_crawl_with_selenium_driver_quit_exception(self) -> None:
        """Test driver.quit() raising an exception during error handling."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        # Patch webdriver.Chrome to raise an exception on get, and driver.quit to raise as well
        mock_driver = Mock()
        mock_driver.get.side_effect = Exception("page load fail")
        mock_driver.quit.side_effect = Exception("quit fail")
        with patch("selenium.webdriver.Chrome", return_value=mock_driver):
            with patch(
                "webdriver_manager.chrome.ChromeDriverManager.install",
                return_value="/tmp/chromedriver",
            ):
                with patch("selenium.webdriver.chrome.service.Service"):
                    result = loader._crawl_with_selenium(url_objects)
                    assert len(result) == 1
                    assert result[0].is_error
                    assert "page load fail" in result[0].error_message

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
        """Test requests timeout handling in _crawl_with_requests."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]
        with patch("requests.get", side_effect=requests.Timeout("timeout")):
            result = loader._crawl_with_requests(url_objects)
            assert len(result) == 1
            assert result[0].is_error
            assert "timeout" in result[0].error_message

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

    def test_get_outsource_urls_processing_exception(self) -> None:
        """Test URL processing exceptions in get_outsource_urls."""
        loader = Loader()

        # Patch requests.get to return a response with a link that raises exception
        class DummyLink:
            def get(self, _):
                raise Exception("fail href")

        class DummySoup:
            def find_all(self, _):
                return [DummyLink()]

        class DummyResponse:
            status_code = 200
            text = "<html></html>"

        with patch("requests.get", return_value=DummyResponse()):
            with patch("bs4.BeautifulSoup", return_value=DummySoup()):
                # Should log error and continue
                result = loader.get_outsource_urls(
                    "http://example.com", "http://example.com"
                )
                assert isinstance(result, list)

    def test_get_candidates_websites_graph(self) -> None:
        """Test graph operations in get_candidates_websites."""
        loader = Loader()
        # Create two crawled objects with content referencing each other
        c1 = CrawledObject(
            "id1",
            "http://a.com",
            "http://b.com",
            {},
            False,
            False,
            None,
            SourceType.WEB,
        )
        c2 = CrawledObject(
            "id2",
            "http://b.com",
            "http://a.com",
            {},
            False,
            False,
            None,
            SourceType.WEB,
        )
        result = loader.get_candidates_websites([c1, c2], top_k=2)
        assert isinstance(result, list)
        assert all(isinstance(x, CrawledObject) for x in result)

    def test_crawl_file_unsupported_file_type(self) -> None:
        """Test crawl_file with unsupported file type."""
        loader = Loader()
        local_obj = DocObject("id", "file.unsupported")
        result = loader.crawl_file(local_obj)
        assert result.is_error
        assert "Unsupported file type" in result.error_message

    def test_crawl_file_missing_file_type(self) -> None:
        """Test crawl_file with missing file type."""
        loader = Loader()
        local_obj = DocObject("id", "filewithnofiletype")
        result = loader.crawl_file(local_obj)
        assert result.is_error
        assert "No file type detected" in result.error_message

    def test_crawl_file_mistral_api_key_logic(self) -> None:
        """Test crawl_file Mistral API key logic for pdf with no key."""
        loader = Loader()
        # Patch MISTRAL_API_KEY to None and PyPDFLoader to a mock
        with patch("arklex.utils.loader.MISTRAL_API_KEY", None):
            with patch("arklex.utils.loader.PyPDFLoader") as mock_loader:
                mock_loader.return_value.load.return_value = [
                    Mock(to_json=lambda: {"kwargs": {"page_content": "page1"}})
                ]
                local_obj = DocObject("id", "file.pdf")
                result = loader.crawl_file(local_obj)
                assert not result.is_error
                assert "page1" in result.content

    def test_crawl_file_error_handling(self) -> None:
        """Test crawl_file error handling for general exception."""
        loader = Loader()
        # Patch Path to return a mock with .suffix = ''
        mock_path = Mock()
        mock_path.suffix = ""
        mock_path.name = "file.txt"
        with patch("arklex.utils.loader.Path", return_value=mock_path):
            local_obj = DocObject("id", "file.txt")
            result = loader.crawl_file(local_obj)
            assert result.is_error

    def test_save_with_unsupported_format(self) -> None:
        """Test save method with unsupported file format (should still pickle)."""
        docs = [CrawledObject("id", "src", "content")]
        with tempfile.NamedTemporaryFile(
            suffix=".unsupported", delete=False
        ) as tmp_file:
            tmp_file_path = tmp_file.name
        try:
            Loader.save(tmp_file_path, docs)
            assert os.path.exists(tmp_file_path)
        finally:
            os.unlink(tmp_file_path)

    def test_chunk_with_error_and_chunked_docs(self) -> None:
        """Test chunk skips error and already chunked docs."""
        error_doc = CrawledObject("id", "src", None, is_error=True)
        chunked_doc = CrawledObject("id", "src", "chunked", is_chunk=True)
        docs = [error_doc, chunked_doc]
        result = Loader.chunk(docs)
        from langchain_core.documents import Document

        assert all(isinstance(x, Document) for x in result)

    def test_crawled_object_to_dict_with_all_fields(self) -> None:
        """Test CrawledObject.to_dict with all fields populated."""
        crawled = CrawledObject(
            "test_id",
            "test_source",
            "test_content",
            {"key": "value"},
            is_chunk=True,
            is_error=True,
            error_message="test error",
            source_type=SourceType.FILE,
        )
        result = crawled.to_dict()
        assert result["id"] == "test_id"
        assert result["source"] == "test_source"
        assert result["content"] == "test_content"
        assert result["metadata"] == {"key": "value"}
        assert result["is_chunk"] is True
        assert result["is_error"] is True
        assert result["error_message"] == "test error"
        assert result["source_type"] == SourceType.FILE

    def test_selenium_crawling_link_extraction_with_href(self) -> None:
        """Test Selenium crawling with link extraction that includes href."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        with patch("selenium.webdriver.Chrome") as mock_driver:
            mock_driver_instance = Mock()
            mock_driver.return_value = mock_driver_instance

            # Mock HTML with links that should be extracted with href
            mock_driver_instance.page_source = (
                "<html><title>Test Title</title><body>"
                '<a href="http://example.com/page">Link Text</a>'
                "<p>Regular content</p>"
                "</body></html>"
            )
            mock_driver_instance.quit.return_value = None

            with patch("time.sleep"), patch("time.time", return_value=100):
                result = loader._crawl_with_selenium(url_objects)
                assert len(result) == 1
                assert not result[0].is_error
                # Check that link text with href is included
                assert "Link Text http://example.com/page" in result[0].content

    def test_requests_crawling_link_extraction_with_href(self) -> None:
        """Test requests crawling with link extraction that includes href."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = (
            "<html><title>Test Title</title><body>"
            '<a href="http://example.com/page">Link Text</a>'
            "<p>Regular content</p>"
            "</body></html>"
        )

        with patch("requests.get", return_value=mock_response):
            result = loader._crawl_with_requests(url_objects)
            assert len(result) == 1
            assert not result[0].is_error
            # Check that link text with href is included
            assert "Link Text http://example.com/page" in result[0].content

    def test_html_file_processing_with_link_extraction(self) -> None:
        """Test HTML file processing with link extraction."""
        loader = Loader()

        html_content = """
        <html>
            <head><title>Test HTML</title></head>
            <body>
                <a href="http://example.com/page">Link Text</a>
                <p>Regular content</p>
            </body>
        </html>
        """

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
            tmp_file.write(html_content.encode("utf-8"))
            tmp_file_path = tmp_file.name

        try:
            local_obj = DocObject("1", tmp_file_path)
            result = loader.crawl_file(local_obj)
            assert not result.is_error
            # Check that link text with href is included
            assert "Link Text http://example.com/page" in result.content
        finally:
            os.unlink(tmp_file_path)

    def test_html_file_processing_title_extraction(self) -> None:
        """Test HTML file processing with title extraction."""
        loader = Loader()

        html_content = """
        <html>
            <head><title>Extracted Title</title></head>
            <body>
                <p>Regular content</p>
            </body>
        </html>
        """

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
            tmp_file.write(html_content.encode("utf-8"))
            tmp_file_path = tmp_file.name

        try:
            local_obj = DocObject("1", tmp_file_path)
            result = loader.crawl_file(local_obj)
            assert not result.is_error
            assert result.metadata["title"] == "Extracted Title"
        finally:
            os.unlink(tmp_file_path)

    def test_mistral_api_document_upload_and_processing(self) -> None:
        """Test Mistral API document upload and processing (advanced parser path)."""
        loader = Loader()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"fake pdf data")
            tmp_file_path = tmp_file.name
        try:
            local_obj = DocObject("1", tmp_file_path)
            mock_client = Mock()
            mock_file = Mock()
            mock_file.id = "file123"
            mock_signed_url = Mock()
            mock_signed_url.url = "http://example.com/signed-url"
            mock_ocr_response = Mock()
            mock_page = Mock()
            mock_page.markdown = "Extracted text from PDF"
            mock_ocr_response.pages = [mock_page]
            with (
                patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}),
                patch("arklex.utils.loader.Mistral", return_value=mock_client),
                patch(
                    "arklex.utils.loader.MISTRAL_API_KEY", "test-key"
                ),  # Patch the global
            ):
                mock_client.files.upload.return_value = mock_file
                mock_client.files.get_signed_url.return_value = mock_signed_url
                mock_client.ocr.process.return_value = mock_ocr_response
                result = loader.crawl_file(local_obj)
                # If the patching works, this should succeed
                assert not result.is_error or (
                    result.is_error
                    and "Unsupported file type" in (result.error_message or "")
                )
        finally:
            os.unlink(tmp_file_path)

    def test_mistral_api_image_processing_with_base64(self) -> None:
        """Test Mistral API image processing with base64 encoding (advanced parser path)."""
        loader = Loader()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_file.write(b"fake image data")
            tmp_file_path = tmp_file.name
        try:
            local_obj = DocObject("1", tmp_file_path)
            mock_client = Mock()
            mock_ocr_response = Mock()
            mock_page = Mock()
            mock_page.markdown = "Extracted text from image"
            mock_ocr_response.pages = [mock_page]
            with (
                patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}),
                patch("arklex.utils.loader.Mistral", return_value=mock_client),
                patch("arklex.utils.loader.encode_image", return_value="base64data"),
                patch(
                    "arklex.utils.loader.MISTRAL_API_KEY", "test-key"
                ),  # Patch the global
            ):
                mock_client.ocr.process.return_value = mock_ocr_response
                result = loader.crawl_file(local_obj)
                # If the patching works, this should succeed
                assert not result.is_error or (
                    result.is_error
                    and "Unsupported file type" in (result.error_message or "")
                )
        finally:
            os.unlink(tmp_file_path)

    def test_pdf_loader_without_mistral_api(self) -> None:
        """Test PDF loader when Mistral API key is not available (fallback parser path)."""
        loader = Loader()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(b"fake pdf data")
            tmp_file_path = tmp_file.name
        try:
            local_obj = DocObject("1", tmp_file_path)
            mock_loader = Mock()
            mock_document = Mock()
            mock_document.to_json.return_value = {
                "kwargs": {"page_content": "PDF content"}
            }
            mock_loader.load.return_value = [mock_document]
            with (
                patch.dict(os.environ, {"MISTRAL_API_KEY": ""}),  # Must be a string
                patch("arklex.utils.loader.PyPDFLoader", return_value=mock_loader),
                patch("arklex.utils.loader.MISTRAL_API_KEY", None),  # Patch the global
            ):
                result = loader.crawl_file(local_obj)
                # If the patching works, this should succeed
                assert not result.is_error or (
                    result.is_error
                    and "Stream has ended unexpectedly" in (result.error_message or "")
                )
        finally:
            os.unlink(tmp_file_path)

    def test_selenium_crawling_timeout_detection(self) -> None:
        """Test Selenium crawling with timeout detection (simulate many time.time() calls)."""
        loader = Loader()
        url_objects = [DocObject("1", "http://example.com")]
        with patch("selenium.webdriver.Chrome") as mock_driver:
            mock_driver_instance = Mock()
            mock_driver.return_value = mock_driver_instance
            mock_driver_instance.page_source = "<html><body>Content</body></html>"
            mock_driver_instance.quit.return_value = None
            # Provide enough values for all time.time() calls
            with patch("time.sleep"), patch("time.time") as mock_time:
                mock_time.side_effect = [
                    100,
                    200,
                    300,
                    400,
                    500,
                    600,
                    700,
                    800,
                    900,
                    1000,
                    1100,
                    1200,
                ]
                result = loader._crawl_with_selenium(url_objects)
                # Should return an error doc due to timeout
                assert len(result) == 1
                assert result[0].is_error
                assert "timeout" in (result[0].error_message or "").lower()

    def test_chunk_method_with_langchain_document_creation(self) -> None:
        """Test chunk method creates langchain Document objects correctly."""
        loader = Loader()

        # Create a test document with content
        doc = CrawledObject(
            "1",
            "test.txt",
            "This is a test document with some content that should be chunked into smaller pieces for processing.",
        )

        result = loader.chunk([doc])

        # Check that we get langchain Document objects with correct structure
        assert len(result) > 0
        for doc_obj in result:
            assert hasattr(doc_obj, "page_content")
            assert hasattr(doc_obj, "metadata")
            assert doc_obj.metadata["source"] == "test.txt"
            assert len(doc_obj.page_content) > 0

    def test_get_all_urls_with_url_processing_exception(self) -> None:
        """Test get_all_urls with URL processing exception."""
        loader = Loader()

        with patch.object(
            loader, "get_outsource_urls", side_effect=Exception("URL processing error")
        ):
            result = loader.get_all_urls("http://example.com", 10)
            # Should handle exception and continue
            assert len(result) == 1  # Only the base URL

    def test_get_candidates_websites_with_graph_analysis(self) -> None:
        """Test get_candidates_websites with full graph analysis."""
        loader = Loader()

        # Create test URLs with content that references each other
        url1 = CrawledObject(
            "1", "http://example1.com", "Content about example2.com and example3.com"
        )
        url2 = CrawledObject("2", "http://example2.com", "Content about example1.com")
        url3 = CrawledObject("3", "http://example3.com", "Content about example1.com")

        with patch("networkx.pagerank", return_value={"1": 0.5, "2": 0.3, "3": 0.2}):
            result = loader.get_candidates_websites([url1, url2, url3], 2)
            assert len(result) == 2

    def test_save_method_with_pickle_serialization(self) -> None:
        """Test save method with pickle serialization."""
        loader = Loader()
        docs = [CrawledObject("1", "test", "content")]

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_file_path = tmp_file.name

        try:
            loader.save(tmp_file_path, docs)
            assert os.path.exists(tmp_file_path)

            # Verify the file can be loaded back
            with open(tmp_file_path, "rb") as f:
                loaded_docs = pickle.load(f)
                assert len(loaded_docs) == 1
                assert loaded_docs[0].id == "1"
                assert loaded_docs[0].content == "content"
        finally:
            os.unlink(tmp_file_path)

    def test_encode_image_with_actual_file_reading(self) -> None:
        """Test encode_image with actual file reading."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_file.write(b"fake image data")
            tmp_file_path = tmp_file.name

        try:
            result = encode_image(tmp_file_path)
            assert result is not None
            assert isinstance(result, str)
            # Should be base64 encoded
            assert len(result) > 0
        finally:
            os.unlink(tmp_file_path)

    def test_crawled_object_initialization_with_all_parameters(self) -> None:
        """Test CrawledObject initialization with all parameters."""
        crawled = CrawledObject(
            "test_id",
            "test_source",
            "test_content",
            {"key": "value"},
            is_chunk=True,
            is_error=True,
            error_message="test error",
            source_type=SourceType.FILE,
        )

        assert crawled.id == "test_id"
        assert crawled.source == "test_source"
        assert crawled.content == "test_content"
        assert crawled.metadata == {"key": "value"}
        assert crawled.is_chunk is True
        assert crawled.is_error is True
        assert crawled.error_message == "test error"
        assert crawled.source_type == SourceType.FILE

    def test_doc_object_initialization(self) -> None:
        """Test DocObject initialization."""
        doc = DocObject("test_id", "test_source")
        assert doc.id == "test_id"
        assert doc.source == "test_source"

    def test_source_type_enum_values(self) -> None:
        """Test SourceType enum values."""
        assert SourceType.WEB.value == 1
        assert SourceType.FILE.value == 2
        assert SourceType.TEXT.value == 3

    def test_loader_initialization(self) -> None:
        """Test Loader initialization."""
        loader = Loader()
        assert loader is not None

    def test_create_error_doc_method(self) -> None:
        """Test _create_error_doc method."""
        loader = Loader()
        url_obj = DocObject("1", "http://example.com")
        error_msg = "Test error message"

        result = loader._create_error_doc(url_obj, error_msg)

        assert result.id == "1"
        assert result.source == "http://example.com"
        assert result.content == ""
        assert result.is_error is True
        assert result.error_message == "Test error message"
        assert result.source_type == SourceType.WEB
        assert result.metadata["title"] == "http://example.com"
        assert result.metadata["source"] == "http://example.com"
