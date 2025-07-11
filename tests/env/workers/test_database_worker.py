"""
Tests for database worker functionality.

This module contains comprehensive tests for the DataBaseWorker class including
initialization, action verification, and workflow execution.
"""

from unittest.mock import Mock, patch

from arklex.env.workers.database_worker import DataBaseWorker
from arklex.orchestrator.entities.msg_state_entities import MessageState


class TestDataBaseWorkerInitialization:
    """Test DataBaseWorker initialization."""

    def test_database_worker_init_with_default_config(self) -> None:
        """Test DataBaseWorker initialization with default config."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                # Provide a proper model config
                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                assert worker.llm == mock_llm_instance
                assert hasattr(worker, "actions")
                assert hasattr(worker, "DBActions")
                assert hasattr(worker, "action_graph")

    def test_database_worker_init_with_custom_config(self) -> None:
        """Test DataBaseWorker initialization with custom config."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                assert worker.llm == mock_llm_instance
                assert hasattr(worker, "actions")
                assert hasattr(worker, "DBActions")
                assert hasattr(worker, "action_graph")

    def test_database_worker_init_google_provider(self) -> None:
        """Test DataBaseWorker initialization with Google provider."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                google_config = {
                    "llm_provider": "google",
                    "model_type_or_path": "gemini-pro",
                    "api_key": "test-google-api-key",
                }

                worker = DataBaseWorker(model_config=google_config)

                assert worker.llm == mock_llm_instance
                assert hasattr(worker, "actions")
                assert hasattr(worker, "DBActions")
                assert hasattr(worker, "action_graph")

    def test_database_worker_init_other_provider(self) -> None:
        """Test DataBaseWorker initialization with other provider."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                other_config = {
                    "llm_provider": "anthropic",
                    "model_type_or_path": "claude-3-sonnet",
                    "api_key": "test-anthropic-api-key",
                }

                worker = DataBaseWorker(model_config=other_config)

                assert worker.llm == mock_llm_instance
                assert hasattr(worker, "actions")
                assert hasattr(worker, "DBActions")
                assert hasattr(worker, "action_graph")


class TestDataBaseWorkerActions:
    """Test DataBaseWorker action methods."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                self.worker = DataBaseWorker(model_config=custom_config)

    def test_search_show_action(self) -> None:
        """Test search_show action."""
        mock_state = Mock(spec=MessageState)
        mock_result = Mock(spec=MessageState)

        self.worker.DBActions.search_show.return_value = mock_result

        result = self.worker.search_show(mock_state)

        assert result == mock_result
        self.worker.DBActions.search_show.assert_called_once_with(mock_state)

    def test_book_show_action(self) -> None:
        """Test book_show action."""
        mock_state = Mock(spec=MessageState)
        mock_result = Mock(spec=MessageState)

        self.worker.DBActions.book_show.return_value = mock_result

        result = self.worker.book_show(mock_state)

        assert result == mock_result
        self.worker.DBActions.book_show.assert_called_once_with(mock_state)

    def test_check_booking_action(self) -> None:
        """Test check_booking action."""
        mock_state = Mock(spec=MessageState)
        mock_result = Mock(spec=MessageState)

        self.worker.DBActions.check_booking.return_value = mock_result

        result = self.worker.check_booking(mock_state)

        assert result == mock_result
        self.worker.DBActions.check_booking.assert_called_once_with(mock_state)

    def test_cancel_booking_action(self) -> None:
        """Test cancel_booking action."""
        mock_state = Mock(spec=MessageState)
        mock_result = Mock(spec=MessageState)

        self.worker.DBActions.cancel_booking.return_value = mock_result

        result = self.worker.cancel_booking(mock_state)

        assert result == mock_result
        self.worker.DBActions.cancel_booking.assert_called_once_with(mock_state)


class TestDataBaseWorkerVerifyAction:
    """Test DataBaseWorker action verification."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                self.worker = DataBaseWorker(model_config=custom_config)

    def test_verify_action_search_show(self) -> None:
        """Test verify_action with SearchShow intent."""
        mock_state = Mock(spec=MessageState)
        mock_state.intent = "SearchShow"

        # Mock the LLM response
        self.worker.llm.invoke.return_value.content = "SearchShow"

        result = self.worker.verify_action(mock_state)

        assert result == "SearchShow"
        self.worker.llm.invoke.assert_called_once()

    def test_verify_action_book_show(self) -> None:
        """Test verify_action with BookShow intent."""
        mock_state = Mock(spec=MessageState)
        mock_state.intent = "BookShow"

        # Mock the LLM response
        self.worker.llm.invoke.return_value.content = "BookShow"

        result = self.worker.verify_action(mock_state)

        assert result == "BookShow"
        self.worker.llm.invoke.assert_called_once()

    def test_verify_action_check_booking(self) -> None:
        """Test verify_action with CheckBooking intent."""
        mock_state = Mock(spec=MessageState)
        mock_state.intent = "CheckBooking"

        # Mock the LLM response
        self.worker.llm.invoke.return_value.content = "CheckBooking"

        result = self.worker.verify_action(mock_state)

        assert result == "CheckBooking"
        self.worker.llm.invoke.assert_called_once()

    def test_verify_action_cancel_booking(self) -> None:
        """Test verify_action with CancelBooking intent."""
        mock_state = Mock(spec=MessageState)
        mock_state.intent = "CancelBooking"

        # Mock the LLM response
        self.worker.llm.invoke.return_value.content = "CancelBooking"

        result = self.worker.verify_action(mock_state)

        assert result == "CancelBooking"
        self.worker.llm.invoke.assert_called_once()

    def test_verify_action_others(self) -> None:
        """Test verify_action with other intents."""
        mock_state = Mock(spec=MessageState)
        mock_state.intent = "OtherIntent"

        # Mock the LLM response
        self.worker.llm.invoke.return_value.content = "Others"

        result = self.worker.verify_action(mock_state)

        assert result == "Others"
        self.worker.llm.invoke.assert_called_once()

    def test_verify_action_exception(self) -> None:
        """Test verify_action with exception handling."""
        mock_state = Mock(spec=MessageState)
        mock_state.intent = "SearchShow"

        # Mock the LLM to raise an exception
        self.worker.llm.invoke.side_effect = Exception("LLM error")

        result = self.worker.verify_action(mock_state)

        assert result == "Others"
        self.worker.llm.invoke.assert_called_once()


class TestDataBaseWorkerCreateActionGraph:
    """Test DataBaseWorker action graph creation."""

    def test_create_action_graph_structure(self) -> None:
        """Test that action graph is created with correct structure."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                # Verify action graph structure
                assert hasattr(worker, "action_graph")
                assert worker.action_graph is not None


class TestDataBaseWorkerExecute:
    """Test DataBaseWorker execution."""

    def test_execute_workflow(self) -> None:
        """Test complete workflow execution."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                # Mock the verify_action method
                worker.verify_action = Mock(return_value="SearchShow")

                # Mock the search_show method
                mock_result = Mock(spec=MessageState)
                worker.search_show = Mock(return_value=mock_result)

                mock_state = Mock(spec=MessageState)
                mock_state.intent = "SearchShow"

                result = worker.execute(mock_state)

                assert result == mock_result
                worker.verify_action.assert_called_once_with(mock_state)
                worker.search_show.assert_called_once_with(mock_state)


class TestDataBaseWorkerIntegration:
    """Integration tests for DataBaseWorker."""

    def test_database_worker_complete_workflow(self) -> None:
        """Test a complete database worker workflow."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                # Test that worker has all expected attributes
                assert hasattr(worker, "llm")
                assert hasattr(worker, "actions")
                assert hasattr(worker, "DBActions")
                assert hasattr(worker, "action_graph")
                assert hasattr(worker, "description")

                # Test that worker has expected methods
                assert hasattr(worker, "search_show")
                assert hasattr(worker, "book_show")
                assert hasattr(worker, "check_booking")
                assert hasattr(worker, "cancel_booking")
                assert hasattr(worker, "verify_action")
                assert hasattr(worker, "execute")

                # Test that actions are properly defined
                expected_actions = [
                    "search_show",
                    "book_show",
                    "check_booking",
                    "cancel_booking",
                ]
                for action in expected_actions:
                    assert action in worker.actions

    def test_database_worker_description(self) -> None:
        """Test that database worker has appropriate description."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                # Test that description is appropriate
                assert isinstance(worker.description, str)
                assert len(worker.description) > 0
                assert "booking system" in worker.description.lower()

    def test_database_worker_actions_coverage(self) -> None:
        """Test that database worker covers all expected actions."""
        with patch(
            "arklex.utils.model_provider_config.PROVIDER_MAP"
        ) as mock_provider_map:
            mock_llm_class = Mock()
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            mock_provider_map.get.return_value = mock_llm_class

            # Mock the database actions to avoid ChatOpenAI initialization
            with patch(
                "arklex.env.workers.database_worker.DatabaseActions"
            ) as mock_db_actions:
                mock_db_actions_instance = Mock()
                mock_db_actions.return_value = mock_db_actions_instance

                custom_config = {
                    "llm_provider": "openai",
                    "model_type_or_path": "gpt-3.5-turbo",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                # Test that all expected actions are available
                expected_actions = [
                    "search_show",
                    "book_show",
                    "check_booking",
                    "cancel_booking",
                ]
                for action in expected_actions:
                    assert action in worker.actions
