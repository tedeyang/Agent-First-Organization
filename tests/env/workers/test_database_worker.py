"""
Tests for database worker functionality.

This module contains comprehensive tests for the DataBaseWorker class including
initialization, action verification, and workflow execution.
"""

from unittest.mock import Mock, patch

from arklex.env.prompts import BotConfig
from arklex.env.workers.database_worker import DataBaseWorker
from arklex.orchestrator.entities.orch_entities import MessageState


class TestDataBaseWorkerInitialization:
    """Test DataBaseWorker initialization."""

    def test_database_worker_init_with_default_config(self) -> None:
        """Test DataBaseWorker initialization with default configuration."""
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

                worker = DataBaseWorker()

                assert worker is not None
                assert hasattr(worker, "model_config")
                assert hasattr(worker, "llm")
                assert hasattr(worker, "action_graph")

    def test_database_worker_init_with_custom_config(self) -> None:
        """Test DataBaseWorker initialization with custom configuration."""
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

                assert worker is not None
                assert worker.model_config == custom_config
                assert hasattr(worker, "llm")
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

                custom_config = {
                    "llm_provider": "google",
                    "model_type_or_path": "gemini-pro",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                assert worker is not None
                assert worker.model_config == custom_config
                assert hasattr(worker, "llm")
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

                custom_config = {
                    "llm_provider": "anthropic",
                    "model_type_or_path": "claude-3-sonnet",
                    "api_key": "test-api-key",
                }

                worker = DataBaseWorker(model_config=custom_config)

                assert worker is not None
                assert worker.model_config == custom_config
                assert hasattr(worker, "llm")
                assert hasattr(worker, "action_graph")


class TestDataBaseWorkerActions:
    """Test DataBaseWorker actions."""

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
        """Test search show action."""
        mock_state = Mock(spec=MessageState)
        mock_state.slots = {}
        mock_state.bot_config = BotConfig(language="EN")

        result = self.worker.DBActions.search_show(mock_state)

        assert result is not None

    def test_book_show_action(self) -> None:
        """Test book show action."""
        mock_state = Mock(spec=MessageState)
        mock_state.slots = {}
        mock_state.bot_config = BotConfig(language="EN")

        result = self.worker.DBActions.book_show(mock_state)

        assert result is not None

    def test_check_booking_action(self) -> None:
        """Test check booking action."""
        mock_state = Mock(spec=MessageState)
        mock_state.slots = {}
        mock_state.bot_config = BotConfig(language="EN")

        result = self.worker.DBActions.check_booking(mock_state)

        assert result is not None

    def test_cancel_booking_action(self) -> None:
        """Test cancel booking action."""
        mock_state = Mock(spec=MessageState)
        mock_state.slots = {}
        mock_state.bot_config = BotConfig(language="EN")

        result = self.worker.DBActions.cancel_booking(mock_state)

        assert result is not None


class TestDataBaseWorkerVerifyAction:
    """Test DataBaseWorker verify_action method."""

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
        """Test verify_action for search show intent."""
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        # Set up the attribute as a dict with get method
        mock_orchestrator_message.attribute = {"task": "search for shows"}
        mock_state.orchestrator_message = mock_orchestrator_message
        mock_state.bot_config = BotConfig(language="EN")

        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = "SearchShow"
        self.worker.llm.invoke.return_value = mock_response

        result = self.worker.verify_action(mock_state)

        assert result == "SearchShow"

    def test_verify_action_book_show(self) -> None:
        """Test verify_action for book show intent."""
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        # Set up the attribute as a dict with get method
        mock_orchestrator_message.attribute = {"task": "book a show"}
        mock_state.orchestrator_message = mock_orchestrator_message
        mock_state.bot_config = BotConfig(language="EN")

        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = "BookShow"
        self.worker.llm.invoke.return_value = mock_response

        result = self.worker.verify_action(mock_state)

        assert result == "BookShow"

    def test_verify_action_check_booking(self) -> None:
        """Test verify_action for check booking intent."""
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        # Set up the attribute as a dict with get method
        mock_orchestrator_message.attribute = {"task": "check my bookings"}
        mock_state.orchestrator_message = mock_orchestrator_message
        mock_state.bot_config = BotConfig(language="EN")

        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = "CheckBooking"
        self.worker.llm.invoke.return_value = mock_response

        result = self.worker.verify_action(mock_state)

        assert result == "CheckBooking"

    def test_verify_action_cancel_booking(self) -> None:
        """Test verify_action for cancel booking intent."""
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        # Set up the attribute as a dict with get method
        mock_orchestrator_message.attribute = {"task": "cancel my booking"}
        mock_state.orchestrator_message = mock_orchestrator_message
        mock_state.bot_config = BotConfig(language="EN")

        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = "CancelBooking"
        self.worker.llm.invoke.return_value = mock_response

        result = self.worker.verify_action(mock_state)

        assert result == "CancelBooking"

    def test_verify_action_others(self) -> None:
        """Test verify_action for other intents."""
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        # Set up the attribute as a dict with get method
        mock_orchestrator_message.attribute = {"task": "general inquiry"}
        mock_state.orchestrator_message = mock_orchestrator_message
        mock_state.bot_config = BotConfig(language="EN")

        # Mock the LLM response
        self.worker.llm.invoke.return_value.content = "Others"

        result = self.worker.verify_action(mock_state)

        assert result == "Others"

    def test_verify_action_exception(self) -> None:
        """Test verify_action with exception."""
        mock_state = Mock(spec=MessageState)
        mock_orchestrator_message = Mock()
        # Set up the attribute as a dict with get method
        mock_orchestrator_message.attribute = {"task": "test task"}
        mock_state.orchestrator_message = mock_orchestrator_message
        mock_state.bot_config = BotConfig(language="EN")

        # Mock the LLM to raise an exception
        self.worker.llm.invoke.side_effect = Exception("LLM error")

        result = self.worker.verify_action(mock_state)

        assert result == "Others"


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

                # Verify action graph has the expected nodes
                graph = worker.action_graph
                assert graph is not None
                # The graph should have nodes for each action
                assert hasattr(graph, "nodes")


class TestDataBaseWorkerExecute:
    """Test DataBaseWorker execute method."""

    def test_execute_workflow(self) -> None:
        """Test execute workflow."""
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

                # Mock the message state
                mock_state = Mock(spec=MessageState)
                mock_state.slots = {}
                mock_state.bot_config = BotConfig(language="EN")

                # Mock the graph compilation and execution
                mock_graph = Mock()
                mock_result = Mock(spec=MessageState)
                mock_graph.invoke.return_value = mock_result
                worker.action_graph.compile = Mock(return_value=mock_graph)

                # Mock the login and init_slots methods
                worker.DBActions.log_in = Mock()
                worker.DBActions.init_slots = Mock(return_value={})

                result = worker._execute(mock_state)

                assert result == mock_result
                worker.DBActions.log_in.assert_called_once()
                worker.DBActions.init_slots.assert_called_once_with({}, {})
                mock_graph.invoke.assert_called_once_with(mock_state)


class TestDataBaseWorkerIntegration:
    """Integration tests for DataBaseWorker."""

    def test_database_worker_complete_workflow(self) -> None:
        """Test complete database worker workflow."""
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

                # Verify worker has all expected actions
                expected_actions = {
                    "SearchShow": "Search for shows",
                    "BookShow": "Book a show",
                    "CheckBooking": "Check details of booked show(s)",
                    "CancelBooking": "Cancel a booking",
                    "Others": "Other actions not mentioned above",
                }

                assert worker.actions == expected_actions

    def test_database_worker_description(self) -> None:
        """Test database worker description."""
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

                expected_description = "Help the user with actions related to customer support like a booking system with structured data, always involving search, insert, update, and delete operations."

                assert worker.description == expected_description

    def test_database_worker_actions_coverage(self) -> None:
        """Test that all database actions are covered."""
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

                # Check that all expected actions are present
                expected_action_names = [
                    "SearchShow",
                    "BookShow",
                    "CheckBooking",
                    "CancelBooking",
                    "Others",
                ]

                for action_name in expected_action_names:
                    assert action_name in worker.actions
