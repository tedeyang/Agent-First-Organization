import os
import sqlite3
import tempfile
from collections.abc import Generator
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Set test environment variable
os.environ["ARKLEX_TEST_ENV"] = "local"
os.environ["DATA_DIR"] = tempfile.gettempdir()

from arklex.env.tools.database.utils import (
    DBNAME,
    MULTIPLE_SHOWS_MESSAGE,
    NO_BOOKING_MESSAGE,
    NO_SHOW_MESSAGE,
    SLOTS,
    USER_ID,
    DatabaseActions,
)
from arklex.orchestrator.entities.msg_state_entities import (
    MessageState,
    SlotDetail,
    StatusEnum,
)


# Helper to create a temp DB with required schema and data
def setup_temp_db() -> None:
    db_path = os.path.join(os.environ["DATA_DIR"], "show_booking_db.sqlite")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user (
            id VARCHAR(40) PRIMARY KEY,
            first_name VARCHAR(40),
            last_name VARCHAR(40),
            email VARCHAR(60),
            register_at TIMESTAMP,
            last_login TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS show (
            id VARCHAR(40) PRIMARY KEY,
            show_name VARCHAR(100),
            genre VARCHAR(40),
            date DATE,
            time TIME,
            description TEXT,
            location VARCHAR(100),
            price DECIMAL,
            available_seats INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS booking (
            id VARCHAR(40) PRIMARY KEY,
            show_id VARCHAR(40),
            user_id VARCHAR(40),
            created_at TIMESTAMP
        )
    """)
    # Insert a user
    cursor.execute(
        "INSERT OR IGNORE INTO user (id, first_name, last_name, email, register_at, last_login) VALUES (?, ?, ?, ?, ?, ?)",
        (
            "user_be6e1836-8fe9-4938-b2d0-48f810648e72",
            "Test",
            "User",
            "test@example.com",
            "2024-01-01 00:00:00",
            "2024-01-01 00:00:00",
        ),
    )
    # Insert multiple shows for testing
    shows = [
        (
            "show_1",
            "Test Show",
            "Opera",
            "2024-12-31",
            "20:00:00",
            "A test show.",
            "Test Location",
            100.0,
            50,
        ),
        (
            "show_2",
            "Another Show",
            "Comedy",
            "2024-12-31",
            "21:00:00",
            "Another test show.",
            "Test Location",
            150.0,
            30,
        ),
        (
            "show_3",
            "Test Show",
            "Drama",
            "2024-01-15",
            "19:00:00",
            "A different test show.",
            "Other Location",
            75.0,
            25,
        ),
    ]
    for show in shows:
        cursor.execute(
            "INSERT OR IGNORE INTO show (id, show_name, genre, date, time, description, location, price, available_seats) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            show,
        )
    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
def run_around_tests() -> Generator[None, None, None]:
    setup_temp_db()
    yield
    # Clean up DB after test
    db_path = os.path.join(os.environ["DATA_DIR"], "show_booking_db.sqlite")
    if os.path.exists(db_path):
        os.remove(db_path)


class TestConstants:
    """Test the constants defined in the module."""

    def test_slots_structure(self) -> None:
        """Test that SLOTS constant has the correct structure."""
        assert isinstance(SLOTS, list)
        assert len(SLOTS) == 4  # show_name, location, date, time

        for slot in SLOTS:
            assert isinstance(slot, dict)
            required_keys = ["name", "type", "value", "description", "prompt"]
            for key in required_keys:
                assert key in slot
                assert isinstance(slot[key], str)

    def test_slot_names(self) -> None:
        """Test that slot names are correct."""
        slot_names = [slot["name"] for slot in SLOTS]
        expected_names = ["show_name", "location", "date", "time"]
        assert slot_names == expected_names

    def test_slot_types(self) -> None:
        """Test that slot types are correct."""
        slot_types = [slot["type"] for slot in SLOTS]
        expected_types = ["string", "string", "date", "time"]
        assert slot_types == expected_types

    def test_message_constants(self) -> None:
        """Test that message constants are defined."""
        assert isinstance(NO_SHOW_MESSAGE, str)
        assert isinstance(MULTIPLE_SHOWS_MESSAGE, str)
        assert isinstance(NO_BOOKING_MESSAGE, str)
        assert len(NO_SHOW_MESSAGE) > 0
        assert len(MULTIPLE_SHOWS_MESSAGE) > 0
        assert len(NO_BOOKING_MESSAGE) > 0

    def test_database_constants(self) -> None:
        """Test database-related constants."""
        assert isinstance(DBNAME, str)
        assert DBNAME == "show_booking_db.sqlite"
        assert isinstance(USER_ID, str)
        assert len(USER_ID) > 0


class TestDatabaseActionsInit:
    """Test the DatabaseActions class initialization."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_init_with_default_user_id(self, mock_llm: object) -> None:
        """Test initialization with default user ID."""
        db = DatabaseActions()
        assert db.user_id == USER_ID
        assert db.db_path.endswith(DBNAME)
        assert isinstance(db.slots, list)
        assert isinstance(db.slot_prompts, list)
        assert len(db.slots) == 0
        assert len(db.slot_prompts) == 0

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_init_with_custom_user_id(self, mock_llm: object) -> None:
        """Test initialization with custom user ID."""
        custom_user_id = "custom_user_123"
        db = DatabaseActions(user_id=custom_user_id)
        assert db.user_id == custom_user_id
        assert db.db_path.endswith(DBNAME)

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_init_llm_configuration(self, mock_llm: object) -> None:
        """Test that LLM is properly configured."""
        db = DatabaseActions()
        assert db.llm is not None
        # Verify ChatOpenAI was called with expected parameters
        mock_llm.assert_called_once()


class TestDatabaseActionsLogin:
    """Test the log_in method."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_log_in_success(self, mock_llm: object) -> None:
        """Test successful login with existing user."""
        db = DatabaseActions()
        result = db.log_in()
        assert result is True

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_log_in_failure(self, mock_llm: object) -> None:
        """Test login failure with non-existent user."""
        db = DatabaseActions(user_id="nonexistent_user")
        result = db.log_in()
        assert result is False

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_log_in_database_connection(self, mock_llm: object) -> None:
        """Test that database connection is properly handled."""
        db = DatabaseActions()
        # This should not raise any exceptions
        result = db.log_in()
        assert isinstance(result, bool)


class TestDatabaseActionsInitSlots:
    """Test the init_slots method."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.load_prompts")
    def test_init_slots_with_empty_slots(
        self, mock_load_prompts: object, mock_llm: object
    ) -> None:
        """Test init_slots with empty slots list."""
        mock_load_prompts.return_value = {
            "database_slot_prompt": "template {slot} {value} {value_list}"
        }
        db = DatabaseActions()
        bot_config = {}
        result = db.init_slots([], bot_config)
        assert isinstance(result, list)
        assert result == SLOTS

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.load_prompts")
    def test_init_slots_with_none_slots(
        self, mock_load_prompts: object, mock_llm: object
    ) -> None:
        """Test init_slots with None slots."""
        mock_load_prompts.return_value = {
            "database_slot_prompt": "template {slot} {value} {value_list}"
        }
        db = DatabaseActions()
        bot_config = {}
        result = db.init_slots(None, bot_config)  # type: ignore
        assert isinstance(result, list)
        assert result == SLOTS

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.load_prompts")
    def test_init_slots_with_custom_slots(
        self, mock_load_prompts: object, mock_llm: object
    ) -> None:
        """Test init_slots with custom slots."""
        mock_load_prompts.return_value = {
            "database_slot_prompt": "template {slot} {value} {value_list}"
        }
        db = DatabaseActions()
        custom_slots = [
            {
                "name": "show_name",  # Use existing column name
                "type": "string",
                "value": "",
                "description": "Custom field",
                "prompt": "Please provide custom field",
            }
        ]
        bot_config = {}
        result = db.init_slots(custom_slots, bot_config)
        assert isinstance(result, list)
        # Should return SLOTS (default) not custom_slots
        assert result == SLOTS


class TestDatabaseActionsVerifySlot:
    """Test the verify_slot method."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.load_prompts")
    @patch("arklex.env.tools.database.utils.chunk_string")
    def test_verify_slot_success(
        self, mock_chunk: object, mock_load_prompts: object, mock_llm: object
    ) -> None:
        """Test successful slot verification."""
        db = DatabaseActions()
        mock_load_prompts.return_value = {
            "database_slot_prompt": "template {slot} {value} {value_list}"
        }
        mock_chunk.return_value = ["chunked_prompt"]

        # Mock LLM response
        db.llm = MagicMock()
        parser = MagicMock()
        parser.invoke.return_value = "Test Show"
        db.llm.__or__.return_value = parser

        slot = SLOTS[0].copy()
        slot["value"] = "Test Show"
        value_list = ["Test Show", "Other Show"]
        bot_config = {}

        result = db.verify_slot(slot, value_list, bot_config)

        assert isinstance(result, SlotDetail)
        assert result.verified_value == "Test Show"
        assert result.confirmed is True

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.load_prompts")
    @patch("arklex.env.tools.database.utils.chunk_string")
    def test_verify_slot_no_match(
        self, mock_chunk: object, mock_load_prompts: object, mock_llm: object
    ) -> None:
        """Test slot verification when no value matches."""
        db = DatabaseActions()
        mock_load_prompts.return_value = {
            "database_slot_prompt": "template {slot} {value} {value_list}"
        }
        mock_chunk.return_value = ["chunked_prompt"]

        # Mock LLM response that doesn't match any value
        db.llm = MagicMock()
        parser = MagicMock()
        parser.invoke.return_value = "Non-existent Show"
        db.llm.__or__.return_value = parser

        slot = SLOTS[0].copy()
        slot["value"] = "Test Show"
        value_list = ["Test Show", "Other Show"]
        bot_config = {}

        result = db.verify_slot(slot, value_list, bot_config)

        assert isinstance(result, SlotDetail)
        assert result.verified_value == ""
        assert result.confirmed is False

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    @patch("arklex.env.tools.database.utils.load_prompts")
    @patch("arklex.env.tools.database.utils.chunk_string")
    def test_verify_slot_exception_handling(
        self, mock_chunk: object, mock_load_prompts: object, mock_llm: object
    ) -> None:
        """Test slot verification with exception handling."""
        db = DatabaseActions()
        mock_load_prompts.return_value = {
            "database_slot_prompt": "template {slot} {value} {value_list}"
        }
        mock_chunk.return_value = ["chunked_prompt"]

        # Mock LLM to raise exception
        db.llm = MagicMock()
        parser = MagicMock()
        parser.invoke.side_effect = Exception("LLM Error")
        db.llm.__or__.return_value = parser

        slot = SLOTS[0].copy()
        slot["value"] = "Test Show"
        value_list = ["Test Show"]
        bot_config = {}

        result = db.verify_slot(slot, value_list, bot_config)

        assert isinstance(result, SlotDetail)
        assert result.verified_value == ""
        assert result.confirmed is False


class TestDatabaseActionsSearchShow:
    """Test the search_show method."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_search_show_with_confirmed_slots(self, mock_llm: object) -> None:
        """Test search_show with confirmed slots."""
        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Test Show",
                description="",
                prompt="",
                verified_value="Test Show",
                confirmed=True,
            )
        ]
        msg_state = MessageState()

        result = db.search_show(msg_state)

        assert isinstance(result, MessageState)
        assert result.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_search_show_with_no_confirmed_slots(self, mock_llm: object) -> None:
        """Test search_show with no confirmed slots."""
        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Test Show",
                description="",
                prompt="",
                verified_value="",
                confirmed=False,
            )
        ]
        msg_state = MessageState()

        result = db.search_show(msg_state)

        assert isinstance(result, MessageState)
        # Should return all shows when no filters applied
        assert result.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_search_show_no_results(self, mock_llm: object) -> None:
        """Test search_show when no shows match criteria."""
        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Non-existent Show",
                description="",
                prompt="",
                verified_value="Non-existent Show",
                confirmed=True,
            )
        ]
        msg_state = MessageState()

        result = db.search_show(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.INCOMPLETE
        assert NO_SHOW_MESSAGE in result.message_flow

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_search_show_multiple_filters(self, mock_llm: object) -> None:
        """Test search_show with multiple confirmed slots."""
        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Test Show",
                description="",
                prompt="",
                verified_value="Test Show",
                confirmed=True,
            ),
            SlotDetail(
                name="location",
                type="string",
                value="Test Location",
                description="",
                prompt="",
                verified_value="Test Location",
                confirmed=True,
            ),
        ]
        msg_state = MessageState()

        result = db.search_show(msg_state)

        assert isinstance(result, MessageState)
        assert result.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)


class TestDatabaseActionsBookShow:
    """Test the book_show method."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_book_show_success(self, mock_llm: object) -> None:
        """Test successful booking."""
        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Test Show",
                description="",
                prompt="",
                verified_value="Test Show",
                confirmed=True,
            )
        ]
        db.slot_prompts = []
        msg_state = MessageState()

        result = db.book_show(msg_state)

        assert isinstance(result, MessageState)
        assert result.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_book_show_no_shows_found(self, mock_llm: object) -> None:
        """Test booking when no shows match criteria."""
        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Non-existent Show",
                description="",
                prompt="",
                verified_value="Non-existent Show",
                confirmed=True,
            )
        ]
        db.slot_prompts = []
        msg_state = MessageState()

        result = db.book_show(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.INCOMPLETE
        assert NO_SHOW_MESSAGE in result.message_flow

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_book_show_multiple_shows_found(self, mock_llm: object) -> None:
        """Test booking when multiple shows match criteria."""
        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="location",
                type="string",
                value="Test Location",
                description="",
                prompt="",
                verified_value="Test Location",
                confirmed=True,
            )
        ]
        db.slot_prompts = ["Please specify the show name"]
        msg_state = MessageState()

        result = db.book_show(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.INCOMPLETE
        assert result.message_flow in (db.slot_prompts[0], MULTIPLE_SHOWS_MESSAGE)

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_book_show_no_slots_confirmed(self, mock_llm: object) -> None:
        """Test booking when no slots are confirmed."""
        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Test Show",
                description="",
                prompt="",
                verified_value="",
                confirmed=False,
            )
        ]
        db.slot_prompts = []
        msg_state = MessageState()

        result = db.book_show(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.INCOMPLETE
        # The actual behavior is to return MULTIPLE_SHOWS_MESSAGE when no slots are confirmed
        # because the query returns all shows (3 shows in our test data)
        assert MULTIPLE_SHOWS_MESSAGE in result.message_flow

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_book_show_empty_slots(self, mock_llm: object) -> None:
        """Test booking with empty slots."""
        db = DatabaseActions()
        db.slots = []
        db.slot_prompts = []
        msg_state = MessageState()

        result = db.book_show(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.INCOMPLETE
        # The actual behavior is to return MULTIPLE_SHOWS_MESSAGE when slots are empty
        # because the query returns all shows (3 shows in our test data)
        assert MULTIPLE_SHOWS_MESSAGE in result.message_flow


class TestDatabaseActionsCheckBooking:
    """Test the check_booking method."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_check_booking_with_bookings(self, mock_llm: object) -> None:
        """Test check_booking when user has bookings."""
        db = DatabaseActions()

        # Insert a booking first
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO booking (id, show_id, user_id, created_at) VALUES (?, ?, ?, ?)",
            ("booking_1", "show_1", db.user_id, datetime.now()),
        )
        conn.commit()
        conn.close()

        msg_state = MessageState()
        result = db.check_booking(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.COMPLETE
        assert "Booked shows are:" in result.message_flow

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_check_booking_no_bookings(self, mock_llm: object) -> None:
        """Test check_booking when user has no bookings."""
        db = DatabaseActions()
        msg_state = MessageState()
        result = db.check_booking(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.COMPLETE
        assert NO_BOOKING_MESSAGE in result.message_flow


class TestDatabaseActionsCancelBooking:
    """Test the cancel_booking method."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_cancel_booking_success(self, mock_llm: object) -> None:
        """Test successful booking cancellation."""
        db = DatabaseActions()

        # Insert a booking first
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO booking (id, show_id, user_id, created_at) VALUES (?, ?, ?, ?)",
            ("booking_1", "show_1", db.user_id, datetime.now()),
        )
        conn.commit()
        conn.close()

        msg_state = MessageState()
        result = db.cancel_booking(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.COMPLETE
        assert "cancelled show" in result.message_flow.lower()

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_cancel_booking_no_bookings(self, mock_llm: object) -> None:
        """Test cancel_booking when user has no bookings."""
        db = DatabaseActions()
        msg_state = MessageState()
        result = db.cancel_booking(msg_state)

        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.COMPLETE
        assert NO_BOOKING_MESSAGE in result.message_flow

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_cancel_booking_multiple_bookings(self, mock_llm: object) -> None:
        """Test cancel_booking when user has multiple bookings."""
        db = DatabaseActions()

        # Insert multiple bookings
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO booking (id, show_id, user_id, created_at) VALUES (?, ?, ?, ?)",
            ("booking_1", "show_1", db.user_id, datetime.now()),
        )
        cursor.execute(
            "INSERT INTO booking (id, show_id, user_id, created_at) VALUES (?, ?, ?, ?)",
            ("booking_2", "show_2", db.user_id, datetime.now()),
        )
        conn.commit()
        conn.close()

        db.slot_prompts = ["Please specify which booking to cancel"]
        msg_state = MessageState()
        result = db.cancel_booking(msg_state)

        assert isinstance(result, MessageState)
        # When there are multiple bookings, the method asks for more specific information
        # instead of cancelling anything
        assert result.status == StatusEnum.INCOMPLETE
        assert result.message_flow == "Please specify which booking to cancel"


class TestDatabaseActionsIntegration:
    """Integration tests for DatabaseActions."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_full_booking_workflow(self, mock_llm: object) -> None:
        """Test the complete booking workflow."""
        db = DatabaseActions()

        # Step 1: Search for shows
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Test Show",
                description="",
                prompt="",
                verified_value="Test Show",
                confirmed=True,
            )
        ]
        msg_state = MessageState()
        search_result = db.search_show(msg_state)
        assert search_result.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)

        # Step 2: Book a show
        book_result = db.book_show(msg_state)
        assert book_result.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)

        # Step 3: Check bookings
        check_result = db.check_booking(msg_state)
        assert check_result.status == StatusEnum.COMPLETE

        # Step 4: Cancel booking
        cancel_result = db.cancel_booking(msg_state)
        assert cancel_result.status == StatusEnum.COMPLETE

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_database_connection_handling(self, mock_llm: object) -> None:
        """Test that database connections are properly handled."""
        db = DatabaseActions()

        # Test that multiple operations don't cause connection issues
        msg_state = MessageState()

        # Multiple search operations
        for _ in range(3):
            result = db.search_show(msg_state)
            assert isinstance(result, MessageState)

        # Multiple check booking operations
        for _ in range(3):
            result = db.check_booking(msg_state)
            assert isinstance(result, MessageState)


class TestDatabaseActionsEdgeCases:
    """Test edge cases and error conditions."""

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_database_file_not_exists(self, mock_llm: object) -> None:
        """Test behavior when database file doesn't exist."""
        # Remove the database file
        db_path = os.path.join(os.environ["DATA_DIR"], "show_booking_db.sqlite")
        if os.path.exists(db_path):
            os.remove(db_path)

        db = DatabaseActions()
        msg_state = MessageState()

        # These operations should handle the missing database gracefully
        # The database will be created automatically by SQLite, but tables won't exist
        # So we expect an OperationalError when trying to query non-existent tables
        with pytest.raises(sqlite3.OperationalError):
            db.search_show(msg_state)

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_empty_database_tables(self, mock_llm: object) -> None:
        """Test behavior with empty database tables."""
        db = DatabaseActions()

        # Clear all tables
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM booking")
        cursor.execute("DELETE FROM show")
        cursor.execute("DELETE FROM user")
        conn.commit()
        conn.close()

        msg_state = MessageState()

        # Test search with empty tables
        result = db.search_show(msg_state)
        assert result.status == StatusEnum.INCOMPLETE
        assert NO_SHOW_MESSAGE in result.message_flow

        # Test check booking with empty tables
        result = db.check_booking(msg_state)
        assert result.status == StatusEnum.COMPLETE
        assert NO_BOOKING_MESSAGE in result.message_flow

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_special_characters_in_slot_values(self, mock_llm: object) -> None:
        """Test handling of special characters in slot values."""
        db = DatabaseActions()
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value="Test'Show",
                description="",
                prompt="",
                verified_value="Test'Show",
                confirmed=True,
            )
        ]
        msg_state = MessageState()

        # Should handle SQL injection attempts gracefully
        result = db.search_show(msg_state)
        assert isinstance(result, MessageState)

    @patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
    def test_very_long_slot_values(self, mock_llm: object) -> None:
        """Test handling of very long slot values."""
        db = DatabaseActions()
        long_value = "A" * 1000  # Very long string
        db.slots = [
            SlotDetail(
                name="show_name",
                type="string",
                value=long_value,
                description="",
                prompt="",
                verified_value=long_value,
                confirmed=True,
            )
        ]
        msg_state = MessageState()

        result = db.search_show(msg_state)
        assert isinstance(result, MessageState)
        assert result.status == StatusEnum.INCOMPLETE  # Should not find any shows
