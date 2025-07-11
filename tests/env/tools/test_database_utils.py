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
    NO_BOOKING_MESSAGE,
    SLOTS,
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
    # Insert a show
    cursor.execute(
        "INSERT OR IGNORE INTO show (id, show_name, genre, date, time, description, location, price, available_seats) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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


@patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
def test_init_and_login(mock_llm: object) -> None:
    db = DatabaseActions()
    assert db.user_id == "user_be6e1836-8fe9-4938-b2d0-48f810648e72"
    assert db.db_path.endswith("show_booking_db.sqlite")
    assert db.log_in() is True
    # Test login with non-existent user
    db2 = DatabaseActions(user_id="nonexistent")
    assert db2.log_in() is False


@patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
@patch("arklex.env.tools.database.utils.load_prompts")
@patch("arklex.env.tools.database.utils.chunk_string", return_value=["prompt"])
def test_init_slots_and_verify_slot(
    mock_chunk: object, mock_load_prompts: object, mock_llm: object
) -> None:
    db = DatabaseActions()
    # Mock prompt and LLM
    mock_load_prompts.return_value = {
        "database_slot_prompt": "template {slot} {value} {value_list}"
    }
    db.llm = MagicMock()
    parser = MagicMock()
    parser.invoke.return_value = "Test Show"
    db.llm.__or__.return_value = parser
    slot = SLOTS[0].copy()
    slot["value"] = "Test Show"
    bot_config = {}
    # Should confirm slot
    detail = db.verify_slot(slot, ["Test Show"], bot_config)
    assert detail.verified_value == "Test Show"
    assert detail.confirmed is True
    # Should not confirm if value not in answer
    parser.invoke.return_value = "Other Show"
    detail2 = db.verify_slot(slot, ["Test Show"], bot_config)
    assert detail2.confirmed is False
    # Test init_slots
    db.llm.__or__.return_value = parser
    slots = db.init_slots([slot], bot_config)
    assert isinstance(slots, list)


@patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
def test_search_show_and_book_show(mock_llm: object) -> None:
    db = DatabaseActions()
    # Set up slots as confirmed
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
    # Test search_show
    result = db.search_show(msg_state)
    assert (
        result.status == StatusEnum.COMPLETE or result.status == StatusEnum.INCOMPLETE
    )
    # Test book_show
    result2 = db.book_show(msg_state)
    assert result2.status in (StatusEnum.COMPLETE, StatusEnum.INCOMPLETE)
    # Test book_show with ambiguous info
    db.slots = []
    result3 = db.book_show(msg_state)
    assert result3.status == StatusEnum.INCOMPLETE


@patch("arklex.env.tools.database.utils.ChatOpenAI", autospec=True)
def test_check_and_cancel_booking(mock_llm: object) -> None:
    db = DatabaseActions()
    # Insert a booking
    db_path = db.db_path
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO booking (id, show_id, user_id, created_at) VALUES (?, ?, ?, ?)",
        ("booking_1", "show_1", db.user_id, datetime.now()),
    )
    conn.commit()
    conn.close()
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
    # Test check_booking
    result = db.check_booking(msg_state)
    assert NO_BOOKING_MESSAGE not in result.message_flow
    # Test cancel_booking
    result2 = db.cancel_booking(msg_state)
    assert (
        "cancelled show" in result2.message_flow.lower()
        or result2.status == StatusEnum.COMPLETE
    )
    # Test cancel_booking when no booking exists
    result3 = db.cancel_booking(msg_state)
    assert NO_BOOKING_MESSAGE in result3.message_flow
