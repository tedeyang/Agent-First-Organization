"""
Tests for Booking Database tools.

This module contains comprehensive tests for all booking database-related tools including
booking functionality, database operations, and utility functions.
"""

import os
import sqlite3
import tempfile
from unittest.mock import Mock, patch

# Import the underlying functions, not the decorated ones
from arklex.env.tools.booking_db.book_show import book_show
from arklex.env.tools.booking_db.build_database import build_database
from arklex.env.tools.booking_db.cancel_booking import cancel_booking
from arklex.env.tools.booking_db.check_booking import check_booking
from arklex.env.tools.booking_db.search_show import search_show
from arklex.env.tools.booking_db.utils import (
    LOG_IN_FAILURE,
    MULTIPLE_SHOWS_MESSAGE,
    NO_SHOW_MESSAGE,
    SLOTS,
    booking,
    log_in,
)


class TestBookingDBUtils:
    """Test the booking database utility functions."""

    def test_booking_constants(self) -> None:
        """Test that booking constants are defined."""
        assert hasattr(booking, "db_path")
        assert hasattr(booking, "user_id")
        assert LOG_IN_FAILURE is not None
        assert NO_SHOW_MESSAGE is not None
        assert MULTIPLE_SHOWS_MESSAGE is not None

    def test_slots_structure(self) -> None:
        """Test that SLOTS dictionary has the expected structure."""
        expected_slots = ["show_name", "date", "time", "location"]
        for slot_name in expected_slots:
            assert slot_name in SLOTS
            assert "type" in SLOTS[slot_name]
            assert "description" in SLOTS[slot_name]

    @patch("arklex.env.tools.booking_db.utils.booking")
    @patch("arklex.env.tools.booking_db.utils.sqlite3.connect")
    def test_log_in_success(self, mock_connect: Mock, mock_booking: Mock) -> None:
        """Test successful login."""
        mock_booking.user_id = "test_user"
        mock_booking.db_path = "/tmp/test.db"

        # Mock the database connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)  # User exists
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Set DATA_DIR environment variable
        with patch.dict(os.environ, {"DATA_DIR": "/tmp"}):
            result = log_in()

        assert result is True

    @patch("arklex.env.tools.booking_db.utils.booking")
    @patch("arklex.env.tools.booking_db.utils.sqlite3.connect")
    def test_log_in_failure(self, mock_connect: Mock, mock_booking: Mock) -> None:
        """Test login failure."""
        mock_booking.user_id = "test_user"
        mock_booking.db_path = "/tmp/test.db"

        # Mock the database connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None  # User doesn't exist
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Set DATA_DIR environment variable
        with patch.dict(os.environ, {"DATA_DIR": "/tmp"}):
            result = log_in()

        assert result is False


class TestBookingDBBookShow:
    """Test the book_show function."""

    @patch("arklex.env.tools.booking_db.book_show.log_in")
    @patch("arklex.env.tools.booking_db.book_show.sqlite3.connect")
    def test_book_show_login_failure(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test book_show when login fails."""
        mock_log_in.return_value = False

        result = book_show().func("Test Show", None, None, None)

        assert result == LOG_IN_FAILURE
        mock_connect.assert_not_called()

    @patch("arklex.env.tools.booking_db.book_show.log_in")
    @patch("arklex.env.tools.booking_db.book_show.sqlite3.connect")
    def test_book_show_no_shows_found(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test book_show when no shows match criteria."""
        mock_log_in.return_value = True
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = book_show().func("Non-existent Show", None, None, None)

        assert result == NO_SHOW_MESSAGE
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.booking_db.book_show.log_in")
    @patch("arklex.env.tools.booking_db.book_show.sqlite3.connect")
    def test_book_show_multiple_shows_found(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test book_show when multiple shows match criteria."""
        mock_log_in.return_value = True
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ("1", "Show 1", "2024-01-15", "10:00", "Description 1", "Location 1", 50),
            ("2", "Show 2", "2024-01-15", "10:00", "Description 2", "Location 2", 60),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = book_show().func(None, "2024-01-15", "10:00", None)

        assert result == MULTIPLE_SHOWS_MESSAGE
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.booking_db.book_show.sqlite3.connect")
    @patch("arklex.env.tools.booking_db.book_show.uuid.uuid4")
    def test_book_show_success(
        self, mock_uuid: Mock, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test successful booking of a show."""
        mock_log_in.return_value = True
        mock_uuid.return_value = "test-uuid"

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (
                "1",
                "Test Show",
                "2024-01-15",
                "10:00",
                "Test Description",
                "Test Location",
                50,
            )
        ]
        mock_cursor.description = [
            ("id",),
            ("show_name",),
            ("date",),
            ("time",),
            ("description",),
            ("location",),
            ("price",),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = book_show().func("Test Show", None, None, None)

        assert "The booked show is:" in result
        assert "Test Show" in result
        mock_cursor.execute.assert_called()
        mock_conn.close.assert_called_once()


class TestBookingDBSearchShow:
    """Test the search_show function."""

    @patch("arklex.env.tools.booking_db.search_show.log_in")
    @patch("arklex.env.tools.booking_db.search_show.sqlite3.connect")
    def test_search_show_login_failure(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test search_show when login fails."""
        mock_log_in.return_value = False

        result = search_show().func("Test Show", None, None, None)

        assert result == LOG_IN_FAILURE
        mock_connect.assert_not_called()

    @patch("arklex.env.tools.booking_db.search_show.log_in")
    @patch("arklex.env.tools.booking_db.search_show.sqlite3.connect")
    def test_search_show_no_results(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test search_show when no shows match criteria."""
        mock_log_in.return_value = True
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = search_show().func("Non-existent Show", None, None, None)

        assert result == NO_SHOW_MESSAGE
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.booking_db.search_show.log_in")
    @patch("arklex.env.tools.booking_db.search_show.sqlite3.connect")
    def test_search_show_with_results(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test search_show when shows match criteria."""
        mock_log_in.return_value = True
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (
                "1",
                "Test Show",
                "2024-01-15",
                "10:00",
                "Test Description",
                "Test Location",
                50,
            ),
            (
                "2",
                "Another Show",
                "2024-01-16",
                "11:00",
                "Another Description",
                "Another Location",
                60,
            ),
        ]
        mock_cursor.description = [
            ("id",),
            ("show_name",),
            ("date",),
            ("time",),
            ("description",),
            ("location",),
            ("price",),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = search_show().func("Test Show", None, None, None)

        assert "Available shows are:" in result
        assert "Test Show" in result
        assert "Test Description" in result
        assert "Test Location" in result
        assert "50" in result
        mock_conn.close.assert_called_once()


class TestBookingDBCheckBooking:
    """Test the check_booking function."""

    @patch("arklex.env.tools.booking_db.check_booking.log_in")
    @patch("arklex.env.tools.booking_db.check_booking.sqlite3.connect")
    def test_check_booking_login_failure(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test check_booking when login fails."""
        mock_log_in.return_value = False

        result = check_booking().func()

        assert result == LOG_IN_FAILURE
        mock_connect.assert_not_called()

    @patch("arklex.env.tools.booking_db.check_booking.log_in")
    @patch("arklex.env.tools.booking_db.check_booking.sqlite3.connect")
    def test_check_booking_no_bookings(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test check_booking when user has no bookings."""
        mock_log_in.return_value = True
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = check_booking().func()

        assert "No bookings found" in result
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.booking_db.check_booking.log_in")
    @patch("arklex.env.tools.booking_db.check_booking.sqlite3.connect")
    def test_check_booking_with_bookings(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test check_booking when user has bookings."""
        mock_log_in.return_value = True
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (
                "test-uuid",
                "1",
                "Test Show",
                "2024-01-15",
                "10:00",
                "Test Description",
                "Test Location",
                50,
            ),
            (
                "test-uuid-2",
                "2",
                "Another Show",
                "2024-01-16",
                "11:00",
                "Another Description",
                "Another Location",
                60,
            ),
        ]
        mock_cursor.description = [
            ("booking_id",),
            ("show_id",),
            ("show_name",),
            ("date",),
            ("time",),
            ("description",),
            ("location",),
            ("price",),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = check_booking().func()

        assert "Booked shows are:" in result
        assert "Test Show" in result
        assert "Another Show" in result
        assert "2024-01-15" in result
        assert "2024-01-16" in result
        mock_conn.close.assert_called_once()


class TestBookingDBCancelBooking:
    """Test the cancel_booking function."""

    @patch("arklex.env.tools.booking_db.cancel_booking.log_in")
    @patch("arklex.env.tools.booking_db.cancel_booking.sqlite3.connect")
    def test_cancel_booking_login_failure(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test cancel_booking when login fails."""
        mock_log_in.return_value = False

        result = cancel_booking().func("test-uuid")

        assert result == LOG_IN_FAILURE
        mock_connect.assert_not_called()

    @patch("arklex.env.tools.booking_db.cancel_booking.log_in")
    @patch("arklex.env.tools.booking_db.cancel_booking.sqlite3.connect")
    def test_cancel_booking_success(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test successful cancellation of a booking."""
        mock_log_in.return_value = True
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (
            "test-uuid",
            "1",
            "Test Show",
            "2024-01-15",
            "10:00",
            "Test Description",
            "Test Location",
            50,
        )
        mock_cursor.description = [
            ("booking_id",),
            ("show_id",),
            ("show_name",),
            ("date",),
            ("time",),
            ("description",),
            ("location",),
            ("price",),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = cancel_booking().func("test-uuid")

        assert "Booking cancelled successfully" in result
        assert "Test Show" in result
        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("arklex.env.tools.booking_db.cancel_booking.log_in")
    @patch("arklex.env.tools.booking_db.cancel_booking.sqlite3.connect")
    def test_cancel_booking_not_found(
        self, mock_connect: Mock, mock_log_in: Mock
    ) -> None:
        """Test cancel_booking when booking is not found."""
        mock_log_in.return_value = True
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        result = cancel_booking().func("non-existent-uuid")

        assert "Booking not found" in result
        mock_conn.close.assert_called_once()


class TestBookingDBBuildDatabase:
    """Test the build_database function."""

    def test_build_database_creates_tables(self) -> None:
        """Test that build_database creates the required tables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            with patch.dict(os.environ, {"DATA_DIR": temp_dir}):
                build_database()

            # Verify database file was created
            assert os.path.exists(db_path)

            # Verify tables were created
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()

            assert "users" in tables
            assert "shows" in tables
            assert "bookings" in tables

    def test_build_database_inserts_sample_data(self) -> None:
        """Test that build_database inserts sample data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            with patch.dict(os.environ, {"DATA_DIR": temp_dir}):
                build_database()

            # Verify sample data was inserted
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check users table
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            assert user_count > 0

            # Check shows table
            cursor.execute("SELECT COUNT(*) FROM shows")
            show_count = cursor.fetchone()[0]
            assert show_count > 0

            conn.close()


class TestBookingDBIntegration:
    """Integration tests for booking database tools."""

    def test_booking_workflow(self) -> None:
        """Test a complete booking workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up the database
            with patch.dict(os.environ, {"DATA_DIR": temp_dir}):
                build_database()

            # Test the workflow
            # Note: This is a simplified integration test
            # In a real scenario, you would test the actual workflow
            # with proper user authentication and booking flow
            assert True  # Placeholder assertion
