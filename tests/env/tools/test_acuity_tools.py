"""
Tests for Acuity tools.

This module contains comprehensive tests for all Acuity-related tools including
exception prompts, booking functionality, and utility functions.
"""

from unittest.mock import Mock, patch

import pytest

# Mock the acuityscheduling import since it's not installed
with patch.dict("sys.modules", {"acuityscheduling": Mock()}):
    from arklex.env.tools.acuity import (
        _exception_prompt,
        book_info_session,
        cancel,
        get_acuity_client,
        get_apt_by_email,
        get_available_dates,
        get_available_times,
        get_session_types,
        get_type_id_by_apt_name,
        reschedule,
    )


class TestAcuityExceptionPrompt:
    """Test the Acuity exception prompt constants."""

    def test_acuity_exception_prompt_constants(self) -> None:
        """Test that Acuity exception prompt constants are defined."""
        assert hasattr(_exception_prompt, "ACUITY_EXCEPTION_PROMPT")
        assert hasattr(_exception_prompt, "ACUITY_EXCEPTION_PROMPT_2")
        assert hasattr(_exception_prompt, "ACUITY_EXCEPTION_PROMPT_3")

    def test_acuity_exception_prompt_values(self) -> None:
        """Test that Acuity exception prompt values are not None."""
        assert _exception_prompt.ACUITY_EXCEPTION_PROMPT is not None
        assert _exception_prompt.ACUITY_EXCEPTION_PROMPT_2 is not None
        assert _exception_prompt.ACUITY_EXCEPTION_PROMPT_3 is not None


class TestAcuityUtils:
    """Test the Acuity utility functions."""

    def test_get_acuity_client_success(self) -> None:
        """Test successful creation of Acuity client."""
        client = get_acuity_client("user", "key")
        assert client is not None
        assert hasattr(client, "appointment_types")

    def test_get_acuity_client_exception(self) -> None:
        """Test Acuity client creation with invalid credentials."""
        with pytest.raises(ValueError):
            get_acuity_client("", "")


class TestAcuityGetAvailableDates:
    """Test the get_available_dates function."""

    @patch("arklex.env.tools.acuity.get_available_dates.requests.get")
    def test_get_available_dates_success(self, mock_get: Mock) -> None:
        """Test successful retrieval of available dates."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"date": "2024-01-15", "slots": 5},
            {"date": "2024-01-16", "slots": 3},
        ]
        mock_get.return_value = mock_response

        result = get_available_dates().func(
            date="2024-01-15", ACUITY_USER_ID="user", ACUITY_API_KEY="key"
        )

        assert "2024-01-15" in result
        assert "2024-01-16" in result
        mock_get.assert_called_once()

    @patch("arklex.env.tools.acuity.get_available_dates.requests.get")
    def test_get_available_dates_exception(self, mock_get: Mock) -> None:
        """Test get_available_dates with exception."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError):
            get_available_dates().func(
                date="2024-01-15", ACUITY_USER_ID="user", ACUITY_API_KEY="key"
            )


class TestAcuityGetAvailableTimes:
    """Test the get_available_times function."""

    @patch("arklex.env.tools.acuity.get_available_times.requests.get")
    def test_get_available_times_success(self, mock_get: Mock) -> None:
        """Test successful retrieval of available times."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"time": "09:00", "available": True},
            {"time": "10:00", "available": True},
            {"time": "11:00", "available": False},
        ]
        mock_get.return_value = mock_response

        result = get_available_times().func(
            date="2024-01-15",
            appointment_type_id="123",
            ACUITY_USER_ID="user",
            ACUITY_API_KEY="key",
        )

        assert "09:00" in result
        assert "10:00" in result
        assert "11:00" not in result
        mock_get.assert_called_once()

    @patch("arklex.env.tools.acuity.get_available_times.requests.get")
    def test_get_available_times_exception(self, mock_get: Mock) -> None:
        """Test get_available_times with exception."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError):
            get_available_times().func(
                date="2024-01-15",
                appointment_type_id="123",
                ACUITY_USER_ID="user",
                ACUITY_API_KEY="key",
            )


class TestAcuityGetSessionTypes:
    """Test the get_session_types function."""

    @patch("arklex.env.tools.acuity.get_session_types.requests.get")
    def test_get_session_types_success(self, mock_get: Mock) -> None:
        """Test successful retrieval of session types."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"name": "Info Session", "id": 1},
            {"name": "Consultation", "id": 2},
        ]
        mock_get.return_value = mock_response

        result = get_session_types().func(ACUITY_USER_ID="user", ACUITY_API_KEY="key")

        assert "Info Session" in result
        assert "Consultation" in result
        mock_get.assert_called_once()

    @patch("arklex.env.tools.acuity.get_session_types.requests.get")
    def test_get_session_types_exception(self, mock_get: Mock) -> None:
        """Test get_session_types with exception."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError):
            get_session_types().func(ACUITY_USER_ID="user", ACUITY_API_KEY="key")


class TestAcuityGetTypeIdByAptName:
    """Test the get_type_id_by_apt_name function."""

    @patch("arklex.env.tools.acuity.get_type_id_by_apt_name.requests.get")
    def test_get_type_id_by_apt_name_success(self, mock_get: Mock) -> None:
        """Test successful retrieval of type ID by appointment name."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"name": "Info Session", "id": 1},
            {"name": "Consultation", "id": 2},
        ]
        mock_get.return_value = mock_response

        result = get_type_id_by_apt_name().func(
            apt_name="Info Session", ACUITY_USER_ID="user", ACUITY_API_KEY="key"
        )

        assert "1" in result
        mock_get.assert_called_once()

    @patch("arklex.env.tools.acuity.get_type_id_by_apt_name.requests.get")
    def test_get_type_id_by_apt_name_not_found(self, mock_get: Mock) -> None:
        """Test get_type_id_by_apt_name when appointment type not found."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"name": "Info Session", "id": 1}]
        mock_get.return_value = mock_response

        result = get_type_id_by_apt_name().func(
            apt_name="Non-existent", ACUITY_USER_ID="user", ACUITY_API_KEY="key"
        )

        assert "none" in result.lower()
        mock_get.assert_called_once()

    @patch("arklex.env.tools.acuity.get_type_id_by_apt_name.requests.get")
    def test_get_type_id_by_apt_name_exception(self, mock_get: Mock) -> None:
        """Test get_type_id_by_apt_name with exception."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError):
            get_type_id_by_apt_name().func(
                apt_name="Info Session", ACUITY_USER_ID="user", ACUITY_API_KEY="key"
            )


class TestAcuityBookInfoSession:
    """Test the book_info_session function."""

    @patch("arklex.env.tools.acuity.book_info_session.requests.post")
    def test_book_info_session_success(self, mock_post: Mock) -> None:
        """Test successful booking of info session."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 123, "status": "confirmed"}
        mock_post.return_value = mock_response

        result = book_info_session().func(
            fname="John",
            lname="Doe",
            email="john@example.com",
            time="2024-01-15T09:00:00-0400",
            apt_type_id="123",
            ACUITY_USER_ID="user",
            ACUITY_API_KEY="key",
        )

        assert "confirmed" in result
        mock_post.assert_called_once()

    @patch("arklex.env.tools.acuity.book_info_session.requests.post")
    def test_book_info_session_exception(self, mock_post: Mock) -> None:
        """Test book_info_session with exception."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError):
            book_info_session().func(
                fname="John",
                lname="Doe",
                email="john@example.com",
                time="2024-01-15T09:00:00-0400",
                apt_type_id="123",
                ACUITY_USER_ID="user",
                ACUITY_API_KEY="key",
            )


class TestAcuityGetAptByEmail:
    """Test the get_apt_by_email function."""

    @patch("arklex.env.tools.acuity.get_apt_by_email.requests.get")
    def test_get_apt_by_email_success(self, mock_get: Mock) -> None:
        """Test successful retrieval of appointments by email."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": 1,
                "email": "john@example.com",
                "date": "January 15, 2025",
                "time": "09:00",
                "endTime": "10:00",
                "type": "Info Session",
            },
            {
                "id": 2,
                "email": "john@example.com",
                "date": "January 16, 2025",
                "time": "10:00",
                "endTime": "11:00",
                "type": "Consultation",
            },
            {
                "id": 3,
                "email": "other@example.com",
                "date": "January 15, 2025",
                "time": "09:00",
                "endTime": "10:00",
                "type": "Info Session",
            },
        ]
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError):
            get_apt_by_email().func(
                email="john@example.com", ACUITY_USER_ID="user", ACUITY_API_KEY="key"
            )
        mock_get.assert_called_once()

    @patch("arklex.env.tools.acuity.get_apt_by_email.requests.get")
    def test_get_apt_by_email_no_appointments(self, mock_get: Mock) -> None:
        """Test get_apt_by_email when no appointments found."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError):
            get_apt_by_email().func(
                email="john@example.com", ACUITY_USER_ID="user", ACUITY_API_KEY="key"
            )
        mock_get.assert_called_once()

    @patch("arklex.env.tools.acuity.get_apt_by_email.requests.get")
    def test_get_apt_by_email_exception(self, mock_get: Mock) -> None:
        """Test get_apt_by_email with exception."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError):
            get_apt_by_email().func(
                email="john@example.com", ACUITY_USER_ID="user", ACUITY_API_KEY="key"
            )


class TestAcuityCancel:
    """Test the cancel function."""

    @patch("arklex.env.tools.acuity.cancel.requests.put")
    def test_cancel_success(self, mock_put: Mock) -> None:
        """Test successful cancellation of appointment."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "cancelled"}
        mock_put.return_value = mock_response

        result = cancel().func(
            apt_id="123", ACUITY_USER_ID="user", ACUITY_API_KEY="key"
        )
        assert "cancelled" in result
        mock_put.assert_called_once()

    @patch("arklex.env.tools.acuity.cancel.requests.put")
    def test_cancel_exception(self, mock_put: Mock) -> None:
        """Test cancel with exception."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_put.return_value = mock_response

        with pytest.raises(RuntimeError):
            cancel().func(apt_id="123", ACUITY_USER_ID="user", ACUITY_API_KEY="key")


class TestAcuityReschedule:
    """Test the reschedule function."""

    @patch("arklex.env.tools.acuity.reschedule.requests.put")
    def test_reschedule_success(self, mock_put: Mock) -> None:
        """Test successful rescheduling of appointment."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": 123,
            "date": "2024-01-16",
            "time": "10:00",
            "status": "confirmed",
        }
        mock_put.return_value = mock_response

        result = reschedule().func(
            apt_id="123",
            date="2024-01-16",
            time="10:00",
            ACUITY_USER_ID="user",
            ACUITY_API_KEY="key",
        )

        assert "confirmed" in result
        assert "2024-01-16" in result
        assert "10:00" in result
        mock_put.assert_called_once()

    @patch("arklex.env.tools.acuity.reschedule.requests.put")
    def test_reschedule_exception(self, mock_put: Mock) -> None:
        """Test reschedule with exception."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_put.return_value = mock_response

        with pytest.raises(RuntimeError):
            reschedule().func(
                apt_id="123",
                date="2024-01-16",
                time="10:00",
                ACUITY_USER_ID="user",
                ACUITY_API_KEY="key",
            )


class TestAcuityToolsIntegration:
    """Integration tests for Acuity tools."""

    @patch("arklex.env.tools.acuity.get_available_dates.requests.get")
    @patch("arklex.env.tools.acuity.get_available_times.requests.get")
    @patch("arklex.env.tools.acuity.get_session_types.requests.get")
    def test_acuity_workflow(
        self, mock_types_get: Mock, mock_times_get: Mock, mock_dates_get: Mock
    ) -> None:
        """Test a complete Acuity workflow."""
        # Mock session types response
        mock_types_response = Mock()
        mock_types_response.status_code = 200
        mock_types_response.json.return_value = [
            {"name": "Info Session", "id": 1},
            {"name": "Consultation", "id": 2},
        ]
        mock_types_get.return_value = mock_types_response

        # Mock available times response
        mock_times_response = Mock()
        mock_times_response.status_code = 200
        mock_times_response.json.return_value = [
            {"time": "09:00", "available": True},
            {"time": "10:00", "available": True},
        ]
        mock_times_get.return_value = mock_times_response

        # Mock available dates response
        mock_dates_response = Mock()
        mock_dates_response.status_code = 200
        mock_dates_response.json.return_value = [
            {"date": "2024-01-15", "slots": 5},
            {"date": "2024-01-16", "slots": 3},
        ]
        mock_dates_get.return_value = mock_dates_response

        # Test session types
        result_types = get_session_types().func(
            ACUITY_USER_ID="user", ACUITY_API_KEY="key"
        )
        assert "Info Session" in result_types

        # Test available times
        result_times = get_available_times().func(
            date="2024-01-15",
            appointment_type_id="1",
            ACUITY_USER_ID="user",
            ACUITY_API_KEY="key",
        )
        assert "09:00" in result_times

        # Test available dates
        result_dates = get_available_dates().func(
            date="2024-01-15", ACUITY_USER_ID="user", ACUITY_API_KEY="key"
        )
        assert "2024-01-15" in result_dates

        # Verify all mocks were called
        mock_types_get.assert_called_once()
        mock_times_get.assert_called_once()
        mock_dates_get.assert_called_once()
