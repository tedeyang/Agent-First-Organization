"""
Integration tests for HubSpot tools.

This module contains comprehensive integration tests for all HubSpot tools,
including proper mocking of external services and edge case testing.
These tests validate the complete HubSpot integration workflow from API
calls to response processing and error handling.
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from hubspot.crm.objects.emails import ApiException as EmailsApiException
from hubspot.crm.objects.meetings import ApiException as MeetingsApiException

from arklex.env.tools.hubspot._exception_prompt import HubspotExceptionPrompt
from arklex.env.tools.hubspot.check_available import check_available
from arklex.env.tools.hubspot.create_meeting import create_meeting
from arklex.env.tools.hubspot.create_ticket import create_ticket
from arklex.env.tools.hubspot.find_contact_by_email import find_contact_by_email
from arklex.env.tools.hubspot.find_owner_id_by_contact_id import (
    find_owner_id_by_contact_id,
)
from arklex.utils.exceptions import AuthenticationError, ToolExecutionError

# Extract underlying functions from decorated tool functions for direct testing
# This allows us to test the core functionality without the decorator wrapper
find_contact_by_email_func = find_contact_by_email().func
create_ticket_func = create_ticket().func
find_owner_id_by_contact_id_func = find_owner_id_by_contact_id().func
check_available_func = check_available().func
create_meeting_func = create_meeting().func


class TestHubSpotFindContactByEmail:
    """
    Integration tests for find_contact_by_email tool.

    This test class validates the contact search functionality, including
    successful searches, contact not found scenarios, API errors, and
    authentication failures.
    """

    @patch("arklex.env.tools.hubspot.find_contact_by_email.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.find_contact_by_email.authenticate_hubspot")
    def test_find_contact_by_email_success(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """
        Test successful contact search by email with communication tracking.

        This test validates that the find_contact_by_email tool can successfully
        search for contacts by email address, create communication records,
        and associate them with the found contact.
        """
        # Mock authentication to return a valid token
        # This simulates successful HubSpot authentication
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Execute the find_contact_by_email function with test parameters
        # This tests the complete workflow from email to contact details
        result = find_contact_by_email_func(
            email="john.doe@example.com",
            chat="Hello, I have a question about your products",
            access_token="test_token",
        )

        # Parse the JSON result and validate the structure
        # This ensures the response format is correct
        expected_result = {
            "contact_id": "12345",
            "contact_email": "john.doe@example.com",
            "contact_first_name": "John",
            "contact_last_name": "Doe",
        }
        assert json.loads(result) == expected_result, (
            "Contact details should match expected format"
        )

        # Verify all expected API calls were made
        # This ensures the tool performs all required operations
        mock_hubspot_client.crm.contacts.search_api.do_search.assert_called_once()
        mock_hubspot_client.crm.objects.communications.basic_api.create.assert_called_once()
        mock_hubspot_client.crm.associations.v4.basic_api.create.assert_called_once()

    @patch("arklex.env.tools.hubspot.find_contact_by_email.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.find_contact_by_email.authenticate_hubspot")
    def test_find_contact_by_email_not_found(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """
        Test contact search when no matching contact exists.

        This test validates that the find_contact_by_email tool properly handles
        cases where the email address doesn't match any existing contacts
        and raises an appropriate error with a meaningful message.
        """
        # Mock authentication to return a valid token
        # This simulates successful HubSpot authentication
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock empty search results
        # This simulates a valid API response with no matching contacts
        mock_search_response = MagicMock()
        mock_search_response.to_dict.return_value = {"total": 0, "results": []}
        mock_hubspot_client.crm.contacts.search_api.do_search.return_value = (
            mock_search_response
        )

        # Test that the function raises an appropriate error
        # This validates the error handling for no matching contacts
        with pytest.raises(ToolExecutionError) as exc_info:
            find_contact_by_email_func(
                email="nonexistent@example.com", chat="Hello", access_token="test_token"
            )

        # Verify the error message contains the expected prompt
        # This ensures the error message is informative and consistent
        assert HubspotExceptionPrompt.USER_NOT_FOUND_PROMPT in str(exc_info.value), (
            "Error message should contain user not found prompt"
        )

    @patch("arklex.env.tools.hubspot.find_contact_by_email.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.find_contact_by_email.authenticate_hubspot")
    def test_find_contact_by_email_api_exception(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """
        Test contact search when HubSpot API throws an exception.

        This test validates that the find_contact_by_email tool properly handles
        API errors and exceptions, ensuring that failures are caught and
        appropriate error messages are provided to the user.
        """
        # Mock authentication to return a valid token
        # This simulates successful HubSpot authentication
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Simulate API error by making search method raise an exception
        # This simulates a network error, rate limiting, or other API issue
        mock_hubspot_client.crm.contacts.search_api.do_search.side_effect = (
            EmailsApiException(status=400)
        )

        # Test that the function raises an appropriate error
        # This validates the error handling for API exceptions
        with pytest.raises(ToolExecutionError) as exc_info:
            find_contact_by_email_func(
                email="test@example.com", chat="Hello", access_token="test_token"
            )

        # Verify the error message contains the expected prompt
        # This ensures the error message is informative and consistent
        assert HubspotExceptionPrompt.USER_NOT_FOUND_PROMPT in str(exc_info.value), (
            "Error message should contain user not found prompt"
        )

    @patch("arklex.env.tools.hubspot.find_contact_by_email.authenticate_hubspot")
    def test_find_contact_by_email_authentication_error(
        self, mock_authenticate: Mock
    ) -> None:
        """
        Test contact search with invalid or missing access token.

        This test validates that the find_contact_by_email tool properly handles
        authentication failures and provides appropriate error messages
        when the access token is invalid or missing.
        """
        # Mock authentication to raise an authentication error
        # This simulates an invalid or missing access token
        mock_authenticate.side_effect = AuthenticationError("Missing access token")

        # Test that the function raises an authentication error
        # This validates the authentication error handling
        with pytest.raises(AuthenticationError):
            find_contact_by_email_func(
                email="test@example.com", chat="Hello", access_token=""
            )


class TestHubSpotCreateTicket:
    """
    Integration tests for create_ticket tool.

    This test class validates the ticket creation functionality, including
    successful ticket creation, contact association, and error handling.
    """

    @patch("arklex.env.tools.hubspot.create_ticket.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_ticket.authenticate_hubspot")
    def test_create_ticket_success(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_ticket_client: MagicMock,
    ) -> None:
        """
        Test successful ticket creation with contact association.

        This test validates that the create_ticket tool can successfully
        create a support ticket in HubSpot and associate it with the
        specified customer contact.
        """
        # Mock authentication to return a valid token
        # This simulates successful HubSpot authentication
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_ticket_client

        # Execute the create_ticket function with test parameters
        # This tests the complete workflow from issue description to ticket creation
        result = create_ticket_func(
            cus_cid="12345",
            issue="I need help with my order",
            access_token="test_token",
        )

        # Verify the ticket ID is returned
        # This ensures the ticket was created successfully
        expected_result = "ticket_123"
        assert result == expected_result, "Should return the created ticket ID"

        # Verify ticket creation and association calls were made
        # This ensures the tool performs all required operations
        mock_hubspot_ticket_client.crm.tickets.basic_api.create.assert_called_once()
        mock_hubspot_ticket_client.crm.associations.v4.basic_api.create.assert_called_once()

    @patch("arklex.env.tools.hubspot.create_ticket.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_ticket.authenticate_hubspot")
    def test_create_ticket_api_exception_during_creation(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """
        Test ticket creation when HubSpot API throws an exception during ticket creation.

        This test validates that the create_ticket tool properly handles
        API errors during ticket creation and provides appropriate error messages.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Simulate API error during ticket creation
        from hubspot.crm.objects.emails import ApiException

        mock_hubspot_client.crm.tickets.basic_api.create.side_effect = ApiException(
            status=400, reason="Bad Request"
        )

        # Test that the function raises an appropriate error
        with pytest.raises(ToolExecutionError) as exc_info:
            create_ticket_func(
                cus_cid="12345",
                issue="I need help with my order",
                access_token="test_token",
            )

        # Verify the error message contains expected content
        assert HubspotExceptionPrompt.TICKET_CREATION_ERROR_PROMPT in str(
            exc_info.value
        ), "Error message should contain ticket creation error prompt"

    @patch("arklex.env.tools.hubspot.create_ticket.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_ticket.authenticate_hubspot")
    def test_create_ticket_api_exception_during_association(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """
        Test ticket creation when HubSpot API throws an exception during association.

        This test validates that the create_ticket tool properly handles
        API errors during contact-ticket association and provides appropriate error messages.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful ticket creation
        mock_ticket_response = MagicMock()
        mock_ticket_response.to_dict.return_value = {"id": "ticket_123"}
        mock_hubspot_client.crm.tickets.basic_api.create.return_value = (
            mock_ticket_response
        )

        # Simulate API error during association
        from hubspot.crm.objects.emails import ApiException

        mock_hubspot_client.crm.associations.v4.basic_api.create.side_effect = (
            ApiException(status=400, reason="Association Failed")
        )

        # Test that the function raises an appropriate error
        with pytest.raises(ToolExecutionError) as exc_info:
            create_ticket_func(
                cus_cid="12345",
                issue="I need help with my order",
                access_token="test_token",
            )

        # Verify the error message contains expected content
        assert HubspotExceptionPrompt.TICKET_CREATION_ERROR_PROMPT in str(
            exc_info.value
        ), "Error message should contain ticket creation error prompt"

    @patch("arklex.env.tools.hubspot.create_ticket.authenticate_hubspot")
    def test_create_ticket_authentication_error(self, mock_authenticate: Mock) -> None:
        """
        Test ticket creation with invalid or missing access token.

        This test validates that the create_ticket tool properly handles
        authentication failures and provides appropriate error messages.
        """
        # Mock authentication to raise an authentication error
        mock_authenticate.side_effect = AuthenticationError("Invalid access token")

        # Test that the function raises an authentication error
        with pytest.raises(AuthenticationError):
            create_ticket_func(
                cus_cid="12345",
                issue="I need help with my order",
                access_token="invalid_token",
            )

    @patch("arklex.env.tools.hubspot.create_ticket.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_ticket.authenticate_hubspot")
    def test_create_ticket_with_long_issue_description(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """
        Test ticket creation with a very long issue description.

        This test validates that the create_ticket tool can handle
        lengthy issue descriptions without truncation or errors.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful ticket creation
        mock_ticket_response = MagicMock()
        mock_ticket_response.to_dict.return_value = {"id": "ticket_456"}
        mock_hubspot_client.crm.tickets.basic_api.create.return_value = (
            mock_ticket_response
        )
        mock_hubspot_client.crm.associations.v4.basic_api.create.return_value = None

        # Create a very long issue description
        long_issue = (
            "I have been experiencing problems with my order for the past three weeks. "
            * 10
        )

        result = create_ticket_func(
            cus_cid="12345",
            issue=long_issue,
            access_token="test_token",
        )

        assert result == "ticket_456", "Should return the created ticket ID"

    @patch("arklex.env.tools.hubspot.create_ticket.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_ticket.authenticate_hubspot")
    def test_create_ticket_with_special_characters_in_issue(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """
        Test ticket creation with special characters in issue description.

        This test validates that the create_ticket tool can handle
        special characters, emojis, and unicode in issue descriptions.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful ticket creation
        mock_ticket_response = MagicMock()
        mock_ticket_response.to_dict.return_value = {"id": "ticket_789"}
        mock_hubspot_client.crm.tickets.basic_api.create.return_value = (
            mock_ticket_response
        )
        mock_hubspot_client.crm.associations.v4.basic_api.create.return_value = None

        # Issue with special characters, emojis, and unicode
        special_issue = "I need help with my order! 🚀 The product has issues: émojis & symbols @#$%^&*()"

        result = create_ticket_func(
            cus_cid="12345",
            issue=special_issue,
            access_token="test_token",
        )

        assert result == "ticket_789", "Should return the created ticket ID"

    @patch("arklex.env.tools.hubspot.create_ticket.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_ticket.authenticate_hubspot")
    def test_create_ticket_with_numeric_customer_id(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """
        Test ticket creation with numeric customer ID as specified in documentation.

        This test validates that the create_ticket tool works correctly
        with numeric customer IDs as described in the slots documentation.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful ticket creation
        mock_ticket_response = MagicMock()
        mock_ticket_response.to_dict.return_value = {"id": "ticket_999"}
        mock_hubspot_client.crm.tickets.basic_api.create.return_value = (
            mock_ticket_response
        )
        mock_hubspot_client.crm.associations.v4.basic_api.create.return_value = None

        # Use numeric customer ID as specified in documentation
        numeric_customer_id = "97530152525"

        result = create_ticket_func(
            cus_cid=numeric_customer_id,
            issue="I need help with my order",
            access_token="test_token",
        )

        assert result == "ticket_999", "Should return the created ticket ID"

    @patch("arklex.env.tools.hubspot.create_ticket.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_ticket.authenticate_hubspot")
    def test_create_ticket_timestamp_format_validation(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """
        Test that ticket creation generates proper timestamp format in subject.

        This test validates that the timestamp in the subject line
        follows the expected ISO 8601 format with microsecond precision.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful ticket creation
        mock_ticket_response = MagicMock()
        mock_ticket_response.to_dict.return_value = {"id": "ticket_timestamp"}
        mock_hubspot_client.crm.tickets.basic_api.create.return_value = (
            mock_ticket_response
        )
        mock_hubspot_client.crm.associations.v4.basic_api.create.return_value = None

        create_ticket_func(
            cus_cid="12345",
            issue="I need help with my order",
            access_token="test_token",
        )

        # Verify that the ticket creation was called with proper subject format
        create_call = mock_hubspot_client.crm.tickets.basic_api.create.call_args
        ticket_input = create_call[1]["simple_public_object_input_for_create"]
        properties = ticket_input.properties

        # Check that subject contains customer ID and timestamp
        assert "subject" in properties, "Subject should be set in ticket properties"
        subject = properties["subject"]
        assert "Issue of 12345 at" in subject, (
            "Subject should contain customer ID and timestamp"
        )
        assert "T" in subject and "Z" in subject, (
            "Subject should contain ISO 8601 timestamp"
        )


class TestHubSpotFindOwnerIdByContactId:
    """
    Integration tests for find_owner_id_by_contact_id tool.

    This test class validates the owner ID lookup functionality, including
    successful lookups, API errors, and edge cases.
    """

    @pytest.fixture
    def mock_hubspot_client(self) -> MagicMock:
        """
        Create a mock HubSpot client with owner ID lookup responses.

        Returns:
            MagicMock: A mock HubSpot client configured for owner ID testing.

        This fixture provides a mock HubSpot client that simulates owner ID
        lookup operations with realistic responses.
        """
        mock_client = MagicMock()

        # Mock successful owner ID retrieval
        # This simulates a successful API response with owner information
        mock_response = MagicMock()
        mock_response.json.return_value = {"properties": {"hubspot_owner_id": "67890"}}
        mock_client.api_request.return_value = mock_response

        return mock_client

    @patch("arklex.env.tools.hubspot.find_owner_id_by_contact_id.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.find_owner_id_by_contact_id.authenticate_hubspot")
    def test_find_owner_id_by_contact_id_success(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """
        Test successful owner ID retrieval for a contact.

        This test validates that the find_owner_id_by_contact_id tool can
        successfully retrieve the owner ID associated with a specific contact
        from the HubSpot CRM.
        """
        # Mock authentication to return a valid token
        # This simulates successful HubSpot authentication
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Execute the find_owner_id_by_contact_id function with test parameters
        # This tests the complete workflow from contact ID to owner ID
        result = find_owner_id_by_contact_id_func(
            cus_cid="12345", access_token="test_token"
        )

        # Verify the owner ID is returned correctly
        # This ensures the owner lookup worked correctly
        assert result == "67890", "Should return the correct owner ID"
        mock_hubspot_client.api_request.assert_called_once()

    @patch("arklex.env.tools.hubspot.find_owner_id_by_contact_id.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.find_owner_id_by_contact_id.authenticate_hubspot")
    def test_find_owner_id_by_contact_id_api_exception(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """
        Test owner ID retrieval when HubSpot API throws an exception.

        This test validates that the find_owner_id_by_contact_id tool properly
        handles API errors and exceptions, ensuring that failures are caught
        and appropriate error messages are provided.
        """
        # Mock authentication to return a valid token
        # This simulates successful HubSpot authentication
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Simulate API error by making api_request method raise an exception
        # This simulates a network error, rate limiting, or other API issue
        from hubspot.crm.objects.emails import ApiException

        mock_hubspot_client.api_request.side_effect = ApiException("API Error")

        # Test that the function raises an appropriate error
        # This validates the error handling for API exceptions
        with pytest.raises(ToolExecutionError) as exc_info:
            find_owner_id_by_contact_id_func(cus_cid="12345", access_token="test_token")

        # Verify the error message contains expected content
        # This ensures the error message is informative
        assert "Tool find_owner_id_by_contact_id execution failed" in str(
            exc_info.value
        ), "Error message should indicate tool execution failure"


class TestHubSpotCheckAvailable:
    """Integration tests for check_available tool."""

    @pytest.fixture
    def mock_hubspot_client(self) -> MagicMock:
        """Create a mock HubSpot client with meeting availability responses."""
        mock_client = MagicMock()

        # Mock meeting links response
        mock_links_response = MagicMock()
        mock_links_response.json.return_value = {
            "status": "success",
            "results": [{"organizerUserId": "67890", "slug": "veronica-chen"}],
        }
        mock_client.api_request.return_value = mock_links_response

        return mock_client

    @patch("arklex.env.tools.hubspot.check_available.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.check_available.authenticate_hubspot")
    def test_check_available_success(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test successful availability check with meeting links and time slots."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock availability response with 15-minute slots
        mock_availability_response = MagicMock()
        mock_availability_response.json.return_value = {
            "linkAvailability": {
                "linkAvailabilityByDuration": {
                    "900000": {  # 15 minutes in milliseconds
                        "availabilities": [
                            {
                                "startMillisUtc": 1640995200000,  # Example timestamp
                                "endMillisUtc": 1640996100000,
                            }
                        ]
                    }
                }
            }
        }

        # Configure mock to return different responses for different calls
        mock_hubspot_client.api_request.side_effect = [
            mock_hubspot_client.api_request.return_value,  # First call (meeting links)
            mock_availability_response,  # Second call (availability)
        ]

        result = check_available_func(
            owner_id=67890,
            time_zone="America/New_York",
            meeting_date="tomorrow",
            duration=15,
            access_token="test_token",
        )

        assert "veronica-chen" in result
        assert "available times for other dates" in result
        assert mock_hubspot_client.api_request.call_count == 2

    @patch("arklex.env.tools.hubspot.check_available.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.check_available.authenticate_hubspot")
    def test_check_available_no_meeting_links(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test availability check when no meeting links are found for the owner."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock empty meeting links response
        mock_links_response = MagicMock()
        mock_links_response.json.return_value = {"status": "success", "results": []}
        mock_hubspot_client.api_request.return_value = mock_links_response

        with pytest.raises(
            ToolExecutionError, match=HubspotExceptionPrompt.MEETING_LINK_UNFOUND_PROMPT
        ):
            check_available_func(
                owner_id=67890,
                time_zone="America/New_York",
                meeting_date="tomorrow",
                duration=15,
                access_token="test_token",
            )

    @patch("arklex.env.tools.hubspot.check_available.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.check_available.authenticate_hubspot")
    def test_check_available_api_error(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test availability check when HubSpot API returns an error response."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock error response from API
        mock_links_response = MagicMock()
        mock_links_response.json.return_value = {
            "status": "error",
            "message": "API error",
        }
        mock_hubspot_client.api_request.return_value = mock_links_response

        with pytest.raises(
            ToolExecutionError, match=HubspotExceptionPrompt.MEETING_LINK_UNFOUND_PROMPT
        ):
            check_available_func(
                owner_id=67890,
                time_zone="America/New_York",
                meeting_date="tomorrow",
                duration=15,
                access_token="test_token",
            )


class TestHubSpotCreateMeeting:
    """Integration tests for create_meeting tool."""

    @pytest.fixture
    def mock_hubspot_client(self) -> MagicMock:
        """Create a mock HubSpot client with meeting creation responses."""
        mock_client = MagicMock()

        # Mock successful meeting creation
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
            "endTime": "2024-01-15T10:15:00Z",
            "subject": "Meeting with John Doe",
        }
        mock_client.api_request.return_value = mock_meeting_response

        return mock_client

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_success(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test successful meeting creation with contact details."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        result = create_meeting_func(
            cus_fname="John",
            cus_lname="Doe",
            cus_email="john.doe@example.com",
            meeting_date="tomorrow",
            meeting_start_time="10:00 AM",
            duration=15,
            slug="veronica-chen",
            time_zone="America/New_York",
            access_token="test_token",
        )

        meeting_data = json.loads(result)
        assert meeting_data["id"] == "meeting_123"
        assert meeting_data["subject"] == "Meeting with John Doe"
        mock_hubspot_client.api_request.assert_called_once()

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_api_exception(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation when HubSpot API throws an exception."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Simulate API error during meeting creation
        mock_hubspot_client.api_request.side_effect = MeetingsApiException(status=400)

        with pytest.raises(ToolExecutionError) as exc_info:
            create_meeting_func(
                cus_fname="John",
                cus_lname="Doe",
                cus_email="john.doe@example.com",
                meeting_date="tomorrow",
                meeting_start_time="10:00 AM",
                duration=15,
                slug="veronica-chen",
                time_zone="America/New_York",
                access_token="test_token",
            )
        assert HubspotExceptionPrompt.MEETING_UNAVAILABLE_PROMPT in str(exc_info.value)

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_iso8601_time(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with ISO8601 formatted time string."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        result = create_meeting_func(
            cus_fname="John",
            cus_lname="Doe",
            cus_email="john.doe@example.com",
            meeting_date="tomorrow",
            meeting_start_time="2024-01-15T10:00:00Z",
            duration=30,
            slug="veronica-chen",
            time_zone="America/New_York",
            access_token="test_token",
        )

        meeting_data = json.loads(result)
        assert meeting_data["id"] == "meeting_123"
        mock_hubspot_client.api_request.assert_called_once()


class TestHubSpotToolsEdgeCases:
    """Edge case tests for HubSpot tools covering boundary conditions and error scenarios."""

    @patch("arklex.env.tools.hubspot.find_contact_by_email.authenticate_hubspot")
    def test_find_contact_by_email_missing_access_token(
        self, mock_authenticate: Mock
    ) -> None:
        """Test contact search with empty access token."""
        mock_authenticate.side_effect = AuthenticationError("Missing access token")

        with pytest.raises(AuthenticationError):
            find_contact_by_email_func(
                email="test@example.com", chat="Hello", access_token=""
            )

    @patch("arklex.env.tools.hubspot.create_ticket.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_ticket.authenticate_hubspot")
    def test_create_ticket_with_empty_issue(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """Test ticket creation with empty issue description."""
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful ticket creation even with empty issue
        mock_ticket_response = MagicMock()
        mock_ticket_response.to_dict.return_value = {"id": "ticket_123"}
        mock_hubspot_client.crm.tickets.basic_api.create.return_value = (
            mock_ticket_response
        )
        mock_hubspot_client.crm.associations.v4.basic_api.create.return_value = None

        result = create_ticket_func(
            cus_cid="12345",
            issue="",  # Empty issue
            access_token="test_token",
        )

        assert result == "ticket_123"

    @patch("arklex.env.tools.hubspot.check_available.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.check_available.authenticate_hubspot")
    def test_check_available_with_different_durations(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """Test availability check with different meeting durations (15, 30, 60 minutes)."""
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting links response
        mock_links_response = MagicMock()
        mock_links_response.json.return_value = {
            "status": "success",
            "results": [{"organizerUserId": "67890", "slug": "veronica-chen"}],
        }

        # Mock availability response for 30 minutes
        mock_availability_response = MagicMock()
        mock_availability_response.json.return_value = {
            "linkAvailability": {
                "linkAvailabilityByDuration": {
                    "1800000": {  # 30 minutes in milliseconds
                        "availabilities": [
                            {
                                "startMillisUtc": 1640995200000,
                                "endMillisUtc": 1640997000000,
                            }
                        ]
                    }
                }
            }
        }

        mock_hubspot_client.api_request.side_effect = [
            mock_links_response,
            mock_availability_response,
        ]

        result = check_available_func(
            owner_id=67890,
            time_zone="America/Los_Angeles",
            meeting_date="next Monday",
            duration=30,
            access_token="test_token",
        )

        assert "veronica-chen" in result
        assert "available times for other dates" in result

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_different_timezones(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """Test meeting creation with various timezone configurations."""
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
            "endTime": "2024-01-15T10:30:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        # Test with different timezones
        timezones = [
            "America/New_York",
            "America/Los_Angeles",
            "Asia/Tokyo",
            "Europe/London",
        ]

        for timezone in timezones:
            result = create_meeting_func(
                cus_fname="John",
                cus_lname="Doe",
                cus_email="john.doe@example.com",
                meeting_date="tomorrow",
                meeting_start_time="10:00 AM",
                duration=30,
                slug="veronica-chen",
                time_zone=timezone,
                access_token="test_token",
            )

            meeting_data = json.loads(result)
            assert meeting_data["id"] == "meeting_123"
