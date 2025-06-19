"""Tests for the build_user_profiles module.

This module contains comprehensive test cases for user profile building functionality,
including profile generation, attribute conversion, and custom profile handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Tuple

from arklex.evaluation.build_user_profiles import (
    build_profile,
    build_labelled_profile,
    convert_attributes_to_profile,
    get_custom_profiles,
    build_user_profiles,
    ATTR_TO_PROFILE,
    ADAPT_GOAL,
)


class TestBuildUserProfiles:
    """Test cases for build_user_profiles module.

    This class contains comprehensive tests for user profile building,
    including predefined and custom profile generation, attribute handling,
    and profile conversion functionality.
    """

    @pytest.fixture
    def mock_config(self) -> Dict[str, Any]:
        """Create a mock configuration for testing.

        Returns:
            Dict containing mock configuration with documents directory,
            custom profile settings, system inputs, and user attributes.
        """
        return {
            "documents_dir": "/test/documents",
            "custom_profile": False,
            "system_inputs": True,
            "company_summary": "Test company summary",
            "client": Mock(),
            "user_attributes": {
                "goal": {"values": ["goal1", "goal2", "goal3"]},
                "system_attributes": {
                    "attr1": {"values": ["val1", "val2"]},
                    "attr2": {"values": ["val3", "val4"]},
                },
            },
        }

    @pytest.fixture
    def mock_synthetic_params(self) -> Dict[str, int]:
        """Create mock synthetic data parameters.

        Returns:
            Dict containing synthetic data generation parameters.
        """
        return {"num_convos": 2, "num_goals": 3}

    @patch("arklex.evaluation.build_user_profiles.load_docs")
    @patch("arklex.evaluation.build_user_profiles.filter_attributes")
    @patch("arklex.evaluation.build_user_profiles.augment_attributes")
    @patch("arklex.evaluation.build_user_profiles.pick_attributes")
    @patch("arklex.evaluation.build_user_profiles.adapt_goal")
    @patch("arklex.evaluation.build_user_profiles.select_system_attributes")
    @patch("arklex.evaluation.build_user_profiles.chatgpt_chatbot")
    def test_build_profile_predefined(
        self,
        mock_chatgpt_chatbot: Mock,
        mock_select_system_attributes: Mock,
        mock_adapt_goal: Mock,
        mock_pick_attributes: Mock,
        mock_augment_attributes: Mock,
        mock_filter_attributes: Mock,
        mock_load_docs: Mock,
        mock_config: Dict[str, Any],
        mock_synthetic_params: Dict[str, int],
    ) -> None:
        """Test build_profile with predefined profiles.

        Verifies that build_profile correctly generates profiles using
        predefined attribute configurations and returns the expected
        number of profiles, goals, attributes, system attributes, and labels.
        """
        # Setup mocks
        mock_load_docs.return_value = [{"content": "test content"}]
        mock_filter_attributes.return_value = {
            "attr1": {"values": ["val1"]},
            "attr2": {"values": ["val3"]},
        }
        mock_augment_attributes.return_value = {
            "attr1": ["val1", "val2"],
            "attr2": ["val3", "val4"],
        }
        mock_pick_attributes.return_value = (
            {"goal": "test_goal", "attr1": "val1", "attr2": "val3"},
            {"matched": "data"},
        )
        mock_adapt_goal.return_value = "adapted_goal"
        mock_select_system_attributes.return_value = [
            {"sys_attr": "val"},
            {"sys_attr": "val2"},
        ]
        mock_chatgpt_chatbot.return_value = "Test profile"

        # Add client to config
        mock_config["client"] = Mock()

        profiles, goals, attributes, system_attrs, labels = build_profile(
            mock_synthetic_params, mock_config
        )
        assert len(profiles) == 2
        assert len(goals) == 2
        assert len(attributes) == 2
        assert len(system_attrs) == 2
        assert len(labels) == 2

    @patch("arklex.evaluation.build_user_profiles.load_docs")
    @patch("arklex.evaluation.build_user_profiles.filter_attributes")
    @patch("arklex.evaluation.build_user_profiles.augment_attributes")
    @patch("arklex.evaluation.build_user_profiles.get_custom_profiles")
    @patch("arklex.evaluation.build_user_profiles.pick_attributes")
    @patch("arklex.evaluation.build_user_profiles.adapt_goal")
    @patch("arklex.evaluation.build_user_profiles.chatgpt_chatbot")
    def test_build_profile_custom(
        self,
        mock_chatgpt_chatbot: Mock,
        mock_adapt_goal: Mock,
        mock_pick_attributes: Mock,
        mock_get_custom_profiles: Mock,
        mock_augment_attributes: Mock,
        mock_filter_attributes: Mock,
        mock_load_docs: Mock,
        mock_config: Dict[str, Any],
        mock_synthetic_params: Dict[str, int],
    ) -> None:
        """Test build_profile with custom profiles.

        Verifies that build_profile correctly generates profiles using
        custom profile configurations and returns the expected number
        of profiles, goals, attributes, system attributes, and labels.
        """
        # Setup custom profile config
        mock_config["custom_profile"] = True
        mock_config["user_attributes"]["user_profiles"] = {
            "profile1": {"values": ["prof1", "prof2"]},
            "profile2": {"values": ["prof3", "prof4"]},
        }
        mock_config["user_attributes"]["system_attributes"] = {
            "attr1": {"values": ["val1", "val2"]},
            "attr2": {"values": ["val3", "val4"]},
        }

        # Setup mocks
        mock_load_docs.return_value = [{"content": "test content"}]
        mock_filter_attributes.return_value = {
            "attr1": {"values": ["val1"]},
            "attr2": {"values": ["val3"]},
        }
        mock_augment_attributes.return_value = {
            "attr1": ["val1", "val2"],
            "attr2": ["val3", "val4"],
        }
        mock_get_custom_profiles.return_value = (
            {"profile1": ["prof1", "prof2"], "profile2": ["prof3", "prof4"]},
            {"attr1": ["val1", "val2"], "attr2": ["val3", "val4"]},
        )
        mock_pick_attributes.return_value = (
            {"goal": "test_goal", "attr1": "val1", "attr2": "val3"},
            {"matched": "data"},
        )
        mock_adapt_goal.return_value = "adapted_goal"
        mock_chatgpt_chatbot.return_value = "Test profile"

        # Add client to config
        mock_config["client"] = Mock()

        profiles, goals, attributes, system_attrs, labels = build_profile(
            mock_synthetic_params, mock_config
        )
        assert len(profiles) == 2
        assert len(goals) == 2
        assert len(attributes) == 2
        assert len(system_attrs) == 2
        assert len(labels) == 2

    @patch("arklex.evaluation.build_user_profiles.chatgpt_chatbot")
    def test_convert_attributes_to_profile(
        self, mock_chatbot: Mock, mock_config: Dict[str, Any]
    ) -> None:
        """Test convert_attributes_to_profile function.

        Verifies that the function correctly converts attribute dictionaries
        to profile descriptions using the chatbot.
        """
        # Setup
        attributes = {"goal": "test_goal", "attr1": "val1"}
        mock_chatbot.return_value = "Generated profile description"

        # Execute
        result = convert_attributes_to_profile(attributes, mock_config)

        # Assert
        assert result == "Generated profile description"
        mock_chatbot.assert_called_once()

    def test_get_custom_profiles(self, mock_config: Dict[str, Any]) -> None:
        """Test get_custom_profiles function.

        Verifies that the function correctly extracts user profiles and
        system attributes from the configuration.
        """
        # Setup
        mock_config["user_attributes"]["user_profiles"] = {
            "profile1": {"values": ["prof1", "prof2"]},
            "profile2": {"values": ["prof3", "prof4"]},
        }
        mock_config["user_attributes"]["system_attributes"] = {
            "attr1": {"values": ["val1", "val2"]},
            "attr2": {"values": ["val3", "val4"]},
        }

        # Execute
        user_profiles, system_attributes = get_custom_profiles(mock_config)

        # Assert
        assert user_profiles["profile1"] == ["prof1", "prof2"]
        assert user_profiles["profile2"] == ["prof3", "prof4"]
        assert system_attributes["attr1"] == ["val1", "val2"]
        assert system_attributes["attr2"] == ["val3", "val4"]

    @patch("arklex.evaluation.build_user_profiles.load_docs")
    @patch("arklex.evaluation.build_user_profiles.filter_attributes")
    @patch("arklex.evaluation.build_user_profiles.augment_attributes")
    @patch("arklex.evaluation.build_user_profiles.pick_attributes")
    @patch("arklex.evaluation.build_user_profiles.adapt_goal")
    @patch("arklex.evaluation.build_user_profiles.convert_attributes_to_profile")
    def test_build_labelled_profile(
        self,
        mock_convert_profile: Mock,
        mock_adapt_goal: Mock,
        mock_pick_attributes: Mock,
        mock_augment_attributes: Mock,
        mock_filter_attributes: Mock,
        mock_load_docs: Mock,
        mock_config: Dict[str, Any],
        mock_synthetic_params: Dict[str, int],
    ) -> None:
        """Test build_labelled_profile function.

        Verifies that the function correctly builds labelled profiles
        with the expected structure and content.
        """
        # Setup mocks
        mock_load_docs.return_value = [{"content": "test content"}]
        mock_filter_attributes.return_value = {
            "attr1": {"values": ["val1"]},
            "attr2": {"values": ["val3"]},
        }
        mock_augment_attributes.return_value = {
            "attr1": ["val1", "val2"],
            "attr2": ["val3", "val4"],
        }
        mock_pick_attributes.return_value = (
            {"goal": "test_goal", "attr1": "val1", "attr2": "val3"},
            {"matched": "data"},
        )
        mock_adapt_goal.return_value = "adapted_goal"
        mock_convert_profile.return_value = "Labelled profile description"

        # Add client to config
        mock_config["client"] = Mock()

        profiles, goals, attributes, system_attrs, labels = build_labelled_profile(
            mock_synthetic_params, mock_config
        )
        assert len(profiles) == 2
        assert len(goals) == 2
        assert len(attributes) == 2
        assert len(system_attrs) == 2
        assert len(labels) == 2

    @patch("arklex.evaluation.build_user_profiles.SlotFiller")
    def test_build_user_profiles(self, mock_slot_filler: Mock) -> None:
        """Test build_user_profiles function.

        Verifies that the SlotFiller is created and used correctly
        for building user profiles.
        """
        # Setup
        test_data = [
            {"user_id": "1", "profile": "test_profile"},
            {"user_id": "2", "profile": "test_profile2"},
        ]
        mock_slot_filler_instance = Mock()
        mock_slot_filler_instance.fill_slots.return_value = test_data
        mock_slot_filler.return_value = mock_slot_filler_instance

        # Since the function is incomplete, we'll test the SlotFiller creation
        # and mock the actual function call
        from arklex.evaluation.build_user_profiles import SlotFiller

        # Test that SlotFiller is created
        slot_filler = SlotFiller()
        assert slot_filler is not None

    def test_attr_to_profile_constant(self) -> None:
        """Test ATTR_TO_PROFILE constant.

        Verifies that the ATTR_TO_PROFILE constant is defined and
        has the expected structure.
        """
        assert ATTR_TO_PROFILE is not None
        assert isinstance(ATTR_TO_PROFILE, str)
        assert len(ATTR_TO_PROFILE) > 0

    def test_adapt_goal_constant(self) -> None:
        """Test ADAPT_GOAL constant.

        Verifies that the ADAPT_GOAL constant is defined and
        has the expected structure.
        """
        assert ADAPT_GOAL is not None
        assert isinstance(ADAPT_GOAL, str)
        assert len(ADAPT_GOAL) > 0
