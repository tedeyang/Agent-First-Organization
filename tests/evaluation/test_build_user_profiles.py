"""Tests for the build_user_profiles module.

This module contains comprehensive test cases for user profile building functionality,
including profile generation, attribute conversion, and custom profile handling.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from arklex.evaluation.build_user_profiles import (
    augment_attributes,
    build_profile,
    convert_attributes_to_profile,
    get_custom_profiles,
    select_system_attributes,
)


@pytest.fixture
def mock_chatgpt_chatbot():
    """Mock chatgpt_chatbot function."""
    with patch("arklex.evaluation.build_user_profiles.chatgpt_chatbot") as mock:
        mock.return_value = "mocked_response"
        yield mock


@pytest.fixture
def mock_requests_get():
    """Mock requests.get function."""
    with patch("arklex.evaluation.build_user_profiles.requests.get") as mock:
        mock_response = Mock()
        mock_response.json.return_value = ["api_value1", "api_value2"]
        mock.return_value = mock_response
        yield mock


@pytest.fixture
def mock_environment():
    """Mock Environment class."""
    with patch("arklex.evaluation.build_user_profiles.Environment") as mock:
        mock_instance = Mock()
        mock_tool = Mock()
        mock_tool.slots = [Mock(name="slot1", value="value1")]
        mock_tool.description = "Test tool description"
        mock_tool.output = "Test output"
        mock_tool.name = "Test Tool"
        mock_instance.tools = {"tool1": {"execute": Mock(return_value=mock_tool)}}
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_slot_filler():
    """Mock SlotFiller class."""
    with patch("arklex.evaluation.build_user_profiles.SlotFiller") as mock:
        mock_instance = Mock()
        mock_instance.execute.return_value = [Mock(name="slot1", value="filled_value")]
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_find_matched_attribute():
    """Mock find_matched_attribute function."""
    with patch("arklex.evaluation.build_user_profiles.find_matched_attribute") as mock:
        mock.return_value = {"matched": "data"}
        yield mock


@pytest.fixture
def mock_load_docs():
    """Mock load_docs function."""
    with patch("arklex.evaluation.build_user_profiles.load_docs") as mock:
        mock.return_value = [{"content": "test content"}]
        yield mock


@pytest.fixture
def mock_filter_attributes():
    """Mock filter_attributes function."""
    with patch("arklex.evaluation.build_user_profiles.filter_attributes") as mock:
        mock.return_value = {
            "attr1": {"values": ["val1"]},
            "attr2": {"values": ["val3"]},
        }
        yield mock


@pytest.fixture
def mock_augment_attributes():
    """Mock augment_attributes function."""
    with patch("arklex.evaluation.build_user_profiles.augment_attributes") as mock:
        mock.return_value = {
            "attr1": ["val1", "val2"],
            "attr2": ["val3", "val4"],
        }
        yield mock


@pytest.fixture
def mock_pick_attributes():
    """Mock pick_attributes function."""
    with patch("arklex.evaluation.build_user_profiles.pick_attributes") as mock:
        mock.return_value = (
            {"goal": "test_goal", "attr1": "val1", "attr2": "val3"},
            {"matched": "data"},
        )
        yield mock


@pytest.fixture
def mock_adapt_goal():
    """Mock adapt_goal function."""
    with patch("arklex.evaluation.build_user_profiles.adapt_goal") as mock:
        mock.return_value = "adapted_goal"
        yield mock


@pytest.fixture
def mock_select_system_attributes():
    """Mock select_system_attributes function."""
    with patch(
        "arklex.evaluation.build_user_profiles.select_system_attributes"
    ) as mock:
        mock.return_value = [{"sys_attr": "val"}, {"sys_attr": "val2"}]
        yield mock


@pytest.fixture
def mock_get_custom_profiles():
    """Mock get_custom_profiles function."""
    with patch("arklex.evaluation.build_user_profiles.get_custom_profiles") as mock:
        mock.return_value = (
            {"profile1": ["prof1", "prof2"], "profile2": ["prof3", "prof4"]},
            {"attr1": ["val1", "val2"], "attr2": ["val3", "val4"]},
        )
        yield mock


class TestBuildUserProfiles:
    """Test cases for build_user_profiles module.

    This class contains comprehensive tests for user profile building,
    including predefined and custom profile generation, attribute handling,
    and profile conversion functionality.
    """

    @pytest.fixture
    def mock_config(self) -> Dict[str, Any]:
        """Create a mock configuration for testing."""
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
                "user_profiles": {"profile1": {"values": ["p1"]}},
            },
            "tools": [{"id": "tool1", "name": "Test Tool"}],
            "workers": [{"id": "worker1", "name": "Test Worker"}],
        }

    @pytest.fixture
    def mock_synthetic_params(self) -> Dict[str, int]:
        """Create mock synthetic data parameters."""
        return {"num_convos": 2, "num_goals": 3}

    @pytest.fixture
    def basic_mock_setup(
        self,
        mock_load_docs,
        mock_filter_attributes,
        mock_augment_attributes,
        mock_pick_attributes,
        mock_adapt_goal,
        mock_select_system_attributes,
        mock_chatgpt_chatbot,
    ):
        """Create basic mock setup for build_profile tests."""
        return {
            "load_docs": mock_load_docs,
            "filter_attributes": mock_filter_attributes,
            "augment_attributes": mock_augment_attributes,
            "pick_attributes": mock_pick_attributes,
            "adapt_goal": mock_adapt_goal,
            "select_system_attributes": mock_select_system_attributes,
            "chatgpt_chatbot": mock_chatgpt_chatbot,
        }

    @pytest.fixture
    def custom_profile_mock_setup(
        self,
        mock_load_docs,
        mock_filter_attributes,
        mock_augment_attributes,
        mock_get_custom_profiles,
        mock_pick_attributes,
        mock_adapt_goal,
        mock_chatgpt_chatbot,
    ):
        """Create mock setup for custom profile tests."""
        return {
            "load_docs": mock_load_docs,
            "filter_attributes": mock_filter_attributes,
            "augment_attributes": mock_augment_attributes,
            "get_custom_profiles": mock_get_custom_profiles,
            "pick_attributes": mock_pick_attributes,
            "adapt_goal": mock_adapt_goal,
            "chatgpt_chatbot": mock_chatgpt_chatbot,
        }

    @pytest.mark.parametrize(
        "test_config,test_name",
        [
            ({"system_inputs": False}, "system_inputs_false"),
            (
                {
                    "custom_profile": True,
                    "user_attributes": {
                        "goal": {"values": ["goal1", "goal2"]},
                        "user_profiles": {"profile1": {"values": ["prof1", "prof2"]}},
                        "system_attributes": {"attr1": {"values": ["val1", "val2"]}},
                    },
                },
                "custom_with_binding",
            ),
            (
                {
                    "custom_profile": True,
                    "user_attributes": {
                        "goal": {"values": ["goal1", "goal2"]},
                        "user_profiles": {"profile1": {"values": ["prof1", "prof2"]}},
                        "system_attributes": {"attr1": {"values": ["val1", "val2"]}},
                    },
                },
                "custom_without_documents",
            ),
        ],
    )
    def test_build_profile_variations(
        self,
        basic_mock_setup,
        custom_profile_mock_setup,
        mock_config: Dict[str, Any],
        mock_synthetic_params: Dict[str, int],
        test_config: Dict[str, Any],
        test_name: str,
    ) -> None:
        """Test build_profile with various configurations."""
        # Update config with test-specific values
        mock_config.update(test_config)

        try:
            profiles, goals, attributes, system_attrs, labels = build_profile(
                mock_synthetic_params, mock_config
            )
            assert len(profiles) == 2
            assert len(goals) == 2
            assert len(attributes) == 2
            assert len(system_attrs) == 2
            assert len(labels) == 2
        except Exception as e:
            assert False, f"Function raised an exception: {e}"

    def test_convert_attributes_to_profile(
        self, mock_chatgpt_chatbot: Mock, mock_config: Dict[str, Any]
    ) -> None:
        """Test convert_attributes_to_profile function."""
        attributes = {"attr1": "value1", "attr2": "value2"}
        result = convert_attributes_to_profile(attributes, mock_config)
        assert result == "mocked_response"
        mock_chatgpt_chatbot.assert_called_once()

    def test_get_custom_profiles(self, mock_config: Dict[str, Any]) -> None:
        """Test get_custom_profiles function."""
        # Update mock_config to include the required structure for API-based get_custom_profiles
        mock_config["user_attributes"]["system_attributes"] = {
            "attr1": {"api": "http://test.com/api1"}
        }
        mock_config["user_attributes"]["user_profiles"] = {
            "profile1": {"api": "http://test.com/api2"}
        }

        with patch(
            "arklex.evaluation.build_user_profiles.requests.get"
        ) as mock_requests_get:
            # Mock API responses
            mock_response1 = Mock()
            mock_response1.json.return_value = ["api_value1", "api_value2"]
            mock_response2 = Mock()
            mock_response2.json.return_value = ["profile_value1", "profile_value2"]
            mock_requests_get.side_effect = [mock_response1, mock_response2]

            user_profiles, system_attributes = get_custom_profiles(mock_config)
            assert isinstance(user_profiles, dict)
            assert isinstance(system_attributes, dict)
            assert "profile1" in user_profiles
            assert "attr1" in system_attributes

    def test_get_custom_profiles_without_required_structure(self) -> None:
        """Test get_custom_profiles function when required structure is not present."""
        config = {
            "user_attributes": {
                # Missing system_attributes and user_profiles
            }
        }

        user_profiles, system_attributes = get_custom_profiles(config)
        assert isinstance(user_profiles, dict)
        assert isinstance(system_attributes, dict)
        assert len(user_profiles) == 0
        assert len(system_attributes) == 0

    def test_build_profile_with_system_inputs_false_and_custom_profile(
        self, mock_config: Dict[str, Any], mock_synthetic_params: Dict[str, int]
    ) -> None:
        """Test build_profile function when system_inputs is False and custom_profile is True."""
        config = mock_config.copy()
        config["custom_profile"] = True
        config["system_inputs"] = False
        config["user_attributes"] = {
            "goal": {"values": ["goal1", "goal2"]},
            "user_profiles": {
                "profile1": {"values": ["prof1", "prof2"]},
            },
            "system_attributes": {
                "attr1": {"values": ["val1", "val2"]},
            },
        }

        with patch("arklex.evaluation.build_user_profiles.load_docs") as mock_load_docs:
            mock_load_docs.return_value = [{"content": "test content"}]

            with patch(
                "arklex.evaluation.build_user_profiles.filter_attributes"
            ) as mock_filter_attributes:
                mock_filter_attributes.return_value = {
                    "attr1": {"values": ["val1"], "generate_values": False},
                }

                with patch(
                    "arklex.evaluation.build_user_profiles.augment_attributes"
                ) as mock_augment_attributes:
                    mock_augment_attributes.return_value = {
                        "attr1": ["val1"],
                    }

                    with patch(
                        "arklex.evaluation.build_user_profiles.get_custom_profiles"
                    ) as mock_get_custom_profiles:
                        mock_get_custom_profiles.return_value = (
                            {"profile1": ["prof1", "prof2"]},
                            {"attr1": ["val1", "val2"]},
                        )

                        with patch(
                            "arklex.evaluation.build_user_profiles.pick_attributes"
                        ) as mock_pick_attributes:
                            mock_pick_attributes.return_value = (
                                {"goal": "test_goal", "attr1": "val1"},
                                {"matched": "data"},
                            )

                            with patch(
                                "arklex.evaluation.build_user_profiles.adapt_goal"
                            ) as mock_adapt_goal:
                                mock_adapt_goal.return_value = "adapted_goal"

                                with patch(
                                    "arklex.evaluation.build_user_profiles.convert_attributes_to_profiles"
                                ) as mock_convert:
                                    mock_convert.return_value = (
                                        ["profile1", "profile1"],
                                        ["goal1", "goal1"],
                                        [{"sys": "input"}, {"sys": "input"}],
                                    )

                                    (
                                        profiles,
                                        goals,
                                        attributes_list,
                                        system_inputs,
                                        labels_list,
                                    ) = build_profile(mock_synthetic_params, config)

                                    # num_convos is 2, so we expect 2 items
                                    assert len(profiles) == 2
                                    assert len(goals) == 2
                                    assert len(attributes_list) == 2
                                    assert len(system_inputs) == 2
                                    assert len(labels_list) == 2

    def test_build_profile_with_binding_index_mismatch(
        self, mock_config: Dict[str, Any], mock_synthetic_params: Dict[str, int]
    ) -> None:
        """Test build_profile function when binding_index doesn't contain the expected key."""
        config = mock_config.copy()
        config["custom_profile"] = True
        config["user_attributes"] = {
            "goal": {"values": ["goal1", "goal2"]},
            "user_profiles": {
                "profile1": {"values": ["prof1", "prof2"], "bind_to": "system_attr1"},
            },
            "system_attributes": {
                "attr1": {"values": ["val1", "val2"]},  # No bind_to specified
            },
        }

        with patch("arklex.evaluation.build_user_profiles.load_docs") as mock_load_docs:
            mock_load_docs.return_value = [{"content": "test content"}]

            with patch(
                "arklex.evaluation.build_user_profiles.filter_attributes"
            ) as mock_filter_attributes:
                mock_filter_attributes.return_value = {
                    "attr1": {"values": ["val1"], "generate_values": False},
                }

                with patch(
                    "arklex.evaluation.build_user_profiles.augment_attributes"
                ) as mock_augment_attributes:
                    mock_augment_attributes.return_value = {
                        "attr1": ["val1"],
                    }

                    with patch(
                        "arklex.evaluation.build_user_profiles.get_custom_profiles"
                    ) as mock_get_custom_profiles:
                        mock_get_custom_profiles.return_value = (
                            {"profile1": ["prof1", "prof2"]},
                            {"attr1": ["val1", "val2"]},
                        )

                        with patch(
                            "arklex.evaluation.build_user_profiles.pick_attributes"
                        ) as mock_pick_attributes:
                            mock_pick_attributes.return_value = (
                                {"goal": "test_goal", "attr1": "val1"},
                                {"matched": "data"},
                            )

                            with patch(
                                "arklex.evaluation.build_user_profiles.adapt_goal"
                            ) as mock_adapt_goal:
                                mock_adapt_goal.return_value = "adapted_goal"

                                with patch(
                                    "arklex.evaluation.build_user_profiles.convert_attributes_to_profiles"
                                ) as mock_convert:
                                    mock_convert.return_value = (
                                        ["profile1", "profile1"],
                                        ["goal1", "goal1"],
                                        [{"sys": "input"}, {"sys": "input"}],
                                    )

                                    (
                                        profiles,
                                        goals,
                                        attributes_list,
                                        system_inputs,
                                        labels_list,
                                    ) = build_profile(mock_synthetic_params, config)

                                    # num_convos is 2, so we expect 2 items
                                    assert len(profiles) == 2
                                    assert len(goals) == 2
                                    assert len(attributes_list) == 2
                                    assert len(system_inputs) == 2
                                    assert len(labels_list) == 2

    def test_build_profile_with_commented_get_label(
        self, mock_config: Dict[str, Any], mock_synthetic_params: Dict[str, int]
    ) -> None:
        """Test build_profile function with the commented get_label section."""
        config = mock_config.copy()
        config["custom_profile"] = True
        config["user_attributes"] = {
            "goal": {"values": ["goal1", "goal2"]},
            "user_profiles": {
                "profile1": {"values": ["prof1", "prof2"]},
            },
            "system_attributes": {
                "attr1": {"values": ["val1", "val2"]},
            },
        }

        with patch("arklex.evaluation.build_user_profiles.load_docs") as mock_load_docs:
            mock_load_docs.return_value = [{"content": "test content"}]

            with patch(
                "arklex.evaluation.build_user_profiles.filter_attributes"
            ) as mock_filter_attributes:
                mock_filter_attributes.return_value = {
                    "attr1": {"values": ["val1"], "generate_values": False},
                }

                with patch(
                    "arklex.evaluation.build_user_profiles.augment_attributes"
                ) as mock_augment_attributes:
                    mock_augment_attributes.return_value = {
                        "attr1": ["val1"],
                    }

                    with patch(
                        "arklex.evaluation.build_user_profiles.get_custom_profiles"
                    ) as mock_get_custom_profiles:
                        mock_get_custom_profiles.return_value = (
                            {"profile1": ["prof1", "prof2"]},
                            {"attr1": ["val1", "val2"]},
                        )

                        with patch(
                            "arklex.evaluation.build_user_profiles.pick_attributes"
                        ) as mock_pick_attributes:
                            mock_pick_attributes.return_value = (
                                {"goal": "test_goal", "attr1": "val1"},
                                {"matched": "data"},
                            )

                            with patch(
                                "arklex.evaluation.build_user_profiles.adapt_goal"
                            ) as mock_adapt_goal:
                                mock_adapt_goal.return_value = "adapted_goal"

                                with patch(
                                    "arklex.evaluation.build_user_profiles.convert_attributes_to_profiles"
                                ) as mock_convert:
                                    mock_convert.return_value = (
                                        ["profile1", "profile1"],
                                        ["goal1", "goal1"],
                                        [{"sys": "input"}, {"sys": "input"}],
                                    )

                                    (
                                        profiles,
                                        goals,
                                        attributes_list,
                                        system_inputs,
                                        labels_list,
                                    ) = build_profile(mock_synthetic_params, config)

                                    # num_convos is 2, so we expect 2 items
                                    assert len(profiles) == 2
                                    assert len(goals) == 2
                                    assert len(attributes_list) == 2
                                    assert len(system_inputs) == 2
                                    assert len(labels_list) == 2

    def test_build_profile_with_empty_documents_in_custom_mode(
        self, mock_config: Dict[str, Any], mock_synthetic_params: Dict[str, int]
    ) -> None:
        """Test build_profile function with empty documents in custom profile mode."""
        config = mock_config.copy()
        config["custom_profile"] = True
        config["user_attributes"] = {
            "goal": {"values": ["goal1", "goal2"]},
            "user_profiles": {
                "profile1": {"values": ["prof1", "prof2"]},
            },
            "system_attributes": {
                "attr1": {"values": ["val1", "val2"]},
            },
        }

        with patch("arklex.evaluation.build_user_profiles.load_docs") as mock_load_docs:
            mock_load_docs.return_value = []  # Empty documents

            with patch(
                "arklex.evaluation.build_user_profiles.filter_attributes"
            ) as mock_filter_attributes:
                mock_filter_attributes.return_value = {
                    "attr1": {"values": ["val1"], "generate_values": False},
                }

                with patch(
                    "arklex.evaluation.build_user_profiles.augment_attributes"
                ) as mock_augment_attributes:
                    mock_augment_attributes.return_value = {
                        "attr1": ["val1"],
                    }

                    with patch(
                        "arklex.evaluation.build_user_profiles.get_custom_profiles"
                    ) as mock_get_custom_profiles:
                        mock_get_custom_profiles.return_value = (
                            {"profile1": ["prof1", "prof2"]},
                            {"attr1": ["val1", "val2"]},
                        )

                        with patch(
                            "arklex.evaluation.build_user_profiles.pick_attributes"
                        ) as mock_pick_attributes:
                            mock_pick_attributes.return_value = (
                                {"goal": "test_goal", "attr1": "val1"},
                                {"matched": "data"},
                            )

                            with patch(
                                "arklex.evaluation.build_user_profiles.adapt_goal"
                            ) as mock_adapt_goal:
                                mock_adapt_goal.return_value = "adapted_goal"

                                with patch(
                                    "arklex.evaluation.build_user_profiles.convert_attributes_to_profiles"
                                ) as mock_convert:
                                    mock_convert.return_value = (
                                        ["profile1", "profile1"],
                                        ["goal1", "goal1"],
                                        [{"sys": "input"}, {"sys": "input"}],
                                    )

                                    (
                                        profiles,
                                        goals,
                                        attributes_list,
                                        system_inputs,
                                        labels_list,
                                    ) = build_profile(mock_synthetic_params, config)

                                    # num_convos is 2, so we expect 2 items
                                    assert len(profiles) == 2
                                    assert len(goals) == 2
                                    assert len(attributes_list) == 2
                                    assert len(system_inputs) == 2
                                    assert len(labels_list) == 2

    def test_get_custom_profiles_with_api_and_exception(self) -> None:
        """Test get_custom_profiles function when API call raises an exception."""
        config = {
            "user_attributes": {
                "system_attributes": {
                    "attr1": {"api": "http://test.com/api1"},
                },
                "user_profiles": {
                    "profile1": {"api": "http://test.com/api2"},
                },
            }
        }

        with patch(
            "arklex.evaluation.build_user_profiles.requests.get"
        ) as mock_requests_get:
            # Mock the first API call to succeed
            mock_response1 = Mock()
            mock_response1.json.return_value = ["api_value1", "api_value2"]

            # Mock the second API call to raise an exception
            mock_requests_get.side_effect = [mock_response1, Exception("API Error")]

            # The function should raise the exception from the API call
            with pytest.raises(Exception, match="API Error"):
                get_custom_profiles(config)

    def test_get_custom_profiles_with_api_and_invalid_response(self) -> None:
        """Test get_custom_profiles function when API returns invalid response."""
        config = {
            "user_attributes": {
                "system_attributes": {
                    "attr1": {"api": "http://test.com/api1"},
                },
                "user_profiles": {
                    "profile1": {"api": "http://test.com/api2"},
                },
            }
        }

        with patch(
            "arklex.evaluation.build_user_profiles.requests.get"
        ) as mock_requests_get:
            # Mock API calls to return invalid responses
            mock_response1 = Mock()
            mock_response1.json.side_effect = ValueError("Invalid JSON")

            mock_response2 = Mock()
            mock_response2.json.return_value = ["api_value1", "api_value2"]

            mock_requests_get.side_effect = [mock_response1, mock_response2]

            # The function should raise the exception from the invalid JSON response
            with pytest.raises(ValueError, match="Invalid JSON"):
                get_custom_profiles(config)

    def test_select_system_attributes_with_non_dict_values(
        self, mock_synthetic_params: Dict[str, int]
    ) -> None:
        """Test select_system_attributes function when system attributes are not dictionaries."""
        config = {
            "user_attributes": {
                "system_attributes": {
                    "attr1": ["val1", "val2"],  # Not a dict
                }
            }
        }

        # Should raise TypeError when trying to access "values" key on a list
        with pytest.raises(
            TypeError, match="list indices must be integers or slices, not str"
        ):
            select_system_attributes(config, mock_synthetic_params)

    def test_select_system_attributes_with_empty_list(
        self, mock_synthetic_params: Dict[str, int]
    ) -> None:
        """Test select_system_attributes function when system attributes list is empty."""
        config = {
            "user_attributes": {
                "system_attributes": {
                    "attr1": [],  # Empty list
                }
            }
        }

        # Should raise TypeError when trying to access "values" key on a list
        with pytest.raises(
            TypeError, match="list indices must be integers or slices, not str"
        ):
            select_system_attributes(config, mock_synthetic_params)

    def test_augment_attributes_with_empty_values(
        self, mock_config: Dict[str, Any]
    ) -> None:
        """Test augment_attributes function when some attributes have empty values."""
        attributes = {
            "attr1": {"values": [], "augment": True},  # Empty values
            "attr2": {"values": ["val3", "val4"], "augment": True},
        }
        documents = [{"content": "test content"}]

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatgpt_chatbot:
            mock_chatgpt_chatbot.return_value = "new_val1, new_val2"

            result = augment_attributes(attributes, mock_config, documents)

            # Should skip attributes with empty values in text_attribute
            assert (
                "attr1" not in result["attr2"]
            )  # attr1 should not appear in text_attribute
            # Should still process attr2
            assert result["attr2"] == ["val3", "val4", "new_val1", "new_val2"]

    def test_augment_attributes_with_mixed_generate_values_and_documents(
        self, mock_config: Dict[str, Any]
    ) -> None:
        """Test augment_attributes function with mixed generate_values and documents."""
        attributes = {
            "attr1": {"values": ["val1"], "augment": False},
            "attr2": {"values": ["val2"], "augment": True},
        }
        documents = [{"content": "test content"}]

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatgpt_chatbot:
            mock_chatgpt_chatbot.return_value = "new_val1, new_val2"

            result = augment_attributes(attributes, mock_config, documents)

            # Should use original values for attr1 (augment=False)
            assert result["attr1"] == ["val1"]
            # Should generate new values for attr2 (augment=True)
            assert result["attr2"] == ["val2", "new_val1", "new_val2"]
            # Should call chatgpt_chatbot only once for attr2
            assert mock_chatgpt_chatbot.call_count == 1

    def test_augment_attributes_with_mixed_generate_values_and_no_documents(
        self, mock_config: Dict[str, Any]
    ) -> None:
        """Test augment_attributes function with mixed generate_values and no documents."""
        attributes = {
            "attr1": {"values": ["val1"], "augment": False},
            "attr2": {"values": ["val2"], "augment": True},
        }
        documents = []  # No documents

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatgpt_chatbot:
            mock_chatgpt_chatbot.return_value = "new_val1, new_val2"

            result = augment_attributes(attributes, mock_config, documents)

            # Should use original values for attr1 (augment=False)
            assert result["attr1"] == ["val1"]
            # Should generate new values for attr2 (augment=True) without documents
            assert result["attr2"] == ["val2", "new_val1", "new_val2"]
            # Should call chatgpt_chatbot only once for attr2
            assert mock_chatgpt_chatbot.call_count == 1
