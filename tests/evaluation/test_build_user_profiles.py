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
    build_labelled_profile,
    convert_attributes_to_profile,
    filter_attributes,
    get_custom_profiles,
    ATTR_TO_PROFILE,
    ADAPT_GOAL,
    pick_attributes,
    pick_goal,
    find_matched_attribute,
    adapt_goal,
    convert_attributes_to_profiles,
    attributes_to_text,
    get_label,
    select_system_attributes,
    build_user_profiles,
    pick_attributes_random,
    pick_attributes_react,
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
        user_profiles, system_attributes = get_custom_profiles(mock_config)
        assert isinstance(user_profiles, dict)
        assert isinstance(system_attributes, dict)

    def test_build_labelled_profile(
        self,
        basic_mock_setup,
        mock_config: Dict[str, Any],
        mock_synthetic_params: Dict[str, int],
    ) -> None:
        """Test build_labelled_profile function."""
        profiles, goals, attributes, system_attrs, labels = build_labelled_profile(
            mock_synthetic_params, mock_config
        )
        assert len(profiles) == 2
        assert len(goals) == 2
        assert len(attributes) == 2
        assert len(system_attrs) == 2
        assert len(labels) == 2

    def test_build_user_profiles(self, mock_slot_filler: Mock) -> None:
        """Test build_user_profiles function."""
        test_data = [{"goal": "test goal", "attributes": {"attr1": "value1"}}]
        result = build_user_profiles(test_data)
        assert result is None  # Function is incomplete

    def test_build_user_profiles_complete(self, mock_slot_filler: Mock) -> None:
        """Test build_user_profiles function with complete implementation."""
        test_data = [
            {
                "goal": "test goal 1",
                "attributes": {"attr1": "value1", "attr2": "value2"},
            },
            {
                "goal": "test goal 2",
                "attributes": {"attr1": "value3", "attr2": "value4"},
            },
        ]

        # The function is incomplete (ends with comment), so it returns None
        result = build_user_profiles(test_data)
        assert result is None  # Function is incomplete, returns None

    def test_attr_to_profile_constant(self) -> None:
        """Test ATTR_TO_PROFILE constant."""
        assert "company_summary" in ATTR_TO_PROFILE
        assert "user_attr" in ATTR_TO_PROFILE

    def test_adapt_goal_constant(self) -> None:
        """Test ADAPT_GOAL constant."""
        assert "goal" in ADAPT_GOAL
        assert "company_summary" in ADAPT_GOAL

    @pytest.mark.parametrize(
        "strategy,expected_goal",
        [
            ("react", "goal1"),
            ("llm_based", "goal2"),
        ],
    )
    def test_pick_goal(
        self, mock_chatgpt_chatbot: Mock, strategy: str, expected_goal: str
    ) -> None:
        """Test pick_goal function with different strategies."""
        attributes = {"attr1": "value1"}
        goals = ["goal1", "goal2"]

        if strategy == "llm_based":
            mock_chatgpt_chatbot.return_value = f"Goal: {expected_goal}"
        else:  # react strategy
            mock_chatgpt_chatbot.return_value = f"Thought: test\nGoal: {expected_goal}"

        result = pick_goal(attributes, goals, strategy)
        assert result == expected_goal

    def test_pick_goal_invalid_strategy(self) -> None:
        """Test pick_goal with invalid strategy."""
        attributes = {"attr1": "value1"}
        goals = ["goal1", "goal2"]

        with pytest.raises(ValueError, match="Invalid strategy"):
            pick_goal(attributes, goals, "invalid_strategy")

    def test_find_matched_attribute_react_strategy(
        self, mock_chatgpt_chatbot: Mock
    ) -> None:
        """Test find_matched_attribute with react strategy."""
        goal = "test goal"
        user_profile = "test profile"

        # Mock the expected response format
        mock_chatgpt_chatbot.return_value = "Thought: test\nAttribute: test_attr"

        result = find_matched_attribute(goal, user_profile, "react")
        assert result == "test_attr"
        mock_chatgpt_chatbot.assert_called_once()

    def test_pick_attributes_react(self) -> None:
        """Test pick_attributes_react function."""
        user_profile = {"profile": "test"}
        attributes = {
            "category1": ["value1", "value2"],
            "category2": ["value3", "value4"],
            "goal": ["goal1", "goal2"],
        }
        goals = ["goal1", "goal2"]
        client = Mock()

        selected_attributes, matched_attribute_to_goal = pick_attributes_react(
            user_profile, attributes, goals, client
        )

        assert "goal" in selected_attributes
        assert "category1" in selected_attributes
        assert "category2" in selected_attributes
        assert isinstance(matched_attribute_to_goal, dict)

    def test_pick_attributes_random(self) -> None:
        """Test pick_attributes_random function."""
        user_profile = {"profile": "test"}
        attributes = {
            "category1": ["value1", "value2"],
            "category2": ["value3", "value4"],
            "goal": ["goal1", "goal2"],
        }
        goals = ["goal1", "goal2"]

        selected_attributes, matched_attribute_to_goal = pick_attributes_random(
            user_profile, attributes, goals
        )

        assert "goal" in selected_attributes
        assert "category1" in selected_attributes
        assert "category2" in selected_attributes
        assert isinstance(matched_attribute_to_goal, dict)

    def test_adapt_goal(
        self, mock_chatgpt_chatbot: Mock, mock_config: Dict[str, Any]
    ) -> None:
        """Test adapt_goal function."""
        goal = "test goal"
        doc = "test document"
        user_profile = "test profile"

        # Mock the expected response format
        mock_chatgpt_chatbot.return_value = "adapted goal"

        result = adapt_goal(goal, mock_config, doc, user_profile)
        assert result == "adapted goal"
        mock_chatgpt_chatbot.assert_called_once()

    def test_get_label_with_invalid_attribute(
        self, mock_environment: Mock, mock_config: Dict[str, Any]
    ) -> None:
        """Test get_label with invalid attribute."""
        attribute = {"goal": "test goal"}
        config = {"tools": [{"id": "tool1"}], "workers": [], "client": Mock()}

        label, valid = get_label(attribute, config)
        assert isinstance(label, list)
        assert valid is True

    def test_filter_attributes(self, mock_config: Dict[str, Any]) -> None:
        """Test filter_attributes function."""
        result = filter_attributes(mock_config)
        assert isinstance(result, dict)
        # The function returns user_profiles when system_attributes is empty
        assert "user_profiles" in result

    def test_augment_attributes(self, mock_config: Dict[str, Any]) -> None:
        """Test augment_attributes function."""
        attributes = {"category": {"values": ["value1"], "generate_values": True}}
        documents = [{"content": "test document"}]

        result = augment_attributes(attributes, mock_config, documents)
        assert isinstance(result, dict)

    def test_attributes_to_text(self) -> None:
        """Test attributes_to_text function."""
        attribute_list = [{"attr1": "value1", "attr2": "value2"}]
        result = attributes_to_text(attribute_list)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_convert_attributes_to_profiles(
        self, mock_chatgpt_chatbot: Mock, mock_config: Dict[str, Any]
    ) -> None:
        """Test convert_attributes_to_profiles function."""
        attributes_list = [
            {"attr1": "value1", "goal": "goal1"},
            {"attr2": "value2", "goal": "goal2"},
        ]
        system_attributes_list = [{"sys1": "val1"}, {"sys2": "val2"}]

        profiles, goals, system_inputs = convert_attributes_to_profiles(
            attributes_list, system_attributes_list, mock_config
        )

        assert len(profiles) == 2
        assert len(goals) == 2
        assert len(system_inputs) == 2

    def test_select_system_attributes(
        self, mock_config: Dict[str, Any], mock_synthetic_params: Dict[str, int]
    ) -> None:
        """Test select_system_attributes function."""
        result = select_system_attributes(mock_config, mock_synthetic_params)
        assert len(result) == 2
        assert all(isinstance(attr, dict) for attr in result)

    def test_build_labelled_profile_edge_case(
        self, mock_config: Dict[str, Any], mock_synthetic_params: Dict[str, Any]
    ) -> None:
        """Test build_labelled_profile with edge case configuration."""
        # Override the mock_config with the correct structure
        mock_config.update(
            {
                "custom_profile": True,
                "user_attributes": {
                    "goal": {"values": ["goal1"]},
                    "user_profiles": {
                        "values": ["prof1", "prof2"]
                    },  # Direct values structure
                    "system_attributes": {"attr1": {"values": ["val1"]}},
                    # Add some attributes that filter_attributes will return
                    "attr1": {"values": ["val1", "val2"]},
                    "attr2": {"values": ["val3", "val4"]},
                    "category1": {"values": ["cat1", "cat2"]},
                },
            }
        )

        # Mock the client to return proper responses
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "adapted_goal"
        mock_client.chat.completions.create.return_value = mock_response
        mock_config["client"] = mock_client

        profiles, goals, attributes, system_attrs, labels = build_labelled_profile(
            mock_synthetic_params, mock_config
        )

        assert len(profiles) == 2
        assert len(goals) == 2
        assert len(attributes) == 2
        assert len(system_attrs) == 2
        assert len(labels) == 2

    def test_attributes_to_text_empty(self) -> None:
        """Test attributes_to_text with empty list."""
        result = attributes_to_text([])
        assert result == []

    def test_convert_attributes_to_profiles_with_empty_lists(
        self, mock_config: Dict[str, Any]
    ) -> None:
        """Test convert_attributes_to_profiles with empty lists."""
        profiles, goals, system_inputs = convert_attributes_to_profiles(
            [], [], mock_config
        )
        assert profiles == []
        assert goals == []
        assert system_inputs == []

    def test_convert_attributes_to_profile_empty(
        self, mock_chatgpt_chatbot: Mock, mock_config: Dict[str, Any]
    ) -> None:
        """Test convert_attributes_to_profile with empty attributes."""
        attributes = {}
        result = convert_attributes_to_profile(attributes, mock_config)
        assert result == "mocked_response"

    def test_build_profile_custom_with_binding(
        self,
        custom_profile_mock_setup,
        mock_config: Dict[str, Any],
        mock_synthetic_params: Dict[str, int],
    ) -> None:
        """Test build_profile with custom profiles and binding."""
        mock_config["custom_profile"] = True
        mock_config["user_attributes"] = {
            "goal": {"values": ["goal1", "goal2"]},
            "user_profiles": {"profile1": {"values": ["prof1", "prof2"]}},
            "system_attributes": {"attr1": {"values": ["val1", "val2"]}},
        }

        profiles, goals, attributes, system_attrs, labels = build_profile(
            mock_synthetic_params, mock_config
        )

        assert len(profiles) == 2
        assert len(goals) == 2
        assert len(attributes) == 2
        assert len(system_attrs) == 2
        assert len(labels) == 2

    def test_build_profile_custom_without_documents(
        self,
        custom_profile_mock_setup,
        mock_config: Dict[str, Any],
        mock_synthetic_params: Dict[str, int],
    ) -> None:
        """Test build_profile with custom profiles without documents."""
        mock_config["custom_profile"] = True
        mock_config["user_attributes"] = {
            "goal": {"values": ["goal1", "goal2"]},
            "user_profiles": {"profile1": {"values": ["prof1", "prof2"]}},
            "system_attributes": {"attr1": {"values": ["val1", "val2"]}},
        }
        custom_profile_mock_setup["load_docs"].return_value = []

        profiles, goals, attributes, system_attrs, labels = build_profile(
            mock_synthetic_params, mock_config
        )

        assert len(profiles) == 2
        assert len(goals) == 2
        assert len(attributes) == 2
        assert len(system_attrs) == 2
        assert len(labels) == 2

    def test_pick_attributes_with_invalid_strategy(self) -> None:
        """Test pick_attributes with invalid strategy."""
        user_profile = {"profile": "test"}
        attributes = {"category1": ["value1", "value2"]}
        goals = ["goal1", "goal2"]

        # The function doesn't raise ValueError, it falls back to random
        result_attributes, result_matched = pick_attributes(
            user_profile, attributes, goals, strategy="invalid"
        )

        assert isinstance(result_attributes, dict)
        assert isinstance(result_matched, dict)
        assert "goal" in result_attributes

    def test_adapt_goal_with_empty_doc(
        self, mock_chatgpt_chatbot: Mock, mock_config: Dict[str, Any]
    ) -> None:
        """Test adapt_goal with empty document."""
        goal = "test goal"
        doc = ""
        user_profile = "test profile"

        result = adapt_goal(goal, mock_config, doc, user_profile)
        assert result == "mocked_response"

    def test_select_system_attributes_with_empty_config(
        self, mock_synthetic_params: Dict[str, int]
    ) -> None:
        """Test select_system_attributes with empty config."""
        config = {"user_attributes": {"system_attributes": {}}}
        result = select_system_attributes(config, mock_synthetic_params)
        assert len(result) == 2
        assert all(isinstance(attr, dict) for attr in result)

    def test_augment_attributes_with_empty_documents(
        self, mock_config: Dict[str, Any]
    ) -> None:
        """Test augment_attributes with empty documents."""
        attributes = {"category": {"values": ["value1"], "generate_values": True}}
        documents = []

        result = augment_attributes(attributes, mock_config, documents)
        assert isinstance(result, dict)

    def test_filter_attributes_with_empty_config(self) -> None:
        """Test filter_attributes with empty config."""
        config = {"user_attributes": {}}
        result = filter_attributes(config)
        assert isinstance(result, dict)

    def test_get_custom_profiles_with_empty_config(self) -> None:
        """Test get_custom_profiles with empty config."""
        config = {"user_attributes": {}}
        with pytest.raises(KeyError):
            get_custom_profiles(config)

    def test_build_profile_with_empty_config(self) -> None:
        """Test build_profile with empty config."""
        synthetic_data_params = {"num_goals": 2, "num_convos": 2}
        config = {}

        with pytest.raises(KeyError):
            build_profile(synthetic_data_params, config)

    def test_build_profile_with_zero_convos(self) -> None:
        """Test build_profile with zero conversations."""
        synthetic_data_params = {"num_goals": 2, "num_convos": 0}
        config = {
            "documents_dir": "/test/documents",
            "custom_profile": False,
            "system_inputs": True,
            "company_summary": "Test company summary",
            "client": Mock(),
            "user_attributes": {
                "goal": {"values": ["goal1", "goal2"]},
                "user_profiles": {
                    "values": ["prof1", "prof2"]
                },  # Direct values structure
                "system_attributes": {"attr1": {"values": ["val1", "val2"]}},
                # Add some attributes that filter_attributes will return
                "attr1": {"values": ["val1", "val2"]},
                "attr2": {"values": ["val3", "val4"]},
            },
            "tools": [{"id": "tool1", "name": "Test Tool"}],
            "workers": [{"id": "worker1", "name": "Test Worker"}],
        }

        # Mock the client to return proper responses
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "adapted_goal"
        mock_client.chat.completions.create.return_value = mock_response
        config["client"] = mock_client

        profiles, goals, attributes, system_attrs, labels = build_profile(
            synthetic_data_params, config
        )

        assert len(profiles) == 0
        assert len(goals) == 0
        assert len(attributes) == 0
        assert len(system_attrs) == 0
        assert len(labels) == 0

    def test_get_custom_profiles_with_api(self, mock_requests_get: Mock) -> None:
        """Test get_custom_profiles with API calls."""
        config = {
            "user_attributes": {
                "system_attributes": {
                    "location": {
                        "api": "http://api.example.com/location",
                        "values": ["NYC", "LA"],
                    }
                },
                "user_profiles": {
                    "profile1": {
                        "api": "http://api.example.com/profile",
                        "values": ["prof1", "prof2"],
                    }
                },
            }
        }

        user_profiles, system_attributes = get_custom_profiles(config)

        assert len(user_profiles) == 1
        assert len(system_attributes) == 1
        # The actual function doesn't make API calls, it uses the values directly
        mock_requests_get.assert_not_called()

    def test_adapt_goal_missing_company_summary(self) -> None:
        """Test adapt_goal with missing company summary."""
        config = {"client": Mock()}
        with pytest.raises(KeyError):
            adapt_goal("test goal", config, "test doc", "test profile")

    def test_attributes_to_text_various_types(self) -> None:
        """Test attributes_to_text with various data types."""
        attribute_list = [{"str_attr": "string", "int_attr": 123, "bool_attr": True}]
        result = attributes_to_text(attribute_list)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_convert_attributes_to_profiles_mismatched_lengths(
        self, mock_config: Dict[str, Any]
    ) -> None:
        """Test convert_attributes_to_profiles with mismatched list lengths."""
        attributes_list = [{"attr1": "value1", "goal": "goal1"}]
        system_attributes_list = [{"sys1": "val1"}, {"sys2": "val2"}]

        # Mock the client to return proper responses
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "converted_profile"
        mock_client.chat.completions.create.return_value = mock_response
        mock_config["client"] = mock_client

        # The function uses zip, so it will only process the shorter list
        profiles, goals, system_inputs = convert_attributes_to_profiles(
            attributes_list, system_attributes_list, mock_config
        )

        # Should only have one result since attributes_list has length 1
        assert len(profiles) == 1
        assert len(goals) == 1
        assert len(system_inputs) == 1

    def test_filter_attributes_missing_user_attributes(self) -> None:
        """Test filter_attributes with missing user_attributes."""
        config = {}
        with pytest.raises(KeyError):
            filter_attributes(config)

    def test_augment_attributes_with_unexpected_documents(
        self, mock_config: Dict[str, Any]
    ) -> None:
        """Test augment_attributes with unexpected document format."""
        attributes = {"category": {"values": ["value1"], "generate_values": True}}
        documents = ["unexpected_string"]

        result = augment_attributes(attributes, mock_config, documents)
        assert isinstance(result, dict)

    def test_get_label_no_matching_tool(self, mock_environment: Mock) -> None:
        """Test get_label when no matching tool is found."""
        attribute = {"goal": "test goal"}
        config = {"tools": [{"id": "tool1"}], "workers": [], "client": Mock()}

        label, valid = get_label(attribute, config)
        assert isinstance(label, list)
        assert valid is True

    def test_find_matched_attribute_with_long_prompt(self) -> None:
        """Test find_matched_attribute with long prompt."""
        goal = "test goal"
        user_profile = "test profile"
        client = Mock()

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatbot:
            mock_chatbot.return_value = "Thought: test\nAttribute: test_attr"
            result = find_matched_attribute(goal, user_profile, client=client)

            # Verify the prompt is properly formatted
            call_args = mock_chatbot.call_args[0][0]
            # The function passes a list of messages, so we need to check the content
            assert isinstance(call_args, list)
            assert len(call_args) == 1
            assert call_args[0]["role"] == "user"
            assert result == "test_attr"  # The function returns just the attribute part

    def test_find_matched_attribute_with_client(self) -> None:
        """Test find_matched_attribute with client parameter."""
        goal = "test goal"
        user_profile = "test profile"
        client = Mock()

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatbot:
            mock_chatbot.return_value = "Thought: test\nAttribute: test_attr"
            result = find_matched_attribute(goal, user_profile, client=client)

            assert result == "test_attr"
            mock_chatbot.assert_called_once()

    def test_find_matched_attribute_invalid_strategy(self) -> None:
        """Test find_matched_attribute with invalid strategy."""
        goal = "test goal"
        user_profile = "test profile"

        with pytest.raises(ValueError, match="Invalid strategy"):
            find_matched_attribute(goal, user_profile, strategy="invalid")

    def test_find_matched_attribute_dict_version(self) -> None:
        """Test find_matched_attribute with dict version."""
        goal = "test goal"
        user_profile = {"attr1": "value1", "attr2": "value2"}

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatbot:
            mock_chatbot.return_value = "Thought: test\nAttribute: attr1"
            result = find_matched_attribute(goal, user_profile, strategy="react")

            # The function returns a dict when user_profile is a dict
            assert isinstance(result, dict)
            assert "goal" in result
            assert "matched_attribute" in result
            # When user_profile is a dict, chatgpt_chatbot is not called
            mock_chatbot.assert_not_called()

    def test_adapt_goal_with_company_summary(self) -> None:
        """Test adapt_goal with company summary."""
        goal = "test goal"
        config = {"company_summary": "Test company summary", "client": Mock()}
        doc = "test document"
        user_profile = "test profile"

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatbot:
            mock_chatbot.return_value = "adapted goal"
            result = adapt_goal(goal, config, doc, user_profile)

            assert result == "adapted goal"
            mock_chatbot.assert_called_once()

    def test_adapt_goal_without_company_summary(self) -> None:
        """Test adapt_goal without company summary."""
        goal = "test goal"
        config = {"client": Mock()}
        doc = "test document"
        user_profile = "test profile"

        with pytest.raises(KeyError):
            adapt_goal(goal, config, doc, user_profile)

    def test_get_custom_profiles_with_api_endpoint(
        self, mock_requests_get: Mock
    ) -> None:
        """Test get_custom_profiles with API endpoint configuration."""
        mock_requests_get.return_value.json.return_value = ["api_value1", "api_value2"]

        config = {
            "user_attributes": {
                "system_attributes": {"attr1": {"api": "http://test.com/api1"}},
                "user_profiles": {"profile1": {"api": "http://test.com/api2"}},
            }
        }

        with pytest.raises(KeyError, match="values"):
            user_profiles, system_attributes = get_custom_profiles(config)

    def test_build_profile_with_empty_documents(self) -> None:
        """Test build_profile with empty documents."""
        synthetic_data_params = {"num_goals": 2, "num_convos": 2}
        config = {
            "documents_dir": "/test/documents",
            "custom_profile": False,
            "system_inputs": True,
            "company_summary": "Test company summary",
            "client": Mock(),
            "user_attributes": {
                "goal": {"values": ["goal1", "goal2"]},
                "system_attributes": {
                    "attr1": {"values": ["val1", "val2"]},
                    "attr2": {"values": ["val3", "val4"]},
                },
                "user_profiles": {
                    "values": ["prof1", "prof2"]
                },  # Direct values structure
            },
            "tools": [{"id": "tool1", "name": "Test Tool"}],
            "workers": [{"id": "worker1", "name": "Test Worker"}],
        }

        # Mock the client to return proper responses
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "adapted_goal"
        mock_client.chat.completions.create.return_value = mock_response
        config["client"] = mock_client

        with patch("arklex.evaluation.build_user_profiles.load_docs") as mock_load_docs:
            mock_load_docs.return_value = []

            profiles, goals, attributes, system_attrs, labels = build_profile(
                synthetic_data_params, config
            )

            assert len(profiles) == 2
            assert len(goals) == 2
            assert len(attributes) == 2
            assert len(system_attrs) == 2
            assert len(labels) == 2

    def test_build_profile_with_custom_profiles(
        self,
        mock_load_docs,
        mock_filter_attributes,
        mock_augment_attributes,
        mock_get_custom_profiles,
        mock_pick_attributes,
        mock_adapt_goal,
    ) -> None:
        """Test build_profile with custom profiles enabled."""
        mock_load_docs.return_value = [{"content": "test document content"}]
        mock_get_custom_profiles.return_value = (
            {"profile1": ["prof1", "prof2"]},
            {"attr1": ["val1", "val2"]},
        )

        # Mock the client response properly
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "mocked profile"
        mock_client.chat.completions.create.return_value = mock_response

        config = {
            "documents_dir": "/test/documents",
            "custom_profile": True,
            "system_inputs": True,
            "company_summary": "Test company summary",
            "client": mock_client,
            "user_attributes": {
                "goal": {"values": ["goal1", "goal2"]},
                "system_attributes": {"attr1": {"values": ["val1", "val2"]}},
                "user_profiles": {"profile1": {"values": ["prof1", "prof2"]}},
            },
        }

        synthetic_data_params = {"num_convos": 1, "num_goals": 3}

        profiles, goals, attributes_list, system_inputs, labels_list = build_profile(
            synthetic_data_params, config
        )

        assert len(profiles) == 1
        assert len(goals) == 1
        assert len(attributes_list) == 1
        assert len(system_inputs) == 1
        assert len(labels_list) == 1

    def test_build_profile_without_custom_profiles(
        self,
        mock_load_docs,
        mock_filter_attributes,
        mock_augment_attributes,
        mock_pick_attributes,
        mock_adapt_goal,
        mock_select_system_attributes,
    ) -> None:
        """Test build_profile without custom profiles."""
        mock_load_docs.return_value = [{"content": "test document content"}]

        # Mock the client response properly
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "mocked profile"
        mock_client.chat.completions.create.return_value = mock_response

        config = {
            "documents_dir": "/test/documents",
            "custom_profile": False,
            "system_inputs": True,
            "company_summary": "Test company summary",
            "client": mock_client,
            "user_attributes": {
                "goal": {"values": ["goal1", "goal2"]},
                "system_attributes": {
                    "attr1": {"values": ["val1", "val2"]},
                    "attr2": {"values": ["val3", "val4"]},
                },
                "user_profiles": {"profile1": {"values": ["p1"]}},
            },
            "tools": [{"id": "tool1", "name": "Test Tool"}],
            "workers": [{"id": "worker1", "name": "Test Worker"}],
        }

        synthetic_data_params = {"num_convos": 1, "num_goals": 3}

        profiles, goals, attributes_list, system_inputs, labels_list = build_profile(
            synthetic_data_params, config
        )

        assert len(profiles) == 1
        assert len(goals) == 1
        assert len(attributes_list) == 1
        assert len(system_inputs) == 1
        assert len(labels_list) == 1

    def test_get_custom_profiles_with_api_calls(self, mock_requests_get: Mock) -> None:
        """Test get_custom_profiles with multiple API calls."""
        mock_requests_get.return_value.json.side_effect = [
            ["api_value1", "api_value2"],
            ["api_value3", "api_value4"],
        ]

        config = {
            "user_attributes": {
                "system_attributes": {
                    "attr1": {"api": "http://test.com/api1"},
                    "attr2": {"api": "http://test.com/api2"},
                },
                "user_profiles": {
                    "profile1": {"api": "http://test.com/api3"},
                    "profile2": {"api": "http://test.com/api4"},
                },
            }
        }

        with pytest.raises(KeyError, match="values"):
            user_profiles, system_attributes = get_custom_profiles(config)

    def test_get_custom_profiles_without_api_calls(self) -> None:
        """Test get_custom_profiles without API calls."""
        config = {
            "user_attributes": {
                "system_attributes": {"attr1": {"values": ["val1", "val2"]}},
                "user_profiles": {"profile1": {"values": ["prof1", "prof2"]}},
            }
        }

        user_profiles, system_attributes = get_custom_profiles(config)

        assert user_profiles == {"profile1": ["prof1", "prof2"]}
        assert system_attributes == {"attr1": ["val1", "val2"]}

    def test_get_custom_profiles_with_bindings(self) -> None:
        """Test get_custom_profiles with binding configuration."""
        config = {
            "user_attributes": {
                "system_attributes": {
                    "attr1": {"values": ["val1", "val2"], "bind_to": "profile1"}
                },
                "user_profiles": {
                    "profile1": {
                        "values": ["prof1", "prof2"],
                        "bind_to": "system_attributes.attr1",
                    }
                },
            }
        }

        user_profiles, system_attributes = get_custom_profiles(config)

        assert user_profiles == {"profile1": ["prof1", "prof2"]}
        assert system_attributes == {"attr1": ["val1", "val2"]}

    def test_get_label_with_retry_logic(
        self, mock_slot_filler, mock_chatgpt_chatbot, mock_environment
    ) -> None:
        """Test get_label with retry logic on failures."""
        mock_tool = Mock()
        mock_slot = Mock()
        mock_slot.name = "slot1"  # Set the name attribute to a string
        mock_tool.slots = [mock_slot]
        mock_tool.description = "Test tool"
        mock_tool.output = "Test output"
        mock_tool.name = "Test Tool"

        mock_environment.return_value.tools = {
            "tool1": {"execute": Mock(return_value=mock_tool)}
        }

        # First two attempts fail, third succeeds
        mock_chatgpt_chatbot.side_effect = [
            Exception("First attempt failed"),
            Exception("Second attempt failed"),
            "tool1",
        ]

        slot_filled = Mock(name="slot1", value="filled_value")
        slot_filled.name = "slot1"  # Ensure .name is a string, not a Mock
        mock_slot_filler.return_value.execute.return_value = [slot_filled]

        config = {
            "tools": [{"id": "tool1", "name": "Test Tool"}],
            "workers": [],
            "client": Mock(),
        }

        attribute = {"goal": "test goal"}

        label, valid = get_label(attribute, config)

        assert label == [
            {
                "tool_id": "tool1",
                "tool_name": "Test Tool",
                "slots": {"slot1": "filled_value"},
            }
        ]
        assert valid is True
        assert mock_chatgpt_chatbot.call_count == 3

    def test_get_label_all_attempts_fail(
        self, mock_chatgpt_chatbot, mock_environment
    ) -> None:
        """Test get_label when all attempts fail."""
        mock_tool = Mock()
        mock_tool.slots = [Mock(name="slot1")]
        mock_tool.description = "Test tool"
        mock_tool.output = "Test output"
        mock_tool.name = "Test Tool"

        mock_environment.return_value.tools = {
            "tool1": {"execute": Mock(return_value=mock_tool)}
        }

        # All attempts fail
        mock_chatgpt_chatbot.side_effect = [
            Exception("First attempt failed"),
            Exception("Second attempt failed"),
            Exception("Third attempt failed"),
        ]

        config = {
            "tools": [{"id": "tool1", "name": "Test Tool"}],
            "workers": [],
            "client": Mock(),
        }

        attribute = {"goal": "test goal"}

        label, valid = get_label(attribute, config)

        assert label == [{"tool_id": "0", "tool_name": "No tool", "slots": {}}]
        assert valid is True
        assert mock_chatgpt_chatbot.call_count == 3

    def test_get_label_with_tool_id_zero(
        self, mock_slot_filler, mock_chatgpt_chatbot, mock_environment
    ) -> None:
        """Test get_label when tool_id is 0."""
        mock_tool = Mock()
        mock_tool.slots = [Mock(name="slot1")]
        mock_tool.description = "Test tool"
        mock_tool.output = "Test output"
        mock_tool.name = "Test Tool"

        mock_environment.return_value.tools = {
            "tool1": {"execute": Mock(return_value=mock_tool)}
        }

        mock_chatgpt_chatbot.return_value = "0"  # No tool selected

        config = {
            "tools": [{"id": "tool1", "name": "Test Tool"}],
            "workers": [],
            "client": Mock(),
        }

        attribute = {"goal": "test goal"}

        label, valid = get_label(attribute, config)

        assert label == [{"tool_id": "0", "tool_name": "No tool", "slots": {}}]
        assert valid is True
        mock_slot_filler.assert_not_called()

    def test_select_system_attributes_with_value_error(self) -> None:
        """Test select_system_attributes with ValueError."""
        config = {"user_attributes": {"system_attributes": {"attr1": "not_a_list"}}}

        synthetic_data_params = {"num_convos": 1}

        with pytest.raises(TypeError, match="string indices must be integers"):
            select_system_attributes(config, synthetic_data_params)

    def test_select_system_attributes_with_api_calls(
        self, mock_requests_get: Mock
    ) -> None:
        """Test select_system_attributes with API calls."""
        mock_requests_get.return_value.json.return_value = [
            {"key": "value1"},
            {"key": "value2"},
        ]

        config = {
            "user_attributes": {
                "system_attributes": {"attr1": {"api": "http://test.com/api1"}}
            }
        }

        synthetic_data_params = {"num_convos": 2}

        with pytest.raises(KeyError, match="values"):
            result = select_system_attributes(config, synthetic_data_params)

    def test_augment_attributes_without_documents(
        self, mock_chatgpt_chatbot: Mock
    ) -> None:
        """Test augment_attributes without documents."""
        attributes = {"category1": {"values": ["val1"], "generate_values": True}}

        config = {"intro": "Test company", "client": Mock()}
        documents = []

        mock_chatgpt_chatbot.return_value = "new_val1, new_val2"

        result = augment_attributes(attributes, config, documents)

        # Should return original values when no documents are available
        assert result["category1"] == ["val1"]

    def test_augment_attributes_with_empty_values_continue(
        self, mock_chatgpt_chatbot: Mock
    ) -> None:
        """Test augment_attributes with empty values that should be skipped."""
        attributes = {
            "category1": {"values": [], "generate_values": True},
            "category2": {"values": ["val1"], "generate_values": True},
        }

        config = {"intro": "Test company", "client": Mock()}
        documents = [{"content": "test doc"}]

        mock_chatgpt_chatbot.return_value = "new_val1, new_val2"

        result = augment_attributes(attributes, config, documents)

        # category1 should be included even with empty values
        assert "category1" in result
        assert result["category1"] == []
        assert result["category2"] == ["val1"]

    def test_get_custom_profiles_missing_system_attributes(self) -> None:
        """Test get_custom_profiles when system_attributes is missing."""
        config = {
            "user_attributes": {
                "user_profiles": {"profile1": {"values": ["prof1", "prof2"]}}
            }
        }

        with pytest.raises(KeyError, match="system_attributes"):
            user_profiles, system_attributes = get_custom_profiles(config)

    def test_get_custom_profiles_missing_user_profiles(self) -> None:
        """Test get_custom_profiles when user_profiles is missing."""
        config = {
            "user_attributes": {
                "system_attributes": {"attr1": {"values": ["val1", "val2"]}}
            }
        }

        with pytest.raises(KeyError, match="user_profiles"):
            user_profiles, system_attributes = get_custom_profiles(config)

    def test_get_custom_profiles_with_non_dict_values_and_api(
        self, mock_requests_get: Mock
    ) -> None:
        """Test get_custom_profiles with non-dict values and API calls."""
        mock_requests_get.return_value.json.return_value = ["api_value1", "api_value2"]

        config = {
            "user_attributes": {
                "system_attributes": {"attr1": {"api": "http://test.com/api1"}},
                "user_profiles": {"profile1": "not_a_dict"},
            }
        }

        with pytest.raises(TypeError, match="string indices must be integers"):
            user_profiles, system_attributes = get_custom_profiles(config)

    def test_pick_attributes_random_with_empty_goals(self) -> None:
        """Test pick_attributes_random with empty goals list."""
        user_profile = {"attr1": "val1"}
        attributes = {"attr1": ["val1", "val2"]}
        goals = []

        with pytest.raises(IndexError):
            pick_attributes_random(user_profile, attributes, goals)

    def test_pick_attributes_random_with_empty_attributes(self) -> None:
        """Test pick_attributes_random with empty attributes."""
        user_profile = {"attr1": "val1"}
        attributes = {}
        goals = ["goal1"]

        result_attributes, result_matched = pick_attributes_random(
            user_profile, attributes, goals
        )

        assert "goal" in result_attributes
        assert result_attributes["goal"] == "goal1"
        assert result_matched["goal"] == "goal1"

    def test_find_matched_attribute_with_llm_based_strategy(
        self, mock_chatgpt_chatbot: Mock
    ) -> None:
        """Test find_matched_attribute with llm_based strategy."""
        mock_chatgpt_chatbot.return_value = (
            "Thought: User needs product info\nAttribute: product_inquiry"
        )

        goal = "interested in product information"
        user_profile = "user_info: customer details, current_webpage: product page"
        strategy = "llm_based"
        client = Mock()

        with pytest.raises(ValueError, match="Invalid strategy"):
            result = find_matched_attribute(goal, user_profile, strategy, client)

    def test_find_matched_attribute_with_invalid_strategy(self) -> None:
        """Test find_matched_attribute with invalid strategy."""
        goal = "test goal"
        user_profile = "test profile"
        strategy = "invalid_strategy"

        with pytest.raises(ValueError, match="Invalid strategy"):
            find_matched_attribute(goal, user_profile, strategy)

    def test_adapt_goal_with_missing_company_summary(
        self, mock_chatgpt_chatbot: Mock
    ) -> None:
        """Test adapt_goal when company_summary is missing from config."""
        mock_chatgpt_chatbot.return_value = "adapted goal"

        goal = "test goal"
        config = {"client": Mock()}  # Missing company_summary
        doc = "test doc"
        user_profile = "test profile"

        with pytest.raises(KeyError, match="company_summary"):
            result = adapt_goal(goal, config, doc, user_profile)

    def test_convert_attributes_to_profiles_with_mismatched_lengths(
        self, mock_chatgpt_chatbot: Mock
    ) -> None:
        """Test convert_attributes_to_profiles with mismatched list lengths."""
        mock_chatgpt_chatbot.return_value = "test profile"

        attributes_list = [{"attr1": "val1", "goal": "goal1"}]
        system_attributes_list = [
            {"sys1": "val1"},
            {"sys2": "val2"},
        ]  # Different length
        config = {"company_summary": "Test company", "client": Mock()}

        profiles, goals, system_inputs = convert_attributes_to_profiles(
            attributes_list, system_attributes_list, config
        )

        assert len(profiles) == 1
        assert len(goals) == 1
        assert len(system_inputs) == 1

    def test_attributes_to_text_with_various_types(self) -> None:
        """Test attributes_to_text with various data types."""
        attribute_list = [
            {
                "string": "text",
                "number": 123,
                "boolean": True,
                "list": [1, 2, 3],
                "none": None,
            }
        ]

        result = attributes_to_text(attribute_list)

        assert len(result) == 1
        assert "string: text" in result[0]
        assert "number: 123" in result[0]
        assert "boolean: True" in result[0]
        assert "list: [1, 2, 3]" in result[0]
        assert "none: None" in result[0]

    def test_build_user_profiles_with_empty_test_data(
        self, mock_slot_filler: Mock
    ) -> None:
        """Test build_user_profiles with empty test data."""
        test_data = []

        result = build_user_profiles(test_data)

        assert result is None

    def test_build_user_profiles_with_invalid_test_data(
        self, mock_slot_filler: Mock
    ) -> None:
        """Test build_user_profiles with invalid test data."""
        test_data = [{"invalid": "data"}]  # Missing required fields

        result = build_user_profiles(test_data)

        assert result is None

    def test_filter_attributes_with_missing_user_attributes(self) -> None:
        """Test filter_attributes when user_attributes is missing."""
        config = {"synthetic_data_params": {"customer_type": "test"}}

        with pytest.raises(KeyError, match="user_attributes"):
            result = filter_attributes(config)

    def test_filter_attributes_with_missing_synthetic_data_params(self) -> None:
        """Test filter_attributes when synthetic_data_params is missing."""
        config = {
            "user_attributes": {
                "generic": {"attr1": {"values": ["val1"]}},
                "test_type": {"attr2": {"values": ["val2"]}},
            }
        }

        result = filter_attributes(config)

        # Should return all attributes when synthetic_data_params is missing
        assert "generic" in result
        assert "test_type" in result
        assert result["generic"] == {"attr1": {"values": ["val1"]}}
        assert result["test_type"] == {"attr2": {"values": ["val2"]}}

    def test_get_custom_profiles_with_api_and_bindings(
        self, mock_requests_get: Mock
    ) -> None:
        """Test get_custom_profiles with API calls and bindings."""
        mock_requests_get.return_value.json.return_value = ["api_value1", "api_value2"]

        config = {
            "user_attributes": {
                "system_attributes": {
                    "attr1": {"api": "http://test.com/api1", "bind_to": "profile1"}
                },
                "user_profiles": {"profile1": {"bind_to": "system_attributes.attr1"}},
            }
        }

        with pytest.raises(KeyError, match="values"):
            user_profiles, system_attributes = get_custom_profiles(config)
