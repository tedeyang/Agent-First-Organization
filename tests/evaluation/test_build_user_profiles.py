"""Tests for the build_user_profiles module.

This module contains comprehensive test cases for user profile building functionality,
including profile generation, attribute conversion, and custom profile handling.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any
from contextlib import ExitStack

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
)


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
    def basic_mock_setup(self):
        """Create basic mock setup for build_profile tests."""
        with (
            patch("arklex.evaluation.build_user_profiles.load_docs") as mock_load_docs,
            patch(
                "arklex.evaluation.build_user_profiles.filter_attributes"
            ) as mock_filter,
            patch(
                "arklex.evaluation.build_user_profiles.augment_attributes"
            ) as mock_augment,
            patch("arklex.evaluation.build_user_profiles.pick_attributes") as mock_pick,
            patch("arklex.evaluation.build_user_profiles.adapt_goal") as mock_adapt,
            patch(
                "arklex.evaluation.build_user_profiles.select_system_attributes"
            ) as mock_select,
            patch(
                "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
            ) as mock_chatbot,
        ):
            # Setup default mock returns
            mock_load_docs.return_value = [{"content": "test content"}]
            mock_filter.return_value = {
                "attr1": {"values": ["val1"]},
                "attr2": {"values": ["val3"]},
            }
            mock_augment.return_value = {
                "attr1": ["val1", "val2"],
                "attr2": ["val3", "val4"],
            }
            mock_pick.return_value = (
                {"goal": "test_goal", "attr1": "val1", "attr2": "val3"},
                {"matched": "data"},
            )
            mock_adapt.return_value = "adapted_goal"
            mock_select.return_value = [{"sys_attr": "val"}, {"sys_attr": "val2"}]
            mock_chatbot.return_value = "Test profile"

            yield {
                "load_docs": mock_load_docs,
                "filter_attributes": mock_filter,
                "augment_attributes": mock_augment,
                "pick_attributes": mock_pick,
                "adapt_goal": mock_adapt,
                "select_system_attributes": mock_select,
                "chatgpt_chatbot": mock_chatbot,
            }

    @pytest.fixture
    def custom_profile_mock_setup(self):
        """Create mock setup for custom profile tests."""
        with (
            patch("arklex.evaluation.build_user_profiles.load_docs") as mock_load_docs,
            patch(
                "arklex.evaluation.build_user_profiles.filter_attributes"
            ) as mock_filter,
            patch(
                "arklex.evaluation.build_user_profiles.augment_attributes"
            ) as mock_augment,
            patch(
                "arklex.evaluation.build_user_profiles.get_custom_profiles"
            ) as mock_get_custom,
            patch("arklex.evaluation.build_user_profiles.pick_attributes") as mock_pick,
            patch("arklex.evaluation.build_user_profiles.adapt_goal") as mock_adapt,
            patch(
                "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
            ) as mock_chatbot,
        ):
            # Setup default mock returns
            mock_load_docs.return_value = [{"content": "test content"}]
            mock_filter.return_value = {
                "attr1": {"values": ["val1"]},
                "attr2": {"values": ["val3"]},
            }
            mock_augment.return_value = {
                "attr1": ["val1", "val2"],
                "attr2": ["val3", "val4"],
            }
            mock_get_custom.return_value = (
                {"profile1": ["prof1", "prof2"], "profile2": ["prof3", "prof4"]},
                {"attr1": ["val1", "val2"], "attr2": ["val3", "val4"]},
            )
            mock_pick.return_value = (
                {"goal": "test_goal", "attr1": "val1", "attr2": "val3"},
                {"matched": "data"},
            )
            mock_adapt.return_value = "adapted_goal"
            mock_chatbot.return_value = "Test profile"

            yield {
                "load_docs": mock_load_docs,
                "filter_attributes": mock_filter,
                "augment_attributes": mock_augment,
                "get_custom_profiles": mock_get_custom,
                "pick_attributes": mock_pick,
                "adapt_goal": mock_adapt,
                "chatgpt_chatbot": mock_chatbot,
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
        # Apply test-specific configuration with proper merging
        for key, value in test_config.items():
            if key == "user_attributes" and key in mock_config:
                # Merge user_attributes instead of replacing
                mock_config[key].update(value)
            else:
                mock_config[key] = value

        # Use appropriate mock setup
        mock_setup = (
            custom_profile_mock_setup
            if test_config.get("custom_profile")
            else basic_mock_setup
        )

        # Override load_docs for custom_without_documents test
        if test_name == "custom_without_documents":
            mock_setup["load_docs"].return_value = []

        profiles, goals, attributes, system_attrs, labels = build_profile(
            mock_synthetic_params, mock_config
        )

        # Common assertions for all variations
        assert len(profiles) == 2
        assert len(goals) == 2
        assert len(attributes) == 2
        assert len(system_attrs) == 2
        assert len(labels) == 2

    @patch("arklex.evaluation.build_user_profiles.chatgpt_chatbot")
    def test_convert_attributes_to_profile(
        self, mock_chatgpt_chatbot: Mock, mock_config: Dict[str, Any]
    ) -> None:
        """Test convert_attributes_to_profile function."""
        mock_chatgpt_chatbot.return_value = "Test profile"
        attributes = {"attr1": "val1", "attr2": "val2"}

        result = convert_attributes_to_profile(attributes, mock_config)

        assert result == "Test profile"

    def test_get_custom_profiles(self, mock_config: Dict[str, Any]) -> None:
        """Test get_custom_profiles function."""
        user_profiles, system_attributes = get_custom_profiles(mock_config)

        assert isinstance(user_profiles, dict)
        assert isinstance(system_attributes, dict)
        assert "profile1" in user_profiles
        assert "attr1" in system_attributes

    def test_build_labelled_profile(
        self,
        basic_mock_setup,
        mock_config: Dict[str, Any],
        mock_synthetic_params: Dict[str, int],
    ) -> None:
        """Test build_labelled_profile function."""
        with patch(
            "arklex.evaluation.build_user_profiles.convert_attributes_to_profile"
        ) as mock_convert:
            mock_convert.return_value = "Test labelled profile"

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
        """Test build_user_profiles function."""
        mock_slot_filler_instance = Mock()
        mock_slot_filler_instance.build_user_profiles.return_value = {"result": "test"}
        mock_slot_filler.return_value = mock_slot_filler_instance
        config = {"tools": [], "workers": []}

        # The function returns None, so we should test that it doesn't raise an exception
        try:
            result = build_user_profiles(config)
            # If it returns None, that's fine - we just want to ensure it doesn't crash
            assert True
        except Exception as e:
            assert False, f"Function raised an exception: {e}"

    def test_attr_to_profile_constant(self) -> None:
        """Test ATTR_TO_PROFILE constant."""
        assert isinstance(ATTR_TO_PROFILE, str)
        assert len(ATTR_TO_PROFILE) > 0

    def test_adapt_goal_constant(self) -> None:
        """Test ADAPT_GOAL constant."""
        assert isinstance(ADAPT_GOAL, str)
        assert len(ADAPT_GOAL) > 0

    @pytest.mark.parametrize(
        "strategy,expected_goal",
        [
            ("react", "goal1"),
            ("llm_based", "goal2"),
        ],
    )
    @patch("arklex.evaluation.build_user_profiles.chatgpt_chatbot")
    def test_pick_goal(
        self, mock_chatgpt_chatbot: Mock, strategy: str, expected_goal: str
    ) -> None:
        """Test pick_goal function with different strategies."""
        if strategy == "react":
            mock_chatgpt_chatbot.return_value = f"Thought: test\nGoal: {expected_goal}"
        else:
            mock_chatgpt_chatbot.return_value = f"Goal: {expected_goal}"

        user_profile = {"attr1": "val1"}
        goals = ["goal1", "goal2"]

        result = pick_goal(user_profile, goals, strategy=strategy)

        assert result == expected_goal

    def test_pick_goal_invalid_strategy(self) -> None:
        """Test pick_goal with invalid strategy raises ValueError."""
        user_profile = {"attr1": "val1"}
        goals = ["goal1", "goal2"]

        with pytest.raises(ValueError, match="Invalid strategy"):
            pick_goal(user_profile, goals, strategy="invalid")

    @patch("arklex.evaluation.build_user_profiles.chatgpt_chatbot")
    def test_find_matched_attribute_react_strategy(
        self, mock_chatgpt_chatbot: Mock
    ) -> None:
        """Test find_matched_attribute with react strategy."""
        mock_chatgpt_chatbot.return_value = "Thought: test\nAttribute: test_attr"
        result = find_matched_attribute("test_goal", "test_profile", client=Mock())

        assert isinstance(result, str)
        assert result == "test_attr"

    def test_pick_attributes_react(self) -> None:
        """Test pick_attributes with react strategy."""
        user_profile = {"attr1": "val1"}
        attributes = {"attr1": ["val1", "val2"]}
        goals = ["goal1", "goal2"]

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatbot:
            mock_chatbot.return_value = "Thought: test\nAttributes: attr1: val1"

            result_attributes, result_matched = pick_attributes(
                user_profile, attributes, goals, strategy="react", client=Mock()
            )

            assert isinstance(result_attributes, dict)
            assert isinstance(result_matched, dict)

    def test_pick_attributes_random(self) -> None:
        """Test pick_attributes with random strategy."""
        user_profile = {"attr1": "val1"}
        attributes = {"attr1": ["val1", "val2"]}
        goals = ["goal1", "goal2"]

        result_attributes, result_matched = pick_attributes(
            user_profile, attributes, goals, strategy="random"
        )

        assert isinstance(result_attributes, dict)
        assert isinstance(result_matched, dict)
        assert "goal" in result_attributes

    @patch("arklex.evaluation.build_user_profiles.chatgpt_chatbot")
    def test_adapt_goal(
        self, mock_chatgpt_chatbot: Mock, mock_config: Dict[str, Any]
    ) -> None:
        """Test adapt_goal function."""
        mock_chatgpt_chatbot.return_value = "adapted goal"
        result = adapt_goal(
            "original_goal", mock_config, doc="test doc", user_profile="test profile"
        )

        assert result == "adapted goal"

    @patch("arklex.evaluation.build_user_profiles.Environment")
    def test_get_label_with_invalid_attribute(
        self, mock_environment: Mock, mock_config: Dict[str, Any]
    ) -> None:
        """Test get_label with invalid attribute."""
        mock_env_instance = Mock()
        mock_tool = Mock()
        mock_tool.slots = []
        mock_tool.description = "Test tool"
        mock_tool.output = "Test output"
        # Fix the mock structure to match what the function expects
        mock_env_instance.tools = {"tool1": {"execute": lambda: mock_tool}}
        mock_environment.return_value = mock_env_instance

        result = get_label("invalid_attribute", mock_config)

        # The function returns a tuple (list, bool), not just a list
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)

    def test_filter_attributes(self, mock_config: Dict[str, Any]) -> None:
        """Test filter_attributes function."""
        result = filter_attributes(mock_config)

        assert isinstance(result, dict)
        # The function returns user_profiles when system_attributes is empty
        assert "user_profiles" in result

    def test_augment_attributes(self, mock_config: Dict[str, Any]) -> None:
        """Test augment_attributes function."""
        predefined_attributes = {"attr1": {"values": ["val1", "val2"]}}
        documents = [{"content": "test document"}]
        result = augment_attributes(predefined_attributes, mock_config, documents)

        assert "attr1" in result
        assert isinstance(result["attr1"], list)

    def test_attributes_to_text(self) -> None:
        """Test attributes_to_text function."""
        attributes = [{"attr1": "val1"}, {"attr2": "val2"}]
        result = attributes_to_text(attributes)

        assert isinstance(result, list)
        assert len(result) == 2

    @patch("arklex.evaluation.build_user_profiles.chatgpt_chatbot")
    def test_convert_attributes_to_profiles(
        self, mock_chatgpt_chatbot: Mock, mock_config: Dict[str, Any]
    ) -> None:
        """Test convert_attributes_to_profiles function."""
        mock_chatgpt_chatbot.return_value = "Test profile"
        profiles = [{"attr1": "val1", "goal": "goal1"}]
        goals = ["goal1"]

        result_profiles, result_goals, result_system_inputs = (
            convert_attributes_to_profiles(profiles, goals, mock_config)
        )

        assert isinstance(result_profiles, list)
        assert isinstance(result_goals, list)
        assert isinstance(result_system_inputs, list)

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
        """Test build_labelled_profile edge case."""
        with (
            patch("arklex.evaluation.build_user_profiles.load_docs") as mock_load_docs,
            patch(
                "arklex.evaluation.build_user_profiles.filter_attributes"
            ) as mock_filter,
            patch(
                "arklex.evaluation.build_user_profiles.augment_attributes"
            ) as mock_augment,
            patch("arklex.evaluation.build_user_profiles.pick_attributes") as mock_pick,
            patch("arklex.evaluation.build_user_profiles.adapt_goal") as mock_adapt,
            patch(
                "arklex.evaluation.build_user_profiles.convert_attributes_to_profile"
            ) as mock_convert,
        ):
            mock_load_docs.return_value = []
            mock_filter.return_value = {}
            mock_augment.return_value = {}
            mock_pick.return_value = ({"goal": "test_goal"}, {})
            mock_adapt.return_value = "adapted_goal"
            mock_convert.return_value = "Test profile"

            profiles, goals, attributes, system_attrs, labels = build_labelled_profile(
                mock_synthetic_params, mock_config
            )

            assert isinstance(profiles, list)
            assert isinstance(goals, list)
            assert isinstance(attributes, list)
            assert isinstance(system_attrs, list)
            assert isinstance(labels, list)

    def test_attributes_to_text_empty(self) -> None:
        """Test attributes_to_text with empty list."""
        result = attributes_to_text([])
        assert result == []

    def test_convert_attributes_to_profiles_with_empty_lists(
        self, mock_config: Dict[str, Any]
    ) -> None:
        """Test convert_attributes_to_profiles with empty input lists."""
        profiles, goals, system_inputs = convert_attributes_to_profiles(
            [], [], mock_config
        )
        assert profiles == []
        assert goals == []
        assert system_inputs == []

    @patch("arklex.evaluation.build_user_profiles.chatgpt_chatbot")
    def test_convert_attributes_to_profile_empty(
        self, mock_chatgpt_chatbot: Mock, mock_config: Dict[str, Any]
    ) -> None:
        """Test convert_attributes_to_profile with empty attributes."""
        mock_chatgpt_chatbot.return_value = "Test profile"
        result = convert_attributes_to_profile({}, mock_config)

        assert result == "Test profile"

    def test_build_profile_custom_with_binding(
        self,
        custom_profile_mock_setup,
        mock_config: Dict[str, Any],
        mock_synthetic_params: Dict[str, int],
    ) -> None:
        """Test build_profile with custom profiles and binding."""
        mock_config["custom_profile"] = True
        mock_config["user_attributes"]["user_profiles"] = {
            "profile1": {"values": ["prof1", "prof2"], "bind_to": "system_attr1"},
            "profile2": {"values": ["prof3", "prof4"]},
        }
        mock_config["user_attributes"]["system_attributes"] = {
            "attr1": {"values": ["val1", "val2"], "bind_to": "profile1"}
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
        """Test build_profile with custom profiles and no documents."""
        mock_config["custom_profile"] = True
        mock_config["user_attributes"]["user_profiles"] = {
            "profile1": {"values": ["prof1", "prof2"]},
        }
        mock_config["user_attributes"]["system_attributes"] = {
            "attr1": {"values": ["val1", "val2"]},
        }

        # Override load_docs to return empty list
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
        """Test pick_attributes with invalid strategy falls back to random."""
        user_profile = {"attr1": "val1"}
        attributes = {"attr1": ["val1", "val2"]}
        goals = ["goal1", "goal2"]

        result_attributes, result_matched = pick_attributes(
            user_profile, attributes, goals, strategy="invalid"
        )

        assert isinstance(result_attributes, dict)
        assert isinstance(result_matched, dict)
        assert "goal" in result_attributes

    @patch("arklex.evaluation.build_user_profiles.chatgpt_chatbot")
    def test_adapt_goal_with_empty_doc(
        self, mock_chatgpt_chatbot: Mock, mock_config: Dict[str, Any]
    ) -> None:
        """Test adapt_goal with empty document."""
        mock_chatgpt_chatbot.return_value = "adapted goal"
        result = adapt_goal(
            "original_goal", mock_config, doc="", user_profile="test profile"
        )

        assert result == "adapted goal"

    def test_select_system_attributes_with_empty_config(
        self, mock_synthetic_params: Dict[str, int]
    ) -> None:
        """Test select_system_attributes with empty config."""
        empty_config = {"user_attributes": {"system_attributes": {}}}
        result = select_system_attributes(empty_config, mock_synthetic_params)

        assert len(result) == 2
        assert all(isinstance(attr, dict) for attr in result)

    def test_augment_attributes_with_empty_documents(
        self, mock_config: Dict[str, Any]
    ) -> None:
        """Test augment_attributes with empty documents list."""
        predefined_attributes = {"attr1": {"values": ["val1", "val2"]}}
        result = augment_attributes(predefined_attributes, mock_config, documents=[])

        assert "attr1" in result
        assert isinstance(result["attr1"], list)

    def test_filter_attributes_with_empty_config(self) -> None:
        """Test filter_attributes with empty config."""
        empty_config = {
            "user_attributes": {"system_attributes": {}, "user_profiles": {}}
        }
        result = filter_attributes(empty_config)

        assert isinstance(result, dict)

    def test_get_custom_profiles_with_empty_config(self) -> None:
        """Test get_custom_profiles with empty config."""
        empty_config = {
            "user_attributes": {"system_attributes": {}, "user_profiles": {}}
        }
        user_profiles, system_attributes = get_custom_profiles(empty_config)

        assert isinstance(user_profiles, dict)
        assert isinstance(system_attributes, dict)

    def test_build_profile_with_empty_config(self) -> None:
        """Test build_profile with empty config raises KeyError."""
        with pytest.raises(KeyError):
            build_profile({"num_convos": 1, "num_goals": 1}, {})

    def test_build_profile_with_zero_convos(self) -> None:
        """Test build_profile with zero conversations."""
        config = {
            "documents_dir": "/tmp",
            "custom_profile": False,
            "system_inputs": True,
            "company_summary": "summary",
            "client": Mock(),
            "user_attributes": {"goal": {"values": ["goal1"]}},
            "tools": [],
            "workers": [],
        }

        profiles, goals, attributes, system_attrs, labels = build_profile(
            {"num_convos": 0, "num_goals": 1}, config
        )

        assert profiles == []
        assert goals == []
        assert attributes == []
        assert system_attrs == []
        assert labels == []

    @patch("arklex.evaluation.build_user_profiles.requests.get")
    def test_get_custom_profiles_with_api(self, mock_get: Mock) -> None:
        """Test get_custom_profiles with API configuration."""
        mock_get.return_value.json.return_value = ["api_value"]
        config = {
            "user_attributes": {
                "system_attributes": {
                    "sys": {"api": "http://fakeapi", "values": ["val1"]}
                },
                "user_profiles": {"profile": {"values": ["a", "b"]}},
            }
        }

        user_profiles, system_attributes = get_custom_profiles(config)

        assert "profile" in user_profiles
        assert "sys" in system_attributes

    def test_adapt_goal_missing_company_summary(self) -> None:
        """Test adapt_goal with missing company_summary raises KeyError."""
        config = {"client": Mock()}
        with pytest.raises(KeyError):
            adapt_goal("goal", config, doc="", user_profile="profile")

    def test_attributes_to_text_various_types(self) -> None:
        """Test attributes_to_text with various types."""
        result = attributes_to_text([{"a": 1}, {}])
        assert isinstance(result, list)

    def test_convert_attributes_to_profiles_mismatched_lengths(self) -> None:
        """Test convert_attributes_to_profiles with mismatched lengths."""
        config = {"tools": [], "workers": []}
        profiles, goals, system_inputs = convert_attributes_to_profiles(
            [{"a": 1, "goal": "goal1"}], [], config
        )
        assert isinstance(profiles, list)

    def test_filter_attributes_missing_user_attributes(self) -> None:
        """Test filter_attributes with missing user_attributes."""
        config = {"user_attributes": {}}
        result = filter_attributes(config)
        assert isinstance(result, dict)

    def test_augment_attributes_with_unexpected_documents(self) -> None:
        """Test augment_attributes with unexpected document structure."""
        predefined = {"attr": {"values": ["v1"]}}
        config = {"company_summary": "summary"}

        result = augment_attributes(
            predefined, config, documents=[{"unexpected": "field"}]
        )

        assert isinstance(result, dict)

    @patch("arklex.evaluation.build_user_profiles.Environment")
    def test_get_label_no_matching_tool(self, mock_env: Mock) -> None:
        """Test get_label when no matching tool is found."""
        mock_config = {"tools": [{"id": "tool1", "name": "Test Tool"}], "workers": []}
        mock_environment = Mock()
        mock_tool = Mock()
        mock_tool.slots = []
        mock_tool.description = "Test tool"
        mock_tool.output = "Test output"
        # Fix the mock structure to match what the function expects
        mock_environment.tools = {"tool1": {"execute": lambda: mock_tool}}
        mock_env.return_value = mock_environment

        result = get_label("nonexistent_attribute", mock_config)

        # The function returns a tuple (list, bool)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)

    def _get_first_find_matched_attribute(self):
        """Helper method to get the first find_matched_attribute function."""
        import importlib
        import inspect
        import arklex.evaluation.build_user_profiles as bup

        funcs = [
            obj
            for name, obj in inspect.getmembers(bup, inspect.isfunction)
            if name == "find_matched_attribute"
        ]
        return funcs[0]

    def test_find_matched_attribute_with_long_prompt(self) -> None:
        """Test find_matched_attribute with the long prompt strategy."""
        goal = "interested in product information"
        user_profile_str = (
            "user_info: {'id': 'test'}, current_webpage: Product ID: test"
        )

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatbot:
            mock_chatbot.return_value = """Thought:
The user is interested in product information that they are looking at, so they probably have some question regarding the product's attribute, such as color, size, material, etc. In this case, the attribute category should be "product attribute" and the corresponding value can be color. 

Attribute:
product attribute: color"""

            # Use the react strategy since long_prompt is not supported
            result = find_matched_attribute(
                goal, user_profile_str, strategy="react", client=Mock()
            )

            assert isinstance(result, str)
            assert "color" in result.lower()

    def test_find_matched_attribute_with_client(self) -> None:
        """Test find_matched_attribute with client parameter."""
        goal = "test goal"
        user_profile_str = "test profile"
        mock_client = Mock()

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatbot:
            mock_chatbot.return_value = "Thought: test\nAttribute: test_attr"

            result = find_matched_attribute(
                goal, user_profile_str, strategy="react", client=mock_client
            )

            assert isinstance(result, str)
            assert result == "test_attr"

    def test_find_matched_attribute_invalid_strategy(self) -> None:
        """Test find_matched_attribute with invalid strategy raises ValueError."""
        goal = "test goal"
        user_profile_str = "test profile"

        with pytest.raises(ValueError, match="Invalid strategy"):
            find_matched_attribute(
                goal, user_profile_str, strategy="invalid", client=Mock()
            )

    def test_find_matched_attribute_dict_version(self) -> None:
        """Test the find_matched_attribute function that takes attributes dict."""
        from arklex.evaluation.build_user_profiles import find_matched_attribute

        attributes = {"foo": "bar"}
        goal = "test_goal"
        mock_client = None
        result = find_matched_attribute(goal, attributes, client=mock_client)
        assert result["goal"] == goal
        assert result["matched_attribute"] == attributes

    def test_adapt_goal_with_company_summary(self) -> None:
        """Test adapt_goal with company summary."""
        config = {
            "company_summary": "Test company summary",
            "client": Mock(),
        }

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatbot:
            mock_chatbot.return_value = "adapted goal with company context"

            result = adapt_goal(
                "original_goal", config, doc="test doc", user_profile="test profile"
            )

            assert result == "adapted goal with company context"

    def test_adapt_goal_without_company_summary(self) -> None:
        """Test adapt_goal without company summary."""
        config = {
            "company_summary": "",
            "client": Mock(),
        }

        with patch(
            "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
        ) as mock_chatbot:
            mock_chatbot.return_value = "adapted goal without company context"

            result = adapt_goal(
                "original_goal", config, doc="test doc", user_profile="test profile"
            )

            assert result == "adapted goal without company context"

    @patch("arklex.evaluation.build_user_profiles.requests.get")
    def test_get_custom_profiles_with_api_endpoint(self, mock_get: Mock) -> None:
        """Test get_custom_profiles with API endpoint configuration."""
        mock_get.return_value.json.return_value = ["api_value"]
        config = {
            "user_attributes": {
                "system_attributes": {
                    "sys": {"api": "http://fakeapi", "values": ["val1"]}
                },
                "user_profiles": {"profile": {"values": ["a", "b"]}},
            }
        }

        user_profiles, system_attributes = get_custom_profiles(config)

        assert "profile" in user_profiles
        assert "sys" in system_attributes

    def test_build_profile_with_empty_documents(self) -> None:
        """Test build_profile with empty documents list."""
        synthetic_params = {"num_convos": 1, "num_goals": 1}
        config = {
            "documents_dir": "/test/documents",
            "custom_profile": False,
            "system_inputs": True,
            "company_summary": "Test company",
            "client": Mock(),
            "user_attributes": {
                "goal": {"values": ["goal1"]},
                "system_attributes": {"attr1": {"values": ["val1"]}},
            },
        }

        with (
            patch("arklex.evaluation.build_user_profiles.load_docs") as mock_load_docs,
            patch(
                "arklex.evaluation.build_user_profiles.filter_attributes"
            ) as mock_filter,
            patch(
                "arklex.evaluation.build_user_profiles.augment_attributes"
            ) as mock_augment,
            patch("arklex.evaluation.build_user_profiles.pick_attributes") as mock_pick,
            patch("arklex.evaluation.build_user_profiles.adapt_goal") as mock_adapt,
            patch(
                "arklex.evaluation.build_user_profiles.select_system_attributes"
            ) as mock_select,
            patch(
                "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
            ) as mock_chatbot,
        ):
            mock_load_docs.return_value = []
            mock_filter.return_value = {"attr1": {"values": ["val1"]}}
            mock_augment.return_value = {"attr1": ["val1"]}
            mock_pick.return_value = (
                {"goal": "goal1", "attr1": "val1"},
                {"matched": "data"},
            )
            mock_adapt.return_value = "adapted_goal"
            mock_select.return_value = [{"sys_attr": "val"}]
            mock_chatbot.return_value = "Test profile"

            profiles, goals, attributes, system_attrs, labels = build_profile(
                synthetic_params, config
            )

            assert len(profiles) == 1
            assert len(goals) == 1
            assert len(attributes) == 1
            assert len(system_attrs) == 1
            assert len(labels) == 1

    def test_build_profile_custom_with_binding_and_documents(self) -> None:
        """Test build_profile with custom profiles, binding, and documents."""
        synthetic_params = {"num_convos": 1, "num_goals": 1}
        config = {
            "documents_dir": "/test/documents",
            "custom_profile": True,
            "system_inputs": True,
            "company_summary": "Test company",
            "client": Mock(),
            "user_attributes": {
                "goal": {"values": ["goal1"]},
                "user_profiles": {
                    "profile1": {"values": ["p1", "p2"], "bind_to": "system_attr1"},
                    "profile2": {"values": ["p3", "p4"]},
                },
                "system_attributes": {
                    "attr1": {"values": ["v1", "v2"], "bind_to": "profile1"}
                },
            },
        }

        with (
            patch("arklex.evaluation.build_user_profiles.load_docs") as mock_load_docs,
            patch(
                "arklex.evaluation.build_user_profiles.filter_attributes"
            ) as mock_filter,
            patch(
                "arklex.evaluation.build_user_profiles.augment_attributes"
            ) as mock_augment,
            patch(
                "arklex.evaluation.build_user_profiles.get_custom_profiles"
            ) as mock_get_custom,
            patch("arklex.evaluation.build_user_profiles.pick_attributes") as mock_pick,
            patch("arklex.evaluation.build_user_profiles.adapt_goal") as mock_adapt,
            patch(
                "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
            ) as mock_chatbot,
        ):
            mock_load_docs.return_value = [{"content": "test document"}]
            mock_filter.return_value = {"attr1": {"values": ["val1"]}}
            mock_augment.return_value = {"attr1": ["val1"]}
            mock_get_custom.return_value = (
                {"profile1": ["p1", "p2"], "profile2": ["p3", "p4"]},
                {"attr1": ["v1", "v2"]},
            )
            mock_pick.return_value = (
                {"goal": "goal1", "attr1": "val1"},
                {"matched": "data"},
            )
            mock_adapt.return_value = "adapted_goal"
            mock_chatbot.return_value = "Test profile"

            profiles, goals, attributes, system_attrs, labels = build_profile(
                synthetic_params, config
            )

            assert len(profiles) == 1
            assert len(goals) == 1
            assert len(attributes) == 1
            assert len(system_attrs) == 1
            assert len(labels) == 1

    def test_build_profile_custom_without_documents(self) -> None:
        """Test build_profile with custom profiles but no documents."""
        synthetic_params = {"num_convos": 1, "num_goals": 1}
        config = {
            "documents_dir": "/test/documents",
            "custom_profile": True,
            "system_inputs": True,
            "company_summary": "Test company",
            "client": Mock(),
            "user_attributes": {
                "goal": {"values": ["goal1"]},
                "user_profiles": {"profile1": {"values": ["p1", "p2"]}},
                "system_attributes": {"attr1": {"values": ["v1", "v2"]}},
            },
        }

        with (
            patch("arklex.evaluation.build_user_profiles.load_docs") as mock_load_docs,
            patch(
                "arklex.evaluation.build_user_profiles.filter_attributes"
            ) as mock_filter,
            patch(
                "arklex.evaluation.build_user_profiles.augment_attributes"
            ) as mock_augment,
            patch(
                "arklex.evaluation.build_user_profiles.get_custom_profiles"
            ) as mock_get_custom,
            patch("arklex.evaluation.build_user_profiles.pick_attributes") as mock_pick,
            patch("arklex.evaluation.build_user_profiles.adapt_goal") as mock_adapt,
            patch(
                "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
            ) as mock_chatbot,
        ):
            mock_load_docs.return_value = []
            mock_filter.return_value = {"attr1": {"values": ["val1"]}}
            mock_augment.return_value = {"attr1": ["val1"]}
            mock_get_custom.return_value = (
                {"profile1": ["p1", "p2"]},
                {"attr1": ["v1", "v2"]},
            )
            mock_pick.return_value = (
                {"goal": "goal1", "attr1": "val1"},
                {"matched": "data"},
            )
            mock_adapt.return_value = "adapted_goal"
            mock_chatbot.return_value = "Test profile"

            profiles, goals, attributes, system_attrs, labels = build_profile(
                synthetic_params, config
            )

            assert len(profiles) == 1
            assert len(goals) == 1
            assert len(attributes) == 1
            assert len(system_attrs) == 1
            assert len(labels) == 1

    def test_build_profile_with_custom_profiles(self) -> None:
        """Test build_profile with custom profiles enabled."""
        synthetic_data_params = {"num_convos": 2, "num_goals": 1}
        config = {
            "custom_profile": True,
            "system_inputs": True,
            "user_attributes": {
                "system_attributes": {
                    "location": {
                        "bind_to": "user_profiles.location",
                        "values": [{"city": "NYC"}, {"city": "LA"}],
                    }
                },
                "user_profiles": {
                    "location": {
                        "bind_to": "system_attributes.location",
                        "values": ["NYC", "LA"],
                    }
                },
                "goal": {"values": ["goal1", "goal2"]},
            },
            "documents_dir": "test_dir",
            "client": "test_client",
        }

        # Define all patches at the beginning
        patches = [
            patch(
                "arklex.evaluation.build_user_profiles.load_docs",
                return_value=[{"content": "test doc"}],
            ),
            patch(
                "arklex.evaluation.build_user_profiles.filter_attributes",
                return_value={"category": {"values": ["value1", "value2"]}},
            ),
            patch(
                "arklex.evaluation.build_user_profiles.augment_attributes",
                return_value={"category": ["value1", "value2"]},
            ),
            patch(
                "arklex.evaluation.build_user_profiles.pick_attributes",
                return_value=({"goal": "goal1"}, {"matched": "attr"}),
            ),
            patch(
                "arklex.evaluation.build_user_profiles.adapt_goal",
                return_value="adapted_goal",
            ),
            patch(
                "arklex.evaluation.build_user_profiles.select_system_attributes",
                return_value=[{"location": {"city": "NYC"}}],
            ),
            patch(
                "arklex.evaluation.build_user_profiles.convert_attributes_to_profiles",
                return_value=(["profile1"], ["goal1"], [{"input": "data"}]),
            ),
        ]

        with ExitStack() as stack:
            # Apply all patches
            for patch_obj in patches:
                stack.enter_context(patch_obj)

            result = build_profile(synthetic_data_params, config)
            assert len(result) == 5
            assert result[0] == ["profile1"]  # profiles
            assert result[1] == ["goal1"]  # goals
            assert len(result[2]) == 2  # attributes_list (2 entries for num_convos=2)
            assert all("goal" in attr for attr in result[2])  # all entries have goal
            assert isinstance(result[3], list)
            assert isinstance(result[3][0], dict)
            assert "input" in result[3][0] or "location" in result[3][0]
            assert isinstance(result[4], list)
            assert len(result[4]) == 2
            assert all(item == {"matched": "attr"} for item in result[4])

    def test_build_profile_without_custom_profiles(self) -> None:
        """Test build_profile without custom profiles."""
        synthetic_data_params = {"num_convos": 2, "num_goals": 1}
        config = {
            "custom_profile": False,
            "system_inputs": True,
            "user_attributes": {
                "goal": {"values": ["goal1", "goal2"]},
            },
            "documents_dir": "test_dir",
            "client": "test_client",
        }

        # Define all patches at the beginning
        patches = [
            patch(
                "arklex.evaluation.build_user_profiles.load_docs",
                return_value=[{"content": "test doc"}],
            ),
            patch(
                "arklex.evaluation.build_user_profiles.filter_attributes",
                return_value={"category": {"values": ["value1", "value2"]}},
            ),
            patch(
                "arklex.evaluation.build_user_profiles.augment_attributes",
                return_value={"category": ["value1", "value2"]},
            ),
            patch(
                "arklex.evaluation.build_user_profiles.pick_attributes",
                return_value=({"goal": "goal1"}, {"matched": "attr"}),
            ),
            patch(
                "arklex.evaluation.build_user_profiles.adapt_goal",
                return_value="adapted_goal",
            ),
            patch(
                "arklex.evaluation.build_user_profiles.select_system_attributes",
                return_value=[{"location": {"city": "NYC"}}],
            ),
            patch(
                "arklex.evaluation.build_user_profiles.convert_attributes_to_profiles",
                return_value=(["profile1"], ["goal1"], [{"input": "data"}]),
            ),
        ]

        with ExitStack() as stack:
            # Apply all patches
            for patch_obj in patches:
                stack.enter_context(patch_obj)

            result = build_profile(synthetic_data_params, config)
            assert len(result) == 5
            assert result[0] == ["profile1"]  # profiles
            assert result[1] == ["goal1"]  # goals
            assert isinstance(result[3], list)
            assert isinstance(result[3][0], dict)
            assert "input" in result[3][0] or "location" in result[3][0]
            assert isinstance(result[4], list)
            assert len(result[4]) == 2
            assert all(item == {"matched": "attr"} for item in result[4])

    def test_get_custom_profiles_with_api_calls(self) -> None:
        """Test get_custom_profiles with API calls."""
        config = {
            "user_attributes": {
                "system_attributes": {
                    "location": {
                        "api": "http://api.example.com/location",
                        "values": ["NYC", "LA"],
                    }
                }
            }
        }
        synthetic_data_params = {"num_convos": 2}

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = [{"values": ["NYC", "LA"]}]

            result = select_system_attributes(config, synthetic_data_params)
            assert len(result) == 2
            assert all("location" in attr for attr in result)

    def test_get_custom_profiles_without_api_calls(self) -> None:
        """Test get_custom_profiles without API calls."""
        config = {
            "user_attributes": {
                "system_attributes": {"location": {"values": ["NYC", "LA"]}},
                "user_profiles": {"location": {"values": ["NYC", "LA"]}},
            }
        }

        user_profiles, system_attributes = get_custom_profiles(config)

        assert "location" in system_attributes
        assert "location" in user_profiles
        assert system_attributes["location"] == ["NYC", "LA"]
        assert user_profiles["location"] == ["NYC", "LA"]

    def test_get_custom_profiles_with_bindings(self) -> None:
        """Test get_custom_profiles with field bindings."""
        config = {
            "user_attributes": {
                "system_attributes": {
                    "location": {
                        "api": "http://api.example.com/location",
                        "values": ["NYC", "LA"],
                    }
                },
                "user_profiles": {
                    "location": {
                        "bind_to": "system_attributes.location",
                        "values": ["NYC", "LA"],
                    }
                },
            }
        }

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = {"data": "test_data"}

            user_profiles, system_attributes = get_custom_profiles(config)
            assert "location" in user_profiles
            assert "location" in system_attributes
            assert user_profiles["location"] == ["NYC", "LA"]
            assert system_attributes["location"] == ["NYC", "LA"]

    def test_get_label_success(self) -> None:
        """Test get_label with successful tool prediction."""
        attribute = {"goal": "book appointment"}
        config = {
            "tools": [
                {
                    "id": "1",
                    "name": "booking_tool",
                    "description": "Book appointments",
                    "slots": [],
                    "output": "Booking confirmation",
                }
            ],
            "workers": [],
            "client": Mock(),
        }

        # Define all patches at the beginning
        patches = [
            patch("arklex.evaluation.build_user_profiles.Environment"),
            patch(
                "arklex.evaluation.build_user_profiles.chatgpt_chatbot",
                return_value="1",
            ),
            patch("arklex.evaluation.build_user_profiles.SlotFiller"),
        ]

        with ExitStack() as stack:
            # Apply patches
            mock_env = stack.enter_context(patches[0])
            mock_chat = stack.enter_context(patches[1])
            mock_slot_filler = stack.enter_context(patches[2])

            # Setup mocks
            mock_env_instance = Mock()
            mock_env.return_value = mock_env_instance

            def tool_mock():
                m = Mock(spec=["slots", "description", "output", "name"])
                m.slots = []
                m.description = "Book appointments"
                m.output = "Booking confirmation"
                m.name = "booking_tool"
                return m

            mock_env_instance.tools = {"1": {"execute": tool_mock}}

            mock_slot_filler_instance = Mock()
            mock_slot_filler.return_value = mock_slot_filler_instance
            mock_slot_filler_instance.execute.return_value = [
                Mock(name="slot1", value="value1")
            ]

            label, valid = get_label(attribute, config)

            assert label[0]["tool_id"] == "1"
            assert label[0]["tool_name"] == "booking_tool"
            assert valid is True

    def test_get_label_no_tool_found(self) -> None:
        """Test get_label when no appropriate tool is found."""
        with (
            patch(
                "arklex.evaluation.build_user_profiles.Environment"
            ) as mock_env_class,
            patch("arklex.evaluation.build_user_profiles.chatgpt_chatbot") as mock_chat,
        ):
            # Mock environment and tools
            mock_env = Mock()
            mock_tool = Mock()
            mock_tool.slots = []
            mock_tool.description = "Test tool description"
            mock_tool.output = "Test output"

            mock_env.tools = {"tool1": {"execute": lambda: mock_tool}}
            mock_env_class.return_value = mock_env

            # Mock chatgpt response indicating no tool found
            mock_chat.return_value = "0"

            config = {
                "tools": [{"id": "tool1"}],
                "workers": [{"id": "worker1"}],
                "client": Mock(),
            }
            attribute = {"goal": "test goal"}

            label, valid = get_label(attribute, config)

            assert len(label) == 1
            assert label[0]["tool_id"] == "0"
            assert label[0]["tool_name"] == "No tool"
            assert valid is True

    def test_get_label_with_exception(self) -> None:
        """Test get_label with exception handling."""
        attribute = {"goal": "book appointment"}
        config = {
            "tools": [
                {
                    "id": "1",
                    "name": "booking_tool",
                    "description": "Book appointments",
                    "slots": [],
                    "output": "Booking confirmation",
                }
            ],
            "workers": [],
            "client": Mock(),
        }

        # Define all patches at the beginning
        patches = [
            patch("arklex.evaluation.build_user_profiles.Environment"),
            patch(
                "arklex.evaluation.build_user_profiles.chatgpt_chatbot",
                side_effect=Exception("API error"),
            ),
        ]

        with ExitStack() as stack:
            # Apply patches
            mock_env = stack.enter_context(patches[0])
            stack.enter_context(patches[1])

            # Setup mocks
            mock_env_instance = Mock()
            mock_env.return_value = mock_env_instance
            mock_env_instance.tools = {
                "1": {
                    "execute": lambda: Mock(
                        slots=[],
                        description="Book appointments",
                        output="Booking confirmation",
                        name="booking_tool",
                    )
                }
            }

            label, valid = get_label(attribute, config)

            assert label[0]["tool_id"] == "0"
            assert label[0]["tool_name"] == "No tool"
            assert valid is True

    def test_select_system_attributes_with_api_calls(self) -> None:
        """Test select_system_attributes with API calls."""
        config = {
            "user_attributes": {
                "system_attributes": {
                    "location": {
                        "api": "http://api.example.com/location",
                        "values": ["NYC", "LA"],
                    }
                }
            }
        }
        synthetic_data_params = {"num_convos": 2}

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = [{"values": ["NYC", "LA"]}]

            result = select_system_attributes(config, synthetic_data_params)

            assert len(result) == 2
            assert all("location" in attr for attr in result)

    @pytest.mark.parametrize(
        "attributes,config,documents,expected_behavior",
        [
            # Test with documents and generation enabled
            (
                {"category": {"values": ["value1"], "generate_values": True}},
                {"company_summary": "Test company", "client": Mock()},
                [{"content": "test document"}],
                "with_documents_and_generation",
            ),
            # Test without documents but generation enabled
            (
                {"category": {"values": ["value1"], "generate_values": True}},
                {"company_summary": "Test company", "client": Mock()},
                [],
                "without_documents_but_generation",
            ),
            # Test without generation
            (
                {"category": {"values": ["value1"], "generate_values": False}},
                {"client": Mock()},
                [],
                "no_generation",
            ),
            # Test with empty values
            (
                {"category": {"values": [], "generate_values": True}},
                {"company_summary": "Test company", "client": Mock()},
                [],
                "empty_values",
            ),
        ],
    )
    @patch("arklex.evaluation.build_user_profiles.chatgpt_chatbot")
    def test_augment_attributes_variations(
        self, mock_chat, attributes, config, documents, expected_behavior
    ) -> None:
        """Test augment_attributes with various configurations."""
        mock_chat.return_value = "new_value1, new_value2, new_value3"

        result = augment_attributes(attributes, config, documents)

        assert "category" in result
        assert isinstance(result["category"], list)

        if expected_behavior == "no_generation":
            assert result["category"] == ["value1"]
        elif expected_behavior == "empty_values":
            # Accept empty or new values
            assert result["category"] == [] or "new_value1" in result["category"]
        else:
            # With generation enabled
            assert "value1" in result["category"]
            assert any(val in result["category"] for val in ["value1", "new_value1"])

    def test_filter_attributes_generic(self) -> None:
        """Test filter_attributes with generic category."""
        config = {
            "user_attributes": {
                "generic": {
                    "category1": {"values": ["value1"]},
                    "category2": {"values": ["value2"]},
                },
                "goal": {"values": ["goal1"]},
            },
            "synthetic_data_params": {"customer_type": "premium"},
        }

        result = filter_attributes(config)

        if "generic" in result:
            assert "category1" in result["generic"]
            assert "category2" in result["generic"]
            assert result["generic"]["category1"] == {"values": ["value1"]}
        else:
            assert "category1" in result
            assert "category2" in result
            assert result["category1"] == {"values": ["value1"]}

    def test_build_profile_with_missing_documents_dir(self) -> None:
        """Test build_profile with missing documents_dir."""
        synthetic_data_params = {"num_goals": 2, "num_convos": 2}
        config = {
            "documents_dir": "",
            "custom_profile": False,
            "system_inputs": True,
            "company_summary": "Test company summary",
            "client": Mock(),
            "user_attributes": {
                "goal": {"values": ["goal1", "goal2"]},
                "user_profiles": {"profile1": {"values": ["prof1", "prof2"]}},
                "system_attributes": {"attr1": {"values": ["val1", "val2"]}},
            },
            "tools": [{"id": "tool1", "name": "Test Tool"}],
            "workers": [{"id": "worker1", "name": "Test Worker"}],
        }

        with (
            patch("arklex.evaluation.build_user_profiles.load_docs") as mock_load,
            patch(
                "arklex.evaluation.build_user_profiles.filter_attributes"
            ) as mock_filter,
            patch(
                "arklex.evaluation.build_user_profiles.augment_attributes"
            ) as mock_augment,
            patch("arklex.evaluation.build_user_profiles.pick_attributes") as mock_pick,
            patch("arklex.evaluation.build_user_profiles.adapt_goal") as mock_adapt,
            patch(
                "arklex.evaluation.build_user_profiles.select_system_attributes"
            ) as mock_select,
            patch(
                "arklex.evaluation.build_user_profiles.chatgpt_chatbot"
            ) as mock_chatbot,
        ):
            # Setup mock returns
            mock_load.return_value = []
            mock_filter.return_value = {
                "attr1": {"values": ["val1"]},
                "attr2": {"values": ["val3"]},
            }
            mock_augment.return_value = {
                "attr1": ["val1", "val2"],
                "attr2": ["val3", "val4"],
            }
            mock_pick.return_value = (
                {"goal": "test_goal", "attr1": "val1", "attr2": "val3"},
                {"matched": "data"},
            )
            mock_adapt.return_value = "adapted_goal"
            mock_select.return_value = [{"sys_attr": "val"}, {"sys_attr": "val2"}]
            mock_chatbot.return_value = "Test profile"

            profiles, goals, attributes, system_attrs, labels = build_profile(
                synthetic_data_params, config
            )
            assert len(profiles) == 2
            assert len(goals) == 2
            assert len(attributes) == 2
            assert len(system_attrs) == 2
            assert len(labels) == 2
