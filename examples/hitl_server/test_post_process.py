# Go to the parent folder of this file (hitl_server), then Run `pytest `test_post_process.py` to test the code in this file.
import io
import json
import os
from contextlib import redirect_stdout
from typing import Any, Dict, List, Tuple

import pytest
from arklex.env.env import Environment
from run import get_api_bot_response

with open("test_cases_post_process.json", encoding="utf-8") as f:
    TEST_CASES = json.load(f)


@pytest.fixture(scope="session")
def config_and_env() -> Tuple[Dict[str, Any], Environment, str]:
    """Load config and environment once per test session."""
    with open("taskgraph.json", encoding="utf-8") as f:
        config = json.load(f)

    model = {
        "model_type_or_path": "gpt-4o-mini",
        "llm_provider": "openai",
    }
    config["input_dir"] = "./"
    config["model"] = model
    os.environ["DATA_DIR"] = config["input_dir"]

    env = Environment(
        tools=config.get("tools", []),
        workers=config.get("workers", []),
        slot_fill_api=config["slotfillapi"],
        planner_enabled=True,
    )

    for node in config["nodes"]:
        if node[1].get("type", "") == "start":
            start_message = node[1]["attribute"]["value"]
            break
    else:
        raise ValueError("Start node not found in taskgraph.json")

    return config, env, start_message


class TestLiveChatDetection:
    def setup_method(self) -> None:
        self.params: Dict[str, Any] = {}

    # run the same test function with multiple sets of arguments
    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=lambda tc: f"id_{tc['test_id']}_{tc['category']}",
    )
    def test_live_chat_detection(
        self,
        test_case: Dict[str, Any],
        config_and_env: Tuple[Dict[str, Any], Environment, str],
    ) -> None:
        config, env, start_message = config_and_env
        user_text = test_case["user_utterance"]
        expected_live_chat = test_case["expect_live_chat"]

        history: List[Dict[str, str]] = [
            {"role": "assistant", "content": start_message}
        ]

        captured_logs = io.StringIO()
        with redirect_stdout(captured_logs):
            output, self.params, hitl = get_api_bot_response(
                config=config,
                history=history,
                user_text=user_text,
                parameters=self.params,
                env=env,
            )

        live_chat_triggered = hitl is not None
        logs_output = captured_logs.getvalue()

        assert live_chat_triggered == expected_live_chat, (
            f"\nTest Failed\nExpected live chat: {expected_live_chat}, got: {live_chat_triggered}\n"
            f"User: {user_text}\nBot: {output}"
        )

        if expected_live_chat:
            assert (
                "* Live Chat Initiated: Bot was uncertain about the response."
                in logs_output
            )
            assert "* Original Bot Response:" in logs_output
