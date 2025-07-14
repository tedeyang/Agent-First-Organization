"""Utility tools for the Arklex framework.

This module provides utility tools and helper functions for the Arklex framework,
including the ToolGenerator class for generating responses and handling streaming
outputs. It also includes functions for tracing execution flow and managing message
states. The module integrates with various language models and prompt templates to
provide flexible response generation capabilities.
"""

import inspect
import json
from typing import (
    Any,
    Protocol,
    TypedDict,
)

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from arklex.env.prompts import load_prompts
from arklex.orchestrator.entities.msg_state_entities import MessageState
from arklex.types import EventType, StreamType
from arklex.utils.exceptions import ToolError
from arklex.utils.logging_utils import LogContext
from arklex.utils.provider_utils import validate_and_get_model_class

log_context = LogContext(__name__)


class ExecuteToolKwargs(TypedDict, total=False):
    """Type definition for kwargs used in execute_tool function."""

    # Add specific tool parameters as needed
    pass


class ToolExecutor(Protocol):
    """Protocol for objects that can execute tools."""

    tools: dict[str, Any]


def get_prompt_template(state: MessageState, prompt_key: str) -> PromptTemplate:
    """Get the prompt template based on the stream type."""
    prompts: dict[str, str] = load_prompts(state.bot_config)

    if state.stream_type == StreamType.SPEECH:
        # Use speech prompts, but fall back to regular prompts for Chinese
        # since Chinese speech prompts are not available yet
        if state.bot_config.language == "CN":
            return PromptTemplate.from_template(prompts[prompt_key])
        else:
            return PromptTemplate.from_template(prompts[prompt_key + "_speech"])
    else:
        return PromptTemplate.from_template(prompts[prompt_key])


class ToolGenerator:
    @staticmethod
    def generate(state: MessageState) -> MessageState:
        llm_config: dict[str, Any] = state.bot_config.llm_config
        user_message: Any = state.user_message

        model_class = validate_and_get_model_class(llm_config)

        llm: Any = model_class(model=llm_config.model_type_or_path, temperature=0.1)
        prompt: PromptTemplate = get_prompt_template(state, "generator_prompt")
        input_prompt: Any = prompt.invoke(
            {"sys_instruct": state.sys_instruct, "formatted_chat": user_message.history}
        )
        log_context.info(f"Prompt: {input_prompt.text}")
        final_chain: Any = llm | StrOutputParser()
        answer: str = final_chain.invoke(input_prompt.text)

        state.response = answer
        return state

    @staticmethod
    def context_generate(state: MessageState) -> MessageState:
        llm_config: dict[str, Any] = state.bot_config.llm_config

        model_class = validate_and_get_model_class(llm_config)

        llm: Any = model_class(model=llm_config.model_type_or_path, temperature=0.1)
        # get the input message
        user_message: Any = state.user_message
        message_flow: str = state.message_flow

        # Add relevant records to context if available
        if state.relevant_records:
            relevant_context: str = "\nRelevant past interactions:\n"
            for record in state.relevant_records:
                relevant_context += "Record:\n"
                if record.info:
                    relevant_context += f"- Info: {record.info}\n"
                if record.personalized_intent:
                    relevant_context += (
                        f"- Personalized User Intent: {record.personalized_intent}\n"
                    )
                if record.output:
                    relevant_context += f"- Raw Output: {record.output}\n"
                if record.steps:
                    relevant_context += "- Intermediate Steps:\n"
                    for step in record.steps:
                        if isinstance(step, dict):
                            for key, value in step.items():
                                relevant_context += f"  * {key}: {value}\n"
                        else:
                            relevant_context += f"  * {step}\n"
                relevant_context += "\n"
            message_flow = relevant_context + "\n" + message_flow

        log_context.info(
            f"Retrieved texts (from retriever/search engine to generator): {message_flow[:50]} ..."
        )

        # generate answer based on the retrieved texts
        prompt: PromptTemplate = get_prompt_template(state, "context_generator_prompt")
        input_prompt: Any = prompt.invoke(
            {
                "sys_instruct": state.sys_instruct,
                "formatted_chat": user_message.history,
                "context": message_flow,
            }
        )
        final_chain: Any = llm | StrOutputParser()
        log_context.info(f"Prompt: {input_prompt.text}")
        answer: str = final_chain.invoke(input_prompt.text)
        state.message_flow = ""
        state.response = answer
        state = trace(input=answer, state=state)
        return state

    @staticmethod
    def stream_context_generate(state: MessageState) -> MessageState:
        llm_config: dict[str, Any] = state.bot_config.llm_config

        model_class = validate_and_get_model_class(llm_config)

        llm: Any = model_class(model=llm_config.model_type_or_path, temperature=0.1)
        # get the input message
        user_message: Any = state.user_message
        message_flow: str = state.message_flow
        # Add relevant records to context if available
        if state.relevant_records:
            relevant_context: str = "\nRelevant past interactions:\n"
            for record in state.relevant_records:
                relevant_context += "Record:\n"
                if record.info:
                    relevant_context += f"- Info: {record.info}\n"
                if record.personalized_intent:
                    relevant_context += (
                        f"- Personalized User Intent: {record.personalized_intent}\n"
                    )
                if record.output:
                    relevant_context += f"- Raw Output: {record.output}\n"
                if record.steps:
                    relevant_context += "- Intermediate Steps:\n"
                    for step in record.steps:
                        if isinstance(step, dict):
                            for key, value in step.items():
                                relevant_context += f"  * {key}: {value}\n"
                        else:
                            relevant_context += f"  * {step}\n"
                relevant_context += "\n"
            message_flow = relevant_context + "\n" + message_flow
        log_context.info(
            f"Retrieved texts (from retriever/search engine to generator): {message_flow[:50]} ..."
        )

        # generate answer based on the retrieved texts
        prompt: PromptTemplate = get_prompt_template(state, "context_generator_prompt")

        input_prompt: Any = prompt.invoke(
            {
                "sys_instruct": state.sys_instruct,
                "formatted_chat": user_message.history,
                "context": message_flow,
            }
        )
        final_chain: Any = llm | StrOutputParser()
        log_context.info(f"Prompt: {input_prompt.text}")
        answer: str = ""
        for chunk in final_chain.stream(input_prompt.text):
            answer += chunk
            state.message_queue.put(
                {"event": EventType.CHUNK.value, "message_chunk": chunk}
            )

        state.message_flow = ""
        state.response = answer
        state = trace(input=answer, state=state)
        return state

    @staticmethod
    def stream_generate(state: MessageState) -> MessageState:
        user_message: Any = state.user_message

        llm_config: dict[str, Any] = state.bot_config.llm_config

        model_class = validate_and_get_model_class(llm_config)

        llm: Any = model_class(model=llm_config.model_type_or_path, temperature=0.1)
        prompt: PromptTemplate = get_prompt_template(state, "generator_prompt")
        input_prompt: Any = prompt.invoke(
            {"sys_instruct": state.sys_instruct, "formatted_chat": user_message.history}
        )
        final_chain: Any = llm | StrOutputParser()
        answer: str = ""
        for chunk in final_chain.stream(input_prompt.text):
            answer += chunk
            state.message_queue.put(
                {"event": EventType.CHUNK.value, "message_chunk": chunk}
            )

        state.response = answer
        return state


def trace(input: str, state: MessageState) -> MessageState:
    current_frame: inspect.FrameInfo | None = inspect.currentframe()
    previous_frame: inspect.FrameInfo | None = (
        current_frame.f_back if current_frame else None
    )
    previous_function_name: str = (
        previous_frame.f_code.co_name if previous_frame else "unknown"
    )
    response_meta: dict[str, str] = {previous_function_name: input}
    state.trajectory[-1][-1].steps.append(response_meta)
    return state


def execute_tool(
    self: ToolExecutor, tool_name: str, **kwargs: ExecuteToolKwargs
) -> str:
    """Execute a tool.

    Args:
        self: The object instance containing tools
        tool_name: Name of the tool to execute
        **kwargs: Additional arguments for the tool

    Returns:
        Tool execution result

    Raises:
        ToolError: If tool execution fails
    """
    try:
        tool = self.tools[tool_name]
        log_context.info(f"Executing tool: {tool_name}")
        result = tool.execute(**kwargs)
        log_context.info(f"Tool execution completed: {tool_name}")
        return result
    except Exception as e:
        log_context.error(f"Tool execution failed: {tool_name}, error: {e}")
        raise ToolError(f"Tool execution failed: {tool_name}") from e


def generate_multi_slot_cohesive_response(
    raw_data: str, llm_config: dict[str, Any]
) -> str:
    # combine raw data into a singular card list
    json_objects = raw_data.strip().split("\n")
    card_list = []

    for obj in json_objects:
        try:
            data = json.loads(obj)
            if "card_list" in data and isinstance(data["card_list"], list):
                card_list.extend(data["card_list"])
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON object: {e}")
    card_list = dedupe_ordered(card_list)
    llm = PROVIDER_MAP.get(llm_config["llm_provider"], ChatOpenAI)(
        model=llm_config["model_type_or_path"], temperature=0.7
    )
    message = [
        {
            "role": "user",
            "content": f"You are helping a customer search products based on the query and get results below and those results will be presented using product card format.\n\n{json.dumps(card_list)}\n\nGenerate a response to continue the conversation without explicitly mentioning contents of the search result. Include one or two questions about those products to know the user's preference. Keep the response within 50 words.\nDIRECTLY GIVE THE RESPONSE.",
        },
    ]
    answer = llm.invoke(message).content
    return json.dumps({"answer": answer, "card_list": card_list})


def dedupe_ordered(items: list) -> list:
    seen = set()
    unique = []
    for item in items:
        key = json.dumps(item, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
