import logging
import re

from arklex.env.prompts import load_prompts
from arklex.utils.graph_state import MessageState, ResourceRecord
from arklex.utils.model_provider_config import PROVIDER_MAP

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


def post_process_response(message_state: MessageState) -> MessageState:
    context = build_context(message_state)
    answer_links = _extract_links(message_state.response)
    if answer_links:
        context_links = _extract_links(context)
        if not answer_links.issubset(context_links):
            missing_links = answer_links - context_links
            logger.info(
                f"Some answer links are NOT present in the context. Missing: {missing_links}"
            )
            message_state.response = _remove_invalid_links(
                message_state.response, missing_links
            )
            message_state.response = _rephrase_answer(message_state)

    return message_state


def build_context(message_state: MessageState):
    context = message_state.sys_instruct
    for resource_group in message_state.trajectory:
        for resource in resource_group:
            if _include_resource(resource):
                context += resource.output
    return context


def _include_resource(resource: ResourceRecord):
    """Determines whether a ResourceRecord's output should be included in context.

    Excludes any output where a 'context_generate' flag is present in steps.
    """
    return not any(step.get("context_generate") for step in resource.steps)


def _extract_links(text: str) -> set:
    url_pattern = r"(?:https?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    links = re.findall(url_pattern, text)
    return {link.rstrip(".,;)!?\"'") for link in links}


def _remove_invalid_links(text: str, links: set) -> str:
    sorted_links = sorted([re.escape(link)
                          for link in links], key=len, reverse=True)
    links_regex = "|".join(sorted_links)
    cleaned_text = re.sub(links_regex, "", text)
    return re.sub(r"\s+", " ", cleaned_text).strip()


def _rephrase_answer(state: MessageState) -> str:
    """Rephrases the answer using an LLM after link removal."""
    llm_config = state.bot_config.llm_config
    llm = PROVIDER_MAP.get(llm_config.llm_provider, ChatOpenAI)(
        model=llm_config.model_type_or_path, temperature=0.1
    )
    prompt: PromptTemplate = PromptTemplate.from_template(
        load_prompts(state.bot_config)["regenerate_response"]
    )
    input_prompt = prompt.invoke(
        {
            "sys_instruct": state.sys_instruct,
            "original_answer": state.response,
            "formatted_chat": state.user_message.history,
        }
    )
    final_chain = llm | StrOutputParser()
    logger.info(f"Prompt: {input_prompt.text}")
    answer: str = final_chain.invoke(input_prompt.text)
    return answer
