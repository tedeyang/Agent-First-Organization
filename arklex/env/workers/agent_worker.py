import logging
import traceback
from typing import Any, Dict, Optional

from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph

from arklex.env.prompts import load_prompts
from arklex.env.tools.utils import trace
from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.types import EventType, StreamType
from arklex.utils.graph_state import MessageState, StatusEnum
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP
from arklex.utils.utils import chunk_string

logger = logging.getLogger(__name__)


@register_worker
class AgentWorker(BaseWorker):
    description: str = "General-purpose Arklex agent worker for chat or voice."

    def __init__(self) -> None:
        super().__init__()
        self.action_graph: StateGraph = self._create_action_graph()
        self.llm: Optional[BaseChatModel] = None
        self.actions = {
            "generate": "Answer a question or generate a response based on the user input.",
            "end_conversation": "End the conversation",
        }

    def verify_action(self, state: MessageState) -> str:
        agent_task: str = state.orchestrator_message.attribute.get("task", "")
        actions_info: str = "\n".join(
            [f"{name}: {description}" for name, description in self.actions.items()]
        )
        actions_name: str = ", ".join(self.actions.keys())
        logger.info(state)
        personalized_intent: str = state.trajectory[-1][-1].personalized_intent

        prompts: Dict[str, str] = load_prompts(state.bot_config)
        prompt: PromptTemplate = PromptTemplate.from_template(
            prompts["agent_action_prompt"]
        )
        input_prompt = prompt.invoke(
            {
                "agent_task": agent_task,
                "actions_info": actions_info,
                "actions_name": actions_name,
                "formatted_chat": state.user_message.history,
            }
        )
        chunked_prompt: str = chunk_string(
            input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"]
        )
        logger.info(f"Chunked prompt for deciding choosing DB action: {chunked_prompt}")
        final_chain = self.llm | StrOutputParser()
        try:
            answer: str = final_chain.invoke(chunked_prompt)
            for action_name in self.actions.keys():
                if action_name in answer:
                    logger.info(f"Chosen action in the database worker: {action_name}")
                    return action_name
            logger.info(f"Base action chosen in the database worker: Others")
            return "generator"
        except Exception as e:
            logger.error(
                f"Error occurred while choosing action in the database worker: {e}"
            )
            return "generator"

    def end_conversation(self, state: MessageState) -> MessageState:
        logger.info("Ending the conversation.")
        state.response = "Thank you for using Arklex. Goodbye!"
        state.status = StatusEnum.COMPLETE
        state.message_flow = ""
        return state

    def generator(self, state: MessageState) -> MessageState:
        logger.info("\nGenerating response using the agent worker.")
        user_message = state.user_message
        orchestrator_message = state.orchestrator_message
        message_flow: str = state.response + "\n" + state.message_flow

        orch_msg_content: str = (
            "None" if not orchestrator_message.message else orchestrator_message.message
        )
        orch_msg_attr: Dict[str, Any] = orchestrator_message.attribute
        direct_response: bool = orch_msg_attr.get("direct_response", False)
        if direct_response:
            state.message_flow = ""
            state.response = orch_msg_content
            return state

        prompts: Dict[str, str] = load_prompts(state.bot_config)
        if message_flow and message_flow != "\n":
            prompt: PromptTemplate = PromptTemplate.from_template(
                prompts["message_flow_generator_prompt"]
            )
            input_prompt = prompt.invoke(
                {
                    "sys_instruct": state.sys_instruct,
                    "message": orch_msg_content,
                    "formatted_chat": user_message.history,
                    "context": message_flow,
                }
            )
        else:
            prompt: PromptTemplate = PromptTemplate.from_template(
                prompts["message_generator_prompt"]
            )
            input_prompt = prompt.invoke(
                {
                    "sys_instruct": state.sys_instruct,
                    "message": orch_msg_content,
                    "formatted_chat": user_message.history,
                }
            )
        logger.info(f"Prompt: {input_prompt.text}")
        final_chain = self.llm | StrOutputParser()
        answer: str = final_chain.invoke(input_prompt.text)

        state.message_flow = ""
        state.response = answer
        state = trace(input=answer, state=state)
        return state

    def _create_action_graph(self) -> StateGraph:
        workflow = StateGraph(MessageState)
        workflow.add_node("generator", self.generator)
        workflow.add_node("end_conversation", self.end_conversation)
        workflow.add_conditional_edges(START, self.verify_action)

        return workflow

    def _execute(self, msg_state: MessageState, **kwargs: Any) -> Dict[str, Any]:
        self.llm = PROVIDER_MAP.get(
            msg_state.bot_config.llm_config.llm_provider, ChatOpenAI
        )(model=msg_state.bot_config.llm_config.model_type_or_path)
        graph = self.action_graph.compile()
        result: Dict[str, Any] = graph.invoke(msg_state)
        return result

    def execute(self, msg_state: MessageState, **kwargs: Any) -> MessageState:
        try:
            response_return: Dict[str, Any] = self._execute(msg_state, **kwargs)
            response_state: MessageState = MessageState.model_validate(response_return)
            response_state.trajectory[-1][-1].output = (
                response_state.response
                if response_state.response
                else response_state.message_flow
            )
            return response_state
        except Exception as e:
            logger.error(traceback.format_exc())
            msg_state.status = StatusEnum.INCOMPLETE
            return msg_state
