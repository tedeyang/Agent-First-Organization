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
        pass

    def generator(self, state: MessageState) -> MessageState:
        pass

    def _create_action_graph(self) -> StateGraph:
        workflow = StateGraph(MessageState)
        # Add nodes for each worker

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

