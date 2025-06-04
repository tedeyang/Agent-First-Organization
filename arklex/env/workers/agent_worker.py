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

    def verify_action(self, state: MessageState) -> str:
        pass

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
