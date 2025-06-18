import json
import logging
import traceback
from functools import partial
from typing import Any, Dict, Optional

from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph

from arklex.env.agents.agent import BaseAgent, register_agent
from arklex.env.prompts import load_prompts
from arklex.env.tools.tools import register_tool
from arklex.env.tools.utils import trace
from arklex.types import EventType, StreamType
from arklex.utils.graph_state import BotConfig, MessageState, StatusEnum
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP
from arklex.utils.utils import chunk_string

logger = logging.getLogger(__name__)


@register_tool("Ends the conversation with a thank you message.", isResponse=True)
def end_conversation(state: MessageState) -> MessageState:
    logger.info("Ending the conversation.")
    state.response = "Thank you for using Arklex. Goodbye!"
    return state


@register_agent
class OpenAIAgent(BaseAgent):
    description: str = "General-purpose Arklex agent for chat or voice."

    def __init__(
        self, successors: list, predecessors: list, tools: list, state: MessageState
    ) -> None:
        super().__init__()
        self.action_graph: StateGraph = self._create_action_graph()
        self.llm: Optional[BaseChatModel] = None

        self.available_tools: dict[str, Any] = {}

        for node in successors + predecessors:
            if node.resource_id in tools:
                self.available_tools[node.resource_id] = tools[node.resource_id]

        self.tool_map = {}
        self.tool_defs = []

        for tool_id, tool in self.available_tools.items():
            tool = tool["execute"]()
            tool_def = tool.to_openai_tool_def()
            self.tool_defs.append(tool_def)
            self.tool_map[tool_def["function"]["name"]] = tool.func

        end_conversation_tool = end_conversation()
        end_conversation_tool_def = end_conversation_tool.to_openai_tool_def()
        end_conversation_tool_def["function"]["name"] = "end_conversation"
        self.tool_defs.append(end_conversation_tool_def)
        self.tool_map["end_conversation"] = end_conversation_tool.func

        self.messages: list = []
        prompts: Dict[str, str] = load_prompts(state.bot_config)
        orchestrator_message = state.orchestrator_message
        orch_msg_content: str = (
            "None" if not orchestrator_message.message else orchestrator_message.message
        )
        prompt: PromptTemplate = PromptTemplate.from_template(
            prompts["function_calling_agent_prompt"]
        )
        input_prompt = prompt.invoke(
            {
                "sys_instruct": state.sys_instruct,
                "message": orch_msg_content,
            }
        )
        self.messages.append(
            SystemMessage(
                content=input_prompt.text,
            )
        )

        logger.info(f"OpenAIAgent initialized with {len(self.tool_defs)} tools.")

    def generate(self, state: MessageState) -> MessageState:
        logger.info("\nGenerating response using the agent.")

        logger.info(f"\n\n\nmessages: {self.messages}")

        orchestrator_message = state.orchestrator_message
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
        prompt: PromptTemplate = PromptTemplate.from_template(
            prompts["function_calling_agent_prompt"]
        )
        input_prompt = prompt.invoke(
            {
                "sys_instruct": state.sys_instruct,
                "message": orch_msg_content,
            }
        )
        self.message.append(
            SystemMessage(
                content=input_prompt.text,
            )
        )

        user_message = state.user_message
        self.message.append(
            HumanMessage(
                content=user_message.message,
            )
        )

        final_chain = self.llm
        ai_message: AIMessage = final_chain.invoke(self.messages)

        logger.info(f"Generated answer: {ai_message}")

        if ai_message.tool_calls:
            logger.info("Processing tool calls.")
            for tool_call in ai_message.tool_calls:
                tool_name = tool_call.get("name")
                if tool_name in self.tool_map:
                    tool_response = self.tool_map[tool_name](
                        state=state, **tool_call.get("args")
                    )
                    self.messages.append(
                        ToolMessage(
                            name=tool_name,
                            content=json.dumps(tool_response),
                            additional_kwargs={"tool_call_id": tool_call.id},
                        )
                    )
                    ai_message = final_chain.invoke(self.messages)
                    state.response = ai_message.content
                else:
                    logger.warning(f"Tool {tool_name} not found in tool map.")
        else:
            state.message_flow = ""
            state.response = ai_message.content
            state = trace(input=ai_message.content, state=state)
        return state

    def _create_action_graph(self) -> StateGraph:
        workflow = StateGraph(MessageState)
        workflow.add_node("generate", self.generate)
        workflow.add_edge(START, "generate")

        return workflow

    def _execute(self, msg_state: MessageState, **kwargs: Any) -> Dict[str, Any]:
        self.llm = PROVIDER_MAP.get(
            msg_state.bot_config.llm_config.llm_provider, ChatOpenAI
        )(model=msg_state.bot_config.llm_config.model_type_or_path)
        self.llm = self.llm.bind_tools(self.tool_defs)
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
