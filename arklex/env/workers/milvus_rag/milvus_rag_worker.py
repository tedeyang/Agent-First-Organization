"""Milvus RAG worker implementation for the Arklex framework.

This module provides a specialized worker for handling Retrieval-Augmented Generation (RAG)
tasks using Milvus as the vector database. The MilvusRAGWorker class is responsible for
answering user questions based on internal documentation, supporting both streaming and
non-streaming responses. It integrates with Milvus for efficient similarity search and
retrieval of relevant documents.
"""

from typing import Any

from arklex.env.tools.RAG.retrievers.milvus_retriever import RetrieveEngine
from arklex.env.tools.utils import ToolGenerator, trace
from arklex.env.workers.base.base_worker import BaseWorker
from arklex.env.workers.milvus_rag.entities import (
    MilvusRAGWorkerData,
    MilvusRAGWorkerOutput,
)
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)
from arklex.types.stream_types import StreamType
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class MilvusRAGWorker(BaseWorker):
    description: str = "Answer the user's questions based on the company's internal documentations (unstructured text data), such as the policies, FAQs, and product information"

    def __init__(self) -> None:
        super().__init__()

    def init_worker_data(
        self,
        orch_state: OrchestratorState,
        node_specific_data: dict[str, Any],
    ) -> None:
        self.orch_state = orch_state
        self.milvus_rag_worker_data: MilvusRAGWorkerData = MilvusRAGWorkerData(
            **node_specific_data,
        )

    def _execute(self) -> MilvusRAGWorkerOutput:
        retrieved_text, retriever_params = RetrieveEngine.milvus_retrieve(
            self.orch_state.user_message.history,
            self.orch_state.bot_config,
            self.milvus_rag_worker_data.tags,
        )
        self.orch_state = trace(input=retriever_params, state=self.orch_state)
        self.orch_state.message_flow = retrieved_text
        if self.orch_state.stream_type != StreamType.NON_STREAM:
            response = ToolGenerator.stream_context_generate(self.orch_state)
        else:
            response = ToolGenerator.context_generate(self.orch_state)
        return MilvusRAGWorkerOutput(
            response=response,
            status=StatusEnum.COMPLETE,
        )
