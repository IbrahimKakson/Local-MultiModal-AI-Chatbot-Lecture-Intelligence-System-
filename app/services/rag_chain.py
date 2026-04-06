"""RAG chain composition using LangChain orchestration.

Week 8: This module wires together the Hybrid Search retriever (Week 6),
the local Mistral LLM engine (Week 7), and LangChain's LCEL pipeline
with conversational memory to create a full RAG pipeline.
"""
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.services.vector_store import VectorStoreService
from app.services.search_service import SearchService
from app.services.llm_engine import load_model, generate_answer_from_model
from app.core.prompts import MISTRAL_RAG_TEMPLATE


# ---------------------------------------------------------------------------
# 1. LangChain Retriever Wrapper (wraps our custom SearchService)
# ---------------------------------------------------------------------------
class HybridRetriever(BaseRetriever):
    """Adapts our custom SearchService into a LangChain-compatible retriever."""
    search_service: Any = None
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """LangChain calls this method to fetch documents for a query."""
        results = self.search_service.hybrid_search(query, top_k=self.top_k)
        return [
            Document(
                page_content=r.get("text", ""),
                metadata=r.get("metadata", {})
            )
            for r in results
        ]


# ---------------------------------------------------------------------------
# 2. LangChain LLM Wrapper (wraps our custom llm_engine)
# ---------------------------------------------------------------------------
class LocalMistralLLM(BaseLLM):
    """Adapts our raw llm_engine into a LangChain-compatible LLM."""

    @property
    def _llm_type(self) -> str:
        return "local-mistral"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """LangChain calls this method to generate text."""
        llm = load_model()
        response = llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            stop=stop or ["[/INST]", "user:", "User:"],
        )
        return response["choices"][0]["text"].strip()

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        """Required by BaseLLM. Delegates to _call for each prompt."""
        from langchain_core.outputs import Generation, LLMResult
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)


# ---------------------------------------------------------------------------
# 3. RAG Chain Builder (LCEL - Modern LangChain)
# ---------------------------------------------------------------------------
class RAGChain:
    """Orchestrates the full Retrieval-Augmented Generation pipeline.

    Connects:
        - HybridRetriever (Vector + Keyword search from Week 6)
        - LocalMistralLLM (CPU inference from Week 7)
        - Chat history (simple in-memory list)
    """

    def __init__(self, top_k: int = 5, memory_window: int = 4):
        # Initialize dependencies
        vector_store = VectorStoreService()
        search_service = SearchService(vector_store)

        # Wrap into LangChain components
        self.retriever = HybridRetriever(
            search_service=search_service, top_k=top_k
        )
        self.llm = LocalMistralLLM()

        # Simple in-memory chat history
        self.chat_history: list[dict] = []
        self.memory_window = memory_window

        # Build the LCEL chain: retriever -> prompt -> llm -> parse
        self.prompt = PromptTemplate(
            template=MISTRAL_RAG_TEMPLATE,
            input_variables=["context", "question"],
        )
        self.output_parser = StrOutputParser()

    def _format_docs(self, docs: List[Document]) -> str:
        """Join retrieved documents into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_question_with_history(self, question: str) -> str:
        """Prepend recent chat history to the current question."""
        if not self.chat_history:
            return question

        # Take only the last N exchanges
        recent = self.chat_history[-self.memory_window:]
        history_str = "\n".join(
            f"User: {turn['question']}\nAssistant: {turn['answer']}"
            for turn in recent
        )
        return f"Previous conversation:\n{history_str}\n\nCurrent question: {question}"

    def ask(self, question: str) -> dict:
        """Ask a question and get an answer with source documents.

        Returns:
            dict with keys: 'answer' (str), 'source_documents' (list[Document])
        """
        # Enrich question with chat history for context
        enriched_question = self._build_question_with_history(question)

        # Step 1: Retrieve relevant documents
        source_docs = self.retriever.invoke(enriched_question)

        # Step 2: Format context
        context = self._format_docs(source_docs)

        # Step 3: Build prompt and generate answer
        formatted_prompt = self.prompt.format(
            context=context, question=question
        )
        answer = self.llm.invoke(formatted_prompt)

        # Step 4: Save to memory
        self.chat_history.append({
            "question": question,
            "answer": answer,
        })

        return {
            "answer": answer,
            "source_documents": source_docs,
        }

    def clear_memory(self):
        """Clears the conversation history (e.g., on new session)."""
        self.chat_history.clear()


# ---------------------------------------------------------------------------
# 4. Backward-compatible function (keeps old tests working)
# ---------------------------------------------------------------------------
def run_rag(query: str, top_k: int = 5) -> str:
    """Simple wrapper that maintains backward compatibility."""
    return generate_answer_from_model(query, context=None)
