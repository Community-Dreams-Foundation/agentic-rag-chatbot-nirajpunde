"""RAG chain: retriever -> format context -> LLM -> grounded answer with citations."""

import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .retrieval import get_retriever, format_context_with_citations


RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context from uploaded documents.

Rules:
- Answer using ONLY the information in the context. Do not use external knowledge.
- If the context does not contain enough information to answer, say: "I cannot find this in the uploaded documents."
- When you use information from the context, mention the source (e.g., "According to [source]...").
- Never invent or hallucinate information. Never cite sources that are not in the context.
- Be concise and accurate.
"""

RAG_USER_TEMPLATE = """Context from documents:

{context}

---

User question: {question}

Answer (grounded in the context above, or refuse if not found):"""


def get_llm():
    """Create Gemini chat model."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )


def query_rag(question: str, k: int = 2) -> tuple[str, list[dict]]:
    """
    Run RAG: retrieve docs, generate answer, return (answer, citations).
    Citations are derived from retrieved docs, not parsed from LLM output.
    """
    retriever = get_retriever(k=k)
    docs = retriever.invoke(question)

    if not docs:
        return (
            "I cannot find this in the uploaded documents.",
            [],
        )

    context_str, citations = format_context_with_citations(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", RAG_USER_TEMPLATE),
    ])

    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke({"context": context_str, "question": question})

    answer = response.content if hasattr(response, "content") else str(response)
    return answer.strip(), citations
