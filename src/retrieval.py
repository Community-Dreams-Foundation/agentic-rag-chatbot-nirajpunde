"""Retriever setup and citation extraction helpers."""

from .ingestion import load_vector_store


def get_retriever(k: int = 5):
    """Get a retriever from the FAISS store."""
    vectorstore = load_vector_store()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def doc_to_citation(doc) -> dict:
    """Convert a retrieved document to citation format for sanity_output.json."""
    source = doc.metadata.get("source", "unknown")
    locator = doc.metadata.get("locator") or doc.metadata.get("chunk_id", "unknown")
    if isinstance(locator, str) and not locator.startswith("chunk"):
        locator = f"chunk {locator}"
    snippet = doc.page_content[:500].strip()
    if len(doc.page_content) > 500:
        snippet += "..."
    return {
        "source": source,
        "locator": str(locator),
        "snippet": snippet,
    }


def format_context_with_citations(docs: list) -> tuple[str, list[dict]]:
    """Format retrieved docs as context string and extract citations."""
    context_parts = []
    citations = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        locator = doc.metadata.get("locator") or doc.metadata.get("chunk_id", str(i + 1))
        context_parts.append(f"[{i + 1}] (Source: {source}, {locator})\n{doc.page_content}")
        citations.append(doc_to_citation(doc))
    return "\n\n---\n\n".join(context_parts), citations
