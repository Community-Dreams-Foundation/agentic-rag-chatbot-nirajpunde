"""Ingestion pipeline: load .txt files, chunk, embed, and index in FAISS."""

import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Use LOCALAPPDATA to avoid OneDrive sync locks (WinError 5 Access denied)
_APP_DATA = Path(os.environ.get("LOCALAPPDATA", os.environ.get("TEMP", ".")))
FAISS_INDEX_DIR = _APP_DATA / "agentic_rag_chatbot" / "faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create Gemini embeddings. Requires GOOGLE_API_KEY env var."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def load_txt_documents(docs_dir: Path) -> list:
    """Load all .txt files from a directory."""
    docs = []
    if not docs_dir.exists():
        return docs
    for path in docs_dir.glob("*.txt"):
        loader = TextLoader(str(path), encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = path.name
        docs.extend(loaded)
    return docs


def chunk_documents(documents: list) -> list:
    """Split documents into chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(i + 1)
        if "locator" not in chunk.metadata:
            chunk.metadata["locator"] = f"chunk {i + 1}"
    return chunks


def create_vector_store(chunks: list, index_dir: Path | str = None) -> FAISS:
    """Create FAISS vector store from chunks and save to disk."""
    index_path = str(index_dir or FAISS_INDEX_DIR)
    embeddings = get_embeddings()
    store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    store.save_local(index_path)
    return store


def get_or_create_vector_store(docs_dir: Path = None, force_recreate: bool = False) -> FAISS:
    """Load docs, chunk, index, and return FAISS store."""
    docs_path = docs_dir or Path(__file__).resolve().parent.parent / "sample_docs"
    documents = load_txt_documents(docs_path)
    if not documents:
        raise ValueError(f"No .txt files found in {docs_path}")

    chunks = chunk_documents(documents)
    if force_recreate and FAISS_INDEX_DIR.exists():
        shutil.rmtree(FAISS_INDEX_DIR)

    return create_vector_store(chunks, FAISS_INDEX_DIR)


def load_vector_store() -> FAISS:
    """Load existing FAISS store from disk. Auto-index if index does not exist."""
    index_file = FAISS_INDEX_DIR / "index.faiss"
    if not index_file.exists():
        return get_or_create_vector_store()
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
