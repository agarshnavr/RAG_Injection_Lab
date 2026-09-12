"""
ingest.py
---------
Loads .txt and .pdf files from a documents folder, splits them into chunks,
embeds them with a local Ollama model, and persists them to a Chroma vector
store on disk.

This is deliberately a standalone module (not folded into app.py) so you can
also run it directly from the command line to (re)build the index without
opening the Streamlit UI:

    python ingest.py

Kept simple on purpose: no metadata enrichment beyond source filename, no
incremental updates (each run wipes and rebuilds the collection). For a
small research-paper document set this is fine and keeps the pipeline easy
to reason about.
"""

import os
import shutil

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# --- Config -----------------------------------------------------------------
DOCS_DIR = "documents"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_lab_docs"

# Embedding model: doesn't need to match the model that answers questions.
# llama3 works fine here and keeps everything local.
EMBEDDING_MODEL = "llama3"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_documents(docs_dir: str = DOCS_DIR):
    """Load every .txt and .pdf file in docs_dir into LangChain Documents."""
    documents = []
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(
            f"'{docs_dir}' does not exist. Create it and drop your .txt/.pdf files in."
        )

    for filename in sorted(os.listdir(docs_dir)):
        filepath = os.path.join(docs_dir, filename)
        if filename.lower().endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())
        elif filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        # silently skip anything else (e.g. .DS_Store, README)

    if not documents:
        raise ValueError(
            f"No .txt or .pdf files found in '{docs_dir}'. Add some and try again."
        )

    return documents


def build_vectorstore(docs_dir: str = DOCS_DIR, persist_dir: str = CHROMA_DIR):
    """
    Full ingestion pipeline: load -> split -> embed -> persist.
    Wipes any existing Chroma collection first, so this is safe to re-run
    whenever your document set changes.
    """
    documents = load_documents(docs_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Start fresh each time to avoid duplicate/stale chunks from old runs.
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )

    print(f"Ingested {len(documents)} file(s) -> {len(chunks)} chunks -> {persist_dir}")
    return vectorstore


def load_vectorstore(persist_dir: str = CHROMA_DIR):
    """Load an already-built Chroma store from disk without re-embedding."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


if __name__ == "__main__":
    build_vectorstore()
