"""
app.py
------
Stage 1: basic ingestion + retrieval + Streamlit chat UI, UNDEFENDED.

Run with:
    streamlit run app.py

On first load, click "Build / rebuild index" in the sidebar after dropping
your .txt/.pdf files into the documents/ folder. After that the index
persists on disk (chroma_db/) so you don't need to rebuild every run.
"""

import os
import time

import streamlit as st
from langchain_ollama import ChatOllama

from ingest import build_vectorstore, load_vectorstore, CHROMA_DIR, DOCS_DIR
from agent_action import TRIGGER_INSTRUCTION, strip_trigger, draft_email, save_email
# from theme import inject_custom_css  # disabled — reverted to default styling

# --- Config -------------------------------------------------------------
AVAILABLE_MODELS = ["llama3", "mistral"]
RETRIEVAL_K = 4  # number of chunks to retrieve per query

# Plain, undefended prompt template. This is intentionally naive — Stage 3
# will add hardened variants behind toggles. Keeping this version around
# lets you always reproduce the "undefended" baseline for comparison.
UNDEFENDED_PROMPT = """Answer the question using the context below.

Context:
{context}

Question: {question}

Answer:"""


# --- Helpers --------------------------------------------------------------

def format_context(retrieved_docs):
    """Join retrieved chunks into a single context string, plain (no defenses)."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def answer_query(query: str, model_name: str, vectorstore):
    """
    Runs one retrieval + generation pass.
    Returns (clean_answer, retrieved_docs, latency_ms, triggered, context).
    'triggered' indicates whether the model's raw response asked for the
    agentic email action (see agent_action.py) — already stripped out of
    clean_answer by this point.
    """
    start = time.time()

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    retrieved_docs = retriever.invoke(query)

    context = format_context(retrieved_docs)
    prompt = UNDEFENDED_PROMPT.format(context=context, question=query) + TRIGGER_INSTRUCTION

    llm = ChatOllama(model=model_name)
    raw_response = llm.invoke(prompt)

    clean_answer, triggered = strip_trigger(raw_response.content)

    latency_ms = round((time.time() - start) * 1000, 1)
    return clean_answer, retrieved_docs, latency_ms, triggered, context


# --- Streamlit UI -----------------------------------------------------------

st.set_page_config(page_title="RAG Injection Lab", layout="wide")
# inject_custom_css()  # disabled — reverted to default styling
st.title("RAG Document Q&A — Injection Defense Lab (Stage 1: Undefended)")

# Sidebar: index management + model choice
with st.sidebar:
    st.header("Setup")

    st.write(f"Documents folder: `{DOCS_DIR}/`")
    if st.button("Build / rebuild index"):
        with st.spinner("Embedding documents..."):
            try:
                build_vectorstore()
                st.session_state["vectorstore_ready"] = True
                st.success("Index built.")
            except Exception as e:
                st.error(str(e))

    st.divider()
    model_name = st.selectbox("Model", AVAILABLE_MODELS, index=0)

# Load vectorstore (either just-built or previously persisted on disk)
if "vectorstore_ready" not in st.session_state:
    st.session_state["vectorstore_ready"] = os.path.isdir(CHROMA_DIR)

if not st.session_state["vectorstore_ready"]:
    st.info("No index found yet. Add files to documents/ and click 'Build / rebuild index'.")
    st.stop()

vectorstore = load_vectorstore()

# Chat history lives in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of {"role", "content", "sources"?}

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Retrieved chunks"):
                for i, doc in enumerate(msg["sources"], 1):
                    src = doc.metadata.get("source", "unknown")
                    st.markdown(f"**Chunk {i}** (`{src}`)")
                    st.text(doc.page_content)
        if msg.get("action_fired"):
            st.warning(f"📧 Agentic action fired — email drafted and saved to `{msg['action_path']}`")
            with st.expander("View drafted email"):
                st.text(msg["action_email_text"])

query = st.chat_input("Ask a question about your documents...")

if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking ({model_name})..."):
            answer, retrieved_docs, latency_ms, triggered, context = answer_query(
                query, model_name, vectorstore
            )
        st.write(answer)
        st.caption(f"{latency_ms} ms")
        with st.expander("Retrieved chunks"):
            for i, doc in enumerate(retrieved_docs, 1):
                src = doc.metadata.get("source", "unknown")
                st.markdown(f"**Chunk {i}** (`{src}`)")
                st.text(doc.page_content)

        action_fired = False
        action_path = None
        action_email_text = None
        if triggered:
            with st.spinner("Drafting alert email..."):
                action_email_text = draft_email(model_name, query, context, answer)
                action_path = save_email(action_email_text)
            action_fired = True
            st.warning(f"📧 Agentic action fired — email drafted and saved to `{action_path}`")
            with st.expander("View drafted email"):
                st.text(action_email_text)

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": retrieved_docs,
            "action_fired": action_fired,
            "action_path": action_path,
            "action_email_text": action_email_text,
        }
    )
