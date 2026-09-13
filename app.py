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
from defenses import (
    SELF_REMINDER_SYSTEM_MESSAGE,
    BOUNDARY_INSTRUCTION,
    wrap_chunks_with_boundaries,
    classify_response,
)
from logger import log_result
# from theme import inject_custom_css  # disabled — reverted to default styling

# --- Config -------------------------------------------------------------
AVAILABLE_MODELS = ["llama3", "mistral"]
RETRIEVAL_K = 4  # number of chunks to retrieve per query

# Base answer instructions — the part of the prompt that stays constant
# regardless of which defenses are toggled on. Defenses ADD to this; they
# never replace it, so the undefended baseline is just this with nothing
# extra layered on.
BASE_INSTRUCTIONS = "Answer the question using the context below."


# --- Helpers --------------------------------------------------------------

def format_context(retrieved_docs, boundary_marking: bool):
    """
    Joins retrieved chunks into a single context string.
    If boundary_marking is on, each chunk is wrapped in <<DATA>>/<<END DATA>>
    delimiters (defense c); otherwise plain concatenation (Stage 1 behavior).
    """
    if boundary_marking:
        return wrap_chunks_with_boundaries(retrieved_docs)
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def build_prompt(context: str, question: str, self_reminder: bool, boundary_marking: bool) -> str:
    """
    Assembles the full prompt sent to the model, layering on whichever
    defenses are active. Order: [system-style reminder?] + base
    instructions + [boundary instruction?] + context + question + trigger.
    """
    parts = []
    if self_reminder:
        parts.append(SELF_REMINDER_SYSTEM_MESSAGE)
    parts.append(BASE_INSTRUCTIONS)
    if boundary_marking:
        parts.append(BOUNDARY_INSTRUCTION)
    parts.append(f"\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    return "\n\n".join(parts) + TRIGGER_INSTRUCTION


def answer_query(query: str, model_name: str, vectorstore, defenses: dict):
    """
    Runs one retrieval + generation pass, applying whichever defenses are
    active in the `defenses` dict: {"self_reminder": bool,
    "output_classifier": bool, "boundary_marking": bool}.

    Returns a dict with everything Stage 4's logger will eventually need:
      displayed_answer   — what the UI should show (may be a block message)
      raw_answer          — the model's real answer, always kept for logging
      retrieved_docs
      latency_ms
      triggered            — did the model request the email action at all
      action_suppressed   — was it flagged suspicious and blocked from firing
      suspicious           — did the output classifier flag this response
      classifier_reasons  — why, if suspicious
      context
    """
    start = time.time()

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    retrieved_docs = retriever.invoke(query)

    context = format_context(retrieved_docs, defenses["boundary_marking"])
    prompt = build_prompt(context, query, defenses["self_reminder"], defenses["boundary_marking"])

    llm = ChatOllama(model=model_name)
    raw_response = llm.invoke(prompt)

    raw_answer, triggered = strip_trigger(raw_response.content)

    suspicious = False
    classifier_reasons = []
    classifier_score = None
    if defenses["output_classifier"]:
        suspicious, classifier_reasons, classifier_score = classify_response(
            query, raw_answer, triggered
        )

    action_suppressed = triggered and suspicious  # only possible if classifier is on
    displayed_answer = (
        "⚠️ Response blocked: this answer showed signs of following an "
        "injected instruction rather than genuinely answering your question."
        if suspicious
        else raw_answer
    )

    latency_ms = round((time.time() - start) * 1000, 1)
    return {
        "displayed_answer": displayed_answer,
        "raw_answer": raw_answer,
        "retrieved_docs": retrieved_docs,
        "latency_ms": latency_ms,
        "triggered": triggered,
        "action_suppressed": action_suppressed,
        "suspicious": suspicious,
        "classifier_reasons": classifier_reasons,
        "classifier_score": classifier_score,
        "context": context,
        "full_prompt": prompt,
    }


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
    if st.button("Warm up model"):
        with st.spinner(f"Loading {model_name} into memory (can take 30-60s the first time)..."):
            ChatOllama(model=model_name).invoke("Hi")
        st.success(f"{model_name} is warmed up — questions should now answer in a few seconds.")

    st.divider()
    st.subheader("Defenses")
    defense_self_reminder = st.checkbox("(a) Self-reminder / system-prompt hardening")
    defense_output_classifier = st.checkbox("(b) Output classifier")
    defense_boundary_marking = st.checkbox("(c) Boundary-marking")
    defenses = {
        "self_reminder": defense_self_reminder,
        "output_classifier": defense_output_classifier,
        "boundary_marking": defense_boundary_marking,
    }

    st.divider()
    st.subheader("Logging")
    is_attack = st.checkbox("Tag next message as an attack")
    attack_category = None
    if is_attack:
        attack_category = st.selectbox(
            "Attack category", ["direct", "indirect", "obfuscated-multiturn"]
        )
    st.caption("Every query is logged to results.csv regardless of this tag.")

    if os.path.isfile("results.csv"):
        with open("results.csv", "rb") as f:
            st.download_button("Download results.csv", f, file_name="results.csv")

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
        if msg.get("full_prompt"):
            with st.expander("🔍 Full prompt sent to model (for verifying defenses)"):
                st.text(msg["full_prompt"])
        if msg.get("suspicious"):
            with st.expander("⚠️ Output classifier flagged this response"):
                st.write("Reasons:")
                for reason in msg["classifier_reasons"]:
                    st.write(f"- {reason}")
                if msg.get("classifier_score") is not None:
                    st.caption(f"Injection confidence score: {msg['classifier_score']:.3f}")
        if msg.get("action_fired"):
            st.warning(f"📧 Agentic action fired — email drafted and saved to `{msg['action_path']}`")
            with st.expander("View drafted email"):
                st.text(msg["action_email_text"])
        elif msg.get("action_suppressed"):
            st.info("🛡️ Agentic action was requested by the model but suppressed by the output classifier.")

query = st.chat_input("Ask a question about your documents...")

if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking ({model_name})..."):
            result = answer_query(query, model_name, vectorstore, defenses)
        st.write(result["displayed_answer"])
        st.caption(f"{result['latency_ms']} ms")
        with st.expander("Retrieved chunks"):
            for i, doc in enumerate(result["retrieved_docs"], 1):
                src = doc.metadata.get("source", "unknown")
                st.markdown(f"**Chunk {i}** (`{src}`)")
                st.text(doc.page_content)
        with st.expander("🔍 Full prompt sent to model (for verifying defenses)"):
            st.text(result["full_prompt"])

        if result["suspicious"]:
            with st.expander("⚠️ Output classifier flagged this response"):
                st.write("Reasons:")
                for reason in result["classifier_reasons"]:
                    st.write(f"- {reason}")
                if result["classifier_score"] is not None:
                    st.caption(f"Injection confidence score: {result['classifier_score']:.3f}")

        action_fired = False
        action_path = None
        action_email_text = None
        if result["triggered"] and not result["action_suppressed"]:
            with st.spinner("Drafting alert email..."):
                action_email_text = draft_email(
                    model_name, query, result["context"], result["raw_answer"]
                )
                action_path = save_email(action_email_text)
            action_fired = True
            st.warning(f"📧 Agentic action fired — email drafted and saved to `{action_path}`")
            with st.expander("View drafted email"):
                st.text(action_email_text)
        elif result["action_suppressed"]:
            st.info("🛡️ Agentic action was requested by the model but suppressed by the output classifier.")

        log_result(
            model=model_name,
            defenses=defenses,
            query=query,
            is_attack=is_attack,
            attack_category=attack_category,
            displayed_response=result["displayed_answer"],
            raw_response=result["raw_answer"],
            action_fired=action_fired,
            action_suppressed=result["action_suppressed"],
            suspicious=result["suspicious"],
            classifier_score=result["classifier_score"],
            latency_ms=result["latency_ms"],
        )

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": result["displayed_answer"],
            "sources": result["retrieved_docs"],
            "suspicious": result["suspicious"],
            "classifier_reasons": result["classifier_reasons"],
            "action_fired": action_fired,
            "action_suppressed": result["action_suppressed"],
            "action_path": action_path,
            "action_email_text": action_email_text,
            "full_prompt": result["full_prompt"],
            "classifier_score": result["classifier_score"],
        }
    )
