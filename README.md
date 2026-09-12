# RAG Injection Lab — Stage 1 (Undefended baseline)

## Setup

```bash
cd rag_injection_lab
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Make sure Ollama is running (`ollama serve` if it's not already) and that
`llama3` and `mistral` are pulled (you said they already are).

## Add your documents

Drop your `.txt` / `.pdf` files into `documents/`. A sample file is included
so you can smoke-test the pipeline immediately.

## Run

```bash
streamlit run app.py
```

In the sidebar: click **"Build / rebuild index"** once (re-click any time you
change the files in `documents/`), pick a model, then ask questions in the
chat box at the bottom. Each answer shows its retrieved chunks in an
expander so you can sanity-check retrieval.

## What's in Stage 1

- `ingest.py` — loads docs, chunks them, embeds with Ollama, persists to
  Chroma (`chroma_db/`).
- `app.py` — Streamlit chat UI, model selector, plain/undefended prompt.

No defenses, no agentic action, no logging yet — those are Stages 2–4.
