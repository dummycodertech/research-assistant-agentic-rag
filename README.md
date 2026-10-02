# 📚 Papeer — Agentic Research Paper Assistant

> An agentic RAG system that lets you **chat with your research papers**, **verify claims** against recent literature, and **search the web** for the latest findings — all in one place.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://papeer-rag-research-assistant.streamlit.app)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **Chat with Papers** | Upload PDFs, Markdown, or TXT files and ask questions about their content |
| 📡 **ArXiv Loader** | Load papers directly by ArXiv ID or paper title — no manual download needed |
| 🔗 **URL Loader** | Scrape and load any public web page into your knowledge base |
| 🌐 **Live Web Search** | Automatically queries the web via Tavily when current information is needed |
| ✅ **Claim Verification** | Checks whether a specific claim from a paper has been superseded by newer research |
| 💬 **Multi-Session Chat** | Separate conversation histories per session with LLM-generated session names |
| ⚡ **Side Channel `/btw`** | Ask off-topic questions without polluting your research session history |
| 📊 **Graph State Inspector** | Expand any response to see the full LangGraph state for that turn |

---

## 🏗️ Architecture

Papeer is built around a **LangGraph agentic RAG pipeline** with intelligent routing, multi-step retrieval, relevancy checking, and query rewriting.

```
User Query
    │
    ▼
┌──────────┐
│  Router  │  classifies query into one of three routes
└──────────┘
     │
     ├─── retrieve ──────► ┌─────────────┐
     │                     │  Agent Node │ ◄─────────────────────┐
     │                     │ (plans what │                       │
     │                     │  to search) │                       │
     │                     └──────┬──────┘                       │
     │                            │                              │
     │                     tool_calls                      (retry, max 3)
     │                            ▼                              │
     │                     ┌──────────────────┐                  │
     │                     │    Tool Node     │──────────────────┘
     │                     │ ┌──────────────┐ │
     │                     │ │ Vector Search│ │
     │                     │ ├──────────────┤ │
     │                     │ │  Web Search  │ │
     │                     │ └──────────────┘ │
     │                     └──────────────────┘
     │                            │
     │                     ┌──────▼───────────┐
     │                     │  Relevancy Check │
     │                     └──────────────────┘
     │                            │
     │                   relevant │ not relevant (retry once)
     │                            │         │
     │                            │    ┌────▼──────────┐
     │                            │    │ Query Rewrite │──► Agent Node
     │                            │    └───────────────┘
     │                            ▼
     ├─── verify_claim ──► ┌──────────────┐
     │                     │ Verify Claim │  (dual Tavily search: web + arXiv)
     │                     └──────┬───────┘
     │                            │
     └─── direct_answer ──────────┤
                                  ▼
                          ┌────────────────┐
                          │ Generate Answer│
                          └────────────────┘
```

### Graph Nodes

| Node | Role |
|------|------|
| **Router** | Classifies query → `retrieve` / `verify_claim` / `direct_answer` |
| **Agent Node** | Plans tool calls via LLM; falls back to forced retrieval on LLM error |
| **Tool Node** | Executes `retrieve_from_vectorstore` or `web_search`; max 3 retrieval attempts |
| **Relevancy Check** | LLM judge — are the retrieved chunks relevant to the query? |
| **Query Rewrite** | Rewrites failed queries with better keywords (max 1 rewrite per turn) |
| **Verify Claim** | Dual Tavily search (general web + arXiv-targeted) to fact-check claims |
| **Generate Answer** | Final synthesis using retrieved context (truncated to 8,000 chars) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | `openai/gpt-oss-120b` via [Groq](https://groq.com) |
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) — stateful agentic graph with `MemorySaver` checkpointing (per-turn fresh thread) |
| **Vector Store** | [Qdrant](https://qdrant.tech) — in-memory (`:memory:`), per-session collections |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine similarity) with `CacheBackedEmbeddings` + `LocalFileStore` |
| **Web Search** | [Tavily](https://tavily.com) API |
| **Document Loaders** | PyMuPDF (PDF), TextLoader (TXT/MD), WebBaseLoader (URLs), ArXiv Atom API |
| **UI** | [Streamlit](https://streamlit.io) |
| **Evaluation** | [DeepEval](https://docs.confident-ai.com/) with a custom Groq judge wrapper |

---

## 📁 Project Structure

```
papeer/
├── app.py                          # Streamlit UI — sessions, chat, file/URL/ArXiv upload
├── streamlit_secrets_bridge.py     # Bridges st.secrets → os.environ for Streamlit Cloud
├── main.py                         # Entry point for local / Docker runs
├── evaluate.py                     # DeepEval RAG evaluation pipeline
├── goldens.json                    # 10 golden Q&A pairs used for evaluation
├── eval_results.json               # Raw per-question evaluation results
├── Dockerfile                      # Docker image definition (used for AWS deployment)
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python 3.12 pin for Streamlit Cloud
├── backend/
│   ├── rag_graph.py                # LangGraph pipeline — all nodes, routing, graph assembly
│   ├── vector_store.py             # In-memory Qdrant wrapper + CacheBackedEmbeddings
│   ├── paper_loader.py             # PDF / TXT / MD / URL / ArXiv document loaders + chunker
│   ├── btw_handler.py              # /btw side-channel handler (web-aware off-topic answers)
│   └── models.py                   # Pydantic schemas (RouterDecision, RelevancyDecision, etc.)
├── embedding_cache/                # Persisted embedding cache (LocalFileStore)
└── .streamlit/
    ├── config.toml                 # fileWatcherType = "none" (suppresses Streamlit noise)
    └── secrets.toml.example        # Template for required secrets
```

---

## 🚀 Run Locally

### 1. Clone & install

```bash
git clone https://github.com/dummycodertech/research-assistant-agentic-rag.git
cd research-assistant-agentic-rag
pip install -r requirements.txt
```

### 2. Set environment variables

Create a `.env` file in the root:

```env
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
```

### 3. Run

```bash
streamlit run app.py
```

---

## 🐳 Docker (Previous AWS Deployment)

This project was originally deployed on **AWS** as a Dockerized Streamlit app.

```bash
# Build
docker build -t papeer .

# Run
docker run -p 8501:8501 \
  -e GROQ_API_KEY=gsk_... \
  -e TAVILY_API_KEY=tvly-... \
  papeer
```

The Dockerfile uses `python:3.12`, layers dependencies for Docker cache efficiency (requirements → backend → app files), and runs Streamlit in headless mode on port `8501`.

---

## ☁️ Current Deployment: Streamlit Community Cloud

The app is live at **[papeer-rag-research-assistant.streamlit.app](https://papeer-rag-research-assistant.streamlit.app)**

To deploy your own fork:

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New App**
3. Select your fork, branch `master`, entry point `app.py`
4. Add secrets under **Settings → Secrets**:

```toml
GROQ_API_KEY = "gsk_..."
TAVILY_API_KEY = "tvly-..."
```

> **Note on storage:** Qdrant runs in-memory on Streamlit Cloud (no external DB). Documents uploaded in a session persist for the life of the Streamlit worker process. They are cleared on worker restart.

---

## 🧪 Evaluation

The RAG pipeline was evaluated using [DeepEval](https://docs.confident-ai.com/) on **10 adversarial questions** drawn from a long-form technical research report. A custom `GroqJudge` wrapper drives all DeepEval scoring using `openai/gpt-oss-20b`.

```bash
python evaluate.py
```

### Results (threshold = 0.75, 10 test cases)

| Metric | Avg Score | Pass Rate |
|--------|-----------|-----------|
| **Contextual Precision** | 0.84 | 8 / 10 — 80% ✅ |
| **Contextual Recall** | 0.69 | 4 / 10 — 40% ⚠️ |
| **Contextual Relevancy** | 0.61 | 2 / 10 — 20% ❌ |
| **Answer Relevancy** | 0.75 | 6 / 10 — 60% ⚠️ |
| **Faithfulness** | 0.62 | 4 / 10 — 40% ⚠️ |

**Takeaways:**
- **Contextual Precision** is the strongest metric — retrieved chunks are generally well-ranked.
- **Contextual Relevancy** is the weakest — the retriever often returns chunks that are adjacent to but not directly relevant to the exact question asked, an expected limitation of chunk-level cosine similarity without re-ranking.
- **Recall and Faithfulness** failures are partly attributable to the dense, multi-faceted nature of the test questions (e.g. security deep-dives covering 5+ sub-topics in one query).
- Full per-question breakdowns with reasoning are in [`eval_results.json`](./eval_results.json).

---

## ⚡ The `/btw` Command

Type `/btw <question>` in the chat to ask anything off-topic without it being added to your research session history. The `/btw` handler routes internally — it uses Tavily web search for live questions (news, prices, events) and answers from LLM knowledge for stable general questions.

```
/btw What is the current price of gold?
/btw Who won the 2024 Nobel Prize in Physics?
/btw Explain the softmax function
```

---

## 🔑 Required API Keys

| Service | Purpose | Free Tier |
|---------|---------|-----------|
| [Groq](https://console.groq.com) | LLM inference | ✅ Rate-limited free tier |
| [Tavily](https://tavily.com) | Web search | ✅ 1,000 searches/month free |

---

## 📜 License

MIT
