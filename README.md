# 📚 Papeer — Agentic Research Paper Assistant

> An agentic RAG system that lets you **chat with your research papers**, **verify claims** against recent literature, and **search the web** for the latest findings — all in one place.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://research-assistant-agentic-rag.streamlit.app)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **Chat with Papers** | Upload PDFs, Markdown, or TXT files and ask questions about their content |
| 📡 **ArXiv Loader** | Load papers directly by ArXiv ID or paper title — no download needed |
| 🔗 **URL Loader** | Scrape and load any web page into your knowledge base |
| 🌐 **Live Web Search** | Automatically searches the web via Tavily for current or supplementary information |
| ✅ **Claim Verification** | Checks if a specific claim from a paper has been superseded by newer research |
| 💬 **Multi-Session Chat** | Maintains separate conversation histories per session with LLM-generated session names |
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
│  Router  │  ── classifies query into one of three routes ──►
└──────────┘
     │
     ├─── retrieve ──────► ┌─────────────┐     tool_calls      ┌──────────────────┐
     │                     │  Agent Node │ ──────────────────► │   Tool Node      │
     │                     │ (plans what │ ◄────────────────── │ ┌──────────────┐ │
     │                     │  to search) │    tool results     │ │ Vector Search│ │
     │                     └─────────────┘                     │ ├──────────────┤ │
     │                            │                            │ │  Web Search  │ │
     │                            │ no more tools              │ └──────────────┘ │
     │                            ▼                            └──────────────────┘
     │                     ┌──────────────────┐
     │                     │  Relevancy Check │
     │                     └──────────────────┘
     │                            │
     │                   relevant │ not relevant (retry once)
     │                            │         │
     │                            │    ┌────▼──────────┐
     │                            │    │ Query Rewrite │──► back to Agent Node
     │                            │    └───────────────┘
     │                            ▼
     ├─── verify_claim ──► ┌──────────────┐
     │                     │ Verify Claim │ (Tavily + arXiv search)
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
| **Agent Node** | Plans tool calls using bound LLM with retrieval tools |
| **Tool Node** | Executes `retrieve_from_vectorstore` or `web_search` tools |
| **Relevancy Check** | LLM-based judge — are retrieved chunks relevant to the query? |
| **Query Rewrite** | Rewrites failed queries with better keywords (max 1 retry) |
| **Verify Claim** | Dual Tavily search (web + arXiv) to fact-check paper claims |
| **Generate Answer** | Final synthesis with retrieved context or direct LLM response |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | `openai/gpt-oss-120b` via [Groq](https://groq.com) (ultra-fast inference) |
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) — stateful agentic graph with `MemorySaver` checkpointing |
| **Vector Store** | [Qdrant](https://qdrant.tech) — in-memory, per-session collections |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace with `CacheBackedEmbeddings` |
| **Web Search** | [Tavily](https://tavily.com) API |
| **Document Loaders** | PyMuPDF (PDF), TextLoader (TXT/MD), WebBaseLoader (URLs), ArXiv API |
| **UI** | [Streamlit](https://streamlit.io) |
| **Evaluation** | [DeepEval](https://docs.confident-ai.com/) with custom Groq judge |

---

## 📁 Project Structure

```
papeer/
├── app.py                          # Streamlit UI — sessions, chat, document upload
├── streamlit_secrets_bridge.py     # Bridges st.secrets → os.environ for Streamlit Cloud
├── main.py                         # Minimal entry point (for local/Docker runs)
├── evaluate.py                     # DeepEval evaluation pipeline
├── goldens.json                    # Golden Q&A pairs for evaluation
├── eval_results.json               # Latest evaluation results
├── Dockerfile                      # Docker image (used for AWS deployment)
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version pin for Streamlit Cloud
├── backend/
│   ├── rag_graph.py                # LangGraph agentic RAG pipeline (core logic)
│   ├── vector_store.py             # Qdrant vector store wrapper + embedding cache
│   ├── paper_loader.py             # PDF / TXT / MD / URL / ArXiv document loaders
│   ├── btw_handler.py              # /btw side-channel (web-aware off-topic answers)
│   └── models.py                   # Pydantic schemas (RouterDecision, RelevancyDecision, etc.)
├── graph.png                       # LangGraph graph visualisation
└── .streamlit/
    ├── config.toml                 # Streamlit server config
    └── secrets.toml.example        # Secrets template
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

## 🐳 Docker (AWS Deployment)

This project was originally deployed on **AWS** as a Docker container.

```bash
# Build image
docker build -t papeer .

# Run with env vars
docker run -p 8501:8501 \
  -e GROQ_API_KEY=gsk_... \
  -e TAVILY_API_KEY=tvly-... \
  papeer
```

The Dockerfile:
- Uses `python:3.12` base image
- Layers dependencies for maximum Docker cache efficiency (requirements → backend → app files)
- Exposes port `8501` and runs Streamlit in headless mode

---

## ☁️ Deployed on Streamlit Community Cloud

The app is currently live on **Streamlit Community Cloud**.

To deploy your own instance:

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New App**
3. Select this repo, branch `master`, entry point `app.py`
4. Add your secrets under **Settings → Secrets**:

```toml
GROQ_API_KEY = "gsk_..."
TAVILY_API_KEY = "tvly-..."
```

> **Note:** Qdrant runs in-memory on Streamlit Cloud (no external DB needed). Documents uploaded in a session persist as long as the Streamlit worker process stays warm.

---

## 🧪 Evaluation

The project includes a full evaluation pipeline using [DeepEval](https://docs.confident-ai.com/):

```bash
python evaluate.py
```

Metrics evaluated:
- **Contextual Precision** — are retrieved chunks actually relevant?
- **Contextual Recall** — does retrieval cover the necessary information?
- **Answer Relevancy** — does the generated answer address the question?
- **Faithfulness** — is the answer grounded in the retrieved context?

Golden Q&A pairs are stored in [`goldens.json`](./goldens.json). Results are saved to [`eval_results.json`](./eval_results.json).

---

## ⚡ The `/btw` Command

Type `/btw <question>` in the chat to ask anything off-topic (weather, news, general knowledge) without it being stored in your research session history. The `/btw` handler has its own routing: it decides whether to use Tavily web search or answer from LLM knowledge directly.

**Example:**
```
/btw What is the current price of gold?
/btw Who won the 2024 Nobel Prize in Physics?
```

---

## 🔑 Required API Keys

| Service | Free Tier | Sign Up |
|---------|-----------|---------|
| **Groq** | ✅ Free (rate-limited) | [console.groq.com](https://console.groq.com) |
| **Tavily** | ✅ Free (1000 searches/month) | [tavily.com](https://tavily.com) |

---

## 📜 License

MIT
