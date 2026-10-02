# 📚 Papeer — Research Paper Assistant

> An agentic RAG system that lets you **chat with your research papers**, **verify claims** against recent literature, and **search the web** for the latest findings — all in one place.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **Chat with Papers** | Upload PDFs, Markdown, or TXT files and ask questions about their content |
| 🌐 **Web Search** | Automatically searches the web for current or supplementary information |
| ✅ **Claim Verification** | Checks if a claim from a paper has been superseded by newer research |
| 📡 **ArXiv Loader** | Load papers directly by ArXiv ID or paper title |
| 🔗 **URL Loader** | Scrape and load web pages into your knowledge base |
| 💬 **Multi-Session** | Maintains separate chat histories per session with auto-generated names |
| ⚡ **Side Channel `/btw`** | Ask off-topic questions without polluting your session history |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────┐    retrieve     ┌───────────────┐
│  Router │ ──────────────► │  Agent Node   │◄─────────────────────┐
└─────────┘                 └───────────────┘                      │
    │                              │                               │
    │ verify_claim          tool_calls                        (retry loop)
    │                              ▼                               │
    ▼                       ┌─────────────┐                        │
┌──────────────┐            │  Tool Node  │ ──────────────────────►│
│ Verify Claim │            │  - VectorDB │    ┌──────────────────┐│
│  (Tavily +   │            │  - Web Srch │    │  Relevancy Check ││
│   arXiv)     │            └─────────────┘    │  + Query Rewrite ││
└──────┬───────┘                               └──────────────────┘│
       │                                                            │
       │ direct_answer                                              │
       ▼                                                            │
┌────────────────┐◄──────────────────────────────────────────────────
│ Generate Answer│
└────────────────┘
```

**Tech Stack:**
- **LLM**: `llama-3.1-8b-instant` + `llama-3.3-70b-versatile` via [Groq](https://groq.com)
- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) (agentic RAG graph with SQLite checkpointing)
- **Vector Store**: [Qdrant Cloud](https://cloud.qdrant.io) (per-session collections)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace (cached locally)
- **Web Search**: [Tavily](https://tavily.com)
- **UI**: [Streamlit](https://streamlit.io)

---

## 🚀 Run Locally

### 1. Clone & install
```bash
git clone https://github.com/dummycodertech/research-assistant-agentic-rag.git
cd research-assistant-agentic-rag
pip install -r requirements.txt
```

### 2. Set up environment variables
Create a `.env` file in the root:
```env
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
QDRANT_URL=https://your-cluster.eu-central-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```

### 3. Run
```bash
streamlit run app.py
```

---

## ☁️ Deployed on Streamlit Cloud

This app is hosted on **Streamlit Community Cloud**.

To deploy your own instance:
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New App
3. Select this repo, branch `main`, entry point `app.py`
4. Add your secrets under **Settings → Secrets**:
```toml
GROQ_API_KEY = "gsk_..."
TAVILY_API_KEY = "tvly-..."
QDRANT_URL = "https://your-cluster.eu-central-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "your_qdrant_api_key"
```

> **Free services used:** Groq (LLM inference), Qdrant Cloud (vector DB), Tavily (web search) — all have generous free tiers.

---

## 📁 Project Structure

```
papeer/
├── app.py                        # Streamlit UI
├── streamlit_secrets_bridge.py   # Bridges st.secrets → os.environ
├── backend/
│   ├── rag_graph.py              # LangGraph agentic RAG pipeline
│   ├── vector_store.py           # Qdrant vector store wrapper
│   ├── paper_loader.py           # PDF / URL / ArXiv loaders
│   ├── btw_handler.py            # /btw side-channel handler
│   └── models.py                 # Pydantic schemas
├── requirements.txt
└── .streamlit/
    └── secrets.toml.example      # Template for secrets
```
