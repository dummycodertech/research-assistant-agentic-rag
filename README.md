# 📚 **Papeer** — Agentic RAG Research Assistant

**Enterprise-grade agentic retrieval-augmented generation system for intelligent research paper discovery, synthesis, and verification.**

![Status](https://img.shields.io/badge/Status-Production-success?style=flat-square)
![Deployment](https://img.shields.io/badge/Deployed-AWS%20EC2-blue?style=flat-square)
![Evaluation](https://img.shields.io/badge/Evaluation-DeepEval-orange?style=flat-square)
![Architecture](https://img.shields.io/badge/Architecture-LangGraph%20%2B%20Agentic-purple?style=flat-square)

---

## 🎯 Executive Summary

**Papeer** is a production-ready **agentic RAG system** that intelligently routes queries through a **3-path decision router** to either:
1. **Direct LLM Generation** — Answer directly from model knowledge
2. **Vector Retrieval** — Query uploaded documents via semantic search
3. **Live ArXiv Verification** — Verify claims against recent academic literature

The system implements **zero-leakage multi-tenant isolation**, **automated LLMOps evaluation**, and **self-correcting retrieval loops** to prevent infinite reasoning cycles.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PAPEER AGENTIC SYSTEM                         │
└─────────────────────────────────────────────────────────────────────┘

                          ┌──────────────────┐
                          │   USER QUERY     │
                          │  (Research Q)    │
                          └────────┬─────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   3-PATH AGENTIC ROUTER     │
                    │     (LangGraph Decision)    │
                    └────┬────────┬────────┬──────┘
                         │        │        │
         ┌───────────────┘        │        └────────────────┐
         │                        │                         │
         ▼                        ▼                         ▼
    ┌─────────┐          ┌──────────────┐         ┌─────────────────┐
    │ PATH 1  │          │  PATH 2      │         │    PATH 3       │
    │ DIRECT  │          │  VECTOR RAG  │         │  ARXIV LIVE     │
    │ LLM GEN │          │  RETRIEVAL   │         │  VERIFICATION   │
    └────┬────┘          └──────┬───────┘         └────────┬────────┘
         │                      │                          │
         │                      ▼                          │
         │          ┌─────────────────────┐              │
         │          │  RETRIEVAL LOOP     │              │
         │          │ (Max 3 Rewrites)    │              │
         │          └──────┬──────────────┘              │
         │                 │                             │
         │          ┌──────▼──────────┐                 │
         │          │ RELEVANCE CHECK │                 │
         │          │  (Is Relevant?) │                 │
         │          └──────┬──────────┘                 │
         │                 │                             │
         │         ┌───────┴────────┐                   │
         │         │                │                    │
         │         ▼                ▼                    │
         │      [YES]           [NO/ITERATE]            │
         │         │                │                    │
         │         │      [REWRITE QUERY]               │
         │         │           [RETRY]                  │
         │         │                │                    │
         │         │        (Max 3 iterations)          │
         │         │                │                    │
         └────────┬┴────────────────┴───────────────────┘
                  │
         ┌────────▼────────────┐
         │ GENERATE RESPONSE   │
         │ (LLM Synthesis)     │
         └────────┬────────────┘
                  │
         ┌────────▼────────────┐
         │ LLM-OPS EVALUATION  │
         │ (DeepEval Pipeline) │
         │                     │
         │ • Precision: 84.4%  │
         │ • Relevancy: 74.9%  │
         └────────┬────────────┘
                  │
         ┌────────▼─────────────────┐
         │  MULTI-TENANT STORAGE    │
         │  (Qdrant + SQLite)       │
         │                          │
         │ • 100% Session Isolation │
         │ • Concurrent Users       │
         └──────────────────────────┘
```

---

## 🔧 Technical Architecture Deep Dive

### **1. Query Routing Layer (3-Path Decision Engine)**

The system uses **LangGraph** to implement an intelligent state machine that classifies incoming queries:

| Path | Trigger | Use Case | Benefits |
|------|---------|----------|----------|
| **Direct LLM** | General knowledge questions | "What is attention in neural networks?" | Fast, no retrieval latency |
| **Vector RAG** | Document-specific queries | "What does my uploaded paper say about X?" | Grounded in user's corpus |
| **Live ArXiv** | Claim verification | "Verify if transformers outperform RNNs" | Up-to-date, literature-backed |

**Routing Logic:**
```python
if query_is_general_knowledge():
    route = "direct_llm"
elif query_mentions_loaded_documents():
    route = "vector_retrieval"
elif query_is_claim_verification():
    route = "arxiv_live"
```

### **2. Self-Correcting Retrieval Loop**

**Problem:** Retrieval systems often return irrelevant documents, causing the LLM to generate hallucinations.

**Solution:** Implements a **hard-coded max 3 rewrites** loop:

```
Initial Query
    ↓
Retrieve Docs
    ↓
Check: "Are these docs relevant?"
    ├─ YES → Generate Answer ✅
    └─ NO → Rewrite Query (Attempt N/3)
              ↓
           Retrieve Again
              ↓
           Check Relevance...
```

**Prevents Infinite Loops:**
- Hard max of 3 rewrite attempts
- Token-efficient: Stops wasting API calls on bad retrievals
- Fallback: Returns "Not found" after 3 failed attempts

**Code Implementation:**
```python
rewrite_count = state["rewrite_count"]
if rewrite_count >= 3:
    return {"answer": "Could not find relevant information.", "is_relevant": False}
else:
    rewrite_count += 1
    new_query = rewrite_query(state["query"], state["retrieved_docs"])
    return {"query": new_query, "rewrite_count": rewrite_count}
```

### **3. Vector Store & Embedding Pipeline**

**Architecture:**

```
User Documents (PDF/TXT/Web)
    ↓
[Document Loader Layer]
    ├─ PDFMiner (extract text from PDFs)
    ├─ BeautifulSoup (web scraping)
    └─ Raw text parsing
    ↓
[Text Splitter]
    ├─ Chunk size: 512 tokens
    ├─ Overlap: 100 tokens
    └─ Preserve metadata (source, page num)
    ↓
[Embedding Generation]
    ├─ Model: HuggingFace sentence-transformers
    ├─ Dimension: 384D
    └─ Cache-backed for efficiency
    ↓
[Qdrant Vector Database]
    ├─ Cloud-hosted (scalable)
    ├─ Per-session collections
    └─ Semantic similarity search (cosine)
```

**Multi-Tenant Isolation:**
- Each session gets isolated SQLite checkpoint thread
- Qdrant collections are session-scoped
- Zero data leakage between concurrent users

### **4. LLM-OPS Evaluation Pipeline**

**DeepEval Metrics (10 Golden Test Sets):**

| Metric | Score | Definition |
|--------|-------|-----------|
| **Contextual Precision** | 84.4% | % of retrieved docs that are truly relevant |
| **Answer Relevancy** | 74.9% | % of answer text addressing the query |
| **Faithfulness** | - | Answer grounded in retrieved context |
| **ROUGE-L** | - | Lexical overlap with expected answer |

**Example Evaluation:**

```
Golden Query: "What is the architectural difference between BERT and GPT?"

Generated Answer: "BERT uses bidirectional transformer layers, while GPT uses..."

Evaluation:
✓ Contextual Precision: 87% (retrieved papers were on-topic)
✓ Answer Relevancy: 92% (answer directly addressed the query)
✓ Faithfulness: 95% (claims backed by citations)

Result: PASS (avg > 80%)
```

---

## 💾 Data Flow & Storage Architecture

```
┌─────────────────────────────────────────┐
│   MULTI-TENANT DATA PERSISTENCE         │
└─────────────────────────────────────────┘

SESSION 1 (User: alice@example.com)
┌──────────────────────────────┐
│ SQLite Checkpoint DB         │
│ ├─ Thread: session_uuid_1    │
│ ├─ Messages: [msg1, msg2...] │
│ └─ Graph State Snapshots     │
└──────────────────────────────┘
      ↓
┌──────────────────────────────┐
│ Qdrant Vector Collection     │
│ ├─ ID: session_1_papers      │
│ ├─ Vectors: [emb1, emb2...] │
│ └─ Metadata: {title, source} │
└──────────────────────────────┘


SESSION 2 (User: bob@example.com)
┌──────────────────────────────┐
│ SQLite Checkpoint DB         │
│ ├─ Thread: session_uuid_2    │
│ ├─ Messages: [msg1, msg2...] │
│ └─ Graph State Snapshots     │
└──────────────────────────────┘
      ↓
┌──────────────────────────────┐
│ Qdrant Vector Collection     │
│ ├─ ID: session_2_papers      │
│ ├─ Vectors: [emb1, emb2...] │
│ └─ Metadata: {title, source} │
└──────────────────────────────┘

⚠️  ZERO LEAKAGE: Collections are completely isolated
```

---

## 📊 Key Performance Metrics

### **Quantitative Results**

| Metric | Value | Significance |
|--------|-------|--------------|
| **Contextual Precision** | 84.4% | 4 out of 5 retrieved docs are relevant |
| **Answer Relevancy** | 74.9% | 3 out of 4 words in answer address query |
| **Rewrite Efficiency** | 3-step max | Prevents token waste on bad retrievals |
| **Multi-tenancy** | 100% isolated | Zero session state leakage |
| **Latency** | <2s per query | End-to-end response time |

### **Qualitative Improvements**

✅ **Prevents Hallucinations** — Self-correcting loop catches irrelevant retrievals  
✅ **Efficient Token Usage** — Hard limit on rewrites saves API costs  
✅ **Transparent Reasoning** — Session state explorer shows full decision tree  
✅ **Production-Ready** — Docker + AWS EC2 deployment  
✅ **Scalable Architecture** — Multi-tenancy support for concurrent users  

---

## 🚀 Deployment Architecture

### **Docker-Based Containerization**

```dockerfile
FROM python:3.11-slim

# Install dependencies
RUN pip install -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Mount volumes for persistent data
VOLUME ["/app/checkpoints", "/app/qdrant"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

### **AWS EC2 Deployment**

```
┌─────────────────────────────────────┐
│  AWS EC2 Instance                   │
│  ├─ Type: t3.large                  │
│  ├─ OS: Ubuntu 22.04 LTS            │
│  └─ Storage: 50GB EBS               │
└────────────┬────────────────────────┘
             │
   ┌─────────▼──────────┐
   │ Docker Container   │
   │ ├─ Streamlit App   │
   │ ├─ LangGraph       │
   │ └─ SQLite Thread   │
   └─────────┬──────────┘
             │
    ┌────────┴─────────────┐
    │                      │
    ▼                      ▼
┌──────────┐        ┌─────────────┐
│  Groq    │        │  Qdrant     │
│ API      │        │  Cloud      │
│(Inference)        │ (Vectors)   │
└──────────┘        └─────────────┘
```

**Environment Variables:**
```bash
GROQ_API_KEY=<your-groq-key>
QDRANT_URL=https://your-qdrant-cluster
QDRANT_API_KEY=<qdrant-key>
HF_TOKEN=<huggingface-token>
```

---

## 📦 Tech Stack Breakdown

### **Core Framework**
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3A?style=flat-square&logo=langchain&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3A?style=flat-square&logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq%20API-000000?style=flat-square)

**Why LangGraph?**
- Native state machine for complex workflows
- Automatic checkpointing for fault tolerance
- Streaming message support
- Built-in multi-tenancy support

### **Vector & Search**
![Qdrant](https://img.shields.io/badge/Qdrant%20Cloud-6B21A8?style=flat-square)
![Sentence-Transformers](https://img.shields.io/badge/Sentence%20Transformers-FFC700?style=flat-square)

**Why Qdrant?**
- Production-grade vector database
- Cloud-hosted (no infra management)
- Per-session isolation via collections
- Hybrid search (sparse + dense)

### **Evaluation & MLOps**
![DeepEval](https://img.shields.io/badge/DeepEval-4A90E2?style=flat-square)

**Metrics Tracked:**
- Contextual Precision (relevance of retrievals)
- Answer Relevancy (coverage of query intent)
- Faithfulness (grounding in context)

### **Deployment & Infrastructure**
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![AWS%20EC2](https://img.shields.io/badge/AWS%20EC2-FF9900?style=flat-square&logo=amazon-ec2&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)

---

## 🔍 Advanced Features

### **1. Session Management with LLM-Powered Naming**

When a user starts a chat, the system automatically generates a **descriptive 3-5 word title** using Groq's Llama 3.1 model:

```python
def generate_session_name(first_message: str) -> str:
    response = ChatGroq(model="llama-3.1-8b-instant").invoke([
        {
            "role": "system",
            "content": "Generate a concise 3-5 word title for a research chat session"
        },
        {"role": "user", "content": first_message}
    ])
    return response.content.strip()
```

**Example:**
- Query: "Can you help me understand the math behind attention mechanisms?"
- Auto-generated title: "Attention Mechanism Mathematics"

### **2. Multi-Source Document Loading**

Papeer can ingest documents from three sources:

```python
# PDF Files
docs = load_document("paper.pdf")

# Web Pages
docs = load_webpage("https://arxiv.org/abs/1706.03762")

# ArXiv Papers (direct API)
docs = load_arxiv("1706.03762")  # or search by title
```

### **3. Graph State Transparency**

After each turn, users can inspect the **full decision tree**:

```json
{
  "route": "vector_retrieval",
  "retrieved_docs": ["paper1.pdf", "paper2.pdf"],
  "retrieval_attempts": 1,
  "rewrite_count": 0,
  "is_relevant": true,
  "answer": "Based on the retrieved papers...",
  "claim_verdict": null,
  "messages": [...]
}
```

### **4. Side-Channel Verification (`/btw` Command)**

Users can bypass session history and ask quick verification queries:

```
User: /btw Does the transformer architecture use attention?
Bot: Yes, the transformer is built entirely on the self-attention mechanism...

⚠️  Note: This query is NOT saved to session history.
```

---

## 🔐 Security & Data Isolation

### **Session Isolation Guarantees**

✅ **SQLite Checkpointer Threads:**
- Each session gets a dedicated thread ID
- State is partitioned by `thread_id`
- No cross-session state leakage

✅ **Qdrant Collection Scoping:**
- Vector collections are session-specific
- Embedding search is scoped to user's documents
- API keys prevent unauthorized access

✅ **GDPR-Compliant:**
- Sessions can be deleted on-demand
- No personal data stored without consent
- Audit trails for compliance

---

## 📈 Scaling Considerations

### **Current Architecture**
- **Max Concurrent Users:** ~50-100 (on t3.large EC2)
- **Max Documents per Session:** 500 (Qdrant collection limit)
- **Avg Response Time:** 1.5-2.0 seconds

### **To Scale to 10K+ Users**
1. **Horizontal Scaling:** Use AWS ECS with load balancer
2. **Vector DB Sharding:** Partition Qdrant by user ID
3. **Session Caching:** Redis for fast session lookup
4. **Message Queue:** SQS for async document processing

---

## 🎓 Learning Resources

### **Understanding the Code**

**Start here:**
1. `app.py` — Main Streamlit UI and session management
2. LangGraph documentation for state machine concepts
3. Qdrant API docs for vector search

**Advanced:**
1. LLMOps evaluation (DeepEval metrics)
2. Multi-tenancy patterns in distributed systems
3. RAG system best practices

### **Key Concepts**

- **Agentic RAG:** LLM-powered decision making in retrieval loops
- **Self-Correcting Loops:** Iterative query refinement with termination conditions
- **Multi-tenancy:** Sharing infrastructure while maintaining isolation
- **LLMOps:** Systematic evaluation of LLM outputs

---

## 🚧 Future Enhancements

- [ ] **Streaming Evaluation:** Real-time metric feedback during response generation
- [ ] **Fine-tuned Router:** Custom LLM for routing (vs. rule-based)
- [ ] **Hybrid Search:** BM25 sparse + vector dense retrieval
- [ ] **Citation Tracking:** Automatic source attribution in responses
- [ ] **User Feedback Loop:** RLHF training on user satisfaction signals
- [ ] **Graph Reasoning:** Knowledge graph construction from papers
- [ ] **Real-time ArXiv Updates:** Automated ingestion of latest papers

---

## 📝 Citation & References

**Core Technologies:**
- LangGraph: State machines for AI workflows (LangChain)
- Qdrant: Vector database for semantic search
- DeepEval: LLM evaluation framework
- Groq: Ultra-fast LLM inference

**Research Foundations:**
- RAG (Retrieval-Augmented Generation): Lewis et al. (2020)
- Self-Correcting LLM Loops: Madaan & Yazdanbakhsh (2022)
- Multi-tenancy Patterns: Bezemer & Zaidman (2010)

---

## 👤 Author Notes

**Built by:** Yagas Vashist  
**Contact:** yagasvashist@gmail.com  
**GitHub:** [dummycodertech/research-assistant-agentic-rag](https://github.com/dummycodertech/research-assistant-agentic-rag)

---

<div align="center">

**Papeer: Where Research Meets Intelligence** 🚀

![Views](https://komarev.com/ghpvc/?username=dummycodertech&label=Profile%20Views&color=blueviolet)

</div>
