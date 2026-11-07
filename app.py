import streamlit as st  # must be first for secrets bridge
import streamlit_secrets_bridge  # noqa: F401 — injects st.secrets → os.environ

import base64
import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from backend.btw_handler import handle_btw
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.rag_graph import build_graph
from backend.vector_store import add_paper, list_papers
from ui import (
    inject_custom_css,
    render_claim_verification_card,
    render_export_note_button,
    render_hero_hud,
    render_source_citations,
    render_telemetry_trace,
)

st.set_page_config(
    page_title="Papeer — Precision AI Engineering Cockpit",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  FUTURISTIC AI ENGINEERING COCKPIT (PALANTIR / LINEAR / VERCEL DESIGN SYSTEM)
# ══════════════════════════════════════════════════════════════════════════════
inject_custom_css()


# ── Core Graph Engine & Session Management ──────────────────────────────────────
@st.cache_resource
def get_graph():
    return build_graph()


SESSIONS_FILE = Path("sessions.json")
_rename_llm = ChatGroq(model="openai/gpt-oss-20b")


def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def _serialize_state(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if k == "messages":
            out[k] = [
                {
                    "type": type(m).__name__,
                    "content": (
                        m.content[:300]
                        if isinstance(m.content, str)
                        else repr(m.content)[:300]
                    ),
                }
                for m in (v or [])
            ]
        elif k == "retrieved_docs":
            out[k] = [
                {"content": d.page_content[:300], "metadata": d.metadata}
                for d in (v or [])
            ]
        else:
            out[k] = v
    return out


def generate_session_name(first_message: str) -> str:
    try:
        response = _rename_llm.invoke([
            {"role": "system", "content": (
                "Generate a concise 3-5 word title for a research chat session "
                "based on the user's first message. Return only the title, "
                "no punctuation at the end, no quotes."
            )},
            {"role": "user", "content": first_message[:500]},
        ])
        return response.content.strip()
    except Exception:
        return "New Session"


def maybe_rename_session(session_id: str, first_message: str) -> None:
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    save_sessions(st.session_state.sessions_meta)


def create_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.now().isoformat(),
        "is_named": False,
    }
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats[sid] = []
    st.session_state.turns[sid] = 0
    return sid


def load_session_chats(session_id: str) -> list[dict]:
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
        chats = []
        turn = 0
        for msg in state.values.get("messages", []):
            type_name = type(msg).__name__
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if type_name == "HumanMessage":
                chats.append({"role": "user", "content": content})
            elif type_name in ("AIMessage", "AIMessageChunk"):
                turn += 1
                chats.append({"role": "assistant", "content": content, "turn": turn, "graph_state": {}})
        return chats
    except Exception:
        return []


def switch_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:
        st.session_state.chats[session_id] = load_session_chats(session_id)
    if session_id not in st.session_state.turns:
        turn_count = sum(1 for m in st.session_state.chats[session_id] if m["role"] == "assistant")
        st.session_state.turns[session_id] = turn_count


graph = get_graph()

# ── Bootstrap State ─────────────────────────────────────────────────────────────
if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "turns" not in st.session_state:
    st.session_state.turns = {}
if "active_session_id" not in st.session_state:
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
        )
        switch_session(latest["id"])
    else:
        sid = create_session()
        st.session_state.active_session_id = sid

active_sid = st.session_state.active_session_id
active_chats = st.session_state.chats.get(active_sid, [])

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR: MINIMALIST BRANDING, NEW CHAT & SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand Header: 8px cyan square + PAPEER + v1.1 tag
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.2rem 0 1.1rem 0;">
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="width: 8px; height: 8px; background-color: #22D3EE; border-radius: 2px; display: inline-block; box-shadow: 0 0 8px rgba(34, 211, 238, 0.6);"></span>
        <span style="font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 700; letter-spacing: 0.05em; color: #F8FAFC;">PAPEER</span>
      </div>
      <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #38BDF8; background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 4px; padding: 2px 6px;">v1.1</span>
    </div>
    """, unsafe_allow_html=True)

    # Primary Action (+ New Chat Command Bar)
    if st.button("＋ New Chat", key="sidebar_new_chat", use_container_width=True):
        new_sid = create_session()
        st.session_state.active_session_id = new_sid
        active_sid = new_sid
        st.rerun()

    # Session History List
    st.markdown("""
    <div style="margin: 1.25rem 0 0.5rem 0;">
      <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;">
        SESSIONS
      </span>
    </div>
    """, unsafe_allow_html=True)

    sorted_sessions = sorted(
        st.session_state.sessions_meta.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )

    for session_item in sorted_sessions:
        s_id = session_item["id"]
        s_active = s_id == st.session_state.active_session_id
        session_label = session_item["name"]
        
        if st.button(
            session_label,
            key=f"side_sess_{s_id}",
            use_container_width=True,
            type="primary" if s_active else "secondary",
        ):
            if not s_active:
                switch_session(s_id)
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  HERO HEADER & ARCHITECTURE HUD (EMPTY STATE ONLY — AUTO-COLLAPSES ON ACTIVE CONVERSATION)
# ══════════════════════════════════════════════════════════════════════════════
if not active_chats:
    render_hero_hud()

    # 1-Click "Load Benchmark Paper" Action Chip
    col_b1, col_b2, col_b3 = st.columns([0.1, 0.8, 0.1])
    with col_b2:
        if st.button("⚡ Load Benchmark Paper · Attention Is All You Need (arXiv:1706.03762)", key="btn_quick_benchmark", use_container_width=True):
            processed_key = f"processed_files_{active_sid}"
            if processed_key not in st.session_state:
                st.session_state[processed_key] = set()
            with st.spinner("Fetching Attention Is All You Need (1706.03762) from arXiv…"):
                try:
                    docs = load_arxiv("1706.03762")
                    add_paper(docs, active_sid)
                    st.session_state[processed_key].add("1706.03762")
                    st.success("Indexed: Attention Is All You Need (arXiv:1706.03762)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to fetch benchmark paper: {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  CORPUS INGESTION DECK (SMOKED GLASSMORPHISM PANEL)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; margin: 1.2rem 0 0.55rem 0; padding: 0 2px;">
  <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.08em; color: #94A3B8; text-transform: uppercase; font-weight: 600;">
    CORPUS INGESTION
  </span>
  <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.05em; color: #10B981; display: inline-flex; align-items: center; gap: 6px;">
    <span class="status-pulse-emerald"></span> VECTOR PIPELINE READY
  </span>
</div>
""", unsafe_allow_html=True)

loaded_count = len(st.session_state.get(f"processed_files_{active_sid}", set()))
_tab_files, _tab_web, _tab_arxiv, _tab_loaded = st.tabs(
    ["Upload Files", "Web Pages", "ArXiv Papers", f"Loaded Documents ({loaded_count})"]
)

# ── Tab 1: File Upload ──────────────────────────────────────────────────────────
with _tab_files:
    uploaded_files = st.file_uploader(
        "PDF, TXT, or Markdown documents",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        key=f"uploader_{active_sid}",
        label_visibility="collapsed",
    )
    
    col_spacer, col_act = st.columns([2.5, 1.2])
    with col_act:
        btn_add = st.button("Add to Corpus", use_container_width=True, key="btn_add_files", type="primary")

    if btn_add:
        if uploaded_files:
            processed_key = f"processed_files_{active_sid}"
            if processed_key not in st.session_state:
                st.session_state[processed_key] = set()
            with st.spinner("Processing & caching vector embeddings…"):
                for f in uploaded_files:
                    if f.name in st.session_state[processed_key]:
                        st.info(f"Already loaded: {f.name}")
                        continue
                    suffix = Path(f.name).suffix
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(f.read())
                            tmp_path = tmp.name
                        docs = load_document(tmp_path)
                        for doc in docs:
                            doc.metadata["title"] = Path(f.name).stem
                        add_paper(docs, active_sid)
                        st.session_state[processed_key].add(f.name)
                        st.success(f"Indexed: {f.name}")
                    except Exception as e:
                        st.error(f"Failed: {f.name} — {e}")
                    finally:
                        if tmp_path:
                            Path(tmp_path).unlink(missing_ok=True)
            st.rerun()
        else:
            st.warning("Please choose one or more documents to upload.")

# ── Tab 2: Web URL Loader ──────────────────────────────────────────────────────
with _tab_web:
    url_input = st.text_area(
        "Web URLs (one per line)",
        key=f"url_area_{active_sid}",
        height=85,
        label_visibility="collapsed",
        placeholder="https://arxiv.org/html/2310.06825  (one URL per line)",
    )
    col_w_spacer, col_w_act = st.columns([2.5, 1.2])
    with col_w_act:
        btn_urls = st.button("Fetch & Index", use_container_width=True, key="btn_load_urls", type="primary")

    if btn_urls:
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        if urls:
            with st.spinner("Fetching web documents…"):
                for url in urls:
                    try:
                        docs = load_webpage(url)
                        add_paper(docs, active_sid)
                        st.success(f"Loaded: {url[:60]}")
                    except Exception as e:
                        st.error(f"Failed: {url[:60]} — {e}")
            st.rerun()
        else:
            st.warning("Enter at least one web URL.")

# ── Tab 3: ArXiv Paper Loader ──────────────────────────────────────────────────
with _tab_arxiv:
    arxiv_title = st.text_input(
        "Paper Title or ArXiv ID",
        key=f"arxiv_input_{active_sid}",
        label_visibility="collapsed",
        placeholder="e.g. 1706.03762  or  Attention Is All You Need",
    )
    col_a_spacer, col_a_act = st.columns([2.5, 1.2])
    with col_a_act:
        btn_arxiv = st.button("Fetch ArXiv", use_container_width=True, key="btn_load_arxiv", type="primary")

    if btn_arxiv:
        if arxiv_title.strip():
            with st.spinner("Searching ArXiv & ingesting paper…"):
                try:
                    docs = load_arxiv(arxiv_title.strip())
                    add_paper(docs, active_sid)
                    loaded_title = docs[0].metadata.get("title") if docs else arxiv_title.strip()
                    st.success(f"Loaded: {loaded_title}")
                except Exception as e:
                    st.error(f"Failed: {e}")
            st.rerun()
        else:
            st.warning("Enter an ArXiv paper title or identifier.")

# ── Tab 4: Loaded Documents Index ──────────────────────────────────────────────
with _tab_loaded:
    try:
        doc_titles = list_papers(active_sid)
    except Exception:
        doc_titles = None

    if doc_titles is None:
        st.caption("Unable to fetch document index.")
    elif doc_titles:
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#94A3B8;margin-bottom:0.75rem;letter-spacing:0.05em;text-transform:uppercase;">
          Active Corpus · {len(doc_titles)} Papers Embedded
        </div>
        """, unsafe_allow_html=True)
        for _title in doc_titles:
            st.markdown(f"""
            <div style="
              background: rgba(13, 16, 26, 0.75);
              border: 1px solid rgba(255, 255, 255, 0.08);
              border-radius: 8px;
              padding: 0.75rem 1rem;
              margin-bottom: 0.5rem;
              display: flex;
              align-items: center;
              justify-content: space-between;
              gap: 12px;
              backdrop-filter: blur(8px);
              box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
            ">
              <div style="display:flex;align-items:center;gap:10px;overflow:hidden;">
                <span style="color:#22D3EE;font-size:12px;">📄</span>
                <span style="font-family:'Inter',sans-serif;font-size:13px;color:#F8FAFC;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{_title}</span>
              </div>
              <span style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 10px;
                color: #22D3EE;
                background: rgba(34, 211, 238, 0.1);
                border: 1px solid rgba(34, 211, 238, 0.25);
                padding: 2px 7px;
                border-radius: 4px;
                letter-spacing: 0.05em;
              ">INDEXED</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No papers loaded for this session yet. Upload or fetch one above.")

# Subtle divider before research protocols / conversation
st.markdown("""
<div style="height: 1px; margin: 1.6rem 0 1.2rem 0; background: rgba(255, 255, 255, 0.06);"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  RESEARCH PROTOCOLS & STARTERS (2x2 BENTO CARDS — SHOWN IN EMPTY STATE)
# ══════════════════════════════════════════════════════════════════════════════
selected_starter_prompt = None

if not active_chats:
    st.markdown("""
    <div style="margin-bottom: 0.75rem;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:0.08em;color:#94A3B8;text-transform:uppercase;font-weight:600;">
        RESEARCH PROTOCOLS & STARTERS
      </span>
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button(
            "`[ROUTE: RETRIEVE]` **Summarize Key Contributions**\n\nExtract methodology, core findings & benchmark metrics",
            key="starter_sum",
            use_container_width=True,
            type="secondary",
        ):
            selected_starter_prompt = "Summarize the core methodology, key findings, and benchmark results of the loaded papers."
        if st.button(
            "`[ROUTE: VERIFY CLAIM]` **Verify Academic Claim**\n\nCross-examine assertions against arXiv & live literature",
            key="starter_ver",
            use_container_width=True,
            type="secondary",
        ):
            selected_starter_prompt = "Verify whether the claims made in the loaded documents are supported or refuted by broader academic literature."
    with col_s2:
        if st.button(
            "`[ROUTE: TAVILY WEB]` **Web Literature Search**\n\nQuery real-time 2025/2026 preprints and web discoveries",
            key="starter_web",
            use_container_width=True,
            type="secondary",
        ):
            selected_starter_prompt = "Search the web for the latest 2025-2026 breakthroughs and papers related to the loaded topics."
        if st.button(
            "`[SIDE CHANNEL: /BTW]` **Concept Explainer (/btw)**\n\nIsolated side-channel query outside vector memory",
            key="starter_btw",
            use_container_width=True,
            type="secondary",
        ):
            selected_starter_prompt = "/btw Explain the attention mechanism and its computational complexity in intuitive terms."

# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: RENDER VERIFIABLE SOURCE CITATIONS & TELEMETRY TRACE CONSOLE
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  CHAT TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
for i, msg in enumerate(active_chats):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            graph_state = msg.get("graph_state", {})
            citations = graph_state.get("source_citations") or []
            render_claim_verification_card(graph_state)
            render_source_citations(citations)
            render_telemetry_trace(msg.get("turn", 1), graph_state)
            user_q = active_chats[i - 1]["content"] if i > 0 and active_chats[i - 1]["role"] == "user" else ""
            render_export_note_button(
                turn_num=msg.get("turn", 1),
                user_query=user_q,
                assistant_answer=msg["content"],
                citations=citations,
                key=f"dl_history_{i}_{msg.get('turn', 1)}",
            )

# ── Chat Input & Agent Execution ───────────────────────────────────────────────
prompt = st.chat_input("Ask a question, verify a claim, or prefix with /btw for side-channel...")
if selected_starter_prompt:
    prompt = selected_starter_prompt

if prompt:
    is_btw = prompt.strip().lower().startswith("/btw")

    if is_btw:
        query = prompt.strip()[4:].strip()
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption("Side-channel — off-topic query, bypassed vector checkpointer.")
        with st.chat_message("assistant"):
            if not query:
                st.markdown("Specify your question after `/btw`, e.g. `/btw What is quantum entanglement?`")
            else:
                placeholder = st.empty()
                response_text = ""
                for chunk in handle_btw(query):
                    response_text += chunk
                    placeholder.markdown(response_text + "▌")
                placeholder.markdown(response_text)
            st.caption("Side-channel response — not persisted to session history.")

    else:
        if active_sid not in st.session_state.chats:
            st.session_state.chats[active_sid] = []
        if active_sid not in st.session_state.turns:
            st.session_state.turns[active_sid] = 0

        is_first_message = len(st.session_state.chats[active_sid]) == 0

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
        st.session_state.turns[active_sid] += 1
        current_turn = st.session_state.turns[active_sid]

        if is_first_message:
            maybe_rename_session(active_sid, prompt)

        recent_history = [
            {"role": m["role"], "content": m["content"][:200]}
            for m in st.session_state.chats[active_sid][-4:]
        ]
        input_state = {
            "messages": [HumanMessage(content=prompt)],
            "session_id": active_sid,
            "query": prompt,
            "raw_query": prompt,
            "chat_history": recent_history,
            "source_citations": [],
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
        }
        config = {"configurable": {"thread_id": f"{active_sid}_turn_{current_turn}"}}

        with st.chat_message("assistant"):
            placeholder = st.empty()
            status_placeholder = st.empty()
            response_text = ""

            NODE_LABELS = {
                "contextualize_query": "RESOLVING CONVERSATION CONTEXT & CO-REFERENCES",
                "router":              "ROUTING QUERY THROUGH INTENT CLASSIFIER",
                "agent_node":          "PLANNING RETRIEVAL STRATEGY",
                "retrieval":           "HYBRID EMBEDDING & FLASHRANK RERANKING",
                "relevancy_check":     "CROSS-EXAMINING DOCUMENT RELEVANCY",
                "query_rewrite":       "REWRITING & REFINING SUB-QUERIES",
                "verify_claim":        "VERIFYING CLAIMS AGAINST ACADEMIC LITERATURE",
                "generate_answer":     "SYNTHESIZING CITATIONS & RESPONSE",
            }

            for updates in graph.stream(input_state, config, stream_mode="updates"):
                for node_name, node_state in updates.items():
                    if node_name in NODE_LABELS:
                        status_placeholder.markdown(f"""
                        <div class="agent-step-pill">
                          <span class="status-pulse-cyan"></span> {NODE_LABELS[node_name]}
                        </div>
                        """, unsafe_allow_html=True)
                    if node_name == "generate_answer":
                        answer = node_state.get("answer", "")
                        if answer:
                            response_text = answer

            status_placeholder.empty()

            if not response_text:
                final_values = graph.get_state(config).values
                response_text = final_values.get("answer") or "No response generated."

            placeholder.markdown(response_text)

            final_values = graph.get_state(config).values
            state_snapshot = _serialize_state(final_values)

            citations = state_snapshot.get("source_citations") or []
            render_claim_verification_card(state_snapshot)
            render_source_citations(citations)
            render_telemetry_trace(current_turn, state_snapshot)
            render_export_note_button(
                turn_num=current_turn,
                user_query=prompt,
                assistant_answer=response_text,
                citations=citations,
                key=f"dl_live_{current_turn}",
            )

        st.session_state.chats[active_sid].append({
            "role": "assistant",
            "content": response_text,
            "graph_state": state_snapshot,
            "turn": current_turn,
        })

        if is_first_message:
            st.rerun()
