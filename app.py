import streamlit as st  # must be first for secrets bridge
import streamlit_secrets_bridge  # noqa: F401 — injects st.secrets → os.environ

import base64
import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from backend.btw_handler import handle_btw
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.rag_graph import build_graph
from backend.vector_store import add_paper, list_papers

st.set_page_config(
    page_title="Papeer — Precision AI Engineering Cockpit",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  FUTURISTIC AI ENGINEERING COCKPIT (PALANTIR / LINEAR / VERCEL DESIGN SYSTEM)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@1,6..72,400;1,6..72,500&display=swap');

/* ─── SYSTEM DESIGN TOKENS ─────────────────────────────────────── */
:root {
  --bg-app:              #06070B;
  --bg-sidebar:          rgba(8, 10, 16, 0.94);
  --bg-surface:          rgba(13, 16, 26, 0.70);
  --bg-surface-elevated: #161824;
  --bg-surface-hover:    #1A1D2D;
  
  --border-hairline:      rgba(255, 255, 255, 0.08);
  --border-muted:         rgba(99, 102, 241, 0.28);
  --border-active:        rgba(34, 211, 238, 0.5);
  
  --accent-indigo:        #6366F1;
  --accent-cyan:          #22D3EE;
  --accent-cyan-glow:     rgba(34, 211, 238, 0.4);
  
  --text-primary:         #F8FAFC;
  --text-secondary:       #CBD5E1;
  --text-muted:           #64748B;
  --text-faint:           #475569;
  
  --status-emerald:       #10B981;
  --status-amber:         #F59E0B;
  --status-cyan:          #22D3EE;
  
  --font-sans:            'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono:            'JetBrains Mono', monospace;
  --font-serif:           'Newsreader', Georgia, serif;
}

* { box-sizing: border-box; }
::selection { background: rgba(34, 211, 238, 0.25); color: #FFFFFF; }

/* ─── PROCEDURAL BLUEPRINT GRID & DUAL AURORA GLOW ──────────────── */
html, body, [data-testid="stAppViewContainer"] {
  background-color: #06070B !important;
  color: var(--text-primary) !important;
  font-family: var(--font-sans) !important;
  -webkit-font-smoothing: antialiased;
}

[data-testid="stAppViewContainer"], .stApp {
  background-color: #06070B !important;
  background-image: 
    /* Top Dual Spotlight Aurora Light Cones */
    radial-gradient(ellipse 70% 45% at 50% -10%, rgba(99, 102, 241, 0.20), transparent 70%),
    radial-gradient(ellipse 40% 30% at 85% 10%, rgba(6, 182, 212, 0.12), transparent 60%),
    /* Architectural Tech Micro-Grid (32px x 32px) */
    linear-gradient(to right, rgba(255, 255, 255, 0.035) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(255, 255, 255, 0.035) 1px, transparent 1px) !important;
  background-size: 100% 100%, 100% 100%, 32px 32px, 32px 32px !important;
  background-attachment: fixed !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main,
[data-testid="stHeader"] {
  background: transparent !important;
}

/* Hide top-right Streamlit Deploy button, header bar & default chrome */
[data-testid="stToolbar"],
[data-testid="stAppDeployButton"],
.stAppDeployButton,
header[data-testid="stHeader"],
#MainMenu, footer, [data-testid="stDecoration"] {
  display: none !important;
  visibility: hidden !important;
}

/* ─── SCROLLBAR ─────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #06070B; }
::-webkit-scrollbar-thumb { background: #262938; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #3B425E; }

/* ─── MAIN CONTAINER (WIDENED FOR 5-NODE HUD & BENTO CARDS) ────── */
.block-container,
[data-testid="stMainBlockContainer"] {
  padding-top: 2.2rem !important;
  max-width: 960px !important;
  width: 100% !important;
  margin: 0 auto !important;
  padding-bottom: 8rem !important;
}

/* ─── SIDEBAR: MINIMALIST PRECISION CONSOLE ─────────────────────── */
[data-testid="stSidebar"] {
  background: rgba(8, 10, 16, 0.94) !important;
  backdrop-filter: blur(20px) !important;
  -webkit-backdrop-filter: blur(20px) !important;
  border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}

[data-testid="stSidebarContent"] {
  padding: 1.25rem 1rem !important;
}

/* Sidebar Buttons Default (Flat & Left-Aligned) */
[data-testid="stSidebar"] .stButton > button {
  border: 1px solid transparent !important;
  background: transparent !important;
  justify-content: flex-start !important;
  align-items: center !important;
  text-align: left !important;
  padding: 8px 12px !important;
  min-height: 36px !important;
  height: auto !important;
  border-radius: 6px !important;
  color: #94A3B8 !important;
  box-shadow: none !important;
  transition: all 0.15s ease !important;
}

[data-testid="stSidebar"] .stButton > button > div,
[data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] {
  justify-content: flex-start !important;
  align-items: center !important;
  text-align: left !important;
  width: 100% !important;
  display: flex !important;
}

[data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] p {
  text-align: left !important;
  justify-content: flex-start !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  max-width: 210px !important;
  font-size: 13px !important;
  margin: 0 !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(255, 255, 255, 0.04) !important;
  color: #F8FAFC !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* Inactive Session item specifically */
[data-testid="stSidebar"] .stButton:not(:first-of-type) > button:not([kind="primary"]),
[data-testid="stSidebar"] .stButton:not(:first-of-type) > button[kind="secondary"] {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
}

/* Top '+ New Chat' Command Bar */
[data-testid="stSidebar"] .stButton:first-of-type > button {
  background: linear-gradient(180deg, rgba(30, 35, 54, 0.7) 0%, rgba(18, 21, 34, 0.9) 100%) !important;
  border: 1px solid rgba(99, 102, 241, 0.35) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 0 15px rgba(99, 102, 241, 0.12) !important;
  color: #F8FAFC !important;
  justify-content: center !important;
  text-align: center !important;
  margin-bottom: 14px !important;
  font-weight: 500 !important;
}

[data-testid="stSidebar"] .stButton:first-of-type > button > div,
[data-testid="stSidebar"] .stButton:first-of-type > button [data-testid="stMarkdownContainer"] {
  justify-content: center !important;
  text-align: center !important;
}

[data-testid="stSidebar"] .stButton:first-of-type > button:hover {
  border-color: #6366F1 !important;
  background: linear-gradient(180deg, rgba(36, 42, 65, 0.8) 0%, rgba(22, 26, 42, 0.95) 100%) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.15), 0 0 20px rgba(99, 102, 241, 0.25) !important;
  color: #FFFFFF !important;
}

/* Active Session in Sidebar (Subtle glowing left border) */
[data-testid="stSidebar"] .stButton:not(:first-of-type) > button[kind="primary"] {
  background: rgba(22, 25, 37, 0.85) !important;
  border: 1px solid rgba(99, 102, 241, 0.3) !important;
  border-left: 3px solid #6366F1 !important;
  color: #F8FAFC !important;
  box-shadow: 0 0 15px rgba(99, 102, 241, 0.1) !important;
  justify-content: flex-start !important;
  text-align: left !important;
}

[data-testid="stSidebar"] .stButton:not(:first-of-type) > button[kind="primary"] * {
  color: #F8FAFC !important;
  font-weight: 500 !important;
}

/* ─── SMOKED GLASS TABS & INGESTION DECK ─────────────────────────── */
[data-testid="stTabs"] {
  background: rgba(13, 16, 26, 0.68) !important;
  backdrop-filter: blur(16px) !important;
  -webkit-backdrop-filter: blur(16px) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  box-shadow: 
    inset 0 1px 0 0 rgba(255, 255, 255, 0.1),
    0 20px 40px -15px rgba(0, 0, 0, 0.7),
    0 0 30px -5px rgba(99, 102, 241, 0.08) !important;
  border-radius: 12px !important;
  padding: 12px 18px 18px 18px !important;
}

[data-testid="stTabs"] [role="tablist"] {
  background: rgba(10, 12, 20, 0.8) !important;
  border: 1px solid rgba(255, 255, 255, 0.06) !important;
  border-radius: 8px 8px 0 0 !important;
  padding: 4px 6px !important;
  gap: 4px !important;
}

[data-baseweb="tab-highlight"] {
  background-color: #22D3EE !important;
  height: 2px !important;
  box-shadow: 0 0 10px rgba(34, 211, 238, 0.6) !important;
}

[data-testid="stTabs"] button[role="tab"] {
  font-family: var(--font-sans) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  color: #64748B !important;
  background: transparent !important;
  border: none !important;
  border-radius: 6px !important;
  padding: 6px 14px !important;
  transition: all 0.15s ease !important;
}

[data-testid="stTabs"] button[role="tab"]:hover {
  color: #F8FAFC !important;
  background: rgba(255, 255, 255, 0.05) !important;
}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: #F8FAFC !important;
  background: rgba(26, 32, 54, 0.7) !important;
  border: 1px solid rgba(99, 102, 241, 0.25) !important;
  font-weight: 600 !important;
}

[data-testid="stTabContent"] {
  background: transparent !important;
  border: none !important;
  padding: 1.1rem 0 0 0 !important;
  box-shadow: none !important;
}

/* ─── FUTURISTIC DROPZONE & INPUTS ───────────────────────────────── */
[data-testid="stFileUploader"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}

[data-testid="stFileUploaderDropzone"] {
  background: rgba(9, 11, 18, 0.75) !important;
  border: 1px dashed rgba(99, 102, 241, 0.35) !important;
  border-radius: 8px !important;
  transition: all 0.2s ease !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
  border-color: #22D3EE !important;
  background: rgba(15, 20, 35, 0.85) !important;
  box-shadow: 0 0 25px rgba(6, 182, 212, 0.15) !important;
}

[data-testid="stFileUploaderDropzone"] button {
  background: #161824 !important;
  color: #F8FAFC !important;
  border: 1px solid #262938 !important;
  border-radius: 6px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  transition: all 0.15s ease !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
  background: #1E2232 !important;
  border-color: #22D3EE !important;
}

.stTextInput input, .stTextArea textarea {
  background: rgba(9, 11, 18, 0.8) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 6px !important;
  color: #F8FAFC !important;
  font-family: var(--font-sans) !important;
  font-size: 13px !important;
  padding: 0.65rem 0.85rem !important;
  transition: all 0.15s ease !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: #22D3EE !important;
  box-shadow: 0 0 0 1px #22D3EE, 0 0 15px rgba(34, 211, 238, 0.2) !important;
  outline: none !important;
}

/* ─── CYBER-INDIGO GRADIENT PRIMARY BUTTONS (KILL CORAL-RED) ─────── */
button[kind="primary"],
[data-testid="stBaseButton-primary"],
.stButton > button[kind="primary"],
.main .stButton > button[kind="primary"],
[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #4F46E5 0%, #6366F1 50%, #0891B2 100%) !important;
  color: #FFFFFF !important;
  border: 1px solid rgba(165, 180, 252, 0.4) !important;
  box-shadow: 
    inset 0 1px 0 rgba(255, 255, 255, 0.25),
    0 4px 20px rgba(99, 102, 241, 0.35) !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  border-radius: 8px !important;
  height: 38px !important;
  min-height: 38px !important;
  transition: all 0.2s ease !important;
}

button[kind="primary"] *,
[data-testid="stBaseButton-primary"] *,
.stButton > button[kind="primary"] * {
  color: #FFFFFF !important;
  font-weight: 600 !important;
}

button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover,
.stButton > button[kind="primary"]:hover {
  transform: translateY(-1px) !important;
  box-shadow: 
    inset 0 1px 0 rgba(255, 255, 255, 0.35),
    0 6px 25px rgba(6, 182, 212, 0.5) !important;
  border-color: #67E8F9 !important;
}

/* ─── FROSTED GLASS BENTO STARTER CARDS (2-LINE HUD DESIGN) ─────── */
[data-testid="stHorizontalBlock"] .stButton > button:not([kind="primary"]) {
  display: flex !important;
  flex-direction: column !important;
  align-items: flex-start !important;
  justify-content: flex-start !important;
  text-align: left !important;
  padding: 16px 18px !important;
  height: auto !important;
  min-height: 86px !important;
  background: linear-gradient(180deg, rgba(19, 23, 36, 0.78) 0%, rgba(12, 14, 22, 0.88) 100%) !important;
  backdrop-filter: blur(12px) !important;
  -webkit-backdrop-filter: blur(12px) !important;
  border: 1px solid rgba(255, 255, 255, 0.07) !important;
  box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.06), 0 8px 20px rgba(0, 0, 0, 0.4) !important;
  border-radius: 10px !important;
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-testid="stHorizontalBlock"] .stButton > button:not([kind="primary"]):hover {
  border-color: rgba(99, 102, 241, 0.5) !important;
  background: linear-gradient(180deg, rgba(25, 30, 48, 0.88) 0%, rgba(15, 18, 30, 0.95) 100%) !important;
  box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.12), 0 0 25px rgba(99, 102, 241, 0.18) !important;
  transform: translateY(-2px) !important;
}

[data-testid="stHorizontalBlock"] .stButton > button [data-testid="stMarkdownContainer"] {
  text-align: left !important;
  width: 100% !important;
}

[data-testid="stHorizontalBlock"] .stButton > button [data-testid="stMarkdownContainer"] p:first-child {
  color: #F8FAFC !important;
  font-size: 13.5px !important;
  font-weight: 600 !important;
  margin-bottom: 6px !important;
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
  flex-wrap: wrap !important;
}

[data-testid="stHorizontalBlock"] .stButton > button [data-testid="stMarkdownContainer"] code {
  font-family: var(--font-mono) !important;
  font-size: 10px !important;
  color: #38BDF8 !important;
  background: rgba(56, 189, 248, 0.1) !important;
  border: 1px solid rgba(56, 189, 248, 0.25) !important;
  padding: 2px 6px !important;
  border-radius: 4px !important;
}

[data-testid="stHorizontalBlock"] .stButton > button [data-testid="stMarkdownContainer"] p:last-child {
  color: #94A3B8 !important;
  font-size: 12.5px !important;
  font-weight: 400 !important;
  margin: 0 !important;
  line-height: 1.45 !important;
}

/* 1-Click Benchmark Action Chip */
[data-testid="stHorizontalBlock"] .stButton > button:not([kind="primary"]):not(:has(p:last-child:not(:first-child))) {
  background: rgba(15, 18, 30, 0.8) !important;
  border: 1px solid rgba(34, 211, 238, 0.35) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 0 20px rgba(34, 211, 238, 0.1) !important;
  border-radius: 999px !important;
  padding: 6px 18px !important;
  min-height: 38px !important;
  height: 38px !important;
  transition: all 0.2s ease !important;
  justify-content: center !important;
  align-items: center !important;
  text-align: center !important;
}

[data-testid="stHorizontalBlock"] .stButton > button:not([kind="primary"]):not(:has(p:last-child:not(:first-child))):hover {
  border-color: #22D3EE !important;
  background: rgba(22, 28, 48, 0.9) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.15), 0 0 25px rgba(34, 211, 238, 0.3) !important;
  transform: translateY(-1px) !important;
}

[data-testid="stHorizontalBlock"] .stButton > button:not([kind="primary"]):not(:has(p:last-child:not(:first-child))) * {
  font-family: var(--font-mono) !important;
  font-size: 11.5px !important;
  color: #38BDF8 !important;
  letter-spacing: 0.02em !important;
  font-weight: 500 !important;
}

[data-testid="stHorizontalBlock"] .stButton > button:not([kind="primary"]):not(:has(p:last-child:not(:first-child))):hover * {
  color: #FFFFFF !important;
}

/* ─── HERO TELEMETRY PILL & PIPELINE HUD ─────────────────────────── */
.hero-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #CBD5E1;
  background: rgba(15, 18, 30, 0.75);
  border: 1px solid rgba(99, 102, 241, 0.35);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 0 20px rgba(99, 102, 241, 0.12);
  border-radius: 999px;
  padding: 5px 14px;
  margin-bottom: 1.1rem;
  backdrop-filter: blur(12px);
}

.pulse-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #22D3EE;
  box-shadow: 0 0 10px #22D3EE, 0 0 18px rgba(34, 211, 238, 0.6);
  display: inline-block;
  animation: pulseDotCyan 2s infinite ease-in-out;
}

@keyframes pulseDotCyan {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(1.2); }
}

.hud-pipeline, .langgraph-hud, .pipeline-hud-bar {
  display: flex !important;
  flex-wrap: nowrap !important;
  justify-content: center !important;
  align-items: center !important;
  gap: 6px !important;
  white-space: nowrap !important;
  margin: 0.8rem auto 1.3rem auto !important;
  max-width: 100% !important;
  overflow-x: auto !important;
}

.pipeline-node {
  background: rgba(15, 18, 30, 0.75);
  border: 1px solid rgba(99, 102, 241, 0.28);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 0 20px rgba(99, 102, 241, 0.08);
  border-radius: 6px;
  padding: 4px 8px !important;
  font-family: var(--font-mono);
  font-size: 10px !important;
  color: #CBD5E1;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  backdrop-filter: blur(8px);
  white-space: nowrap !important;
  flex-shrink: 0 !important;
  transition: all 0.2s ease;
}

.pipeline-node:hover {
  border-color: #22D3EE;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.12), 0 0 20px rgba(34, 211, 238, 0.2);
  transform: translateY(-1px);
}

.node-num {
  color: #22D3EE;
  font-weight: 700;
  text-shadow: 0 0 8px rgba(34, 211, 238, 0.4);
}

.node-label {
  letter-spacing: 0.03em;
}

.pipeline-arrow {
  color: rgba(99, 102, 241, 0.45);
  font-family: var(--font-mono);
  font-size: 10px !important;
  user-select: none;
  flex-shrink: 0 !important;
}

/* ─── LIVE AGENT EXECUTION TELEMETRY BADGE ───────────────────────── */
.agent-step-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(15, 18, 30, 0.85);
  border: 1px solid rgba(34, 211, 238, 0.3);
  box-shadow: 0 0 15px rgba(34, 211, 238, 0.1);
  border-radius: 999px;
  padding: 4px 12px;
  margin: 0.5rem 0;
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.06em;
  color: #CBD5E1;
  text-transform: uppercase;
}

.status-pulse-cyan {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #22D3EE;
  box-shadow: 0 0 8px rgba(34, 211, 238, 0.8);
  display: inline-block;
  animation: pulseCyan 1.6s infinite ease-in-out;
}

@keyframes pulseCyan {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.35; transform: scale(1.2); }
}

.status-pulse-emerald {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #10B981;
  box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
  display: inline-block;
  animation: pulseEmerald 2s infinite ease-in-out;
}

@keyframes pulseEmerald {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(1.1); }
}

/* ─── CHAT MESSAGES ─────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 0.35rem 0 !important;
}

/* User Message: Precision Obsidian with Cyan Accent */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: rgba(22, 24, 36, 0.85) !important;
  backdrop-filter: blur(12px) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-left: 3px solid #22D3EE !important;
  border-radius: 8px !important;
  padding: 1rem 1.25rem !important;
  margin: 0.5rem 0 !important;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
}

/* Assistant Message: Deep Matte Obsidian with Hairline Border */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
  background: rgba(14, 16, 23, 0.85) !important;
  backdrop-filter: blur(12px) !important;
  border: 1px solid rgba(255, 255, 255, 0.07) !important;
  border-radius: 8px !important;
  padding: 1rem 1.25rem !important;
  margin: 0.5rem 0 !important;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
}

[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
  background: #161822 !important;
  border: 1px solid #262938 !important;
  border-radius: 6px !important;
}

/* ─── CHAT INPUT BAR ────────────────────────────────────────────── */
[data-testid="stChatInput"] {
  background: rgba(17, 19, 28, 0.92) !important;
  backdrop-filter: blur(16px) !important;
  border: 1px solid rgba(255, 255, 255, 0.09) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06), 0 8px 30px rgba(0, 0, 0, 0.5) !important;
  border-radius: 10px !important;
  transition: all 0.15s ease !important;
}

[data-testid="stChatInput"]:focus-within {
  border-color: #22D3EE !important;
  box-shadow: 0 0 0 1px #22D3EE, 0 4px 25px rgba(34, 211, 238, 0.18) !important;
}

[data-testid="stChatInputTextArea"] {
  background: transparent !important;
  color: #F8FAFC !important;
  font-family: var(--font-sans) !important;
  font-size: 13.5px !important;
  caret-color: #22D3EE !important;
}

/* ─── EXPANDERS & TELEMETRY CONSOLE ──────────────────────────────── */
[data-testid="stExpander"] {
  background: rgba(13, 15, 22, 0.85) !important;
  border: 1px solid rgba(255, 255, 255, 0.07) !important;
  border-radius: 6px !important;
  margin-top: 0.5rem !important;
}

[data-testid="stExpander"] summary {
  font-family: var(--font-mono) !important;
  font-size: 11px !important;
  letter-spacing: 0.05em !important;
  color: #94A3B8 !important;
  text-transform: uppercase !important;
  padding: 0.4rem 0.6rem !important;
}

[data-testid="stJson"] {
  background: #08090E !important;
  font-family: var(--font-mono) !important;
  font-size: 11.5px !important;
  border: 1px solid #1A1D27 !important;
  border-radius: 6px !important;
}

/* Prose Markdown inside Chat */
[data-testid="stMarkdownContainer"] code:not(pre code) {
  background: #141722 !important;
  color: #C7D2FE !important;
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
  padding: 2px 6px !important;
  border-radius: 4px !important;
  border: 1px solid #22263A !important;
}

[data-testid="stMarkdownContainer"] pre {
  background: #08090E !important;
  border: 1px solid #1E202B !important;
  border-left: 2px solid #22D3EE !important;
  border-radius: 6px !important;
  padding: 0.85rem !important;
}

[data-testid="stMarkdownContainer"] pre code {
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
  color: #F8FAFC !important;
}

[data-testid="stMarkdownContainer"] a {
  color: #38BDF8 !important;
  text-decoration: underline !important;
  text-underline-offset: 3px !important;
}

[data-testid="stMarkdownContainer"] a:hover {
  color: #7DD3FC !important;
}

[data-testid="stBottom"] {
  background: linear-gradient(0deg, #06070B 65%, transparent 100%) !important;
}
</style>
""", unsafe_allow_html=True)


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
    st.markdown("""
    <div style="text-align: center; padding: 1.8rem 0 0.8rem 0;">
      <!-- Hero Telemetry Pill -->
      <div class="hero-pill">
        <span class="pulse-dot"></span>LANGGRAPH v1.1 STATE MACHINE · QDRANT IN-MEMORY · DEEPEVAL 84.4% PRECISION
      </div>
      
      <!-- Editorial Title -->
      <h1 style="font-family: 'Inter', sans-serif; font-size: 40px; font-weight: 600; letter-spacing: -0.03em; color: #F8FAFC; margin: 0 0 0.75rem 0; line-height: 1.15;">
        Research papers, <span style="font-family: 'Newsreader', Georgia, serif; font-style: italic; font-weight: 500; color: #CBD5E1;">distilled.</span>
      </h1>
      
      <!-- Subtitle -->
      <p style="font-family: 'Inter', sans-serif; font-size: 14px; color: #64748B; max-width: 540px; margin: 0 auto 1.3rem auto; line-height: 1.55;">
        Drop any research paper, arXiv link, or academic claim to extract instant summaries, verify citations against recent literature, and synthesize findings.
      </p>

      <!-- Live LangGraph Architecture HUD Pipeline Bar -->
      <div class="pipeline-hud-bar hud-pipeline langgraph-hud">
        <div class="pipeline-node"><span class="node-num">01</span> <span class="node-label">INGEST: PyMuPDF/ArXiv</span></div>
        <span class="pipeline-arrow">──►</span>
        <div class="pipeline-node"><span class="node-num">02</span> <span class="node-label">ROUTER: Pydantic</span></div>
        <span class="pipeline-arrow">──►</span>
        <div class="pipeline-node"><span class="node-num">03</span> <span class="node-label">RETRIEVE: Qdrant/Tavily</span></div>
        <span class="pipeline-arrow">──►</span>
        <div class="pipeline-node"><span class="node-num">04</span> <span class="node-label">JUDGE: Self-Correction</span></div>
        <span class="pipeline-arrow">──►</span>
        <div class="pipeline-node"><span class="node-num">05</span> <span class="node-label">SYNTHESIS: GPT-OSS-120B</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

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

_tab_files, _tab_web, _tab_arxiv, _tab_loaded = st.tabs(
    ["Upload Files", "Web Pages", "ArXiv Papers", "Loaded Documents"]
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
def render_source_citations(citations: list[dict]):
    if not citations:
        return
    with st.expander(f"📚 VERIFIED SOURCE CITATIONS ({len(citations)})", expanded=False):
        for cit in citations:
            st.markdown(f"""
            <div style="
              background: rgba(13, 16, 26, 0.7);
              border: 1px solid rgba(34, 211, 238, 0.2);
              border-left: 3px solid #22D3EE;
              border-radius: 6px;
              padding: 8px 12px;
              margin-bottom: 8px;
            ">
              <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">
                <span style="font-family:'JetBrains Mono',monospace; font-weight:700; color:#22D3EE; font-size:12px;">[{cit.get('index', 1)}] {cit.get('title', 'Document')}</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#94A3B8;">Section: {cit.get('section', 'General')} · Page {cit.get('page', 1)}</span>
              </div>
              <div style="font-family:'Newsreader',Georgia,serif; font-style:italic; font-size:12px; color:#CBD5E1; margin-top:5px; line-height:1.4;">
                "{cit.get('snippet', '')}..."
              </div>
            </div>
            """, unsafe_allow_html=True)


def render_telemetry_trace(turn_num: int, state_snapshot: dict):
    with st.expander(f"TELEMETRY TRACE · TURN {turn_num}", expanded=False):
        if state_snapshot:
            route = state_snapshot.get("route") or "retrieve"
            relevant = state_snapshot.get("is_relevant")
            rewrites = state_snapshot.get("rewrite_count", 0)
            attempts = state_snapshot.get("retrieval_attempts", 0)
            verdict = state_snapshot.get("claim_verdict")

            relevant_badge = (
                f'<span style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);color:#10B981;padding:2px 8px;border-radius:4px;">relevant: True</span>'
                if relevant is True else
                f'<span style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.25);color:#F59E0B;padding:2px 8px;border-radius:4px;">relevant: {relevant}</span>'
            )
            verdict_badge = (
                f'<span style="background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.3);color:#F59E0B;padding:2px 8px;border-radius:4px;">verdict: <strong>{verdict}</strong></span>'
                if verdict else ""
            )

            st.markdown(f"""
            <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;font-family:'JetBrains Mono',monospace;font-size:11px;">
              <span style="background:rgba(15,18,30,0.85);border:1px solid rgba(255,255,255,0.08);color:#94A3B8;padding:2px 8px;border-radius:4px;">route: <strong style="color:#F8FAFC;">{route}</strong></span>
              {relevant_badge}
              <span style="background:rgba(15,18,30,0.85);border:1px solid rgba(255,255,255,0.08);color:#94A3B8;padding:2px 8px;border-radius:4px;">rewrites: <strong style="color:#F8FAFC;">{rewrites}</strong></span>
              <span style="background:rgba(15,18,30,0.85);border:1px solid rgba(255,255,255,0.08);color:#94A3B8;padding:2px 8px;border-radius:4px;">attempts: <strong style="color:#F8FAFC;">{attempts}</strong></span>
              {verdict_badge}
            </div>
            """, unsafe_allow_html=True)
        st.json(state_snapshot)

# ══════════════════════════════════════════════════════════════════════════════
#  CHAT TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
for msg in active_chats:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            citations = msg.get("graph_state", {}).get("source_citations") or []
            render_source_citations(citations)
            render_telemetry_trace(msg.get("turn", 1), msg.get("graph_state", {}))

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
            render_source_citations(citations)
            render_telemetry_trace(current_turn, state_snapshot)

        st.session_state.chats[active_sid].append({
            "role": "assistant",
            "content": response_text,
            "graph_state": state_snapshot,
            "turn": current_turn,
        })

        if is_first_message:
            st.rerun()
