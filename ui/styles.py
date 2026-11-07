"""Custom design system, typography tokens, and CSS overrides for Papeer."""

import streamlit as st


def inject_custom_css() -> None:
    """Injects the high-craft Palantir / Linear / Vercel dark theme into the Streamlit DOM."""
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
.main {
  background: transparent !important;
}

/* 1. Keep stHeader in the DOM (transparent) so the sidebar '>>' reopen button survives */
header[data-testid="stHeader"] {
  background: transparent !important;
  border-bottom: none !important;
  pointer-events: none !important;
}

/* Re-enable pointer clicks ONLY on the sidebar expand button inside the header */
header[data-testid="stHeader"] button,
[data-testid="collapsedControl"],
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"] {
  pointer-events: auto !important;
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  color: #CBD5E1 !important;
  background: rgba(15, 18, 30, 0.85) !important;
  border: 1px solid rgba(99, 102, 241, 0.35) !important;
  border-radius: 8px !important;
  box-shadow: 0 0 15px rgba(99, 102, 241, 0.15) !important;
  transition: all 0.15s ease !important;
}

[data-testid="collapsedControl"]:hover,
[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapsedControl"]:hover {
  color: #22D3EE !important;
  border-color: #22D3EE !important;
  background: rgba(20, 25, 42, 0.95) !important;
}

/* 2. Hide ONLY the right-side Deploy button, status widget, and 3-dot menu */
[data-testid="stAppDeployButton"],
.stAppDeployButton,
[data-testid="stMainMenu"],
#MainMenu,
footer,
[data-testid="stStatusWidget"],
[data-testid="stDecoration"] {
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
  max-width: 1020px !important;
  width: 100% !important;
  margin: 0 auto !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
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

/* Add clean horizontal spacing between tab buttons */
[data-testid="stTabs"] [data-baseweb="tab-list"],
[data-testid="stTabs"] [role="tablist"] {
  display: flex !important;
  gap: 24px !important;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
  padding-bottom: 0px !important;
  margin-bottom: 14px !important;
  background: transparent !important;
  border-top: none !important;
  border-left: none !important;
  border-right: none !important;
}

/* Individual tab styling */
[data-testid="stTabs"] [data-baseweb="tab"],
[data-testid="stTabs"] button[role="tab"] {
  padding: 8px 4px 10px 4px !important;
  margin: 0 !important;
  font-family: 'Inter', -apple-system, sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  color: #64748B !important;
  background: transparent !important;
  border: none !important;
  transition: color 0.15s ease !important;
}

[data-testid="stTabs"] [data-baseweb="tab"] p,
[data-testid="stTabs"] button[role="tab"] p {
  color: inherit !important;
  font-size: 13px !important;
  font-weight: inherit !important;
  margin: 0 !important;
}

[data-testid="stTabs"] [data-baseweb="tab"]:hover,
[data-testid="stTabs"] button[role="tab"]:hover {
  color: #CBD5E1 !important;
  background: transparent !important;
}

[data-testid="stTabs"] [data-baseweb="tab"]:hover p,
[data-testid="stTabs"] button[role="tab"]:hover p {
  color: #CBD5E1 !important;
}

/* Active tab: crisp white text + cyan/indigo underline (NO RED) */
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: #F8FAFC !important;
  font-weight: 600 !important;
  background: transparent !important;
  border: none !important;
}

[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] p,
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] span {
  color: #F8FAFC !important;
  font-weight: 600 !important;
}

/* Override Streamlit's red tab highlight bar */
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-baseweb="tab-highlight"] {
  background-color: #22D3EE !important;
  height: 2px !important;
  border-radius: 2px !important;
  box-shadow: 0 0 10px rgba(34, 211, 238, 0.6) !important;
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
button[kind="primary"] p,
button[kind="primary"] span,
button[kind="primary"] div,
[data-testid="stBaseButton-primary"] *,
[data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-primary"] span,
[data-testid="stBaseButton-primary"] div,
.stButton > button[kind="primary"] *,
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span {
  color: #FFFFFF !important;
  font-weight: 600 !important;
  opacity: 1 !important;
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

/* Ensure crisp contrast on disabled primary buttons */
button:disabled,
button[disabled],
button:disabled *,
button[disabled] *,
[data-testid="stBaseButton-primary"]:disabled,
[data-testid="stBaseButton-primary"]:disabled * {
  color: rgba(255, 255, 255, 0.85) !important;
  opacity: 0.85 !important;
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

/* Ensure the HUD container never clips its left/right children */
.pipeline-hud-bar, .hud-pipeline, .langgraph-hud {
  display: flex !important;
  flex-wrap: nowrap !important;
  justify-content: space-between !important;
  align-items: center !important;
  gap: 6px !important;
  width: 100% !important;
  max-width: 100% !important;
  overflow: visible !important;
  margin: 14px auto 18px auto !important;
  box-sizing: border-box !important;
}

/* Make each HUD node compact enough that all 5 fit cleanly across 100% width */
.pipeline-hud-bar > div, .hud-node, .pipeline-node {
  flex: 1 1 auto !important;
  text-align: center !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10px !important;
  padding: 5px 7px !important;
  white-space: nowrap !important;
  background: rgba(15, 18, 30, 0.8) !important;
  border: 1px solid rgba(99, 102, 241, 0.28) !important;
  border-radius: 6px !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 0 15px rgba(99, 102, 241, 0.08) !important;
  color: #CBD5E1 !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 5px !important;
  backdrop-filter: blur(8px) !important;
  transition: all 0.2s ease !important;
}

.pipeline-hud-bar > div:hover, .hud-node:hover, .pipeline-node:hover {
  border-color: #22D3EE !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.12), 0 0 20px rgba(34, 211, 238, 0.2) !important;
  transform: translateY(-1px) !important;
}

.node-num {
  color: #22D3EE !important;
  font-weight: 700 !important;
  text-shadow: 0 0 8px rgba(34, 211, 238, 0.4) !important;
}

.node-label {
  letter-spacing: 0.03em !important;
}

.pipeline-arrow {
  color: rgba(99, 102, 241, 0.45) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10px !important;
  user-select: none !important;
  flex: 0 0 auto !important;
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

/* ─── EXPORT RESEARCH NOTE (.MD) GHOST PILL BUTTON ───────────────── */
[data-testid="stDownloadButton"] {
  display: inline-block !important;
  margin-top: 6px !important;
  margin-bottom: 4px !important;
}

[data-testid="stDownloadButton"] > button {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: 0.03em !important;
  padding: 4px 12px !important;
  min-height: 28px !important;
  height: 28px !important;
  background: rgba(15, 18, 30, 0.65) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  color: #94A3B8 !important;
  border-radius: 6px !important;
  box-shadow: none !important;
  transition: all 0.15s ease !important;
  display: inline-flex !important;
  align-items: center !important;
  gap: 6px !important;
}

[data-testid="stDownloadButton"] > button:hover {
  background: rgba(22, 28, 48, 0.9) !important;
  border-color: #22D3EE !important;
  color: #22D3EE !important;
  box-shadow: 0 0 12px rgba(34, 211, 238, 0.25) !important;
  transform: translateY(-1px) !important;
}

[data-testid="stDownloadButton"] > button p,
[data-testid="stDownloadButton"] > button span,
[data-testid="stDownloadButton"] > button div {
  color: inherit !important;
  font-size: 11px !important;
  font-weight: inherit !important;
}

[data-testid="stBottom"] {
  background: linear-gradient(0deg, #06070B 65%, transparent 100%) !important;
}
</style>
""", unsafe_allow_html=True)





