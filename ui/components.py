"""Reusable, high-craft UI presentation components for Papeer."""

from datetime import datetime
import streamlit as st


def render_hero_hud() -> None:
    """Renders the empty-state hero header and the 5-stage LangGraph architecture HUD pipeline bar."""
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
        <div class="pipeline-node"><span class="node-num">01</span> <span class="node-label">INGEST · PyMuPDF4LLM</span></div>
        <span class="pipeline-arrow">──►</span>
        <div class="pipeline-node"><span class="node-num">02</span> <span class="node-label">ROUTER · Pydantic</span></div>
        <span class="pipeline-arrow">──►</span>
        <div class="pipeline-node"><span class="node-num">03</span> <span class="node-label">HYBRID · Qdrant+BM25</span></div>
        <span class="pipeline-arrow">──►</span>
        <div class="pipeline-node"><span class="node-num">04</span> <span class="node-label">RERANK · FlashRank</span></div>
        <span class="pipeline-arrow">──►</span>
        <div class="pipeline-node"><span class="node-num">05</span> <span class="node-label">SYNTHESIS · 120B</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_source_citations(citations: list[dict]) -> None:
    """Renders the expandable verified source citations drawer."""
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


def render_telemetry_trace(turn_num: int, state_snapshot: dict) -> None:
    """Renders the per-turn LangGraph telemetry expander with status badges and raw state JSON."""
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


def render_claim_verification_card(state_snapshot: dict) -> None:
    """
    Renders a dedicated HUD Verdict Banner when a claim verification check was performed.
    Displays color-coded badge: Emerald (Supported/Valid), Amber (Superseded/Partially Valid),
    or Rose (Refuted/Outdated), along with glass cards for any superseding papers.
    """
    if not state_snapshot:
        return

    route = state_snapshot.get("route")
    verdict = state_snapshot.get("claim_verdict")
    papers = state_snapshot.get("superseding_papers") or []

    # Only render if route is verify_claim or claim_verdict/superseding_papers are present
    if route != "verify_claim" and not verdict and not papers:
        return

    verdict_text = (verdict or "").strip()
    v_lower = verdict_text.lower()

    if any(k in v_lower for k in ["supersed", "partial", "replaced", "qualified"]):
        badge_label = "SUPERSEDED / QUALIFIED"
        badge_color = "#F59E0B"
        badge_bg = "rgba(245, 158, 11, 0.12)"
        badge_border = "rgba(245, 158, 11, 0.35)"
        banner_border = "#F59E0B"
    elif any(k in v_lower for k in ["refut", "false", "outdated", "disproven", "incorrect"]):
        badge_label = "REFUTED / OUTDATED"
        badge_color = "#F43F5E"
        badge_bg = "rgba(244, 63, 94, 0.12)"
        badge_border = "rgba(244, 63, 94, 0.35)"
        banner_border = "#F43F5E"
    else:
        badge_label = "LITERATURE SUPPORTED"
        badge_color = "#10B981"
        badge_bg = "rgba(16, 185, 129, 0.12)"
        badge_border = "rgba(16, 185, 129, 0.35)"
        banner_border = "#10B981"

    papers_html = ""
    if papers:
        paper_items = []
        for p in papers:
            title = p.get("title", "Academic Literature")
            summary = p.get("summary") or p.get("findings", "")
            url = p.get("url", "#")
            paper_items.append(f"""
            <div style="
              background: #0D101A;
              border: 1px solid rgba(255, 255, 255, 0.07);
              border-left: 3px solid #6366F1;
              border-radius: 6px;
              padding: 10px 14px;
              margin-top: 8px;
            ">
              <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 8px;">
                <span style="font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 600; color: #F8FAFC;">
                  {title}
                </span>
                <a href="{url}" target="_blank" style="
                  font-family: 'JetBrains Mono', monospace;
                  font-size: 11px;
                  color: #38BDF8;
                  text-decoration: none;
                  background: rgba(56, 189, 248, 0.1);
                  border: 1px solid rgba(56, 189, 248, 0.25);
                  border-radius: 4px;
                  padding: 2px 7px;
                  white-space: nowrap;
                ">arXiv / Source ↗</a>
              </div>
              <div style="font-family: 'Inter', sans-serif; font-size: 12px; color: #94A3B8; margin-top: 5px; line-height: 1.45;">
                {summary}
              </div>
            </div>
            """)
        papers_html = "".join(paper_items)

    verdict_display = f"""
    <div style="font-family: 'Inter', sans-serif; font-size: 13.5px; color: #F8FAFC; line-height: 1.55; margin-top: 6px;">
      {verdict_text if verdict_text else "Claim analysis completed against academic index."}
    </div>
    """ if verdict_text else ""

    st.markdown(f"""
    <div style="
      background: rgba(13, 16, 26, 0.85);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-left: 3px solid {banner_border};
      border-radius: 8px;
      padding: 12px 16px;
      margin: 10px 0 14px 0;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    ">
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap;">
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; font-weight: 600;">
          LITERATURE CLAIM VERIFICATION
        </span>
        <span style="
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px;
          font-weight: 700;
          letter-spacing: 0.04em;
          color: {badge_color};
          background: {badge_bg};
          border: 1px solid {badge_border};
          padding: 2px 8px;
          border-radius: 4px;
        ">
          {badge_label}
        </span>
      </div>
      {verdict_display}
      {papers_html}
    </div>
    """, unsafe_allow_html=True)


def render_export_note_button(
    turn_num: int,
    user_query: str,
    assistant_answer: str,
    citations: list[dict],
    key: str,
) -> None:
    """Renders a 1-click Markdown export button beneath assistant responses."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Papeer Research Note · Turn {turn_num}",
        f"**Generated:** {timestamp}",
        "",
        "## Research Query",
        f"> {user_query}" if user_query else "> *N/A*",
        "",
        "## Synthesized Findings",
        assistant_answer or "*No response content.*",
        "",
    ]

    if citations:
        lines.append("## Verified Source Citations")
        for cit in citations:
            idx = cit.get("index", 1)
            title = cit.get("title", "Document")
            section = cit.get("section", "General")
            page = cit.get("page", 1)
            snippet = cit.get("snippet", "").strip()
            lines.append(f"- `[{idx}]` **{title}** — *Section: {section} (Page {page})*")
            if snippet:
                lines.append(f'  > "{snippet}"')
        lines.append("")

    markdown_payload = "\n".join(lines)
    st.download_button(
        label="⬇ Export Note (.md)",
        data=markdown_payload,
        file_name=f"papeer_research_turn_{turn_num}.md",
        mime="text/markdown",
        key=key,
    )

