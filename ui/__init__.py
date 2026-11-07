"""UI package for Papeer: precision design system tokens, styles, and reusable cockpit components."""

from ui.components import (
    render_claim_verification_card,
    render_export_note_button,
    render_hero_hud,
    render_source_citations,
    render_telemetry_trace,
)
from ui.styles import inject_custom_css

__all__ = [
    "inject_custom_css",
    "render_hero_hud",
    "render_source_citations",
    "render_telemetry_trace",
    "render_claim_verification_card",
    "render_export_note_button",
]

