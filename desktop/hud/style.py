"""CSS y tablas visuales del HUD (sin gi; el CSS se aplica en window.py)."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CSS — paleta cian/neón sobre fondo oscuro translúcido
# ---------------------------------------------------------------------------

CSS = """
/* ============================================================
   Crotolamo HUD  —  paleta Jarvis/Neon (GTK3)
   ============================================================ */

/* wrapper externo transparente (necesario para el visual RGBA) */
.hud-root {
    background: transparent;
}

/* tarjeta principal */
.hud-card {
    background: rgba(5, 10, 25, 0.82);
    border-radius: 20px;
    border: 1.5px solid rgba(0, 210, 255, 0.30);
    padding: 18px 24px 20px 24px;
    box-shadow: 0 0 32px rgba(0, 210, 255, 0.15),
                0 4px 20px rgba(0, 0, 0, 0.55);
}

/* marca / título */
.hud-brand {
    font-family: monospace;
    font-weight: 900;
    font-size: 11pt;
    letter-spacing: 5px;
    color: rgba(0, 210, 255, 0.75);
}

/* orbe central */
.hud-orb-wrap {
    padding: 6px 0 10px 0;
}

.hud-orb {
    border-radius: 999px;
    padding: 20px;
    border: 2px solid rgba(0, 210, 255, 0.20);
    background: rgba(0, 210, 255, 0.06);
    min-width: 80px;
    min-height: 80px;
}

/* ---- estados del orbe por modo ---- */

.mode-idle .hud-orb {
    border-color: rgba(100, 120, 140, 0.25);
    background: rgba(40, 50, 60, 0.30);
}
.mode-idle .hud-orb image {
    color: rgba(120, 140, 160, 0.60);
}

.mode-listening .hud-orb {
    border-color: rgba(0, 210, 255, 0.70);
    background: rgba(0, 210, 255, 0.12);
    box-shadow: 0 0 22px rgba(0, 210, 255, 0.35),
                inset 0 0 12px rgba(0, 210, 255, 0.10);
    animation: pulse-listen 1.2s ease-in-out infinite;
}
.mode-listening .hud-orb image {
    color: #00d2ff;
}

.mode-thinking .hud-orb {
    border-color: rgba(100, 80, 255, 0.70);
    background: rgba(90, 60, 220, 0.14);
    box-shadow: 0 0 26px rgba(100, 80, 255, 0.40),
                inset 0 0 14px rgba(100, 80, 255, 0.12);
    animation: pulse-think 1.6s ease-in-out infinite;
}
.mode-thinking .hud-orb image {
    color: #a07fff;
}

.mode-speaking .hud-orb {
    border-color: rgba(0, 255, 180, 0.70);
    background: rgba(0, 200, 140, 0.12);
    box-shadow: 0 0 26px rgba(0, 255, 180, 0.40),
                inset 0 0 14px rgba(0, 200, 140, 0.10);
    animation: pulse-speak 0.9s ease-in-out infinite;
}
.mode-speaking .hud-orb image {
    color: #00ffb4;
}

/* spinner GTK para "thinking" (Gtk.Spinner, auto-animado) */
.hud-spinner {
    color: #a07fff;
}

/* etiqueta de modo */
.hud-mode-label {
    font-family: monospace;
    font-size: 9pt;
    letter-spacing: 3px;
    color: rgba(0, 210, 255, 0.70);
    padding-top: 2px;
}

/* texto reconocido / respuesta del asistente */
.hud-text {
    font-size: 10.5pt;
    color: rgba(200, 230, 255, 0.90);
    padding-top: 8px;
}

/* separador decorativo */
.hud-sep {
    background: rgba(0, 210, 255, 0.12);
    min-height: 1px;
    margin: 6px 0;
}

/* ---- @keyframes (GTK3: box-shadow, opacity, color) ---- */

@keyframes pulse-listen {
    0%   { box-shadow: 0 0 10px rgba(0,210,255,0.20), inset 0 0 6px rgba(0,210,255,0.08); }
    50%  { box-shadow: 0 0 32px rgba(0,210,255,0.55), inset 0 0 16px rgba(0,210,255,0.18); }
    100% { box-shadow: 0 0 10px rgba(0,210,255,0.20), inset 0 0 6px rgba(0,210,255,0.08); }
}

@keyframes pulse-think {
    0%   { box-shadow: 0 0 12px rgba(100,80,255,0.25), inset 0 0 8px rgba(100,80,255,0.10); }
    50%  { box-shadow: 0 0 36px rgba(100,80,255,0.60), inset 0 0 18px rgba(100,80,255,0.20); }
    100% { box-shadow: 0 0 12px rgba(100,80,255,0.25), inset 0 0 8px rgba(100,80,255,0.10); }
}

@keyframes pulse-speak {
    0%   { box-shadow: 0 0 10px rgba(0,255,180,0.20), inset 0 0 6px rgba(0,200,140,0.08); }
    50%  { box-shadow: 0 0 34px rgba(0,255,180,0.55), inset 0 0 16px rgba(0,200,140,0.18); }
    100% { box-shadow: 0 0 10px rgba(0,255,180,0.20), inset 0 0 6px rgba(0,200,140,0.08); }
}
"""

# ---------------------------------------------------------------------------
# Labels e iconos por modo
# ---------------------------------------------------------------------------

MODE_LABELS: dict[str, str] = {
    "idle":      "  IDLE  ",
    "listening": "ESCUCHANDO",
    "thinking":  "PENSANDO",
    "speaking":  "HABLANDO",
}

MODE_ICONS: dict[str, str] = {
    "idle":      "microphone-sensitivity-low-symbolic",
    "listening": "microphone-sensitivity-high-symbolic",
    "thinking":  "system-search-symbolic",
    "speaking":  "audio-volume-high-symbolic",
}
