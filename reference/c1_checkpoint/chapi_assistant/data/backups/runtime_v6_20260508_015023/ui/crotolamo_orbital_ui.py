"""
CROTOLAMO ORBITAL UI
Interfaz Tkinter cyberpunk/memética para el Crotolamo local.

Fase 5: usa core/crotolamo_runtime.py como tronco común, con modos de proyecto y contexto local.
La UI ya no le habla directo a veinte cables sueltos: pasa por runtime.
Milagro menor, pero milagro.
"""

from __future__ import annotations

import json
import math
import os
import platform
import queue
import random
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import tkinter as tk
from tkinter import messagebox, simpledialog

# Permite ejecutar desde la raíz del proyecto aunque el archivo esté dentro de ui/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Runtime v5: tronco común oficial. Si falla, caemos al modo viejo para no
# convertir una mejora en un pisapapeles con neón.
try:
    from core.crotolamo_runtime import CrotolamoRuntime
except Exception as runtime_error:  # pragma: no cover
    CrotolamoRuntime = None
    RUNTIME_IMPORT_ERROR = runtime_error
else:
    RUNTIME_IMPORT_ERROR = None

# Importación por capas: la UI no debe morir completa solo porque la voz o
# algún módulo se puso dramático. Pequeño milagro de ingeniería civilizada.
try:
    from core.chapi_shell import ask_ollama, normalize_plan
    from core.skills import handle_direct_skill
except Exception as import_error:  # pragma: no cover
    ask_ollama = None
    normalize_plan = None
    handle_direct_skill = None
    CORE_IMPORT_ERROR = import_error
else:
    CORE_IMPORT_ERROR = None

try:
    from core.voice_in import listen_once
except Exception as voice_in_error:  # pragma: no cover
    listen_once = None
    VOICE_IN_ERROR = voice_in_error
else:
    VOICE_IN_ERROR = None

try:
    from core.voice_out import speak
except Exception as voice_out_error:  # pragma: no cover
    speak = None
    VOICE_OUT_ERROR = voice_out_error
else:
    VOICE_OUT_ERROR = None

IMPORT_ERROR = CORE_IMPORT_ERROR
LOG_DIR = PROJECT_ROOT / "data" / "orbital_logs"
CONFIG_DIR = PROJECT_ROOT / "config"
SETTINGS_PATH = CONFIG_DIR / "orbital_ui.json"


# -----------------------------
# Tema visual
# -----------------------------

@dataclass(frozen=True)
class NeonTheme:
    bg: str = "#02040c"
    bg_2: str = "#060a18"
    panel: str = "#07101f"
    panel_2: str = "#0b1428"
    cyan: str = "#00e5ff"
    cyan_2: str = "#0077ff"
    blue: str = "#1a4dff"
    purple: str = "#8b3dff"
    magenta: str = "#ff2bd6"
    green: str = "#00ff9d"
    yellow: str = "#ffd166"
    red: str = "#ff3864"
    text: str = "#d7f7ff"
    muted: str = "#7da4c2"
    dark_text: str = "#041018"


THEME = NeonTheme()


class OrbitalButton(tk.Canvas):
    """Botón de canvas con borde neón."""

    def __init__(
        self,
        master: tk.Misc,
        text: str,
        command: Callable[[], None] | None = None,
        width: int = 170,
        height: int = 38,
        accent: str = THEME.cyan,
        active: bool = False,
    ) -> None:
        super().__init__(
            master,
            width=width,
            height=height,
            bg=THEME.panel,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.command = command
        self.text = text
        self.accent = accent
        self.active = active
        self.width_px = width
        self.height_px = height
        self._hover = False
        self._draw()
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _draw(self) -> None:
        self.delete("all")
        fill = "#09233d" if self.active or self._hover else THEME.panel
        glow = self.accent if self.active or self._hover else "#16435a"
        self.create_rectangle(2, 2, self.width_px - 2, self.height_px - 2, outline=glow, width=1, fill=fill)
        self.create_line(8, 2, 34, 2, fill=self.accent, width=2)
        self.create_line(self.width_px - 34, self.height_px - 2, self.width_px - 8, self.height_px - 2, fill=self.accent, width=2)
        self.create_text(18, self.height_px / 2, text="◇", fill=self.accent, font=("JetBrains Mono", 12, "bold"), anchor="w")
        self.create_text(45, self.height_px / 2, text=self.text, fill=THEME.text, font=("JetBrains Mono", 10, "bold"), anchor="w")

    def _on_enter(self, _event: tk.Event) -> None:
        self._hover = True
        self._draw()

    def _on_leave(self, _event: tk.Event) -> None:
        self._hover = False
        self._draw()

    def _on_click(self, _event: tk.Event) -> None:
        if self.command:
            self.command()

    def set_text(self, text: str, active: bool | None = None, accent: str | None = None) -> None:
        self.text = text
        if active is not None:
            self.active = active
        if accent is not None:
            self.accent = accent
        self._draw()


class OrbitalPanel(tk.Frame):
    """Frame con header estilo HUD."""

    def __init__(self, master: tk.Misc, title: str, accent: str = THEME.cyan, **kwargs: Any) -> None:
        super().__init__(master, bg=THEME.panel, highlightbackground=accent, highlightthickness=1, **kwargs)
        self.title = title
        self.accent = accent
        header = tk.Frame(self, bg=THEME.bg_2, height=28)
        header.pack(fill="x")
        tk.Label(
            header,
            text=f"◇ {title}",
            bg=THEME.bg_2,
            fg=accent,
            font=("JetBrains Mono", 10, "bold"),
            anchor="w",
            padx=8,
        ).pack(side="left", fill="both", expand=True)
        tk.Label(
            header,
            text="×",
            bg=THEME.bg_2,
            fg=THEME.muted,
            font=("JetBrains Mono", 10, "bold"),
            padx=8,
        ).pack(side="right")
        self.body = tk.Frame(self, bg=THEME.panel)
        self.body.pack(fill="both", expand=True, padx=8, pady=8)


class CoreCanvas(tk.Canvas):
    """Núcleo circular animado."""

    def __init__(self, master: tk.Misc, width: int = 560, height: int = 430) -> None:
        super().__init__(master, width=width, height=height, bg=THEME.bg, highlightthickness=0, bd=0)
        self.width_px = width
        self.height_px = height
        self.t = 0.0
        self.after(50, self.animate)

    def animate(self) -> None:
        self.t += 0.04
        self.draw()
        self.after(50, self.animate)

    def draw(self) -> None:
        self.delete("all")
        w, h = self.width_px, self.height_px
        cx, cy = w // 2, h // 2

        # Fondo de túnel/grid
        for i in range(0, 18):
            y = int(h * (i / 18) ** 1.7)
            self.create_line(0, y, w, y, fill="#06284a", width=1)
        for i in range(-10, 11):
            x0 = cx + i * 22
            self.create_line(x0, h, cx + i * 3, 0, fill="#063a62", width=1)

        # Estrellas mínimas
        random.seed(67)
        for _ in range(90):
            x = random.randint(0, w)
            y = random.randint(0, h)
            c = random.choice(["#113355", "#1d7cff", "#c8f7ff", "#7e48ff"])
            self.create_oval(x, y, x + 1, y + 1, fill=c, outline="")

        # Cubos wireframe falsos, porque hacer física 3D en Tkinter sería castigar al silicio
        for base_x, base_y, size, phase in [(78, 310, 58, 0), (430, 305, 46, 1.7), (380, 72, 36, 3.2)]:
            wobble = math.sin(self.t + phase) * 6
            self._wire_cube(base_x + wobble, base_y - wobble, size)

        # Anillos HUD
        rings = [72, 102, 132, 162]
        for idx, radius in enumerate(rings):
            start = (self.t * 45 + idx * 38) % 360
            for k in range(0, 360, 42):
                extent = 22 + 8 * math.sin(self.t + idx)
                color = [THEME.cyan, THEME.cyan_2, THEME.purple, "#19d3ff"][idx % 4]
                self.create_arc(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    start=start + k,
                    extent=extent,
                    style="arc",
                    outline=color,
                    width=5 if idx in {1, 2} else 2,
                )
            self.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline="#0d426d", width=1)

        # Ticks
        for a in range(0, 360, 10):
            rad = math.radians(a + self.t * 12)
            r1 = 178
            r2 = 188 if a % 30 == 0 else 183
            x1, y1 = cx + math.cos(rad) * r1, cy + math.sin(rad) * r1
            x2, y2 = cx + math.cos(rad) * r2, cy + math.sin(rad) * r2
            self.create_line(x1, y1, x2, y2, fill="#1dbdff", width=1)

        # Núcleo geométrico
        points = []
        for i in range(6):
            a = self.t + math.radians(i * 60)
            points.append((cx + math.cos(a) * 32, cy + math.sin(a) * 32))
        for i, p1 in enumerate(points):
            p2 = points[(i + 1) % len(points)]
            self.create_line(*p1, *p2, fill=THEME.cyan, width=2)
            self.create_line(cx, cy, *p1, fill="#3388ff", width=1)
        self.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill=THEME.cyan, outline="")

        self.create_text(cx, cy + 68, text="NÚCLEO", fill=THEME.cyan, font=("JetBrains Mono", 20, "bold"))
        self.create_text(cx, cy + 94, text="ESTABLE", fill=THEME.green, font=("JetBrains Mono", 11, "bold"))
        self.create_text(cx, cy + 125, text="67%", fill=THEME.cyan, font=("JetBrains Mono", 26, "bold"))

    def _wire_cube(self, x: float, y: float, size: float) -> None:
        s = size
        d = size * 0.32
        color = "#168dff"
        self.create_rectangle(x, y, x + s, y + s, outline=color, width=1)
        self.create_rectangle(x + d, y - d, x + s + d, y + s - d, outline=color, width=1)
        self.create_line(x, y, x + d, y - d, fill=color)
        self.create_line(x + s, y, x + s + d, y - d, fill=color)
        self.create_line(x, y + s, x + d, y + s - d, fill=color)
        self.create_line(x + s, y + s, x + s + d, y + s - d, fill=color)


class Sparkline(tk.Canvas):
    def __init__(self, master: tk.Misc, width: int = 330, height: int = 96, accent: str = THEME.cyan) -> None:
        super().__init__(master, width=width, height=height, bg=THEME.panel, highlightthickness=0)
        self.width_px = width
        self.height_px = height
        self.accent = accent
        self.values = [random.random() for _ in range(70)]
        self.after(120, self.animate)

    def animate(self) -> None:
        self.values = self.values[1:] + [max(0.05, min(1.0, self.values[-1] + random.uniform(-0.2, 0.25)))]
        self.draw()
        self.after(140, self.animate)

    def draw(self) -> None:
        self.delete("all")
        w, h = self.width_px, self.height_px
        for i in range(0, w, 28):
            self.create_line(i, 0, i, h, fill="#0a2740")
        for i in range(0, h, 24):
            self.create_line(0, i, w, i, fill="#0a2740")
        pts = []
        for idx, val in enumerate(self.values):
            x = idx * (w / (len(self.values) - 1))
            y = h - val * (h - 12) - 6
            pts.extend([x, y])
        if len(pts) >= 4:
            self.create_line(*pts, fill=self.accent, width=2, smooth=True)
            shifted = []
            for i, coord in enumerate(pts):
                shifted.append(coord if i % 2 == 0 else min(h, coord + 8))
            self.create_line(*shifted, fill=THEME.purple, width=1, smooth=True)


class CrotolamoOrbitalUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CROTOLAMO :: Orbital Suite v5 Project Modes")
        self.geometry("1420x860")
        self.minsize(1180, 720)
        self.configure(bg=THEME.bg)

        self.result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.current_plan: dict[str, Any] | None = None
        self.running = False
        self.listening = False
        self.last_prompt = ""
        self.voice_enabled = self._load_voice_enabled()
        self.session_log_path = self._new_session_log_path()
        self.runtime = CrotolamoRuntime(PROJECT_ROOT) if CrotolamoRuntime is not None else None
        self.mode_var = tk.StringVar(value=self._active_mode_key())
        self.mode_detail_labels: dict[str, tk.Label] = {}
        self.real_metric_labels: dict[str, tk.Label] = {}
        self.status_canvas: tk.Canvas | None = None

        self._build_ui()
        self.after(80, self._poll_queue)
        self.after(1200, self._refresh_real_panels)

    # -----------------------------
    # Construcción visual
    # -----------------------------
    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=0)
        self.rowconfigure(1, weight=1)

        self._build_topbar()
        self._build_sidebar()
        self._build_center()
        self._build_rightbar()
        self._build_bottom_prompt()

        self._log("SISTEMA", f"Log de sesión: {self.session_log_path}")
        if RUNTIME_IMPORT_ERROR is not None:
            self._log("ALERTA", f"Runtime v5 no disponible: {RUNTIME_IMPORT_ERROR}")
        if CORE_IMPORT_ERROR is not None and self.runtime is None:
            self._log("ALERTA", f"No pude importar el núcleo de Crotolamo: {CORE_IMPORT_ERROR}")
            self._set_status("IMPORT ERROR", THEME.red)
        else:
            self._log("SISTEMA", "Crotolamo Orbital UI v5 cargado. Modos de proyecto y contexto local activados, porque responder todo en genérico era demasiado siglo pasado.")
            self._set_status("ONLINE", THEME.green)

        if VOICE_IN_ERROR is not None:
            self._log("VOZ-IN", f"Entrada de voz no disponible: {VOICE_IN_ERROR}")
        if VOICE_OUT_ERROR is not None:
            self._log("VOZ-OUT", f"Salida de voz no disponible: {VOICE_OUT_ERROR}")
        self._update_connection_panel()
        self._update_mode_panel()

    def _build_topbar(self) -> None:
        top = tk.Frame(self, bg=THEME.bg, height=64)
        top.grid(row=0, column=0, columnspan=3, sticky="ew", padx=12, pady=(10, 0))
        top.columnconfigure(1, weight=1)

        title_box = tk.Frame(top, bg=THEME.bg)
        title_box.grid(row=0, column=0, sticky="w")
        tk.Label(title_box, text="CROTOLAMO", bg=THEME.bg, fg=THEME.cyan, font=("JetBrains Mono", 28, "bold")).pack(anchor="w")
        tk.Label(title_box, text="クロトラモ  ·  ORBITAL SUITE  ·  v5 PROJECT MODES", bg=THEME.bg, fg=THEME.muted, font=("JetBrains Mono", 9)).pack(anchor="w")

        slogan = tk.Label(
            top,
            text="EL CAOS TAMBIÉN ES UN SISTEMA",
            bg=THEME.bg,
            fg=THEME.magenta,
            font=("JetBrains Mono", 13, "bold"),
        )
        slogan.grid(row=0, column=1, sticky="n", pady=8)

        status_box = tk.Frame(top, bg=THEME.panel, highlightbackground=THEME.cyan_2, highlightthickness=1)
        status_box.grid(row=0, column=2, sticky="e")
        tk.Label(status_box, text="USUARIO: CAOS ORBITAL", bg=THEME.panel, fg=THEME.text, font=("JetBrains Mono", 9, "bold"), padx=12, pady=4).pack(anchor="w")
        self.status_label = tk.Label(status_box, text="INICIANDO", bg=THEME.panel, fg=THEME.yellow, font=("JetBrains Mono", 9, "bold"), padx=12, pady=4)
        self.status_label.pack(anchor="w")

    def _build_sidebar(self) -> None:
        side = tk.Frame(self, bg=THEME.bg, width=210)
        side.grid(row=1, column=0, sticky="ns", padx=(12, 6), pady=10)
        side.grid_propagate(False)

        meme_wall = tk.Canvas(side, width=205, height=118, bg="#17091f", highlightthickness=1, highlightbackground=THEME.purple)
        meme_wall.pack(fill="x", pady=(0, 8))
        meme_wall.create_text(20, 18, text="☆", fill=THEME.yellow, font=("JetBrains Mono", 30, "bold"), anchor="w")
        meme_wall.create_text(72, 24, text="LATIN\nMAFIA", fill=THEME.magenta, font=("JetBrains Mono", 15, "bold"), anchor="w")
        meme_wall.create_text(18, 78, text="MALO DEL\nCUENTO", fill=THEME.text, font=("JetBrains Mono", 11, "bold"), anchor="w")
        meme_wall.create_text(130, 78, text="NO ES BUG\nES FUNCIÓN", fill=THEME.yellow, font=("JetBrains Mono", 8, "bold"), anchor="center")

        for idx, item in enumerate(["NÚCLEO", "SISTEMA", "LABORATORIO", "ARCHIVOS", "TERMINAL", "RED ORBITAL", "MÓDULOS", "BITÁCORA"]):
            OrbitalButton(side, item, width=205, active=(idx == 0), command=lambda name=item: self._log("NAV", f"{name}: sección visual lista."))\
                .pack(fill="x", pady=4)

        quick = tk.Frame(side, bg=THEME.bg)
        quick.pack(fill="x", pady=(8, 0))
        OrbitalButton(quick, "ABRIR RAÍZ", width=205, accent=THEME.yellow, command=self.open_project_root).pack(fill="x", pady=4)
        OrbitalButton(quick, "DIAGNÓSTICO", width=205, accent=THEME.green, command=self.run_diagnostics).pack(fill="x", pady=4)

        self._build_mode_selector(side)

        tk.Frame(side, bg=THEME.bg).pack(fill="both", expand=True)
        alert = tk.Frame(side, bg="#160823", highlightbackground=THEME.magenta, highlightthickness=1)
        alert.pack(fill="x", pady=(8, 0))
        tk.Label(alert, text="ALERTA MEMÉTICA", bg="#160823", fg=THEME.magenta, font=("JetBrains Mono", 10, "bold"), pady=4).pack()
        tk.Label(alert, text="NIVEL DE ABSURDO:\nMÁXIMO", bg="#160823", fg=THEME.text, font=("JetBrains Mono", 9), pady=6).pack()

    def _build_center(self) -> None:
        center = tk.Frame(self, bg=THEME.bg)
        center.grid(row=1, column=1, sticky="nsew", padx=4, pady=10)
        center.columnconfigure(0, weight=0)
        center.columnconfigure(1, weight=1)
        center.rowconfigure(0, weight=1)
        center.rowconfigure(1, weight=0)

        left_stack = tk.Frame(center, bg=THEME.bg, width=310)
        left_stack.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        left_stack.grid_propagate(False)

        op = OrbitalPanel(left_stack, "OPERADOR PRINCIPAL")
        op.pack(fill="x", pady=(0, 8))
        portrait = tk.Canvas(op.body, width=270, height=175, bg="#061226", highlightthickness=0)
        portrait.pack(fill="x")
        self._draw_operator_portrait(portrait)
        tk.Label(op.body, text="● ESTADO: ENFOCADO", bg=THEME.panel, fg=THEME.green, font=("JetBrains Mono", 9, "bold")).pack(anchor="w", pady=(8, 0))
        tk.Label(op.body, text="Hackeando el sistema\ncomo si fuera un juego.", bg=THEME.panel, fg=THEME.text, font=("JetBrains Mono", 9), justify="left").pack(anchor="w")

        mascot = OrbitalPanel(left_stack, "MASCOTA ASISTENTE", accent=THEME.purple)
        mascot.pack(fill="x", pady=(0, 8))
        mcan = tk.Canvas(mascot.body, width=270, height=108, bg=THEME.panel, highlightthickness=0)
        mcan.pack(fill="x")
        self._draw_pezlin(mcan)

        hacker = OrbitalPanel(left_stack, "HACKER MODE", accent=THEME.green)
        hacker.pack(fill="x")
        hcan = tk.Canvas(hacker.body, width=270, height=118, bg="#03130a", highlightthickness=0)
        hcan.pack(fill="x")
        self._draw_hacker_meme(hcan)

        core_area = tk.Frame(center, bg=THEME.bg)
        core_area.grid(row=0, column=1, sticky="nsew")
        core_area.columnconfigure(0, weight=1)
        core_panel = OrbitalPanel(core_area, "ORBITAL CORE")
        core_panel.pack(fill="both", expand=True)
        self.core_canvas = CoreCanvas(core_panel.body, width=600, height=455)
        self.core_canvas.pack(fill="both", expand=True)

        bottom = tk.Frame(center, bg=THEME.bg)
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)

        graph = OrbitalPanel(bottom, "ACTIVIDAD EN TIEMPO REAL", accent=THEME.purple)
        graph.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        Sparkline(graph.body, width=470, height=112, accent=THEME.cyan).pack(fill="x")

        term = OrbitalPanel(bottom, "TERMINAL ORBITAL", accent=THEME.green)
        term.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.log_text = tk.Text(
            term.body,
            height=7,
            bg="#02070e",
            fg=THEME.green,
            insertbackground=THEME.cyan,
            relief="flat",
            font=("JetBrains Mono", 9),
            wrap="word",
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

    def _build_rightbar(self) -> None:
        right = tk.Frame(self, bg=THEME.bg, width=330)
        right.grid(row=1, column=2, sticky="ns", padx=(6, 12), pady=10)
        right.grid_propagate(False)

        mode_panel = OrbitalPanel(right, "MODO ACTIVO v5", accent=THEME.magenta)
        mode_panel.pack(fill="x", pady=(0, 8))
        for key in ["MODO", "RUTA", "EXISTE"]:
            row = tk.Frame(mode_panel.body, bg=THEME.panel)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{key}:", bg=THEME.panel, fg=THEME.muted, font=("JetBrains Mono", 8, "bold"), width=8, anchor="w").pack(side="left")
            lab = tk.Label(row, text="--", bg=THEME.panel, fg=THEME.text, font=("JetBrains Mono", 8), anchor="w", wraplength=210, justify="left")
            lab.pack(side="left", fill="x", expand=True)
            self.mode_detail_labels[key] = lab

        state = OrbitalPanel(right, "ESTADO REAL DEL SISTEMA")
        state.pack(fill="x", pady=(0, 8))
        scan = tk.Canvas(state.body, width=290, height=120, bg=THEME.panel, highlightthickness=0)
        scan.pack(fill="x")
        self.status_canvas = scan
        self._draw_status(scan)
        metrics = tk.Frame(state.body, bg=THEME.panel)
        metrics.pack(fill="x", pady=(4, 0))
        for key in ["CPU", "RAM", "DISCO", "BATERÍA", "OLLAMA", "MIC", "AUDIO"]:
            row = tk.Frame(metrics, bg=THEME.panel)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{key}:", bg=THEME.panel, fg=THEME.muted, font=("JetBrains Mono", 8, "bold"), width=9, anchor="w").pack(side="left")
            lab = tk.Label(row, text="--", bg=THEME.panel, fg=THEME.text, font=("JetBrains Mono", 8), anchor="w")
            lab.pack(side="left", fill="x", expand=True)
            self.real_metric_labels[key] = lab

        conn = OrbitalPanel(right, "CONEXIONES REALES", accent=THEME.green)
        conn.pack(fill="x", pady=(0, 8))
        self.connection_text = tk.Text(
            conn.body,
            height=5,
            bg="#02070e",
            fg=THEME.text,
            relief="flat",
            font=("JetBrains Mono", 8),
            wrap="word",
        )
        self.connection_text.pack(fill="x")
        self.connection_text.configure(state="disabled")

        pigeon = OrbitalPanel(right, "MÓDULO: PALOMA SUPREMA", accent=THEME.magenta)
        pigeon.pack(fill="x", pady=(0, 8))
        pcan = tk.Canvas(pigeon.body, width=290, height=165, bg="#13061f", highlightthickness=0)
        pcan.pack(fill="x")
        self._draw_pigeon(pcan)

        modules = OrbitalPanel(right, "MÓDULOS ACTIVOS")
        modules.pack(fill="x", pady=(0, 8))
        for name, val in [("MOTOR CREATIVO", 0.84), ("IA EXPERIMENTAL", 0.67), ("GENERADOR DE CAOS", 0.91), ("COMPILADOR DE SUEÑOS", 0.58), ("INTERFAZ NEURAL", 0.77)]:
            self._module_row(modules.body, name, val)

        notes = OrbitalPanel(right, "NOTAS DEL LABORATORIO", accent=THEME.yellow)
        notes.pack(fill="both", expand=True)
        ncan = tk.Canvas(notes.body, width=290, height=120, bg="#201a14", highlightthickness=0)
        ncan.pack(fill="both", expand=True)
        self._draw_notes(ncan)

    def _build_bottom_prompt(self) -> None:
        bottom = tk.Frame(self, bg=THEME.bg)
        bottom.grid(row=2, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 12))
        bottom.columnconfigure(1, weight=1)

        tk.Label(bottom, text="PATRÓN >", bg=THEME.bg, fg=THEME.cyan, font=("JetBrains Mono", 12, "bold")).grid(row=0, column=0, padx=(0, 8))
        self.prompt_var = tk.StringVar()
        entry = tk.Entry(
            bottom,
            textvariable=self.prompt_var,
            bg="#02070e",
            fg=THEME.text,
            insertbackground=THEME.cyan,
            relief="flat",
            font=("JetBrains Mono", 12),
        )
        entry.grid(row=0, column=1, sticky="ew", ipady=10)
        entry.bind("<Return>", lambda _event: self.submit_prompt())
        entry.focus_set()

        OrbitalButton(bottom, "PENSAR", command=self.submit_prompt, width=120, accent=THEME.cyan).grid(row=0, column=2, padx=6)
        OrbitalButton(bottom, "ESCUCHAR", command=self.listen_from_mic, width=135, accent=THEME.magenta).grid(row=0, column=3, padx=6)
        self.voice_button = OrbitalButton(bottom, "VOZ: ON" if self.voice_enabled else "VOZ: OFF", command=self.toggle_voice, width=120, accent=THEME.green if self.voice_enabled else THEME.yellow, active=self.voice_enabled)
        self.voice_button.grid(row=0, column=4, padx=6)
        OrbitalButton(bottom, "EJECUTAR", command=self.execute_current_plan, width=130, accent=THEME.green).grid(row=0, column=5)

    # -----------------------------
    # Modos de proyecto v5
    # -----------------------------
    def _active_mode_key(self) -> str:
        try:
            if self.runtime is not None:
                return str(self.runtime.mode_manager.active_key())
        except Exception:
            pass
        return "crotolamo"

    def _mode_keys(self) -> list[str]:
        try:
            if self.runtime is not None:
                keys = self.runtime.mode_manager.mode_keys()
                return keys or ["crotolamo"]
        except Exception:
            pass
        return ["crotolamo", "huevonitis", "tletl", "fedora", "escuela", "laboratorio"]

    def _build_mode_selector(self, master: tk.Misc) -> None:
        box = tk.Frame(master, bg="#07101f", highlightbackground=THEME.magenta, highlightthickness=1)
        box.pack(fill="x", pady=(10, 4))
        tk.Label(box, text="MODO DE TRABAJO", bg="#07101f", fg=THEME.magenta, font=("JetBrains Mono", 10, "bold"), pady=5).pack(anchor="w", padx=8)
        menu = tk.OptionMenu(box, self.mode_var, *self._mode_keys(), command=lambda _v: self.set_mode_from_ui())
        menu.configure(bg=THEME.panel_2, fg=THEME.text, activebackground=THEME.bg_2, activeforeground=THEME.cyan, highlightthickness=0, font=("JetBrains Mono", 9), relief="flat")
        menu["menu"].configure(bg=THEME.panel_2, fg=THEME.text, activebackground=THEME.cyan_2, activeforeground=THEME.text, font=("JetBrains Mono", 9))
        menu.pack(fill="x", padx=8, pady=(0, 6))
        OrbitalButton(box, "ABRIR MODO", width=187, accent=THEME.yellow, command=self.open_active_mode).pack(fill="x", padx=8, pady=3)
        OrbitalButton(box, "TERMINAL MODO", width=187, accent=THEME.green, command=self.terminal_active_mode).pack(fill="x", padx=8, pady=3)
        OrbitalButton(box, "DOCTOR MODO", width=187, accent=THEME.cyan, command=self.diagnostics_active_mode).pack(fill="x", padx=8, pady=(3, 8))

    def set_mode_from_ui(self) -> None:
        key = self.mode_var.get().strip()
        if self.runtime is None:
            self._log("MODO", "Runtime no disponible; no puedo cambiar modo.")
            return
        try:
            result = self.runtime.set_mode(key)
            self._log("MODO", result.get("text", f"Modo: {key}"))
            self.mode_var.set(self.runtime.mode_manager.active_key())
            self._update_mode_panel()
            self._update_connection_panel()
        except Exception as error:
            self._log("MODO", f"No pude cambiar modo: {error}")

    def _submit_mode_command(self, command: str) -> None:
        self.prompt_var.set(command)
        self.submit_prompt()

    def open_active_mode(self) -> None:
        self._submit_mode_command("abrir modo")

    def terminal_active_mode(self) -> None:
        self._submit_mode_command("terminal modo")

    def diagnostics_active_mode(self) -> None:
        self._submit_mode_command("diagnóstico modo")

    def _update_mode_panel(self) -> None:
        if not hasattr(self, "mode_detail_labels"):
            return
        try:
            if self.runtime is None:
                mode_key = self.mode_var.get() or "crotolamo"
                values = {"MODO": mode_key, "RUTA": "runtime no disponible", "EXISTE": "?"}
            else:
                payload = self.runtime.mode_payload()
                mode = payload.get("mode", {})
                from pathlib import Path as _Path
                path = _Path(str(mode.get("path") or "~")).expanduser()
                values = {
                    "MODO": f"{mode.get('title')} ({mode.get('key')})",
                    "RUTA": str(path),
                    "EXISTE": "sí" if path.exists() else "no",
                }
                if self.mode_var.get() != self.runtime.mode_manager.active_key():
                    self.mode_var.set(self.runtime.mode_manager.active_key())
            for key, val in values.items():
                lab = self.mode_detail_labels.get(key)
                if lab is not None:
                    color = THEME.green if key == "EXISTE" and val == "sí" else THEME.red if key == "EXISTE" and val == "no" else THEME.text
                    lab.configure(text=str(val), fg=color)
        except Exception as error:
            self._log("MODO", f"No pude actualizar panel de modo: {error}")

    # -----------------------------
    # Dibujos/memes en canvas
    # -----------------------------
    def _draw_operator_portrait(self, c: tk.Canvas) -> None:
        w, h = 270, 175
        random.seed(12)
        for _ in range(55):
            x, y = random.randint(0, w), random.randint(0, h)
            c.create_oval(x, y, x + 2, y + 2, fill=random.choice([THEME.cyan, THEME.purple, "white"]), outline="")
        c.create_oval(95, 35, 178, 122, fill="#1b2f55", outline=THEME.cyan, width=2)
        c.create_polygon(90, 52, 124, 18, 178, 45, 140, 9, fill="#0d1830", outline=THEME.cyan)
        c.create_line(105, 67, 166, 58, fill="white", width=9)
        c.create_line(105, 69, 166, 60, fill=THEME.magenta, width=3)
        c.create_line(118, 112, 156, 108, fill=THEME.green, width=2)
        c.create_text(190, 34, text="♛", fill=THEME.yellow, font=("JetBrains Mono", 26, "bold"))
        c.create_rectangle(55, 128, 218, 160, outline=THEME.cyan, width=2)
        c.create_line(55, 128, 136, 100, 218, 128, fill="#0a77ff", width=1)
        c.create_text(136, 146, text="HOLOGRAMA", fill=THEME.cyan, font=("JetBrains Mono", 9, "bold"))

    def _draw_pezlin(self, c: tk.Canvas) -> None:
        c.create_oval(18, 28, 92, 100, fill="#5a2ca0", outline=THEME.purple, width=3)
        c.create_oval(33, 15, 54, 38, fill="#dbeaff", outline=THEME.cyan, width=2)
        c.create_oval(56, 15, 77, 38, fill="#dbeaff", outline=THEME.cyan, width=2)
        c.create_oval(41, 24, 48, 31, fill="#0b1020")
        c.create_oval(64, 24, 71, 31, fill="#0b1020")
        c.create_arc(39, 48, 78, 78, start=180, extent=180, outline=THEME.yellow, width=2)
        c.create_text(120, 30, text="PEZLÍN 3000", fill=THEME.cyan, font=("JetBrains Mono", 12, "bold"), anchor="w")
        c.create_text(120, 58, text="Asistente de caos\ny soluciones absurdas", fill=THEME.text, font=("JetBrains Mono", 9), anchor="w")
        c.create_text(120, 94, text="● ONLINE", fill=THEME.green, font=("JetBrains Mono", 9, "bold"), anchor="w")

    def _draw_hacker_meme(self, c: tk.Canvas) -> None:
        for x in range(0, 270, 14):
            for y in range(0, 118, 18):
                c.create_text(x, y, text=random.choice(["0", "1", "λ", "Σ"]), fill="#00ff66", font=("JetBrains Mono", 8))
        c.create_oval(22, 22, 92, 95, fill="#2f1f16", outline=THEME.green, width=2)
        c.create_line(35, 50, 78, 45, fill="white", width=8)
        c.create_line(35, 51, 78, 46, fill="#ff8bd6", width=3)
        c.create_arc(42, 62, 78, 84, start=190, extent=140, outline=THEME.green, width=2)
        c.create_rectangle(115, 38, 245, 92, fill="#1384d8", outline=THEME.cyan, width=2)
        c.create_text(180, 65, text="SHS", fill="white", font=("JetBrains Mono", 24, "bold"))
        c.create_text(135, 20, text="MATRIZ_MAFIOSA.OK", fill=THEME.green, font=("JetBrains Mono", 9, "bold"), anchor="w")
        c.create_text(135, 104, text="CONFIDENCIALIDAD: 67%", fill=THEME.text, font=("JetBrains Mono", 8), anchor="w")

    def _draw_status(self, c: tk.Canvas) -> None:
        cx, cy = 75, 78
        c.create_oval(cx - 55, cy - 55, cx + 55, cy + 55, outline="#16435a", width=8)
        c.create_arc(cx - 55, cy - 55, cx + 55, cy + 55, start=90, extent=-240, style="arc", outline=THEME.cyan, width=8)
        c.create_text(cx, cy, text="67", fill=THEME.cyan, font=("JetBrains Mono", 26, "bold"))
        rows = [("Integridad", "100%"), ("Seguridad", "67%"), ("Rendimiento", "88%"), ("Estabilidad", "90%"), ("Caos", "69%")]
        for i, (a, b) in enumerate(rows):
            y = 28 + i * 24
            c.create_text(155, y, text=a, fill=THEME.muted, font=("JetBrains Mono", 9), anchor="w")
            c.create_text(270, y, text=b, fill=THEME.cyan, font=("JetBrains Mono", 9, "bold"), anchor="e")

    def _draw_pigeon(self, c: tk.Canvas) -> None:
        random.seed(6)
        for _ in range(45):
            x, y = random.randint(0, 290), random.randint(0, 165)
            c.create_oval(x, y, x + 1, y + 1, fill=random.choice([THEME.cyan, THEME.purple, "white"]), outline="")
        c.create_oval(76, 35, 198, 132, fill="#8274cb", outline=THEME.purple, width=3)
        c.create_oval(106, 15, 170, 76, fill="#6a67aa", outline=THEME.cyan, width=2)
        c.create_oval(122, 36, 130, 44, fill=THEME.red, outline="")
        c.create_oval(150, 35, 158, 43, fill=THEME.red, outline="")
        c.create_line(126, 40, 275, 18, fill=THEME.red, width=4)
        c.create_line(154, 39, 280, 54, fill=THEME.red, width=4)
        c.create_arc(85, 58, 190, 142, start=200, extent=140, outline=THEME.yellow, width=5)
        c.create_text(142, 101, text="67", fill=THEME.yellow, font=("JetBrains Mono", 24, "bold"))
        c.create_text(34, 122, text="6", fill=THEME.yellow, font=("JetBrains Mono", 34, "bold"))
        c.create_text(236, 125, text="7", fill=THEME.yellow, font=("JetBrains Mono", 34, "bold"))
        c.create_text(143, 150, text="CONTROL TOTAL O NADA", fill=THEME.yellow, font=("JetBrains Mono", 10, "bold"))

    def _draw_notes(self, c: tk.Canvas) -> None:
        c.create_text(16, 18, text="ideas:", fill="#1a1712", font=("JetBrains Mono", 11, "bold"), anchor="w")
        c.create_rectangle(170, 30, 235, 92, outline="#1a1712", width=2)
        c.create_line(170, 30, 202, 12, 235, 30, fill="#1a1712", width=2)
        c.create_text(80, 74, text="☆  λ  BUG\nROMPER\nPARA CREAR", fill="#1a1712", font=("JetBrains Mono", 12, "bold"), anchor="center")
        for _ in range(20):
            x, y = random.randint(5, 280), random.randint(5, 115)
            c.create_text(x, y, text=random.choice(["+", "◇", "x", "67", "??"]), fill="#1a1712", font=("JetBrains Mono", 8))

    def _module_row(self, master: tk.Misc, name: str, val: float) -> None:
        row = tk.Frame(master, bg=THEME.panel)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=name, bg=THEME.panel, fg=THEME.text, font=("JetBrains Mono", 8), anchor="w").pack(side="left", fill="x", expand=True)
        bar = tk.Canvas(row, width=82, height=12, bg=THEME.panel, highlightthickness=0)
        bar.pack(side="right")
        bar.create_rectangle(1, 2, 80, 10, outline="#16435a")
        bar.create_rectangle(2, 3, 2 + int(76 * val), 9, fill=THEME.cyan, outline="")

    # -----------------------------
    # Lógica Crotolamo
    # -----------------------------
    def submit_prompt(self) -> None:
        prompt = self.prompt_var.get().strip()
        if not prompt:
            self._log("SISTEMA", "Escribe algo, patrón. La telepatía sigue en beta y qué desgracia.")
            return
        if self.running:
            self._log("SISTEMA", "Ya estoy procesando una petición. Un apocalipsis a la vez.")
            return
        if self.runtime is None and (CORE_IMPORT_ERROR is not None or ask_ollama is None or handle_direct_skill is None):
            self._log("ERROR", f"No está disponible el núcleo: {CORE_IMPORT_ERROR or RUNTIME_IMPORT_ERROR}")
            return

        self.last_prompt = prompt
        self.current_plan = None
        self.running = True
        self._set_status("PENSANDO", THEME.yellow)
        self._log("PATRÓN", prompt)
        self.prompt_var.set("")

        thread = threading.Thread(target=self._worker_prompt, args=(prompt,), daemon=True)
        thread.start()

    def _worker_prompt(self, prompt: str) -> None:
        try:
            if self.runtime is not None:
                result = self.runtime.process_text(prompt)
                if result.get("kind") == "direct":
                    self.result_queue.put(("direct", result.get("text", "")))
                    return
                plan = {
                    "safe": bool(result.get("safe", False)),
                    "risk": result.get("risk", "safe"),
                    "explanation": result.get("explanation") or result.get("text") or "Sin explicación.",
                    "commands": result.get("commands", []),
                    "meta": result.get("meta", {}),
                }
                self.result_queue.put(("plan", plan))
                return

            direct = handle_direct_skill(prompt) if handle_direct_skill else None
            if direct is not None:
                self.result_queue.put(("direct", direct))
                return
            if ask_ollama is None:
                raise RuntimeError("ask_ollama no está disponible.")
            plan = ask_ollama(prompt)
            if normalize_plan is not None:
                plan = normalize_plan(plan)
            self.result_queue.put(("plan", plan))
        except Exception as error:
            self.result_queue.put(("error", error))

    def execute_current_plan(self) -> None:
        if self.running:
            self._log("SISTEMA", "Todavía estoy ocupado. Ni los robots multitarea somos magia barata.")
            return
        if not self.current_plan:
            self._log("SISTEMA", "No hay plan listo para ejecutar.")
            return
        commands = self.current_plan.get("commands", [])
        safe = bool(self.current_plan.get("safe", False))
        risk = str(self.current_plan.get("risk", "safe"))
        safety = (self.current_plan.get("meta", {}) or {}).get("safety", {})
        if not commands:
            self._log("SISTEMA", "El plan no trae comandos. No voy a ejecutar aire.")
            return
        if not safe or risk == "blocked":
            self._log("SEGURIDAD", "Plan inseguro bloqueado por Seguridad v5.")
            messagebox.showwarning("Crotolamo Seguridad v5", "Plan bloqueado. No lo ejecuto desde la UI.")
            return

        preview = "\n".join(commands)
        if len(preview) > 1200:
            preview = preview[:1200] + "\n..."

        allow_confirm = False
        if risk == "confirm":
            details = []
            for check in safety.get("checks", []) if isinstance(safety, dict) else []:
                reasons = check.get("reasons", []) if isinstance(check, dict) else []
                if reasons:
                    details.extend(str(r) for r in reasons[:3])
            detail_text = "\n".join(f"- {r}" for r in details[:8]) or "- Modifica o ejecuta cosas fuera del modo solo lectura."
            self._log("SEGURIDAD", "Confirmación fuerte requerida. Escribe EJECUTAR en el diálogo si realmente quieres correrlo.")
            typed = simpledialog.askstring(
                "Confirmación fuerte · Crotolamo v4",
                "Estos comandos requieren confirmación fuerte:\n\n"
                + preview
                + "\n\nMotivos:\n"
                + detail_text
                + "\n\nEscribe EJECUTAR para continuar:",
                parent=self,
            )
            if typed != "EJECUTAR":
                self._log("SEGURIDAD", "Ejecución cancelada: confirmación fuerte no recibida.")
                return
            allow_confirm = True
        else:
            if not messagebox.askyesno("Ejecutar comandos seguros", "¿Ejecutar estos comandos de bajo riesgo?\n\n" + preview):
                self._log("SISTEMA", "Ejecución cancelada.")
                return

        self.running = True
        self._set_status("EJECUTANDO", THEME.green)
        threading.Thread(target=self._worker_execute, args=(commands, allow_confirm), daemon=True).start()

    def _worker_execute(self, commands: list[str], allow_confirm: bool = False) -> None:
        if self.runtime is not None:
            for event in self.runtime.execute_commands(commands, allow_confirm=allow_confirm):
                self.result_queue.put(("log", (event.get("label", "LOG"), event.get("text", ""))))
            self.result_queue.put(("done", None))
            return

        for cmd in commands:
            self.result_queue.put(("log", ("$", cmd)))
            result = subprocess.run(
                cmd,
                shell=True,
                text=True,
                capture_output=True,
                executable="/bin/bash",
            )
            if result.stdout.strip():
                self.result_queue.put(("log", ("OUT", result.stdout.rstrip())))
            if result.stderr.strip():
                self.result_queue.put(("log", ("ERR", result.stderr.rstrip())))
            if result.returncode != 0:
                self.result_queue.put(("log", ("ERROR", f"Comando terminó con código {result.returncode}")))
                break
        self.result_queue.put(("done", None))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.result_queue.get_nowait()
                if kind == "direct":
                    self._log("CROTOLAMO", str(payload))
                    self.current_plan = None
                    self.running = False
                    self._set_status("ONLINE", THEME.green)
                    self._update_mode_panel()
                    self._speak_async(str(payload))
                elif kind == "plan":
                    self.current_plan = payload
                    self.running = False
                    self._set_status("PLAN LISTO", THEME.cyan)
                    explanation = payload.get("explanation", "Sin explicación.")
                    safe = payload.get("safe", False)
                    risk = payload.get("risk", "safe")
                    commands = payload.get("commands", [])
                    self._log("PLAN", explanation)
                    self._log("SEGURIDAD", f"{'SEGURO' if safe else 'BLOQUEADO'} | riesgo={risk}")
                    self._update_mode_panel()
                    self._speak_async(explanation)
                    if commands:
                        self._log("COMANDOS", "\n".join(f"  {cmd}" for cmd in commands))
                    else:
                        self._log("COMANDOS", "Sin comandos. Solo respuesta/conversación.")
                elif kind == "heard":
                    self.running = False
                    self.listening = False
                    self._set_status("ONLINE", THEME.green)
                    text = str(payload).strip()
                    if text:
                        self.prompt_var.set(text)
                        self._log("MIC", f"Escuché: {text}")
                        self._speak_async(f"Escuché: {text}")
                    else:
                        self._log("MIC", "No escuché nada claro.")
                elif kind == "diagnostics":
                    self.running = False
                    self._set_status("ONLINE", THEME.green)
                    self._log("DIAGNÓSTICO", payload)
                    self._update_connection_panel(extra=str(payload))
                elif kind == "error":
                    self.running = False
                    self.listening = False
                    self._set_status("ERROR", THEME.red)
                    self._log("ERROR", str(payload))
                elif kind == "log":
                    label, text = payload
                    self._log(label, text)
                elif kind == "done":
                    self.running = False
                    self._set_status("ONLINE", THEME.green)
                    self._log("SISTEMA", "Hecho, patrón.")
                    self._speak_async("Hecho, patrón.")
        except queue.Empty:
            pass
        self.after(80, self._poll_queue)

    def _load_voice_enabled(self) -> bool:
        try:
            if SETTINGS_PATH.exists():
                data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                return bool(data.get("voice_enabled", False))
        except Exception:
            pass
        return False

    def _save_settings(self) -> None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(
                json.dumps({"voice_enabled": self.voice_enabled}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as error:
            self._log("CONFIG", f"No pude guardar configuración: {error}")

    def _new_session_log_path(self) -> Path:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return LOG_DIR / f"orbital_ui_{stamp}.log"

    def _write_log_file(self, line: str) -> None:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with self.session_log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            # Si falla el log, no matamos la UI. Sería demasiado humano.
            pass

    def _log(self, label: str, text: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {label}> {text}"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self._write_log_file(line)

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.configure(text=f"ESTADO: {text}", fg=color)

    def _speak_async(self, text: str) -> None:
        if not self.voice_enabled:
            return
        clean = str(text).strip()
        if not clean:
            return
        if self.runtime is not None:
            threading.Thread(target=lambda: self.runtime.say(clean[:420]), daemon=True).start()
            return
        if speak is None:
            return
        # Piper/ffplay bloquean. Lo mandamos a hilo para que la UI no se congele como laptop con 47 pestañas.
        threading.Thread(target=lambda: speak(clean[:420]), daemon=True).start()

    def toggle_voice(self) -> None:
        voice_available = (self.runtime is not None and callable(getattr(self.runtime, "speak", None))) or speak is not None
        if not voice_available:
            self.voice_enabled = False
            self._log("VOZ", f"Salida de voz no disponible: {VOICE_OUT_ERROR}")
            self.voice_button.set_text("VOZ: OFF", active=False, accent=THEME.yellow)
            return
        self.voice_enabled = not self.voice_enabled
        self._save_settings()
        self.voice_button.set_text("VOZ: ON" if self.voice_enabled else "VOZ: OFF", active=self.voice_enabled, accent=THEME.green if self.voice_enabled else THEME.yellow)
        self._log("VOZ", "Salida de voz activada." if self.voice_enabled else "Salida de voz apagada.")
        if self.voice_enabled:
            self._speak_async("Voz activada, patrón.")
        self._update_connection_panel()

    def listen_from_mic(self) -> None:
        mic_available = (self.runtime is not None and callable(getattr(self.runtime, "listen_once", None))) or listen_once is not None
        if not mic_available:
            self._log("MIC", f"Entrada de voz no disponible: {VOICE_IN_ERROR}")
            return
        if self.running:
            self._log("MIC", "Estoy ocupado. Un caos a la vez, patrón.")
            return
        self.running = True
        self.listening = True
        self._set_status("ESCUCHANDO", THEME.magenta)
        self._log("MIC", "Escuchando 8 segundos...")
        threading.Thread(target=self._worker_listen, daemon=True).start()

    def _worker_listen(self) -> None:
        try:
            if self.runtime is not None:
                text = self.runtime.listen(seconds=8)
            else:
                text = listen_once(seconds=8) if listen_once else ""
            self.result_queue.put(("heard", text))
        except Exception as error:
            self.result_queue.put(("error", error))

    def open_project_root(self) -> None:
        try:
            subprocess.Popen(["xdg-open", str(PROJECT_ROOT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._log("ARCHIVOS", f"Abrí la raíz del proyecto: {PROJECT_ROOT}")
        except Exception as error:
            self._log("ARCHIVOS", f"No pude abrir carpeta: {error}")

    def run_diagnostics(self) -> None:
        if self.running:
            self._log("DIAGNÓSTICO", "Ya estoy ocupado. La impaciencia humana no acelera Python.")
            return
        self.running = True
        self._set_status("DIAGNÓSTICO", THEME.yellow)
        threading.Thread(target=self._worker_diagnostics, daemon=True).start()

    def _worker_diagnostics(self) -> None:
        if self.runtime is not None:
            self.result_queue.put(("diagnostics", self.runtime.diagnostics_text()))
            return

        lines: list[str] = []
        lines.append(f"Python: {sys.version.split()[0]} | OS: {platform.system()} {platform.release()}")
        lines.append(f"Proyecto: {PROJECT_ROOT}")
        lines.append(f"Core: {'OK' if CORE_IMPORT_ERROR is None else 'ERROR: ' + str(CORE_IMPORT_ERROR)}")
        lines.append(f"Skills: {'OK' if callable(handle_direct_skill) else 'NO'}")
        lines.append(f"ask_ollama: {'OK' if callable(ask_ollama) else 'NO'}")
        lines.append(f"Voz entrada: {'OK' if callable(listen_once) else 'NO: ' + str(VOICE_IN_ERROR)}")
        lines.append(f"Voz salida: {'OK' if callable(speak) else 'NO: ' + str(VOICE_OUT_ERROR)}")
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=2) as response:
                ok = response.status == 200
            lines.append(f"Ollama API: {'OK' if ok else 'RESPUESTA RARA'}")
        except Exception as error:
            lines.append(f"Ollama API: NO ({error})")
        self.result_queue.put(("diagnostics", "\n".join(lines)))

    def _update_connection_panel(self, extra: str = "") -> None:
        if not hasattr(self, "connection_text"):
            return
        if self.runtime is not None:
            state = self.runtime.state()
            imports = state.get("imports", {})
            api = state.get("ollama_api", {})
            core = "OK"
            skills = "OK" if imports.get("skills") else "NO"
            ollama = "OK" if api.get("available") else "NO"
            voice_in = "OK" if imports.get("voice_in") else "NO"
            voice_out = "OK" if imports.get("voice_out") else "NO"
            ram = state.get("memory", {}).get("percent", "?")
            disk = state.get("disk_home", {}).get("percent", "?")
            voice_state = "ON" if self.voice_enabled else "OFF"
            mode = state.get("mode", {}).get("active", "?")
            text = (
                f"runtime=v5  modo={mode}  core={core}  skills={skills}  ollama={ollama}\n"
                f"mic={voice_in}  voz={voice_out}/{voice_state}\n"
                f"ram={ram}%  disco_home={disk}%\n"
                f"logs=data/runtime_logs + data/orbital_logs"
            )
        else:
            core = "OK" if CORE_IMPORT_ERROR is None else "ERROR"
            skills = "OK" if callable(handle_direct_skill) else "NO"
            ollama = "LISTO*" if callable(ask_ollama) else "NO"
            voice_in = "OK" if callable(listen_once) else "NO"
            voice_out = "OK" if callable(speak) else "NO"
            voice_state = "ON" if self.voice_enabled else "OFF"
            text = (
                f"runtime=NO  core={core}  skills={skills}  ollama={ollama}\n"
                f"mic={voice_in}  voz={voice_out}/{voice_state}\n"
                f"logs=data/orbital_logs\n"
                "*Modo viejo. Instala runtime v5 si quieres seguridad y datos reales, detalle menor."
            )
        if extra:
            text += "\n\nÚltimo diagnóstico:\n" + extra[-420:]
        self.connection_text.configure(state="normal")
        self.connection_text.delete("1.0", "end")
        self.connection_text.insert("1.0", text)
        self.connection_text.configure(state="disabled")


    def _refresh_real_panels(self) -> None:
        """Actualiza paneles con datos reales cada pocos segundos."""
        try:
            if self.runtime is None:
                self.after(2500, self._refresh_real_panels)
                return
            state = self.runtime.state()
            cpu = state.get("cpu", {})
            mem = state.get("memory", {})
            disk = state.get("disk_home", {})
            bat = state.get("battery", {})
            api = state.get("ollama_api", {})
            imports = state.get("imports", {})
            audio = state.get("audio", {})

            cpu_val = cpu.get("percent") if cpu.get("available") else None
            ram_val = mem.get("percent") if mem.get("available") else None
            disk_val = disk.get("percent")
            bat_txt = "NO"
            if bat.get("available"):
                bat_txt = f"{bat.get('capacity')}% {bat.get('status') or ''}".strip()
            ollama_txt = "OK" if api.get("available") else "NO"
            if api.get("running"):
                ollama_txt += " · " + ", ".join(api.get("running", [])[:2])
            mic_txt = "OK" if imports.get("voice_in") else "NO"
            audio_txt = "OK" if imports.get("voice_out") else "NO"
            if audio.get("available"):
                audio_txt += f" · in:{audio.get('inputs')} out:{audio.get('outputs')}"

            values = {
                "CPU": f"{cpu_val}%" if cpu_val is not None else "NO",
                "RAM": f"{ram_val}%" if ram_val is not None else "NO",
                "DISCO": f"{disk_val}% HOME" if disk_val is not None else "NO",
                "BATERÍA": bat_txt,
                "OLLAMA": ollama_txt,
                "MIC": mic_txt,
                "AUDIO": audio_txt,
            }
            for key, val in values.items():
                lab = self.real_metric_labels.get(key)
                if lab is not None:
                    color = THEME.text
                    if key in {"CPU", "RAM", "DISCO"}:
                        try:
                            num = float(str(val).split("%")[0])
                            color = THEME.red if num >= 90 else THEME.yellow if num >= 75 else THEME.green
                        except Exception:
                            color = THEME.text
                    elif val.startswith("NO"):
                        color = THEME.red
                    elif val.startswith("OK"):
                        color = THEME.green
                    lab.configure(text=str(val), fg=color)

            if self.status_canvas is not None:
                self._draw_status_dynamic(self.status_canvas, cpu_val or 0, ram_val or 0, disk_val or 0)
            self._update_connection_panel()
            self._update_mode_panel()
        except Exception as error:
            self._log("PANEL", f"No pude refrescar datos reales: {error}")
        finally:
            self.after(3000, self._refresh_real_panels)

    def _draw_status_dynamic(self, c: tk.Canvas, cpu: float, ram: float, disk: float) -> None:
        c.delete("all")
        cx, cy = 70, 62
        score = max(0, min(100, int(100 - ((cpu + ram + disk) / 3) * 0.55)))
        c.create_oval(cx - 48, cy - 48, cx + 48, cy + 48, outline="#16435a", width=8)
        c.create_arc(cx - 48, cy - 48, cx + 48, cy + 48, start=90, extent=-int(360 * score / 100), style="arc", outline=THEME.cyan, width=8)
        c.create_text(cx, cy - 6, text=str(score), fill=THEME.cyan, font=("JetBrains Mono", 24, "bold"))
        c.create_text(cx, cy + 22, text="SALUD", fill=THEME.muted, font=("JetBrains Mono", 8, "bold"))
        rows = [("CPU", cpu), ("RAM", ram), ("DISCO", disk)]
        for i, (name, val) in enumerate(rows):
            y = 24 + i * 30
            c.create_text(140, y, text=name, fill=THEME.muted, font=("JetBrains Mono", 9), anchor="w")
            c.create_rectangle(190, y - 6, 278, y + 6, outline="#16435a")
            fill = THEME.red if val >= 90 else THEME.yellow if val >= 75 else THEME.cyan
            c.create_rectangle(192, y - 4, 192 + int(82 * max(0, min(100, val)) / 100), y + 4, fill=fill, outline="")
            c.create_text(282, y, text=f"{val:.0f}%", fill=fill, font=("JetBrains Mono", 8, "bold"), anchor="e")


def main() -> None:
    app = CrotolamoOrbitalUI()
    app.mainloop()


if __name__ == "__main__":
    main()
