"""
CROTOLAMO ORBITAL UI PRO v16
Interfaz visual limpia, organizada y relativamente implementable.

Hecha en Tkinter puro para no meter dependencias raras.
Si quieres algo todavía más brutal después: PySide6 o web app local.
"""
from __future__ import annotations

import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import Tk, Canvas, Frame, Label, Button, Entry, Text, StringVar, BOTH, LEFT, RIGHT, X, Y, END, DISABLED, NORMAL
from tkinter import ttk


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Paleta
BG = "#050814"
PANEL = "#09111f"
PANEL_2 = "#0b1427"
BORDER = "#19385c"
CYAN = "#00d9ff"
BLUE = "#3a7bff"
PURPLE = "#b23cff"
MAGENTA = "#ff33cc"
GREEN = "#00ff88"
RED = "#ff405c"
YELLOW = "#ffd166"
TEXT = "#d8edff"
MUTED = "#7894b0"


def safe_import_runtime():
    try:
        from core.crotolamo_runtime import CrotolamoRuntime
        return CrotolamoRuntime()
    except Exception as e:
        return e


def render_result(result) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "response", "message", "output", "result"):
            if key in result:
                return str(result[key])
        return str(result)
    return str(result)


class GlowFrame(Frame):
    def __init__(self, master, title="", number=None, accent=CYAN, **kwargs):
        super().__init__(master, bg=PANEL, highlightbackground=accent, highlightthickness=1, **kwargs)
        self.accent = accent
        self.title = title
        if title:
            header = Frame(self, bg=PANEL)
            header.pack(fill=X, padx=8, pady=(6, 2))
            label_text = f"{number}. {title}" if number else title
            Label(
                header,
                text=label_text.upper(),
                fg=accent,
                bg=PANEL,
                font=("JetBrains Mono", 9, "bold"),
                anchor="w",
            ).pack(side=LEFT)


class OrbitalVisualizer(Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, bg="#020711", highlightthickness=0, **kwargs)
        self.t = 0
        self.after(40, self.animate)

    def animate(self):
        self.t += 0.04
        self.draw()
        self.after(40, self.animate)

    def draw(self):
        self.delete("all")
        w = max(self.winfo_width(), 10)
        h = max(self.winfo_height(), 10)
        cx, cy = w * 0.52, h * 0.50

        # Fondo estrellas
        random.seed(67)
        for _ in range(90):
            x = random.randint(0, w)
            y = random.randint(0, h)
            c = random.choice(["#20406b", "#3768ff", "#6b2cff", "#00d9ff"])
            self.create_oval(x, y, x+1, y+1, fill=c, outline="")

        # Túnel perspectiva
        for i in range(9):
            scale = i / 8
            x1 = cx - (w * 0.06 + scale * w * 0.42)
            x2 = cx + (w * 0.06 + scale * w * 0.42)
            y1 = cy - (h * 0.05 + scale * h * 0.40)
            y2 = cy + (h * 0.05 + scale * h * 0.40)
            color = "#122a5f" if i % 2 else "#193b83"
            self.create_rectangle(x1, y1, x2, y2, outline=color)
        for a in range(0, 360, 15):
            rad = math.radians(a)
            self.create_line(cx, cy, cx + math.cos(rad)*w*.48, cy + math.sin(rad)*h*.45, fill="#102a58")

        # Órbitas
        for i, r in enumerate([55, 85, 120, 155]):
            self.create_oval(cx-r*1.7, cy-r*.58, cx+r*1.7, cy+r*.58, outline=["#0e83ff", "#7b2cff", "#00d9ff", "#ff33cc"][i], width=1)

        # Galaxias
        for gx, gy, s in [(w*.25, h*.32, 1.0), (w*.78, h*.35, 1.2), (w*.72, h*.72, .7)]:
            for k in range(28):
                ang = k*.55 + self.t
                rr = k * s * 1.5
                x = gx + math.cos(ang) * rr
                y = gy + math.sin(ang) * rr * .42
                self.create_oval(x, y, x+2, y+2, fill=random.choice([PURPLE, CYAN, BLUE, MAGENTA]), outline="")
            self.create_oval(gx-4, gy-4, gx+4, gy+4, fill="#ffd166", outline="")

        # Cubo wireframe central
        size = 72 + math.sin(self.t)*4
        off = size * 0.38
        pts_front = [
            (cx-size/2, cy-size/2),
            (cx+size/2, cy-size/2),
            (cx+size/2, cy+size/2),
            (cx-size/2, cy+size/2),
        ]
        pts_back = [(x+off, y-off) for x, y in pts_front]
        for pts in (pts_front, pts_back):
            self.create_polygon(*sum(pts, ()), outline=CYAN, fill="", width=2)
        for p1, p2 in zip(pts_front, pts_back):
            self.create_line(*p1, *p2, fill=BLUE, width=2)
        self.create_text(cx+off/2, cy-off/2, text="◇", fill="#ffffff", font=("JetBrains Mono", 26, "bold"))

        # Plataforma
        self.create_oval(cx-110, cy+78, cx+110, cy+120, outline=PURPLE, width=2)
        self.create_oval(cx-70, cy+86, cx+70, cy+112, outline=CYAN, width=1)

        # Chips inferiores
        chips = [("ORBITAL CORE", "96.7%"), ("LATENCY", "12ms"), ("ENTROPY", "67%"), ("REALITY ROOT", "STABLE")]
        x0 = w * .08
        y0 = h - 44
        for i, (k, v) in enumerate(chips):
            x = x0 + i * 145
            self.create_rectangle(x, y0, x+125, y0+34, outline=BORDER, fill="#071020")
            self.create_text(x+8, y0+9, text=k, fill=MUTED, anchor="w", font=("JetBrains Mono", 7))
            self.create_text(x+8, y0+24, text=v, fill=GREEN if i != 2 else CYAN, anchor="w", font=("JetBrains Mono", 10, "bold"))


class MiniGauge(Canvas):
    def __init__(self, master, value=97, label="PASSED", **kwargs):
        super().__init__(master, bg=PANEL_2, highlightthickness=0, **kwargs)
        self.value = value
        self.label = label
        self.bind("<Configure>", lambda e: self.draw())

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        r = min(w, h) * .34
        cx, cy = w/2, h/2
        self.create_oval(cx-r, cy-r, cx+r, cy+r, outline="#163151", width=10)
        extent = 360 * self.value / 100
        self.create_arc(cx-r, cy-r, cx+r, cy+r, start=90, extent=-extent, outline=GREEN, width=10, style="arc")
        self.create_text(cx, cy-6, text=f"{self.value}%", fill=GREEN, font=("JetBrains Mono", 18, "bold"))
        self.create_text(cx, cy+18, text=self.label, fill=TEXT, font=("JetBrains Mono", 8))


class MascotCard(GlowFrame):
    def __init__(self, master, kind="pezlin", **kwargs):
        title = "PEZLÍN 3000" if kind == "pezlin" else "PALOMA SUPREMA 67"
        super().__init__(master, title=title, accent=PURPLE, **kwargs)
        self.kind = kind
        self.canvas = Canvas(self, bg=PANEL_2, highlightthickness=0, height=145)
        self.canvas.pack(fill=BOTH, expand=True, padx=8, pady=8)
        self.canvas.bind("<Configure>", lambda e: self.draw())

    def draw(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if self.kind == "pezlin":
            cx, cy = w/2, h*.48
            c.create_oval(cx-35, cy-38, cx+35, cy+38, fill="#4b2a85", outline=PURPLE, width=2)
            c.create_oval(cx-28, cy-22, cx-2, cy+4, fill="#d8edff", outline=TEXT, width=2)
            c.create_oval(cx+2, cy-22, cx+28, cy+4, fill="#d8edff", outline=TEXT, width=2)
            c.create_oval(cx-17, cy-12, cx-7, cy-2, fill="#111", outline="")
            c.create_oval(cx+11, cy-12, cx+21, cy-2, fill="#111", outline="")
            c.create_arc(cx-16, cy+8, cx+16, cy+24, start=0, extent=-180, outline=GREEN, width=2)
            c.create_text(cx, h-24, text="STATUS: ONLINE", fill=GREEN, font=("JetBrains Mono", 8, "bold"))
            c.create_text(cx, h-10, text="MODE: READY", fill=CYAN, font=("JetBrains Mono", 8))
        else:
            cx, cy = w/2, h*.45
            c.create_oval(cx-48, cy-34, cx+48, cy+34, fill="#5b5575", outline="#c4c8ff", width=2)
            c.create_oval(cx-22, cy-8, cx-4, cy+8, fill=RED, outline="")
            c.create_oval(cx+4, cy-8, cx+22, cy+8, fill=RED, outline="")
            c.create_line(cx-12, cy, w-15, cy-38, fill=RED, width=3)
            c.create_line(cx+12, cy, w-15, cy-18, fill=RED, width=3)
            c.create_oval(cx-38, cy+18, cx+38, cy+58, outline="#ffd166", width=4)
            c.create_text(cx, cy+52, text="67", fill="#ffd166", font=("JetBrains Mono", 18, "bold"))
            c.create_text(cx, h-24, text="STATUS: MAXIMUM", fill=RED, font=("JetBrains Mono", 8, "bold"))
            c.create_text(cx, h-10, text="MODE: DOMINANDO", fill=MAGENTA, font=("JetBrains Mono", 8))


class OrbitalUIPro:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("CROTOLAMO ORBITAL UI PRO")
        self.root.configure(bg=BG)
        self.root.geometry("1500x850")
        self.root.minsize(1200, 720)

        self.runtime = safe_import_runtime()
        self.output_queue = queue.Queue()

        self.command_var = StringVar()
        self.status_var = StringVar(value="STATUS: ONLINE")
        self.active_section = StringVar(value="Command Core")

        self.build()
        self.poll_output()

    def build(self):
        self.build_topbar()

        body = Frame(self.root, bg=BG)
        body.pack(fill=BOTH, expand=True, padx=8, pady=(0, 6))

        self.sidebar = Frame(body, bg=PANEL, width=190, highlightbackground=BORDER, highlightthickness=1)
        self.sidebar.pack(side=LEFT, fill=Y, padx=(0, 8))
        self.sidebar.pack_propagate(False)
        self.build_sidebar()

        main = Frame(body, bg=BG)
        main.pack(side=LEFT, fill=BOTH, expand=True)

        self.build_main_grid(main)
        self.build_footer()

    def build_topbar(self):
        top = Frame(self.root, bg=PANEL, height=58, highlightbackground=PURPLE, highlightthickness=1)
        top.pack(fill=X, padx=8, pady=8)
        top.pack_propagate(False)

        Label(top, text="◉ CROTOLAMO", fg=TEXT, bg=PANEL, font=("JetBrains Mono", 22, "bold")).pack(side=LEFT, padx=(16, 6))
        Label(top, text="ORBITAL UI PRO", fg=MAGENTA, bg=PANEL, font=("JetBrains Mono", 11, "bold")).pack(side=LEFT)

        chips = Frame(top, bg=PANEL)
        chips.pack(side=LEFT, padx=30)
        for title, value, color in [
            ("SYSTEM STATUS", "ALL SYSTEMS OPERATIONAL", GREEN),
            ("CPU", "67%", CYAN),
            ("RAM", "72%", CYAN),
            ("GPU", "81%", CYAN),
            ("NET", "1.42 Gbps", CYAN),
            ("ORBITAL TIME", "06-07-26 UTC", GREEN),
        ]:
            f = Frame(chips, bg="#071020", highlightbackground=BORDER, highlightthickness=1)
            f.pack(side=LEFT, padx=5, pady=8)
            Label(f, text=title, fg=MUTED, bg="#071020", font=("JetBrains Mono", 7)).pack(anchor="w", padx=8, pady=(3,0))
            Label(f, text=value, fg=color, bg="#071020", font=("JetBrains Mono", 9, "bold")).pack(anchor="w", padx=8, pady=(0,3))

        Label(top, text="⚙  🔔  CROTOLAMO", fg=TEXT, bg=PANEL, font=("JetBrains Mono", 10)).pack(side=RIGHT, padx=18)

    def nav_button(self, parent, text, color=CYAN):
        b = Button(
            parent, text=text, anchor="w",
            bg="#071020", fg=color, activebackground="#0e2444", activeforeground=TEXT,
            relief="flat", bd=0, font=("JetBrains Mono", 9),
            command=lambda: self.quick_command_from_nav(text),
        )
        b.pack(fill=X, padx=10, pady=3)
        return b

    def sidebar_label(self, parent, text, color=MUTED):
        Label(parent, text=text, fg=color, bg=PANEL, font=("JetBrains Mono", 8, "bold"), anchor="w").pack(fill=X, padx=12, pady=(12, 4))

    def build_sidebar(self):
        self.sidebar_label(self.sidebar, "CORE", CYAN)
        self.nav_button(self.sidebar, "Command Core", CYAN)

        self.sidebar_label(self.sidebar, "PROJECTS", PURPLE)
        self.nav_button(self.sidebar, "Project Map", PURPLE)
        self.nav_button(self.sidebar, "Inspector", PURPLE)

        self.sidebar_label(self.sidebar, "TOOLS", CYAN)
        for item in ["Task Planner", "Patch Builder", "Test Runner", "Safe Executor"]:
            self.nav_button(self.sidebar, item, CYAN)

        self.sidebar_label(self.sidebar, "SYSTEM", GREEN)
        for item in ["Memory & Context", "Brain Engine", "Voice", "Meme Reactor"]:
            self.nav_button(self.sidebar, item, GREEN)

        self.sidebar_label(self.sidebar, "ASSETS", PURPLE)
        self.nav_button(self.sidebar, "Mascots", PURPLE)

        node = Frame(self.sidebar, bg="#071020", highlightbackground=BORDER, highlightthickness=1)
        node.pack(fill=X, padx=10, pady=16)
        Label(node, text="ORBITAL NODE", fg=MUTED, bg="#071020", font=("JetBrains Mono", 7)).pack(anchor="w", padx=8, pady=(8,0))
        Label(node, text="CROTOLAMO-67  ONLINE", fg=GREEN, bg="#071020", font=("JetBrains Mono", 9, "bold")).pack(anchor="w", padx=8, pady=4)
        Label(node, text="LATENCY 12ms  |  UPTIME 86:70:26", fg=CYAN, bg="#071020", font=("JetBrains Mono", 7)).pack(anchor="w", padx=8, pady=(0,8))

    def build_main_grid(self, main):
        # Configurar columnas
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=4)
        main.grid_rowconfigure(1, weight=2)
        main.grid_rowconfigure(2, weight=2)

        # Command Core
        command = GlowFrame(main, title="COMMAND CORE / RUNTIME CONSOLE", number="01", accent=PURPLE)
        command.grid(row=0, column=0, sticky="nsew", padx=(0,8), pady=(0,8))
        self.build_command_core(command)

        # Project workspace
        workspace = GlowFrame(main, title="PROJECT WORKSPACE", number="02", accent=PURPLE)
        workspace.grid(row=0, column=1, sticky="nsew", padx=(0,0), pady=(0,8))
        self.build_workspace(workspace)

        # Tools
        tools = GlowFrame(main, title="ORBITAL TOOLS & MODULES", number="03", accent=CYAN)
        tools.grid(row=1, column=0, sticky="nsew", padx=(0,8), pady=(0,8))
        self.build_tools(tools)

        mascots = GlowFrame(main, title="MASCOTS", number="04", accent=PURPLE)
        mascots.grid(row=1, column=1, sticky="nsew", pady=(0,8))
        self.build_mascots(mascots)

        lower = Frame(main, bg=BG)
        lower.grid(row=2, column=0, columnspan=2, sticky="nsew")
        lower.grid_columnconfigure(0, weight=1)
        lower.grid_columnconfigure(1, weight=1)
        self.build_input(lower)

    def build_command_core(self, parent):
        inner = Frame(parent, bg=PANEL)
        inner.pack(fill=BOTH, expand=True, padx=8, pady=8)
        inner.grid_columnconfigure(0, weight=1)
        inner.grid_columnconfigure(1, weight=2)
        inner.grid_rowconfigure(0, weight=1)

        self.terminal = Text(inner, bg="#030812", fg=GREEN, insertbackground=CYAN, relief="flat", font=("JetBrains Mono", 9), wrap="word")
        self.terminal.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        self.terminal.insert(END, "crotolamo@orbital:~$ orbital run --mode=chaos\n\n")
        for line in [
            "> initializing orbital grid... ok",
            "> loading cores online [cores: 67]... ok",
            "> sandbox: MODO_MUTÍSIMO",
            "> system: latino_labs handmade",
            "> STATUS: READY TO CROTALIZE",
            "",
            "\"Si no hay caos, no hay historia.\"",
            "",
        ]:
            self.terminal.insert(END, line + "\n")
        self.terminal.insert(END, "crotolamo@orbital:~$ ")
        self.terminal.configure(state=DISABLED)

        self.visual = OrbitalVisualizer(inner)
        self.visual.grid(row=0, column=1, sticky="nsew")

    def build_workspace(self, parent):
        tabs = Frame(parent, bg=PANEL)
        tabs.pack(fill=X, padx=8, pady=6)
        for t in ["PROJECT MAP", "INSPECTOR", "TASK PLANNER", "PATCH PREVIEW"]:
            Button(tabs, text=t, bg="#071020", fg=CYAN if t=="PROJECT MAP" else MUTED,
                   activebackground="#0e2444", relief="flat", font=("JetBrains Mono", 8),
                   command=lambda x=t: self.workspace_action(x)).pack(side=LEFT, padx=4)

        split = Frame(parent, bg=PANEL)
        split.pack(fill=BOTH, expand=True, padx=8, pady=(0,8))
        split.grid_columnconfigure(0, weight=1)
        split.grid_columnconfigure(1, weight=1)
        split.grid_rowconfigure(0, weight=1)

        self.project_tree = Text(split, bg="#030812", fg=TEXT, relief="flat", font=("JetBrains Mono", 8))
        self.project_tree.grid(row=0, column=0, sticky="nsew", padx=(0,6))
        tree = """/mnt/orbital/mafia
├── core_system
│   ├── main_core.py
│   └── orbital_link.py
├── modules
│   ├── brain_engine.py
│   ├── meme_reactor.py
│   └── voice_core.py
├── assets
│   ├── pezlin_3000.png
│   └── paloma_suprema.png
├── tests
│   ├── test_latency.py
│   ├── test_cores.py
│   └── test_voices.py
└── config.yaml"""
        self.project_tree.insert(END, tree)
        self.project_tree.configure(state=DISABLED)

        info = Frame(split, bg="#071020", highlightbackground=BORDER, highlightthickness=1)
        info.grid(row=0, column=1, sticky="nsew")
        for label, val in [
            ("INSPECTOR", "main_core.py"),
            ("Type", "Python File"),
            ("Size", "12.8 KB"),
            ("Status", "Tracked"),
            ("Lines", "312"),
        ]:
            Label(info, text=f"{label}: {val}", fg=CYAN if label=="INSPECTOR" else TEXT, bg="#071020",
                  font=("JetBrains Mono", 8), anchor="w").pack(fill=X, padx=10, pady=5)
        Button(info, text="OPEN IN EDITOR", bg="#15102c", fg=PURPLE, relief="flat",
               font=("JetBrains Mono", 8), command=lambda: self.run_runtime_command("mapa crotolamo")).pack(fill=X, padx=10, pady=12)

        actions = Frame(parent, bg=PANEL)
        actions.pack(fill=X, padx=8, pady=(0,8))
        for cmd in ["proyectos", "mapa crotolamo", "inspeccionar crotolamo", "plan limpiar estructura de crotolamo"]:
            Button(actions, text=cmd.upper(), bg="#071020", fg=GREEN, relief="flat", font=("JetBrains Mono", 7),
                   command=lambda c=cmd: self.run_runtime_command(c)).pack(side=LEFT, padx=4)

    def small_card(self, parent, title, accent=CYAN):
        card = Frame(parent, bg=PANEL_2, highlightbackground=accent, highlightthickness=1)
        Label(card, text=title.upper(), fg=accent, bg=PANEL_2, font=("JetBrains Mono", 8, "bold")).pack(anchor="w", padx=8, pady=(8,3))
        return card

    def build_tools(self, parent):
        row = Frame(parent, bg=PANEL)
        row.pack(fill=BOTH, expand=True, padx=8, pady=8)
        for i in range(6):
            row.grid_columnconfigure(i, weight=1)

        # Test
        c = self.small_card(row, "Test Runner", GREEN); c.grid(row=0, column=0, sticky="nsew", padx=3)
        MiniGauge(c, value=97, height=110).pack(fill=BOTH, expand=True, padx=4)
        Button(c, text="RUN ALL TESTS", bg="#071020", fg=CYAN, relief="flat", font=("JetBrains Mono", 7),
               command=lambda: self.run_runtime_command("test crotolamo")).pack(fill=X, padx=6, pady=6)

        # Executor
        c = self.small_card(row, "Safe Executor", GREEN); c.grid(row=0, column=1, sticky="nsew", padx=3)
        for item in ["Sandbox Mode", "Secure I/O", "Policy Lock", "Auto Rollback"]:
            Label(c, text="✓ " + item, fg=GREEN, bg=PANEL_2, font=("JetBrains Mono", 7)).pack(anchor="w", padx=10, pady=2)
        Button(c, text="EXECUTE", bg="#072415", fg=GREEN, relief="flat", font=("JetBrains Mono", 7),
               command=lambda: self.run_runtime_command("executor")).pack(fill=X, padx=6, pady=6)

        # Memory
        c = self.small_card(row, "Memory & Context", PURPLE); c.grid(row=0, column=2, sticky="nsew", padx=3)
        for item in ["Context Units 2.8M", "Memory Nodes 7.2K", "Tokens Active 1.3M", "Cache Hit 94%"]:
            Label(c, text=item, fg=TEXT, bg=PANEL_2, font=("JetBrains Mono", 7)).pack(anchor="w", padx=10, pady=2)
        Button(c, text="MEMORIA", bg="#15102c", fg=PURPLE, relief="flat", font=("JetBrains Mono", 7),
               command=lambda: self.run_runtime_command("memoria")).pack(fill=X, padx=6, pady=6)

        # Brain
        c = self.small_card(row, "Brain Engine", GREEN); c.grid(row=0, column=3, sticky="nsew", padx=3)
        for item in ["Profile: PRO", "Depth: 5/5", "Self-review: ON", "Creativity 91%"]:
            Label(c, text=item, fg=TEXT, bg=PANEL_2, font=("JetBrains Mono", 7)).pack(anchor="w", padx=10, pady=2)
        Button(c, text="INTELIGENCIA", bg="#072415", fg=GREEN, relief="flat", font=("JetBrains Mono", 7),
               command=lambda: self.run_runtime_command("inteligencia")).pack(fill=X, padx=6, pady=6)

        # Voice
        c = self.small_card(row, "Voice", PURPLE); c.grid(row=0, column=4, sticky="nsew", padx=3)
        Label(c, text="LATIN_MAFIA_3000", fg=TEXT, bg=PANEL_2, font=("JetBrains Mono", 7)).pack(anchor="w", padx=10, pady=3)
        Label(c, text="Latencia: 12ms", fg=CYAN, bg=PANEL_2, font=("JetBrains Mono", 7)).pack(anchor="w", padx=10)
        Label(c, text="🎙", fg=PURPLE, bg=PANEL_2, font=("Arial", 36)).pack()
        Button(c, text="PUSH TO TALK", bg="#15102c", fg=PURPLE, relief="flat", font=("JetBrains Mono", 7)).pack(fill=X, padx=6, pady=6)

        # Meme
        c = self.small_card(row, "Meme Reactor", MAGENTA); c.grid(row=0, column=5, sticky="nsew", padx=3)
        Label(c, text="CHAOS LEVEL: ULTRA", fg=GREEN, bg=PANEL_2, font=("JetBrains Mono", 7)).pack(anchor="w", padx=10, pady=3)
        Label(c, text="REACTION SPEED: MAX", fg=CYAN, bg=PANEL_2, font=("JetBrains Mono", 7)).pack(anchor="w", padx=10)
        Label(c, text="😎💻", fg=YELLOW, bg=PANEL_2, font=("Arial", 28)).pack(expand=True)
        Button(c, text="GENERATE MEME", bg="#1d0a26", fg=MAGENTA, relief="flat", font=("JetBrains Mono", 7)).pack(fill=X, padx=6, pady=6)

    def build_mascots(self, parent):
        row = Frame(parent, bg=PANEL)
        row.pack(fill=BOTH, expand=True, padx=8, pady=8)
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=1)
        MascotCard(row, "pezlin").grid(row=0, column=0, sticky="nsew", padx=(0,5))
        MascotCard(row, "paloma").grid(row=0, column=1, sticky="nsew", padx=(5,0))

    def build_input(self, parent):
        box = Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        box.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        Label(box, text="COMMAND INPUT", fg=CYAN, bg=PANEL, font=("JetBrains Mono", 8, "bold")).pack(anchor="w", padx=8, pady=(6,0))
        entry_row = Frame(box, bg=PANEL)
        entry_row.pack(fill=X, padx=8, pady=6)
        self.input_entry = Entry(entry_row, textvariable=self.command_var, bg="#030812", fg=TEXT, insertbackground=CYAN,
                                 relief="flat", font=("JetBrains Mono", 10))
        self.input_entry.pack(side=LEFT, fill=X, expand=True, ipady=8)
        self.input_entry.bind("<Return>", lambda e: self.submit_command())
        Button(entry_row, text="RUN", bg="#072415", fg=GREEN, relief="flat", command=self.submit_command).pack(side=RIGHT, padx=(8,0), ipadx=18, ipady=6)

        quick = Frame(box, bg=PANEL)
        quick.pack(fill=X, padx=8, pady=(0,8))
        for cmd in ["contexto", "proyectos", "test crotolamo", "executor", "parches"]:
            Button(quick, text=cmd, bg="#071020", fg=MUTED, relief="flat", font=("JetBrains Mono", 7),
                   command=lambda c=cmd: self.run_runtime_command(c)).pack(side=LEFT, padx=3)

        log = Frame(parent, bg=PANEL, highlightbackground=PURPLE, highlightthickness=1)
        log.grid(row=0, column=1, sticky="nsew")
        Label(log, text="STATUS FEED", fg=PURPLE, bg=PANEL, font=("JetBrains Mono", 8, "bold")).pack(anchor="w", padx=8, pady=(6,0))
        self.feed = Text(log, bg="#030812", fg=TEXT, height=5, relief="flat", font=("JetBrains Mono", 8))
        self.feed.pack(fill=BOTH, expand=True, padx=8, pady=6)
        self.feed.insert(END, "[INFO] Orbital UI Pro listo.\n[INFO] no panic. I got logic & el sistema\n")
        self.feed.configure(state=DISABLED)

    def build_footer(self):
        footer = Frame(self.root, bg=PANEL, height=46, highlightbackground=BORDER, highlightthickness=1)
        footer.pack(fill=X, padx=8, pady=(0,8))
        footer.pack_propagate(False)
        Label(footer, text="LATIN MAFIA  |  CAOS MODE  |  HECHO EN HUEVONITIS  |  67  |  LOCO PERO SERIO",
              fg=YELLOW, bg=PANEL, font=("JetBrains Mono", 10, "bold")).pack(side=LEFT, padx=14)
        Label(footer, textvariable=self.status_var, fg=GREEN, bg=PANEL, font=("JetBrains Mono", 10, "bold")).pack(side=RIGHT, padx=14)

    def quick_command_from_nav(self, text):
        mapping = {
            "Command Core": "runtime",
            "Project Map": "mapa crotolamo",
            "Inspector": "inspeccionar crotolamo",
            "Task Planner": "plan limpiar estructura de crotolamo",
            "Patch Builder": "patch limpiar estructura de crotolamo",
            "Test Runner": "test crotolamo",
            "Safe Executor": "executor",
            "Memory & Context": "memoria",
            "Brain Engine": "inteligencia",
            "Voice": "voz",
            "Meme Reactor": "contexto",
            "Mascots": "contexto",
        }
        self.run_runtime_command(mapping.get(text, text.lower()))

    def workspace_action(self, tab):
        mapping = {
            "PROJECT MAP": "mapa crotolamo",
            "INSPECTOR": "inspeccionar crotolamo",
            "TASK PLANNER": "plan limpiar estructura de crotolamo",
            "PATCH PREVIEW": "parches",
        }
        self.run_runtime_command(mapping.get(tab, "proyectos"))

    def submit_command(self):
        cmd = self.command_var.get().strip()
        if not cmd:
            return
        self.command_var.set("")
        self.run_runtime_command(cmd)

    def append_terminal(self, text):
        self.terminal.configure(state=NORMAL)
        self.terminal.insert(END, "\n" + text + "\n")
        self.terminal.see(END)
        self.terminal.configure(state=DISABLED)

    def append_feed(self, text):
        self.feed.configure(state=NORMAL)
        self.feed.insert(END, text + "\n")
        self.feed.see(END)
        self.feed.configure(state=DISABLED)

    def run_runtime_command(self, cmd):
        self.append_terminal(f"crotolamo@orbital:~$ {cmd}")
        self.status_var.set("STATUS: RUNNING")
        threading.Thread(target=self._worker, args=(cmd,), daemon=True).start()

    def _worker(self, cmd):
        if isinstance(self.runtime, Exception):
            self.output_queue.put(("error", f"No pude cargar runtime: {self.runtime}"))
            return

        try:
            result = self.runtime.process_text(cmd)
            self.output_queue.put(("ok", render_result(result)))
        except Exception as e:
            self.output_queue.put(("error", f"{type(e).__name__}: {e}"))

    def poll_output(self):
        try:
            while True:
                kind, text = self.output_queue.get_nowait()
                if kind == "ok":
                    self.append_terminal(text[:6000])
                    self.append_feed("[OK] comando ejecutado")
                    self.status_var.set("STATUS: ONLINE")
                else:
                    self.append_terminal("[ERROR] " + text)
                    self.append_feed("[ERROR] " + text)
                    self.status_var.set("STATUS: ERROR")
        except queue.Empty:
            pass
        self.root.after(120, self.poll_output)


def main():
    app = Tk()
    try:
        style = ttk.Style(app)
        style.theme_use("clam")
    except Exception:
        pass
    OrbitalUIPro(app)
    app.mainloop()


if __name__ == "__main__":
    main()
