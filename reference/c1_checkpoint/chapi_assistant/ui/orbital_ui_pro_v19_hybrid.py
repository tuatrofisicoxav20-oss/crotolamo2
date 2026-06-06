from __future__ import annotations

import os
import sys
import threading
import queue
from pathlib import Path
from tkinter import (
    Tk, Canvas, Frame, Label, Button, Entry, Text, StringVar,
    BOTH, LEFT, RIGHT, X, Y, END, NORMAL, DISABLED
)
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.orbital_theme import THEME as T
from ui.orbital_widgets_v17 import SystemPulse, StickerStrip, RadarBadge
from ui.orbital_assets import get_asset

try:
    import psutil
except Exception:
    psutil = None


def safe_runtime():
    try:
        from core.crotolamo_runtime import CrotolamoRuntime
        return CrotolamoRuntime()
    except Exception as e:
        return e


def render_result(result):
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for k in ("text", "response", "message", "output", "result"):
            if k in result:
                return str(result[k])
        return str(result)
    return str(result)


def build_tree_lines(root: Path, max_depth: int = 3, max_entries: int = 8):
    lines = [str(root)]
    ignore = {
        ".git", "__pycache__", ".venv", "venv", "node_modules",
        "backups", ".mypy_cache", ".pytest_cache", ".ruff_cache"
    }

    def walk(path: Path, prefix: str, depth: int):
        if depth > max_depth:
            return
        try:
            entries = sorted(
                [p for p in path.iterdir() if p.name not in ignore],
                key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except Exception:
            return

        shown = entries[:max_entries]
        for idx, item in enumerate(shown):
            connector = "└── " if idx == len(shown) - 1 else "├── "
            lines.append(prefix + connector + item.name)
            if item.is_dir():
                ext_prefix = "    " if idx == len(shown) - 1 else "│   "
                walk(item, prefix + ext_prefix, depth + 1)
        if len(entries) > max_entries:
            lines.append(prefix + f"└── ... +{len(entries)-max_entries} elementos")

    walk(root, "", 1)
    return "\n".join(lines)


def collect_project_stats(root: Path):
    stats = {
        "dirs": 0,
        "files": 0,
        "py_files": 0,
        "largest_name": "N/A",
        "largest_size": 0,
    }
    ignore = {
        ".git", "__pycache__", ".venv", "venv", "node_modules",
        "backups", ".mypy_cache", ".pytest_cache", ".ruff_cache"
    }

    try:
        for path in root.rglob("*"):
            if any(part in ignore for part in path.parts):
                continue
            if path.is_dir():
                stats["dirs"] += 1
            elif path.is_file():
                stats["files"] += 1
                if path.suffix == ".py":
                    stats["py_files"] += 1
                try:
                    size = path.stat().st_size
                    if size > stats["largest_size"]:
                        stats["largest_size"] = size
                        stats["largest_name"] = str(path.relative_to(root))
                except Exception:
                    pass
    except Exception:
        pass
    return stats


def format_bytes(n: int):
    units = ["B", "KB", "MB", "GB"]
    value = float(n)
    for u in units:
        if value < 1024 or u == units[-1]:
            if u == "B":
                return f"{int(value)} {u}"
            return f"{value:.1f} {u}"
        value /= 1024.0
    return f"{n} B"


class Panel(Frame):
    def __init__(self, master, title="", number="", accent=None, **kwargs):
        accent = accent or T["CYAN"]
        super().__init__(master, bg=T["PANEL"], highlightbackground=accent, highlightthickness=1, **kwargs)
        if title:
            head = Frame(self, bg=T["PANEL"])
            head.pack(fill=X, padx=10, pady=(7, 2))
            Label(
                head,
                text=(f"{number}. " if number else "") + title.upper(),
                bg=T["PANEL"], fg=accent, font=("JetBrains Mono", 9, "bold")
            ).pack(side=LEFT)
            Label(head, text="● ○", bg=T["PANEL"], fg=T["MUTED"], font=("JetBrains Mono", 8)).pack(side=RIGHT)


class OrbitCanvas(Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, bg="#020711", highlightthickness=0, **kwargs)
        self.t = 0
        self.after(40, self.tick)

    def tick(self):
        self.t += 0.04
        self.draw()
        self.after(40, self.tick)

    def draw(self):
        import math
        self.delete("all")
        w, h = max(self.winfo_width(), 10), max(self.winfo_height(), 10)
        bg = get_asset("galaxy_bg.png")
        if bg:
            self.create_image(w // 2, h // 2, image=bg)

        cx, cy = w * .55, h * .50
        for i in range(12):
            s = i / 11
            self.create_rectangle(
                cx - (w * .04 + s * w * .42),
                cy - (h * .04 + s * h * .42),
                cx + (w * .04 + s * w * .42),
                cy + (h * .04 + s * h * .42),
                outline="#12366d"
            )
        for i in range(-8, 9):
            x = cx + i * w * .045
            self.create_line(cx, cy, x, h, fill="#112c5b")
            self.create_line(cx, cy, x, 0, fill="#112c5b")

        size = 96 + math.sin(self.t * 2) * 5
        off = size * .34
        front = [(cx-size/2, cy-size/2), (cx+size/2, cy-size/2), (cx+size/2, cy+size/2), (cx-size/2, cy+size/2)]
        back = [(x+off, y-off) for x, y in front]
        for poly, color in [(front, T["CYAN"]), (back, T["BLUE"])]:
            self.create_polygon(*sum(poly, ()), outline=color, fill="", width=2)
        for a, b in zip(front, back):
            self.create_line(*a, *b, fill=T["PURPLE"], width=2)
        for i, r in enumerate([78, 118, 160, 205]):
            self.create_oval(
                cx-r*1.45, cy-r*.45, cx+r*1.45, cy+r*.45,
                outline=[T["CYAN"], T["PURPLE"], T["BLUE"], T["MAGENTA"]][i], width=1
            )


class AssetImage(Canvas):
    def __init__(self, master, asset_name, **kwargs):
        self.asset_name = asset_name
        super().__init__(master, bg=T["CARD"], highlightthickness=0, **kwargs)
        self.bind("<Configure>", lambda e: self.draw())

    def draw(self):
        self.delete("all")
        w, h = max(self.winfo_width(), 10), max(self.winfo_height(), 10)
        img = get_asset(self.asset_name)
        if img:
            self.create_image(w//2, h//2, image=img)
        else:
            self.create_text(w//2, h//2, text=self.asset_name, fill=T["MUTED"], font=("JetBrains Mono", 9))


class OrbitalUIProV19:
    def __init__(self, root):
        self.root = root
        self.root.title("CROTOLAMO ORBITAL UI PRO v19 Hybrid")
        self.root.geometry("1600x930")
        self.root.minsize(1300, 780)
        self.root.configure(bg=T["BG"])

        self.runtime = safe_runtime()
        self.queue = queue.Queue()
        self.command_var = StringVar()
        self.status_var = StringVar(value="STATUS: ONLINE")
        self.workspace_mode = StringVar(value="PROJECT MAP")
        self.last_command = "startup"
        self.project_root = ROOT
        self.tool_summary_labels = {}

        self.build()
        self.refresh_local_stats()
        self.update_metrics()
        self.poll()

    def build(self):
        self.build_topbar()
        self.build_content()
        self.build_footer()

    def build_topbar(self):
        top = Frame(self.root, bg=T["PANEL"], height=62, highlightbackground=T["PURPLE"], highlightthickness=1)
        top.pack(fill=X, padx=8, pady=8)
        top.pack_propagate(False)

        logo = Frame(top, bg=T["PANEL"])
        logo.pack(side=LEFT, padx=14)
        Label(logo, text="CROTOLAMO", fg=T["TEXT"], bg=T["PANEL"], font=("JetBrains Mono", 21, "bold")).pack(anchor="w")
        Label(logo, text="ORBITAL UI PRO v19 / HYBRID", fg=T["MAGENTA"], bg=T["PANEL"], font=("JetBrains Mono", 9, "bold")).pack(anchor="w")

        status = Frame(top, bg=T["CARD"], highlightbackground=T["BORDER"], highlightthickness=1)
        status.pack(side=LEFT, padx=20, pady=8)
        Label(status, text="SYSTEM STATUS", fg=T["MUTED"], bg=T["CARD"], font=("JetBrains Mono", 7, "bold")).pack(anchor="w", padx=10, pady=(4, 0))
        Label(status, text="● ALL SYSTEMS OPERATIONAL", fg=T["GREEN"], bg=T["CARD"], font=("JetBrains Mono", 9, "bold")).pack(anchor="w", padx=10, pady=(0, 4))

        metrics = Frame(top, bg=T["PANEL"])
        metrics.pack(side=LEFT, fill=Y, expand=True)
        self.cpu = SystemPulse(metrics, "CPU", 67); self.cpu.pack(side=LEFT, padx=4, pady=8)
        self.ram = SystemPulse(metrics, "RAM", 72); self.ram.pack(side=LEFT, padx=4, pady=8)
        self.gpu = SystemPulse(metrics, "GPU", 81); self.gpu.pack(side=LEFT, padx=4, pady=8)

        right = Frame(top, bg=T["PANEL"])
        right.pack(side=RIGHT, padx=10)
        RadarBadge(right).pack(side=LEFT, padx=8)
        Label(right, text="CROTOLAMO-67", fg=T["TEXT"], bg=T["PANEL"], font=("JetBrains Mono", 10, "bold")).pack(anchor="e")
        Label(right, text="Orbital Operator", fg=T["MUTED"], bg=T["PANEL"], font=("JetBrains Mono", 8)).pack(anchor="e")

    def build_content(self):
        body = Frame(self.root, bg=T["BG"])
        body.pack(fill=BOTH, expand=True, padx=8, pady=(0, 6))

        side = Panel(body, accent=T["BORDER"], width=205)
        side.pack(side=LEFT, fill=Y, padx=(0, 8))
        side.pack_propagate(False)
        self.build_sidebar(side)

        main = Frame(body, bg=T["BG"])
        main.pack(side=LEFT, fill=BOTH, expand=True)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=5)
        main.grid_rowconfigure(1, weight=2)
        main.grid_rowconfigure(2, weight=2)

        p1 = Panel(main, "COMMAND CORE / RUNTIME CONSOLE", "01", T["PURPLE"])
        p1.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        self.build_command_core(p1)

        p2 = Panel(main, "PROJECT WORKSPACE / LIVE DATA", "02", T["PURPLE"])
        p2.grid(row=0, column=1, sticky="nsew", pady=(0, 8))
        self.build_workspace(p2)

        p3 = Panel(main, "ORBITAL TOOLS & MODULES", "03", T["CYAN"])
        p3.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        self.build_tools(p3)

        p4 = Panel(main, "MASCOTS / PERSONALITY / HACKER MODE", "04", T["PURPLE"])
        p4.grid(row=1, column=1, sticky="nsew", pady=(0, 8))
        self.build_mascots(p4)

        p5 = Panel(main, "COMMAND INPUT", "05", T["GREEN"])
        p5.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        self.build_command_input(p5)

        p6 = Panel(main, "ACTIVITY FEED / TELEMETRY", "06", T["CYAN"])
        p6.grid(row=2, column=1, sticky="nsew")
        self.build_activity_feed(p6)

    def build_sidebar(self, parent):
        def heading(t, c):
            Label(parent, text=t, fg=c, bg=T["PANEL"], font=("JetBrains Mono", 8, "bold"), anchor="w").pack(fill=X, padx=12, pady=(12, 4))

        def item(t, cmd, c):
            Button(
                parent, text=t, anchor="w",
                bg=T["CARD"], fg=c, activebackground=T["PANEL_2"], activeforeground=T["TEXT"],
                relief="flat", font=("JetBrains Mono", 9),
                command=lambda: self.run_runtime_command(cmd)
            ).pack(fill=X, padx=10, pady=3)

        heading("CORE", T["CYAN"]); item("Command Core", "runtime", T["CYAN"])
        heading("PROJECTS", T["PURPLE"])
        item("Project Map", "__LOCAL_PROJECT_MAP__", T["PURPLE"])
        item("Inspector", "__LOCAL_INSPECTOR__", T["PURPLE"])
        heading("TOOLS", T["CYAN"])
        item("Task Planner", "plan limpiar estructura de crotolamo", T["CYAN"])
        item("Patch Builder", "patch limpiar estructura de crotolamo", T["CYAN"])
        item("Test Runner", "test crotolamo", T["CYAN"])
        item("Safe Executor", "executor", T["CYAN"])
        heading("SYSTEM", T["GREEN"])
        item("Memory & Context", "memoria", T["GREEN"])
        item("Brain Engine", "inteligencia", T["GREEN"])
        item("Voice", "voz", T["GREEN"])
        item("Meme Reactor", "contexto", T["GREEN"])
        heading("ASSETS", T["PURPLE"])
        item("Hacker Mode", "__LOCAL_HACKER__", T["PURPLE"])

        box = Frame(parent, bg=T["CARD"], highlightbackground=T["BORDER"], highlightthickness=1)
        box.pack(fill=X, padx=10, pady=16)
        Label(box, text="ORBITAL NODE", fg=T["MUTED"], bg=T["CARD"], font=("JetBrains Mono", 7)).pack(anchor="w", padx=8, pady=(8, 0))
        self.node_label = Label(box, text="CROTOLAMO-67", fg=T["CYAN"], bg=T["CARD"], font=("JetBrains Mono", 10, "bold"))
        self.node_label.pack(anchor="w", padx=8)
        self.node_status = Label(box, text="ONLINE ●", fg=T["GREEN"], bg=T["CARD"], font=("JetBrains Mono", 8, "bold"))
        self.node_status.pack(anchor="w", padx=8, pady=(0, 8))

    def build_command_core(self, parent):
        inner = Frame(parent, bg=T["PANEL"])
        inner.pack(fill=BOTH, expand=True, padx=8, pady=8)
        inner.grid_columnconfigure(0, weight=1)
        inner.grid_columnconfigure(1, weight=2)
        inner.grid_rowconfigure(0, weight=1)

        self.terminal = Text(inner, bg="#020710", fg=T["GREEN"], insertbackground=T["CYAN"], relief="flat", font=("JetBrains Mono", 9), wrap="word")
        self.terminal.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        boot = [
            "crotolamo@orbital:~$ orbital run --mode=chaos",
            "",
            "> loading orbital assets... ok",
            "> pezlin_3000.png linked",
            "> paloma_suprema_67.png armed",
            "> hacker_mode.png active",
            "> STATUS: READY TO CROTALIZE",
            "",
            "\"Si no hay caos, no hay historia.\"",
            "",
        ]
        self.terminal.insert(END, "\n".join(boot))
        self.terminal.configure(state=DISABLED)

        OrbitCanvas(inner).grid(row=0, column=1, sticky="nsew")

        chips = Frame(parent, bg=T["PANEL"])
        chips.pack(fill=X, padx=8, pady=(0, 8))
        self.core_chip_labels = {}
        for k, v, c in [
            ("ORBITAL CORE", "96.7%", T["GREEN"]),
            ("SYNC LATENCY", "12ms", T["CYAN"]),
            ("ENTROPY", "67%", T["PURPLE"]),
            ("REALITY ROOT", "STABLE", T["GREEN"]),
        ]:
            box = Frame(chips, bg=T["CARD"], highlightbackground=T["BORDER"], highlightthickness=1)
            box.pack(side=LEFT, fill=X, expand=True, padx=4)
            Label(box, text=k, fg=T["MUTED"], bg=T["CARD"], font=("JetBrains Mono", 7)).pack()
            label = Label(box, text=v, fg=c, bg=T["CARD"], font=("JetBrains Mono", 11, "bold"))
            label.pack(pady=(0, 4))
            self.core_chip_labels[k] = label

    def build_workspace(self, parent):
        tabs = Frame(parent, bg=T["PANEL"])
        tabs.pack(fill=X, padx=8, pady=6)

        for label, cmd in [
            ("PROJECT MAP", "__LOCAL_PROJECT_MAP__"),
            ("INSPECTOR", "__LOCAL_INSPECTOR__"),
            ("TASK PLANNER", "plan limpiar estructura de crotolamo"),
            ("PATCH PREVIEW", "parches"),
        ]:
            Button(
                tabs, text=label, bg=T["CARD"], fg=T["CYAN"] if label == "PROJECT MAP" else T["MUTED"],
                relief="flat", font=("JetBrains Mono", 8),
                command=lambda c=cmd, l=label: self.switch_workspace(l, c)
            ).pack(side=LEFT, padx=3)

        content = Frame(parent, bg=T["PANEL"])
        content.pack(fill=BOTH, expand=True, padx=8, pady=(0, 8))
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=2)
        content.grid_rowconfigure(0, weight=1)

        self.workspace_text = Text(content, bg="#020710", fg=T["TEXT"], relief="flat", font=("JetBrains Mono", 8), wrap="word")
        self.workspace_text.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        right = Frame(content, bg=T["CARD"], highlightbackground=T["BORDER"], highlightthickness=1)
        right.grid(row=0, column=1, sticky="nsew")

        Label(right, text="INSPECTOR SNAPSHOT", fg=T["CYAN"], bg=T["CARD"], font=("JetBrains Mono", 8, "bold")).pack(anchor="w", padx=10, pady=(10, 4))
        self.inspector_text = Text(right, bg=T["CARD"], fg=T["TEXT"], relief="flat", font=("JetBrains Mono", 8), wrap="word", height=12)
        self.inspector_text.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))

        footer = Frame(parent, bg=T["PANEL"])
        footer.pack(fill=X, padx=8, pady=(0, 8))
        self.workspace_status = Label(footer, text="Workspace ready", fg=T["GREEN"], bg=T["PANEL"], font=("JetBrains Mono", 8, "bold"))
        self.workspace_status.pack(side=LEFT)

    def build_tools(self, parent):
        row = Frame(parent, bg=T["PANEL"])
        row.pack(fill=BOTH, expand=True, padx=8, pady=8)
        for i in range(6):
            row.grid_columnconfigure(i, weight=1)

        cards = [
            ("Test Runner", "test crotolamo", T["GREEN"], "Click OPEN to run"),
            ("Safe Executor", "executor", T["GREEN"], "Secure shell control"),
            ("Memory & Context", "memoria", T["PURPLE"], "Local memory summary"),
            ("Brain Engine", "inteligencia", T["GREEN"], "Reasoning profile"),
            ("Voice", "voz", T["PURPLE"], "Voice bridge / TTS"),
            ("Meme Reactor", "contexto", T["MAGENTA"], "Chaos context"),
        ]

        for i, (title, cmd, color, body) in enumerate(cards):
            card = Frame(row, bg=T["CARD"], highlightbackground=color, highlightthickness=1)
            card.grid(row=0, column=i, sticky="nsew", padx=3)
            Label(card, text=title.upper(), fg=color, bg=T["CARD"], font=("JetBrains Mono", 8, "bold")).pack(anchor="w", padx=8, pady=(8, 3))
            summary = Label(card, text=body, fg=T["TEXT"], bg=T["CARD"], font=("JetBrains Mono", 7), justify="left", anchor="w")
            summary.pack(anchor="w", padx=8, pady=6)
            self.tool_summary_labels[cmd] = summary
            Button(
                card, text="OPEN", bg=T["PANEL_2"], fg=color, relief="flat", font=("JetBrains Mono", 8),
                command=lambda c=cmd: self.run_runtime_command(c)
            ).pack(side="bottom", fill=X, padx=8, pady=8)

    def build_mascots(self, parent):
        row = Frame(parent, bg=T["PANEL"])
        row.pack(fill=BOTH, expand=True, padx=8, pady=8)
        for i in range(3):
            row.grid_columnconfigure(i, weight=1)

        items = [
            ("PEZLÍN 3000", "pezlin_3000.png", "STATUS: ONLINE\nMODE: READY", T["PURPLE"]),
            ("PALOMA SUPREMA 67", "paloma_suprema_67.png", "LASER: ARMED\nMODE: DOMINANDO", T["MAGENTA"]),
            ("HACKER MODE", "hacker_mode.png", "CHAOS LEVEL: MAX\nAESTHETIC: ONLINE", T["CYAN"]),
        ]

        for col, (name, asset, status, color) in enumerate(items):
            box = Frame(row, bg=T["CARD"], highlightbackground=color, highlightthickness=1)
            box.grid(row=0, column=col, sticky="nsew", padx=4)
            Label(box, text=name, fg=color, bg=T["CARD"], font=("JetBrains Mono", 10, "bold")).pack(pady=(8, 4))
            AssetImage(box, asset, height=140).pack(fill=BOTH, expand=True, padx=8)
            Label(box, text=status, fg=T["GREEN"] if col == 0 else (T["RED"] if col == 1 else T["CYAN"]),
                  bg=T["CARD"], font=("JetBrains Mono", 8, "bold")).pack(pady=6)

    def build_command_input(self, parent):
        row = Frame(parent, bg=T["PANEL"])
        row.pack(fill=X, padx=8, pady=8)

        self.entry = Entry(
            row, textvariable=self.command_var,
            bg="#020710", fg=T["TEXT"], insertbackground=T["CYAN"],
            relief="flat", font=("JetBrains Mono", 11)
        )
        self.entry.pack(side=LEFT, fill=X, expand=True, ipady=8)
        self.entry.bind("<Return>", lambda e: self.submit())
        Button(
            row, text="RUN", bg="#072417", fg=T["GREEN"], relief="flat",
            font=("JetBrains Mono", 9, "bold"), command=self.submit
        ).pack(side=LEFT, padx=(8, 0), ipadx=22, ipady=6)

        quick = Frame(parent, bg=T["PANEL"])
        quick.pack(fill=X, padx=8, pady=(0, 8))
        for cmd in ["contexto", "__LOCAL_PROJECT_MAP__", "test crotolamo", "executor", "parches"]:
            label = "project map" if cmd == "__LOCAL_PROJECT_MAP__" else cmd
            Button(
                quick, text=label, bg=T["CARD"], fg=T["MUTED"],
                relief="flat", font=("JetBrains Mono", 7),
                command=lambda c=cmd: self.run_runtime_command(c)
            ).pack(side=LEFT, padx=3)

    def build_activity_feed(self, parent):
        self.feed = Text(parent, bg="#020710", fg=T["TEXT"], relief="flat", font=("JetBrains Mono", 8), wrap="word")
        self.feed.pack(fill=BOTH, expand=True, padx=8, pady=8)
        self.feed.insert(END, "[INFO] Orbital UI v19 híbrida lista.\n[INFO] Mezcla de estética + utilidad activada.\n")
        self.feed.configure(state=DISABLED)

    def build_footer(self):
        foot = Frame(self.root, bg=T["PANEL"], height=72, highlightbackground=T["BORDER"], highlightthickness=1)
        foot.pack(fill=X, padx=8, pady=(0, 8))
        foot.pack_propagate(False)

        sticker = get_asset("sticker_strip.png")
        if sticker:
            lab = Label(foot, image=sticker, bg=T["PANEL"])
            lab.image = sticker
            lab.pack(side=LEFT, fill=Y, padx=4)
        else:
            StickerStrip(foot).pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 8))

        status = Frame(foot, bg=T["CARD"], highlightbackground=T["BORDER"], highlightthickness=1, width=380)
        status.pack(side=RIGHT, fill=Y)
        status.pack_propagate(False)
        Label(status, text="ASSET PACK: v18 | UI CORE: v19", fg=T["CYAN"], bg=T["CARD"], font=("JetBrains Mono", 8)).pack(anchor="w", padx=10, pady=(10, 0))
        Label(status, textvariable=self.status_var, fg=T["GREEN"], bg=T["CARD"], font=("JetBrains Mono", 10, "bold")).pack(anchor="w", padx=10)

    def log_feed(self, text: str):
        self.feed.configure(state=NORMAL)
        self.feed.insert(END, text + "\n")
        self.feed.see(END)
        self.feed.configure(state=DISABLED)

    def write_terminal(self, text: str):
        self.terminal.configure(state=NORMAL)
        self.terminal.insert(END, text[:9000])
        self.terminal.see(END)
        self.terminal.configure(state=DISABLED)

    def set_workspace_text(self, text: str):
        self.workspace_text.configure(state=NORMAL)
        self.workspace_text.delete("1.0", END)
        self.workspace_text.insert(END, text[:20000])
        self.workspace_text.configure(state=DISABLED)

    def set_inspector_text(self, text: str):
        self.inspector_text.configure(state=NORMAL)
        self.inspector_text.delete("1.0", END)
        self.inspector_text.insert(END, text[:12000])
        self.inspector_text.configure(state=DISABLED)

    def submit(self):
        cmd = self.command_var.get().strip()
        if cmd:
            self.command_var.set("")
            self.run_runtime_command(cmd)

    def switch_workspace(self, label: str, cmd: str):
        self.workspace_mode.set(label)
        self.run_runtime_command(cmd)

    def refresh_local_stats(self):
        stats = collect_project_stats(self.project_root)
        inspector = [
            f"Root: {self.project_root}",
            f"Directorios: {stats['dirs']}",
            f"Archivos: {stats['files']}",
            f"Python: {stats['py_files']}",
            f"Archivo más grande: {stats['largest_name']}",
            f"Tamaño: {format_bytes(stats['largest_size'])}",
            "",
            f"Último comando: {self.last_command}",
        ]
        self.set_inspector_text("\n".join(inspector))

    def run_runtime_command(self, cmd: str):
        self.last_command = cmd
        self.write_terminal(f"\n\ncrotolamo@orbital:~$ {cmd}\n")
        self.status_var.set("STATUS: RUNNING")
        self.node_status.configure(text="WORKING ●", fg=T["YELLOW"])
        self.workspace_status.configure(text=f"Running: {cmd}", fg=T["YELLOW"])
        self.log_feed(f"[RUN] {cmd}")

        if cmd == "__LOCAL_PROJECT_MAP__":
            self.workspace_mode.set("PROJECT MAP")
            text = build_tree_lines(self.project_root)
            self.set_workspace_text(text)
            self.refresh_local_stats()
            self.status_var.set("STATUS: ONLINE")
            self.node_status.configure(text="ONLINE ●", fg=T["GREEN"])
            self.workspace_status.configure(text="Workspace: local project map", fg=T["GREEN"])
            self.log_feed("[OK] Mapa local generado.")
            return

        if cmd == "__LOCAL_INSPECTOR__":
            self.workspace_mode.set("INSPECTOR")
            stats = collect_project_stats(self.project_root)
            body = [
                "PROJECT INSPECTOR",
                "-----------------",
                f"Proyecto: {self.project_root.name}",
                f"Ruta: {self.project_root}",
                f"Directorios: {stats['dirs']}",
                f"Archivos: {stats['files']}",
                f"Archivos .py: {stats['py_files']}",
                f"Archivo más grande: {stats['largest_name']}",
                f"Tamaño: {format_bytes(stats['largest_size'])}",
                "",
                "Entrypoints sugeridos:",
                "- launch_runtime_shell.py",
                "- launch_orbital_ui_pro.py",
                "- tools/crotolamo_doctor.py",
            ]
            self.set_workspace_text("\n".join(body))
            self.refresh_local_stats()
            self.status_var.set("STATUS: ONLINE")
            self.node_status.configure(text="ONLINE ●", fg=T["GREEN"])
            self.workspace_status.configure(text="Workspace: local inspector", fg=T["GREEN"])
            self.log_feed("[OK] Inspector local actualizado.")
            return

        if cmd == "__LOCAL_HACKER__":
            self.set_workspace_text("HACKER MODE ACTIVATED\n\n- aesthetic layer: ONLINE\n- chaos factor: MAX\n- seriousness: preserved")
            self.set_inspector_text("Asset preview active:\n- hacker_mode.png\n- sticker_strip.png\n- galaxy_bg.png")
            self.status_var.set("STATUS: ONLINE")
            self.node_status.configure(text="ONLINE ●", fg=T["GREEN"])
            self.workspace_status.configure(text="Workspace: hacker mode", fg=T["GREEN"])
            self.log_feed("[OK] Hacker mode preview.")
            return

        threading.Thread(target=self.worker, args=(cmd,), daemon=True).start()

    def worker(self, cmd: str):
        if isinstance(self.runtime, Exception):
            self.queue.put(("error", cmd, f"Runtime no cargó: {self.runtime}"))
            return
        try:
            out = self.runtime.process_text(cmd)
            self.queue.put(("ok", cmd, render_result(out)))
        except Exception as e:
            self.queue.put(("error", cmd, f"{type(e).__name__}: {e}"))

    def update_tool_summary(self, cmd: str, text: str):
        label = self.tool_summary_labels.get(cmd)
        if not label:
            return
        summary = text.strip().replace("\r", "")
        lines = [ln for ln in summary.splitlines() if ln.strip()]
        if not lines:
            label.configure(text="Sin salida.")
            return
        label.configure(text="\n".join(lines[:3])[:160])

    def poll(self):
        try:
            while True:
                kind, cmd, msg = self.queue.get_nowait()
                if kind == "ok":
                    preview = str(msg)[:9000]
                    self.write_terminal(preview + "\n")
                    self.status_var.set("STATUS: ONLINE")
                    self.node_status.configure(text="ONLINE ●", fg=T["GREEN"])
                    self.workspace_status.configure(text=f"Done: {cmd}", fg=T["GREEN"])
                    self.log_feed(f"[OK] {cmd}")
                    self.update_tool_summary(cmd, preview)
                    # route useful outputs to workspace
                    if cmd in {"proyectos", "mapa crotolamo", "inspeccionar crotolamo", "plan limpiar estructura de crotolamo", "parches", "test crotolamo", "executor", "memoria", "inteligencia", "contexto"}:
                        self.set_workspace_text(preview)
                    self.refresh_local_stats()
                else:
                    self.write_terminal("[ERROR] " + str(msg) + "\n")
                    self.status_var.set("STATUS: ERROR")
                    self.node_status.configure(text="ERROR ●", fg=T["RED"])
                    self.workspace_status.configure(text=f"Error: {cmd}", fg=T["RED"])
                    self.log_feed(f"[ERROR] {cmd}: {msg}")
                    self.update_tool_summary(cmd, f"ERROR\n{msg}")
        except queue.Empty:
            pass
        self.root.after(120, self.poll)

    def update_metrics(self):
        if psutil:
            try:
                self.cpu.value = int(psutil.cpu_percent(interval=None))
                self.ram.value = int(psutil.virtual_memory().percent)
            except Exception:
                pass
        self.root.after(1500, self.update_metrics)


def main():
    app = Tk()
    try:
        ttk.Style(app).theme_use("clam")
    except Exception:
        pass
    OrbitalUIProV19(app)
    app.mainloop()


if __name__ == "__main__":
    main()
