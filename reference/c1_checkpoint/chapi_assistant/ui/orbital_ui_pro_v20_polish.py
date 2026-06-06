from __future__ import annotations

import math
import os
import sys
import queue
import random
import threading
from pathlib import Path
from tkinter import (
    Tk, Canvas, Frame, Label, Button, Entry, Text, StringVar,
    BOTH, LEFT, RIGHT, X, Y, END, NORMAL, DISABLED
)
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.orbital_theme import THEME as T
from ui.orbital_widgets_v17 import SystemPulse, RadarBadge
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


def format_bytes(n: int):
    units = ["B", "KB", "MB", "GB"]
    value = float(max(n, 0))
    for u in units:
        if value < 1024 or u == units[-1]:
            return f"{int(value)} {u}" if u == "B" else f"{value:.1f} {u}"
        value /= 1024.0
    return f"{n} B"


def detect_projects(root: Path):
    candidates: dict[str, Path] = {"Crotolamo": root}
    docs = Path.home() / "Documentos"
    search_roots = [docs, root.parent]
    seen: set[Path] = set()

    for parent in search_roots:
        if not parent.exists():
            continue
        for item in parent.iterdir():
            try:
                rp = item.resolve()
            except Exception:
                rp = item
            if rp in seen:
                continue
            seen.add(rp)
            if not item.is_dir():
                continue
            name = item.name.lower()
            if item == root:
                continue
            if "huevonitis" in name:
                candidates[f"Huevonitis · {item.name}"] = item
            elif "tletl" in name:
                candidates[f"Tletl · {item.name}"] = item
            elif "crotolamo" in name or "chapi_assistant" in name:
                candidates[f"Crotolamo · {item.name}"] = item
    return candidates


def collect_project_stats(root: Path):
    stats = {
        "dirs": 0,
        "files": 0,
        "py_files": 0,
        "largest_name": "N/A",
        "largest_size": 0,
        "total_size": 0,
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
                    stats["total_size"] += size
                    if size > stats["largest_size"]:
                        stats["largest_size"] = size
                        stats["largest_name"] = str(path.relative_to(root))
                except Exception:
                    pass
    except Exception:
        pass
    return stats


def build_tree_lines(root: Path, max_depth: int = 3, max_entries: int = 10):
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
            lines.append(prefix + f"└── ... +{len(entries) - max_entries} elementos")

    walk(root, "", 1)
    return "\n".join(lines)


def find_project_assets(root: Path):
    hits = []
    for rel in [
        "assets/orbital_ui/galaxy_bg.png",
        "assets/orbital_ui/pezlin_3000.png",
        "assets/orbital_ui/paloma_suprema_67.png",
        "assets/orbital_ui/hacker_mode.png",
        "assets/orbital_ui/sticker_strip.png",
    ]:
        path = root / rel
        if path.exists():
            hits.append(path)
    return hits


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
                bg=T["PANEL"], fg=accent, font=("JetBrains Mono", 9, "bold"),
            ).pack(side=LEFT)
            Label(head, text="● ○", bg=T["PANEL"], fg=T["MUTED"], font=("JetBrains Mono", 8)).pack(side=RIGHT)


class FittedAssetImage(Canvas):
    def __init__(self, master, asset_name, bg=None, **kwargs):
        super().__init__(master, bg=bg or T["CARD"], highlightthickness=0, **kwargs)
        self.asset_name = asset_name
        self.tk_img = None
        self.bind("<Configure>", lambda e: self.draw())

    def set_asset(self, asset_name: str):
        self.asset_name = asset_name
        self.draw()

    def draw(self):
        self.delete("all")
        w = max(self.winfo_width(), 30)
        h = max(self.winfo_height(), 30)
        img = get_asset(self.asset_name)
        if not img:
            self.create_text(w/2, h/2, text=self.asset_name, fill=T["MUTED"], font=("JetBrains Mono", 8))
            return

        iw = max(img.width(), 1)
        ih = max(img.height(), 1)
        sx = math.ceil(iw / max(w - 12, 1))
        sy = math.ceil(ih / max(h - 12, 1))
        factor = max(1, sx, sy)
        fitted = img.subsample(factor, factor) if factor > 1 else img
        self.tk_img = fitted
        self.create_image(w/2, h/2, image=self.tk_img)


class StickerFooter(Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, bg=T["PANEL"], highlightthickness=0, **kwargs)
        self.tk_img = None
        self.bind("<Configure>", lambda e: self.draw())

    def draw(self):
        self.delete("all")
        w = max(self.winfo_width(), 40)
        h = max(self.winfo_height(), 40)
        img = get_asset("sticker_strip.png")
        if img:
            iw = max(img.width(), 1)
            ih = max(img.height(), 1)
            sx = math.ceil(iw / max(w - 10, 1))
            sy = math.ceil(ih / max(h - 8, 1))
            factor = max(1, sx, sy)
            fitted = img.subsample(factor, factor) if factor > 1 else img
            self.tk_img = fitted
            self.create_image(w/2, h/2, image=self.tk_img)
        else:
            labels = [("LATIN MAFIA", T["YELLOW"]), ("CAOS MODE", T["GREEN"]), ("67", T["YELLOW"]), ("I ♥ CAOS", T["MAGENTA"])]
            x = 10
            for text, color in labels:
                self.create_rectangle(x, 8, x + 120, h - 8, outline=color)
                self.create_text(x + 60, h/2, text=text, fill=color, font=("JetBrains Mono", 10, "bold"))
                x += 130


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
        self.delete("all")
        w, h = max(self.winfo_width(), 40), max(self.winfo_height(), 40)

        bg = get_asset("galaxy_bg.png")
        if bg:
            iw = max(bg.width(), 1)
            ih = max(bg.height(), 1)
            sx = math.ceil(iw / max(w, 1))
            sy = math.ceil(ih / max(h, 1))
            factor = max(1, sx, sy)
            fitted = bg.subsample(factor, factor) if factor > 1 else bg
            self._bg = fitted
            self.create_image(w/2, h/2, image=self._bg)

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


class ToolCard(Frame):
    def __init__(self, master, title, cmd, accent, app, initial="Click OPEN", **kwargs):
        super().__init__(master, bg=T["CARD"], highlightbackground=accent, highlightthickness=1, **kwargs)
        self.cmd = cmd
        self.app = app
        self.accent = accent
        Label(self, text=title.upper(), fg=accent, bg=T["CARD"], font=("JetBrains Mono", 8, "bold")).pack(anchor="w", padx=8, pady=(8, 3))
        self.summary = Label(self, text=initial, fg=T["TEXT"], bg=T["CARD"], font=("JetBrains Mono", 7), justify="left", anchor="w")
        self.summary.pack(anchor="w", padx=8, pady=(0, 6), fill=X)
        meter_box = Frame(self, bg=T["CARD"])
        meter_box.pack(fill=X, padx=8, pady=(0, 8))
        self.progress = ttk.Progressbar(meter_box, mode="determinate", maximum=100)
        self.progress.pack(fill=X)
        self.progress["value"] = 35
        self.status_label = Label(self, text="IDLE", fg=T["MUTED"], bg=T["CARD"], font=("JetBrains Mono", 7, "bold"))
        self.status_label.pack(anchor="w", padx=8, pady=(0, 6))
        Button(
            self, text="OPEN", bg=T["PANEL_2"], fg=accent,
            relief="flat", font=("JetBrains Mono", 8),
            command=lambda: app.run_runtime_command(cmd)
        ).pack(side="bottom", fill=X, padx=8, pady=8)

    def set_summary(self, text: str):
        lines = [ln for ln in text.replace("\r", "").splitlines() if ln.strip()]
        self.summary.configure(text="\n".join(lines[:3])[:180] if lines else "Sin salida.")
        score = min(100, max(8, len(text.strip()) % 101))
        self.progress["value"] = score
        self.status_label.configure(text="UPDATED", fg=T["GREEN"])

    def set_state_running(self):
        self.status_label.configure(text="RUNNING", fg=T["YELLOW"])

    def set_state_error(self, text="ERROR"):
        self.status_label.configure(text=text, fg=T["RED"])


class OrbitalUIProV20:
    def __init__(self, root):
        self.root = root
        self.root.title("CROTOLAMO ORBITAL UI PRO v20")
        self.root.geometry("1640x960")
        self.root.minsize(1320, 800)
        self.root.configure(bg=T["BG"])

        self.runtime = safe_runtime()
        self.queue = queue.Queue()
        self.command_var = StringVar()
        self.status_var = StringVar(value="STATUS: ONLINE")
        self.workspace_mode = StringVar(value="PROJECT MAP")
        self.project_selector_var = StringVar()
        self.last_command = "startup"
        self.tool_cards: dict[str, ToolCard] = {}
        self.asset_preview_name = "hacker_mode.png"

        self.projects = detect_projects(ROOT)
        default_name = next(iter(self.projects.keys()))
        self.project_selector_var.set(default_name)
        self.project_root = self.projects[default_name]

        self.build()
        self.refresh_project_views(force_map=True)
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
        Label(logo, text="ORBITAL UI PRO v20 / POLISH + LIVE WORKSPACE", fg=T["MAGENTA"], bg=T["PANEL"], font=("JetBrains Mono", 9, "bold")).pack(anchor="w")

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

        side = Panel(body, accent=T["BORDER"], width=215)
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

        p2 = Panel(main, "PROJECT WORKSPACE / LIVE", "02", T["PURPLE"])
        p2.grid(row=0, column=1, sticky="nsew", pady=(0, 8))
        self.build_workspace(p2)

        p3 = Panel(main, "ORBITAL TOOLS & MODULES", "03", T["CYAN"])
        p3.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        self.build_tools(p3)

        p4 = Panel(main, "MASCOTS / PERSONALITY", "04", T["PURPLE"])
        p4.grid(row=1, column=1, sticky="nsew", pady=(0, 8))
        self.build_mascots(p4)

        p5 = Panel(main, "COMMAND INPUT", "05", T["GREEN"])
        p5.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        self.build_command_input(p5)

        p6 = Panel(main, "ACTIVITY FEED / TELEMETRY", "06", T["CYAN"])
        p6.grid(row=2, column=1, sticky="nsew")
        self.build_activity_feed(p6)

    def build_sidebar(self, parent):
        def heading(text, color):
            Label(parent, text=text, fg=color, bg=T["PANEL"], font=("JetBrains Mono", 8, "bold"), anchor="w").pack(fill=X, padx=12, pady=(12, 4))

        def item(text, cmd, color):
            Button(
                parent, text=text, anchor="w",
                bg=T["CARD"], fg=color, activebackground=T["PANEL_2"], activeforeground=T["TEXT"],
                relief="flat", font=("JetBrains Mono", 9),
                command=lambda: self.run_runtime_command(cmd)
            ).pack(fill=X, padx=10, pady=3)

        heading("CORE", T["CYAN"]); item("Command Core", "runtime", T["CYAN"])
        heading("PROJECTS", T["PURPLE"]); item("Project Map", "__LOCAL_PROJECT_MAP__", T["PURPLE"]); item("Inspector", "__LOCAL_INSPECTOR__", T["PURPLE"])
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
            "> workspace live bridge... ok",
            "> visual polish layer... ok",
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
            ("ORBITAL CORE", "98.2%", T["GREEN"]),
            ("SYNC LATENCY", "10ms", T["CYAN"]),
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
        top = Frame(parent, bg=T["PANEL"])
        top.pack(fill=X, padx=8, pady=(6, 4))

        Label(top, text="PROJECT", fg=T["MUTED"], bg=T["PANEL"], font=("JetBrains Mono", 8, "bold")).pack(side=LEFT, padx=(0, 6))
        self.project_combo = ttk.Combobox(top, textvariable=self.project_selector_var, values=list(self.projects.keys()), state="readonly", width=34)
        self.project_combo.pack(side=LEFT, padx=(0, 8))
        self.project_combo.bind("<<ComboboxSelected>>", lambda e: self.on_project_change())
        Button(top, text="REFRESH", bg=T["CARD"], fg=T["GREEN"], relief="flat", font=("JetBrains Mono", 8), command=lambda: self.refresh_project_views(force_map=False)).pack(side=LEFT, padx=(0, 8))
        self.workspace_status = Label(top, text="Workspace ready", fg=T["GREEN"], bg=T["PANEL"], font=("JetBrains Mono", 8, "bold"))
        self.workspace_status.pack(side=RIGHT)

        tabs = Frame(parent, bg=T["PANEL"])
        tabs.pack(fill=X, padx=8, pady=(0, 6))
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
        Label(right, text="PROJECT SNAPSHOT", fg=T["CYAN"], bg=T["CARD"], font=("JetBrains Mono", 8, "bold")).pack(anchor="w", padx=10, pady=(10, 4))

        self.inspector_text = Text(right, bg=T["CARD"], fg=T["TEXT"], relief="flat", font=("JetBrains Mono", 8), wrap="word", height=9)
        self.inspector_text.pack(fill=X, padx=10, pady=(0, 10))

        Label(right, text="ASSET PREVIEW", fg=T["PURPLE"], bg=T["CARD"], font=("JetBrains Mono", 8, "bold")).pack(anchor="w", padx=10, pady=(2, 6))
        self.asset_preview = FittedAssetImage(right, self.asset_preview_name, bg=T["CARD"], height=150)
        self.asset_preview.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))

    def build_tools(self, parent):
        row = Frame(parent, bg=T["PANEL"])
        row.pack(fill=BOTH, expand=True, padx=8, pady=8)
        for i in range(6):
            row.grid_columnconfigure(i, weight=1)

        items = [
            ("Test Runner", "test crotolamo", T["GREEN"], "Run suite and inspect output."),
            ("Safe Executor", "executor", T["GREEN"], "Shell guard and execution policy."),
            ("Memory & Context", "memoria", T["PURPLE"], "Memory status and context depth."),
            ("Brain Engine", "inteligencia", T["GREEN"], "Reasoning and complexity profile."),
            ("Voice", "voz", T["PURPLE"], "Voice input / output bridge."),
            ("Meme Reactor", "contexto", T["MAGENTA"], "Context chaos and meme layer."),
        ]
        for i, (title, cmd, color, body) in enumerate(items):
            card = ToolCard(row, title, cmd, color, self, initial=body)
            card.grid(row=0, column=i, sticky="nsew", padx=3)
            self.tool_cards[cmd] = card

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
            FittedAssetImage(box, asset, bg=T["CARD"], height=145).pack(fill=BOTH, expand=True, padx=8)
            Label(
                box, text=status,
                fg=T["GREEN"] if col == 0 else (T["RED"] if col == 1 else T["CYAN"]),
                bg=T["CARD"], font=("JetBrains Mono", 8, "bold")
            ).pack(pady=6)

    def build_command_input(self, parent):
        row = Frame(parent, bg=T["PANEL"])
        row.pack(fill=X, padx=8, pady=8)

        self.entry = Entry(row, textvariable=self.command_var, bg="#020710", fg=T["TEXT"], insertbackground=T["CYAN"], relief="flat", font=("JetBrains Mono", 11))
        self.entry.pack(side=LEFT, fill=X, expand=True, ipady=8)
        self.entry.bind("<Return>", lambda e: self.submit())
        Button(row, text="RUN", bg="#072417", fg=T["GREEN"], relief="flat", font=("JetBrains Mono", 9, "bold"), command=self.submit).pack(side=LEFT, padx=(8, 0), ipadx=22, ipady=6)

        quick = Frame(parent, bg=T["PANEL"])
        quick.pack(fill=X, padx=8, pady=(0, 8))
        for cmd in ["__LOCAL_PROJECT_MAP__", "__LOCAL_INSPECTOR__", "test crotolamo", "executor", "parches"]:
            label = cmd.replace("__LOCAL_", "").replace("__", "").replace("_", " ").lower() if cmd.startswith("__LOCAL_") else cmd
            Button(quick, text=label, bg=T["CARD"], fg=T["MUTED"], relief="flat", font=("JetBrains Mono", 7), command=lambda c=cmd: self.run_runtime_command(c)).pack(side=LEFT, padx=3)

    def build_activity_feed(self, parent):
        self.feed = Text(parent, bg="#020710", fg=T["TEXT"], relief="flat", font=("JetBrains Mono", 8), wrap="word")
        self.feed.pack(fill=BOTH, expand=True, padx=8, pady=8)
        self.feed.insert(END, "[INFO] Orbital UI v20 lista.\n[INFO] Layout polish + live workspace activados.\n")
        self.feed.configure(state=DISABLED)

    def build_footer(self):
        foot = Frame(self.root, bg=T["PANEL"], height=86, highlightbackground=T["BORDER"], highlightthickness=1)
        foot.pack(fill=X, padx=8, pady=(0, 8))
        foot.pack_propagate(False)

        left = Frame(foot, bg=T["PANEL"])
        left.pack(side=LEFT, fill=BOTH, expand=True, padx=(4, 0), pady=4)
        StickerFooter(left, height=74).pack(fill=BOTH, expand=True)

        status = Frame(foot, bg=T["CARD"], highlightbackground=T["BORDER"], highlightthickness=1, width=420)
        status.pack(side=RIGHT, fill=Y, padx=4, pady=4)
        status.pack_propagate(False)
        Label(status, text="ASSET PACK: v18 | UI CORE: v20", fg=T["CYAN"], bg=T["CARD"], font=("JetBrains Mono", 8)).pack(anchor="w", padx=10, pady=(10, 0))
        Label(status, textvariable=self.status_var, fg=T["GREEN"], bg=T["CARD"], font=("JetBrains Mono", 10, "bold")).pack(anchor="w", padx=10)
        self.footer_project_label = Label(status, text="", fg=T["TEXT"], bg=T["CARD"], font=("JetBrains Mono", 8))
        self.footer_project_label.pack(anchor="w", padx=10, pady=(4, 0))

    def log_feed(self, text: str):
        self.feed.configure(state=NORMAL)
        self.feed.insert(END, text + "\n")
        self.feed.see(END)
        self.feed.configure(state=DISABLED)

    def write_terminal(self, text: str):
        self.terminal.configure(state=NORMAL)
        self.terminal.insert(END, text[:12000])
        self.terminal.see(END)
        self.terminal.configure(state=DISABLED)

    def set_workspace_text(self, text: str):
        self.workspace_text.configure(state=NORMAL)
        self.workspace_text.delete("1.0", END)
        self.workspace_text.insert(END, text[:30000])
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

    def on_project_change(self):
        name = self.project_selector_var.get()
        self.project_root = self.projects.get(name, ROOT)
        self.refresh_project_views(force_map=True)
        self.log_feed(f"[PROJECT] {name}")

    def refresh_project_views(self, force_map: bool = False):
        stats = collect_project_stats(self.project_root)
        inspector = [
            f"Proyecto: {self.project_selector_var.get()}",
            f"Ruta: {self.project_root}",
            f"Directorios: {stats['dirs']}",
            f"Archivos: {stats['files']}",
            f"Python: {stats['py_files']}",
            f"Tamaño total: {format_bytes(stats['total_size'])}",
            f"Archivo más grande: {stats['largest_name']}",
            f"Tamaño: {format_bytes(stats['largest_size'])}",
            "",
            f"Último comando: {self.last_command}",
        ]
        self.set_inspector_text("\n".join(inspector))
        self.footer_project_label.configure(text=f"Proyecto activo: {self.project_selector_var.get()}")

        assets = find_project_assets(self.project_root)
        if assets:
            self.asset_preview_name = assets[-1].name
            self.asset_preview.set_asset(self.asset_preview_name)

        if force_map or self.workspace_mode.get() == "PROJECT MAP":
            self.workspace_mode.set("PROJECT MAP")
            self.set_workspace_text(build_tree_lines(self.project_root))
            self.workspace_status.configure(text="Workspace: project map", fg=T["GREEN"])

    def switch_workspace(self, label: str, cmd: str):
        self.workspace_mode.set(label)
        self.run_runtime_command(cmd)

    def run_runtime_command(self, cmd: str):
        self.last_command = cmd
        self.write_terminal(f"\n\ncrotolamo@orbital:~$ {cmd}\n")
        self.status_var.set("STATUS: RUNNING")
        self.node_status.configure(text="WORKING ●", fg=T["YELLOW"])
        self.workspace_status.configure(text=f"Running: {cmd}", fg=T["YELLOW"])
        self.log_feed(f"[RUN] {cmd}")

        if cmd in self.tool_cards:
            self.tool_cards[cmd].set_state_running()

        if cmd == "__LOCAL_PROJECT_MAP__":
            self.workspace_mode.set("PROJECT MAP")
            self.set_workspace_text(build_tree_lines(self.project_root))
            self.refresh_project_views(force_map=False)
            self.finish_local_ok("Mapa local generado.")
            return

        if cmd == "__LOCAL_INSPECTOR__":
            self.workspace_mode.set("INSPECTOR")
            stats = collect_project_stats(self.project_root)
            body = [
                "PROJECT INSPECTOR",
                "-----------------",
                f"Proyecto: {self.project_selector_var.get()}",
                f"Ruta: {self.project_root}",
                f"Directorios: {stats['dirs']}",
                f"Archivos: {stats['files']}",
                f"Archivos .py: {stats['py_files']}",
                f"Tamaño total: {format_bytes(stats['total_size'])}",
                f"Archivo más grande: {stats['largest_name']}",
                f"Tamaño: {format_bytes(stats['largest_size'])}",
                "",
                "Entrypoints posibles:",
                "- launch_runtime_shell.py",
                "- launch_orbital_ui_pro.py",
                "- tools/crotolamo_doctor.py",
            ]
            self.set_workspace_text("\n".join(body))
            self.refresh_project_views(force_map=False)
            self.finish_local_ok("Inspector local actualizado.")
            return

        if cmd == "__LOCAL_HACKER__":
            self.asset_preview.set_asset("hacker_mode.png")
            self.set_workspace_text("HACKER MODE ACTIVATED\n\n- aesthetic layer: ONLINE\n- chaos factor: MAX\n- seriousness: preserved")
            self.set_inspector_text("Asset preview activo:\n- hacker_mode.png\n- sticker_strip.png\n- galaxy_bg.png")
            self.finish_local_ok("Hacker mode preview.")
            return

        threading.Thread(target=self.worker, args=(cmd,), daemon=True).start()

    def finish_local_ok(self, message: str):
        self.status_var.set("STATUS: ONLINE")
        self.node_status.configure(text="ONLINE ●", fg=T["GREEN"])
        self.workspace_status.configure(text=message, fg=T["GREEN"])
        self.log_feed(f"[OK] {message}")

    def worker(self, cmd: str):
        if isinstance(self.runtime, Exception):
            self.queue.put(("error", cmd, f"Runtime no cargó: {self.runtime}"))
            return
        try:
            out = self.runtime.process_text(cmd)
            self.queue.put(("ok", cmd, render_result(out)))
        except Exception as e:
            self.queue.put(("error", cmd, f"{type(e).__name__}: {e}"))

    def poll(self):
        try:
            while True:
                kind, cmd, msg = self.queue.get_nowait()
                if kind == "ok":
                    preview = str(msg)[:12000]
                    self.write_terminal(preview + "\n")
                    self.status_var.set("STATUS: ONLINE")
                    self.node_status.configure(text="ONLINE ●", fg=T["GREEN"])
                    self.workspace_status.configure(text=f"Done: {cmd}", fg=T["GREEN"])
                    self.log_feed(f"[OK] {cmd}")
                    if cmd in self.tool_cards:
                        self.tool_cards[cmd].set_summary(preview)
                    if cmd in {"proyectos", "mapa crotolamo", "inspeccionar crotolamo", "plan limpiar estructura de crotolamo", "parches", "test crotolamo", "executor", "memoria", "inteligencia", "contexto"}:
                        self.set_workspace_text(preview)
                    self.refresh_project_views(force_map=False)
                else:
                    self.write_terminal("[ERROR] " + str(msg) + "\n")
                    self.status_var.set("STATUS: ERROR")
                    self.node_status.configure(text="ERROR ●", fg=T["RED"])
                    self.workspace_status.configure(text=f"Error: {cmd}", fg=T["RED"])
                    self.log_feed(f"[ERROR] {cmd}: {msg}")
                    if cmd in self.tool_cards:
                        self.tool_cards[cmd].set_state_error()
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
        try:
            self.core_chip_labels["SYNC LATENCY"].configure(text=f"{random.randint(8, 17)}ms")
            self.core_chip_labels["ENTROPY"].configure(text=f"{random.randint(61, 77)}%")
        except Exception:
            pass
        self.root.after(1500, self.update_metrics)


def main():
    app = Tk()
    try:
        style = ttk.Style(app)
        style.theme_use("clam")
    except Exception:
        pass
    OrbitalUIProV20(app)
    app.mainloop()


if __name__ == "__main__":
    main()
