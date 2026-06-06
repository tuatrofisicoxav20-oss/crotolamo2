from __future__ import annotations
import random, sys, threading, queue, math
from pathlib import Path
from tkinter import Tk, Canvas, Frame, Label, Button, Entry, Text, StringVar, BOTH, LEFT, RIGHT, X, Y, END, NORMAL, DISABLED
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.orbital_theme import THEME as T
from ui.orbital_widgets_v17 import SystemPulse, StickerStrip, RadarBadge

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

class Panel(Frame):
    def __init__(self, master, title="", number="", accent=None, **kwargs):
        accent = accent or T["CYAN"]
        super().__init__(master, bg=T["PANEL"], highlightbackground=accent, highlightthickness=1, **kwargs)
        if title:
            head = Frame(self, bg=T["PANEL"])
            head.pack(fill=X, padx=10, pady=(7, 2))
            prefix = f"{number}. " if number else ""
            Label(head, text=(prefix + title).upper(), bg=T["PANEL"], fg=accent,
                  font=("JetBrains Mono", 9, "bold"), anchor="w").pack(side=LEFT)
            Label(head, text="● ○", bg=T["PANEL"], fg=T["MUTED"], font=("JetBrains Mono", 8)).pack(side=RIGHT)

class OrbitCanvas(Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, bg="#020711", highlightthickness=0, **kwargs)
        self.t = 0
        self.after(33, self.tick)
    def tick(self):
        self.t += .035
        self.draw()
        self.after(33, self.tick)
    def draw(self):
        self.delete("all")
        w, h = max(self.winfo_width(), 10), max(self.winfo_height(), 10)
        cx, cy = w * .55, h * .50
        random.seed(670)
        for _ in range(120):
            x, y = random.randint(0, w), random.randint(0, h)
            c = random.choice(["#12335c", "#204b94", "#5c2cff", "#00d9ff", "#ff33cc"])
            self.create_oval(x, y, x+1, y+1, fill=c, outline="")
        for i in range(12):
            s = i/11
            self.create_rectangle(cx-(w*.04+s*w*.42), cy-(h*.04+s*h*.42), cx+(w*.04+s*w*.42), cy+(h*.04+s*h*.42), outline="#12366d")
        for i in range(-8, 9):
            x = cx + i*w*.045
            self.create_line(cx, cy, x, h, fill="#112c5b")
            self.create_line(cx, cy, x, 0, fill="#112c5b")
        for gx, gy, scale in [(w*.24, h*.32, 1.0), (w*.82, h*.35, 1.25), (w*.75, h*.75, .75)]:
            for k in range(32):
                a = self.t + k*.5
                r = k*1.75*scale
                x = gx + math.cos(a)*r
                y = gy + math.sin(a)*r*.45
                self.create_oval(x, y, x+2, y+2, fill=random.choice([T["PURPLE"], T["CYAN"], T["MAGENTA"], T["BLUE"]]), outline="")
            self.create_oval(gx-5, gy-5, gx+5, gy+5, fill=T["YELLOW"], outline="")
        size = 96 + math.sin(self.t*2)*5
        off = size*.34
        front = [(cx-size/2, cy-size/2), (cx+size/2, cy-size/2), (cx+size/2, cy+size/2), (cx-size/2, cy+size/2)]
        back = [(x+off, y-off) for x,y in front]
        for poly, color in [(front, T["CYAN"]), (back, T["BLUE"])]:
            self.create_polygon(*sum(poly, ()), outline=color, fill="", width=2)
        for a,b in zip(front, back):
            self.create_line(*a, *b, fill=T["PURPLE"], width=2)
        for i, r in enumerate([78, 118, 160, 205]):
            self.create_oval(cx-r*1.45, cy-r*.45, cx+r*1.45, cy+r*.45,
                             outline=[T["CYAN"], T["PURPLE"], T["BLUE"], T["MAGENTA"]][i], width=1)

class OrbitalUIProV17:
    def __init__(self, root):
        self.root = root
        self.root.title("CROTOLAMO ORBITAL UI PRO v17")
        self.root.geometry("1540x880")
        self.root.minsize(1250, 760)
        self.root.configure(bg=T["BG"])
        self.runtime = safe_runtime()
        self.queue = queue.Queue()
        self.command_var = StringVar()
        self.status_var = StringVar(value="STATUS: ONLINE")
        self.build()
        self.update_metrics()
        self.poll()

    def build(self):
        self.topbar()
        self.content()
        self.footer()

    def topbar(self):
        top = Frame(self.root, bg=T["PANEL"], height=62, highlightbackground=T["PURPLE"], highlightthickness=1)
        top.pack(fill=X, padx=8, pady=8)
        top.pack_propagate(False)
        logo = Frame(top, bg=T["PANEL"]); logo.pack(side=LEFT, padx=14)
        Label(logo, text="CROTOLAMO", fg=T["TEXT"], bg=T["PANEL"], font=("JetBrains Mono", 21, "bold")).pack(anchor="w")
        Label(logo, text="ORBITAL UI PRO v17", fg=T["MAGENTA"], bg=T["PANEL"], font=("JetBrains Mono", 9, "bold")).pack(anchor="w")
        status = Frame(top, bg=T["CARD"], highlightbackground=T["BORDER"], highlightthickness=1); status.pack(side=LEFT, padx=20, pady=8)
        Label(status, text="SYSTEM STATUS", fg=T["MUTED"], bg=T["CARD"], font=("JetBrains Mono", 7, "bold")).pack(anchor="w", padx=10, pady=(4,0))
        Label(status, text="● ALL SYSTEMS OPERATIONAL", fg=T["GREEN"], bg=T["CARD"], font=("JetBrains Mono", 9, "bold")).pack(anchor="w", padx=10, pady=(0,4))
        metrics = Frame(top, bg=T["PANEL"]); metrics.pack(side=LEFT, fill=Y, expand=True)
        self.cpu_pulse = SystemPulse(metrics, "CPU", 67); self.cpu_pulse.pack(side=LEFT, padx=4, pady=8)
        self.ram_pulse = SystemPulse(metrics, "RAM", 72); self.ram_pulse.pack(side=LEFT, padx=4, pady=8)
        self.gpu_pulse = SystemPulse(metrics, "GPU", 81); self.gpu_pulse.pack(side=LEFT, padx=4, pady=8)
        right = Frame(top, bg=T["PANEL"]); right.pack(side=RIGHT, padx=10)
        RadarBadge(right).pack(side=LEFT, padx=8)
        Label(right, text="CROTOLAMO-67", fg=T["TEXT"], bg=T["PANEL"], font=("JetBrains Mono", 10, "bold")).pack(anchor="e")
        Label(right, text="Orbital Operator", fg=T["MUTED"], bg=T["PANEL"], font=("JetBrains Mono", 8)).pack(anchor="e")

    def content(self):
        body = Frame(self.root, bg=T["BG"]); body.pack(fill=BOTH, expand=True, padx=8, pady=(0,6))
        side = Panel(body, accent=T["BORDER"], width=205); side.pack(side=LEFT, fill=Y, padx=(0,8)); side.pack_propagate(False)
        self.sidebar(side)
        main = Frame(body, bg=T["BG"]); main.pack(side=LEFT, fill=BOTH, expand=True)
        main.grid_columnconfigure(0, weight=3); main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=5); main.grid_rowconfigure(1, weight=2); main.grid_rowconfigure(2, weight=1)
        command = Panel(main, "COMMAND CORE / RUNTIME CONSOLE", "01", T["PURPLE"]); command.grid(row=0, column=0, sticky="nsew", padx=(0,8), pady=(0,8)); self.command_core(command)
        workspace = Panel(main, "PROJECT WORKSPACE", "02", T["PURPLE"]); workspace.grid(row=0, column=1, sticky="nsew", pady=(0,8)); self.workspace(workspace)
        tools = Panel(main, "ORBITAL TOOLS & MODULES", "03", T["CYAN"]); tools.grid(row=1, column=0, sticky="nsew", padx=(0,8), pady=(0,8)); self.tools(tools)
        mascots = Panel(main, "MASCOTS / PERSONALITY", "04", T["PURPLE"]); mascots.grid(row=1, column=1, sticky="nsew", pady=(0,8)); self.mascots(mascots)
        inp = Panel(main, "COMMAND INPUT", "05", T["GREEN"]); inp.grid(row=2, column=0, columnspan=2, sticky="nsew"); self.command_input(inp)

    def sidebar(self, parent):
        def heading(t, color):
            Label(parent, text=t, fg=color, bg=T["PANEL"], font=("JetBrains Mono", 8, "bold"), anchor="w").pack(fill=X, padx=12, pady=(12,4))
        def item(t, cmd, color):
            Button(parent, text=t, anchor="w", bg=T["CARD"], fg=color, activebackground=T["PANEL_2"],
                   activeforeground=T["TEXT"], relief="flat", font=("JetBrains Mono", 9),
                   command=lambda: self.run(cmd)).pack(fill=X, padx=10, pady=3)
        heading("CORE", T["CYAN"]); item("Command Core", "runtime", T["CYAN"])
        heading("PROJECTS", T["PURPLE"]); item("Project Map", "mapa crotolamo", T["PURPLE"]); item("Inspector", "inspeccionar crotolamo", T["PURPLE"])
        heading("TOOLS", T["CYAN"])
        for name, cmd in [("Task Planner","plan limpiar estructura de crotolamo"),("Patch Builder","patch limpiar estructura de crotolamo"),("Test Runner","test crotolamo"),("Safe Executor","executor")]:
            item(name, cmd, T["CYAN"])
        heading("SYSTEM", T["GREEN"])
        for name, cmd in [("Memory & Context","memoria"),("Brain Engine","inteligencia"),("Voice","voz"),("Meme Reactor","contexto")]:
            item(name, cmd, T["GREEN"])
        heading("ASSETS", T["PURPLE"]); item("Mascots", "contexto", T["PURPLE"])
        box = Frame(parent, bg=T["CARD"], highlightbackground=T["BORDER"], highlightthickness=1); box.pack(fill=X, padx=10, pady=16)
        Label(box, text="ORBITAL NODE", fg=T["MUTED"], bg=T["CARD"], font=("JetBrains Mono", 7)).pack(anchor="w", padx=8, pady=(8,0))
        Label(box, text="CROTOLAMO-67", fg=T["CYAN"], bg=T["CARD"], font=("JetBrains Mono", 10, "bold")).pack(anchor="w", padx=8)
        Label(box, text="ONLINE ●", fg=T["GREEN"], bg=T["CARD"], font=("JetBrains Mono", 8, "bold")).pack(anchor="w", padx=8, pady=(0,8))

    def command_core(self, parent):
        inner = Frame(parent, bg=T["PANEL"]); inner.pack(fill=BOTH, expand=True, padx=8, pady=8)
        inner.grid_columnconfigure(0, weight=1); inner.grid_columnconfigure(1, weight=2); inner.grid_rowconfigure(0, weight=1)
        self.terminal = Text(inner, bg="#020710", fg=T["GREEN"], insertbackground=T["CYAN"], relief="flat", font=("JetBrains Mono", 9), wrap="word")
        self.terminal.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        self.terminal.insert(END, "crotolamo@orbital:~$ orbital run --mode=chaos\n\n")
        for line in ["> initializing orbital grid... ok", "> loading cores online [cores: 67]... ok", "> sandbox: MODO_MUTÍSIMO", "> system: latino_labs handmade", "> STATUS: READY TO CROTALIZE", "", "\"Si no hay caos, no hay historia.\"", ""]:
            self.terminal.insert(END, line + "\n")
        self.terminal.configure(state=DISABLED)
        OrbitCanvas(inner).grid(row=0, column=1, sticky="nsew")
        chips = Frame(parent, bg=T["PANEL"]); chips.pack(fill=X, padx=8, pady=(0,8))
        for k, v, c in [("ORBITAL CORE","96.7%",T["GREEN"]),("SYNC LATENCY","12ms",T["CYAN"]),("ENTROPY","67%",T["PURPLE"]),("REALITY ROOT","STABLE",T["GREEN"])]:
            box = Frame(chips, bg=T["CARD"], highlightbackground=T["BORDER"], highlightthickness=1); box.pack(side=LEFT, fill=X, expand=True, padx=4)
            Label(box, text=k, fg=T["MUTED"], bg=T["CARD"], font=("JetBrains Mono", 7)).pack()
            Label(box, text=v, fg=c, bg=T["CARD"], font=("JetBrains Mono", 11, "bold")).pack(pady=(0,4))

    def workspace(self, parent):
        tabs = Frame(parent, bg=T["PANEL"]); tabs.pack(fill=X, padx=8, pady=6)
        for label, cmd in [("PROJECT MAP","mapa crotolamo"),("INSPECTOR","inspeccionar crotolamo"),("TASK PLANNER","plan limpiar estructura de crotolamo"),("PATCH PREVIEW","parches")]:
            Button(tabs, text=label, bg=T["CARD"], fg=T["CYAN"] if label=="PROJECT MAP" else T["MUTED"], relief="flat", font=("JetBrains Mono", 8), command=lambda c=cmd: self.run(c)).pack(side=LEFT, padx=3)
        content = Frame(parent, bg=T["PANEL"]); content.pack(fill=BOTH, expand=True, padx=8, pady=(0,8))
        content.grid_columnconfigure(0, weight=1); content.grid_columnconfigure(1, weight=1); content.grid_rowconfigure(0, weight=1)
        tree = Text(content, bg="#020710", fg=T["TEXT"], relief="flat", font=("JetBrains Mono", 8)); tree.grid(row=0, column=0, sticky="nsew", padx=(0,6))
        tree.insert(END, "/mnt/orbital/mafia\n├── core_system\n│   ├── main_core.py\n│   └── orbital_link.py\n├── modules\n│   ├── brain_engine.py\n│   ├── meme_reactor.py\n│   └── voice_core.py\n└── tests\n    ├── test_latency.py\n    └── test_cores.py")
        tree.configure(state=DISABLED)
        info = Frame(content, bg=T["CARD"], highlightbackground=T["BORDER"], highlightthickness=1); info.grid(row=0, column=1, sticky="nsew")
        for a,b in [("INSPECTOR","main_core.py"),("Type","Python File"),("Size","12.8 KB"),("Status","Tracked"),("Lines","312")]:
            Label(info, text=f"{a}: {b}", fg=T["CYAN"] if a=="INSPECTOR" else T["TEXT"], bg=T["CARD"], font=("JetBrains Mono", 8), anchor="w").pack(fill=X, padx=10, pady=5)
        Button(info, text="OPEN IN EDITOR", bg="#15102c", fg=T["PURPLE"], relief="flat", font=("JetBrains Mono", 8), command=lambda: self.run("mapa crotolamo")).pack(fill=X, padx=10, pady=10)

    def tools(self, parent):
        row = Frame(parent, bg=T["PANEL"]); row.pack(fill=BOTH, expand=True, padx=8, pady=8)
        for i in range(6): row.grid_columnconfigure(i, weight=1)
        cards = [("Test Runner","test crotolamo",T["GREEN"],"97% PASSED\nWarnings: 2\nFailed: 1"),("Safe Executor","executor",T["GREEN"],"Sandbox ON\nPolicy Lock\nAuto Rollback"),("Memory & Context","memoria",T["PURPLE"],"Tokens Active\nCache Hit 94%"),("Brain Engine","inteligencia",T["GREEN"],"Depth 5/5\nSelf Review ON"),("Voice","voz",T["PURPLE"],"Push to Talk\nLatency 12ms"),("Meme Reactor","contexto",T["MAGENTA"],"Chaos Ultra\nMeme Index 99%")]
        for i,(title,cmd,color,body) in enumerate(cards):
            card = Frame(row, bg=T["CARD"], highlightbackground=color, highlightthickness=1); card.grid(row=0, column=i, sticky="nsew", padx=3)
            Label(card, text=title.upper(), fg=color, bg=T["CARD"], font=("JetBrains Mono", 8, "bold")).pack(anchor="w", padx=8, pady=(8,3))
            Label(card, text=body, fg=T["TEXT"], bg=T["CARD"], font=("JetBrains Mono", 7), justify="left").pack(anchor="w", padx=8, pady=6)
            Button(card, text="OPEN", bg=T["PANEL_2"], fg=color, relief="flat", font=("JetBrains Mono", 8), command=lambda c=cmd: self.run(c)).pack(side="bottom", fill=X, padx=8, pady=8)

    def mascots(self, parent):
        row = Frame(parent, bg=T["PANEL"]); row.pack(fill=BOTH, expand=True, padx=8, pady=8)
        row.grid_columnconfigure(0, weight=1); row.grid_columnconfigure(1, weight=1)
        for col,(name,desc,emoji,color) in enumerate([("PEZLÍN 3000","STATUS: ONLINE\nMODE: READY","◉_◉",T["PURPLE"]),("PALOMA SUPREMA 67","STATUS: MAXIMUM\nMODE: DOMINANDO","🕊 67",T["MAGENTA"])]):
            box = Frame(row, bg=T["CARD"], highlightbackground=color, highlightthickness=1); box.grid(row=0, column=col, sticky="nsew", padx=4)
            Label(box, text=name, fg=color, bg=T["CARD"], font=("JetBrains Mono", 10, "bold")).pack(pady=(12,6))
            Label(box, text=emoji, fg=T["YELLOW"] if col else T["TEXT"], bg=T["CARD"], font=("Arial", 42, "bold")).pack(expand=True)
            Label(box, text=desc, fg=T["GREEN"] if col==0 else T["RED"], bg=T["CARD"], font=("JetBrains Mono", 8)).pack(pady=8)

    def command_input(self, parent):
        row = Frame(parent, bg=T["PANEL"]); row.pack(fill=X, padx=8, pady=8)
        self.entry = Entry(row, textvariable=self.command_var, bg="#020710", fg=T["TEXT"], insertbackground=T["CYAN"], relief="flat", font=("JetBrains Mono", 11))
        self.entry.pack(side=LEFT, fill=X, expand=True, ipady=8)
        self.entry.bind("<Return>", lambda e: self.submit())
        Button(row, text="RUN", bg="#072417", fg=T["GREEN"], relief="flat", font=("JetBrains Mono", 9, "bold"), command=self.submit).pack(side=LEFT, padx=(8,0), ipadx=22, ipady=6)
        quick = Frame(parent, bg=T["PANEL"]); quick.pack(fill=X, padx=8, pady=(0,8))
        for cmd in ["contexto","proyectos","test crotolamo","executor","parches"]:
            Button(quick, text=cmd, bg=T["CARD"], fg=T["MUTED"], relief="flat", font=("JetBrains Mono", 7), command=lambda c=cmd: self.run(c)).pack(side=LEFT, padx=3)

    def footer(self):
        foot = Frame(self.root, bg=T["PANEL"], height=60, highlightbackground=T["BORDER"], highlightthickness=1)
        foot.pack(fill=X, padx=8, pady=(0,8)); foot.pack_propagate(False)
        StickerStrip(foot).pack(side=LEFT, fill=BOTH, expand=True, padx=(0,8))
        status = Frame(foot, bg=T["CARD"], highlightbackground=T["BORDER"], highlightthickness=1, width=350); status.pack(side=RIGHT, fill=Y); status.pack_propagate(False)
        Label(status, text="CONNECTION: orbital_link_secure", fg=T["CYAN"], bg=T["CARD"], font=("JetBrains Mono", 8)).pack(anchor="w", padx=10, pady=(8,0))
        Label(status, textvariable=self.status_var, fg=T["GREEN"], bg=T["CARD"], font=("JetBrains Mono", 10, "bold")).pack(anchor="w", padx=10)

    def submit(self):
        cmd = self.command_var.get().strip()
        if cmd:
            self.command_var.set("")
            self.run(cmd)

    def run(self, cmd):
        self.write_terminal(f"\ncrotolamo@orbital:~$ {cmd}\n")
        self.status_var.set("STATUS: RUNNING")
        threading.Thread(target=self.worker, args=(cmd,), daemon=True).start()

    def worker(self, cmd):
        if isinstance(self.runtime, Exception):
            self.queue.put(("error", f"Runtime no cargó: {self.runtime}"))
            return
        try:
            out = self.runtime.process_text(cmd)
            self.queue.put(("ok", render_result(out)))
        except Exception as e:
            self.queue.put(("error", f"{type(e).__name__}: {e}"))

    def write_terminal(self, text):
        self.terminal.configure(state=NORMAL)
        self.terminal.insert(END, text[:8000])
        self.terminal.see(END)
        self.terminal.configure(state=DISABLED)

    def poll(self):
        try:
            while True:
                kind, msg = self.queue.get_nowait()
                if kind == "ok":
                    self.write_terminal(str(msg)[:8000] + "\n")
                    self.status_var.set("STATUS: ONLINE")
                else:
                    self.write_terminal("[ERROR] " + str(msg) + "\n")
                    self.status_var.set("STATUS: ERROR")
        except queue.Empty:
            pass
        self.root.after(120, self.poll)

    def update_metrics(self):
        if psutil:
            try:
                self.cpu_pulse.value = int(psutil.cpu_percent(interval=None))
                self.ram_pulse.value = int(psutil.virtual_memory().percent)
            except Exception:
                pass
        self.root.after(1500, self.update_metrics)

def main():
    app = Tk()
    try:
        ttk.Style(app).theme_use("clam")
    except Exception:
        pass
    OrbitalUIProV17(app)
    app.mainloop()

if __name__ == "__main__":
    main()
