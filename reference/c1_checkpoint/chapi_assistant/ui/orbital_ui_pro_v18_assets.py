from __future__ import annotations
import sys, threading, queue, math, random
from pathlib import Path
from tkinter import Tk, Canvas, Frame, Label, Button, Entry, Text, StringVar, BOTH, LEFT, RIGHT, X, Y, END, NORMAL, DISABLED
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
    if result is None: return ""
    if isinstance(result, str): return result
    if isinstance(result, dict):
        for k in ("text","response","message","output","result"):
            if k in result: return str(result[k])
        return str(result)
    return str(result)

class Panel(Frame):
    def __init__(self, master, title="", number="", accent=None, **kwargs):
        accent = accent or T["CYAN"]
        super().__init__(master, bg=T["PANEL"], highlightbackground=accent, highlightthickness=1, **kwargs)
        if title:
            head = Frame(self, bg=T["PANEL"])
            head.pack(fill=X, padx=10, pady=(7,2))
            Label(head, text=(f"{number}. " if number else "") + title.upper(), bg=T["PANEL"], fg=accent, font=("JetBrains Mono",9,"bold")).pack(side=LEFT)
            Label(head, text="● ○", bg=T["PANEL"], fg=T["MUTED"], font=("JetBrains Mono",8)).pack(side=RIGHT)

class OrbitCanvas(Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, bg="#020711", highlightthickness=0, **kwargs)
        self.bg_img = None
        self.t = 0
        self.after(33, self.tick)
    def tick(self):
        self.t += .035
        self.draw()
        self.after(33, self.tick)
    def draw(self):
        self.delete("all")
        w,h=max(self.winfo_width(),10),max(self.winfo_height(),10)
        img=get_asset("galaxy_bg.png")
        if img:
            self.create_image(w//2,h//2,image=img)
        cx,cy=w*.55,h*.50
        for i in range(12):
            s=i/11
            self.create_rectangle(cx-(w*.04+s*w*.42),cy-(h*.04+s*h*.42),cx+(w*.04+s*w*.42),cy+(h*.04+s*h*.42),outline="#12366d")
        for i in range(-8,9):
            x=cx+i*w*.045
            self.create_line(cx,cy,x,h,fill="#112c5b")
            self.create_line(cx,cy,x,0,fill="#112c5b")
        size=96+math.sin(self.t*2)*5
        off=size*.34
        front=[(cx-size/2,cy-size/2),(cx+size/2,cy-size/2),(cx+size/2,cy+size/2),(cx-size/2,cy+size/2)]
        back=[(x+off,y-off) for x,y in front]
        for poly,color in [(front,T["CYAN"]),(back,T["BLUE"])]:
            self.create_polygon(*sum(poly,()),outline=color,fill="",width=2)
        for a,b in zip(front,back):
            self.create_line(*a,*b,fill=T["PURPLE"],width=2)
        for i,r in enumerate([78,118,160,205]):
            self.create_oval(cx-r*1.45,cy-r*.45,cx+r*1.45,cy+r*.45,outline=[T["CYAN"],T["PURPLE"],T["BLUE"],T["MAGENTA"]][i],width=1)

class AssetImage(Canvas):
    def __init__(self, master, asset_name, **kwargs):
        self.asset_name=asset_name
        super().__init__(master,bg=T["CARD"],highlightthickness=0,**kwargs)
        self.bind("<Configure>", lambda e: self.draw())
    def draw(self):
        self.delete("all")
        img=get_asset(self.asset_name)
        w,h=max(self.winfo_width(),10),max(self.winfo_height(),10)
        if img:
            self.create_image(w//2,h//2,image=img)
        else:
            self.create_text(w//2,h//2,text=self.asset_name,fill=T["MUTED"],font=("JetBrains Mono",9))

class OrbitalUIProV18:
    def __init__(self, root):
        self.root=root
        self.root.title("CROTOLAMO ORBITAL UI PRO v18 Assets")
        self.root.geometry("1540x880")
        self.root.minsize(1250,760)
        self.root.configure(bg=T["BG"])
        self.runtime=safe_runtime()
        self.queue=queue.Queue()
        self.command_var=StringVar()
        self.status_var=StringVar(value="STATUS: ONLINE")
        self.build()
        self.update_metrics()
        self.poll()
    def build(self):
        self.topbar(); self.content(); self.footer()
    def topbar(self):
        top=Frame(self.root,bg=T["PANEL"],height=62,highlightbackground=T["PURPLE"],highlightthickness=1)
        top.pack(fill=X,padx=8,pady=8); top.pack_propagate(False)
        logo=Frame(top,bg=T["PANEL"]); logo.pack(side=LEFT,padx=14)
        Label(logo,text="CROTOLAMO",fg=T["TEXT"],bg=T["PANEL"],font=("JetBrains Mono",21,"bold")).pack(anchor="w")
        Label(logo,text="ORBITAL UI PRO v18",fg=T["MAGENTA"],bg=T["PANEL"],font=("JetBrains Mono",9,"bold")).pack(anchor="w")
        status=Frame(top,bg=T["CARD"],highlightbackground=T["BORDER"],highlightthickness=1); status.pack(side=LEFT,padx=20,pady=8)
        Label(status,text="SYSTEM STATUS",fg=T["MUTED"],bg=T["CARD"],font=("JetBrains Mono",7,"bold")).pack(anchor="w",padx=10,pady=(4,0))
        Label(status,text="● ALL SYSTEMS OPERATIONAL",fg=T["GREEN"],bg=T["CARD"],font=("JetBrains Mono",9,"bold")).pack(anchor="w",padx=10,pady=(0,4))
        metrics=Frame(top,bg=T["PANEL"]); metrics.pack(side=LEFT,fill=Y,expand=True)
        self.cpu=SystemPulse(metrics,"CPU",67); self.cpu.pack(side=LEFT,padx=4,pady=8)
        self.ram=SystemPulse(metrics,"RAM",72); self.ram.pack(side=LEFT,padx=4,pady=8)
        self.gpu=SystemPulse(metrics,"GPU",81); self.gpu.pack(side=LEFT,padx=4,pady=8)
        right=Frame(top,bg=T["PANEL"]); right.pack(side=RIGHT,padx=10)
        RadarBadge(right).pack(side=LEFT,padx=8)
        Label(right,text="CROTOLAMO-67",fg=T["TEXT"],bg=T["PANEL"],font=("JetBrains Mono",10,"bold")).pack(anchor="e")
        Label(right,text="Orbital Operator",fg=T["MUTED"],bg=T["PANEL"],font=("JetBrains Mono",8)).pack(anchor="e")
    def content(self):
        body=Frame(self.root,bg=T["BG"]); body.pack(fill=BOTH,expand=True,padx=8,pady=(0,6))
        side=Panel(body,accent=T["BORDER"],width=205); side.pack(side=LEFT,fill=Y,padx=(0,8)); side.pack_propagate(False); self.sidebar(side)
        main=Frame(body,bg=T["BG"]); main.pack(side=LEFT,fill=BOTH,expand=True)
        main.grid_columnconfigure(0,weight=3); main.grid_columnconfigure(1,weight=2)
        main.grid_rowconfigure(0,weight=5); main.grid_rowconfigure(1,weight=2); main.grid_rowconfigure(2,weight=1)
        command=Panel(main,"COMMAND CORE / RUNTIME CONSOLE","01",T["PURPLE"]); command.grid(row=0,column=0,sticky="nsew",padx=(0,8),pady=(0,8)); self.command_core(command)
        work=Panel(main,"PROJECT WORKSPACE","02",T["PURPLE"]); work.grid(row=0,column=1,sticky="nsew",pady=(0,8)); self.workspace(work)
        tools=Panel(main,"ORBITAL TOOLS & MODULES","03",T["CYAN"]); tools.grid(row=1,column=0,sticky="nsew",padx=(0,8),pady=(0,8)); self.tools(tools)
        masc=Panel(main,"MASCOTS / PERSONALITY","04",T["PURPLE"]); masc.grid(row=1,column=1,sticky="nsew",pady=(0,8)); self.mascots(masc)
        inp=Panel(main,"COMMAND INPUT","05",T["GREEN"]); inp.grid(row=2,column=0,columnspan=2,sticky="nsew"); self.command_input(inp)
    def sidebar(self,parent):
        def heading(t,c): Label(parent,text=t,fg=c,bg=T["PANEL"],font=("JetBrains Mono",8,"bold"),anchor="w").pack(fill=X,padx=12,pady=(12,4))
        def item(t,cmd,c): Button(parent,text=t,anchor="w",bg=T["CARD"],fg=c,activebackground=T["PANEL_2"],activeforeground=T["TEXT"],relief="flat",font=("JetBrains Mono",9),command=lambda:self.run(cmd)).pack(fill=X,padx=10,pady=3)
        heading("CORE",T["CYAN"]); item("Command Core","runtime",T["CYAN"])
        heading("PROJECTS",T["PURPLE"]); item("Project Map","mapa crotolamo",T["PURPLE"]); item("Inspector","inspeccionar crotolamo",T["PURPLE"])
        heading("TOOLS",T["CYAN"])
        for name,cmd in [("Task Planner","plan limpiar estructura de crotolamo"),("Patch Builder","patch limpiar estructura de crotolamo"),("Test Runner","test crotolamo"),("Safe Executor","executor")]: item(name,cmd,T["CYAN"])
        heading("SYSTEM",T["GREEN"])
        for name,cmd in [("Memory & Context","memoria"),("Brain Engine","inteligencia"),("Voice","voz"),("Meme Reactor","contexto")]: item(name,cmd,T["GREEN"])
        heading("ASSETS",T["PURPLE"]); item("Mascots","contexto",T["PURPLE"])
        box=Frame(parent,bg=T["CARD"],highlightbackground=T["BORDER"],highlightthickness=1); box.pack(fill=X,padx=10,pady=16)
        Label(box,text="ORBITAL NODE",fg=T["MUTED"],bg=T["CARD"],font=("JetBrains Mono",7)).pack(anchor="w",padx=8,pady=(8,0))
        Label(box,text="CROTOLAMO-67",fg=T["CYAN"],bg=T["CARD"],font=("JetBrains Mono",10,"bold")).pack(anchor="w",padx=8)
        Label(box,text="ONLINE ●",fg=T["GREEN"],bg=T["CARD"],font=("JetBrains Mono",8,"bold")).pack(anchor="w",padx=8,pady=(0,8))
    def command_core(self,parent):
        inner=Frame(parent,bg=T["PANEL"]); inner.pack(fill=BOTH,expand=True,padx=8,pady=8)
        inner.grid_columnconfigure(0,weight=1); inner.grid_columnconfigure(1,weight=2); inner.grid_rowconfigure(0,weight=1)
        self.terminal=Text(inner,bg="#020710",fg=T["GREEN"],insertbackground=T["CYAN"],relief="flat",font=("JetBrains Mono",9),wrap="word")
        self.terminal.grid(row=0,column=0,sticky="nsew",padx=(0,8))
        self.terminal.insert(END,"crotolamo@orbital:~$ orbital run --mode=chaos\n\n")
        for line in ["> loading orbital assets... ok","> pezlin_3000.png linked","> paloma_suprema_67.png armed","> hacker_mode.png active","> STATUS: READY TO CROTALIZE","","\"Si no hay caos, no hay historia.\"",""]:
            self.terminal.insert(END,line+"\n")
        self.terminal.configure(state=DISABLED)
        OrbitCanvas(inner).grid(row=0,column=1,sticky="nsew")
    def workspace(self,parent):
        tabs=Frame(parent,bg=T["PANEL"]); tabs.pack(fill=X,padx=8,pady=6)
        for label,cmd in [("PROJECT MAP","mapa crotolamo"),("INSPECTOR","inspeccionar crotolamo"),("TASK PLANNER","plan limpiar estructura de crotolamo"),("PATCH PREVIEW","parches")]:
            Button(tabs,text=label,bg=T["CARD"],fg=T["CYAN"] if label=="PROJECT MAP" else T["MUTED"],relief="flat",font=("JetBrains Mono",8),command=lambda c=cmd:self.run(c)).pack(side=LEFT,padx=3)
        txt=Text(parent,bg="#020710",fg=T["TEXT"],relief="flat",font=("JetBrains Mono",8))
        txt.pack(fill=BOTH,expand=True,padx=8,pady=(0,8))
        txt.insert(END,"/assets/orbital_ui\n├── galaxy_bg.png\n├── pezlin_3000.png\n├── paloma_suprema_67.png\n├── hacker_mode.png\n└── sticker_strip.png\n\nQuick Actions:\n- mapa crotolamo\n- inspeccionar crotolamo\n- test crotolamo")
        txt.configure(state=DISABLED)
    def tools(self,parent):
        row=Frame(parent,bg=T["PANEL"]); row.pack(fill=BOTH,expand=True,padx=8,pady=8)
        for i in range(6): row.grid_columnconfigure(i,weight=1)
        cards=[("Test Runner","test crotolamo",T["GREEN"],"97% PASSED"),("Safe Executor","executor",T["GREEN"],"Sandbox ON"),("Memory","memoria",T["PURPLE"],"Context active"),("Brain","inteligencia",T["GREEN"],"Depth 5/5"),("Voice","voz",T["PURPLE"],"Push to talk"),("Meme Reactor","contexto",T["MAGENTA"],"Assets linked")]
        for i,(title,cmd,color,body) in enumerate(cards):
            card=Frame(row,bg=T["CARD"],highlightbackground=color,highlightthickness=1); card.grid(row=0,column=i,sticky="nsew",padx=3)
            Label(card,text=title.upper(),fg=color,bg=T["CARD"],font=("JetBrains Mono",8,"bold")).pack(anchor="w",padx=8,pady=(8,3))
            Label(card,text=body,fg=T["TEXT"],bg=T["CARD"],font=("JetBrains Mono",8)).pack(anchor="w",padx=8,pady=6)
            Button(card,text="OPEN",bg=T["PANEL_2"],fg=color,relief="flat",font=("JetBrains Mono",8),command=lambda c=cmd:self.run(c)).pack(side="bottom",fill=X,padx=8,pady=8)
    def mascots(self,parent):
        row=Frame(parent,bg=T["PANEL"]); row.pack(fill=BOTH,expand=True,padx=8,pady=8)
        row.grid_columnconfigure(0,weight=1); row.grid_columnconfigure(1,weight=1)
        for col,(name,asset,status,color) in enumerate([("PEZLÍN 3000","pezlin_3000.png","STATUS: ONLINE",T["PURPLE"]),("PALOMA SUPREMA 67","paloma_suprema_67.png","LASER: ARMED",T["MAGENTA"])]):
            box=Frame(row,bg=T["CARD"],highlightbackground=color,highlightthickness=1); box.grid(row=0,column=col,sticky="nsew",padx=4)
            Label(box,text=name,fg=color,bg=T["CARD"],font=("JetBrains Mono",10,"bold")).pack(pady=(8,4))
            AssetImage(box,asset,height=165).pack(fill=BOTH,expand=True,padx=8)
            Label(box,text=status,fg=T["GREEN"] if col==0 else T["RED"],bg=T["CARD"],font=("JetBrains Mono",8,"bold")).pack(pady=6)
    def command_input(self,parent):
        row=Frame(parent,bg=T["PANEL"]); row.pack(fill=X,padx=8,pady=8)
        self.entry=Entry(row,textvariable=self.command_var,bg="#020710",fg=T["TEXT"],insertbackground=T["CYAN"],relief="flat",font=("JetBrains Mono",11))
        self.entry.pack(side=LEFT,fill=X,expand=True,ipady=8); self.entry.bind("<Return>",lambda e:self.submit())
        Button(row,text="RUN",bg="#072417",fg=T["GREEN"],relief="flat",font=("JetBrains Mono",9,"bold"),command=self.submit).pack(side=LEFT,padx=(8,0),ipadx=22,ipady=6)
    def footer(self):
        foot=Frame(self.root,bg=T["PANEL"],height=72,highlightbackground=T["BORDER"],highlightthickness=1); foot.pack(fill=X,padx=8,pady=(0,8)); foot.pack_propagate(False)
        img=get_asset("sticker_strip.png")
        if img:
            lab=Label(foot,image=img,bg=T["PANEL"]); lab.image=img; lab.pack(side=LEFT,fill=Y,padx=4)
        else:
            StickerStrip(foot).pack(side=LEFT,fill=BOTH,expand=True,padx=(0,8))
        status=Frame(foot,bg=T["CARD"],highlightbackground=T["BORDER"],highlightthickness=1,width=380); status.pack(side=RIGHT,fill=Y); status.pack_propagate(False)
        Label(status,text="ASSET PACK: v18 loaded",fg=T["CYAN"],bg=T["CARD"],font=("JetBrains Mono",8)).pack(anchor="w",padx=10,pady=(10,0))
        Label(status,textvariable=self.status_var,fg=T["GREEN"],bg=T["CARD"],font=("JetBrains Mono",10,"bold")).pack(anchor="w",padx=10)
    def submit(self):
        cmd=self.command_var.get().strip()
        if cmd:
            self.command_var.set("")
            self.run(cmd)
    def run(self,cmd):
        self.write_terminal(f"\ncrotolamo@orbital:~$ {cmd}\n")
        self.status_var.set("STATUS: RUNNING")
        threading.Thread(target=self.worker,args=(cmd,),daemon=True).start()
    def worker(self,cmd):
        if isinstance(self.runtime,Exception):
            self.queue.put(("error",f"Runtime no cargó: {self.runtime}")); return
        try:
            self.queue.put(("ok",render_result(self.runtime.process_text(cmd))))
        except Exception as e:
            self.queue.put(("error",f"{type(e).__name__}: {e}"))
    def write_terminal(self,text):
        self.terminal.configure(state=NORMAL); self.terminal.insert(END,text[:8000]); self.terminal.see(END); self.terminal.configure(state=DISABLED)
    def poll(self):
        try:
            while True:
                kind,msg=self.queue.get_nowait()
                self.write_terminal(("[ERROR] " if kind!="ok" else "")+str(msg)[:8000]+"\n")
                self.status_var.set("STATUS: ONLINE" if kind=="ok" else "STATUS: ERROR")
        except queue.Empty:
            pass
        self.root.after(120,self.poll)
    def update_metrics(self):
        if psutil:
            try:
                self.cpu.value=int(psutil.cpu_percent(interval=None)); self.ram.value=int(psutil.virtual_memory().percent)
            except Exception: pass
        self.root.after(1500,self.update_metrics)

def main():
    app=Tk()
    try: ttk.Style(app).theme_use("clam")
    except Exception: pass
    OrbitalUIProV18(app)
    app.mainloop()

if __name__=="__main__":
    main()
