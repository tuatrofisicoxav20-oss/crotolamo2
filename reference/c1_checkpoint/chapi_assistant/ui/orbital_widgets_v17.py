from __future__ import annotations
import math, random
from tkinter import Canvas
from ui.orbital_theme import THEME as T

class SystemPulse(Canvas):
    def __init__(self, master, label="CPU", value=67, **kwargs):
        self.label = label
        self.value = value
        self.t = 0
        super().__init__(master, bg=T["CARD"], highlightthickness=0, width=120, height=42, **kwargs)
        self.after(90, self.animate)

    def animate(self):
        self.t += .17
        self.draw()
        self.after(90, self.animate)

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        self.create_rectangle(0, 0, w-1, h-1, outline=T["BORDER"], fill=T["CARD"])
        self.create_text(8, 10, text=self.label, fill=T["MUTED"], anchor="w", font=("JetBrains Mono", 7, "bold"))
        self.create_text(8, 28, text=f"{self.value}%", fill=T["CYAN"], anchor="w", font=("JetBrains Mono", 11, "bold"))
        for i in range(18):
            x = 58 + i * 3
            amp = 6 + 12 * abs(math.sin(self.t + i * .7))
            self.create_line(x, h-8-amp, x, h-8, fill=T["GREEN"] if i % 3 else T["CYAN"])

class StickerStrip(Canvas):
    def __init__(self, master, **kwargs):
        self.t = 0
        super().__init__(master, bg=T["PANEL"], highlightthickness=0, height=58, **kwargs)
        self.after(100, self.animate)

    def animate(self):
        self.t += .06
        self.draw()
        self.after(100, self.animate)

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        self.create_rectangle(0, 0, w, h, fill=T["CARD"], outline=T["BORDER"])
        labels = [("LATIN\nMAFIA", T["YELLOW"], 17), ("CAOS\nMODE", T["GREEN"], 10), ("67", T["YELLOW"], 22),
                  ("LOCO PERO\nSERIO", T["MAGENTA"], 9), ("HECHO EN\nHUEVONITIS", T["TEXT"], 8), ("I ♥\nCAOS", T["CYAN"], 12)]
        x = 12
        for text, color, size in labels:
            y = 8 + math.sin(self.t + x*.01)*2
            self.create_rectangle(x, y, x+110, h-8+y-8, outline=color, fill="#0b0a13")
            self.create_text(x+55, h/2, text=text, fill=color, font=("JetBrains Mono", size, "bold"))
            x += 120
            if x > w - 100:
                break
        random.seed(67)
        for _ in range(14):
            x = random.randint(0, max(w, 1))
            y = random.randint(8, max(h-8, 9))
            c = random.choice([T["CYAN"], T["PURPLE"], T["MAGENTA"], T["YELLOW"]])
            self.create_oval(x, y, x+6, y+6, outline=c)

class RadarBadge(Canvas):
    def __init__(self, master, **kwargs):
        self.t = 0
        super().__init__(master, bg=T["PANEL"], highlightthickness=0, width=64, height=64, **kwargs)
        self.after(60, self.animate)

    def animate(self):
        self.t += .08
        self.draw()
        self.after(60, self.animate)

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        cx, cy = w/2, h/2
        for r in (10, 20, 29):
            self.create_oval(cx-r, cy-r, cx+r, cy+r, outline=T["BORDER"])
        self.create_line(cx, cy, cx+math.cos(self.t)*29, cy+math.sin(self.t)*29, fill=T["CYAN"], width=2)
        self.create_text(cx, h-8, text="67", fill=T["YELLOW"], font=("JetBrains Mono", 8, "bold"))
