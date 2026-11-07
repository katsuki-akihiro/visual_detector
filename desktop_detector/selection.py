"""Full-screen selector for a fixed primary-screen region."""
from __future__ import annotations

import tkinter as tk
from typing import Callable

from PIL import Image, ImageEnhance, ImageTk

from .core import Region
from .theme import BOX, CANVAS


class RegionSelector:
    """Select from a frozen screenshot; Esc cancels."""

    def __init__(self, root: tk.Tk, snapshot: Image.Image, size: tuple[int, int],
                 done: Callable[[Region | None], None]) -> None:
        self.done = done
        self.size = size
        self.origin: tuple[int, int] | None = None
        self.top = tk.Toplevel(root)
        self.top.overrideredirect(True)
        self.top.attributes("-topmost", True)
        self.top.geometry(f"{size[0]}x{size[1]}+0+0")
        self.canvas = tk.Canvas(self.top, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        display = snapshot.resize(size, Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(ImageEnhance.Brightness(display).enhance(0.62), master=self.top)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.create_rectangle(0, 0, size[0], 54, fill=CANVAS, outline="")
        self.help_id = self.canvas.create_text(
            22, 26, anchor="w", fill="white", font=("TkDefaultFont", 12),
            text="監視する範囲をドラッグしてください。Esc：キャンセル / 最小32×32")
        self.rect_id: int | None = None
        self.text_id: int | None = None
        self.canvas.bind("<ButtonPress-1>", self._down)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", self._up)
        self.top.bind("<Escape>", lambda event: self.finish(None))
        self.top.protocol("WM_DELETE_WINDOW", lambda: self.finish(None))
        self.top.after(60, self._focus)

    def _focus(self) -> None:
        if self.top.winfo_exists():
            self.top.focus_force()
            self.top.grab_set()

    def _point(self, event: tk.Event) -> tuple[int, int]:
        return max(0, min(self.size[0], int(event.x))), max(0, min(self.size[1], int(event.y)))

    def _down(self, event: tk.Event) -> None:
        self.origin = self._point(event)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        if self.text_id:
            self.canvas.delete(self.text_id)
        x, y = self.origin
        self.rect_id = self.canvas.create_rectangle(x, y, x, y, outline=BOX, width=3)
        self.text_id = self.canvas.create_text(x + 8, y + 8, anchor="nw", fill="white", font=("TkDefaultFont", 11))

    def _move(self, event: tk.Event) -> None:
        if self.origin is None:
            return
        x, y = self._point(event)
        ox, oy = self.origin
        self.canvas.coords(self.rect_id, ox, oy, x, y)
        self.canvas.itemconfigure(self.text_id, text=f"{abs(x-ox)} × {abs(y-oy)}")

    def _up(self, event: tk.Event) -> None:
        if self.origin is None:
            return
        x, y = self._point(event)
        ox, oy = self.origin
        width, height = abs(x - ox), abs(y - oy)
        if width < 32 or height < 32:
            self.canvas.itemconfigure(self.help_id, text="32×32以上の領域を、もう一度ドラッグしてください。Esc：キャンセル")
            self.origin = None
            return
        self.finish(Region(min(x, ox), min(y, oy), width, height))

    def finish(self, region: Region | None) -> None:
        self.top.grab_release()
        self.top.destroy()
        self.done(region)
