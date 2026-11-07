"""Canvas rendering for the three inference modes."""
from __future__ import annotations

from datetime import datetime
import tkinter as tk

from PIL import Image, ImageTk

from .core import MODE_BINARY, MODE_CHOICE, Detection, fit_image
from .theme import BOX, CANVAS, INK
from .worker import FrameResult


class PreviewCanvas(tk.Canvas):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, background=CANVAS, highlightthickness=0, height=40)
        self.mode = ""
        self.result: FrameResult | None = None
        self.history: tuple[FrameResult, ...] = ()
        self.blocked = False
        self.photos: list[ImageTk.PhotoImage] = []
        self.bind("<Configure>", self._render)

    def show(self, mode: str, result: FrameResult | None,
             history: tuple[FrameResult, ...], blocked: bool) -> None:
        self.mode = mode
        self.result = result
        self.history = history
        self.blocked = blocked
        self._render()

    def _render(self, event=None) -> None:
        self.delete("all")
        self.photos.clear()
        width, height = max(1, self.winfo_width()), max(1, self.winfo_height())
        if self.mode == MODE_BINARY:
            self._draw_history(width, height)
        elif self.result is None:
            message = "小窓を監視領域の外へ移動" if self.blocked else "領域選択 → 開始"
            self.create_text(width / 2, height / 2, text=message, fill="#B3C2DA",
                             justify="center", font=("TkDefaultFont", 11))
        else:
            self._draw_result(width, height)

    def _photo(self, image: Image.Image, width: int, height: int) -> ImageTk.PhotoImage:
        photo = ImageTk.PhotoImage(image.resize((width, height), Image.Resampling.BILINEAR), master=self)
        self.photos.append(photo)
        return photo

    def _draw_history(self, width: int, height: int) -> None:
        if not self.history:
            self.create_text(width / 2, height / 2, text="陽性の画像はまだありません", fill="#B3C2DA")
            return
        card_width = width / 3
        for index, item in enumerate(self.history):
            left = round(index * card_width)
            area_width = max(1, round(card_width) - 12)
            area_height = max(1, height - 47)
            image = item.task.image
            w, h, ox, oy = fit_image(image.width, image.height, area_width, area_height)
            x, y = left + 6 + ox, 6 + oy
            self.create_image(x, y, anchor="nw", image=self._photo(image, w, h))
            for detection in item.detections:
                self._box(detection, x, y, w, h)
            stamp = datetime.fromtimestamp(item.task.captured_wall).strftime("%H:%M:%S.%f")[:-3]
            self.create_text(left + 6, height - 35, anchor="nw", fill="white",
                             text=f"検出 {item.score:.2f}  {stamp}")
            if index < 2:
                edge = round((index + 1) * card_width)
                self.create_line(edge, 0, edge, height, fill="#42526C")

    def _draw_result(self, width: int, height: int) -> None:
        result = self.result
        image = result.task.image
        w, h, ox, oy = fit_image(image.width, image.height, width, height)
        self.create_image(ox, oy, anchor="nw", image=self._photo(image, w, h))
        if result.task.settings.mode == MODE_CHOICE:
            self.create_rectangle(ox, oy, ox + w, oy + 30, fill=CANVAS, outline="")
            self.create_text(ox + 8, oy + 15, anchor="w", text=f"{result.choice}  {result.score:.2f}",
                             fill="white", font=("TkDefaultFont", 11, "bold"))
            return
        for index, detection in enumerate(result.detections, start=1):
            self._box(detection, ox, oy, w, h)
            x1, y1, _, _ = detection.pixels(w, h)
            label = f"#{index}"
            if detection.confidence is not None:
                label += f"  {detection.confidence:.2f}"
            tag = self.create_text(ox + x1 + 4, max(oy, oy + y1 - 18) + 2, anchor="nw",
                                   text=label, fill=INK, font=("TkDefaultFont", 9, "bold"))
            bounds = self.bbox(tag)
            if bounds:
                background = self.create_rectangle(bounds[0] - 3, bounds[1] - 1,
                                                   bounds[2] + 3, bounds[3] + 1,
                                                   fill=BOX, outline="")
                self.tag_lower(background, tag)

    def _box(self, detection: Detection, x: int, y: int, width: int, height: int) -> None:
        x1, y1, x2, y2 = detection.pixels(width, height)
        self.create_rectangle(x + x1, y + y1, x + x2, y + y2, outline=BOX, width=2)
