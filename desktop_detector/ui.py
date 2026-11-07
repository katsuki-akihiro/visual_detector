"""A single floating preview: region selector + off-thread local inference."""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import queue
import sys
import time
import tkinter as tk
from tkinter import font as tkfont, ttk
from .backend import DEFAULT_MODEL_DIR, D1Backend
from .capture import check_capture_session, enable_dpi_awareness, grab
from .core import (DEFAULT_PROMPT, MODE_REGION, MODE_BINARY, MODE_CHOICE, MODES,
                   INTERVAL_CHOICES, InferenceSettings, Region, SingleFlight,
                   make_questions, validate_categories, validate_interval, validate_threshold)
from .preview import PreviewCanvas
from .selection import RegionSelector
from .theme import ACCENT, BG, INK, MUTED, WARNING
from .worker import FrameResult, FrameTask, InferenceWorker


def available_gpu_device() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


class DetectorApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace, backend=None) -> None:
        self.root = root
        self.args = args
        self.gate = SingleFlight()
        self.ready = False
        self.loading = False
        self.closing = False
        self.selecting = False
        self.blocked = False
        self.region: Region | None = None
        self.region_screen: tuple[int, int] | None = None
        self.selector: RegionSelector | None = None
        self.last_result: FrameResult | None = None
        self.positive_history: deque[FrameResult] = deque(maxlen=3)
        self.completed_times: deque[float] = deque(maxlen=31)
        self.settings = InferenceSettings(MODE_REGION, args.prompt, args.threshold, args.max_boxes, args.max_edge)
        self.tick_id: str | None = None
        self.gpu_device = available_gpu_device()
        self.initial_device = args.device if args.device != "auto" else (self.gpu_device or "cpu")
        self.backend = backend or D1Backend(args.model_dir, self.initial_device, args.dtype, args.compile)
        self.worker = InferenceWorker(self.backend)
        root.title("Visual Detector — Liquid AI d1-2B")
        root.configure(bg=BG)
        root.attributes("-topmost", True)
        root.minsize(400, 300)
        screen_width = root.winfo_screenwidth()
        root.geometry(args.geometry or f"400x300+{max(0, screen_width-424)}+48")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_ui()
        self._controls()
        root.bind("<Configure>", self._configured)
        root.bind("<Escape>", lambda event: self.stop() if not self.selecting else None)
        self.tick_id = root.after(30, self._tick)

    def _build_ui(self) -> None:
        families = set(tkfont.families(self.root))
        for family in ("Yu Gothic UI", "Hiragino Sans", "Noto Sans CJK JP", "Noto Sans JP"):
            if family in families:
                tkfont.nametofont("TkDefaultFont").configure(family=family, size=9)
                tkfont.nametofont("TkTextFont").configure(family=family, size=9)
                break
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=INK)
        style.configure("TButton", padding=(7, 4))
        style.configure("Start.TButton", background=ACCENT, foreground="white")
        style.map("Start.TButton", background=[("active", "#0D5D57"), ("disabled", "#CDD8D8")],
                  foreground=[("disabled", "#708080")])
        self.body = ttk.Frame(self.root, padding=(10, 7, 10, 6))
        self.body.pack(fill="both", expand=True)
        header = ttk.Frame(self.body)
        header.pack(fill="x", pady=(0, 5))
        ttk.Label(header, text="VISUAL DETECTOR", font=(tkfont.nametofont("TkDefaultFont").actual("family"), 11, "bold")).pack(side="left")
        self.device_choice = tk.StringVar(value="CPU" if self.initial_device == "cpu" else "GPU")
        self.device_combo = ttk.Combobox(header, values=("GPU", "CPU"), textvariable=self.device_choice,
                                         state="readonly", width=5)
        self.device_combo.pack(side="right")
        self.device_combo.bind("<<ComboboxSelected>>", self._device_changed)
        ttk.Label(header, text="実行", foreground=MUTED).pack(side="right", padx=(0, 4))
        mode_row = ttk.Frame(self.body)
        mode_row.pack(fill="x", pady=(0, 5))
        ttk.Label(mode_row, text="モード", width=10).pack(side="left")
        self.mode = tk.StringVar(value=MODE_REGION)
        self.mode_combo = ttk.Combobox(mode_row, values=MODES, textvariable=self.mode, state="readonly", width=14)
        self.mode_combo.pack(side="left")
        self.mode_combo.bind("<<ComboboxSelected>>", self._mode_changed)
        self.mode_options = ttk.Frame(self.body)
        self.mode_options.pack(fill="x")
        self.region_options = ttk.Frame(self.mode_options)
        prompt_row = ttk.Frame(self.region_options)
        prompt_row.pack(fill="x", pady=(0, 5))
        ttk.Label(prompt_row, text="対象指示", width=10).pack(side="left")
        self.prompt = tk.StringVar(value=self.args.prompt)
        self.prompt_entry = ttk.Entry(prompt_row, textvariable=self.prompt)
        self.prompt_entry.pack(side="left", fill="x", expand=True)
        threshold_row = ttk.Frame(self.region_options)
        threshold_row.pack(fill="x", pady=(0, 5))
        ttk.Label(threshold_row, text="信頼度下限", width=10).pack(side="left")
        self.threshold = tk.StringVar(value=f"{self.args.threshold:.2f}")
        self.threshold_spinbox = ttk.Spinbox(threshold_row, from_=0, to=1, increment=0.01,
                                             format="%.2f", width=6, textvariable=self.threshold)
        self.threshold_spinbox.pack(side="left")
        self.binary_options = ttk.Frame(self.mode_options)
        binary_row = ttk.Frame(self.binary_options)
        binary_row.pack(fill="x", pady=(0, 5))
        ttk.Label(binary_row, text="判定指示", width=10).pack(side="left")
        self.binary_prompt = tk.StringVar(value="魚が映っているか？")
        self.binary_entry = ttk.Entry(binary_row, textvariable=self.binary_prompt)
        self.binary_entry.pack(side="left", fill="x", expand=True)
        binary_threshold_row = ttk.Frame(self.binary_options)
        binary_threshold_row.pack(fill="x", pady=(0, 5))
        ttk.Label(binary_threshold_row, text="判定しきい値", width=10).pack(side="left")
        self.binary_threshold = tk.StringVar(value="0.20")
        self.binary_spinbox = ttk.Spinbox(binary_threshold_row, from_=0, to=1, increment=0.01,
                                          format="%.2f", width=6, textvariable=self.binary_threshold)
        self.binary_spinbox.pack(side="left")
        location_row = ttk.Frame(self.binary_options)
        location_row.pack(fill="x", pady=(0, 5))
        ttk.Label(location_row, text="位置の指示", width=10).pack(side="left")
        self.location_prompt = tk.StringVar(value="映っている魚をすべて囲む")
        self.location_entry = ttk.Entry(location_row, textvariable=self.location_prompt)
        self.location_entry.pack(side="left", fill="x", expand=True)
        ttk.Label(location_row, text="任意", foreground=MUTED).pack(side="left", padx=(4, 0))
        self.choice_options = ttk.Frame(self.mode_options)
        choice_row = ttk.Frame(self.choice_options)
        choice_row.pack(fill="x", pady=(0, 5))
        self.categories: tuple[str, ...] = ("魚がある",)
        self.category_summary = tk.StringVar()
        ttk.Label(choice_row, textvariable=self.category_summary).pack(side="left", fill="x", expand=True)
        self.category_button = ttk.Button(choice_row, text="候補を編集", command=self.edit_categories)
        self.category_button.pack(side="right")
        self._update_category_summary()
        self.region_options.pack(fill="x")
        toolbar = ttk.Frame(self.body)
        toolbar.pack(fill="x")
        self.region_button = ttk.Button(toolbar, text="領域選択", command=self.choose_region)
        self.region_button.pack(side="left")
        self.start_button = ttk.Button(toolbar, text="開始", width=4, style="Start.TButton", command=self.start)
        self.start_button.pack(side="left", padx=(5, 3))
        self.stop_button = ttk.Button(toolbar, text="停止", width=4, command=self.stop, state="disabled")
        self.stop_button.pack(side="left")
        ttk.Label(toolbar, text="更新間隔").pack(side="left", padx=(8, 2))
        self.interval = tk.StringVar(value=str(self.args.interval))
        self.interval_combo = ttk.Combobox(toolbar, values=INTERVAL_CHOICES, textvariable=self.interval,
                                           state="readonly", width=4)
        self.interval_combo.pack(side="left")
        ttk.Label(toolbar, text="秒").pack(side="left", padx=(3, 0))
        self.region_text = tk.StringVar(value="領域：未選択（主画面の固定座標）")
        ttk.Label(self.body, textvariable=self.region_text, foreground=MUTED).pack(fill="x", pady=(5, 4))
        self.canvas = PreviewCanvas(self.body)
        self.canvas.pack(fill="both", expand=True)
        self.status = tk.StringVar(value="領域を選び、開始してください。")
        self.status_label = ttk.Label(self.body, textvariable=self.status, foreground=MUTED, wraplength=380)
        self.status_label.pack(fill="x", pady=(4, 0))
        self.metrics = tk.StringVar(value="検出 —  |  — fps  |  推論 — ms  |  処理合計 — ms")
        ttk.Label(self.body, textvariable=self.metrics).pack(fill="x")
        self._redraw()

    def _update_category_summary(self) -> None:
        preview = " / ".join(label[:16] + ("…" if len(label) > 16 else "") for label in self.categories[:2])
        if len(self.categories) > 2:
            preview += f" / 他{len(self.categories)-2}件"
        self.category_summary.set(f"分類候補：{preview} / 該当しない")

    def _device_changed(self, event=None) -> None:
        if self.gate.running or self.gate.busy or self.loading or self.selecting:
            self.device_choice.set("CPU" if getattr(self.backend, "requested_device", self.args.device) == "cpu" else "GPU")
            return
        if self.device_choice.get() == "GPU":
            if self.gpu_device is None:
                self.device_choice.set("CPU")
                self._set_status("利用できるGPUがありません。CPUを選択してください。", True)
                return
            selected = self.gpu_device
        else:
            selected = "cpu"
        self.backend.requested_device = selected
        self.backend.requested_dtype = "auto"
        self.ready = False
        self._set_status(f"{self.device_choice.get()}を選択しました。次の開始時にモデルを読み込みます。")
        self._controls()

    def _mode_changed(self, event=None) -> None:
        if self.gate.running or self.gate.busy or self.loading or self.selecting:
            self.mode.set(self.settings.mode)
            return
        for frame in (self.region_options, self.binary_options, self.choice_options):
            frame.pack_forget()
        {MODE_REGION: self.region_options, MODE_BINARY: self.binary_options,
         MODE_CHOICE: self.choice_options}[self.mode.get()].pack(fill="x")
        self.last_result = None
        self.positive_history.clear()
        self.metrics.set("推論 — ms  |  処理合計 — ms")
        self._set_status("")
        width = 620 if self.mode.get() == MODE_BINARY else 460 if self.mode.get() == MODE_CHOICE else 400
        height = 470 if self.mode.get() == MODE_BINARY else 340 if self.mode.get() == MODE_CHOICE else 300
        x = max(0, min(self.root.winfo_x(), self.root.winfo_screenwidth() - width - 20))
        self.root.geometry(f"{width}x{height}+{x}+{self.root.winfo_y()}")
        self._redraw()
        self._controls()

    def edit_categories(self) -> None:
        if self.gate.running or self.gate.busy or self.loading or self.selecting:
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("分類候補を編集")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        ttk.Label(dialog, text="候補を1行に1件ずつ入力してください。『該当しない』は自動で追加されます。").pack(padx=12, pady=(12, 6))
        editor = tk.Text(dialog, width=48, height=9)
        editor.pack(padx=12)
        editor.insert("1.0", "\n".join(self.categories))
        error = tk.StringVar()
        ttk.Label(dialog, textvariable=error, foreground=WARNING).pack(padx=12, pady=(4, 0))
        def save() -> None:
            try:
                self.categories = validate_categories(editor.get("1.0", "end"))
            except ValueError as exc:
                error.set(str(exc))
                return
            self._update_category_summary()
            dialog.destroy()
        ttk.Button(dialog, text="保存", command=save).pack(pady=(6, 12))
        dialog.grab_set()
        editor.focus_set()

    def _configured(self, event: tk.Event) -> None:
        if event.widget is self.root:
            self.status_label.configure(wraplength=max(350, self.root.winfo_width() - 20))

    def _set_status(self, text: str, warning: bool = False) -> None:
        self.status.set(text)
        self.status_label.configure(foreground=WARNING if warning else MUTED)

    def _controls(self) -> None:
        active = self.gate.running
        busy = self.gate.busy or self.loading or self.selecting or self.closing
        self.start_button.configure(state="disabled" if active or busy else "normal")
        self.stop_button.configure(state="normal" if active and not self.closing else "disabled")
        self.region_button.configure(state="disabled" if active or busy else "normal")
        self.prompt_entry.configure(state="disabled" if active or busy else "normal")
        self.mode_combo.configure(state="disabled" if active or busy else "readonly")
        self.device_combo.configure(state="disabled" if active or busy else "readonly")
        self.binary_entry.configure(state="disabled" if active or busy else "normal")
        self.binary_spinbox.configure(state="disabled" if active or busy else "normal")
        self.location_entry.configure(state="disabled" if active or busy else "normal")
        self.category_button.configure(state="disabled" if active or busy else "normal")
        self.interval_combo.configure(state="disabled" if active or busy else "readonly")
        self.threshold_spinbox.configure(state="disabled" if active or busy else "normal")

    def set_region(self, region: Region) -> None:
        width, height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        if not region.inside(width, height):
            raise ValueError("領域が主画面の範囲外です。")
        self.region = region
        self.region_screen = (width, height)
        self.region_text.set(f"領域：({region.left}, {region.top})  {region.width}×{region.height}")
        self.last_result = None
        self.positive_history.clear()
        self._redraw()

    def choose_region(self) -> None:
        if self.gate.running or self.gate.busy or self.loading or self.selecting:
            return
        self.selecting = True
        self._controls()
        self.root.withdraw()
        self.root.after(180, self._show_selector)

    def _show_selector(self) -> None:
        try:
            size = (self.root.winfo_screenwidth(), self.root.winfo_screenheight())
            snapshot = grab(Region(0, 0, *size))
            self.selector = RegionSelector(self.root, snapshot, size, self._selection_done)
        except Exception as exc:
            self._selection_done(None)
            self._set_status(f"領域選択に失敗：{exc}", True)

    def _selection_done(self, region: Region | None) -> None:
        self.selector = None
        self.selecting = False
        self.root.deiconify()
        self.root.lift()
        if region:
            self.set_region(region)
            self._set_status("小窓を領域外に置き、開始してください。")
        else:
            self._set_status("領域選択をキャンセルしました。")
        self._controls()

    def preview_bounds(self) -> Region:
        # Conservative margin includes native title bar / shadow across platforms.
        x = min(self.root.winfo_x(), self.root.winfo_rootx()) - 8
        y = min(self.root.winfo_y(), self.root.winfo_rooty()) - 32
        right = self.root.winfo_rootx() + self.root.winfo_width() + 8
        bottom = self.root.winfo_rooty() + self.root.winfo_height() + 12
        return Region(x, y, max(1, right - x), max(1, bottom - y))

    def overlaps(self) -> bool:
        if self.region is None:
            return False
        return self.region.overlaps(self.preview_bounds())

    def _settings_from_controls(self) -> InferenceSettings:
        mode = self.mode.get()
        if mode == MODE_REGION:
            return InferenceSettings(mode, self.prompt.get(), validate_threshold(self.threshold.get()),
                                     self.args.max_boxes, self.args.max_edge)
        if mode == MODE_BINARY:
            return InferenceSettings(mode, self.binary_prompt.get(),
                                     validate_threshold(self.binary_threshold.get()),
                                     self.args.max_boxes, self.args.max_edge, self.location_prompt.get())
        if mode == MODE_CHOICE:
            return InferenceSettings(mode, "", 0.0, self.args.max_boxes, self.args.max_edge,
                                     categories=self.categories)
        raise ValueError("モードを選択してください。")

    def start(self) -> None:
        if self.gate.running or self.gate.busy or self.loading or self.selecting or self.closing:
            return
        if self.region is None:
            self._set_status("先に『領域選択』で監視する範囲を指定してください。", True)
            return
        if self.overlaps():
            self._set_status("小窓が監視領域に重なっています。外へ移動してください。", True)
            return
        try:
            settings = self._settings_from_controls()
            interval = validate_interval(self.interval.get())
            self.gate.start(interval, time.perf_counter())
        except ValueError as exc:
            self._set_status(str(exc), True)
            return
        self.last_result = None
        self.positive_history.clear()
        self.settings = settings
        self.completed_times.clear()
        self.blocked = False
        self._redraw()
        if not self.ready:
            self.loading = True
            self._set_status("ローカルモデルを読込中。")
            self.worker.load()
        else:
            self._set_status("")
        self._controls()

    def stop(self) -> None:
        self.gate.stop()
        if self.settings.mode != MODE_BINARY:
            self.last_result = None
        self.completed_times.clear()
        self.blocked = False
        self.metrics.set("推論 — ms  |  処理合計 — ms")
        self._set_status("停止済み。")
        self._redraw()
        self._controls()

    def _handle_event(self, kind: str, data) -> None:
        if kind == "loaded":
            self.ready = True
            self.loading = False
            self._set_status("" if self.gate.running else "モデル読込済み・停止中。")
        elif kind == "error":
            task_kind, generation, text = data
            self.loading = False
            valid = self.gate.finish(generation) if task_kind == "frame" else self.gate.running
            if valid or task_kind == "load":
                self.stop()
                self._set_status(text, True)
                print(text, file=sys.stderr)
        elif kind == "result":
            result: FrameResult = data
            if self.gate.finish(result.task.generation) and not self.overlaps():
                self.last_result = result
                if result.task.settings.mode == MODE_BINARY and result.score is not None and result.score > result.task.settings.threshold:
                    self.positive_history.appendleft(result)
                self._redraw()
                now = time.perf_counter()
                self.completed_times.append(now)
                fps = (len(self.completed_times) - 1) / (self.completed_times[-1] - self.completed_times[0]) if len(self.completed_times) > 1 else 0.0
                total_ms = (now - result.task.captured_at) * 1000
                if result.task.settings.mode == MODE_REGION:
                    prefix = f"検出 {len(result.detections)}"
                elif result.task.settings.mode == MODE_BINARY:
                    prefix = f"{'検出' if result.score > result.task.settings.threshold else '該当なし'} {result.score:.2f}  |  陽性履歴 {len(self.positive_history)}/3"
                else:
                    prefix = f"{result.choice} {result.score:.2f}"
                self.metrics.set(f"{prefix}  |  {fps:.2f} fps  |  推論 {result.inference_ms:.0f} ms  |  処理合計 {total_ms:.0f} ms")
                self._set_status("")
        self._controls()

    def _tick(self) -> None:
        if self.closing:
            return
        try:
            while True:
                kind, data = self.worker.events.get_nowait()
                self._handle_event(kind, data)
        except queue.Empty:
            pass
        now = time.perf_counter()
        if self.gate.running and self.region:
            if self.region_screen != (self.root.winfo_screenwidth(), self.root.winfo_screenheight()):
                self.stop()
                self._set_status("画面構成が変わりました。領域を再選択してください。", True)
            elif self.root.state() == "iconic" or not self.root.winfo_viewable():
                self._set_status("小窓の最小化中は画面取得を停止しています。")
            elif self.overlaps():
                if not self.blocked:
                    self.blocked = True
                    self.last_result = None
                    self.completed_times.clear()
                    self.metrics.set("検出 —  |  一時停止  |  推論 — ms  |  処理合計 — ms")
                    self._redraw()
                self._set_status("重なりのため取得停止。小窓を領域外へ移動してください。", True)
            else:
                if self.blocked:
                    self.blocked = False
                    self._redraw()
                if self.ready and not self.loading:
                    ticket = self.gate.claim(now)
                    if ticket is not None:
                        try:
                            # Capture and overlap check both happen on the UI thread:
                            # no GUI event can move the window between these operations.
                            captured_at = time.perf_counter()
                            image = grab(self.region)
                            task = FrameTask(ticket, image, captured_at, time.time(), self.settings)
                            self.worker.submit(task)
                        except Exception as exc:
                            self.gate.finish(ticket)
                            self.stop()
                            self._set_status(f"画面取得に失敗：{exc}", True)
        self.tick_id = self.root.after(30, self._tick)

    def _redraw(self) -> None:
        self.canvas.show(self.mode.get(), self.last_result, tuple(self.positive_history), self.blocked)

    def close(self) -> None:
        if self.closing:
            return
        self.stop()
        self.closing = True
        if self.tick_id:
            self.root.after_cancel(self.tick_id)
        self.worker.close()
        self._controls()
        self._set_status("終了処理中。実行中のモデル処理が完了すると閉じます。")
        self._wait_close()

    def _wait_close(self) -> None:
        if self.worker.thread.is_alive():
            self.root.after(80, self._wait_close)
        else:
            self.root.destroy()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="指定領域をローカルd1-2Bで検出・判定・分類する小窓")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    p.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    p.add_argument("--compile", action="store_true", help="model.compile()を使用。初回遅延が増えるため初期検証では非推奨")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--max-boxes", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.0, help="confidenceの追加フィルター。初期値0は無効")
    p.add_argument("--max-edge", type=int, default=768, help="推論前画像の長辺上限。0は縮小しない")
    p.add_argument("--geometry", help="小窓の初期位置・寸法。例: 400x300+900+50")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        make_questions(args.prompt, args.max_boxes)
        validate_threshold(args.threshold)
        validate_interval(args.interval)
        if args.max_edge < 0:
            raise ValueError("max-edgeは0以上にしてください。")
        check_capture_session()
        enable_dpi_awareness()
        root = tk.Tk()
        DetectorApp(root, args)
        root.mainloop()
        return 0
    except (ValueError, RuntimeError, tk.TclError) as exc:
        print(f"起動できません: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
