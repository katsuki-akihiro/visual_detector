"""Real model through the desktop GUI: run, stop, resume, switch device, switch back.

    python -m tests.real_gui_device_switch path/to/cats.jpg [--accelerator GPU|NPU]

The accelerator must really run: the test fails if the backend reports any other
device after loading (no silent CPU fallback is accepted).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import tkinter as tk

from PIL import Image, ImageGrab, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from desktop_detector.capture import enable_dpi_awareness  # noqa: E402
from desktop_detector.core import Region  # noqa: E402
from desktop_detector.ui import DetectorApp, parser  # noqa: E402


def pump(root, app, frames: int, seconds: float) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline and len(app.completed_times) < frames:
        root.update()
        if not app.gate.running:
            raise RuntimeError(app.status.get())
        time.sleep(.01)
    if len(app.completed_times) < frames:
        raise TimeoutError(f"{frames} frames not reached: {app.status.get()}")


def stop_and_wait(root, app) -> None:
    app.stop()
    deadline = time.perf_counter() + 300
    while app.gate.busy and time.perf_counter() < deadline:
        root.update()
        time.sleep(.01)
    assert not app.gate.busy, "in-flight frame did not finish after stop"


def main() -> int:
    cli = argparse.ArgumentParser(description="実モデルで停止・再開・デバイス切替をGUI上で確認")
    cli.add_argument("image", type=Path)
    cli.add_argument("--accelerator", default="GPU", choices=("GPU", "NPU"))
    cli.add_argument("--screenshot", type=Path, help="実行先表示の確認用スクリーンショット")
    options = cli.parse_args()
    enable_dpi_awareness()  # same as app.py: Tk and screen capture in one pixel space
    root = tk.Tk()
    args = parser().parse_args(["--geometry", f"400x320+{max(0, root.winfo_screenwidth()-424)}+70",
                                "--prompt", "映っている猫をすべて囲む",
                                "--device", "npu" if options.accelerator == "NPU" else "auto"])
    app = DetectorApp(root, args)
    target = tk.Toplevel(root)
    target.title("device switch test image")
    with Image.open(options.image) as source:
        image = source.convert("RGB")
    target.geometry(f"{image.width}x{image.height}+40+80")
    photo = ImageTk.PhotoImage(image, master=root)
    tk.Label(target, image=photo, borderwidth=0, highlightthickness=0).pack()
    target.attributes("-topmost", True)
    root.update()
    app.set_region(Region(target.winfo_rootx(), target.winfo_rooty(), image.width, image.height))
    steps = []

    def phase(name: str, expect: str, frames: int, seconds: float) -> None:
        start = time.perf_counter()
        app.start()
        pump(root, app, frames, seconds)
        actual = app.backend.device
        if actual != expect:
            raise AssertionError(f"{name}: expected {expect}, ran on {actual}")
        steps.append({"step": name, "device": actual, "frames": len(app.completed_times),
                      "detections": len(app.last_result.detections),
                      "last_inference_ms": round(app.last_result.inference_ms),
                      "wall_s": round(time.perf_counter() - start, 1), "label": app.backend_text.get()})
        if expect not in app.backend_text.get():
            raise AssertionError(f"{name}: the screen shows {app.backend_text.get()!r}, not {expect}")
        if options.screenshot and name == "run":
            root.update()
            x, y = root.winfo_rootx(), root.winfo_rooty()
            ImageGrab.grab((x, y, x + root.winfo_width(), y + root.winfo_height())).save(options.screenshot)
        stop_and_wait(root, app)

    try:
        accel = options.accelerator
        expect = app.gpu_device if accel == "GPU" else "npu"
        if expect is None:
            raise RuntimeError("GPUが検出されません。")
        assert app.device_choice.get() == accel, app.device_choice.get()
        phase("run", expect, 3, 300)
        phase("resume", expect, 3, 120)
        app.device_choice.set("CPU")
        app._device_changed()
        assert not app.ready and app.backend.requested_device == "cpu"
        phase("switch_to_cpu", "cpu", 1, 600)
        app.device_choice.set(accel)
        app._device_changed()
        phase("switch_back", expect, 3, 300)
        print(json.dumps({"backend": app.backend.description, "steps": steps}, ensure_ascii=False, indent=2))
        assert all(step["detections"] == 2 for step in steps), "expected two cats on every device"
        return 0
    finally:
        app.stop()
        app.close()
        deadline = time.monotonic() + 60
        while app.worker.thread.is_alive() and time.monotonic() < deadline:
            root.update()
            time.sleep(.025)
        target.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
