"""Exercise the real local model through the Windows desktop GUI once."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys
import time
import tkinter as tk
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from desktop_detector.core import Region
from desktop_detector.ui import DetectorApp, parser


def main() -> int:
    cli = argparse.ArgumentParser(description="猫2匹の画像で実モデルの画面取得を確認")
    cli.add_argument("image", type=Path, help="猫2匹が写った画像のパス")
    options = cli.parse_args()
    root = tk.Tk()
    args = parser().parse_args(["--geometry", f"400x300+{max(0, root.winfo_screenwidth()-424)}+70", "--prompt", "Every visible cat.",
                                "--threshold", "1.00"])
    app = DetectorApp(root, args)
    target = tk.Toplevel(root)
    target.title("d1-2B real model test image")
    target.resizable(False, False)
    with Image.open(options.image) as source:
        image = source.convert("RGB")
    target.geometry(f"{image.width}x{image.height}+40+80")
    photo = ImageTk.PhotoImage(image, master=root)
    tk.Label(target, image=photo, borderwidth=0, highlightthickness=0).pack()
    target.attributes("-topmost", True)
    root.update()
    target.lift()
    root.update()
    app.set_region(Region(target.winfo_rootx(), target.winfo_rooty(), image.width, image.height))
    try:
        assert app.threshold.get() == "1.00"
        app.start()
        deadline = time.perf_counter() + 90
        while time.perf_counter() < deadline and len(app.completed_times) < 2:
            root.update()
            if not app.gate.running:
                raise RuntimeError(app.status.get())
            time.sleep(.01)
        if len(app.completed_times) < 2 or app.last_result is None:
            raise TimeoutError(f"Real GUI inference timed out: {app.status.get()}")
        filtered = len(app.last_result.detections)
        first_frames = len(app.completed_times)
        assert app.last_result.task.settings.threshold == 1.0 and filtered == 0
        app.stop()
        deadline = time.perf_counter() + 30
        while app.gate.busy and time.perf_counter() < deadline:
            root.update()
            time.sleep(.01)
        assert not app.gate.busy
        app.threshold.set("0.00")
        app.start()
        deadline = time.perf_counter() + 30
        while time.perf_counter() < deadline and len(app.completed_times) < 2:
            root.update()
            if not app.gate.running:
                raise RuntimeError(app.status.get())
            time.sleep(.01)
        if len(app.completed_times) < 2 or app.last_result is None:
            raise TimeoutError(f"Real GUI inference timed out: {app.status.get()}")
        result = {
            "backend": app.backend.description,
            "frames_rendered": first_frames + len(app.completed_times),
            "detections_at_1_00": filtered,
            "detections": len(app.last_result.detections),
            "threshold": app.last_result.task.settings.threshold,
            "inference_ms": round(app.last_result.inference_ms, 1),
            "status": app.status.get(),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["detections"] != 2:
            raise AssertionError(f"Expected two cats, got {result['detections']}")
        return 0
    finally:
        app.stop()
        app.close()
        target.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
