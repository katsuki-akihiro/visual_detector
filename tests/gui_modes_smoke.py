"""Desktop UI integration for the two new modes, using a deterministic local backend."""
from __future__ import annotations

import time
import tkinter as tk

from PIL import Image
from desktop_detector.core import Detection, INTERVAL_CHOICES, MODE_BINARY, MODE_CHOICE, Region
from desktop_detector.ui import DetectorApp, parser
import desktop_detector.ui as ui


class FakeBackend:
    device = "cpu"

    def load(self):
        return "fake"

    def predict_binary(self, image, prompt, threshold, location_prompt, max_boxes):
        assert prompt == "魚が映っているか？"
        return .8, [Detection(.1, .1, .8, .8)] if location_prompt else []

    def predict_choice(self, image, categories):
        assert categories == ("魚がある", "猫がいる")
        return "猫がいる", .9, {"魚がある": .05, "猫がいる": .9, "該当しない": .05}


def pump(root, condition, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        root.update()
        if condition():
            return
        time.sleep(.025)
    raise AssertionError("GUI result timeout")


def main():
    root = tk.Tk()
    args = parser().parse_args(["--geometry", f"400x300+{max(300, root.winfo_screenwidth()-650)}+60"])
    original_grab = ui.grab
    ui.grab = lambda region: Image.new("RGB", (region.width, region.height), "#345678")
    app = DetectorApp(root, args, backend=FakeBackend())
    try:
        root.update()
        assert tuple(app.interval_combo["values"]) == INTERVAL_CHOICES
        app.mode.set(MODE_BINARY)
        app._mode_changed()
        root.update()
        app.set_region(Region(10, 100, 120, 90))
        assert app.location_prompt.get() == "映っている魚をすべて囲む"
        assert app.binary_threshold.get() == "0.20"
        app.start()
        pump(root, lambda: len(app.completed_times) >= 4)
        assert len(app.positive_history) == 3
        assert all(item.task.captured_wall > 0 and item.detections for item in app.positive_history)
        stamps = [item.task.captured_wall for item in app.positive_history]
        assert stamps == sorted(stamps, reverse=True)
        app.stop()
        assert len(app.positive_history) == 3
        app.mode.set(MODE_CHOICE)
        app._mode_changed()
        app.categories = ("魚がある", "猫がいる")
        app._update_category_summary()
        app.interval.set("10.0")
        app.start()
        assert app.gate.interval == 10.0
        pump(root, lambda: len(app.completed_times) >= 1)
        assert app.last_result.choice == "猫がいる"
        assert "猫がいる" in app.metrics.get()
        app.stop()
        app.device_choice.set("CPU")
        app._device_changed()
        assert app.backend.requested_device == "cpu" and not app.ready
        assert app.backend.requested_dtype == "auto"
        gpu = app.gpu_device
        app.gpu_device = None
        app.device_choice.set("GPU")
        app._device_changed()
        assert app.device_choice.get() == "CPU" and app.backend.requested_device == "cpu"
        app.gpu_device = gpu
        if app.gpu_device:
            app.device_choice.set("GPU")
            app._device_changed()
            assert app.backend.requested_device == app.gpu_device
        print("PASS: binary history, timestamps, localization, choice mode, device selection")
    finally:
        ui.grab = original_grab
        app.close()
        deadline = time.monotonic() + 5
        while app.worker.thread.is_alive() and time.monotonic() < deadline:
            root.update()
            time.sleep(.025)


if __name__ == "__main__":
    main()
