"""One long-lived model thread. At most one frame can be submitted at a time."""
from __future__ import annotations
from dataclasses import dataclass
import queue
import threading
import time
from typing import Any
from PIL import Image
from .core import Detection, InferenceSettings, MODE_REGION, MODE_BINARY, MODE_CHOICE


@dataclass(frozen=True)
class FrameTask:
    generation: int
    image: Image.Image
    captured_at: float
    captured_wall: float
    settings: InferenceSettings


@dataclass
class FrameResult:
    task: FrameTask
    detections: list[Detection]
    inference_ms: float
    score: float | None = None
    choice: str | None = None


class InferenceWorker:
    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.commands: queue.Queue = queue.Queue(maxsize=1)
        self.events: queue.Queue = queue.Queue()
        self.shutdown = threading.Event()
        self.thread = threading.Thread(target=self._run, name="local-d1", daemon=True)
        self.thread.start()

    def load(self) -> None:
        self.commands.put_nowait(("load", None))

    def submit(self, task: FrameTask) -> None:
        self.commands.put_nowait(("frame", task))

    def close(self) -> None:
        # Stop new work; a running torch forward is not forcibly interrupted.
        self.shutdown.set()

    def _run(self) -> None:
        while not self.shutdown.is_set():
            try:
                kind, task = self.commands.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.shutdown.is_set():
                break
            try:
                if kind == "load":
                    self.events.put(("loaded", self.backend.load()))
                else:
                    start = time.perf_counter()
                    image = task.image.copy()
                    settings = task.settings
                    if settings.max_edge:
                        image.thumbnail((settings.max_edge, settings.max_edge), Image.Resampling.LANCZOS)
                    score = None
                    choice = None
                    if settings.mode == MODE_REGION:
                        detections = self.backend.predict(image, settings.prompt, settings.max_boxes, settings.threshold)
                    elif settings.mode == MODE_BINARY:
                        score, detections = self.backend.predict_binary(
                            image, settings.prompt, settings.threshold, settings.location_prompt, settings.max_boxes)
                    elif settings.mode == MODE_CHOICE:
                        choice, score, _ = self.backend.predict_choice(image, settings.categories)
                        detections = []
                    else:
                        raise ValueError(f"不明なモード: {settings.mode}")
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    self.events.put(("result", FrameResult(task, detections, elapsed_ms, score, choice)))
            except Exception as exc:
                # No image contents or model answer are written to disk/logs.
                self.events.put(("error", (kind, task.generation if task else None, f"{type(exc).__name__}: {exc}")))
            finally:
                self.commands.task_done()
