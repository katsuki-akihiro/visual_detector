"""Print environment information only; no screen capture, download, or inference."""
from __future__ import annotations
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from desktop_detector.backend import DEFAULT_MODEL_DIR, verify_model_directory

info = {"python": sys.version, "os": platform.platform(),
        "session": os.environ.get("XDG_SESSION_TYPE", ""), "packages": {}}
for name in ("Pillow", "transformers", "huggingface-hub", "torch", "torchvision"):
    try:
        info["packages"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        info["packages"][name] = "NOT INSTALLED"
try:
    import tkinter
    info["tk"] = tkinter.TkVersion
except ImportError as exc:
    info["tk"] = str(exc)
try:
    import torch
    info["accelerators"] = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(),
                            "torch_cuda": torch.version.cuda}
    if torch.cuda.is_available():
        info["accelerators"]["gpu"] = torch.cuda.get_device_name(0)
        info["accelerators"]["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
except Exception as exc:
    info["accelerators"] = str(exc)
try:
    lock = verify_model_directory(DEFAULT_MODEL_DIR)
    info["model"] = {"id": lock["model_id"], "revision": lock["revision"], "directory": str(DEFAULT_MODEL_DIR)}
except Exception as exc:
    info["model"] = str(exc)
print(json.dumps(info, ensure_ascii=False, indent=2))
