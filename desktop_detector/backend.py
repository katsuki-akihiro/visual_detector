"""Local d1-2B adapter; no HTTP inference server, no generative chat endpoint."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image
from .core import (Detection, MODEL_ID, make_questions, parse_boxes, make_binary_question,
                   make_choice_question, parse_binary, parse_choice)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "d1-2B"


def verify_model_directory(model_dir: Path) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    lock_path = model_dir / "model.lock.json"
    if not lock_path.is_file():
        raise RuntimeError("モデル未準備です。先に python tools/download_model.py --accept-model-code を実行してください。")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("model_id") != MODEL_ID:
        raise RuntimeError(f"このモックは {MODEL_ID} 専用です。")
    if not (model_dir / "config.json").is_file() or not list(model_dir.glob("*.safetensors")):
        raise RuntimeError("config.json またはモデル重みがありません。ダウンロードを完了してください。")
    hashes = lock.get("code_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("モデルコードの整合性記録がありません。再ダウンロードしてください。")
    for name, digest in hashes.items():
        file = (model_dir / name).resolve()
        if not file.is_relative_to(model_dir) or not file.is_file():
            raise RuntimeError("モデルコードのパスまたはファイルが不正です。")
        if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"モデルコードがダウンロード時から変更されています: {name}")
    return lock


class D1Backend:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, device: str = "auto", dtype: str = "auto", compile_model: bool = False) -> None:
        self.model_dir = Path(model_dir)
        self.requested_device = device
        self.requested_dtype = dtype
        self.compile_model = compile_model
        self.model: Any = None
        self.torch: Any = None
        self.device = ""
        self.description = "LiquidAI/d1-2B / 未読込"

    def load(self) -> str:
        lock = verify_model_directory(self.model_dir)
        # Switching devices reloads weights; release the previous copy first.
        self.model = None
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        # Force local loading, including the processor/tokenizer's lazy paths.
        # This is not an OS-level sandbox for third-party Python code.
        os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["DO_NOT_TRACK"] = "1"
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("推論依存関係が不足しています。構築手順書の『実モデル導入』を実施してください。") from exc
        self.torch = torch
        device = self.requested_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDAを利用できません。GPUドライバーとCUDA対応PyTorchを確認してください。")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPSを利用できません。Apple SiliconとPyTorchの構成を確認してください。")
        self.device = device
        dtype_name = self.requested_dtype
        if dtype_name == "auto":
            dtype_name = ("bfloat16" if torch.cuda.is_bf16_supported(including_emulation=False) else "float16") if device == "cuda" else "float32"
        if device == "cpu" and dtype_name != "float32":
            raise RuntimeError("このMVPのCPU実行はfloat32に限定しています。")
        dtype = getattr(torch, dtype_name)
        self.model = AutoModel.from_pretrained(
            str(self.model_dir.resolve()), trust_remote_code=True,
            local_files_only=True, dtype=dtype,
        ).to(device).eval()
        if not callable(getattr(self.model, "system_one", None)):
            raise RuntimeError("system_one() がありません。d1-2BのモデルコードとTransformersを確認してください。")
        if self.compile_model:
            self.model.compile()
        self.description = f"d1-2B / {device} / {dtype_name} / {lock['revision'][:8]}"
        return self.description

    def synchronize(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def _ask(self, image: Image.Image, question: dict) -> dict:
        if self.model is None:
            raise RuntimeError("モデルを先に読み込んでください。")
        with self.torch.inference_mode():
            answer = self.model.system_one(None, question, media=[image.convert("RGB")])
        self.synchronize()
        return answer

    def predict(self, image: Image.Image, prompt: str, max_boxes: int = 8, threshold: float = 0.0) -> list[Detection]:
        return parse_boxes(self._ask(image, make_questions(prompt, max_boxes)), threshold)

    def predict_binary(self, image: Image.Image, prompt: str, threshold: float,
                       location_prompt: str, max_boxes: int) -> tuple[float, list[Detection]]:
        score = parse_binary(self._ask(image, make_binary_question(prompt)))
        boxes: list[Detection] = []
        if score > threshold and location_prompt.strip():
            boxes = parse_boxes(self._ask(image, make_questions(location_prompt, max_boxes)))
        return score, boxes

    def predict_choice(self, image: Image.Image, categories: tuple[str, ...]) -> tuple[str, float, dict[str, float]]:
        return parse_choice(self._ask(image, make_choice_question(categories)), categories)
