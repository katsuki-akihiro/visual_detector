"""Intel NPU backend: d1-2B's vision encoder + language model compiled by OpenVINO.

Only the heavy forward (`text_backbone(model)(input_ids, attention_mask, pixel_values, ...)`)
runs on the NPU. Prompt rendering, tokenization, slot read-out and calibration stay in
d1's own Python code (`SystemOne`), so the three modes answer in the same format as
the PyTorch backend. The NPU needs static shapes, so every picture is resized to
FIXED_SIZE and every sequence is right-padded to SEQ_LEN (a right pad never reaches a
real token under d1's masks). Boxes are fractions of the picture, so the stretch to
FIXED_SIZE does not change their meaning.

No CPU fallback: the compiled model is pinned to the "NPU" device and the backend
reports the device OpenVINO actually uses.
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from .backend import DEFAULT_MODEL_DIR, PROJECT_ROOT, verify_model_directory
from .core import (Detection, make_binary_question, make_choice_question, make_questions,
                   parse_binary, parse_boxes, parse_choice)

FIXED_SIZE = (640, 480)          # width, height -> a 40 x 30 grid of 16 px patches
SEQ_LEN = 640                    # longest prompt: picture (~306 tokens) + question
DEFAULT_IR_DIR = PROJECT_ROOT / "models" / "d1-2B-npu"
IR_NAME = "d1_backbone.xml"
INPUT_NAMES = ("input_ids", "attention_mask", "state_len", "question", "pixel_values")


def npu_problem(ir_dir: Path = DEFAULT_IR_DIR) -> str | None:
    """Why the NPU cannot be chosen now, or None when it can."""
    try:
        import openvino as ov
    except ImportError:
        return "OpenVINOが未導入です。構築手順書の『NPU版』を実施してください。"
    try:
        devices = ov.Core().available_devices
    except Exception as exc:
        return f"OpenVINOのデバイス一覧を取得できません: {exc}"
    if "NPU" not in devices:
        return "OpenVINOからNPUが見えません。Intel NPUドライバーを確認してください。"
    if not (Path(ir_dir) / IR_NAME).is_file():
        return "NPU用モデルが未変換です。python tools/export_npu.py を実行してください。"
    return None


def _d1_module(model: Any, name: str):
    """A module of the model's own code (loaded by transformers under a dynamic package)."""
    return sys.modules[type(model).__module__.rsplit(".", 1)[0] + "." + name]


def fixed_picture(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    return image if image.size == FIXED_SIZE else image.resize(FIXED_SIZE, Image.Resampling.BICUBIC)


def pad_inputs(ids, mask, state, question, seq_len: int = SEQ_LEN):
    """Right-pad one row to seq_len: pad id, mask 0, question -1 (d1's own padding)."""
    import torch
    width = ids.shape[1]
    if width > seq_len:
        raise RuntimeError(f"入力が {width} トークンで、NPU用の固定長 {seq_len} を超えます。指示や候補を短くしてください。")
    extra = seq_len - width
    ids = torch.nn.functional.pad(ids, (0, extra), value=0)
    mask = torch.nn.functional.pad(mask, (0, extra), value=0)
    question = torch.nn.functional.pad(question, (0, extra), value=-1)
    return ids, mask, state, question


class static_masked_scatter:
    """Within the block, Tensor.masked_scatter is a static-shape equivalent.

    d1's VL wrapper places the picture features with `inputs_embeds.masked_scatter(mask, features)`,
    which OpenVINO converts to NonZero + ScatterNDUpdate: a data-dependent shape the NPU
    compiler rejects ("Failed to create a valid MLIR module"). The same result as a gather:
    the k-th masked position takes source row k (cumsum of the mask), elsewhere self.
    """

    def __enter__(self):
        import torch
        self.original = torch.Tensor.masked_scatter

        def masked_scatter(target, mask, source):
            mask = mask.expand_as(target)
            rows = mask.reshape(-1, target.shape[-1])[:, 0]           # one flag per token
            index = (rows.long().cumsum(0) - 1).clamp(min=0)          # k-th masked token -> source row k
            placed = source.reshape(-1, target.shape[-1])[index].reshape(target.shape)
            return torch.where(mask, placed.to(target.dtype), target)

        torch.Tensor.masked_scatter = masked_scatter
        return self

    def __exit__(self, *exc):
        import torch
        torch.Tensor.masked_scatter = self.original
        return False


class sdpa_without_gqa:
    """Within the block, grouped-query SDPA calls get K/V repeated to the query heads.

    d1-2B's vision encoder (16 query / 8 KV heads) and language model (16 / 4) call
    F.scaled_dot_product_attention(..., enable_gqa=True). The NPU compiler's
    DecomposeAttention pass fails on that op ("Incompatible MatMul batch dimension",
    [8,2,1200,64] x [1,8,1200,64]). Repeating K/V first is the same computation.
    """

    def __enter__(self):
        import torch.nn.functional as F
        self.original = F.scaled_dot_product_attention
        original = self.original

        def sdpa(q, k, v, *args, enable_gqa=False, **kwargs):
            if enable_gqa and k.shape[-3] != q.shape[-3]:
                groups = q.shape[-3] // k.shape[-3]
                k = k.repeat_interleave(groups, dim=-3)
                v = v.repeat_interleave(groups, dim=-3)
            return original(q, k, v, *args, **kwargs)

        F.scaled_dot_product_attention = sdpa
        return self

    def __exit__(self, *exc):
        import torch.nn.functional as F
        F.scaled_dot_product_attention = self.original
        return False


class BackboneForExport:
    """torch.nn.Module factory kept lazy so importing this file does not need torch."""

    @staticmethod
    def build(model: Any, grid):
        import torch
        attention = _d1_module(model, "attention")

        class Backbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model
                self.register_buffer("grid", grid.clone(), persistent=False)

            def forward(self, input_ids, attention_mask, state_len, question, pixel_values):
                with attention.state_attention(self.model, state_len, question):
                    out = attention.text_backbone(self.model)(
                        input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                        pixel_values=pixel_values, image_grid_thw=self.grid)
                hidden = getattr(out, "last_hidden_state", None)
                return out[0] if hidden is None else hidden

        return Backbone().eval()


class NPUBackend:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, ir_dir: Path = DEFAULT_IR_DIR,
                 device: str = "NPU") -> None:
        self.model_dir = Path(model_dir)
        self.ir_dir = Path(ir_dir)
        self.ov_device = device
        self.requested_device = "npu"
        self.requested_dtype = "auto"
        self.model: Any = None
        self.torch: Any = None
        self.compiled: Any = None
        self.request: Any = None
        self.head: Any = None
        self.grid: Any = None
        self.seq_len = SEQ_LEN
        self.device = ""
        self.description = "LiquidAI/d1-2B / NPU / 未読込"

    # ------------------------------------------------------------------ load
    def load(self) -> str:
        lock = verify_model_directory(self.model_dir)
        meta_path = self.ir_dir / "export.json"
        if not (self.ir_dir / IR_NAME).is_file() or not meta_path.is_file():
            raise RuntimeError("NPU用モデルがありません。先に python tools/export_npu.py を実行してください。")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("revision") != lock["revision"]:
            raise RuntimeError("NPU用モデルが現在のd1-2Bと異なる版から変換されています。再変換してください。")
        self.model = self.compiled = self.request = None
        gc.collect()
        os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        try:
            import openvino as ov
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("NPU実行には openvino が必要です。構築手順書の『NPU版』を実施してください。") from exc
        self.torch = torch
        core = ov.Core()
        if self.ov_device not in core.available_devices:
            raise RuntimeError(f"OpenVINOから {self.ov_device} が見えません。Intel NPUドライバーを確認してください。")
        # The d1 runner (tokenizer, prompt, read-out) needs the model object; its weights
        # are dropped below except the LM head, which projects the few answer slots.
        model = AutoModel.from_pretrained(str(self.model_dir.resolve()), trust_remote_code=True,
                                          local_files_only=True, dtype=torch.bfloat16).eval()
        engine = model.engine
        self.head = _d1_module(model, "slots").output_head(model).weight.detach().float().clone()
        model.to("meta")  # free every backbone weight; the NPU holds its own copy
        gc.collect()
        engine._slot_logits = lambda seqs, batch, slots, pointer: self._slot_logits(engine, seqs, batch, slots, pointer)
        self.model = model
        self.grid = torch.tensor(meta["image_grid_thw"], dtype=torch.long)
        self.seq_len = int(meta.get("seq_len", SEQ_LEN))
        cache = PROJECT_ROOT / ".ov_cache"
        cache.mkdir(exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache)})
        self.compiled = core.compile_model(str(self.ir_dir / IR_NAME), self.ov_device)
        actual = self.compiled.get_property("EXECUTION_DEVICES")
        actual = [actual] if isinstance(actual, str) else list(actual)  # a str or a list by OpenVINO version
        if actual != [self.ov_device]:
            raise RuntimeError(f"{self.ov_device}ではなく {actual} で実行されます。")
        self.request = self.compiled.create_infer_request()
        # "npu" only when the NPU itself runs it; the CPU plugin is a diagnostic, named as such.
        self.device = "npu" if self.ov_device == "NPU" else f"openvino-{self.ov_device.lower()}"
        name = core.get_property(self.ov_device, "FULL_DEVICE_NAME")
        self.description = (f"d1-2B / {self.device} ({name}) / OpenVINO {ov.__version__.split('-')[0]} / "
                            f"{meta.get('weights', 'fp16')} / {lock['revision'][:8]}")
        self.short_description = f"{self.device}・{meta.get('weights', 'fp16')}"
        return self.description

    # -------------------------------------------------------------- forward
    def _forward(self, ids, mask, state, question, media) -> Any:
        torch = self.torch
        if media.get("image_grid_thw") is None or not torch.equal(media["image_grid_thw"].cpu(), self.grid):
            raise RuntimeError("NPU用の固定画像サイズと異なる入力です。")
        width = ids.shape[1]
        ids, mask, state, question = pad_inputs(ids, mask, state, question, self.seq_len)
        feeds = {"input_ids": ids.numpy(), "attention_mask": mask.numpy(), "state_len": state.numpy(),
                 "question": question.numpy(), "pixel_values": media["pixel_values"].float().numpy()}
        out = self.request.infer(feeds)
        return torch.from_numpy(out[self.compiled.output(0)].copy())[:, :width]

    def _slot_logits(self, engine, seqs, batch, slots, pointer):
        """Replacement of SystemOne._slot_logits: the same rows, the backbone on the NPU."""
        torch = self.torch
        slots_mod = _d1_module(self.model, "slots")
        if engine.attention != "set":
            raise RuntimeError("NPU版はd1-2Bの set attention 専用です。")
        hidden_rows = []
        for b in batch:
            ids, lengths, media = engine._rows(seqs, [b])
            mask = torch.ones_like(ids)
            state = torch.tensor([seqs[b][3]])
            question = torch.tensor([seqs[b][4] + [-1] * (ids.shape[1] - len(seqs[b][4]))])
            hidden_rows.append(self._forward(ids, mask, state, question, media))
        width = max(h.shape[1] for h in hidden_rows)
        hidden = torch.cat([torch.nn.functional.pad(h, (0, 0, 0, width - h.shape[1])) for h in hidden_rows])
        rows = torch.tensor([s[0] for s in slots])  # a slot's row within this batch, as in SystemOne
        last = torch.tensor([s[3] for s in slots])
        picked = hidden[rows, last]
        states = hidden[pointer["rows"], : pointer["width"]] if pointer is not None else None
        out = slots_mod.split(picked, states, pointer, lambda x: x.float() @ self.head.T)
        codes = out.codes.float()
        return codes - torch.logsumexp(codes, dim=-1, keepdim=True), out.positions

    # ----------------------------------------------------------------- modes
    def _ask(self, image: Image.Image, question: dict) -> dict:
        if self.request is None:
            raise RuntimeError("モデルを先に読み込んでください。")
        with self.torch.inference_mode():
            return self.model.system_one(None, question, media=[fixed_picture(image)])

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
