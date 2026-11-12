"""Convert d1-2B's vision encoder + language model to OpenVINO IR with static shapes for the NPU.

    python tools/export_npu.py [--weights fp16|int8|int4] [--skip-npu]

Steps: load the local model (bfloat16, CPU) -> record one real backbone call on a
FIXED_SIZE picture -> right-pad it to SEQ_LEN -> ov.convert_model (TorchScript trace)
-> compare the IR on the OpenVINO CPU plugin against PyTorch -> save -> compile for NPU.
Each step prints its time and peak memory; a failure names the step.
No image or text is written; the picture used for tracing is synthetic.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from desktop_detector.backend import DEFAULT_MODEL_DIR, PROJECT_ROOT, verify_model_directory  # noqa: E402
from desktop_detector.core import make_questions  # noqa: E402
from desktop_detector.npu_backend import (DEFAULT_IR_DIR, FIXED_SIZE, INPUT_NAMES, IR_NAME, SEQ_LEN,  # noqa: E402
                                          BackboneForExport, pad_inputs, sdpa_without_gqa,
                                          static_masked_scatter)


def peak_mb() -> int:
    try:
        sys.path.insert(0, str(ROOT / "tools"))
        from benchmark import process_memory
        return round(process_memory()["peak_commit_mb"])
    except Exception:
        return -1


def step(name: str, report: dict):
    class Step:
        def __enter__(self):
            self.t = time.perf_counter()
            print(f"[{name}] ...", flush=True)

        def __exit__(self, kind, exc, tb):
            entry = {"step": name, "s": round(time.perf_counter() - self.t, 1), "peak_commit_mb": peak_mb()}
            if exc is not None:
                entry["error"] = f"{kind.__name__}: {str(exc)[:2000]}"
                entry["where"] = [f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in traceback.extract_tb(tb)][-15:]
            report["steps"].append(entry)
            print(json.dumps(entry, ensure_ascii=False), flush=True)
            return False
    return Step()


def main() -> int:
    p = argparse.ArgumentParser(description="d1-2BをNPU向けOpenVINO IRへ変換")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_IR_DIR)
    p.add_argument("--weights", choices=("fp16", "int8", "int4"), default="fp16",
                   help="IRの重み形式。int8/int4はnncfが必要")
    p.add_argument("--skip-npu", action="store_true", help="NPUコンパイル確認を省略")
    p.add_argument("--seq-len", type=int, default=SEQ_LEN,
                   help="固定トークン長。長い指示・候補を使う場合に増やす（推論は遅くなる）")
    p.add_argument("--trace-dtype", choices=("float32", "bfloat16"), default="float32",
                   help="トレース時の精度。bfloat16はメモリが少なく済むがNPUでコンパイルできない")
    p.add_argument("--report", type=Path, help="各ステップの結果JSON")
    args = p.parse_args()
    report: dict = {"seq_len": args.seq_len, "fixed_size": FIXED_SIZE, "weights": args.weights, "steps": []}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import numpy as np
        import openvino as ov
        import torch
        from PIL import Image
        from transformers import AutoModel
        report["openvino"] = ov.__version__
        report["torch"] = torch.__version__
        lock = verify_model_directory(args.model_dir)
        # Trace in float32: a bfloat16 trace leaves bf16 compute ops in the IR, which the
        # NPU compiler rejects ("Failed to create a valid MLIR module"). Weights are
        # still stored as fp16 (compress_to_fp16) or int8/int4 below.
        with step(f"load_pytorch_{args.trace_dtype}", report):
            model = AutoModel.from_pretrained(str(args.model_dir.resolve()), trust_remote_code=True,
                                              local_files_only=True, dtype=getattr(torch, args.trace_dtype)).eval()
            engine = model.engine
        captured = {}
        with step("record_backbone_call", report):
            original = engine._slot_logits

            def capture(seqs, batch, slots, pointer):
                ids, lengths, media = engine._rows(seqs, batch)
                captured.update(ids=ids, state=torch.tensor([seqs[b][3] for b in batch]),
                                question=torch.tensor([seqs[b][4] + [-1] * (ids.shape[1] - len(seqs[b][4])) for b in batch]),
                                media=media)
                return original(seqs, batch, slots, pointer)
            engine._slot_logits = capture
            rng = np.random.default_rng(0)
            picture = Image.fromarray(rng.integers(0, 255, (FIXED_SIZE[1], FIXED_SIZE[0], 3), dtype=np.uint8))
            with torch.inference_mode():
                model.system_one(None, make_questions("映っている猫をすべて囲む", 8), media=[picture])
            engine._slot_logits = original
            grid = captured["media"]["image_grid_thw"].cpu()
            extra = sorted(set(captured["media"]) - {"pixel_values", "image_grid_thw"})
            if extra:
                raise RuntimeError(f"想定外のmedia入力: {extra}")
            report["image_grid_thw"] = grid.tolist()
            report["recorded_tokens"] = int(captured["ids"].shape[1])
            print(f"grid={grid.tolist()} tokens={captured['ids'].shape[1]} pixels={list(captured['media']['pixel_values'].shape)}")
        ids, mask, state, question = pad_inputs(captured["ids"], torch.ones_like(captured["ids"]),
                                                captured["state"], captured["question"], args.seq_len)
        pixels = captured["media"]["pixel_values"].float()
        example = {"input_ids": ids, "attention_mask": mask, "state_len": state, "question": question,
                   "pixel_values": pixels}
        wrapper = BackboneForExport.build(model, grid)
        with step("pytorch_reference", report):
            with torch.no_grad():
                reference = wrapper(**example).float()
                unpadded = wrapper(captured["ids"], torch.ones_like(captured["ids"]), captured["state"],
                                   captured["question"], pixels).float()
                with static_masked_scatter(), sdpa_without_gqa():
                    patched = wrapper(**example).float()
            width = captured["ids"].shape[1]
            pad_diff = (reference[:, :width] - unpadded).abs().max().item()
            report["padding_max_abs_diff"] = pad_diff
            report["export_patches_max_abs_diff"] = (patched - reference).abs().max().item()
            print(f"right-padding effect (max |diff|): {pad_diff:.4g}; "
                  f"export patches vs original: {report['export_patches_max_abs_diff']:.4g}")
            if report["export_patches_max_abs_diff"] > 1e-2:
                raise RuntimeError("変換用の置き換え（masked_scatter・GQA）が元の処理と一致しません。")
        with step("convert_model", report):
            with torch.no_grad(), static_masked_scatter(), sdpa_without_gqa():
                ov_model = ov.convert_model(wrapper, example_input=example)
            for port, name in zip(ov_model.inputs, INPUT_NAMES):
                port.get_tensor().set_names({name})
            ov_model.reshape({name: list(example[name].shape) for name in INPUT_NAMES})
            report["ops"] = sorted({op.get_type_name() for op in ov_model.get_ops()})
            report["bf16_compute_ops"] = sum(1 for op in ov_model.get_ops() if op.get_type_name() != "Constant"
                                             and any(str(o.get_element_type()) == "<Type: 'bfloat16'>"
                                                     for o in op.outputs()))
            dynamic = sorted({op.get_friendly_name() for op in ov_model.get_ops()
                              if any(o.get_partial_shape().is_dynamic for o in op.outputs())})
            report["dynamic_shape_ops"] = dynamic[:20]
            if dynamic:
                print("WARNING: dynamic-shaped ops remain (the NPU needs static shapes):", dynamic[:5])
        del wrapper, model, engine
        gc.collect()
        if args.weights != "fp16":
            with step(f"compress_weights_{args.weights}", report):
                import nncf
                mode = nncf.CompressWeightsMode.INT8_SYM if args.weights == "int8" else nncf.CompressWeightsMode.INT4_SYM
                kwargs = {} if args.weights == "int8" else {"group_size": 128, "ratio": 1.0}
                ov_model = nncf.compress_weights(ov_model, mode=mode, **kwargs)
        args.output.mkdir(parents=True, exist_ok=True)
        with step("save_ir", report):
            ov.save_model(ov_model, str(args.output / IR_NAME), compress_to_fp16=True)
            del ov_model
            gc.collect()
            report["ir_mb"] = round(sum(f.stat().st_size for f in args.output.glob("d1_backbone.*")) / 2**20)
        feeds = {k: v.numpy() for k, v in example.items()}
        core = ov.Core()
        with step("check_on_openvino_cpu", report):
            compiled = core.compile_model(str(args.output / IR_NAME), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
            out = compiled.create_infer_request().infer(feeds)[compiled.output(0)]
            diff = np.abs(out[:, :width] - reference[:, :width].numpy())
            report["cpu_vs_pytorch"] = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                                        "ref_mean_abs": float(reference[:, :width].abs().mean())}
            print(report["cpu_vs_pytorch"])
            del compiled
            gc.collect()
        meta = {"model_id": lock["model_id"], "revision": lock["revision"], "seq_len": args.seq_len,
                "fixed_size": FIXED_SIZE, "image_grid_thw": report["image_grid_thw"], "weights": args.weights,
                "openvino": ov.__version__, "inputs": list(INPUT_NAMES)}
        (args.output / "export.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        if not args.skip_npu:
            with step("compile_npu", report):
                cache = PROJECT_ROOT / ".ov_cache"
                cache.mkdir(exist_ok=True)
                core.set_property({"CACHE_DIR": str(cache)})
                # LOG_ERROR prints the NPU compiler's own reason (e.g. the failing pass) to stderr.
                compiled = core.compile_model(str(args.output / IR_NAME), "NPU", {"LOG_LEVEL": "LOG_ERROR"})
                report["npu_execution_devices"] = list(compiled.get_property("EXECUTION_DEVICES"))
            with step("check_on_npu", report):
                request = compiled.create_infer_request()
                start = time.perf_counter()
                out = request.infer(feeds)[compiled.output(0)]
                report["npu_first_infer_ms"] = round((time.perf_counter() - start) * 1000)
                diff = np.abs(out[:, :width] - reference[:, :width].numpy())
                report["npu_vs_pytorch"] = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean())}
                print(report["npu_vs_pytorch"], report["npu_first_infer_ms"], "ms")
        report["status"] = "ok"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {str(exc)[:2000]}"
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
