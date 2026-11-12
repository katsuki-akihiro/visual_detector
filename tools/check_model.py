"""Run one local image through real d1-2B, without GUI or desktop permissions."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from desktop_detector.backend import D1Backend, DEFAULT_MODEL_DIR
from desktop_detector.core import DEFAULT_PROMPT, make_questions


def main() -> int:
    p = argparse.ArgumentParser(description="d1-2Bのモデル読込とBBox応答を、手元の画像1枚で確認")
    p.add_argument("image", type=Path)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--device", choices=("auto", "cuda", "xpu", "mps", "cpu"), default="auto")
    p.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    p.add_argument("--max-boxes", type=int, default=8)
    p.add_argument("--max-edge", type=int, default=768)
    p.add_argument("--runs", type=int, default=1, help="2回目以降がwarm計測。キャプチャ/描画は含まない")
    args = p.parse_args()
    try:
        make_questions(args.prompt, args.max_boxes)
        if args.runs < 1 or args.max_edge < 0:
            raise ValueError("runsは1以上、max-edgeは0以上にしてください。")
        with Image.open(args.image) as source:
            image = source.convert("RGB")
        if args.max_edge:
            image.thumbnail((args.max_edge, args.max_edge), Image.Resampling.LANCZOS)
        backend = D1Backend(args.model_dir, args.device, args.dtype)
        start = time.perf_counter()
        description = backend.load()
        load_ms = (time.perf_counter() - start) * 1000
        times = []
        boxes = []
        for _ in range(args.runs):
            start = time.perf_counter()
            boxes = backend.predict(image, args.prompt, args.max_boxes)
            times.append((time.perf_counter() - start) * 1000)
        out = {"backend": description, "input_size": image.size, "model_load_ms": round(load_ms, 1),
               "first_inference_ms": round(times[0], 1),
               "warm_median_ms": round(statistics.median(times[1:]), 1) if len(times) > 1 else None,
               "detections": [asdict(box) for box in boxes]}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"モデル検証失敗: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
