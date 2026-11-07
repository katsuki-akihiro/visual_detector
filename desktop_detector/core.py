"""Geometry, strict response parsing, and single-flight control (no GUI/ML imports)."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import math
from typing import Any

MODEL_ID = "LiquidAI/d1-2B"
DEFAULT_PROMPT = "映っている魚をすべて囲む"
MODE_REGION = "領域検出"
MODE_BINARY = "全体判定"
MODE_CHOICE = "全体推論"
MODES = (MODE_REGION, MODE_BINARY, MODE_CHOICE)
NONE_LABEL = "該当しない"
INTERVAL_CHOICES = ("0.5", "1.0", "2.0", "5.0", "10.0")


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("領域の幅・高さは正の値にしてください。")

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.left + self.width, self.top + self.height

    def overlaps(self, other: Region) -> bool:
        a, b, c, d = self.bbox
        e, f, g, h = other.bbox
        return a < g and e < c and b < h and f < d

    def inside(self, width: int, height: int) -> bool:
        return self.left >= 0 and self.top >= 0 and self.left + self.width <= width and self.top + self.height <= height


@dataclass(frozen=True)
class Detection:
    """Edges are fractions of the exact image sent to system_one()."""
    left: float
    top: float
    right: float
    bottom: float
    confidence: float | None = None

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (round(self.left * width), round(self.top * height),
                round(self.right * width), round(self.bottom * height))


class ResponseSchemaError(ValueError):
    """A changed/broken response must not be silently shown as zero detections."""


def finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ResponseSchemaError(f"{field}: 数値の代わりに bool が返りました。")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResponseSchemaError(f"{field}: 数値ではありません。") from exc
    if not math.isfinite(result):
        raise ResponseSchemaError(f"{field}: NaN/無限大です。")
    return result


def probability(value: Any, field: str) -> float:
    result = finite_number(value, field)
    if not 0.0 <= result <= 1.0:
        raise ResponseSchemaError(f"{field}: 0〜1の範囲外です。")
    return result


def make_questions(prompt: str, max_boxes: int = 8) -> dict:
    if not prompt.strip():
        raise ValueError("検出対象の指示を入力してください。")
    if isinstance(max_boxes, bool) or not isinstance(max_boxes, int) or not 1 <= max_boxes <= 16:
        raise ValueError("max_boxes は1〜16の整数にしてください。")
    return {"objects": {"type": "boxes", "instructions": prompt.strip(), "max": max_boxes}}


def make_binary_question(prompt: str) -> dict:
    if not prompt.strip():
        raise ValueError("判定する指示を入力してください。")
    return {"decision": {"type": "noul", "instructions": prompt.strip()}}


def validate_categories(lines: str | list[str]) -> tuple[str, ...]:
    categories = [line.strip() for line in (lines.splitlines() if isinstance(lines, str) else lines) if line.strip()]
    if not categories:
        raise ValueError("分類候補を1件以上入力してください。")
    if len(categories) > 15:
        raise ValueError("分類候補は15件以内にしてください。")
    if len(set(categories)) != len(categories) or NONE_LABEL in categories:
        raise ValueError("分類候補の重複と『該当しない』の入力はできません。")
    return tuple(categories)


def make_choice_question(categories: tuple[str, ...]) -> dict:
    categories = validate_categories(list(categories))
    criteria = {f"c{i}": label for i, label in enumerate(categories)}
    criteria["none"] = "上記のどれにも該当しない"
    return {"category": {"type": "choice", "instructions": "画像全体に最も合う説明を一つ選んでください。", "criteria": criteria}}


def parse_binary(raw: Any) -> float:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("answers"), Mapping):
        raise ResponseSchemaError("answers がありません。")
    answer = raw["answers"].get("decision")
    if not isinstance(answer, Mapping) or "noul" not in answer:
        raise ResponseSchemaError("answers.decision.noul がありません。")
    return probability(answer["noul"], "noul")


def parse_choice(raw: Any, categories: tuple[str, ...]) -> tuple[str, float, dict[str, float]]:
    categories = validate_categories(list(categories))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("answers"), Mapping):
        raise ResponseSchemaError("answers がありません。")
    answer = raw["answers"].get("category")
    scores = answer.get("probabilities") if isinstance(answer, Mapping) else None
    if not isinstance(scores, Mapping):
        raise ResponseSchemaError("answers.category.probabilities がありません。")
    names = {f"c{i}": label for i, label in enumerate(categories)}
    names["none"] = NONE_LABEL
    if set(scores) != set(names):
        raise ResponseSchemaError("分類候補のスコアが揃っていません。")
    probabilities = {names[key]: probability(scores[key], key) for key in names}
    label = max(probabilities, key=probabilities.__getitem__)
    return label, probabilities[label], probabilities


def validate_threshold(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("信頼度下限は0〜1の数値にしてください。")
    try:
        threshold = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("信頼度下限は0〜1の数値にしてください。") from exc
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("信頼度下限は0〜1の数値にしてください。")
    return threshold


def validate_interval(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("更新間隔は0.1〜60秒にしてください。")
    try:
        interval = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("更新間隔は0.1〜60秒にしてください。") from exc
    if not math.isfinite(interval) or not 0.1 <= interval <= 60:
        raise ValueError("更新間隔は0.1〜60秒にしてください。")
    return interval


@dataclass(frozen=True)
class InferenceSettings:
    mode: str
    prompt: str
    threshold: float
    max_boxes: int
    max_edge: int
    location_prompt: str = ""
    categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_threshold(self.threshold)
        if self.max_edge < 0:
            raise ValueError("推論前画像の長辺上限は0以上にしてください。")
        if self.mode == MODE_REGION:
            make_questions(self.prompt, self.max_boxes)
        elif self.mode == MODE_BINARY:
            make_binary_question(self.prompt)
            if self.location_prompt.strip():
                make_questions(self.location_prompt, self.max_boxes)
        elif self.mode == MODE_CHOICE:
            make_choice_question(self.categories)
        else:
            raise ValueError("モードを選択してください。")


def parse_boxes(raw: Any, threshold: float = 0.0) -> list[Detection]:
    """Read documented boxes answers; accept nested box or direct edge records.

    threshold filters *confidence* when present. It does not reinterpret
    confidence as P(present); the model's own found < .5 rule is independent.
    Empty boxes is valid. Missing keys or bad coordinates are errors, not negatives.
    """
    threshold = validate_threshold(threshold)
    if not isinstance(raw, Mapping):
        raise ResponseSchemaError("応答が辞書ではありません。")
    answers = raw.get("answers")
    if not isinstance(answers, Mapping) or not isinstance(answers.get("objects"), Mapping):
        raise ResponseSchemaError("answers.objects がありません。モデル/APIの版を確認してください。")
    answer = answers["objects"]
    boxes = answer.get("boxes")
    if not isinstance(boxes, list):
        raise ResponseSchemaError("answers.objects.boxes が配列ではありません。")
    if len(boxes) > 16:
        raise ResponseSchemaError("BBoxが16件を超えました。モデル/APIの版を確認してください。")
    parsed: list[Detection] = []
    for idx, entry in enumerate(boxes):
        if not isinstance(entry, Mapping):
            raise ResponseSchemaError(f"boxes[{idx}] が辞書ではありません。")
        found = probability(entry["found"], "found") if entry.get("found") is not None else None
        if found is not None and found < 0.5:
            continue
        box = entry.get("box", entry)
        if box is None:
            # Null is the model's missing-box representation.
            continue
        if not isinstance(box, Mapping) or not all(k in box for k in ("left", "top", "right", "bottom")):
            raise ResponseSchemaError(f"boxes[{idx}] の4辺がありません。")
        edges = [finite_number(box[k], k) for k in ("left", "top", "right", "bottom")]
        if any(v < -0.01 or v > 1.01 for v in edges):
            raise ResponseSchemaError("正規化BBoxの範囲外です。ピクセル座標とは解釈しません。")
        left, top, right, bottom = [min(1.0, max(0.0, v)) for v in edges]
        if left >= right or top >= bottom:
            raise ResponseSchemaError("BBoxの辺が逆転、または面積が0です。")
        conf = probability(entry["confidence"], "confidence") if entry.get("confidence") is not None else None
        if conf is not None and conf < threshold:
            continue
        parsed.append(Detection(left, top, right, bottom, conf))
    return parsed


def fit_image(image_width: int, image_height: int, canvas_width: int, canvas_height: int) -> tuple[int, int, int, int]:
    """Return display width, height, x offset, y offset; preserve aspect ratio."""
    if min(image_width, image_height, canvas_width, canvas_height) < 1:
        raise ValueError("画像・表示領域の寸法は正の値が必要です。")
    scale = min(canvas_width / image_width, canvas_height / image_height)
    w, h = max(1, round(image_width * scale)), max(1, round(image_height * scale))
    return w, h, (canvas_width - w) // 2, (canvas_height - h) // 2


class SingleFlight:
    """No queued frames. Stop invalidates a result already in flight."""
    def __init__(self) -> None:
        self.generation = 0
        self.running = False
        self.busy = False
        self.interval = 0.5
        self.next_due = 0.0

    def start(self, interval: float, now: float) -> None:
        if self.busy:
            raise RuntimeError("直前の処理が完了するまで開始できません。")
        interval = validate_interval(interval)
        self.generation += 1
        self.running = True
        self.interval = interval
        self.next_due = now

    def stop(self) -> None:
        self.running = False
        self.generation += 1

    def claim(self, now: float) -> int | None:
        if not self.running or self.busy or now < self.next_due:
            return None
        self.busy = True
        self.next_due = now + self.interval
        return self.generation

    def finish(self, generation: int) -> bool:
        self.busy = False
        return self.running and generation == self.generation
