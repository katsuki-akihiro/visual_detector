"""Exercise the two whole-region modes with the installed local d1-2B model."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

from PIL import Image
from desktop_detector.backend import D1Backend


def main() -> None:
    parser = argparse.ArgumentParser(description="手元の猫画像で全体判定・全体推論を確認")
    parser.add_argument("image", type=Path, help="猫が写った画像のパス")
    args = parser.parse_args()
    model = D1Backend()
    model.load()
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    score, boxes = model.predict_binary(image, "Are there cats in the image?", 0.20, "Every visible cat.", 8)
    label, confidence, probabilities = model.predict_choice(image, ("魚がある", "猫がいる"))
    assert 0 <= score <= 1
    assert label in ("魚がある", "猫がいる", "該当しない")
    assert confidence == max(probabilities.values())
    print(json.dumps({"binary_score": score, "binary_positive": score > .2,
                      "localized_boxes": len(boxes), "choice": label,
                      "choice_score": confidence, "probabilities": probabilities}, ensure_ascii=False))


if __name__ == "__main__":
    main()
