"""Download only LiquidAI/d1-2B. Resolve a revision once and record it locally."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from desktop_detector.core import MODEL_ID
from desktop_detector.backend import DEFAULT_MODEL_DIR, verify_model_directory


def main() -> int:
    parser = argparse.ArgumentParser(description="公式d1-2Bをローカルに取得。モデル独自Pythonコードの実行許可が必要です。")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--revision", default="main", help="初回取得するbranch/tag/commit。解決したcommitをlockへ保存")
    parser.add_argument("--accept-model-code", action="store_true", help="LFMライセンスと独自Pythonコード利用を確認したうえで指定")
    args = parser.parse_args()
    if not args.accept_model_code:
        parser.error("公式モデルのLICENSE・Pythonコードを確認し、--accept-model-codeを指定してください。")
    try:
        if (args.output / "model.lock.json").exists():
            lock = verify_model_directory(args.output)
            if args.revision != "main" and args.revision != lock["revision"]:
                raise RuntimeError("既存モデルとは異なるrevisionです。別の--outputを指定してください。")
            print(f"既存の固定モデルを利用: {lock['revision']}\n{args.output.resolve()}")
            return 0
        from huggingface_hub import HfApi, snapshot_download
        revision = HfApi().model_info(MODEL_ID, revision=args.revision).sha
        if not revision:
            raise RuntimeError("モデルのcommitを取得できません。")
        print(f"取得元: {MODEL_ID}\n固定commit: {revision}\n保存先: {args.output.resolve()}")
        snapshot_download(repo_id=MODEL_ID, revision=revision, local_dir=str(args.output),
            allow_patterns=["*.py", "*.json", "*.jinja", "*.safetensors", "*.model", "LICENSE*", "README.md"])
        if not (args.output / "config.json").is_file() or not list(args.output.glob("*.safetensors")):
            raise RuntimeError("必須ファイルが取得できませんでした。")
        hashes = {}
        for file in sorted(args.output.rglob("*")):
            if file.is_file() and ".cache" not in file.parts and file.suffix in (".py", ".json", ".jinja"):
                hashes[file.relative_to(args.output).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
        lock = {"model_id": MODEL_ID, "revision": revision, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "code_sha256": hashes,
                "note": "Hashes record the downloaded code/config; they do not sandbox or audit that code."}
        (args.output / "model.lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("取得完了。以後、アプリのモデル読込はローカル限定です。\npython app.py")
        return 0
    except Exception as exc:
        print(f"取得に失敗: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
