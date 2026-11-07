"""Primary-screen capture, preserving the returned image's real pixel dimensions."""
from __future__ import annotations
import ctypes
import os
import sys
from PIL import Image, ImageGrab
from .core import Region


def enable_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()


def check_capture_session() -> None:
    if sys.platform.startswith("linux"):
        if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" or os.environ.get("WAYLAND_DISPLAY"):
            raise RuntimeError("このMVPのLinux画面取得はX11専用です。WaylandではX11セッションに切り替えてください。")
        if not os.environ.get("DISPLAY"):
            raise RuntimeError("DISPLAYがありません。デスクトップのターミナルから起動してください。")


def grab(region: Region) -> Image.Image:
    check_capture_session()
    kwargs = {"bbox": region.bbox}
    if sys.platform == "win32":
        kwargs["include_layered_windows"] = True
    if sys.platform.startswith("linux"):
        kwargs["xdisplay"] = ""  # No external screenshot-tool fallback.
    image = ImageGrab.grab(**kwargs).convert("RGB")
    if image.width < 1 or image.height < 1:
        raise RuntimeError("画面を取得できません。画面収録権限を確認してください。")
    # macOS Retina may return twice the logical size. The preview and normalized
    # boxes always use image.size, not the UI's logical monitor coordinates.
    if abs(image.width / image.height - region.width / region.height) > 0.05:
        raise RuntimeError("取得画像と選択領域の縦横比が一致しません。画面倍率・モニター構成を確認してください。")
    return image
