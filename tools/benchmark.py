"""Measure the three modes on one local image: time, memory, GPU/NPU use, paging.

Windows only for the system counters (PDH). Image contents and model answers
other than compact numeric summaries are not written. No user name or local
path is recorded in the output.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image  # noqa: E402
from desktop_detector.backend import DEFAULT_MODEL_DIR, create_backend  # noqa: E402

MB = 2**20
REGION_PROMPT = "映っている猫をすべて囲む"
BINARY_PROMPT = "猫が映っているか？"
LOCATION_PROMPT = "映っている猫をすべて囲む"
CATEGORIES = ("魚がある", "猫がいる")


# ---------------------------------------------------------------- process memory
class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


_kernel32 = ctypes.WinDLL("kernel32")
_kernel32.GetCurrentProcess.restype = wintypes.HANDLE  # a 64-bit pseudo-handle, not a C int
_psapi = ctypes.WinDLL("psapi")
_psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)


def process_memory() -> dict:
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(pmc)
    if not _psapi.GetProcessMemoryInfo(_kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return {"working_set_mb": pmc.WorkingSetSize / MB, "peak_working_set_mb": pmc.PeakWorkingSetSize / MB,
            "private_mb": pmc.PrivateUsage / MB, "peak_commit_mb": pmc.PeakPagefileUsage / MB,
            "page_faults": pmc.PageFaultCount}


def system_memory() -> dict:
    status = _MemStatus()
    status.dwLength = ctypes.sizeof(status)
    _kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return {"total_mb": status.ullTotalPhys / MB, "available_mb": status.ullAvailPhys / MB,
            "load_percent": status.dwMemoryLoad}


# ---------------------------------------------------------------- PDH sampler
class _FmtValue(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]


class _FmtItem(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _FmtValue)]


PDH_FMT_DOUBLE = 0x200
PDH_FMT_NOCAP100 = 0x8000
PDH_MORE_DATA = 0x800007D2


class Sampler(threading.Thread):
    """1 s samples: this process's GPU/NPU engine use and memory, system paging."""
    COUNTERS = {
        "engine": r"\GPU Engine(*)\Utilization Percentage",
        "gpu_shared": r"\GPU Process Memory(*)\Shared Usage",
        "gpu_dedicated": r"\GPU Process Memory(*)\Dedicated Usage",
        "pages_input": r"\Memory\Pages Input/sec",       # includes reading the mapped model file
        "pages_output": r"\Memory\Pages Output/sec",     # pages written to the page file: memory pressure
        "pagefile_percent": r"\Paging File(_Total)\% Usage",
        "available_mb": r"\Memory\Available MBytes",
    }

    def __init__(self, interval: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples: list[dict] = []
        self.prefix = f"pid_{os.getpid()}_"
        self.error = ""

    def _array(self, pdh, handle) -> list[tuple[str, float]]:
        size, count = wintypes.DWORD(0), wintypes.DWORD(0)
        fmt = PDH_FMT_DOUBLE | PDH_FMT_NOCAP100
        status = pdh.PdhGetFormattedCounterArrayW(handle, fmt, ctypes.byref(size), ctypes.byref(count), None)
        if status & 0xFFFFFFFF != PDH_MORE_DATA:
            return []
        buffer = (ctypes.c_byte * size.value)()
        if pdh.PdhGetFormattedCounterArrayW(handle, fmt, ctypes.byref(size), ctypes.byref(count), buffer) != 0:
            return []
        items = ctypes.cast(buffer, ctypes.POINTER(_FmtItem))
        return [(items[i].szName, items[i].FmtValue.doubleValue) for i in range(count.value)
                if items[i].FmtValue.CStatus in (0, 1)]

    def run(self) -> None:
        try:
            pdh = ctypes.windll.pdh
            query = wintypes.HANDLE()
            if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
                raise OSError("PdhOpenQuery failed")
            handles = {}
            for key, path in self.COUNTERS.items():
                handle = wintypes.HANDLE()
                if pdh.PdhAddEnglishCounterW(query, path, 0, ctypes.byref(handle)) == 0:
                    handles[key] = handle
            pdh.PdhCollectQueryData(query)
            while not self.stop_event.wait(self.interval):
                pdh.PdhCollectQueryData(query)
                sample = {"t": time.perf_counter(), **process_memory()}
                engines: dict[str, float] = {}
                for name, value in self._array(pdh, handles.get("engine")) if "engine" in handles else []:
                    if name.startswith(self.prefix):
                        engtype = name.rsplit("engtype_", 1)[-1] or "other"
                        luid = name.split("_luid_", 1)[-1].split("_phys", 1)[0]
                        key = f"{engtype}@{luid[-5:]}"
                        engines[key] = engines.get(key, 0.0) + value
                sample["engines"] = engines
                for key in ("gpu_shared", "gpu_dedicated"):
                    if key in handles:
                        sample[key + "_mb"] = sum(v for n, v in self._array(pdh, handles[key])
                                                  if n.startswith(self.prefix)) / MB
                for key in ("pages_input", "pages_output", "pagefile_percent", "available_mb"):
                    if key in handles:
                        value = _FmtValue()
                        if pdh.PdhGetFormattedCounterValue(handles[key], PDH_FMT_DOUBLE, None, ctypes.byref(value)) == 0:
                            sample[key] = value.doubleValue
                self.samples.append(sample)
            pdh.PdhCloseQuery(query)
        except Exception as exc:  # counters are diagnostics, not a benchmark failure
            self.error = f"{type(exc).__name__}: {exc}"

    def stop(self) -> None:
        self.stop_event.set()
        self.join(5)

    def window(self, start: float, end: float) -> dict:
        rows = [s for s in self.samples if start <= s["t"] <= end + self.interval]
        if not rows:
            return {}
        engine_keys = sorted({k for s in rows for k in s.get("engines", {})})
        engines = {k: {"mean": round(statistics.mean(s.get("engines", {}).get(k, 0.0) for s in rows), 1),
                       "max": round(max(s.get("engines", {}).get(k, 0.0) for s in rows), 1)} for k in engine_keys}
        return {"samples": len(rows),
                "engine_util_percent": {k: v for k, v in engines.items() if v["max"] >= 1.0},
                "max_working_set_mb": round(max(s["working_set_mb"] for s in rows)),
                "max_private_mb": round(max(s["private_mb"] for s in rows)),
                "max_gpu_shared_mb": round(max(s.get("gpu_shared_mb", 0) for s in rows)),
                "max_gpu_dedicated_mb": round(max(s.get("gpu_dedicated_mb", 0) for s in rows)),
                "min_system_available_mb": round(min(s.get("available_mb", 0) for s in rows)),
                "pages_input_per_s_max": round(max(s.get("pages_input", 0) for s in rows), 1),
                "pages_input_total_est": round(sum(s.get("pages_input", 0) for s in rows) * self.interval),
                "pages_output_total_est": round(sum(s.get("pages_output", 0) for s in rows) * self.interval),
                "pagefile_percent_max": round(max(s.get("pagefile_percent", 0) for s in rows), 1)}


# ---------------------------------------------------------------- host info
def host_info() -> dict:
    script = ("$c=Get-CimInstance Win32_Processor|Select -First 1 Name;"
              "$g=Get-CimInstance Win32_VideoController|Select Name,DriverVersion;"
              "$n=Get-PnpDevice -FriendlyName 'Intel(R) AI Boost' -ErrorAction SilentlyContinue;"
              "$nv=if($n){($n|Get-PnpDeviceProperty DEVPKEY_Device_DriverVersion).Data};"
              "@{cpu=$c.Name;gpus=@($g);npu_driver=$nv}|ConvertTo-Json -Compress")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True,
                             text=True, timeout=60).stdout
        info = json.loads(out)
    except Exception:
        info = {}
    info.update({"os": platform.platform(), "python": platform.python_version(),
                 "ram_gb": round(system_memory()["total_mb"] / 1024, 1)})
    return info


def package_versions() -> dict:
    import importlib.metadata as md
    out = {}
    for name in ("torch", "torchvision", "transformers", "openvino", "nncf", "optimum-intel"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    return out


# ---------------------------------------------------------------- benchmark
def summarize_boxes(boxes) -> list[list[float]]:
    return [[round(b.left, 3), round(b.top, 3), round(b.right, 3), round(b.bottom, 3),
             None if b.confidence is None else round(b.confidence, 3)] for b in boxes]


def run_mode(backend, mode: str, source: Image.Image, max_edge: int) -> tuple[float, dict]:
    start = time.perf_counter()
    image = source.copy()  # same pre-processing as worker.py
    if max_edge:
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    if mode == "region":
        result = {"boxes": summarize_boxes(backend.predict(image, REGION_PROMPT, 8, 0.0))}
    elif mode == "binary":
        score, boxes = backend.predict_binary(image, BINARY_PROMPT, 0.20, LOCATION_PROMPT, 8)
        result = {"score": round(score, 4), "positive": score > 0.20, "boxes": summarize_boxes(boxes)}
    else:
        label, score, probs = backend.predict_choice(image, CATEGORIES)
        result = {"choice": label, "score": round(score, 4), "probabilities": {k: round(v, 4) for k, v in probs.items()}}
    return (time.perf_counter() - start) * 1000, result


def main() -> int:
    p = argparse.ArgumentParser(description="3モードの処理時間・メモリ・GPU/NPU使用率・ページングを計測")
    p.add_argument("image", type=Path, help="手元の画像（猫が写った写真を想定）")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "xpu", "mps", "npu"))
    p.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--runs", type=int, default=3, help="各モードの実行回数（1回目はwarm-up込み）")
    p.add_argument("--loops", type=int, default=1, help="3モードを順に繰り返す回数（連続実行の確認）")
    p.add_argument("--max-edge", type=int, default=768)
    p.add_argument("--output", type=Path, help="JSONの保存先")
    p.add_argument("--ov-device", default="NPU", choices=("NPU", "CPU"),
                   help="--device npu のOpenVINO実行先。CPUは変換結果の切り分け用で、NPU実行としては扱わない")
    args = p.parse_args()
    sampler = Sampler()
    sampler.start()
    out: dict = {"host": host_info(), "packages": package_versions(), "device_requested": args.device,
                 "image_size": None, "max_edge": args.max_edge, "status": "running"}
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    out["image_size"] = list(image.size)
    try:
        before = system_memory()
        t0 = time.perf_counter()
        backend = create_backend(args.device, args.model_dir, args.dtype, ov_device=args.ov_device)
        out["backend"] = backend.load()
        t1 = time.perf_counter()
        out["actual_device"] = backend.device
        expected = "openvino-cpu" if args.device == "npu" and args.ov_device == "CPU" else args.device
        if backend.device != expected:
            raise RuntimeError(f"要求 {expected} に対し実行先が {backend.device} です。")
        out["load"] = {"ms": round((t1 - t0) * 1000), "system_available_before_mb": round(before["available_mb"]),
                       **{k: round(v) for k, v in process_memory().items()}, "counters": sampler.window(t0, t1)}
        if args.device == "xpu":
            torch = backend.torch
            out["load"]["xpu_allocated_mb"] = round(torch.xpu.memory_allocated() / MB)
            torch.xpu.reset_peak_memory_stats()
        modes: dict[str, dict] = {m: {"ms": [], "results": []} for m in ("region", "binary", "choice")}
        faults_before = process_memory()["page_faults"]
        t2 = time.perf_counter()
        for loop in range(args.loops):
            for mode in modes:
                for _ in range(args.runs):
                    ws = time.perf_counter()
                    ms, result = run_mode(backend, mode, image, args.max_edge)
                    modes[mode]["ms"].append(round(ms))
                    if result not in modes[mode]["results"]:
                        modes[mode]["results"].append(result)
                    modes[mode].setdefault("windows", []).append((ws, time.perf_counter()))
            print(f"loop {loop + 1}/{args.loops} done", file=sys.stderr, flush=True)
        t3 = time.perf_counter()
        for mode, data in modes.items():
            windows = data.pop("windows")
            data["first_ms"] = data["ms"][0]
            warm = [ms for i, ms in enumerate(data["ms"]) if i % args.runs != 0] or data["ms"]
            data["warm_median_ms"] = round(statistics.median(warm))
            data["counters"] = sampler.window(windows[0][0], windows[-1][1])
        out["modes"] = modes
        out["inference"] = {"total_s": round(t3 - t2, 1), "frames": sum(len(m["ms"]) for m in modes.values()),
                            "process_page_faults": process_memory()["page_faults"] - faults_before,
                            "counters": sampler.window(t2, t3)}
        if args.device == "xpu":
            out["inference"]["xpu_peak_allocated_mb"] = round(backend.torch.xpu.max_memory_allocated() / MB)
            out["inference"]["xpu_peak_reserved_mb"] = round(backend.torch.xpu.max_memory_reserved() / MB)
        out["process_peak"] = {k: round(v) for k, v in process_memory().items()}
        out["status"] = "ok"
    except Exception as exc:
        import traceback
        out["status"] = "failed"
        out["error"] = f"{type(exc).__name__}: {exc}"
        # Keep only frames from this project / model code, without absolute paths.
        tb = traceback.extract_tb(exc.__traceback__)
        out["traceback"] = [f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in tb][-12:]
        out["process_peak"] = {k: round(v) for k, v in process_memory().items()}
    finally:
        sampler.stop()
        if sampler.error:
            out["sampler_error"] = sampler.error
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if out["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
