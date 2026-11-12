"""Low-frequency, flushed resource telemetry for diagnosing interrupted runs."""

import csv
import ctypes
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import subprocess
import threading
from time import monotonic

import psutil
import torch

log = logging.getLogger(__name__)
MIB = 1024 ** 2
FIELDS = (
    "timestamp_utc", "session_pid", "elapsed_s", "phase", "episode", "plan_call",
    "system_ram_total_mib", "system_ram_available_mib", "system_ram_percent",
    "system_commit_used_mib", "system_commit_limit_mib", "process_rss_mib", "process_private_mib",
    "torch_allocated_mib", "torch_reserved_mib", "torch_peak_allocated_mib", "torch_peak_reserved_mib",
    "gpu_memory_used_mib", "gpu_memory_total_mib", "gpu_utilization_percent", "gpu_temperature_c",
    "gpu_power_w", "sampling_error",
)


def windows_commit_memory():
    if os.name != "nt":
        return None, None

    class PerformanceInfo(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in (
                "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
                "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize",
            )
        ] + [(name, ctypes.c_ulong) for name in ("HandleCount", "ProcessCount", "ThreadCount")]

    info = PerformanceInfo()
    info.cb = ctypes.sizeof(info)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
        raise OSError("GetPerformanceInfo failed")
    return info.CommitTotal * info.PageSize / MIB, info.CommitLimit * info.PageSize / MIB


def gpu_stats():
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    result = subprocess.run(
        ["nvidia-smi", "--id=" + gpu_id,
         "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=3, check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    values = result.stdout.strip().splitlines()[0].split(",")
    keys = ("gpu_memory_used_mib", "gpu_memory_total_mib", "gpu_utilization_percent", "gpu_temperature_c", "gpu_power_w")
    if len(values) != len(keys):
        raise ValueError("Unexpected nvidia-smi output")
    return {key: float(value.strip()) if value.strip() not in ("N/A", "[N/A]", "[Not Supported]") else ""
            for key, value in zip(keys, values)}


class ResourceMonitor:
    def __init__(self, folder, interval_s=None):
        if interval_s is not None and interval_s <= 0:
            raise ValueError("resource_monitor_interval_s must be positive or null")
        self.interval_s = interval_s
        self.path = Path(folder) / "resource_usage.csv"
        self.started = monotonic()
        self.process = psutil.Process()
        self.context = {"phase": "startup", "episode": "", "plan_call": ""}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

    def set_context(self, phase, episode="", plan_call=""):
        with self.lock:
            self.context = dict(phase=phase, episode=episode, plan_call=plan_call)

    def sample(self):
        with self.lock:
            row = dict(self.context)
        row.update(timestamp_utc=datetime.now(timezone.utc).isoformat(), session_pid=os.getpid(),
                   elapsed_s=round(monotonic() - self.started, 3))
        errors = []
        try:
            ram, process = psutil.virtual_memory(), self.process.memory_info()
            row.update(system_ram_total_mib=ram.total / MIB, system_ram_available_mib=ram.available / MIB,
                       system_ram_percent=ram.percent, process_rss_mib=process.rss / MIB,
                       process_private_mib=getattr(process, "private", process.vms) / MIB)
            used, limit = windows_commit_memory()
            row.update(system_commit_used_mib=used, system_commit_limit_mib=limit)
        except Exception as exc:
            errors.append("ram: " + str(exc))
        # Avoid initializing CUDA from this background thread during startup.
        if torch.cuda.is_initialized():
            try:
                row.update(torch_allocated_mib=torch.cuda.memory_allocated(0) / MIB,
                           torch_reserved_mib=torch.cuda.memory_reserved(0) / MIB,
                           torch_peak_allocated_mib=torch.cuda.max_memory_allocated(0) / MIB,
                           torch_peak_reserved_mib=torch.cuda.max_memory_reserved(0) / MIB)
            except Exception as exc:
                errors.append("torch: " + str(exc))
        try:
            row.update(gpu_stats())
        except Exception as exc:
            errors.append("gpu: " + str(exc))
        row["sampling_error"] = "; ".join(errors)
        return row

    def _write_sample(self, handle, writer):
        writer.writerow(self.sample())
        handle.flush()
        os.fsync(handle.fileno())

    def _run(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            header = not self.path.exists() or self.path.stat().st_size == 0
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                if header:
                    writer.writeheader()
                while not self.stop_event.is_set():
                    self._write_sample(handle, writer)
                    self.stop_event.wait(self.interval_s)
                self._write_sample(handle, writer)
        except Exception:
            # Monitoring failure should not invalidate or kill an evaluation.
            log.exception("Resource monitoring stopped; evaluation continues")

    def __enter__(self):
        if self.interval_s is not None:
            self.thread = threading.Thread(target=self._run, daemon=True, name="resource-monitor")
            self.thread.start()
            log.info("Resource telemetry every %s seconds: %s", self.interval_s, self.path)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.thread is not None:
            self.set_context("failed" if exc_type else "finished")
            self.stop_event.set()
            self.thread.join(timeout=8)
