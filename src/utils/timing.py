"""Temporizacion y memoria de GPU (sec. 29): sincronizacion CUDA, warm-up y estadisticas.

``time_callable`` implementa el protocolo de latencia: ``warmup`` llamadas descartadas y ``iterations``
medidas, con ``torch.cuda.synchronize()`` antes y despues de cada una (si ``sync`` y hay CUDA).
"""

from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence

import numpy as np


def cuda_available() -> bool:
    """``True`` si PyTorch ve una GPU CUDA."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def synchronize() -> None:
    """``torch.cuda.synchronize()`` si hay CUDA; no hace nada en CPU."""
    if cuda_available():
        import torch

        torch.cuda.synchronize()


def reset_peak_vram() -> None:
    """Reinicia el contador de pico de VRAM (asignada) del proceso."""
    if cuda_available():
        import torch

        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> float | None:
    """Pico de VRAM asignada del proceso en MB (``None`` sin CUDA)."""
    if not cuda_available():
        return None
    import torch

    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


def free_cuda_memory() -> None:
    """Libera la cache de CUDA y recolecta basura (entre folds/trials)."""
    import gc

    gc.collect()
    if cuda_available():
        import torch

        torch.cuda.empty_cache()


@contextmanager
def timed(sync: bool = True) -> Iterator[dict[str, float]]:
    """Mide un bloque en ms: ``with timed() as t: ...; t['ms']`` (sincroniza CUDA antes/despues)."""
    out: dict[str, float] = {}
    if sync:
        synchronize()
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        if sync:
            synchronize()
        out["ms"] = (time.perf_counter() - t0) * 1000.0


def summarize_ms(samples: Sequence[float]) -> dict[str, float | int | None]:
    """Media, SD, mediana, p95, min, max y FPS de una lista de latencias en ms."""
    a = np.asarray(list(samples), dtype=float)
    if a.size == 0:
        return {k: None for k in ("n", "mean_ms", "std_ms", "median_ms", "p95_ms", "min_ms", "max_ms", "fps")}
    mean = float(a.mean())
    return {
        "n": int(a.size), "mean_ms": mean, "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()),
        "max_ms": float(a.max()), "fps": (1000.0 / mean) if mean > 0 else None,
    }


def time_callable(fn: Callable[[int], Any], *, warmup: int, iterations: int, sync: bool = True) -> list[float]:
    """Latencias (ms) de ``fn(i)``: ``warmup`` llamadas descartadas + ``iterations`` medidas."""
    for i in range(warmup):
        fn(i)
    synchronize() if sync else None
    samples: list[float] = []
    for i in range(iterations):
        if sync:
            synchronize()
        t0 = time.perf_counter()
        fn(i)
        if sync:
            synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def gpu_compute_processes() -> list[dict[str, Any]]:
    """Procesos con contexto de computo en la GPU (via ``nvidia-smi``); ``[]`` si no hay o no se puede consultar."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False, encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return []
    procs: list[dict[str, Any]] = []
    if out.returncode != 0:
        return procs
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            procs.append({"pid": int(parts[0]), "name": parts[1], "used_mb": parts[2]})
    return procs
