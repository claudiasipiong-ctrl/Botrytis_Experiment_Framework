"""Utilidades comunes de entrenamiento: dispositivo, hash de pesos, resumenes de folds."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from utils.timing import cuda_available


def resolve_device(cfg: Mapping[str, Any]) -> str:
    """``cuda:0`` si hay GPU (``compute.device: auto``), ``cpu`` en caso contrario o si se fuerza."""
    want = str(cfg.get("compute", {}).get("device", "auto")).lower()
    if want in ("cpu",):
        return "cpu"
    if want in ("auto", ""):
        return "cuda:0" if cuda_available() else "cpu"
    return f"cuda:{want}" if want.isdigit() else want


def ultralytics_device(cfg: Mapping[str, Any]) -> Any:
    """Valor del argumento ``device`` de Ultralytics (``0`` | ``'cpu'``)."""
    dev = resolve_device(cfg)
    return "cpu" if dev == "cpu" else int(dev.split(":")[1]) if ":" in dev else dev


def state_dict_sha256(state: Mapping[str, Any]) -> str:
    """SHA-256 determinista de un ``state_dict`` (claves ordenadas + bytes de cada tensor en CPU)."""
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k]
        h.update(k.encode("utf-8"))
        if hasattr(t, "detach"):
            arr = t.detach().cpu().contiguous().numpy()
            h.update(str(arr.dtype).encode())
            h.update(arr.tobytes())
        else:
            h.update(repr(t).encode("utf-8"))
    return h.hexdigest()


def mean_sd(values: Sequence[float]) -> tuple[float | None, float | None]:
    """Media y desviacion estandar muestral (``ddof=1``; ``0.0`` si hay un solo valor)."""
    v = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not v:
        return None, None
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def is_oom(exc: BaseException) -> bool:
    """``True`` si la excepcion es una falta de memoria CUDA (o de Ultralytics/torch equivalente)."""
    name = type(exc).__name__
    return name == "OutOfMemoryError" or "out of memory" in str(exc).lower()
