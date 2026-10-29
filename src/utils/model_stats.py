"""Parametros y FLOPs de los 10 modelos con el MISMO criterio (comparables entre familias).

* Parametros: suma de ``p.numel()`` (millones).
* FLOPs: ``torch.utils.flop_counter.FlopCounterMode`` sobre un forward con batch 1 (convoluciones,
  matmul/bmm, atencion). FLOPs = 2 x MACs. Se mide en CPU y en ``float32`` para que no dependa de la GPU.
  Las operaciones no matriciales (activaciones, NMS, top-k) no se cuentan: es una cota inferior comun.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)  # silencia el aviso de triton (no afecta a conv/matmul)


def count_params_m(module: torch.nn.Module) -> float:
    """Parametros totales en millones."""
    return float(sum(p.numel() for p in module.parameters()) / 1e6)


@torch.no_grad()
def count_gflops(module: torch.nn.Module, input_shape: tuple[int, int, int, int]) -> float:
    """GFLOPs (2 x MACs) de un forward con un tensor de ceros ``input_shape`` (N, C, H, W)."""
    from torch.utils.flop_counter import FlopCounterMode

    m = copy.deepcopy(module).float().cpu().eval()
    x = torch.zeros(*input_shape, dtype=torch.float32)
    with FlopCounterMode(display=False) as fc:
        m(x)
    return float(fc.get_total_flops() / 1e9)


def classifier_stats(module: torch.nn.Module, image_size: int = 224) -> dict[str, float]:
    """``{'params_m', 'gflops'}`` de un clasificador con entrada ``image_size``."""
    return {"params_m": count_params_m(module), "gflops": count_gflops(module, (1, 3, image_size, image_size))}


def ultralytics_stats(yolo: Any, image_size: int = 640) -> dict[str, float]:
    """``{'params_m', 'gflops'}`` de un modelo Ultralytics (``YOLO``/``RTDETR``) a ``image_size``."""
    inner = yolo.model
    return {"params_m": count_params_m(inner), "gflops": count_gflops(inner, (1, 3, image_size, image_size))}
