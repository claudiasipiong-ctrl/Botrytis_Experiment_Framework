"""Metricas de deteccion: matching, P/R/F1, mAP y barridos de umbral (conf / NMS).

Las funciones base viven en ``utils.metrics``; aqui se exponen bajo el nombre de modulo de la
especificacion (sec. 30) y se anaden los barridos que usa ``11_tune_inference_thresholds.py``.

El matching usa IoU (sec. 25) y independiente del pairing GT (que solo crea etiquetas GT).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "ap_over_iou_table", "average_precision_detection", "conf_sweep", "counts_at", "detection_prf", "filter_by_conf", "map_over_iou", "match_detections",
    "pick_conf", "pipeline_confusion_matrix", "pipeline_error_rates",
]

from utils.metrics import (
    average_precision_detection, detection_prf, map_over_iou, match_detections, pipeline_confusion_matrix, pipeline_error_rates,
)

Preds = Sequence[tuple[np.ndarray, np.ndarray]]


def filter_by_conf(preds: Preds, conf: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Conserva las detecciones con ``score >= conf`` (por imagen)."""
    out = []
    for boxes, scores in preds:
        b = np.asarray(boxes, dtype=float).reshape(-1, 4)
        s = np.asarray(scores, dtype=float).reshape(-1)
        keep = s >= conf
        out.append((b[keep], s[keep]))
    return out


def counts_at(preds: Preds, gts: Sequence[np.ndarray], conf: float, iou_thr: float) -> dict[str, Any]:
    """TP/FP/FN, precision, recall y F1 con matching uno-a-uno a ``iou_thr`` tras filtrar por ``conf``."""
    tp = fp = fn = 0
    for (b, s), g in zip(filter_by_conf(preds, conf), gts):
        res = match_detections(b, s, np.asarray(g, dtype=float).reshape(-1, 4), iou_thr)
        tp += len(res["matches"])
        fp += len(res["fp"])
        fn += len(res["fn"])
    return {"conf": float(conf), "tp": tp, "fp": fp, "fn": fn, **detection_prf(tp, fp, fn)}


def conf_sweep(preds: Preds, gts: Sequence[np.ndarray], confs: Sequence[float], iou_thr: float) -> pd.DataFrame:
    """Tabla ``conf, tp, fp, fn, precision, recall, f1`` para cada umbral de confianza."""
    return pd.DataFrame([counts_at(preds, gts, c, iou_thr) for c in confs])


def ap_over_iou_table(preds: Preds, gts: Sequence[np.ndarray], thresholds: Sequence[float] | None = None) -> pd.DataFrame:
    """AP a cada umbral de IoU (0.50:0.05:0.95) -- figura ``AP vs IoU threshold``."""
    thr = list(thresholds) if thresholds is not None else [round(0.5 + 0.05 * i, 2) for i in range(10)]
    return pd.DataFrame({"iou": thr, "ap": [average_precision_detection(preds, gts, t) for t in thr]})


def pick_conf(sweep: pd.DataFrame, metric: str = "f1") -> tuple[float, float]:
    """Confianza que maximiza ``metric``; ante empate elige la mayor (menos FP)."""
    best = float(sweep[metric].max())
    tied = sweep.loc[np.isclose(sweep[metric], best, atol=1e-12), "conf"].to_numpy(float)
    return float(tied.max()), best
