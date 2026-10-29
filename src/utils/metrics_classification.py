"""Metricas de clasificacion binaria (``con_signos`` = clase positiva) y barrido de umbral.

Las funciones base viven en ``utils.metrics`` (sin dependencias de modelos); aqui se exponen bajo el
nombre de modulo de la especificacion (sec. 30) y se anade el barrido de umbral de probabilidad,
que es **distinto de el pairing GT** (sec. 21.3).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "average_precision_np", "best_threshold", "binary_classification_report", "class_balance", "confusion_matrix_binary", "fbeta_score",
    "pick_threshold", "precision_recall_curve_np", "roc_auc_np", "roc_curve_np", "threshold_sweep",
]

from utils.metrics import (
    average_precision_np, best_threshold, binary_classification_report, class_balance, confusion_matrix_binary,
    fbeta_score, precision_recall_curve_np, roc_auc_np, roc_curve_np,
)


def threshold_sweep(y_true: Sequence[int], y_score: Sequence[float], thresholds: Sequence[float] | None = None, n_points: int = 101) -> pd.DataFrame:
    """Tabla ``threshold, precision, recall, f1, f2, accuracy, specificity, tp, fp, fn, tn`` para cada umbral.

    La prediccion es ``p >= umbral`` (mismo criterio que el pipeline, sec. 23).
    """
    yt = np.asarray(y_true).astype(int)
    ys = np.asarray(y_score, dtype=float)
    grid = np.linspace(0.0, 1.0, int(n_points)) if thresholds is None else np.asarray(list(thresholds), dtype=float)
    rows: list[dict[str, Any]] = []
    for t in grid:
        rep = binary_classification_report(yt, ys, threshold=float(t))
        rows.append({"threshold": float(t), "precision": rep["precision"], "recall": rep["recall"], "f1": rep["f1"], "f2": rep["f2"],
                     "accuracy": rep["accuracy"], "specificity": rep["specificity"], "tp": rep["tp"], "fp": rep["fp"], "fn": rep["fn"], "tn": rep["tn"]})
    return pd.DataFrame(rows)


def pick_threshold(sweep: pd.DataFrame, metric: str = "f1") -> tuple[float, float]:
    """Umbral que maximiza ``metric`` (``f1`` | ``f2``); ante empates, la mediana de los umbrales empatados.

    Returns:
        ``(umbral, valor_de_la_metrica)``.
    """
    if metric not in ("f1", "f2"):
        raise ValueError("classifier_threshold_metric debe ser 'f1' o 'f2'")
    best = float(sweep[metric].max())
    tied = sweep.loc[np.isclose(sweep[metric], best, atol=1e-12), "threshold"].to_numpy(float)
    return float(np.median(tied)), best
