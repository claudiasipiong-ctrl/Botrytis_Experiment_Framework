"""Metricas reutilizables (solo NumPy/pandas; sin dependencia de modelos).

Incluye:

* estadisticas descriptivas y de balance del dataset (sec. 4.2);
* metricas binarias de clasificacion con ``con_signos`` como clase positiva (sec. 13.2, 28.2);
* matching deteccion-GT one-to-one, precision/recall/F1 y mAP (sec. 13.1, 25, 28.1);
* errores del pipeline E_det, E_clas, E_flujo (sec. 28.3-28.5).

El umbral/metrica de *matching* es TODO BLOQUEANTE (sec. 25): estas funciones reciben el
umbral como argumento explicito y no asumen ningun valor por defecto metodologico.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from utils.boxes import iou_matrix


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def describe_series(values: Sequence[float] | pd.Series) -> dict[str, float | None]:
    """Resumen estadistico robusto (``None`` si la serie esta vacia)."""
    s = pd.Series(values, dtype="float64").dropna()
    if s.empty:
        return {k: None for k in ("count", "mean", "std", "min", "p25", "median", "p75", "max")}
    return {
        "count": float(len(s)), "mean": float(s.mean()), "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "min": float(s.min()), "p25": float(s.quantile(0.25)), "median": float(s.median()),
        "p75": float(s.quantile(0.75)), "max": float(s.max()),
    }


def class_balance(
    labels: Sequence[str] | pd.Series,
    positive_label: str = "con_signos",
    negative_label: str = "sin_signos",
) -> dict[str, Any]:
    """Totales, porcentajes y razon de desbalance (mayoritaria / minoritaria).

    ``imbalance_ratio`` es ``None`` si alguna clase tiene 0 ejemplos.
    """
    s = pd.Series(labels)
    n_pos = int((s == positive_label).sum())
    n_neg = int((s == negative_label).sum())
    total = n_pos + n_neg
    lo, hi = min(n_pos, n_neg), max(n_pos, n_neg)
    return {
        "total": total, positive_label: n_pos, negative_label: n_neg,
        f"pct_{positive_label}": (100.0 * n_pos / total) if total else None,
        f"pct_{negative_label}": (100.0 * n_neg / total) if total else None,
        "imbalance_ratio": (hi / lo) if lo > 0 else None,
        "majority_class": positive_label if n_pos >= n_neg else negative_label,
    }


def classification_dataset_stats(
    fruit_table: pd.DataFrame,
    image_summary: pd.DataFrame,
    positive_label: str = "con_signos",
    negative_label: str = "sin_signos",
) -> dict[str, Any]:
    """Estadisticas de la sec. 4.2: totales, %, desbalance, frutos/crops por imagen (+ por split)."""
    stats: dict[str, Any] = {
        "n_images": int(len(image_summary)),
        "n_fruits_total": int(len(fruit_table)),
        "n_signs_total": int(image_summary["n_signs"].sum()) if "n_signs" in image_summary else None,
        "class_balance": class_balance(fruit_table["label"], positive_label, negative_label),
        "fruits_per_image": describe_series(image_summary["n_fruits"]),
        "positive_crops_per_image": describe_series(image_summary["n_positive"]),
        "pct_positive_crops_per_image": describe_series(image_summary["pct_positive"]),
        "images_with_positive_signs": int(image_summary["has_positive_signs"].sum()),
        "images_without_fruits": int((image_summary["n_fruits"] == 0).sum()),
        "max_sign_coverage": describe_series(fruit_table["max_sign_coverage"]),
    }
    if "outer_split" in fruit_table.columns:
        per_split: dict[str, Any] = {}
        for split, grp in fruit_table.groupby("outer_split"):
            per_split[str(split)] = {
                "n_images": int(grp["image_id"].nunique()),
                "n_fruits": int(len(grp)),
                "class_balance": class_balance(grp["label"], positive_label, negative_label),
            }
        stats["per_split"] = per_split
    return stats


# --------------------------------------------------------------------------- #
# Clasificacion binaria
# --------------------------------------------------------------------------- #
def confusion_matrix_binary(y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    """Matriz ``[[TN, FP], [FN, TP]]`` (positivo = 1)."""
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tp = int(((yt == 1) & (yp == 1)).sum())
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def fbeta_score(precision: float, recall: float, beta: float = 1.0) -> float:
    """F-beta a partir de precision y recall (``0`` si ambos son 0). ``beta=2`` da F2."""
    b2 = beta * beta
    return _safe_div((1 + b2) * precision * recall, b2 * precision + recall)


def roc_curve_np(y_true: Sequence[int], y_score: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curva ROC ``(fpr, tpr, thresholds)`` sin dependencias externas."""
    yt = np.asarray(y_true).astype(int)
    ys = np.asarray(y_score, dtype=float)
    order = np.argsort(-ys, kind="mergesort")
    yt, ys = yt[order], ys[order]
    distinct = np.where(np.diff(ys))[0]
    idx = np.r_[distinct, len(ys) - 1]
    tps = np.cumsum(yt)[idx]
    fps = (1 + idx) - tps
    p, n = yt.sum(), len(yt) - yt.sum()
    tpr = np.r_[0.0, tps / p] if p else np.zeros(len(idx) + 1)
    fpr = np.r_[0.0, fps / n] if n else np.zeros(len(idx) + 1)
    return fpr, tpr, np.r_[np.inf, ys[idx]]


def roc_auc_np(y_true: Sequence[int], y_score: Sequence[float]) -> float | None:
    """ROC-AUC (Mann-Whitney con empates). ``None`` si falta alguna clase."""
    yt = np.asarray(y_true).astype(int)
    if not len(yt) or yt.min() == yt.max():
        return None
    fpr, tpr, _ = roc_curve_np(yt, y_score)
    return float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))


def precision_recall_curve_np(
    y_true: Sequence[int], y_score: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curva PR ``(precision, recall, thresholds)`` (recall creciente, como scikit-learn)."""
    yt = np.asarray(y_true).astype(int)
    ys = np.asarray(y_score, dtype=float)
    order = np.argsort(-ys, kind="mergesort")
    yt, ys = yt[order], ys[order]
    distinct = np.where(np.diff(ys))[0]
    idx = np.r_[distinct, len(ys) - 1]
    tps = np.cumsum(yt)[idx].astype(float)
    fps = (1 + idx) - tps
    precision = tps / (tps + fps)
    total_pos = yt.sum()
    recall = tps / total_pos if total_pos else np.zeros_like(tps)
    return np.r_[precision[::-1], 1.0], np.r_[recall[::-1], 0.0], ys[idx][::-1]


def average_precision_np(y_true: Sequence[int], y_score: Sequence[float]) -> float | None:
    """Average Precision (PR-AUC escalonada, definicion de scikit-learn). ``None`` sin positivos."""
    yt = np.asarray(y_true).astype(int)
    if yt.sum() == 0:
        return None
    precision, recall, _ = precision_recall_curve_np(yt, y_score)
    return float(-np.sum(np.diff(recall) * precision[:-1]))


def binary_classification_report(
    y_true: Sequence[int],
    y_score: Sequence[float] | None = None,
    *,
    threshold: float = 0.5,
    y_pred: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Metricas de la sec. 13.2/28.2 con ``con_signos`` = clase positiva (1).

    Args:
        y_true: Etiquetas 0/1.
        y_score: ``P(con_signos)``. Necesaria para ROC-AUC y AP.
        threshold: Umbral de probabilidad (NO es el pairing GT; sec. 21.3).
        y_pred: Predicciones duras; si se omite se derivan de ``y_score >= threshold``.
    """
    yt = np.asarray(y_true).astype(int)
    if y_pred is None:
        if y_score is None:
            raise ValueError("Se requiere y_score o y_pred")
        yp = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    else:
        yp = np.asarray(y_pred).astype(int)
    cm = confusion_matrix_binary(yt, yp)
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    report: dict[str, Any] = {
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": fbeta_score(precision, recall, 1.0),
        "f2": fbeta_score(precision, recall, 2.0),
        "specificity": _safe_div(tn, tn + fp),
        "confusion_matrix": cm.tolist(),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "threshold": float(threshold),
        "n": int(len(yt)),
        "n_positive": int(yt.sum()),
    }
    per_class = {}
    for label, a, b, c in (("sin_signos", tn, fn, fp), ("con_signos", tp, fp, fn)):
        cp, cr = _safe_div(a, a+b), _safe_div(a, a+c)
        per_class[label] = {"precision": cp, "recall": cr, "f1-score": fbeta_score(cp,cr,1.0), "support": a+c}
    report["classification_report"] = {**per_class, "accuracy": report["accuracy"],
        "macro avg": {k: sum(r[k] for r in per_class.values())/2 for k in ("precision","recall","f1-score")}}
    report["classification_report"]["macro avg"]["support"] = len(yt)
    report["classification_report"]["weighted avg"] = {
        k: _safe_div(sum(r[k]*r["support"] for r in per_class.values()), len(yt)) for k in ("precision","recall","f1-score")}
    report["classification_report"]["weighted avg"]["support"] = len(yt)
    if y_score is not None:
        report["roc_auc"] = roc_auc_np(yt, y_score)
        report["average_precision"] = average_precision_np(yt, y_score)
        report["average_precision_positive"] = report["average_precision"]
        if len(yt) and 0 < yt.sum() < len(yt):
            fpr, tpr, _ = roc_curve_np(yt,y_score)
            report["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        else:
            report["roc"] = None
    return report


def best_threshold(
    y_true: Sequence[int], y_score: Sequence[float], metric: str = "f1", beta: float = 1.0
) -> tuple[float, float]:
    """Umbral que maximiza F-beta sobre ``y_score`` (usar SOLO en validation externo, sec. 21).

    Returns:
        ``(umbral, valor_de_la_metrica)``.
    """
    if metric not in ("f1", "fbeta"):
        raise ValueError("metric debe ser 'f1' o 'fbeta'")
    b = 1.0 if metric == "f1" else beta
    ys = np.asarray(y_score, dtype=float)
    best_t, best_v = 0.5, -1.0
    for t in np.unique(ys):
        rep = binary_classification_report(y_true, ys, threshold=float(t))
        v = fbeta_score(rep["precision"], rep["recall"], b)
        if v > best_v:
            best_t, best_v = float(t), float(v)
    return best_t, best_v


# --------------------------------------------------------------------------- #
# Deteccion
# --------------------------------------------------------------------------- #
def match_detections(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float,
) -> dict[str, Any]:
    """Matching greedy one-to-one por score descendente con ``IoU >= iou_threshold``.

    Las predicciones sobrantes (duplicados o sin GT) son FP de localizacion; los GT sin pareja
    son FN. El matching de detecciones es independiente del pairing de anotaciones.

    Returns:
        ``{"matches": [(pred_idx, gt_idx, iou)], "fp": [pred_idx], "fn": [gt_idx]}``.
    """
    pb = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    gb = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    ps = np.asarray(pred_scores, dtype=float).reshape(-1)
    order = np.argsort(-ps, kind="mergesort")
    ious = iou_matrix(pb, gb)
    taken: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    fps: list[int] = []
    for pi in order:
        best_gt, best_iou = -1, 0.0
        for gi in range(len(gb)):
            if gi in taken:
                continue
            if ious[pi, gi] >= iou_threshold and ious[pi, gi] > best_iou:
                best_gt, best_iou = gi, float(ious[pi, gi])
        if best_gt >= 0:
            taken.add(best_gt)
            matches.append((int(pi), best_gt, best_iou))
        else:
            fps.append(int(pi))
    fns = [g for g in range(len(gb)) if g not in taken]
    return {"matches": sorted(matches), "fp": sorted(fps), "fn": fns}


def detection_prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision, recall y F1 a partir de conteos."""
    p, r = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    return {"precision": p, "recall": r, "f1": fbeta_score(p, r, 1.0)}


def average_precision_detection(
    preds: Sequence[tuple[np.ndarray, np.ndarray]],
    gts: Sequence[np.ndarray],
    iou_threshold: float,
) -> float:
    """AP a un IoU dado, una sola clase, interpolacion COCO de 101 puntos de recall.

    Un detector perfecto da exactamente 1.0. Ultralytics integra con trapecios y reporta
    ~0.995 en ese caso, por lo que sus mAP pueden diferir ligeramente (~0.005) de estos.

    Args:
        preds: Por imagen ``(boxes (N,4), scores (N,))``.
        gts: Por imagen ``boxes (M,4)``.
        iou_threshold: Umbral de IoU del matching TP/FP.
    """
    n_gt = int(sum(len(np.asarray(g).reshape(-1, 4)) for g in gts))
    if n_gt == 0:
        return 0.0
    records: list[tuple[float, int]] = []
    for (pb, ps), gb in zip(preds, gts):
        pb = np.asarray(pb, dtype=float).reshape(-1, 4)
        ps = np.asarray(ps, dtype=float).reshape(-1)
        gb = np.asarray(gb, dtype=float).reshape(-1, 4)
        res = match_detections(pb, ps, gb, iou_threshold)
        tp_idx = {m[0] for m in res["matches"]}
        records.extend((float(ps[i]), 1 if i in tp_idx else 0) for i in range(len(ps)))
    if not records:
        return 0.0
    records.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records], dtype=float)
    fp = np.cumsum([1 - r[1] for r in records], dtype=float)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-16)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]       # precision monotona decreciente
    thresholds = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, thresholds, side="left")         # primer punto con recall >= umbral
    sampled = np.zeros_like(thresholds)
    valid = idx < len(envelope)
    sampled[valid] = envelope[idx[valid]]                          # recall no alcanzado => precision 0
    return float(sampled.mean())


def map_over_iou(
    preds: Sequence[tuple[np.ndarray, np.ndarray]],
    gts: Sequence[np.ndarray],
    thresholds: Sequence[float] | None = None,
) -> dict[str, float]:
    """``mAP@0.5`` y ``mAP@0.5:0.95`` (paso 0.05) para deteccion de una clase."""
    thr = list(thresholds) if thresholds is not None else [round(0.5 + 0.05 * i, 2) for i in range(10)]
    aps = [average_precision_detection(preds, gts, t) for t in thr]
    ap50 = average_precision_detection(preds, gts, 0.5) if 0.5 not in thr else aps[thr.index(0.5)]
    return {"map50": float(ap50), "map50_95": float(np.mean(aps))}


# --------------------------------------------------------------------------- #
# Pipeline integrado (sec. 28)
# --------------------------------------------------------------------------- #
def pipeline_error_rates(n_gt: int, n_fn: int, n_ec: int, n_matched: int) -> dict[str, float | None]:
    """GT-normalized error decomposition. Undefined denominators return None.

    Legacy e_clas/e_flujo keys are aliases for report compatibility; FP never
    enter these numerators. Conditional classification error is NA if no match.
    """
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 0 for x in (n_gt,n_fn,n_ec,n_matched)):
        raise ValueError("Counts must be nonnegative integers")
    if n_fn + n_matched != n_gt or n_ec > n_matched:
        raise ValueError("GT must partition into misses and one-to-one matches; errors <= matches")
    return {
        "E_det": n_fn/n_gt if n_gt else None,
        "E_cls": n_ec/n_matched if n_matched else None,
        "E_cls_cond": n_ec/n_matched if n_matched else None,
        "E_cls_abs": n_ec/n_gt if n_gt else None,
        "E_global": (n_fn+n_ec)/n_gt if n_gt else None,
        "N_GT": int(n_gt), "N_miss": int(n_fn), "N_cls_error": int(n_ec), "N_match": int(n_matched),
        "e_det": (n_fn / n_gt) if n_gt else None,
        "e_clas": (n_ec / n_matched) if n_matched else None,
        "e_flujo": ((n_fn + n_ec) / n_gt) if n_gt else None,
        "n_gt": int(n_gt), "n_fn": int(n_fn), "n_ec": int(n_ec), "n_matched": int(n_matched),
    }


def pipeline_confusion_matrix(
    gt_labels_matched: Sequence[str],
    pred_labels_matched: Sequence[str],
    n_fn_by_gt_label: Mapping[str, int],
    fp_pred_labels: Sequence[str],
    classes: Sequence[str] = ("con_signos", "sin_signos"),
) -> np.ndarray:
    """Matriz de errores del pipeline, ``(K+1) x (K+1)`` con ``K = len(classes)``.

    * Filas = verdad: las ``K`` clases GT + ``fondo`` (detecciones sin GT valido).
    * Columnas = salida del pipeline: las ``K`` clases predichas + ``sin deteccion``.

    Celdas:

    * ``[gt_i, pred_j]``: frutos GT asociados a una deteccion (aciertos en la diagonal,
      errores de clasificacion fuera de ella);
    * ``[gt_i, sin deteccion]``: FN de deteccion de esa clase GT;
    * ``[fondo, pred_j]``: FP de localizacion (deteccion sin GT valido o duplicada) con la
      clase que el clasificador le asigno;
    * ``[fondo, sin deteccion]``: no aplica (siempre 0).
    """
    k = len(classes)
    cm = np.zeros((k + 1, k + 1), dtype=int)
    idx = {c: i for i, c in enumerate(classes)}
    for g, p in zip(gt_labels_matched, pred_labels_matched):
        cm[idx[g], idx[p]] += 1
    for c, n in n_fn_by_gt_label.items():
        cm[idx[c], k] += int(n)
    for p in fp_pred_labels:
        cm[k, idx[p]] += 1
    return cm
