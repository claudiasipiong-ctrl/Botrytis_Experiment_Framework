"""Evaluacion del pipeline integrado detector + clasificador (sec. 23-25, 28).

Se usa en ``14_validate_integrated_pipeline.py`` (validation externo, PREDICTED crops) y en
``16_evaluate_test.py`` (test, una sola vez). Cada caja predicha se asocia a un fruto GT por IoU
(matching uno-a-uno por confianza; independiente del pairing GT). Errores (sec. 28):

* ``E_det = FN / N_GT``        ``E_clas = EC / N_matched``        ``E_flujo = (FN + EC) / N_GT``
* detecciones sin GT valido o duplicadas = errores de localizacion (FP), reportados aparte.

El resumen se calcula con :func:`summarize_from_table` a partir de la tabla de detecciones, de modo que
es **reproducible sin modelos** (``pipeline_detections.csv`` + ``pipeline_misses.csv``) y permite
reevaluar el umbral del clasificador sobre los mismos crops predichos.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING
if TYPE_CHECKING:
    import torch

import numpy as np
import pandas as pd
from PIL import Image

from utils.boxes import iou_matrix
from utils.io import load_image_rgb
from utils.metrics import binary_classification_report, match_detections, pipeline_confusion_matrix, pipeline_error_rates

DET_COLUMNS = ["image_id", "det_id", "x1", "y1", "x2", "y2", "det_conf", "p_con_signos", "matched_fruit_id", "gt_label", "iou", "outcome", "x_norm", "y_norm"]
MISS_COLUMNS = ["image_id", "fruit_id", "gt_label", "x1", "y1", "x2", "y2", "x_norm", "y_norm"]


@dataclass
class PipelineResult:
    """Resultados completos de una evaluacion de pipeline."""

    summary: dict[str, Any]
    detections: pd.DataFrame
    misses: pd.DataFrame
    per_image: pd.DataFrame
    confusion: np.ndarray
    examples: list[dict[str, Any]] = field(default_factory=list)
    errors: pd.DataFrame = field(default_factory=pd.DataFrame)


def gt_from_fruit_table(fruits: pd.DataFrame, image_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """``image_id -> {'boxes': (n,4), 'labels': [...], 'fruit_ids': [...], 'w', 'h'}`` desde la tabla GT de frutos."""
    out: dict[str, dict[str, Any]] = {}
    groups = {k: g.sort_values("fruit_id") for k, g in fruits.groupby("image_id")}
    for iid in image_ids:
        g = groups.get(iid)
        if g is None:
            out[iid] = {"boxes": np.zeros((0, 4)), "labels": [], "fruit_ids": [], "w": None, "h": None}
        else:
            out[iid] = {"boxes": g[["x1", "y1", "x2", "y2"]].to_numpy(float), "labels": g["label"].tolist(),
                        "fruit_ids": g["fruit_id"].astype(int).tolist(), "w": int(g["image_w"].iloc[0]), "h": int(g["image_h"].iloc[0])}
    return out


def summarize_from_table(
    det: pd.DataFrame, misses: pd.DataFrame, *, n_images: int, pos: str, neg: str, op: Mapping[str, float],
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """Metricas del pipeline desde las tablas de detecciones y FN (reproducible sin modelos).

    Args:
        det: Detecciones (``DET_COLUMNS``); ``matched_fruit_id`` nulo = deteccion sin GT valido.
        misses: Frutos GT sin deteccion (``MISS_COLUMNS``).
        op: Punto de operacion: ``det_conf``, ``nms_iou``, ``cls_threshold``, ``match_iou``.

    Returns:
        ``(resumen, tabla de detecciones con pred_label/outcome recalculados, matriz de errores)``.
    """
    thr = float(op["cls_threshold"])
    d = det.copy()
    if d.duplicated(["image_id", "det_id"]).any():
        raise ValueError("Detection counted multiple times; matching must be one-to-one")
    d["pred_label"] = np.where(d["p_con_signos"].astype(float) >= thr, pos, neg) if len(d) else []
    matched = d[d["matched_fruit_id"].notna()]
    unmatched = d[d["matched_fruit_id"].isna()]
    if matched.duplicated(["image_id", "matched_fruit_id"]).any() or misses.duplicated(["image_id", "fruit_id"]).any():
        raise ValueError("GT counted multiple times; matching must be one-to-one")
    if set(zip(matched["image_id"], matched["matched_fruit_id"])) & set(zip(misses["image_id"], misses["fruit_id"])):
        raise ValueError("GT cannot be both matched and missed")
    if not matched["gt_label"].isin([pos,neg]).all() or not misses["gt_label"].isin([pos,neg]).all():
        raise ValueError("Unresolved GT labels cannot enter integrated classification evaluation")
    if len(d):
        d.loc[matched.index, "outcome"] = np.where(matched["pred_label"] == matched["gt_label"], "TP", "EC")
    tp, fp, fn = len(matched), len(unmatched), len(misses)
    n_dup = int((unmatched["outcome"] == "FP_duplicate").sum()) if len(unmatched) else 0
    n_gt = tp + fn
    n_ec = int((matched["pred_label"] != matched["gt_label"]).sum()) if len(matched) else 0
    rates = pipeline_error_rates(n_gt, fn, n_ec, tp)
    det_p = tp / (tp + fp) if (tp + fp) else 0.0
    det_r = tp / (tp + fn) if (tp + fn) else 0.0
    y_true = (matched["gt_label"] == pos).astype(int).to_numpy() if len(matched) else np.zeros(0, dtype=int)
    cls_rep = binary_classification_report(y_true, matched["p_con_signos"].to_numpy(float), threshold=thr) if len(matched) else {}
    fn_by = {pos: int((misses["gt_label"] == pos).sum()) if len(misses) else 0, neg: int((misses["gt_label"] == neg).sum()) if len(misses) else 0}
    cm = pipeline_confusion_matrix(matched["gt_label"].tolist(), matched["pred_label"].tolist(), fn_by, d.loc[unmatched.index, "pred_label"].tolist() if len(unmatched) else [],
                                   classes=(pos, neg))
    summary = {
        "n_images": int(n_images), "n_gt": int(n_gt), "n_pred": int(len(d)), "tp": tp, "fp": fp, "fn": fn, "n_duplicate_fp": n_dup, "n_no_gt_fp": fp - n_dup,
        "detection": {"precision": det_p, "recall": det_r, "f1": (2 * det_p * det_r / (det_p + det_r)) if (det_p + det_r) else 0.0},
        "classification_on_matched": cls_rep, "errors": rates, "operating_point": {k: float(v) for k, v in op.items()},
        "pipeline_confusion_matrix": cm.tolist(), "pipeline_confusion_rows": [pos, neg, "fondo"], "pipeline_confusion_cols": [f"pred {pos}", f"pred {neg}", "sin deteccion"],
    }
    return summary, d, cm


def _classify_timed(model: torch.nn.Module, crops: Sequence[Image.Image], norm: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[np.ndarray, float, float]:
    """``(P(con_signos), ms_crop_preprocess, ms_clasificador)`` con sincronizacion CUDA."""
    import torch
    from utils.classifier_data import build_transforms
    from utils.timing import synchronize
    from utils.training_common import resolve_device
    if not crops:
        return np.zeros(0), 0.0, 0.0
    device = resolve_device(cfg)
    use_amp = device != "cpu" and str(cfg["classifier"]["precision"]).lower() == "fp16"
    tf = build_transforms(cfg, norm, train=False)
    synchronize()
    t0 = time.perf_counter()
    x = torch.stack([tf(c.convert("RGB")) for c in crops]).to(device)
    synchronize()
    t1 = time.perf_counter()
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        logits = model(x)
    p = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
    synchronize()
    t2 = time.perf_counter()
    return p, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0


def evaluate_pipeline(
    cfg: Mapping[str, Any], *, detector: Any, classifier: torch.nn.Module, norm: Mapping[str, Any], image_paths: Mapping[str, Path],
    gt: Mapping[str, Mapping[str, Any]], conf: float, nms_iou: float, cls_threshold: float, match_iou: float, max_examples: int = 6,
) -> PipelineResult:
    """Ejecuta detector -> crops predichos -> clasificador sobre ``image_paths`` y calcula TODAS las metricas."""
    from utils.inference import crops_from_boxes, predict_detector
    from utils.timing import synchronize
    pos = str(cfg["classification_labels"]["positive_label"])
    neg = str(cfg["classification_labels"]["negative_label"])
    det_rows: list[dict[str, Any]] = []
    miss_rows: list[dict[str, Any]] = []
    img_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for iid in sorted(image_paths):
        img = load_image_rgb(image_paths[iid])
        g = gt[iid]
        gboxes = np.asarray(g["boxes"], dtype=float).reshape(-1, 4)
        synchronize()
        t0 = time.perf_counter()
        boxes, scores = predict_detector(detector, [img], conf=conf, iou=nms_iou, cfg=cfg)[0]
        synchronize()
        t_det = (time.perf_counter() - t0) * 1000.0
        tc0 = time.perf_counter()
        crops = crops_from_boxes(img, boxes, int(cfg["classification_dataset"]["margin_px"]))
        t_crop = (time.perf_counter() - tc0) * 1000.0
        probs, t_pre, t_cls = _classify_timed(classifier, crops, norm, cfg)
        res = match_detections(boxes, scores, gboxes, match_iou)
        ious = iou_matrix(boxes, gboxes) if len(boxes) and len(gboxes) else np.zeros((len(boxes), len(gboxes)))
        matched_pred = {pi: gi for pi, gi, _ in res["matches"]}
        for pi in range(len(boxes)):
            cx, cy = (boxes[pi, 0] + boxes[pi, 2]) / 2 / img.width, (boxes[pi, 1] + boxes[pi, 3]) / 2 / img.height
            rec = {"image_id": iid, "det_id": pi, "x1": boxes[pi, 0], "y1": boxes[pi, 1], "x2": boxes[pi, 2], "y2": boxes[pi, 3], "det_conf": float(scores[pi]),
                   "p_con_signos": float(probs[pi]), "matched_fruit_id": None, "gt_label": None, "iou": None, "outcome": None, "x_norm": cx, "y_norm": cy}
            if pi in matched_pred:
                gi = matched_pred[pi]
                rec.update(matched_fruit_id=int(g["fruit_ids"][gi]), gt_label=g["labels"][gi], iou=float(ious[pi, gi]))
            else:
                rec["outcome"] = "FP_duplicate" if (len(gboxes) and ious[pi].max() >= match_iou) else "FP_no_gt"
            det_rows.append(rec)
        for gi in res["fn"]:
            cx, cy = (gboxes[gi, 0] + gboxes[gi, 2]) / 2 / img.width, (gboxes[gi, 1] + gboxes[gi, 3]) / 2 / img.height
            miss_rows.append({"image_id": iid, "fruit_id": int(g["fruit_ids"][gi]), "gt_label": g["labels"][gi], "x1": gboxes[gi, 0], "y1": gboxes[gi, 1],
                              "x2": gboxes[gi, 2], "y2": gboxes[gi, 3], "x_norm": cx, "y_norm": cy})
        img_rows.append({"image_id": iid, "n_detections": len(boxes), "t_detector_ms": t_det, "t_crop_preprocess_ms": t_crop + t_pre,
                         "t_classifier_ms": t_cls, "t_total_ms": t_det + t_crop + t_pre + t_cls})
        if len(boxes) or len(res["fn"]):
            candidates.append({"image_path": str(image_paths[iid]), "title": iid, "tp": [tuple(boxes[pi]) for pi in matched_pred],
                               "fp": [tuple(boxes[pi]) for pi in res["fp"]], "fn": [tuple(gboxes[gi]) for gi in res["fn"]]})

    det = pd.DataFrame(det_rows, columns=DET_COLUMNS)
    misses = pd.DataFrame(miss_rows, columns=MISS_COLUMNS)
    op = {"det_conf": float(conf), "nms_iou": float(nms_iou), "cls_threshold": float(cls_threshold), "match_iou": float(match_iou)}
    summary, det, cm = summarize_from_table(det, misses, n_images=len(image_paths), pos=pos, neg=neg, op=op)
    summary["timing_note"] = "tiempos medidos en esta PC durante la evaluacion; NO son el benchmark controlado (17_benchmark_latency.py)"
    examples = sorted(candidates, key=lambda e: (-(len(e["fp"]) + len(e["fn"])), e["title"]))[:max_examples]
    return PipelineResult(summary, det, misses, pd.DataFrame(img_rows), cm, examples, error_table(det, misses))


def error_table(det: pd.DataFrame, misses: pd.DataFrame) -> pd.DataFrame:
    """Errores con su posicion normalizada (FP, FN y EC) para el mapa espacial."""
    rows: list[dict[str, Any]] = []
    if len(det):
        fp = det[det["matched_fruit_id"].isna()]
        ec = det[det["outcome"] == "EC"]
        rows += [{"image_id": r.image_id, "x_norm": r.x_norm, "y_norm": r.y_norm, "error_type": "FP"} for r in fp.itertuples(index=False)]
        rows += [{"image_id": r.image_id, "x_norm": r.x_norm, "y_norm": r.y_norm, "error_type": "EC"} for r in ec.itertuples(index=False)]
    rows += [{"image_id": r.image_id, "x_norm": r.x_norm, "y_norm": r.y_norm, "error_type": "FN"} for r in misses.itertuples(index=False)]
    return pd.DataFrame(rows, columns=["image_id", "x_norm", "y_norm", "error_type"])
