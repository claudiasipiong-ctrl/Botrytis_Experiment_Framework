"""Box geometry; sign coverage is descriptive only."""
from __future__ import annotations

from typing import Iterable

import numpy as np

Box = tuple[float, float, float, float]



# --------------------------------------------------------------------------- #
# Conversion YOLO <-> xyxy
# --------------------------------------------------------------------------- #
def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> Box:
    """Convierte YOLO normalizado a pixeles (sec. 3.3) **sin** recortar.

    x1 = (xc - w/2) * W ; x2 = (xc + w/2) * W ; idem para y.
    """
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return (x1, y1, x2, y2)


def clip_xyxy(box: Box, img_w: int, img_h: int) -> Box:
    """Recorta la caja a ``[0, W] x [0, H]`` (sec. 3.3)."""
    x1, y1, x2, y2 = box
    return (
        float(np.clip(x1, 0, img_w)),
        float(np.clip(y1, 0, img_h)),
        float(np.clip(x2, 0, img_w)),
        float(np.clip(y2, 0, img_h)),
    )


def xyxy_to_yolo(box: Box, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convierte pixeles a YOLO normalizado ``(xc, yc, w, h)``."""
    x1, y1, x2, y2 = box
    return (
        ((x1 + x2) / 2.0) / img_w,
        ((y1 + y2) / 2.0) / img_h,
        (x2 - x1) / img_w,
        (y2 - y1) / img_h,
    )


# --------------------------------------------------------------------------- #
# Areas e interseccion
# --------------------------------------------------------------------------- #
def box_area(box: Box) -> float:
    """Area de la caja (0 si es degenerada o invertida)."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(box_a: Box, box_b: Box) -> float:
    """Area de la interseccion entre dos cajas (0 si no se solapan)."""
    iw = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    ih = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    return iw * ih


def coverage_ratio(fruit_box: Box, sign_box: Box) -> float:
    """Cobertura descriptiva ``A(Bf ∩ Bs) / A(Bf)``; nunca determina etiquetas.

    Si el area del fruto no es positiva devuelve ``0.0``.
    """
    fruit_area = box_area(fruit_box)
    if fruit_area <= 0.0:
        return 0.0
    return intersection_area(fruit_box, sign_box) / fruit_area


def max_sign_coverage(fruit_box: Box, sign_boxes: Iterable[Box]) -> tuple[float, int]:
    """Maximo individual de cobertura sobre todas las cajas de signo.

    No suma coberturas ni une cajas (sec. 4).

    Returns:
        ``(max_coverage, indice_de_la_caja_ganadora)``; indice ``-1`` si no hay signos.
    """
    best, best_idx = 0.0, -1
    for idx, sb in enumerate(sign_boxes):
        cov = coverage_ratio(fruit_box, sb)
        if cov > best:
            best, best_idx = cov, idx
    return best, best_idx


# IoU for annotation identity and detection evaluation.
def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Matriz de IoU ``(len(a), len(b))`` para arrays ``(N, 4)`` ``xyxy``."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


def iou(box_a: Box, box_b: Box) -> float:
    """IoU entre dos cajas."""
    return float(iou_matrix(np.array([box_a]), np.array([box_b]))[0, 0])


# --------------------------------------------------------------------------- #
# Recortes
# --------------------------------------------------------------------------- #
def crop_window(box: Box, img_w: int, img_h: int, margin_px: int = 0) -> tuple[int, int, int, int]:
    """Ventana entera ``(left, top, right, bottom)`` que CUBRE la caja (floor/ceil).

    Garantiza al menos 1 px por lado y que quede dentro de la imagen.

    Args:
        box: Caja ``xyxy`` en pixeles.
        margin_px: Margen adicional (0 con la metodologia actual, sec. 8).
    """
    x1, y1, x2, y2 = box
    left = int(np.floor(x1)) - margin_px
    top = int(np.floor(y1)) - margin_px
    right = int(np.ceil(x2)) + margin_px
    bottom = int(np.ceil(y2)) + margin_px
    left, top = max(0, left), max(0, top)
    right, bottom = min(img_w, right), min(img_h, bottom)
    if right <= left:
        right = min(img_w, left + 1)
        left = max(0, right - 1)
    if bottom <= top:
        bottom = min(img_h, top + 1)
        top = max(0, bottom - 1)
    return left, top, right, bottom
