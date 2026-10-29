"""Inferencia: detector (Ultralytics) sobre imagenes completas y clasificador sobre crops.

Flujo del pipeline (sec. 23): imagen -> detector -> cajas -> crops de la imagen ORIGINAL -> preprocesamiento
del clasificador -> ``P(con_signos)`` -> umbral. El crop usa la MISMA ventana entera que los GT crops
(``boxes.crop_window``, sin margen; sec. 8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from utils.architectures import ArchSpec
from utils.boxes import crop_window
from utils.classifier_data import build_transforms
from utils.io import load_image_rgb
from utils.pretrained import configure_ultralytics
from utils.training_common import resolve_device, ultralytics_device


def load_detector(weights: str | Path, spec: ArchSpec) -> Any:
    """Carga un checkpoint de deteccion entrenado (``YOLO``/``RTDETR``)."""
    configure_ultralytics()
    from ultralytics import RTDETR, YOLO

    return RTDETR(str(weights)) if spec.family == "rtdetr" else YOLO(str(weights))


def rgb_to_bgr(arr: np.ndarray) -> np.ndarray:
    """``HxWx3`` RGB -> BGR contiguo (Ultralytics espera BGR para arrays de NumPy)."""
    return np.ascontiguousarray(arr[..., ::-1])


def predict_detector(
    model: Any, images: Sequence[Image.Image | np.ndarray], *, conf: float, iou: float, cfg: Mapping[str, Any], max_det: int = 300,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Predice cajas ``xyxy`` (pixeles de la imagen orientada) y scores para cada imagen.

    Args:
        conf: Umbral de confianza. Usar ``detector.val_conf`` (0.001) para evaluar mAP y el umbral
            afinado para el pipeline.
        iou: IoU de NMS (no tiene efecto en modelos sin NMS como YOLO26/RT-DETR).
    """
    arrays = [np.asarray(im) if isinstance(im, Image.Image) else im for im in images]
    half = str(cfg["detector"]["precision"]).lower() == "fp16" and resolve_device(cfg) != "cpu"
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for arr in arrays:
        res = model.predict(source=rgb_to_bgr(arr), conf=float(conf), iou=float(iou), imgsz=int(cfg["detector"]["image_size"]),
                            half=half, device=ultralytics_device(cfg), verbose=False, max_det=max_det)[0]
        boxes = res.boxes.xyxy.detach().cpu().numpy().astype(float) if len(res.boxes) else np.zeros((0, 4))
        scores = res.boxes.conf.detach().cpu().numpy().astype(float) if len(res.boxes) else np.zeros(0)
        out.append((boxes.reshape(-1, 4), scores.reshape(-1)))
    return out


def predict_paths(model: Any, paths: Sequence[Path], *, conf: float, iou: float, cfg: Mapping[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Como :func:`predict_detector` leyendo las imagenes del disco (orientacion EXIF aplicada)."""
    return predict_detector(model, [load_image_rgb(p) for p in paths], conf=conf, iou=iou, cfg=cfg)


def crops_from_boxes(image: Image.Image, boxes: np.ndarray, margin_px: int = 0) -> list[Image.Image]:
    """Recortes de la imagen original con la ventana entera de cada caja (sin margen por defecto)."""
    crops = []
    for b in boxes:
        left, top, right, bottom = crop_window((float(b[0]), float(b[1]), float(b[2]), float(b[3])), image.width, image.height, margin_px)
        crops.append(image.crop((left, top, right, bottom)))
    return crops


@torch.no_grad()
def classify_crops(model: torch.nn.Module, crops: Sequence[Image.Image], norm: Mapping[str, Any], cfg: Mapping[str, Any], *, batch_size: int = 64) -> np.ndarray:
    """``P(con_signos)`` de cada crop (preprocesamiento determinista del clasificador)."""
    if not crops:
        return np.zeros(0)
    device = resolve_device(cfg)
    use_amp = device != "cpu" and str(cfg["classifier"]["precision"]).lower() == "fp16"
    tf = build_transforms(cfg, norm, train=False)
    model.eval()
    probs: list[np.ndarray] = []
    for i in range(0, len(crops), batch_size):
        x = torch.stack([tf(c.convert("RGB")) for c in crops[i:i + batch_size]]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)
