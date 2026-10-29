"""Grad-CAM para los 3 clasificadores (ResNet50, ViT-B/16, Swin-Tiny).

Capa objetivo: ``layer4[-1]`` (ResNet50), ``blocks[-1].norm1`` (ViT-B/16) y
``layers[-1].blocks[-1].norm1`` (Swin-Tiny). Los tokens se reorganizan en una rejilla espacial.
Se calcula en float32 y con gradientes (sin autocast).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.architectures import ArchSpec
from utils.classifier_data import build_transforms


def target_layer(model: torch.nn.Module, spec: ArchSpec) -> torch.nn.Module:
    """Capa cuyas activaciones/gradientes definen el mapa de calor."""
    if spec.family == "resnet50":
        return model.layer4[-1]  # type: ignore[index,no-any-return]
    if spec.family == "vit":
        return model.blocks[-1].norm1  # type: ignore[index,no-any-return]
    if spec.family == "swin":
        return model.layers[-1].blocks[-1].norm1  # type: ignore[index,no-any-return]
    raise ValueError(f"Grad-CAM no definido para {spec.name}")


def to_spatial(act: torch.Tensor, spec: ArchSpec) -> torch.Tensor:
    """Activacion -> ``(B, C, H, W)``."""
    if spec.family == "resnet50":
        return act
    if spec.family == "vit":  # (B, 1 + N, C): descarta el token de clase
        tokens = act[:, 1:, :]
        side = int(round(tokens.shape[1] ** 0.5))
        return tokens.reshape(tokens.shape[0], side, side, tokens.shape[2]).permute(0, 3, 1, 2)
    if act.dim() == 4:  # swin de timm >= 0.9: (B, H, W, C)
        return act.permute(0, 3, 1, 2)
    side = int(round(act.shape[1] ** 0.5))  # (B, N, C)
    return act.reshape(act.shape[0], side, side, act.shape[2]).permute(0, 3, 1, 2)


def gradcam(model: torch.nn.Module, spec: ArchSpec, x: torch.Tensor, *, target_class: int = 1) -> tuple[np.ndarray, float]:
    """Mapa Grad-CAM normalizado a [0, 1] (tamano de ``x``) y ``P(target_class)`` para un crop ``x`` (1, 3, H, W)."""
    store: dict[str, torch.Tensor] = {}
    layer = target_layer(model, spec)

    def hook(_m: Any, _i: Any, out: torch.Tensor) -> None:
        store["act"] = out

    h = layer.register_forward_hook(hook)
    model.eval()
    try:
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = model(x.float())
            prob = torch.softmax(logits.float(), dim=1)[0, target_class].item()
            act = store["act"]
            grads = torch.autograd.grad(logits[0, target_class], act, retain_graph=False)[0]
    finally:
        h.remove()
    a, g = to_spatial(act.detach(), spec), to_spatial(grads.detach(), spec)
    weights = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * a).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max() if float(cam.max()) > 0 else cam
    return cam.cpu().numpy(), float(prob)


def cam_for_image(model: torch.nn.Module, spec: ArchSpec, norm: Mapping[str, Any], cfg: Mapping[str, Any], image: Image.Image, device: str) -> tuple[np.ndarray, float]:
    """Grad-CAM de un crop PIL (preprocesamiento determinista del clasificador)."""
    tf = build_transforms(cfg, norm, train=False)
    x = tf(image.convert("RGB")).unsqueeze(0).to(device)
    return gradcam(model.to(device), spec, x)


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpone el mapa de calor (``jet``) sobre el crop (ambos a ``cam.shape``)."""
    import matplotlib

    size = (cam.shape[1], cam.shape[0])
    base = np.asarray(image.convert("RGB").resize(size)).astype(float) / 255.0
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)
