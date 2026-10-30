"""Clasificadores binarios (``sin_signos`` = 0, ``con_signos`` = 1): ResNet50, ViT-B/16 y Swin-Tiny.

* Preentrenados en ImageNet (``pretrained: true`` es FIJO y comun a las tres arquitecturas, sec. 14.2).
* Cabeza de 2 logits; ``P(con_signos) = softmax(logits)[:, 1]``.
* Normalizacion EXIGIDA por el checkpoint preentrenado (media/desviacion de su ``pretrained_cfg``).
* ResNet50 via torchvision; ViT-B/16 y Swin-Tiny via timm (``drop_path_rate`` de Swin es hiperparametro HPO).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from utils.architectures import ArchSpec

CLASS_NAMES: tuple[str, str] = ("sin_signos", "con_signos")
POSITIVE_INDEX = 1


def build_classifier(
    spec: ArchSpec, hp: Mapping[str, Any] | None = None, *, pretrained: bool = True, cfg: Mapping[str, Any] | None = None, implementation_model: str | None = None
) -> tuple[nn.Module, dict[str, tuple[float, ...]]]:
    """Construye el modelo y devuelve ``(modelo, {'mean': ..., 'std': ...})``.

    Args:
        spec: Arquitectura de clasificacion.
        hp: Hiperparametros (solo ``drop_path_rate`` afecta a la construccion).
        pretrained: Cargar checkpoint local verificado; False crea arquitectura sin adquirir pesos.
    """
    hp = dict(hp or {})
    source = None
    if pretrained:
        if cfg is None:
            raise RuntimeError("Pretrained requiere cfg y pesos locales verificados; no se descarga")
        from utils.weights import verified_local_weight
        source = verified_local_weight(spec, cfg)
    if spec.family == "resnet50":
        import torchvision

        weights = None
        model = torchvision.models.resnet50(weights=None)
        if source:
            model.load_state_dict(torch.load(source, map_location="cpu", weights_only=True), strict=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if weights is not None:
            tr = weights.transforms()
            mean, std = tuple(float(v) for v in tr.mean), tuple(float(v) for v in tr.std)
        model._botrytis_implementation_model = "resnet50"
        return model, {"mean": mean, "std": std}
    if spec.family in ("vit", "swin"):
        import timm

        kwargs: dict[str, Any] = {}
        if spec.family == "swin":
            kwargs["drop_path_rate"] = float(hp.get("drop_path_rate", 0.0))
        from utils.weights import entry_for
        exact_model = implementation_model or (entry_for(spec, cfg)["implementation_model"] if cfg else spec.weights)
        model = timm.create_model(exact_model, pretrained=False, num_classes=1000 if source else 2, **kwargs)
        if source:
            state = torch.load(source, map_location="cpu", weights_only=True)
            if spec.family == "swin":
                # Use timm's checkpoint adapter: older Swin files contain derived
                # attention masks/indices that current models regenerate locally.
                from timm.models.swin_transformer import checkpoint_filter_fn
                state = checkpoint_filter_fn(state, model)
            model.load_state_dict(state, strict=True)
            model.reset_classifier(2)
        pc = getattr(model, "pretrained_cfg", {}) or {}
        mean = tuple(float(v) for v in pc.get("mean", (0.485, 0.456, 0.406)))
        std = tuple(float(v) for v in pc.get("std", (0.229, 0.224, 0.225)))
        model._botrytis_implementation_model = exact_model
        return model, {"mean": mean, "std": std}
    raise ValueError(f"{spec.name} no es un clasificador")


def save_classifier(path: Any, model: nn.Module, spec: ArchSpec, hp: Mapping[str, Any], norm: Mapping[str, Any], **extra: Any) -> None:
    """Guarda un checkpoint autocontenido (pesos + hiperparametros + normalizacion)."""
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"arch": spec.name, "implementation_model":getattr(model,"_botrytis_implementation_model",spec.weights), "state_dict": state, "hp": dict(hp), "norm": {k: list(v) for k, v in norm.items()},
                "image_size": 224, "classes": list(CLASS_NAMES), **extra}, path)


def load_classifier(path: Any, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Reconstruye un clasificador desde un checkpoint propio (sin descargar pesos).

    Returns:
        ``(modelo en eval, meta)`` con ``meta`` = checkpoint sin ``state_dict`` (arch, hp, norm...).
    """
    from utils.architectures import get_arch

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("classes")!=list(CLASS_NAMES) or set(ckpt.get("norm",{}))!={"mean","std"}:raise ValueError("Checkpoint de otro contrato de clases/normalizacion")
    if len(ckpt["norm"]["mean"])!=3 or len(ckpt["norm"]["std"])!=3 or any(float(v)<=0 for v in ckpt["norm"]["std"]):raise ValueError("Normalizacion invalida")
    spec = get_arch(ckpt["arch"])
    model, _ = build_classifier(spec, ckpt.get("hp"), pretrained=False, implementation_model=ckpt.get("implementation_model"))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta
