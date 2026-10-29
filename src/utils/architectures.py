"""Registro de las 7 arquitecturas (4 detectores + 3 clasificadores) y nombres de estudio.

Un estudio Optuna POR ARQUITECTURA (sec. 10.4): nunca un ``study_global``. El nombre del estudio es
``<prefijo>_<slug>`` (``botrytis_yolov8n``...; sec. 17.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

DETECTION = "detection"
CLASSIFICATION = "classification"


@dataclass(frozen=True)
class ArchSpec:
    """Definicion inmutable de una arquitectura.

    Attributes:
        name: Nombre de la metodologia (``YOLOv8n``).
        slug: Identificador de fichero/estudio (``yolov8n``).
        task: ``detection`` | ``classification``.
        family: ``yolov8`` | ``yolo26`` | ``rtdetr`` | ``resnet50`` | ``vit`` | ``swin``.
        weights: Nombre del checkpoint preentrenado (Ultralytics) o modelo timm/torchvision.
        space_key: Clave en ``configs/search_spaces.yaml``.
    """

    name: str
    slug: str
    task: str
    family: str
    weights: str
    space_key: str

    def study_name(self, prefix: str = "botrytis") -> str:
        """Nombre del estudio Optuna de esta arquitectura."""
        return f"{prefix}_{self.slug}"

    @property
    def is_detector(self) -> bool:
        """``True`` para los 4 detectores."""
        return self.task == DETECTION

    @property
    def is_ultralytics(self) -> bool:
        """``True`` si se entrena con Ultralytics (YOLOv8, YOLO26, RT-DETR)."""
        return self.family in ("yolov8", "yolo26", "rtdetr")


ARCHITECTURES: tuple[ArchSpec, ...] = (
    ArchSpec("YOLOv8n", "yolov8n", DETECTION, "yolov8", "yolov8n.pt", "yolov8"),
    ArchSpec("YOLOv8l", "yolov8l", DETECTION, "yolov8", "yolov8l.pt", "yolov8"),
    ArchSpec("YOLO26l", "yolo26l", DETECTION, "yolo26", "yolo26l.pt", "yolo26"),
    ArchSpec("RT-DETR-R50", "rtdetr_r50", DETECTION, "rtdetr", "rtdetr-resnet50", "rtdetr"),
    ArchSpec("ResNet50", "resnet50", CLASSIFICATION, "resnet50", "resnet50", "resnet50"),
    ArchSpec("ViT-B/16", "vit_b16", CLASSIFICATION, "vit", "vit_base_patch16_224", "vit"),
    ArchSpec("Swin-Tiny", "swin_tiny", CLASSIFICATION, "swin", "swin_tiny_patch4_window7_224", "swin"),
)

_BY_KEY: dict[str, ArchSpec] = {}
for _a in ARCHITECTURES:
    for _k in (_a.name, _a.slug, _a.name.lower(), _a.name.lower().replace("/", "_").replace("-", "_")):
        _BY_KEY[_k] = _a


class UnknownArchitecture(KeyError):
    """Nombre de arquitectura no incluido en la metodologia (p. ej. U-Net, sec. 2.3)."""


def get_arch(name: str) -> ArchSpec:
    """Devuelve la :class:`ArchSpec` por nombre o slug (insensible a mayusculas/guiones).

    Raises:
        UnknownArchitecture: Si no pertenece a las 7 arquitecturas de la metodologia.
    """
    key = str(name).strip()
    spec = _BY_KEY.get(key) or _BY_KEY.get(key.lower()) or _BY_KEY.get(key.lower().replace("/", "_").replace("-", "_"))
    if spec is None:
        raise UnknownArchitecture(
            f"Arquitectura desconocida '{name}'. Validas: {[a.name for a in ARCHITECTURES]} (U-Net NO se programa, sec. 2.3)."
        )
    return spec


def archs(task: str | None = None) -> list[ArchSpec]:
    """Arquitecturas (opcionalmente filtradas por tarea), en el orden de la metodologia."""
    return [a for a in ARCHITECTURES if task is None or a.task == task]


def trials_for(spec: ArchSpec, cfg: Mapping[str, Any]) -> int:
    """Presupuesto HPO de la arquitectura (15 detectores / 20 clasificadores)."""
    section = "detector" if spec.is_detector else "classifier"
    return int(cfg[section]["trials_per_architecture"])


def section_for(spec: ArchSpec, cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Seccion de la configuracion (``detector``/``classifier``) de la arquitectura."""
    return cfg["detector" if spec.is_detector else "classifier"]
