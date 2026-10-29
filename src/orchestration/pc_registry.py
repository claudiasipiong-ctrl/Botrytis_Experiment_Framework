"""Registro de PCs (PC-01, PC-02, PC-03): identidad, asignacion inicial y coherencia de entornos.

Reglas (sec. 16, Estrategia_3PCs sec. 4 y 9):

* 3 PCs iguales; maximo 1 trabajo GPU simultaneo por PC y 3 globales.
* Primera tanda sugerida (PC-01: YOLOv8n/m, PC-02: YOLOv8l/YOLO26l, PC-03: ResNet50/ViT-B/16); despues
  la cola es dinamica (la PC libre toma la siguiente arquitectura pendiente).
* ``BENCHMARK_PC`` (PC-01) es la unica que mide latencia.
* Antes de lanzar HPO, los ``environment_pc*.json`` deben coincidir en los campos criticos.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.architectures import get_arch  # noqa: E402
from utils.config_loader import cfg_path, get_nested  # noqa: E402
from utils.io import read_json  # noqa: E402
from utils.reproducibility import compare_environments  # noqa: E402

#: Campos de HARDWARE/SOFTWARE comparados entre PCs (los hashes de datos/codigo se comprueban aparte).
HW_SW_FIELDS = (
    "python_version", "os_system", "gpu_model", "gpu_vram_mb", "nvidia_driver", "cuda_runtime", "cudnn", "torch", "torchvision",
    "timm", "ultralytics", "optuna", "numpy", "precision_mode", "cpu_model", "ram_total_mb",
)


#: Hashes que las 3 PCs deben compartir (mismo codigo, dataset y split; sec. 16.6). Se ignoran si una PC aun no los tiene.
SHARED_HASH_FIELDS = ("code_hash", "dataset_hash", "split_hash")


class UnknownPC(ValueError):
    """``pc_id`` no pertenece al registro de PCs."""


@dataclass(frozen=True)
class PCInfo:
    """Datos de una PC del registro."""

    pc_id: str
    is_benchmark: bool
    preferred_slugs: tuple[str, ...]


def pc_token(pc_id: str) -> str:
    """``PC-01`` -> ``pc01``."""
    return re.sub(r"[^a-z0-9]", "", pc_id.lower()) or "unknown"


class PCRegistry:
    """Consulta del registro de PCs de la configuracion."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        reg = cfg.get("pc_registry", {})
        self.pcs: list[str] = list(reg.get("pcs", ["PC-01", "PC-02", "PC-03"]))
        self.first_batch: dict[str, list[str]] = {k: list(v) for k, v in (reg.get("first_batch") or {}).items()}
        self.benchmark_pc = str(get_nested(cfg, "benchmark.benchmark_pc", default="PC-01"))

    def validate(self, pc_id: str) -> PCInfo:
        """Devuelve la info de ``pc_id`` o lanza :class:`UnknownPC`."""
        if pc_id not in self.pcs:
            raise UnknownPC(f"PC '{pc_id}' desconocida; registro: {self.pcs}. Use --pc-id o BOTRYTIS_PC_ID (PC-01|PC-02|PC-03).")
        slugs = tuple(get_arch(a).slug for a in self.first_batch.get(pc_id, []))
        return PCInfo(pc_id, pc_id == self.benchmark_pc, slugs)

    def is_preferred(self, pc_id: str, slug: str) -> bool:
        """``True`` si ``slug`` esta en la primera tanda sugerida de ``pc_id``."""
        return slug in self.validate(pc_id).preferred_slugs

    def environment_file(self, pc_id: str) -> Path:
        """``artifacts/environments/environment_pcNN.json``."""
        return cfg_path(self.cfg, "paths.environments_dir") / f"environment_{pc_token(pc_id)}.json"

    def environment_mismatches(self) -> list[dict[str, Any]]:
        """Diferencias de hardware/software entre los ``environment_pc*.json`` disponibles."""
        files = {pc: self.environment_file(pc) for pc in self.pcs}
        present = {pc: read_json(p) for pc, p in files.items() if p.is_file()}
        mism: list[dict[str, Any]] = []
        items = sorted(present.items())
        for (pa, ea), (pb, eb) in zip(items, items[1:]):
            for d in compare_environments(ea, eb, HW_SW_FIELDS + SHARED_HASH_FIELDS):
                if d["field"] in SHARED_HASH_FIELDS and (d["reference"] is None or d["candidate"] is None):
                    continue  # una PC aun no tiene ese hash (p. ej. split no creado): no es una discrepancia
                mism.append({"pcs": f"{pa} vs {pb}", **d})
        return mism

    def missing_environments(self) -> list[str]:
        """PCs sin ``environment_pc*.json`` (no han ejecutado ``00_check_environment.py``)."""
        return [pc for pc in self.pcs if not self.environment_file(pc).is_file()]


    def ownership(self):
        mapping={}
        for pc,names in self.first_batch.items():
            for n in names:
                slug=get_arch(n).slug
                if slug in mapping:raise ValueError("Arquitectura con dos propietarios: "+slug)
                mapping[slug]=pc
        from utils.architectures import archs
        if set(mapping)!={a.slug for a in archs()}:raise ValueError("Reparto incompleto de las diez arquitecturas")
        return mapping

    def owner(self,slug):
        return self.ownership()[get_arch(slug).slug]
