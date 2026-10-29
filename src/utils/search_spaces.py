"""Espacios de busqueda HPO (sec. 15), hiperparametros baseline e identidad de configuraciones.

Los espacios viven en ``configs/search_spaces.yaml`` (versionado); aqui solo se interpretan. Un
estudio Optuna por arquitectura: cada arquitectura usa SU espacio (sec. 10.4).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

from utils.architectures import ArchSpec
from utils.config_loader import PROJECT_ROOT
from utils.hashing import sha256_json, sha256_text_file

SEARCH_SPACES_PATH: Path = PROJECT_ROOT / "configs" / "search_spaces.yaml"


def load_search_spaces(path: Path | None = None) -> dict[str, Any]:
    """Carga ``configs/search_spaces.yaml``."""
    with (path or SEARCH_SPACES_PATH).open("r", encoding="utf-8") as fh:
        return dict(yaml.safe_load(fh))


def search_space_hash(path: Path | None = None) -> str:
    """SHA-256 (saltos de linea normalizados) del fichero de espacios de busqueda."""
    return sha256_text_file(path or SEARCH_SPACES_PATH)


def space_for(spec: ArchSpec, cfg: Mapping[str, Any], spaces: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Espacio de la arquitectura (sin ``optional_augmentation`` salvo que este habilitada)."""
    spaces = spaces or load_search_spaces()
    raw = copy.deepcopy(dict(spaces[spec.space_key]))
    optional = raw.pop("optional_augmentation", None)
    if optional and spec.family == "yolo26" and bool(cfg["detector"].get("yolo26_tune_augmentation", False)):
        raw.update(optional)
    return raw


def suggest_hyperparameters(trial: Any, spec: ArchSpec, cfg: Mapping[str, Any], spaces: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Muestrea los hiperparametros de un trial Optuna para ``spec``.

    Devuelve el diccionario CANONICO (``lr``, ``weight_decay``...) que consumen los entrenadores;
    los nombres internos de Optuna (``lr_if_adamw``, ``weight_decay_is_zero``) no se filtran.
    """
    space = space_for(spec, cfg, spaces)
    hp: dict[str, Any] = {}
    for name, d in space.items():
        kind = d["type"]
        if kind == "fixed":
            hp[name] = d["value"]
        elif kind == "categorical":
            hp[name] = trial.suggest_categorical(name, list(d["values"]))
        elif kind == "float":
            hp[name] = trial.suggest_float(name, float(d["min"]), float(d["max"]))
        elif kind == "log_float":
            hp[name] = trial.suggest_float(name, float(d["min"]), float(d["max"]), log=True)
        elif kind == "log_float_with_zero":
            is_zero = trial.suggest_categorical(f"{name}_is_zero", [False, True])
            hp[name] = 0.0 if is_zero else trial.suggest_float(name, float(d["min_nonzero"]), float(d["max"]), log=True)
        elif kind == "conditional_log_float":
            parent = hp[d["depends_on"]]
            lo, hi = d["ranges"][parent]
            hp[name] = trial.suggest_float(f"{name}_if_{str(parent).lower()}", float(lo), float(hi), log=True)
        else:
            raise ValueError(f"Tipo de espacio desconocido '{kind}' para {spec.name}.{name}")
    return hp


def baseline_hyperparameters(spec: ArchSpec, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Hiperparametros del baseline (sec. 41.2) en el mismo vocabulario que el espacio HPO."""
    if spec.is_detector:
        b = dict(cfg["detector"]["baseline"])
        b.update(cfg["detector"].get("baseline_overrides", {}).get(spec.name, {}))
        if spec.family == "rtdetr":
            return {"optimizer": "AdamW", "lr0": float(b["rtdetr_lr0"]), "backbone_lr_ratio": float(b["rtdetr_backbone_lr_ratio"]),
                    "weight_decay": float(b["weight_decay"])}
        hp = {k: b[k] for k in ("optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs")}
        if spec.family == "yolo26":
            hp["freeze"] = int(b.get("freeze", 0))
        return hp
    b = dict(cfg["classifier"]["baseline"])
    lr = float(b["lr"] if spec.family == "resnet50" else b["lr_transformers"])
    hp = {"optimizer": b["optimizer"], "lr": lr, "weight_decay": float(b["weight_decay"]),
          "warmup_epochs": int(b["warmup_epochs"]), "label_smoothing": float(b["label_smoothing"])}
    if spec.family == "swin":
        hp["drop_path_rate"] = float(b["drop_path_rate"])
    return hp


def enqueue_params_from_hp(spec: ArchSpec, hp: Mapping[str, Any], cfg: Mapping[str, Any], spaces: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Traduce un ``hp`` canonico a los parametros de Optuna (para ``study.enqueue_trial``).

    Solo incluye los parametros que pertenecen al espacio de la arquitectura; los valores fuera de
    rango se recortan al limite del espacio (un trial encolado nunca sale del espacio declarado).
    """
    space = space_for(spec, cfg, spaces)
    params: dict[str, Any] = {}
    for name, d in space.items():
        if name not in hp or d["type"] == "fixed":
            continue
        v = hp[name]
        kind = d["type"]
        if kind == "categorical":
            params[name] = v if v in d["values"] else d["values"][0]
        elif kind in ("float", "log_float"):
            params[name] = min(max(float(v), float(d["min"])), float(d["max"]))
        elif kind == "log_float_with_zero":
            if float(v) == 0.0:
                params[f"{name}_is_zero"] = True
            else:
                params[f"{name}_is_zero"] = False
                params[name] = min(max(float(v), float(d["min_nonzero"])), float(d["max"]))
        elif kind == "conditional_log_float":
            parent = hp[d["depends_on"]]
            lo, hi = d["ranges"][parent]
            params[f"{name}_if_{str(parent).lower()}"] = min(max(float(v), float(lo)), float(hi))
    return params


def _canonical(hp: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in sorted(hp.items()):
        out[k] = float(f"{v:.6g}") if isinstance(v, float) else v
    return out


def config_id(hp: Mapping[str, Any]) -> str:
    """Identificador estable de una configuracion (``cfg_<8 hex>``) independiente del orden de claves."""
    return "cfg_" + sha256_json(_canonical(hp))[:8]


def split_fixed_and_searched(spec: ArchSpec, cfg: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """``(parametros_fijos, parametros_optimizados)`` del espacio (para informes)."""
    space = space_for(spec, cfg)
    fixed = [k for k, d in space.items() if d["type"] == "fixed"]
    return fixed, [k for k in space if k not in fixed]
