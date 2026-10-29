"""Lectura y verificacion del ``freeze_manifest.json`` (sec. 26-27).

``verify_freeze`` comprueba que NADA cambio desde la congelacion: sello SHA-256 del manifiesto, hashes de
codigo/dataset/split, checkpoints, umbrales definitivos y versiones criticas del entorno.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from utils.checks import MethodologyViolation
from utils.config_loader import cfg_path
from utils.hashing import sha256_file, verify_checksum_file
from utils.io import read_json
from utils.semantics import semantic_fingerprint


def freeze_path(cfg: Mapping[str, Any]) -> Path:
    """Ruta de ``freeze_manifest.json``."""
    return cfg_path(cfg, "test.freeze_manifest")


def load_freeze(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Carga el manifiesto verificando su sello SHA-256.

    Raises:
        MethodologyViolation: Si no existe o fue alterado.
    """
    p = freeze_path(cfg)
    if not p.is_file():
        raise MethodologyViolation(f"No existe {p.name}: ejecute 15_freeze_experiment.py antes de abrir el test (sec. 26)")
    if not verify_checksum_file(p, p.with_suffix(".sha256")):
        raise MethodologyViolation(f"{p.name} NO coincide con su SHA-256: fue modificado despues de congelar")
    manifest = dict(read_json(p))
    if manifest.get("semantic_fingerprint") != semantic_fingerprint(cfg):
        raise MethodologyViolation("Freeze incompatible con semantica/pairing/split vigente: regenerar experimento")
    return manifest


def verify_freeze(cfg: Mapping[str, Any], freeze: Mapping[str, Any], hashes: Mapping[str, Any], env: Mapping[str, Any], *, check_code: bool = True) -> None:
    """Aborta si algo cambio tras la congelacion.

    Args:
        hashes: ``ctx.hashes`` (``code_hash``, ``dataset_hash``, ``split_hash``) vigentes.
        env: ``ctx.env`` actual (versiones criticas).
        check_code: Comparar ``code_hash`` (el benchmark/exportacion pueden desactivarlo si solo leen).
    """
    from utils.comparison import verify_candidates
    if "comparison" not in freeze:
        raise MethodologyViolation("Freeze anterior sin candidatos comparativos; completar validation y congelar con esta version antes del test")
    errors: list[str] = []
    if freeze.get("semantic_fingerprint") != semantic_fingerprint(cfg):
        errors.append("semantic_fingerprint cambio o ausente")
    if check_code and hashes.get("code_hash") != freeze["code"]["code_hash"]:
        errors.append(f"code_hash cambio tras el freeze ({freeze['code']['code_hash'][:12]}... -> {str(hashes.get('code_hash'))[:12]}...)")
    for key in ("dataset_hash", "split_hash"):
        if hashes.get(key) != freeze["data"][key]:
            errors.append(f"{key} cambio tras el freeze")
    art = cfg_path(cfg, "paths.artifacts_dir")
    for name in ("detector", "classifier"):
        ck = Path(freeze[name]["checkpoint"])
        if not ck.is_file():
            errors.append(f"falta el checkpoint congelado del {name}: {ck}")
        elif sha256_file(ck) != freeze[name]["checkpoint_hash"]:
            errors.append(f"el checkpoint del {name} cambio tras el freeze")
    ft_path = art / "final_thresholds.json"
    if ft_path.is_file():
        ft = read_json(ft_path)
        if ft["detector"]["conf"] != freeze["detector"]["confidence_threshold"] or ft["detector"]["nms_iou"] != freeze["detector"]["nms_iou"]:
            errors.append("los umbrales del detector cambiaron tras el freeze")
        if ft["classifier"]["probability_threshold"] != freeze["classifier"]["probability_threshold"]:
            errors.append("el umbral del clasificador cambio tras el freeze")
    fz_env = freeze["environment"]
    for k in ("python_version", "torch", "torchvision", "timm", "ultralytics", "optuna"):
        if fz_env.get(k) is not None and env.get(k) != fz_env.get(k):
            errors.append(f"version de {k} distinta a la congelada ({fz_env.get(k)} -> {env.get(k)})")
    if errors:
        raise MethodologyViolation("El experimento congelado fue alterado: " + "; ".join(errors))
    verify_candidates(cfg, freeze["comparison"])
