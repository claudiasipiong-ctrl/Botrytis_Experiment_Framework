"""15 - Congelacion formal del experimento (Especificacion s2, sec. 26).

Crea ``artifacts/freeze_manifest.json`` (+ ``.sha256``) con el detector y el clasificador seleccionados
(arquitectura, hash del checkpoint, hiperparametros, tamano de imagen, umbrales), el pipeline (estrategia
de crops, preprocesamiento, regla de matching), los hashes de datos/split/codigo/entorno y las versiones.
La existencia del manifiesto **desbloquea el test** (``utils.splits.assert_split_allowed``) y, a la vez,
**prohibe** reentrenar, reseleccionar o reajustar umbrales (``utils.guards.assert_not_frozen``).

Requisitos previos (aborta si falta alguno): detector/clasificador seleccionados, validacion del pipeline con
predicted crops en PASS, umbrales definitivos, ``selection_policy.yaml`` inmutable, sin ``TODO`` en la
configuracion y ``decisions.confirmed_by_user: true`` (las decisiones del anexo quedan CONFIRMADAS por escrito).

Ejemplo::

    python src/15_freeze_experiment.py
"""

from __future__ import annotations

# Stdlib bootstrap: records import/CLI/config failures before optional dependencies.
import sys as _boot_sys
from pathlib import Path as _BootPath
_boot_root = _BootPath(__file__).resolve().parents[1]
_boot_sys.path.insert(0, str(_boot_root / "src"))
from utils import bootstrap as _bootstrap
_bootstrap.install(__file__, _boot_root)

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.checks import MethodologyViolation  # noqa: E402
from utils.config_loader import cfg_path, get_nested, list_todos  # noqa: E402
from utils.guards import is_official  # noqa: E402
from utils.hashing import canonical_json, sha256_file, write_checksum_file  # noqa: E402
from utils.io import read_json, write_json  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.run_common import prepare_for_inference  # noqa: E402
from utils.search_spaces import search_space_hash  # noqa: E402

SCRIPT = "15_freeze_experiment"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    parser.add_argument("--allow-unconfirmed-decisions", action="store_true", help="solo perfiles no oficiales: congela sin decisions.confirmed_by_user")
    parser.add_argument("--skip-env-check", action="store_true", help="no comparar el entorno con la referencia")


def _require(path: Path, hint: str) -> dict[str, Any]:
    if not path.is_file():
        raise MethodologyViolation(f"Falta {path.name}: {hint}")
    return dict(read_json(path))


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Valida los prerrequisitos, escribe el manifiesto de congelacion y lo sella."""
    cfg, log = ctx.cfg, ctx.logger
    data = prepare_for_inference(ctx, args)
    from utils.split_summary import export_split_summary
    export_split_summary(cfg, data)
    art = cfg_path(cfg, "paths.artifacts_dir")
    freeze_path = cfg_path(cfg, "test.freeze_manifest")
    sd = _require(art / "selected_detector.json", "ejecute 12_select_detector.py")
    sc = _require(art / "selected_classifier.json", "ejecute 13_select_classifier.py")
    pv = _require(art / "pipeline_validation.json", "ejecute 14_validate_integrated_pipeline.py (validacion con predicted crops, sec. 24)")
    ft = _require(art / "final_thresholds.json", "ejecute 14_validate_integrated_pipeline.py")
    if pv.get("status") != "PASS" or not pv.get("predicted_crops_used"):
        raise MethodologyViolation("La validacion del pipeline con predicted crops no esta en PASS (sec. 24)")
    todos = list_todos(cfg)
    if todos:
        raise MethodologyViolation(f"Quedan TODO sin resolver en la configuracion: {todos} (sec. 39)")
    confirmed = bool(get_nested(cfg, "decisions.confirmed_by_user", default=False))
    if not confirmed:
        if is_official(cfg) or not args.allow_unconfirmed_decisions:
            raise MethodologyViolation("decisions.confirmed_by_user es false: confirme por escrito las decisiones tomadas del anexo (configs/default.yaml > decisions) antes de congelar")
        log.warning("PERFIL NO OFICIAL: se congela sin decisiones confirmadas (--allow-unconfirmed-decisions)")
    for name, sel in (("detector", sd), ("classifier", sc)):
        ck = Path(sel["checkpoint"])
        if not ck.is_file():
            raise MethodologyViolation(f"No existe el checkpoint del {name}: {ck}")
        if sha256_file(ck) != sel["checkpoint_sha256"]:
            raise MethodologyViolation(f"El checkpoint del {name} cambio desde su seleccion (SHA-256 distinto)")
        if ft[name]["checkpoint_sha256"] != sel["checkpoint_sha256"]:
            raise MethodologyViolation(f"Los umbrales definitivos no corresponden al checkpoint del {name} seleccionado")
    for key in ("dataset_hash", "split_hash"):
        if pv.get(key) != ctx.hashes.get(key):
            raise MethodologyViolation(f"{key} de la validacion del pipeline ({pv.get(key)}) != actual ({ctx.hashes.get(key)})")
    policy_path = art / "selection_policy.yaml"
    if not policy_path.is_file():
        raise MethodologyViolation("Falta selection_policy.yaml (se materializa en 12/13; sec. 22.3)")
    lock = read_json(art / "test_lock.json") if (art / "test_lock.json").is_file() else {}
    env = ctx.env
    from utils.semantics import semantic_fingerprint
    from utils.comparison import freeze_candidates
    comparison = freeze_candidates(cfg, ctx.hashes)
    manifest = {
        "comparison": comparison,
        "semantic_fingerprint": semantic_fingerprint(cfg),
        "frozen_at": datetime.now(timezone.utc).isoformat(), "pc_id": ctx.pc_id, "profile": get_nested(cfg, "project.profile", default="official"),
        "detector": {
            "architecture": sd["architecture"], "checkpoint": sd["checkpoint"], "checkpoint_hash": sd["checkpoint_sha256"], "hyperparameters": sd["hp"], "config_id": sd["config_id"],
            "image_size": sd["image_size"], "confidence_threshold": ft["detector"]["conf"], "nms_iou": ft["detector"]["nms_iou"], "nms_applies": ft["detector"]["nms_applies"],
        },
        "classifier": {
            "architecture": sc["architecture"], "checkpoint": sc["checkpoint"], "checkpoint_hash": sc["checkpoint_sha256"], "hyperparameters": sc["hp"], "config_id": sc["config_id"],
            "image_size": sc["image_size"], "normalization": _norm(Path(sc["checkpoint"])), "probability_threshold": ft["classifier"]["probability_threshold"],
            "threshold_source": ft["classifier"]["threshold_source"], "positive_class": cfg["classifier"]["positive_class"],
        },
        "pipeline": {"crop_strategy": ft["crop_strategy"], "preprocessing": "Resize((224,224)) bicubico + Normalize(media/desv. del checkpoint), sin augmentation",
                     "matching_rule": ft["matching"]},
        "data": {"dataset_hash": ctx.hashes["dataset_hash"], "split_hash": ctx.hashes["split_hash"], "test_image_ids_sha256": lock.get("test_image_ids_sha256")},
        "code": {"git_commit": env.get("git_commit"), "code_hash": ctx.hashes["code_hash"], "search_space_sha256": search_space_hash()},
        "environment": {"environment_hash": env.get("environment_hash"), "python_version": env.get("python_version"), "torch": env.get("torch"),
                        "torchvision": env.get("torchvision"), "timm": env.get("timm"), "ultralytics": env.get("ultralytics"), "optuna": env.get("optuna"),
                        "cuda_runtime": env.get("cuda_runtime"), "cudnn": env.get("cudnn"), "nvidia_driver": env.get("nvidia_driver"), "gpu_model": env.get("gpu_model"),
                        "versions": env.get("packages")},
        "selection": {"policy_file": "artifacts/selection_policy.yaml", "policy_sha256": sha256_file(policy_path), "detector_reason": sd["reason"], "classifier_reason": sc["reason"]},
        "validation_pipeline": {"summary_file": "artifacts/pipeline_validation.json", "e_flujo": pv["summary"]["errors"]["e_flujo"]},
        "decisions_confirmed_by_user": confirmed, "decisions_adopted_from_annex": list(get_nested(cfg, "decisions.adopted_from_annex", default=[])),
        "nothing_may_change_after_this_file": True,
    }
    if freeze_path.is_file():
        old = read_json(freeze_path)
        if canonical_json(_strip_volatile(old)) != canonical_json(_strip_volatile(manifest)):
            raise MethodologyViolation(f"{freeze_path.name} ya existe y difiere: el experimento esta congelado (sec. 26); no se sobrescribe")
        log.info("freeze_manifest.json ya existe y es identico: no se reescribe")
        return
    write_json(freeze_path, manifest)
    digest = write_checksum_file(freeze_path, freeze_path.with_suffix(".sha256"))
    ctx.register_artifact(freeze_path, "manifiesto de congelacion")
    ctx.log_metrics({"freeze_sha256": digest, "detector": sd["architecture"], "classifier": sc["architecture"], "test_unlocked": True})
    log.info("EXPERIMENTO CONGELADO: detector=%s clasificador=%s | freeze sha256=%s", sd["architecture"], sc["architecture"], digest)
    log.info("A partir de ahora: NO cambiar nada en funcion de resultados de test (sec. 26-27).")


def _strip_volatile(manifest: dict[str, Any]) -> dict[str, Any]:
    """Manifiesto sin los campos que cambian al repetir el freeze (fecha y PC)."""
    return {k: v for k, v in manifest.items() if k not in ("frozen_at", "pc_id")}


def _norm(checkpoint: Path) -> dict[str, Any]:
    import torch

    meta = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return dict(meta.get("norm", {}))


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments))
