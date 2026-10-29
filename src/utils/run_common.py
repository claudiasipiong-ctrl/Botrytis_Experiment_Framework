"""Piezas comunes de los scripts 06-18: argumentos de arquitectura, preflight metodologico y runs anidados.

* ``preflight_training`` aplica las guardias de la sec. 35 ANTES de gastar GPU: dataset validado y sin
  cambios, split/folds sellados, presupuesto/preentrenamiento/tamano de imagen, entorno identico al de
  referencia, test bloqueado y experimento no congelado.
* ``nested_run`` crea el run ``runs/<arquitectura>/<etapa>`` (los 6 ficheros obligatorios) reutilizando el
  ``environment.json`` del proceso padre.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from utils.architectures import ArchSpec, get_arch
from utils.checks import MethodologyViolation
from utils.config_loader import cfg_path, get_nested, resolve_path
from utils.guards import (
    assert_equal_hpo_budget, assert_image_size_fixed, assert_not_frozen, assert_same_pretrained_policy, is_official,
)
from utils.io import read_json
from utils.logging_utils import RunContext
from utils.prepared import PreparedData, load_prepared_data
from utils.reproducibility import compare_environments

#: Campos que no se comparan contra la referencia (datos/codigo se verifican por separado).
_ENV_NON_HW = ("git_commit", "code_hash", "dataset_hash", "split_hash")


def add_arch_arguments(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    """``--architecture`` (nombre o slug) y ``--skip-env-check``."""
    parser.add_argument("--architecture", "-a", required=required, default=None,
                        help="YOLOv8n|YOLOv8m|YOLOv8l|YOLO26n|YOLO26s|YOLO26l|RT-DETR-R50|ResNet50|ViT-B/16|Swin-Tiny (o el slug)")
    parser.add_argument("--skip-env-check", action="store_true", help="no comparar el entorno con la referencia (solo desarrollo; queda WARNING)")


def arch_from_args(args: argparse.Namespace) -> ArchSpec:
    """Arquitectura pedida por ``--architecture``."""
    return get_arch(args.architecture)


def layout_for(stage: str | None = None) -> Any:
    """``run_layout`` para ``run_cli``: ``runs/<slug>/<stage>`` (se reutiliza al reanudar) o ``runs/<slug>/<ts>_<script>``."""

    def _layout(args: argparse.Namespace) -> tuple[str | None, str | None, bool]:
        spec = get_arch(args.architecture)
        return spec.slug, stage, stage is not None

    return _layout


def check_environment_matches_reference(ctx: RunContext, *, strict: bool = True) -> None:
    """Compara el entorno de esta PC con ``environment_reference.json`` (sec. 16.6)."""
    ref_path = resolve_path(get_nested(ctx.cfg, "compute.reference_environment", default="artifacts/environments/environment_reference.json"))
    if not ref_path.is_file():
        msg = f"No existe la referencia de entorno {ref_path}: ejecute 00_check_environment.py --write-reference en la PC de referencia."
        if strict:
            raise MethodologyViolation(msg)
        ctx.logger.warning(msg)
        return
    fields = [f for f in get_nested(ctx.cfg, "compute.critical_environment_fields", default=[]) if f not in _ENV_NON_HW]
    ref = read_json(ref_path)
    # git/code/dataset/split no se comparan aqui: los datos se verifican por hash sellado (require_validated_dataset,
    # manifiestos) y el codigo se contrasta entre PCs (registro) y contra el freeze (16).
    diffs = compare_environments(ref, ctx.env, [f for f in fields if ref.get(f) is not None])
    if diffs:
        detail = "; ".join(f"{d['field']}: ref={d['reference']!r} esta_pc={d['candidate']!r}" for d in diffs)
        if strict:
            raise MethodologyViolation(f"El entorno de esta PC difiere del de referencia ({detail}). Las 3 PCs deben ser identicas (sec. 16.6).")
        ctx.logger.warning("Entorno distinto al de referencia (--skip-env-check): %s", detail)


def preflight_training(ctx: RunContext, args: argparse.Namespace, *, with_folds: bool = True, check_derived: bool = True) -> PreparedData:
    """Guardias metodologicas previas a cualquier etapa de entrenamiento/seleccion.

    Returns:
        ``PreparedData`` con dataset validado, split y (opcionalmente) folds verificados.
    """
    cfg = ctx.cfg
    assert_equal_hpo_budget(cfg, declared_override=not is_official(cfg))
    assert_same_pretrained_policy(cfg)
    assert_image_size_fixed(cfg)
    if not is_official(cfg):
        ctx.logger.warning("PERFIL '%s': presupuestos/tamanos reducidos; los resultados NO son cientificos.", get_nested(cfg, "project.profile"))
    data = load_prepared_data(cfg, with_folds=with_folds)
    from utils.isolation import require_near_duplicate_review
    require_near_duplicate_review(cfg, data)
    ctx.set_hashes(dataset_hash=data.dataset_hash, split_hash=data.split_hash)
    if check_derived:
        det_csv = cfg_path(cfg, "paths.detection_dir") / "manifests" / "detection_images.csv"
        cls_csv = cfg_path(cfg, "paths.classification_dir") / str(cfg["classification_dataset"]["manifest_name"])
        for p, script in ((det_csv, "04_build_detection_dataset.py"), (cls_csv, "05_build_classification_crops.py")):
            if not Path(p).is_file():
                raise MethodologyViolation(f"Falta {p}: ejecute {script} primero")
    from utils.detector_data import detection_images_table
    from utils.classifier_data import load_crop_manifest
    detection_images_table(cfg)
    load_crop_manifest(cfg)
    check_environment_matches_reference(ctx, strict=not bool(getattr(args, "skip_env_check", False)))
    return data


def prepare_for_inference(ctx: RunContext, args: argparse.Namespace) -> PreparedData:
    """Preflight ligero para etapas posteriores (evaluacion/umbrales/pipeline): datos + split, sin exigir folds."""
    cfg = ctx.cfg
    data = load_prepared_data(cfg, with_folds=True)
    from utils.isolation import require_near_duplicate_review
    require_near_duplicate_review(cfg, data)
    ctx.set_hashes(dataset_hash=data.dataset_hash, split_hash=data.split_hash)
    from utils.detector_data import detection_images_table
    from utils.classifier_data import load_crop_manifest
    detection_images_table(cfg)
    load_crop_manifest(cfg)
    check_environment_matches_reference(ctx, strict=not bool(getattr(args, "skip_env_check", False)))
    return data


def guard_not_frozen(ctx: RunContext, action: str) -> None:
    """Las etapas de entrenamiento/seleccion/umbrales no pueden ejecutarse tras el freeze."""
    assert_not_frozen(ctx.cfg, action)


def nested_run(parent: RunContext, spec: ArchSpec, stage_path: str, *, extra_cfg: Mapping[str, Any] | None = None, experiment: str | None = None,
               reuse_existing: bool = True) -> RunContext:
    """Run anidado ``runs/<slug>/<stage_path>`` (p. ej. ``hpo/trial_003``) con los 6 ficheros obligatorios.

    ``config.yaml`` del run incluye el bloque ``trial`` con lo especifico (hiperparametros, seed, folds...).
    """
    cfg = dict(parent.cfg)
    if extra_cfg:
        cfg["trial"] = dict(extra_cfg)
    ctx = RunContext(cfg, parent.script_name, runs_dir=parent.runs_base, pc_id=parent.pc_id, architecture=spec.slug, trial_name=stage_path,
                     experiment_name=experiment or parent.experiment_name, reuse_existing=reuse_existing, env_override=parent.env,
                     cli_args=parent.cli_args, argv=parent.argv)
    return ctx
