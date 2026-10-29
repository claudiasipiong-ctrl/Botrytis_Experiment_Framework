"""Predeclare ALL validated candidates before opening test; never reselect on test."""
from pathlib import Path
from utils.architectures import archs
from utils.checks import MethodologyViolation
from utils.config_loader import cfg_path
from utils.hashing import sha256_file, sha256_json
from utils.hpo_results import final_checkpoint, load_best_config, stage_json_path
from utils.io import read_json


def protocol(cfg):
    return {"detector": cfg["detector"], "classifier": cfg["classifier"],
            "comparison_classifier_threshold": .5, "test_use": "descriptive_only_no_reselection"}


def freeze_candidates(cfg, hashes):
    candidates = []
    for spec in archs():
        val = stage_json_path(cfg, "outer_validation", spec)
        ck = final_checkpoint(cfg, spec)
        if not val.exists() and not ck.exists():
            continue
        if not val.is_file() or not ck.is_file():
            raise MethodologyViolation(f"Candidato incompleto antes del freeze: {spec.name}; completar 09/10")
        metrics = read_json(val)
        meta_path = ck.with_name("final_meta.json")
        if not meta_path.is_file():
            raise MethodologyViolation(f"Falta procedencia de entrenamiento: {spec.name}")
        meta = read_json(meta_path)
        if any(meta.get(k) != hashes.get(k) for k in ("dataset_hash", "split_hash")):
            raise MethodologyViolation(f"Entrenamiento pertenece a otros datos/split: {spec.name}")
        if metrics.get("checkpoint_sha256") != sha256_file(ck):
            raise MethodologyViolation(f"Checkpoint distinto al validado: {spec.name}")
        for key in ("dataset_hash", "split_hash"):
            if metrics.get(key) != hashes.get(key):
                raise MethodologyViolation(f"{key} obsoleto: {spec.name}")
        ev = metrics.get("evidence", {})
        from utils.evidence import split_dir
        evidence = split_dir(cfg, spec, "validation") / "metrics" / "metrics.json"
        if not evidence.is_file():
            raise MethodologyViolation(f"Faltan evidencias completas de validation: {spec.name}; repetir 10 antes del freeze")
        best = load_best_config(cfg, spec)
        if metrics.get("config_id") != best.get("config_id"):
            raise MethodologyViolation(f"Configuracion distinta a la validada: {spec.name}")
        candidates.append({"architecture": spec.name, "slug": spec.slug, "task": spec.task, "checkpoint": str(ck), "checkpoint_sha256": sha256_file(ck),
                           "validation_file": str(val), "validation_sha256": sha256_file(val), "best": best})
    if not candidates:
        raise MethodologyViolation("No hay candidatos validados para sellar")
    derived = {task: sha256_file(cfg_path(cfg, f"paths.{task}_dir") / "content_integrity.json") for task in ("detection", "classification")}
    review_path = cfg_path(cfg, "paths.artifacts_dir") / "near_duplicate_review.json"
    return {"review_sha256": sha256_file(review_path) if review_path.is_file() else None, "candidates": candidates, "protocol": protocol(cfg), "protocol_sha256": sha256_json(protocol(cfg)), "derived_content_sha256": derived,
            "purpose": "comparacion descriptiva final de todos los candidatos; seleccion previa inmutable"}


def verify_candidates(cfg, comparison):
    if comparison.get("protocol_sha256") != sha256_json(protocol(cfg)):
        raise MethodologyViolation("Protocolo de comparacion modificado tras freeze")
    review_path = cfg_path(cfg, "paths.artifacts_dir") / "near_duplicate_review.json"
    if comparison.get("review_sha256") != (sha256_file(review_path) if review_path.is_file() else None):
        raise MethodologyViolation("Revision de fuentes modificada tras freeze")
    for task, digest in comparison["derived_content_sha256"].items():
        p = cfg_path(cfg, f"paths.{task}_dir") / "content_integrity.json"
        if not p.is_file() or sha256_file(p) != digest:
            raise MethodologyViolation(f"Contenido derivado {task} modificado tras freeze")
    for candidate in comparison["candidates"]:
        for pathkey, hashkey in (("checkpoint", "checkpoint_sha256"), ("validation_file", "validation_sha256")):
            p = Path(candidate[pathkey])
            if not p.is_file() or sha256_file(p) != candidate[hashkey]:
                raise MethodologyViolation(f"Candidato congelado alterado: {candidate['architecture']} / {pathkey}")
