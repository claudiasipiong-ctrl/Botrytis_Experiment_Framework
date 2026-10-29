"""Fingerprint and fail-closed validation of all derived experimental inputs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from utils.pairing import SEMANTICS_VERSION


def semantic_fingerprint(cfg: Mapping[str,Any]) -> str:
    """Identity of annotation, pairing, split and fold semantics."""
    payload = {"version": SEMANTICS_VERSION, "classes": cfg["classes"],
               "pairing": cfg["pairing"], "labels": cfg["classification_labels"],
               "split": cfg["outer_split"], "folds": cfg["inner_hpo"],
               "initialization": cfg.get("initialization"), "schema": 4}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def derived_fingerprint(cfg: Mapping[str,Any], dataset_hash: str, split_hash: str | None = None) -> str:
    """Bind semantics to source bytes and the original-image split."""
    return hashlib.sha256(f"{semantic_fingerprint(cfg)}:{dataset_hash}:{split_hash}".encode()).hexdigest()


def write_stamp(root: Path, cfg: Mapping[str,Any], dataset_hash: str, split_hash: str | None) -> None:
    """Write a completion stamp only after successful derived-data construction."""
    from utils.io import write_json
    write_json(Path(root)/"semantic_stamp.json", {"version": SEMANTICS_VERSION,
        "semantic_fingerprint": semantic_fingerprint(cfg), "dataset_hash": dataset_hash,
        "split_hash": split_hash, "derived_fingerprint": derived_fingerprint(cfg, dataset_hash, split_hash)})


def require_stamp(root: Path, cfg: Mapping[str,Any]) -> None:
    """Reject absent, stale or incompatible derived data before training."""
    from utils.checks import MethodologyViolation
    from utils.dataset import require_validated_dataset
    from utils.splits import load_split_manifest
    from utils.config_loader import cfg_path
    from utils.io import read_json
    path = Path(root)/"semantic_stamp.json"
    if not path.is_file():
        raise MethodologyViolation(f"Derivado sin version semantica: {root}. Regenerar 01-05.")
    stamp = read_json(path)
    dataset_hash = require_validated_dataset(cfg)
    split = load_split_manifest(cfg_path(cfg, "paths.artifacts_dir"), cfg["outer_split"]["manifest_name"])
    split_hash = str(split["split_hash"].iloc[0])
    if stamp.get("derived_fingerprint") != derived_fingerprint(cfg, dataset_hash, split_hash):
        raise MethodologyViolation(f"Derivado incompatible con dataset, semantica, pairing o split: {root}. Regenerar.")
