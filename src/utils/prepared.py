"""Carga verificada de las entradas de los scripts 02-05 (dataset + split + folds).

Centraliza los controles "abortar si..." de la sec. 35 que se comprueban al leer artefactos:
dataset validado y sin cambios, manifiesto de split sellado, folds sellados y coherentes con el
split vigente.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from utils.checks import (
    assert_folds_consistent_across_tasks,
    assert_folds_only_train,
    assert_one_split_per_image,
)
from utils.config_loader import cfg_path, get_nested
from utils.dataset import load_annotations, require_validated_dataset
from utils.folds import CLS_FOLD_COL, DET_FOLD_COL, FoldIntegrityError, load_inner_folds
from utils.io import read_json
from utils.labels import build_fruit_table, image_summary
from utils.splits import SplitIntegrityError, load_split_manifest

FOLDS_METADATA = "inner_folds_metadata.json"
IMAGE_MANIFEST = "image_manifest.csv"


@dataclass
class PreparedData:
    """Dataset ya validado y (opcionalmente) con split y folds verificados."""

    dataset_hash: str
    artifacts_dir: Path
    images: pd.DataFrame
    annotations: pd.DataFrame
    fruits: pd.DataFrame
    summary: pd.DataFrame
    manifest: pd.DataFrame | None = None
    det_folds: pd.DataFrame | None = None
    cls_folds: pd.DataFrame | None = None

    @property
    def split_hash(self) -> str | None:
        """``split_hash`` del manifiesto cargado."""
        return str(self.manifest["split_hash"].iloc[0]) if self.manifest is not None and len(self.manifest) else None


def load_prepared_data(
    cfg: Mapping[str, Any], *, with_split: bool = True, with_folds: bool = False, recheck_hash: bool = True
) -> PreparedData:
    """Carga y verifica las entradas.

    Args:
        cfg: Configuracion.
        with_split: Cargar y verificar ``split_manifest.csv`` (+ ``.sha256``).
        with_folds: Cargar y verificar los folds internos (implica ``with_split``).
        recheck_hash: Recalcular el ``dataset_hash`` de todos los ficheros.

    Raises:
        DatasetNotValidatedError, SplitIntegrityError, FoldIntegrityError,
        MethodologyViolation: Si algun control falla.
    """
    artifacts = cfg_path(cfg, "paths.artifacts_dir")
    dataset_hash = require_validated_dataset(cfg, recheck=recheck_hash)
    images, annotations = load_annotations(cfg, validated_inventory=True)
    fruits = build_fruit_table(annotations, cfg)
    from utils.io import write_csv, write_json
    from utils.checks import MethodologyViolation
    schemas = {
        "pairing_audit": ["image_id","suspect_ordinal","suspect_box","fruit_ordinal","fruit_box","iou","status","reason"],
        "pairing_candidates": ["image_id","suspect_ordinal","suspect_box","fruit_ordinal","fruit_box","iou","selected","status","reason"],
        "pairing_anomalies": ["image_id","class_id","ordinal","reason"],
    }
    for key, columns in schemas.items():
        records = fruits.attrs[key]
        write_csv(pd.DataFrame(records) if records else pd.DataFrame(columns=columns), artifacts / f"{key}.csv")
    anomalies = fruits.attrs["pairing_anomalies"]
    # The full audit is now on disk. pandas deep-copies attrs for each group,
    # which otherwise replicates every image's candidates for every crop/image.
    fruits.attrs.clear()
    if anomalies:
        bad_ids = sorted({r["image_id"] for r in anomalies})
        if cfg["pairing"]["anomaly_policy"] == "error":
            raise MethodologyViolation(f"{len(anomalies)} unresolved annotation anomalies in {bad_ids}. Review pairing audit; no crops permitted.")
        write_json(artifacts / "quarantine_manifest.json", {
            "explicit_policy": "quarantine_images", "dataset_hash": dataset_hash,
            "image_ids": bad_ids, "anomalies": anomalies,
            "excluded_physical_fruits": int(fruits["image_id"].isin(bad_ids).sum())})
        images = images[~images["image_id"].isin(bad_ids)].copy()
        annotations = annotations[~annotations["image_id"].isin(bad_ids)].copy()
        fruits = fruits[~fruits["image_id"].isin(bad_ids)].copy()
    summary = image_summary(fruits, images, annotations, cfg)
    data = PreparedData(dataset_hash, artifacts, images, annotations, fruits, summary)

    if with_split or with_folds:
        manifest_name = str(get_nested(cfg, "outer_split.manifest_name", default="split_manifest.csv"))
        manifest = load_split_manifest(artifacts, manifest_name)
        assert_one_split_per_image(manifest)
        if set(manifest["image_id"]) != set(images["image_id"]):
            extra = sorted(set(manifest["image_id"]) - set(images["image_id"]))[:5]
            missing = sorted(set(images["image_id"]) - set(manifest["image_id"]))[:5]
            raise SplitIntegrityError(
                f"El manifiesto de split no coincide con las imagenes del dataset (sobran {extra}, faltan {missing}). "
                "El dataset cambio despues de crear el split."
            )
        from utils.semantics import semantic_fingerprint
        split_meta = read_json(artifacts / "split_metadata.json") if (artifacts / "split_metadata.json").is_file() else {}
        if split_meta.get("dataset_hash") != dataset_hash or split_meta.get("semantic_fingerprint") != semantic_fingerprint(cfg):
            raise SplitIntegrityError("Split incompatible with current dataset, classes, pairing or split configuration; regenerate 02-05")
        from utils.isolation import audit_source
        source_audit = audit_source(images, manifest)
        write_csv(source_audit, artifacts / "split_content_audit.csv")
        data.manifest = manifest

    if with_folds:
        det, cls = load_inner_folds(artifacts)
        k = int(cfg["inner_hpo"]["folds"])
        assert_folds_only_train(det, data.manifest, DET_FOLD_COL, k)  # type: ignore[arg-type]
        assert_folds_only_train(cls, data.manifest, CLS_FOLD_COL, k)  # type: ignore[arg-type]
        assert_folds_consistent_across_tasks(det, cls)
        meta_path = artifacts / FOLDS_METADATA
        if not meta_path.is_file():
            raise FoldIntegrityError(f"Falta {meta_path}; repita 03_create_inner_folds.py")
        meta = read_json(meta_path)
        if meta.get("split_hash") != data.split_hash or meta.get("dataset_hash") != dataset_hash or meta.get("semantic_fingerprint") != semantic_fingerprint(cfg):
            raise FoldIntegrityError(
                "Los folds internos se crearon con OTRO split (split_hash distinto). "
                "Regenere 03_create_inner_folds.py --force si el cambio de split fue intencional."
            )
        data.det_folds, data.cls_folds = det, cls
    return data
