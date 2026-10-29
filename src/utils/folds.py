"""Folds internos para HPO (sec. 6): a nivel de imagen original y solo dentro del outer train.

* Clasificacion: estratificado por ``has_positive_signs`` (opcionalmente + bin de % positivos).
* Deteccion: balanceado por bins de numero de frutos por imagen.
* Los crops heredan el fold de su imagen; NUNCA se crean folds sobre crops (sec. 6.3).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from utils.hashing import sha256_json, verify_checksum_file, write_checksum_file
from utils.io import read_csv, write_csv

DET_FOLD_COL = "inner_fold_det"
CLS_FOLD_COL = "inner_fold_cls"
DET_FILE = "inner_folds_detection.csv"
CLS_FILE = "inner_folds_classification.csv"


class FoldIntegrityError(RuntimeError):
    """Los folds internos fueron alterados o violan las reglas metodologicas."""


def assign_stratified_folds(
    image_ids: Sequence[str],
    strata: Mapping[str, str],
    k: int,
    seed: int,
) -> dict[str, int]:
    """Asigna cada imagen a un fold ``0..k-1`` estratificando por ``strata``.

    Dentro de cada estrato se baraja (semilla fija) y se reparte en ronda, continuando el
    contador entre estratos, de modo que los tamanos de fold difieren como maximo en 1.
    """
    if k < 2:
        raise ValueError("Se requieren al menos 2 folds")
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for iid in sorted(image_ids):
        groups.setdefault(str(strata[iid]), []).append(iid)
    folds: dict[str, int] = {}
    offset = 0
    for key in sorted(groups):
        members = groups[key]
        rng.shuffle(members)
        for i, iid in enumerate(members):
            folds[iid] = (offset + i) % k
        offset += len(members)
    return folds


def detection_strata(n_fruits: pd.Series, n_bins: int) -> dict[str, str]:
    """Bins por cuantiles (por rango) del numero de frutos por imagen.

    Args:
        n_fruits: Serie indexada por ``image_id``.
        n_bins: Numero de bins (se limita al numero de imagenes).
    """
    ordered = sorted(n_fruits.items(), key=lambda kv: (int(kv[1]), kv[0]))
    n = len(ordered)
    bins = max(1, min(n_bins, n))
    return {iid: f"det_bin{(rank * bins) // n}" for rank, (iid, _) in enumerate(ordered)}


def classification_strata(summary: pd.DataFrame, extra_positive_bin: bool = False) -> dict[str, str]:
    """Estratos de clasificacion: ``has_positive_signs`` (+ mediana de % positivos si se pide)."""
    strata: dict[str, str] = {}
    positives = summary[summary["has_positive_signs"]].sort_values(["pct_positive", "image_id"])
    hi_ids: set[str] = set()
    if extra_positive_bin and len(positives) >= 2:
        hi_ids = set(positives["image_id"].iloc[len(positives) // 2:])
    for row in summary.itertuples(index=False):
        if not row.has_positive_signs:
            strata[row.image_id] = "cls_neg"
        elif extra_positive_bin:
            strata[row.image_id] = "cls_pos_hi" if row.image_id in hi_ids else "cls_pos_lo"
        else:
            strata[row.image_id] = "cls_pos"
    return strata


def build_inner_folds(
    train_summary: pd.DataFrame,
    cfg: Mapping[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crea los folds de deteccion y clasificacion sobre las imagenes de outer train.

    Args:
        train_summary: ``labels.image_summary`` **restringido a outer train**.
        cfg: Configuracion (``inner_hpo.*``).
        seed: Semilla de folds.

    Returns:
        ``(detection_folds, classification_folds)`` DataFrames.
    """
    k = int(cfg["inner_hpo"]["folds"])
    ids = train_summary["image_id"].tolist()

    det_strata = detection_strata(train_summary.set_index("image_id")["n_fruits"], int(cfg["inner_hpo"]["detection_count_bins"]))
    det_assign = assign_stratified_folds(ids, det_strata, k, seed)
    det = pd.DataFrame({
        "image_id": sorted(ids),
    })
    det[DET_FOLD_COL] = det["image_id"].map(det_assign).astype(int)
    det["n_fruits"] = det["image_id"].map(train_summary.set_index("image_id")["n_fruits"]).astype(int)
    det["stratum"] = det["image_id"].map(det_strata)

    cls_strata = classification_strata(train_summary, bool(cfg["inner_hpo"].get("classification_extra_positive_bin", False)))
    cls_assign = assign_stratified_folds(ids, cls_strata, k, seed)
    cls = pd.DataFrame({"image_id": sorted(ids)})
    cls[CLS_FOLD_COL] = cls["image_id"].map(cls_assign).astype(int)
    idx = train_summary.set_index("image_id")
    cls["has_positive_signs"] = cls["image_id"].map(idx["has_positive_signs"]).astype(bool)
    cls["n_fruits"] = cls["image_id"].map(idx["n_fruits"]).astype(int)
    cls["n_positive"] = cls["image_id"].map(idx["n_positive"]).astype(int)
    cls["stratum"] = cls["image_id"].map(cls_strata)
    return det, cls


def write_inner_folds(det: pd.DataFrame, cls: pd.DataFrame, artifacts_dir: Path) -> dict[str, Any]:
    """Escribe ambos CSV de folds + sus ``.sha256`` y devuelve ``folds_hash`` y rutas."""
    det_path, cls_path = artifacts_dir / DET_FILE, artifacts_dir / CLS_FILE
    write_csv(det, det_path)
    write_csv(cls, cls_path)
    det_sha = write_checksum_file(det_path, det_path.with_suffix(".sha256"))
    cls_sha = write_checksum_file(cls_path, cls_path.with_suffix(".sha256"))
    folds_hash = sha256_json({
        "det": sorted(zip(det["image_id"], det[DET_FOLD_COL].astype(int))),
        "cls": sorted(zip(cls["image_id"], cls[CLS_FOLD_COL].astype(int))),
    })
    return {
        "folds_hash": folds_hash, "det_file": str(det_path), "cls_file": str(cls_path),
        "det_sha256": det_sha, "cls_sha256": cls_sha,
    }


def load_inner_folds(artifacts_dir: Path, *, verify: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los folds verificando sus ``.sha256``.

    Raises:
        FoldIntegrityError: Ausentes o alterados.
    """
    out: list[pd.DataFrame] = []
    for name in (DET_FILE, CLS_FILE):
        path = artifacts_dir / name
        if not path.is_file():
            raise FoldIntegrityError(f"No existe {path}. Ejecute 03_create_inner_folds.py primero.")
        if verify and not verify_checksum_file(path, path.with_suffix(".sha256")):
            raise FoldIntegrityError(f"{name} no coincide con su SHA-256: los folds son inmutables una vez creados.")
        out.append(read_csv(path))
    return out[0], out[1]


def fold_balance_report(folds: pd.DataFrame, fold_col: str, positive_col: str | None = None) -> dict[str, Any]:
    """Resumen de tamanos (y de positivos) por fold para el log/metrics."""
    report: dict[str, Any] = {"images_per_fold": {int(k): int(v) for k, v in folds[fold_col].value_counts().sort_index().items()}}
    if positive_col is not None:
        report["positive_images_per_fold"] = {
            int(k): int(v) for k, v in folds.groupby(fold_col)[positive_col].sum().sort_index().items()
        }
    return report
