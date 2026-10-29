"""Controles metodologicos que deben hacer FALLAR el programa (Especificacion s2, sec. 35).

Cada funcion lanza :class:`MethodologyViolation` ante una violacion; nunca devuelve ``False``
en silencio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from utils.folds import CLS_FOLD_COL, DET_FOLD_COL
from utils.splits import SPLITS


class MethodologyViolation(RuntimeError):
    """Se violo una regla metodologica FIJA o de aborto obligatorio."""


def assert_one_split_per_image(manifest: pd.DataFrame) -> None:
    """[sec. 35] Una imagen no puede aparecer en mas de un outer split."""
    dup = manifest["image_id"][manifest["image_id"].duplicated()]
    if len(dup):
        raise MethodologyViolation(f"Imagenes en mas de un split: {sorted(set(dup))[:10]}")
    bad = set(manifest["outer_split"]) - set(SPLITS)
    if bad:
        raise MethodologyViolation(f"Splits desconocidos: {sorted(bad)}")


def assert_crops_inherit_split(crops: pd.DataFrame, manifest: pd.DataFrame) -> None:
    """[sec. 35] Un crop debe tener el split de su ``image_id`` (nunca se dividen crops)."""
    expected = manifest.set_index("image_id")["outer_split"]
    mapped = crops["image_id"].map(expected)
    if mapped.isna().any():
        raise MethodologyViolation("Hay crops cuyo image_id no existe en split_manifest.csv")
    bad = crops[mapped.to_numpy() != crops["outer_split"].to_numpy()]
    if len(bad):
        raise MethodologyViolation(
            f"{len(bad)} crops con split distinto al de su imagen, p. ej. {bad['image_id'].iloc[0]}"
        )


def assert_same_image_same_fold(crops: pd.DataFrame, fold_col: str = "inner_fold") -> None:
    """[sec. 35] Todos los crops de una imagen deben caer en el mismo fold interno."""
    train = crops[crops["outer_split"] == "train"]
    n_folds = train.groupby("image_id")[fold_col].nunique(dropna=False)
    bad = n_folds[n_folds > 1]
    if len(bad):
        raise MethodologyViolation(f"Crops de una misma imagen en folds distintos: {list(bad.index[:10])}")


def assert_folds_only_train(folds: pd.DataFrame, manifest: pd.DataFrame, fold_col: str, k: int) -> None:
    """Los folds internos cubren exactamente outer train y usan valores ``0..k-1``."""
    train_ids = set(manifest.loc[manifest["outer_split"] == "train", "image_id"])
    fold_ids = set(folds["image_id"])
    if fold_ids != train_ids:
        extra = sorted(fold_ids - train_ids)[:5]
        missing = sorted(train_ids - fold_ids)[:5]
        raise MethodologyViolation(
            f"Los folds deben cubrir EXACTAMENTE outer train. Sobran (val/test?): {extra}; faltan: {missing}"
        )
    if not set(folds[fold_col].astype(int)) <= set(range(k)):
        raise MethodologyViolation(f"Valores de fold fuera de 0..{k - 1}")
    if folds["image_id"].duplicated().any():
        raise MethodologyViolation("image_id repetido en los folds")


def assert_folds_consistent_across_tasks(det: pd.DataFrame, cls: pd.DataFrame) -> None:
    """Deteccion y clasificacion usan el mismo conjunto de imagenes de train."""
    if set(det["image_id"]) != set(cls["image_id"]):
        raise MethodologyViolation("Los folds de deteccion y clasificacion cubren imagenes distintas")


def assert_detection_labels_fruit_only(labels_dir: Path) -> int:
    """[sec. 35] Las cajas de signo NO son targets del detector: toda linea debe ser clase 0.

    Returns:
        Numero total de cajas de fruto verificadas.
    """
    total = 0
    for path in sorted(labels_dir.rglob("*.txt")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            cls = line.split()[0]
            if cls != "0":
                raise MethodologyViolation(
                    f"{path.name}:{line_no}: class_id={cls} en labels de deteccion (solo se permite 0 = fruit)"
                )
            total += 1
    return total


def assert_pairing_labels_consistent(fruit_table: pd.DataFrame, cfg: Mapping[str, object]) -> None:
    """Verify labels from accepted class 1 assignments only."""
    labels = cfg["classification_labels"]
    if fruit_table["label"].isna().any():
        raise MethodologyViolation("Unresolved annotation anomalies; cannot become negatives")
    positive = fruit_table["suspect_ordinal"].astype(int) >= 0
    expected = positive.map({True: labels["positive_label"], False: labels["negative_label"]})
    if not (expected == fruit_table["label"]).all():
        raise MethodologyViolation("Labels differ from suspicious annotation pairing")
    if (fruit_table.loc[positive, "pairing_iou"] < cfg["pairing"]["min_iou"]).any():
        raise MethodologyViolation("Accepted pair below minimum IoU")
    if fruit_table.loc[positive].duplicated(["image_id", "suspect_ordinal"]).any():
        raise MethodologyViolation("Suspicious annotation assigned multiple times")


def assert_fold_columns_present(manifest: pd.DataFrame) -> None:
    """Las columnas de fold esperadas existen en el manifiesto combinado."""
    for col in (DET_FOLD_COL, CLS_FOLD_COL):
        if col not in manifest.columns:
            raise MethodologyViolation(f"Falta la columna {col}")
