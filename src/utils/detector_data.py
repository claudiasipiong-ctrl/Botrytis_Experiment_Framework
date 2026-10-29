"""Datos del detector para Ultralytics: YAML por fold interno y del 70 % / 20 % externo.

Reglas (sec. 6, 7, 35):

* El HPO solo ve outer train: cada fold usa ``images/train`` partido por ``inner_fold_det``.
* El YAML que se entrega a Ultralytics NUNCA contiene la clave ``test`` (test bloqueado).
* Solo una clase (``0: fruit``); las cajas de signo no existen en estos labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from utils.checks import MethodologyViolation
from utils.config_loader import cfg_path
from utils.io import atomic_write_text, ensure_dir, read_csv
from utils.splits import assert_split_allowed


def list_file_encoding() -> str:
    """Codificacion con la que Ultralytics LEE los ``train.txt``/``val.txt``.

    Ultralytics (version fijada en ``requirements.txt``) abre las listas con ``open(p, encoding="utf-8")``; se
    escriben en UTF-8 para que rutas con acentos (p. ej. ``...\\SEMINARIO DE INVESTIGACIÓN II\\...``) se lean bien.
    ``tests/test_derived_inputs_and_audit.py`` y la prueba de humo con ruta acentuada comprueban que sigue siendo asi.
    """
    return "utf-8"


def detection_images_table(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Tabla ``image_id, outer_split, inner_fold_det, n_fruits, image_file`` escrita por ``04``."""
    from utils.semantics import require_stamp
    require_stamp(cfg_path(cfg, "paths.detection_dir"), cfg)
    path = cfg_path(cfg, "paths.detection_dir") / "manifests" / "detection_images.csv"
    if not path.is_file():
        raise MethodologyViolation(f"No existe {path}. Ejecute 04_build_detection_dataset.py primero.")
    table = read_csv(path)
    from utils.isolation import validate_derived_manifest
    validate_derived_manifest(cfg, table, "detection")
    return table


def _image_path(cfg: Mapping[str, Any], split: str, image_file: str) -> Path:
    names = cfg["detection_dataset"]["split_dir_names"]
    return cfg_path(cfg, "paths.detection_dir") / "images" / names[split] / image_file


def write_fold_yaml(cfg: Mapping[str, Any], fold: int, out_dir: Path, *, table: pd.DataFrame | None = None) -> tuple[Path, dict[str, int]]:
    """Crea ``train.txt``/``val.txt`` y ``data.yaml`` del fold ``fold`` (solo outer train).

    Returns:
        ``(ruta_data_yaml, {'n_train': .., 'n_val': ..})``.
    """
    assert_split_allowed("train", "hpo_trial", cfg)
    table = table if table is not None else detection_images_table(cfg)
    tr = table[table["outer_split"] == "train"]
    if tr["inner_fold_det"].isna().any():
        raise MethodologyViolation("Hay imagenes de outer train sin inner_fold_det")
    val_rows = tr[tr["inner_fold_det"].astype(int) == int(fold)]
    trn_rows = tr[tr["inner_fold_det"].astype(int) != int(fold)]
    if set(val_rows["image_id"]) & set(trn_rows["image_id"]):
        raise MethodologyViolation("Fuga: imagen en train y validation del mismo fold")
    out_dir = ensure_dir(out_dir)
    lists = {}
    enc = list_file_encoding()
    for name, rows in (("train", trn_rows), ("val", val_rows)):
        p = out_dir / f"{name}.txt"
        text = "\n".join(_image_path(cfg, "train", f).resolve().as_posix() for f in rows["image_file"]) + "\n"
        try:
            p.write_bytes(text.encode(enc))
        except UnicodeEncodeError as exc:  # ruta con caracteres fuera de la codificacion del sistema
            raise MethodologyViolation(
                f"La ruta del dataset contiene caracteres no representables en '{enc}' (Ultralytics lee las listas con esa codificacion). "
                "Mueva el proyecto/datos a una ruta sin caracteres especiales o ejecute Python con PYTHONUTF8=1."
            ) from exc
        lists[name] = p.resolve().as_posix()
    data = {"train": lists["train"], "val": lists["val"], "nc": 1, "names": {0: "fruit"}}
    yml = out_dir / "data.yaml"
    atomic_write_text(yml, yaml.safe_dump(data, sort_keys=False))
    return yml, {"n_train": len(trn_rows), "n_val": len(val_rows)}


def write_outer_yaml(cfg: Mapping[str, Any], out_dir: Path, *, val_split: str = "validation", stage: str = "outer_train") -> Path:
    """YAML para entrenar con TODO el outer train y validar en ``val_split`` (sin clave ``test``).

    Args:
        val_split: ``validation`` (seleccion/umbrales) o ``test`` (solo con ``stage='final_test'`` y freeze).
    """
    assert_split_allowed("train", stage, cfg)
    assert_split_allowed(val_split, stage, cfg)
    detection_images_table(cfg)
    from utils.semantics import require_stamp
    require_stamp(cfg_path(cfg, "paths.detection_dir"), cfg)
    names = cfg["detection_dataset"]["split_dir_names"]
    root = cfg_path(cfg, "paths.detection_dir")
    data = {"train": (root / "images" / names["train"]).resolve().as_posix(),
            "val": (root / "images" / names[val_split]).resolve().as_posix(), "nc": 1, "names": {0: "fruit"}}
    out_dir = ensure_dir(out_dir)
    yml = out_dir / f"data_train_{val_split}.yaml"
    atomic_write_text(yml, yaml.safe_dump(data, sort_keys=False))
    return yml


def split_images(cfg: Mapping[str, Any], split: str, stage: str) -> list[Path]:
    """Rutas de las imagenes de un split con la guardia de acceso (``test`` exige freeze)."""
    assert_split_allowed(split, stage, cfg)
    table = detection_images_table(cfg)
    rows = table[table["outer_split"] == split].sort_values("image_id")
    return [_image_path(cfg, split, f) for f in rows["image_file"]]
