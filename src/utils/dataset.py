"""Lectura del dataset YOLO exportado desde Label Studio.

Convenciones
------------
* ``image_id`` = nombre de fichero de la imagen **sin extension** (``IMG_0001.jpg`` -> ``IMG_0001``).
* La anotacion de una imagen es ``<labels_dir>/<image_id>.txt`` con lineas
  ``class_id x_center y_center width height`` (normalizado a la imagen **orientada por EXIF**).
* Clases 0/1/2: ``fruit``/``suspect``/``sign``. Solo 0 es fruto fisico.
  ``class_id`` original se conserva; clase 1 nunca es otro target del detector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from utils.boxes import yolo_to_xyxy
from utils.config_loader import cfg_path, require
from utils.io import get_image_size, read_json

ERROR = "ERROR"
WARNING = "WARNING"

#: Informe que escribe ``01_validate_dataset.py`` (contiene ``passed`` y ``dataset_hash``).
VALIDATION_REPORT = "dataset_validation.json"


def recorded_dataset_hash(artifacts_dir: Path) -> str | None:
    """``dataset_hash`` registrado por ``01_validate_dataset.py`` (``None`` si aun no existe)."""
    path = artifacts_dir / VALIDATION_REPORT
    if not path.is_file():
        return None
    value = read_json(path).get("dataset_hash")
    return str(value) if value else None


@dataclass(frozen=True)
class Issue:
    """Un hallazgo de validacion."""

    image_id: str
    severity: str
    code: str
    message: str
    line_no: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Representacion serializable (fila del CSV de incidencias)."""
        return {
            "image_id": self.image_id, "severity": self.severity, "code": self.code,
            "line_no": self.line_no, "message": self.message,
        }


@dataclass(frozen=True)
class RawAnnotation:
    """Linea YOLO ya parseada (sin convertir a pixeles)."""

    line_no: int
    class_id: int
    xc: float
    yc: float
    w: float
    h: float


@dataclass
class ParsedLabelFile:
    """Resultado de parsear un fichero de etiquetas."""

    annotations: list[RawAnnotation] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)


def discover_images(images_dir: Path, extensions: Iterable[str]) -> list[Path]:
    """Lista ordenada (por nombre) de imagenes soportadas directamente bajo ``images_dir``."""
    exts = {e.lower() for e in extensions}
    return sorted(
        (p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts),
        key=lambda p: p.name.lower(),
    )


def label_path_for(image_id: str, labels_dir: Path, label_ext: str = ".txt") -> Path:
    """Ruta esperada de la anotacion de ``image_id``."""
    return labels_dir / f"{image_id}{label_ext}"


def parse_yolo_label_file(path: Path, image_id: str) -> ParsedLabelFile:
    """Parsea un fichero YOLO detectando problemas de formato.

    Reporta ERROR por: numero de tokens != 5 (p. ej. poligonos de segmentacion o ``conf``),
    ``class_id`` no entero, valores no numericos o no finitos.
    """
    parsed = ParsedLabelFile()
    try:
        text = path.read_text(encoding="utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        parsed.issues.append(Issue(image_id, ERROR, "label_not_utf8", "El fichero de etiquetas no es UTF-8 valido"))
        return parsed
    except OSError as exc:
        parsed.issues.append(Issue(image_id, ERROR, "label_unreadable", f"No se puede leer el fichero: {exc}"))
        return parsed

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) != 5:
            parsed.issues.append(Issue(
                image_id, ERROR, "bad_format",
                f"Se esperaban 5 campos (class xc yc w h) y hay {len(tokens)}: '{line[:80]}'", line_no,
            ))
            continue
        try:
            class_f = float(tokens[0])
            if not class_f.is_integer() or "." in tokens[0] or "e" in tokens[0].lower():
                raise ValueError("class_id no es entero")
            class_id = int(class_f)
            xc, yc, w, h = (float(t) for t in tokens[1:])
        except ValueError:
            parsed.issues.append(Issue(image_id, ERROR, "bad_number", f"Valor no numerico/entero: '{line[:80]}'", line_no))
            continue
        if not all(math.isfinite(v) for v in (xc, yc, w, h)):
            parsed.issues.append(Issue(image_id, ERROR, "non_finite", f"Valor NaN/inf: '{line[:80]}'", line_no))
            continue
        parsed.annotations.append(RawAnnotation(line_no, class_id, xc, yc, w, h))
    return parsed


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def class_ids_from_cfg(cfg: Mapping[str, Any]) -> tuple[tuple[int, ...], int]:
    """Only class 0 is physical fruit; class 2 is auxiliary."""
    fruit, sign = require(cfg, "classes.fruit_class_id", "classes.sign_class_id", spec_ref="sec. 3.2")
    fruits = list(fruit) if isinstance(fruit, (list, tuple)) else [fruit]
    if not fruits or not all(_is_int(f) for f in fruits) or not _is_int(sign):
        raise ValueError(f"Los IDs de clase deben ser enteros, recibido fruit={fruit!r}, sign={sign!r}")
    if len(set(fruits)) != len(fruits) or sign in fruits:
        raise ValueError(f"IDs de clase repetidos o solapados: fruit={fruits}, sign={sign}")
    if fruits != [0] or sign != 2 or cfg["classes"]["suspicious_class_id"] != 1:
        raise ValueError("Dataset semantics require physical=0, suspicious=1, auxiliary=2")
    return (0,), 2


def dataset_dirs(cfg: Mapping[str, Any]) -> tuple[Path, Path]:
    """``(images_dir, labels_dir)`` resueltos desde la configuracion."""
    return cfg_path(cfg, "paths.raw_images_dir"), cfg_path(cfg, "paths.raw_labels_dir")


def load_annotations(cfg: Mapping[str, Any], *, validated_inventory: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga el dataset completo a tablas (asume que ``01_validate_dataset.py`` ya paso).

    Returns:
        ``(images, annotations)``:

        * ``images``: ``image_id, image_path, label_path, image_w, image_h``.
        * ``annotations``: ``image_id, kind ('fruit'|'sign'), ordinal, class_id, line_no, xc, yc, w, h,
          x1, y1, x2, y2, image_w, image_h`` con cajas ya recortadas (clip) a la imagen.
          ``ordinal`` numera cada tipo dentro de la imagen en orden de fichero (``fruit_id``).

    Raises:
        ValueError: Falta una anotacion, hay ``class_id`` desconocido o formato invalido.
    """
    fruit_ids, sign_id = class_ids_from_cfg(cfg)
    images_dir, labels_dir = dataset_dirs(cfg)
    label_ext = str(cfg["dataset"].get("label_extension", ".txt"))
    files = discover_images(images_dir, cfg["dataset"]["image_extensions"])
    # PreparedData has just rehashed the bytes and checked validation. Reuse its
    # measured image dimensions instead of decoding headers for every stage.
    sizes = {}
    if validated_inventory:
        from utils.io import read_csv
        inv = read_csv(cfg_path(cfg, "paths.artifacts_dir") / "dataset_inventory.csv")
        sizes = {str(r.image_id): (int(r.image_w), int(r.image_h)) for r in inv.itertuples()}

    img_rows: list[dict[str, Any]] = []
    ann_rows: list[dict[str, Any]] = []
    for img in files:
        image_id = img.stem
        lab = label_path_for(image_id, labels_dir, label_ext)
        if not lab.is_file():
            raise ValueError(f"Falta la anotacion de {image_id}: {lab}")
        parsed = parse_yolo_label_file(lab, image_id)
        if parsed.issues:
            first = parsed.issues[0]
            raise ValueError(f"Anotacion invalida en {image_id} (linea {first.line_no}): {first.message}")
        w, h = sizes[image_id] if image_id in sizes else get_image_size(img)
        img_rows.append({"image_id": image_id, "image_path": str(img), "label_path": str(lab), "image_w": w, "image_h": h})
        counters = {"fruit": 0, "suspect": 0, "sign": 0}
        for a in parsed.annotations:
            if a.class_id in fruit_ids:
                kind = "fruit"
            elif a.class_id == 1:
                kind = "suspect"
            elif a.class_id == sign_id:
                kind = "sign"
            else:
                raise ValueError(f"class_id desconocido {a.class_id} en {image_id} (linea {a.line_no})")
            x1, y1, x2, y2 = yolo_to_xyxy(a.xc, a.yc, a.w, a.h, w, h)
            ann_rows.append({
                "image_id": image_id, "kind": kind, "ordinal": counters[kind], "class_id": a.class_id,
                "line_no": a.line_no, "xc": a.xc, "yc": a.yc, "w": a.w, "h": a.h,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2, "image_w": w, "image_h": h,
            })
            counters[kind] += 1

    images = pd.DataFrame(img_rows, columns=["image_id", "image_path", "label_path", "image_w", "image_h"])
    ann_cols = ["image_id", "kind", "ordinal", "class_id", "line_no", "xc", "yc", "w", "h",
                "x1", "y1", "x2", "y2", "image_w", "image_h"]
    annotations = pd.DataFrame(ann_rows, columns=ann_cols)
    if annotations.empty:
        annotations = annotations.astype({"image_id": str})
    return images, annotations


def unique_sorted(values: Sequence[str]) -> list[str]:
    """Lista ordenada sin duplicados (determinista)."""
    return sorted(set(values))


# --------------------------------------------------------------------------- #
# Hash del dataset y prerrequisitos entre scripts
# --------------------------------------------------------------------------- #
class DatasetNotValidatedError(RuntimeError):
    """``01_validate_dataset.py`` no se ejecuto, no paso, o el dataset cambio despues."""


def compute_dataset_hash(cfg: Mapping[str, Any]) -> tuple[str, list[tuple[str, str, str]]]:
    """Calcula ``dataset_hash`` = SHA-256 sobre ``(image_id, sha256(imagen), sha256(label))``.

    Las etiquetas se hashean con saltos de linea normalizados (igual en Windows y Linux).

    Returns:
        ``(dataset_hash, registros)``.
    """
    from utils.hashing import compute_dataset_hash as _combine
    from utils.hashing import sha256_file, sha256_text_file

    images_dir, labels_dir = dataset_dirs(cfg)
    label_ext = str(cfg["dataset"].get("label_extension", ".txt"))
    records: list[tuple[str, str, str]] = []
    for img in discover_images(images_dir, cfg["dataset"]["image_extensions"]):
        lab = label_path_for(img.stem, labels_dir, label_ext)
        records.append((img.stem, sha256_file(img), sha256_text_file(lab) if lab.is_file() else "MISSING_LABEL"))
    import hashlib
    from utils.semantics import derived_fingerprint
    label_digest = hashlib.sha256()
    for lp in sorted(labels_dir.glob("*.txt")):
        label_digest.update(lp.name.encode() + b"\0" + sha256_text_file(lp).encode() + b"\0")
    return derived_fingerprint(cfg, _combine(records) + ":" + label_digest.hexdigest()), records


def require_validated_dataset(cfg: Mapping[str, Any], *, recheck: bool = True) -> str:
    """Exige que el dataset haya pasado ``01_validate_dataset.py`` y no haya cambiado.

    Args:
        recheck: Recalcula el hash de todos los ficheros y lo compara con el registrado.

    Returns:
        El ``dataset_hash`` vigente.

    Raises:
        DatasetNotValidatedError: Validacion ausente, fallida o hash distinto.
    """
    artifacts = cfg_path(cfg, "paths.artifacts_dir")
    report_path = artifacts / VALIDATION_REPORT
    if not report_path.is_file():
        raise DatasetNotValidatedError(f"No existe {report_path}. Ejecute 01_validate_dataset.py primero.")
    report = read_json(report_path)
    if not report.get("passed", False):
        raise DatasetNotValidatedError(
            f"La validacion del dataset FALLO ({report.get('n_errors')} errores). Corrija y repita 01_validate_dataset.py."
        )
    from utils.semantics import semantic_fingerprint
    if report.get("semantic_fingerprint") != semantic_fingerprint(cfg):
        raise DatasetNotValidatedError("Semantica/configuracion incompatible; regenerar 01-05")
    recorded = str(report["dataset_hash"])
    if recheck:
        current, _ = compute_dataset_hash(cfg)
        if current != recorded:
            raise DatasetNotValidatedError(
                "El dataset cambio desde la validacion (dataset_hash distinto).\n"
                f"  registrado: {recorded}\n  actual:     {current}\n"
                "Repita 01_validate_dataset.py (y, si ya existia, NO regenere el split: verifique el cambio)."
            )
    return recorded
