"""Entrada/salida comun: ficheros atomicos, CSV/JSON/YAML deterministas e imagenes.

Todos los CSV se escriben en UTF-8 con ``\\n`` (nunca ``\\r\\n``) para que el SHA-256 sea
identico en Windows y Linux.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageOps


# --------------------------------------------------------------------------- #
# Directorios y escritura atomica
# --------------------------------------------------------------------------- #
def ensure_dir(path: str | Path) -> Path:
    """Crea el directorio (y padres) si no existe y lo devuelve."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Escribe texto UTF-8 con ``\\n`` de forma atomica (tmp + replace)."""
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def to_posix(path: str | Path, start: str | Path | None = None) -> str:
    """Ruta con ``/`` (opcionalmente relativa a ``start``) para manifiestos portables."""
    p = Path(path)
    if start is not None:
        try:
            p = p.relative_to(start)
        except ValueError:
            p = Path(os.path.relpath(p, start))
    return p.as_posix()


# --------------------------------------------------------------------------- #
# JSON / YAML / CSV
# --------------------------------------------------------------------------- #
def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def write_json(path: str | Path, data: Any, *, indent: int = 2, sort_keys: bool = False) -> Path:
    """Escribe JSON UTF-8 (``NaN``/``inf`` -> ``null``) de forma atomica."""
    if isinstance(data, dict):
        import contextvars
        try:
            from utils.logging_utils import ACTIVE_CONFIG
            cfg = ACTIVE_CONFIG.get()
            if cfg is not None:
                from utils.semantics import semantic_fingerprint
                data = {**data, "semantic_fingerprint": semantic_fingerprint(cfg), "schema_version":4}
        except (ImportError,KeyError):
            pass
    text = json.dumps(_sanitize(data), indent=indent, ensure_ascii=False, sort_keys=sort_keys, default=_json_default)
    return atomic_write_text(path, text + "\n")


def _sanitize(obj: Any) -> Any:
    """Reemplaza floats no finitos (no validos en JSON estandar) por ``None``."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, np.floating):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def read_json(path: str | Path) -> Any:
    """Lee un JSON UTF-8."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_yaml(path: str | Path, data: Mapping[str, Any]) -> Path:
    """Escribe YAML UTF-8 de forma atomica."""
    text = yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=True, default_flow_style=False)
    return atomic_write_text(path, text)


def read_yaml(path: str | Path) -> Any:
    """Lee un YAML UTF-8."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_csv(df: pd.DataFrame, path: str | Path, *, float_format: str | None = None) -> Path:
    """Escribe un DataFrame como CSV determinista (UTF-8, ``\\n``, sin indice) de forma atomica."""
    text = df.to_csv(index=False, lineterminator="\n", float_format=float_format)
    return atomic_write_text(path, text)


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Lee un CSV forzando ``image_id`` a texto (evita perder ceros a la izquierda)."""
    dtype = dict(kwargs.pop("dtype", {}) or {})
    dtype.setdefault("image_id", str)
    return pd.read_csv(path, dtype=dtype, keep_default_na=True, **kwargs)


# --------------------------------------------------------------------------- #
# Imagenes
# --------------------------------------------------------------------------- #
def image_orientation(path: str | Path) -> int:
    """Valor EXIF de orientacion (1 = normal). Devuelve 1 si no hay EXIF."""
    with Image.open(path) as im:
        try:
            return int(im.getexif().get(0x0112, 1) or 1)
        except Exception:  # EXIF corrupto: se trata como normal
            return 1


def get_image_size(path: str | Path) -> tuple[int, int]:
    """``(ancho, alto)`` tras aplicar la orientacion EXIF (sin decodificar los pixeles).

    Label Studio y Ultralytics muestran/cargan las imagenes ya rotadas segun EXIF, por lo que
    las coordenadas YOLO normalizadas se refieren a la imagen **orientada**.
    """
    with Image.open(path) as im:
        w, h = im.size
        try:
            orientation = int(im.getexif().get(0x0112, 1) or 1)
        except Exception:
            orientation = 1
    if orientation in (5, 6, 7, 8):
        w, h = h, w
    return int(w), int(h)


def load_image_rgb(path: str | Path) -> Image.Image:
    """Carga una imagen como RGB aplicando la orientacion EXIF; libera el fichero."""
    with Image.open(path) as im:
        im.load()
        out = ImageOps.exif_transpose(im)
        return out.convert("RGB") if out.mode != "RGB" else out.copy()


def save_image(image: Image.Image, path: str | Path, *, fmt: str = "png", jpg_quality: int = 95) -> Path:
    """Guarda una imagen PIL (PNG lossless o JPEG)."""
    path = Path(path)
    ensure_dir(path.parent)
    if fmt.lower() in ("jpg", "jpeg"):
        image.convert("RGB").save(path, format="JPEG", quality=jpg_quality, subsampling=0)
    else:
        image.save(path, format="PNG", compress_level=6)
    return path


# --------------------------------------------------------------------------- #
# Copia de ficheros
# --------------------------------------------------------------------------- #
def link_or_copy(src: str | Path, dst: str | Path, mode: str = "copy") -> str:
    """Copia (o enlaza duro) ``src`` en ``dst``.

    Args:
        mode: ``copy`` | ``hardlink``. ``hardlink`` cae a ``copy`` si falla (p. ej. otro volumen).

    Returns:
        Modo realmente usado.
    """
    src, dst = Path(src), Path(dst)
    ensure_dir(dst.parent)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def clear_directory(path: str | Path, suffixes: Iterable[str] | None = None) -> int:
    """Elimina ficheros (recursivo) de ``path``; devuelve cuantos se borraron.

    Args:
        suffixes: Si se indica, solo se borran ficheros con esas extensiones.
    """
    path = Path(path)
    n = 0
    if not path.exists():
        return 0
    sfx = {s.lower() for s in suffixes} if suffixes else None
    for p in path.rglob("*"):
        if p.is_file() and (sfx is None or p.suffix.lower() in sfx):
            p.unlink()
            n += 1
    return n
