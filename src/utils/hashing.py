"""Hashes SHA-256 reproducibles entre sistemas operativos (Windows/Linux).

Claves de diseno
----------------
* Los ficheros de texto (labels YOLO, YAML, codigo) se hashean **normalizando saltos de linea**
  (``\\r\\n`` -> ``\\n``) para que las 3 PCs obtengan el mismo hash aunque git o el editor
  conviertan finales de linea.
* Las imagenes se hashean byte a byte (sin normalizar).
* ``dataset_hash`` y ``split_hash`` se calculan sobre representaciones canonicas ordenadas, no
  sobre el orden del sistema de ficheros.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

_CHUNK = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hexadecimal de ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 de ``text`` codificado en UTF-8 (sin normalizar)."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    """SHA-256 de un fichero leido por bloques (byte a byte)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text_bytes(data: bytes) -> bytes:
    """Normaliza saltos de linea a ``\\n`` y elimina espacios finales y BOM UTF-8."""
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return data.rstrip() + b"\n" if data.strip() else b""


def sha256_text_file(path: str | Path) -> str:
    """SHA-256 de un fichero de texto con saltos de linea normalizados."""
    return sha256_bytes(normalize_text_bytes(Path(path).read_bytes()))


def canonical_json(obj: Any) -> str:
    """Serializacion JSON determinista (claves ordenadas, sin espacios)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_json(obj: Any) -> str:
    """SHA-256 de la serializacion JSON canonica de ``obj``."""
    return sha256_text(canonical_json(obj))


def compute_dataset_hash(records: Iterable[tuple[str, str, str]]) -> str:
    """Hash del dataset a partir de ``(image_id, image_sha256, label_sha256)``.

    El orden de entrada es irrelevante (se ordena por ``image_id``).
    """
    lines = [f"{iid}\t{ish}\t{lsh}" for iid, ish, lsh in sorted(records, key=lambda r: r[0])]
    return sha256_text("\n".join(lines) + "\n")


def compute_split_hash(assignments: Iterable[tuple[str, str]]) -> str:
    """Hash canonico de una asignacion ``(image_id, outer_split)``."""
    lines = [f"{iid},{split}" for iid, split in sorted(assignments, key=lambda r: r[0])]
    return sha256_text("\n".join(lines) + "\n")


def compute_code_hash(
    root: str | Path,
    patterns: Sequence[str] = ("src/**/*.py", "tools/*.py", "scripts/*", "configs/*.yaml", "configs/*.json", "requirements*.txt"),
) -> str:
    """Hash del codigo (contenido normalizado + ruta relativa) bajo ``root``."""
    root = Path(root)
    files: set[Path] = set()
    for pat in patterns:
        files.update(p for p in root.glob(pat) if p.is_file() and "__pycache__" not in p.parts)
    parts = [f"{p.relative_to(root).as_posix()}\t{sha256_text_file(p)}" for p in sorted(files)]
    return sha256_text("\n".join(parts) + "\n")


def write_checksum_file(target: str | Path, checksum_path: str | Path) -> str:
    """Escribe ``<sha256>  <nombre>`` (compatible con ``sha256sum -c``) y devuelve el hash.

    El hash se calcula byte a byte sobre ``target`` (por eso los CSV se escriben siempre
    con ``\\n``).
    """
    target = Path(target)
    digest = sha256_file(target)
    Path(checksum_path).write_text(f"{digest}  {target.name}\n", encoding="utf-8", newline="\n")
    return digest


def read_checksum_file(checksum_path: str | Path) -> str:
    """Lee el hash de un fichero ``.sha256`` (primera palabra de la primera linea)."""
    text = Path(checksum_path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Fichero de checksum vacio: {checksum_path}")
    return text.split()[0].lower()


def verify_checksum_file(target: str | Path, checksum_path: str | Path) -> bool:
    """``True`` si el SHA-256 de ``target`` coincide con el registrado en ``checksum_path``."""
    if not Path(target).is_file() or not Path(checksum_path).is_file():
        return False
    return sha256_file(target) == read_checksum_file(checksum_path)


# --------------------------------------------------------------------------- #
# Hash perceptual (deteccion de casi-duplicados)
# --------------------------------------------------------------------------- #
def image_dhash(image: Any, hash_size: int = 8) -> int:
    """dHash (difference hash) de 64 bits para una imagen PIL.

    Args:
        image: ``PIL.Image.Image``.
        hash_size: Lado de la rejilla (8 -> 64 bits).
    """
    from PIL import Image  # import local: PIL solo hace falta aqui

    small = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    px = list(small.getdata())
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = px[row * (hash_size + 1) + col]
            right = px[row * (hash_size + 1) + col + 1]
            bits = (bits << 1) | int(left > right)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Distancia de Hamming entre dos enteros."""
    return bin(a ^ b).count("1")


def scientific_config_hash(cfg):
    """Scientific protocol identity, portable between manually operated PCs."""
    from utils.semantics import semantic_fingerprint
    return sha256_json({"semantics":semantic_fingerprint(cfg), **{k:cfg.get(k) for k in ("detector","classifier","robustness","inference_tuning","matching","selection","initialization","benchmark")}})
