"""Split externo 70/20/10 por ``image_id``, manifiesto inmutable y bloqueo del test (sec. 5).

La unidad del split es SIEMPRE la imagen original. Los frutos y crops heredan el split de su
imagen; jamas se dividen crops.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from utils.config_loader import get_nested, resolve_path
from utils.hashing import compute_split_hash, sha256_text, verify_checksum_file, write_checksum_file
from utils.io import read_csv, write_csv, write_json

SPLITS: tuple[str, ...] = ("train", "validation", "test")
MANIFEST_COLUMNS: list[str] = ["image_id", "outer_split", "split_hash"]


class TestSetLockedError(RuntimeError):
    """Se intento acceder al split de test (o al validation externo) fuera de la fase permitida."""

    __test__ = False  # evita que pytest lo recolecte como clase de tests


class SplitIntegrityError(RuntimeError):
    """El manifiesto de split fue alterado o es inconsistente."""


# --------------------------------------------------------------------------- #
# Asignacion
# --------------------------------------------------------------------------- #
def allocate_counts(n: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """Reparte ``n`` items entre splits por el metodo del mayor resto (aritmetica exacta).

    El desempate entre restos iguales sigue el orden ``train > validation > test``.
    El mayor resto conserva el total descubierto, sin constantes del dataset.
    """
    exact = {s: Fraction(str(ratios[s])) * n for s in SPLITS}
    counts = {s: int(exact[s]) for s in SPLITS}
    leftover = n - sum(counts.values())
    order = sorted(SPLITS, key=lambda s: (-(exact[s] - counts[s]), SPLITS.index(s)))
    for s in order[:leftover]:
        counts[s] += 1
    return counts


def _allocate_group(size: int, remaining: Mapping[str, int]) -> dict[str, int]:
    """Cuota de un estrato proporcional a la capacidad restante de cada split."""
    total = sum(remaining.values())
    if size > total:
        raise ValueError("Estrato mayor que la capacidad restante (error interno del split)")
    exact = {s: Fraction(size * remaining[s], total) for s in SPLITS}
    quota = {s: int(exact[s]) for s in SPLITS}
    leftover = size - sum(quota.values())
    order = sorted(SPLITS, key=lambda s: (-(exact[s] - quota[s]), SPLITS.index(s)))
    for s in order:
        if leftover == 0:
            break
        if quota[s] < remaining[s]:
            quota[s] += 1
            leftover -= 1
    if leftover:
        raise ValueError("No se pudo asignar el estrato completo (error interno del split)")
    return quota


def assign_outer_split(
    image_ids: Sequence[str],
    ratios: Mapping[str, float],
    seed: int,
    strata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Asigna cada ``image_id`` a un split, una sola vez y de forma determinista.

    Args:
        image_ids: IDs unicos de imagen.
        ratios: ``{"train": .7, "validation": .2, "test": .1}``.
        seed: Semilla predeclarada (TODO BLOQUEANTE, sec. 5.3).
        strata: Opcional ``image_id -> estrato``. Si se da, el reparto es estratificado
            manteniendo exactamente los totales globales.

    Returns:
        ``{image_id: split}``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"La semilla del split debe ser un entero, recibido {seed!r}")
    ids = sorted(image_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("image_ids duplicados: el split exige IDs unicos")
    targets = allocate_counts(len(ids), ratios)
    rng = random.Random(seed)

    groups: dict[str, list[str]] = {}
    for iid in ids:
        key = str(strata[iid]) if strata is not None else "all"
        groups.setdefault(key, []).append(iid)
    ordered_groups = sorted(groups.items(), key=lambda kv: (len(kv[1]), kv[0]))

    remaining = dict(targets)
    assignment: dict[str, str] = {}
    for _, members in ordered_groups:
        members = sorted(members)
        rng.shuffle(members)
        quota = _allocate_group(len(members), remaining)
        pos = 0
        for s in SPLITS:
            for iid in members[pos:pos + quota[s]]:
                assignment[iid] = s
            pos += quota[s]
            remaining[s] -= quota[s]
    return assignment


# --------------------------------------------------------------------------- #
# Manifiesto
# --------------------------------------------------------------------------- #
def build_split_manifest(assignment: Mapping[str, str]) -> pd.DataFrame:
    """DataFrame ``image_id, outer_split, split_hash`` ordenado por ``image_id``."""
    split_hash = compute_split_hash(assignment.items())
    df = pd.DataFrame(
        {"image_id": sorted(assignment), "outer_split": [assignment[i] for i in sorted(assignment)]}
    )
    df["split_hash"] = split_hash
    return df[MANIFEST_COLUMNS]


def manifest_paths(artifacts_dir: Path, manifest_name: str = "split_manifest.csv") -> tuple[Path, Path]:
    """``(split_manifest.csv, split_manifest.sha256)`` dentro de ``artifacts_dir``."""
    csv_path = artifacts_dir / manifest_name
    return csv_path, csv_path.with_suffix(".sha256")


def write_split_manifest(
    manifest: pd.DataFrame, artifacts_dir: Path, manifest_name: str = "split_manifest.csv"
) -> tuple[Path, Path, str]:
    """Escribe ``split_manifest.csv`` y ``split_manifest.sha256``.

    Returns:
        ``(csv_path, sha_path, sha256_del_fichero)``.
    """
    csv_path, sha_path = manifest_paths(artifacts_dir, manifest_name)
    write_csv(manifest, csv_path)
    digest = write_checksum_file(csv_path, sha_path)
    return csv_path, sha_path, digest


def load_split_manifest(
    artifacts_dir: Path, manifest_name: str = "split_manifest.csv", *, verify: bool = True
) -> pd.DataFrame:
    """Carga el manifiesto verificando su ``.sha256`` y la coherencia de ``split_hash``.

    Raises:
        SplitIntegrityError: Manifiesto ausente, alterado o inconsistente.
    """
    csv_path, sha_path = manifest_paths(artifacts_dir, manifest_name)
    if not csv_path.is_file():
        raise SplitIntegrityError(f"No existe {csv_path}. Ejecute 02_create_outer_split.py primero.")
    if verify:
        if not sha_path.is_file():
            raise SplitIntegrityError(f"Falta {sha_path}: el manifiesto no esta sellado.")
        if not verify_checksum_file(csv_path, sha_path):
            raise SplitIntegrityError(
                f"El manifiesto {csv_path.name} NO coincide con su SHA-256 registrado: fue modificado "
                "despues de crearse. El split externo es inmutable (sec. 5.2)."
            )
    df = read_csv(csv_path)
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise SplitIntegrityError(f"Columnas ausentes en el manifiesto: {missing}")
    recomputed = compute_split_hash(zip(df["image_id"], df["outer_split"]))
    if not (df["split_hash"] == recomputed).all():
        raise SplitIntegrityError("split_hash de las filas no coincide con el recomputado desde las asignaciones.")
    if df["image_id"].duplicated().any():
        raise SplitIntegrityError("image_id duplicado en el manifiesto (una imagen en mas de un split)")
    bad = set(df["outer_split"]) - set(SPLITS)
    if bad:
        raise SplitIntegrityError(f"Splits desconocidos en el manifiesto: {sorted(bad)}")
    return df


def current_split_hash(artifacts_dir: Path, manifest_name: str = "split_manifest.csv") -> str | None:
    """``split_hash`` del manifiesto vigente (``None`` si aun no existe). No verifica."""
    csv_path, _ = manifest_paths(artifacts_dir, manifest_name)
    if not csv_path.is_file():
        return None
    df = read_csv(csv_path)
    return str(df["split_hash"].iloc[0]) if len(df) else None


# --------------------------------------------------------------------------- #
# Bloqueo del test
# --------------------------------------------------------------------------- #
def write_test_lock(artifacts_dir: Path, manifest: pd.DataFrame, dataset_hash: str | None) -> Path:
    """Crea ``artifacts/test_lock.json`` (el test queda cerrado hasta ``freeze_manifest.json``)."""
    test_ids = sorted(manifest.loc[manifest["outer_split"] == "test", "image_id"])
    payload = {
        "locked": True,
        "n_test_images": len(test_ids),
        "test_image_ids_sha256": sha256_text("\n".join(test_ids) + "\n"),
        "split_hash": str(manifest["split_hash"].iloc[0]),
        "dataset_hash": dataset_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "unlock_condition": "artifacts/freeze_manifest.json debe existir (sec. 26-27)",
        "rule": "NO usar test en HPO, seleccion de modelos ni ajuste de thresholds",
    }
    return write_json(artifacts_dir / "test_lock.json", payload)


def assert_split_allowed(split: str, stage: str, cfg: Mapping[str, Any]) -> None:
    """Guardia de acceso a splits para los modulos posteriores (06-18).

    Reglas (sec. 35):

    * ``hpo_trial``: solo ``train`` (el validation externo y el test NO entran en trials).
    * ``threshold_tuning`` / ``model_selection`` / ``pipeline_validation``: ``train`` y ``validation``.
    * ``final_test``: ``test`` unicamente si existe ``freeze_manifest.json``.

    Raises:
        TestSetLockedError: Acceso no permitido.
    """
    if split not in SPLITS:
        raise ValueError(f"Split desconocido: {split!r}")
    if split == "test":
        freeze = resolve_path(get_nested(cfg, "test.freeze_manifest", default="artifacts/freeze_manifest.json"))
        if stage != "final_test":
            raise TestSetLockedError(f"El split 'test' esta BLOQUEADO en la etapa '{stage}' (solo 'final_test').")
        if not freeze.is_file():
            raise TestSetLockedError(f"El split 'test' sigue bloqueado: no existe {freeze} (sec. 26-27).")
        try:
            from utils.freeze import load_freeze
            from utils.comparison import verify_candidates
            sealed = load_freeze(cfg)
            if not sealed.get("comparison", {}).get("candidates"):
                raise RuntimeError("Freeze sin candidatos sellados")
            verify_candidates(cfg, sealed["comparison"])
        except (RuntimeError, ValueError, KeyError, OSError) as exc:
            raise TestSetLockedError(f"Test bloqueado: freeze invalido o alterado: {exc}") from exc
        from utils.manual_gates import require_test_release
        require_test_release(cfg, freeze)
        return
    if split == "validation" and stage == "hpo_trial":
        raise TestSetLockedError("El validation externo no puede usarse dentro de trials de HPO (sec. 35).")
