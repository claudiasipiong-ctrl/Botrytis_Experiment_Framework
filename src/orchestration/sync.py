"""Sincronizacion por ZIP + Google Drive entre las 3 PCs (Estrategia_3PCs sec. 6, 10): empaquetar y verificar.

* ``pack-base``: ``sync_001_base.zip`` = dataset crudo + configs + src + requirements + manifiestos/hashes.
  NO incluye checkpoints pesados, logs temporales, caches ni datasets derivados (cada PC reconstruye 04/05 de forma
  determinista y comprueba el SHA-256 del manifiesto de crops).
* ``pack-results``: ``sync_results_<fecha>_<pc>.zip`` = runs/, artefactos de resultados (Optuna, configuraciones
  congeladas, evaluaciones, umbrales, benchmark), estado de la cola, environment y ``manifest.json``.
  ``--with-checkpoints`` agrega los ``final.pt`` (necesarios en la PC que ejecuta 10-16).
* ``unpack``: verifica SHA-256 del ZIP, tamano y SHA-256 de cada fichero, y que dataset_hash/split_hash coincidan con
  los locales; un ZIP corrupto o inconsistente NO se usa. Fusiona el estado de la cola (gana el mayor rango).

Ejemplos::

    python src/orchestration/sync.py pack-base --out sync_001_base.zip
    python src/orchestration/sync.py pack-results --pc-id PC-02 --out sync_results_20261001_pc02.zip
    python src/orchestration/sync.py unpack --zip sync_results_20261001_pc02.zip
"""

from __future__ import annotations

# Stdlib bootstrap: records import/CLI/config failures before optional dependencies.
import sys as _boot_sys
from pathlib import Path as _BootPath
_boot_root = _BootPath(__file__).resolve().parents[2]
_boot_sys.path.insert(0, str(_boot_root / "src"))
from utils import bootstrap as _bootstrap
_bootstrap.install(__file__, _boot_root)

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from orchestration.study_queue import StudyQueue  # noqa: E402
from utils.checks import MethodologyViolation  # noqa: E402
from utils.config_loader import PROJECT_ROOT, cfg_path  # noqa: E402
from utils.dataset import recorded_dataset_hash  # noqa: E402
from utils.hashing import compute_code_hash, read_checksum_file, sha256_file  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.splits import current_split_hash  # noqa: E402

SCRIPT = "sync"
BASE_ARTIFACTS = ("split_manifest.csv", "split_manifest.sha256", "split_metadata.json", "test_lock.json", "inner_folds_detection.csv", "inner_folds_detection.sha256",
                  "inner_folds_classification.csv", "inner_folds_classification.sha256", "inner_folds_metadata.json", "image_manifest.csv", "dataset_validation.json",
                  "dataset_inventory.csv", "classification_manifest.csv", "classification_manifest.sha256", "classification_stats.json",
                  "environments/environment_reference.json", "quarantine_manifest.json",
                  "pairing_audit.csv", "pairing_candidates.csv", "pairing_anomalies.csv",
                  "near_duplicate_cross_split.csv", "near_duplicate_review.json", "split_content_audit.csv", "crop_content_audit.csv")
RESULT_ARTIFACT_DIRS = ("optuna", "best_configs", "outer_validation", "inference_thresholds", "benchmark", "queue", "environments")
RESULT_ARTIFACT_FILES = ("selection_policy.yaml", "selection_policy.sha256", "selected_detector.json", "selected_classifier.json", "final_thresholds.json",
                         "pipeline_validation.json", "pipeline_validation_detections.csv", "pipeline_validation_misses.csv", "freeze_manifest.json", "freeze_manifest.sha256",
                         "test_results.json", "test_results.sha256", "master_results.csv", "provenance.json")
HEAVY = {".pt", ".pth", ".onnx", ".engine", ".cache"}


def _files(root: Path, sub: str, skip_heavy: bool) -> Iterable[Path]:
    base = root / sub
    if base.is_file():
        yield base
        return
    if base.is_dir():
        for p in sorted(base.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and not (skip_heavy and p.suffix.lower() in HEAVY) and p.name != ".gitkeep" and not p.name.endswith(".lock") \
                    and p.name != "desktop.ini":
                yield p


def _write_zip(out: Path, members: list[tuple[Path, str]], manifest: dict[str, Any]) -> dict[str, Any]:
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = [{"path": arc, "sha256": sha256_file(p), "size": p.stat().st_size} for p, arc in members]
    manifest = {**manifest, "created_at": datetime.now(timezone.utc).isoformat(), "n_files": len(entries), "files": entries}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p, arc in members:
            z.write(p, arc)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    digest = sha256_file(out)
    out.with_suffix(".zip.sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8", newline="\n")
    manifest["zip_sha256"], manifest["zip_size"] = digest, out.stat().st_size
    return manifest


def _hashes(cfg: dict[str, Any]) -> dict[str, Any]:
    art = cfg_path(cfg, "paths.artifacts_dir")
    cm = art / "classification_manifest.sha256"
    return {"dataset_hash": recorded_dataset_hash(art), "split_hash": current_split_hash(art, str(cfg["outer_split"]["manifest_name"])), "code_hash": compute_code_hash(PROJECT_ROOT),
            "classification_manifest_sha256": read_checksum_file(cm) if cm.is_file() else None}


def pack_base(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    """Crea ``sync_001_base.zip`` (dataset crudo + configs + src + manifiestos; sin derivados ni checkpoints)."""
    art = cfg_path(cfg, "paths.artifacts_dir")
    members: list[tuple[Path, str]] = []
    for sub in ("src", "configs", "tools", "tests", "requirements.txt", "requirements-ml.txt", "requirements-dev.txt", "README.md", "pytest.ini"):
        members += [(p, p.relative_to(PROJECT_ROOT).as_posix()) for p in _files(PROJECT_ROOT, sub, True)]
    members += [(art / n, f"artifacts/{n}") for n in BASE_ARTIFACTS if (art / n).is_file()]
    for key, arc in (("paths.raw_images_dir", "dataset/images"), ("paths.raw_labels_dir", "dataset/labels")):
        d = cfg_path(cfg, key)
        members += [(p, f"{arc}/{p.name}") for p in sorted(d.iterdir()) if p.is_file()]
    h = _hashes(cfg)
    hj = PROJECT_ROOT / "artifacts" / "_hashes_tmp.json"
    hj.parent.mkdir(parents=True, exist_ok=True)
    hj.write_text(json.dumps(h, indent=2), encoding="utf-8")
    members.append((hj, "hashes/hashes.json"))
    try:
        man = _write_zip(out, members, {"kind": "base", "hashes": h})
    finally:
        hj.unlink(missing_ok=True)
    return man


def pack_results(cfg: dict[str, Any], out: Path, pc_id: str, runs_dir: Path, *, with_checkpoints: bool) -> dict[str, Any]:
    """Crea ``sync_results_<fecha>_<pc>.zip`` con runs, artefactos de resultados y estado de la cola."""
    art = cfg_path(cfg, "paths.artifacts_dir")
    StudyQueue.from_cfg(cfg).export_state(art / "queue" / f"state_{pc_id.lower().replace('-', '')}.json")
    members: list[tuple[Path, str]] = []
    if runs_dir.is_dir():
        members += [(p, f"runs/{p.relative_to(runs_dir).as_posix()}") for p in _files(runs_dir.parent, runs_dir.name, True)]
    for sub in RESULT_ARTIFACT_DIRS:
        members += [(p, f"artifacts/{p.relative_to(art).as_posix()}") for p in _files(art, sub, True)
                    if not (p.name.startswith("queue.sqlite") or p.name.endswith(".sqlite-journal"))]  # la BD de la cola es LOCAL de cada PC: solo viaja su estado JSON
    members += [(art / n, f"artifacts/{n}") for n in RESULT_ARTIFACT_FILES if (art / n).is_file()]
    if with_checkpoints:
        ck = art / "checkpoints"
        members += [(p, f"artifacts/checkpoints/{p.relative_to(ck).as_posix()}") for p in sorted(ck.rglob("*")) if p.is_file()] if ck.is_dir() else []
    fig = cfg_path(cfg, "paths.figures_dir")
    members += [(p, f"reports/figures/{p.name}") for p in sorted(fig.glob("*")) if p.is_file()] if fig.is_dir() else []
    tbl = cfg_path(cfg, "paths.tables_dir")
    members += [(p, f"reports/tables/{p.name}") for p in sorted(tbl.glob("*")) if p.is_file()] if tbl.is_dir() else []
    results = cfg_path(cfg, "paths.results_dir")
    if results.is_dir():
        members += [(p, f"results/{p.relative_to(results).as_posix()}") for p in sorted(results.rglob("*")) if p.is_file()]
    return _write_zip(out, members, {"kind": "results", "pc_id": pc_id, "hashes": _hashes(cfg), "with_checkpoints": with_checkpoints})


def verify_zip(zip_path: Path) -> dict[str, Any]:
    """Verifica SHA-256 del ZIP (si hay ``.zip.sha256``), tamano y SHA-256 de cada fichero contra ``manifest.json``.

    Raises:
        MethodologyViolation: ZIP corrupto o inconsistente (NO debe usarse).
    """
    sha_file = zip_path.with_suffix(".zip.sha256")
    if sha_file.is_file() and read_checksum_file(sha_file) != sha256_file(zip_path):
        raise MethodologyViolation(f"{zip_path.name}: SHA-256 del ZIP no coincide con {sha_file.name} (corrupto o incompleto)")
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad:
            raise MethodologyViolation(f"{zip_path.name}: miembro corrupto {bad}")
        manifest = json.loads(z.read("manifest.json"))
        for e in manifest["files"]:
            info = z.getinfo(e["path"])
            if info.file_size != e["size"]:
                raise MethodologyViolation(f"{zip_path.name}: tamano distinto para {e['path']}")
            import hashlib

            h = hashlib.sha256()
            with z.open(e["path"]) as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != e["sha256"]:
                raise MethodologyViolation(f"{zip_path.name}: SHA-256 distinto para {e['path']}")
    return manifest


def unpack(cfg: dict[str, Any], zip_path: Path, dest_root: Path, *, runs_dir: Path, force: bool) -> dict[str, Any]:
    """Verifica y extrae un ZIP de resultados/base; fusiona la cola. No sobrescribe ficheros distintos salvo ``force``."""
    manifest = verify_zip(zip_path)
    local = _hashes(cfg)
    kind = manifest["kind"]
    if kind == "results":
        for key in ("dataset_hash", "split_hash"):
            theirs, mine = manifest["hashes"].get(key), local.get(key)
            if theirs and mine and theirs != mine:
                raise MethodologyViolation(f"{key} del ZIP ({theirs[:12]}...) != local ({mine[:12]}...): las PCs no comparten el mismo dataset/split")
    art = cfg_path(cfg, "paths.artifacts_dir")
    copied = skipped = 0
    conflicts: list[str] = []
    with zipfile.ZipFile(zip_path) as z:
        for e in manifest["files"]:
            arc = e["path"]
            if arc.startswith("runs/"):
                target = runs_dir / arc[len("runs/"):]
            elif arc.startswith("results/"):
                target = cfg_path(cfg, "paths.results_dir") / arc[len("results/"):]
            elif arc.startswith("dataset/images/"):
                target = cfg_path(cfg, "paths.raw_images_dir") / arc[len("dataset/images/"):]
            elif arc.startswith("dataset/labels/"):
                target = cfg_path(cfg, "paths.raw_labels_dir") / arc[len("dataset/labels/"):]
            elif arc.startswith("reports/figures/"):
                target = cfg_path(cfg, "paths.figures_dir") / arc[len("reports/figures/"):]
            elif arc.startswith("reports/tables/"):
                target = cfg_path(cfg, "paths.tables_dir") / arc[len("reports/tables/"):]
            elif arc.startswith("reports/"):
                target = cfg_path(cfg, "paths.reports_dir") / arc[len("reports/"):]
            elif arc.startswith("artifacts/"):
                target = art / arc[len("artifacts/"):]
            else:
                target = dest_root / arc
            if target.exists():
                if sha256_file(target) == e["sha256"]:
                    skipped += 1
                    continue
                if not force and not arc.startswith(("runs/", "reports/", "artifacts/optuna", "artifacts/queue", "artifacts/benchmark")):
                    conflicts.append(arc)
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(arc) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            copied += 1
    merged: list[dict[str, Any]] = []
    q = StudyQueue.from_cfg(cfg)
    for st in sorted((art / "queue").glob("state_*.json")):
        merged.append({"file": st.name, **q.merge_state(st)})
    return {"kind": kind, "copied": copied, "identical_skipped": skipped, "conflicts": conflicts, "queue_merge": merged}


# v4 uses collision-safe immutable receipts. Legacy implementations above are historical API context.
from orchestration.exchange import pack_base, pack_results, verify_zip, unpack

def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    parser.add_argument("action", choices=("pack-base", "pack-results", "unpack", "verify", "recover-import"))
    parser.add_argument("--out", default=None, help="ZIP de salida (pack-*)")
    parser.add_argument("--zip", default=None, help="ZIP a verificar/extraer (unpack/verify)")
    parser.add_argument("--with-checkpoints", action="store_true", help="incluye los checkpoints definitivos (pack-results)")
    parser.add_argument("--dest", default=None, help="raiz de destino para ficheros fuera de runs/artifacts/reports (unpack de un ZIP base)")
    parser.add_argument("--force", action="store_true", help="sobrescribe ficheros en conflicto")


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Empaqueta, verifica o extrae ZIPs de sincronizacion."""
    cfg, log = ctx.cfg, ctx.logger
    if args.action == "recover-import":
        from orchestration.exchange import _apply_journal
        _apply_journal(cfg, ctx.runs_base)
        return
    if args.action == "pack-base":
        out = Path(args.out or "sync_001_base.zip")
        man = pack_base(cfg, out)
        log.info("ZIP base: %s (%d ficheros, %.1f MB, sha256=%s)", out, man["n_files"], man["zip_size"] / 1e6, man["zip_sha256"])
    elif args.action == "pack-results":
        out = Path(args.out or f"sync_results_{datetime.now().strftime('%Y%m%d')}_{ctx.pc_id.lower().replace('-', '')}.zip")
        man = pack_results(cfg, out, ctx.pc_id, ctx.runs_base, with_checkpoints=args.with_checkpoints)
        log.info("ZIP de resultados: %s (%d ficheros, %.1f MB, sha256=%s)", out, man["n_files"], man["zip_size"] / 1e6, man["zip_sha256"])
    else:
        if not args.zip:
            raise ValueError("--zip requerido")
        z = Path(args.zip)
        if args.action == "verify":
            man = verify_zip(z)
            log.info("ZIP VALIDO: %s (%d ficheros, tipo %s)", z, man["n_files"], man["kind"])
        else:
            if args.force:
                raise RuntimeError("--force deshabilitado: preservar conflictos para revision manual")
            res = unpack(cfg, z, Path(args.dest or PROJECT_ROOT), runs_dir=ctx.runs_base)
            log.info("ZIP extraido: %s", res)
            ctx.log_metrics(res)
            if res["conflicts"]:
                log.warning("ficheros en conflicto NO sobrescritos: %s", res["conflicts"][:10])


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments))
