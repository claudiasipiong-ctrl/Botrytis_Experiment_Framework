"""Snapshots publicados y sincronizacion sin servidor.

Solo se exportan archivos confirmados e inmutables. Las colas extranjeras son
evidencia, nunca autoridad sobre una cola local. No se carga codigo ni pickle
durante la verificacion/importacion de ZIP.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
import zipfile
from contextlib import closing
from datetime import datetime, timezone, timedelta
from pathlib import Path, PurePosixPath

from orchestration.pc_registry import PCRegistry
from orchestration.machine import fleet_mapping, read_binding
from utils.config_loader import PROJECT_ROOT, cfg_path
from utils.dataset import recorded_dataset_hash
from utils.durable import FileLock, atomic_copy, atomic_json, atomic_bytes
from utils.hashing import compute_code_hash, sha256_file, sha256_json, read_checksum_file, scientific_config_hash
from utils.splits import current_split_hash

SCHEMA = 2
BASE_ARTIFACTS = (
    "split_manifest.csv", "split_manifest.sha256", "split_metadata.json", "test_lock.json",
    "inner_folds_detection.csv", "inner_folds_detection.sha256",
    "inner_folds_classification.csv", "inner_folds_classification.sha256", "inner_folds_metadata.json",
    "image_manifest.csv", "classification_manifest.csv", "classification_manifest.sha256", "classification_stats.json",
    "fleet_mapping.json", "environments/environment_reference.json",
)
GLOBAL_FILES = (
    "selection_policy.yaml", "selection_policy.sha256", "selected_detector.json",
    "selected_classifier.json", "final_thresholds.json", "pipeline_validation.json",
    "pipeline_validation_detections.csv", "pipeline_validation_misses.csv",
    "freeze_manifest.json", "freeze_manifest.sha256", "test_results.json",
    "test_results.sha256", "master_results.csv", "provenance.json",
)

def sync_root(cfg):
    return cfg_path(cfg, "paths.artifacts_dir") / "sync"

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def science_hash(cfg):
    # Rutas y hardware son locales. Workers se incluyen: el stream RNG puede depender de ellos.
    return scientific_config_hash(cfg)

def hashes(cfg):
    art = cfg_path(cfg, "paths.artifacts_dir")
    result = {"code_hash": compute_code_hash(PROJECT_ROOT), "science_hash": science_hash(cfg),
              "dataset_hash": recorded_dataset_hash(art),
              "split_hash": current_split_hash(art, str(cfg["outer_split"]["manifest_name"]))}
    for key in ("inner_folds_detection", "inner_folds_classification", "classification_manifest"):
        p = art / (key + ".sha256")
        result[key] = read_checksum_file(p) if p.is_file() else None
    p = art / "fleet_mapping.json"
    result["fleet_hash"] = sha256_json(_json(p)) if p.is_file() else None
    return result

def require_hashes(h):
    missing = [k for k, v in h.items() if not isinstance(v, str) or not v.strip()]
    if missing:
        raise RuntimeError(f"Faltan hashes obligatorios para sincronizar resultados: {missing}")

def sqlite_backup(src, dst, timeout=30):
    """Backup online consistente, con limite de tiempo y comprobacion de integridad."""
    end = time.monotonic() + timeout
    def progress(status, remaining, total):
        if time.monotonic() > end:
            raise TimeoutError(f"SQLite backup excedio {timeout}s: {src}")
    with closing(sqlite3.connect(Path(src).resolve().as_uri() + "?mode=ro", uri=True, timeout=2)) as inp:
        with closing(sqlite3.connect(dst, timeout=2)) as out:
            inp.backup(out, pages=256, progress=progress, sleep=0.05)
            if out.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError(f"Backup SQLite invalido: {src}")

def _safe_name(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\0" in name:
        raise ValueError(f"Ruta ZIP no permitida: {name!r}")
    p = PurePosixPath(name)
    if p.is_absolute() or any(x in ("", ".", "..") for x in name.split("/")):
        raise ValueError(f"Ruta ZIP insegura: {name!r}")
    for part in p.parts:
        if part.rstrip(" .") != part or part.split(".")[0].upper() in {
            "CON", "PRN", "AUX", "NUL", *("COM" + str(i) for i in range(1, 10)),
            *("LPT" + str(i) for i in range(1, 10))
        }:
            raise ValueError(f"Nombre Windows inseguro: {part}")
    return name

def _object(cfg, source, *, database=False):
    root = sync_root(cfg)
    objects = root / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    if not database:
        before = source.stat()
        digest = sha256_file(source)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Archivo activo durante publicacion: {source}")
        existing = objects / digest
        if existing.is_file():
            if sha256_file(existing) != digest:
                raise RuntimeError(f"Objeto previamente publicado corrupto: {existing}")
            return {"sha256": digest, "size": existing.stat().st_size}
    fd, temp = tempfile.mkstemp(dir=objects, suffix=".partial")
    os.close(fd)
    temp = Path(temp)
    try:
        if database:
            sqlite_backup(source, temp)
        else:
            before = source.stat()
            atomic_copy(source, temp)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Archivo activo durante publicacion: {source}")
        digest = sha256_file(temp)
        dest = objects / digest
        if not dest.exists():
            os.replace(temp, dest)
        elif sha256_file(dest) != digest:
            raise RuntimeError(f"Objeto previamente publicado corrupto: {dest}")
        return {"sha256": digest, "size": dest.stat().st_size}
    finally:
        temp.unlink(missing_ok=True)

def publish_architecture(cfg, pc_id, slug, runs_dir, *, stage, complete=True):
    """Llamar solo ENTRE jobs/trials de esta arquitectura; nunca desde un timer."""
    reg = PCRegistry(cfg)
    if reg.owner(slug) != pc_id:
        raise ValueError(f"{pc_id} no es propietario de {slug}")
    h = hashes(cfg)
    require_hashes(h)
    art = cfg_path(cfg, "paths.artifacts_dir")
    members = []
    base = Path(runs_dir) / slug
    if base.is_dir():
        for p in sorted(base.rglob("*")):
            if p.is_file() and not p.is_symlink() and p.suffix.lower() in {".json", ".jsonl", ".csv", ".yaml", ".log", ".txt", ".svg", ".pdf", ".png", ".html"}:
                if "__pycache__" not in p.parts:
                    members.append((p, "runs/" + p.relative_to(runs_dir).as_posix(), False))
    for directory in ("best_configs", "outer_validation", "inference_thresholds"):
        for p in sorted((art / directory).glob(slug + "*")):
            if p.is_file() and p.suffix in {".json", ".csv"}:
                members.append((p, f"artifacts/{directory}/{p.name}", False))
    checkpoint = art / "checkpoints" / slug / "final.pt"
    meta = checkpoint.with_name("final_meta.json")
    if checkpoint.exists() or meta.exists():
        if not checkpoint.is_file() or not meta.is_file() or _json(meta).get("sha256") != sha256_file(checkpoint):
            raise RuntimeError("Checkpoint final sin metadatos confirmados o con SHA distinto")
        members += [(checkpoint, f"artifacts/checkpoints/{slug}/final.pt", False),
                    (meta, f"artifacts/checkpoints/{slug}/final_meta.json", False)]
    from utils.architectures import get_arch
    from utils.optuna_storage import study_db_path
    db = study_db_path(cfg, get_arch(slug))
    if db.is_file():
        members.append((db, f"artifacts/optuna/{db.name}", True))
    with FileLock(sync_root(cfg) / ("publish_" + slug + ".guard")):
        entries = [{"path": _safe_name(arc), **_object(cfg, src, database=database)}
                   for src, arc, database in members]
        receipt = {"schema": SCHEMA, "pc_id": pc_id, "slug": slug, "stage": stage,
                   "stage_complete": complete, "published_at": utcnow(), "hashes": h, "files": entries}
        atomic_json(sync_root(cfg) / "published" / (slug + ".json"), receipt)
    return receipt

def _write_zip(cfg, out, entries, manifest):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() or out.with_suffix(".zip.sha256").exists():
        raise FileExistsError(f"No se sobrescribe una exportacion: {out}")
    total = sum(e["size"] for e in entries)
    if shutil.disk_usage(out.parent).free < total + (16 << 20):
        raise OSError(f"Espacio insuficiente: se requiere al menos {total + (16 << 20)} bytes libres")
    temp = out.with_name(out.name + "." + uuid.uuid4().hex + ".partial")
    man = {**manifest, "schema": SCHEMA, "created_at": utcnow(), "n_files": len(entries), "files": entries}
    started, copied, last_update = time.monotonic(), 0, 0.0
    deadline = datetime.fromisoformat(manifest["deadline_at"]) if manifest.get("deadline_at") else None
    def progress(state, *, force=False):
        nonlocal last_update
        current = time.monotonic()
        if not force and current - last_update < 1:
            return
        last_update = current
        elapsed = max(current - started, 1e-6)
        rate = copied / elapsed if copied else None
        estimate = datetime.now(timezone.utc) + timedelta(seconds=(total - copied) / rate) if rate else None
        actual_now = datetime.now(timezone.utc)
        risk = "DEADLINE_MISSED" if deadline and actual_now >= deadline else (
            "RISK_ESTIMATED" if deadline and estimate and estimate >= deadline else
            "UNMEASURED" if rate is None else "ESTIMATE_ONLY")
        atomic_json(sync_root(cfg) / "last_export_progress.json", {
            "state": state, "at": actual_now.isoformat(), "output": str(out), "deadline_at": manifest.get("deadline_at"),
            "uncompressed_total_bytes": total, "copied_bytes": copied, "elapsed_seconds": elapsed,
            "observed_bytes_per_second": rate, "estimated_copy_end": estimate.isoformat() if estimate else None,
            "deadline_risk": risk, "estimate_excludes_final_validation": True})
    progress("PREPARING", force=True)
    try:
        with zipfile.ZipFile(temp, "x", allowZip64=True) as z:
            for e in entries:
                source = sync_root(cfg) / "objects" / e["sha256"]
                compression = zipfile.ZIP_STORED if e["path"].endswith((".pt", ".pth")) else zipfile.ZIP_DEFLATED
                info = zipfile.ZipInfo(e["path"])
                info.compress_type = compression
                digest = hashlib.sha256()
                with source.open("rb") as inp, z.open(info, "w", force_zip64=True) as target:
                    for chunk in iter(lambda: inp.read(1 << 20), b""):
                        target.write(chunk)
                        digest.update(chunk)
                        copied += len(chunk)
                        progress("WRITING")
                if digest.hexdigest() != e["sha256"]:
                    raise RuntimeError(f"Objeto cambio durante exportacion: {e['path']}")
            z.writestr("manifest.json", json.dumps(man, indent=2, ensure_ascii=False), compress_type=zipfile.ZIP_DEFLATED)
        progress("VERIFYING", force=True)
        verify_zip(temp, require_sidecar=False)
        digest = sha256_file(temp)
        with temp.open("r+b") as f:
            os.fsync(f.fileno())
        # El checksum esta listo ANTES de publicar el nombre final.
        atomic_bytes(out.with_suffix(".zip.sha256"), f"{digest}  {out.name}\n".encode())
        if out.exists():
            raise FileExistsError(out)
        os.rename(temp, out)
        progress("COMPLETE", force=True)
        return {**man, "zip_sha256": digest, "zip_size": out.stat().st_size}
    except BaseException:
        # Solo .partial es visible si la operacion falla; conserva la evidencia.
        raise

def pack_results(cfg, out, pc_id, runs_dir, *, with_checkpoints=True, deadline_at=None):
    h = hashes(cfg)
    require_hashes(h)
    b = read_binding(cfg)
    if b["pc_id"] != pc_id:
        raise ValueError("pc_id distinto de la vinculacion")
    with FileLock(sync_root(cfg) / "export.guard"):
        counter_path = sync_root(cfg) / "export_counter.json"
        sequence = int(_json(counter_path)["sequence"]) + 1 if counter_path.exists() else 1
        atomic_json(counter_path, {"sequence": sequence})  # huecos son validos; nunca reutiliza
        entries = {}
        receipts = []
        for slug in PCRegistry(cfg).owned_slugs(pc_id):
            p = sync_root(cfg) / "published" / (slug + ".json")
            if not p.is_file():
                continue
            receipt = _json(p)
            if receipt["hashes"] != h or receipt["pc_id"] != pc_id:
                raise RuntimeError(f"Publicacion incompatible: {slug}; no se mezclan experimentos")
            receipts.append({k: v for k, v in receipt.items() if k != "files"})
            for entry in receipt["files"]:
                if not with_checkpoints and entry["path"].endswith(".pt"):
                    continue
                entries[entry["path"]] = entry
        # Evidencia de cola consistente (dos SELECT dentro de una transaccion); no se importa como autoridad.
        from orchestration.study_queue import StudyQueue
        q = StudyQueue.from_cfg(cfg)
        state = sync_root(cfg) / "queue_snapshot.json"
        q.export_state(state)
        arc = f"evidence/{pc_id}/queue.json"
        entries[arc] = {"path": arc, **_object(cfg, state)}
        env = cfg_path(cfg, "paths.environments_dir") / ("environment_" + pc_id.lower().replace("-", "") + ".json")
        if env.exists():
            arc = f"evidence/{pc_id}/environment.json"
            entries[arc] = {"path": arc, **_object(cfg, env)}
        # Configuracion cientifica sin rutas ni credenciales. Hash efectivo en manifiesto.
        config = sync_root(cfg) / "configuration.json"
        sanitized = {k: v for k, v in cfg.items() if k not in {"paths", "_meta", "operations", "compute"}}
        from utils.bootstrap import redact
        atomic_json(config, redact(sanitized))
        arc = f"evidence/{pc_id}/configuration.json"
        entries[arc] = {"path": arc, **_object(cfg, config)}
        return _write_zip(cfg, out, list(entries.values()), {
            "kind": "results", "pc_id": pc_id, "machine_id": b["machine_id"],
            "sequence": sequence, "hashes": h, "receipts": receipts,
            "deadline_at": deadline_at,
            "with_checkpoints": with_checkpoints, "live_epoch_checkpoint_included": False})

def pack_base(cfg, out):
    """Solo manifiestos y referencia: el dataset ZIP ya existe en las tres PCs."""
    art = cfg_path(cfg, "paths.artifacts_dir")
    entries = [{"path": "artifacts/" + n, **_object(cfg, art / n)}
               for n in BASE_ARTIFACTS if (art / n).is_file()]
    required = {"split_manifest.csv", "split_manifest.sha256", "inner_folds_detection.csv",
                "inner_folds_detection.sha256", "inner_folds_classification.csv",
                "inner_folds_classification.sha256", "inner_folds_metadata.json", "fleet_mapping.json"}
    if not required <= {e["path"].removeprefix("artifacts/") for e in entries}:
        raise RuntimeError("Base incompleta: prepare 01-05 y fleet_mapping.json primero")
    from utils.prepared import load_prepared_data
    load_prepared_data(cfg, with_folds=True)
    return _write_zip(cfg, out, entries, {"kind": "base", "hashes": hashes(cfg), "pc_id": "PC-01",
                                        "machine_id": fleet_mapping(cfg)["PC-01"], "sequence": 0})

def verify_zip(zip_path, *, require_sidecar=True):
    zip_path = Path(zip_path)
    if require_sidecar:
        side = zip_path.with_suffix(".zip.sha256")
        if not side.is_file() or read_checksum_file(side) != sha256_file(zip_path):
            raise ValueError("Falta checksum externo o SHA-256 del ZIP distinto")
    with zipfile.ZipFile(zip_path) as z:
        infos = z.infolist()
        names = [_safe_name(i.filename) for i in infos]
        if len(names) != len(set(n.casefold() for n in names)):
            raise ValueError("Miembros ZIP duplicados o colision por mayusculas")
        if any(((i.external_attr >> 16) & 0o170000) == 0o120000 for i in infos):
            raise ValueError("ZIP con symlinks")
        if "manifest.json" not in names or z.getinfo("manifest.json").file_size > 16 << 20:
            raise ValueError("Manifiesto ausente o excesivo")
        man = json.loads(z.read("manifest.json"))
        if man.get("schema") != SCHEMA or man.get("kind") not in {"base", "results"}:
            raise ValueError("Version/tipo de ZIP no soportado; el formato antiguo inseguro no se importa")
        entries = man["files"]
        if man.get("n_files") != len(entries) or len(entries) > 200000:
            raise ValueError("Conteo de miembros invalido")
        if set(names) != {"manifest.json", *(e["path"] for e in entries)} or len(entries) != len({e["path"].casefold() for e in entries}):
            raise ValueError("Miembros extra, omitidos o repetidos")
        for e in entries:
            _safe_name(e["path"])
            if not re.fullmatch("[0-9a-f]{64}", e["sha256"]) or type(e["size"]) is not int or e["size"] < 0:
                raise ValueError("Metadatos de miembro invalidos")
            info = z.getinfo(e["path"])
            if info.file_size != e["size"]:
                raise ValueError(f"Tamano distinto: {e['path']}")
            h = hashlib.sha256()
            with z.open(e["path"]) as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != e["sha256"]:
                raise ValueError(f"SHA-256 distinto: {e['path']}")
    return man

def _target(cfg, arc, runs_dir):
    _safe_name(arc)
    if arc.startswith("runs/"):
        base, rel = Path(runs_dir), arc[5:]
    elif arc.startswith("artifacts/"):
        base, rel = cfg_path(cfg, "paths.artifacts_dir"), arc[10:]
    elif arc.startswith("evidence/"):
        base, rel = sync_root(cfg) / "foreign_evidence", arc[9:]
    else:
        raise ValueError(f"Destino no permitido: {arc}")
    dest = base / rel
    if not dest.resolve().is_relative_to(base.resolve()):
        raise ValueError("Destino fuera de su raiz (incluyendo enlaces locales)")
    parent = dest
    while parent != base.parent:
        if parent.is_symlink():
            raise ValueError("Destino contiene symlink")
        if parent == base:
            break
        parent = parent.parent
    return dest

def _authorized(cfg, man, arc):
    if man["kind"] == "base":
        return man["pc_id"] == "PC-01" and arc in {"artifacts/" + n for n in BASE_ARTIFACTS}
    pc = man["pc_id"]
    if arc.startswith(f"evidence/{pc}/"):
        return arc.rsplit("/", 1)[-1] in {"queue.json", "environment.json", "configuration.json"}
    for slug, owner in PCRegistry(cfg).ownership().items():
        if pc != owner:
            continue
        if arc.startswith("runs/" + slug + "/"):
            return Path(arc).suffix in {".json", ".jsonl", ".csv", ".yaml", ".log", ".txt", ".svg", ".pdf", ".png", ".html"}
        if arc.startswith("artifacts/checkpoints/" + slug + "/"):
            return arc.rsplit("/", 1)[-1] in {"final.pt", "final_meta.json"}
        for directory in ("best_configs", "outer_validation", "inference_thresholds"):
            if arc.startswith(f"artifacts/{directory}/{slug}") and Path(arc).suffix in {".json", ".csv"}:
                return True
        prefix = str(cfg["optuna"]["study_prefix"])
        if arc == f"artifacts/optuna/{prefix}_{slug}.db":
            return True
    return False

def assert_no_pending_import(cfg):
    if (sync_root(cfg) / "import_journal.json").exists():
        raise RuntimeError("Importacion pendiente de recuperacion; ejecute sync.py recover-import antes de leer resultados")

def _apply_journal(cfg, runs_dir):
    root = sync_root(cfg)
    jp = root / "import_journal.json"
    if not jp.exists():
        return
    journal = _json(jp)
    ledger_path = root / "import_ledger.json"
    ledger = _json(ledger_path) if ledger_path.exists() else {"files": {}, "sequences": {}, "bundles": []}
    for e in journal["entries"]:
        target = _target(cfg, e["path"], runs_dir)
        current = sha256_file(target) if target.exists() else None
        if current not in (e["old_sha256"], e["sha256"]):
            raise RuntimeError(f"Conflicto durante recuperacion: {e['path']}")
        if current != e["sha256"]:
            src = root / "received" / journal["bundle"] / e["path"]
            if sha256_file(src) != e["sha256"]:
                raise RuntimeError("Staging de importacion corrupto")
            atomic_copy(src, target)
        ledger["files"][e["path"]] = {"sha256": e["sha256"], "pc_id": journal["pc_id"]}
    ledger["sequences"][journal["pc_id"]] = journal["sequence"]
    ledger["bundles"] = sorted(set(ledger["bundles"]) | {journal["bundle"]})
    atomic_json(ledger_path, ledger)
    jp.unlink()

def recover_import(cfg, runs_dir):
    with FileLock(sync_root(cfg) / "import.guard"):
        _apply_journal(cfg, runs_dir)

def unpack(cfg, zip_path, dest_root, *, runs_dir, force=False):
    if force:
        raise ValueError("--force no esta permitido: preserve y resuelva los conflictos explicitamente")
    man = verify_zip(zip_path)
    local = hashes(cfg)
    if man["kind"] == "results":
        require_hashes(local)
        require_hashes(man["hashes"])
        if local != man["hashes"]:
            raise ValueError(f"ZIP incompatible: hashes diferentes {[k for k in local if local[k] != man['hashes'].get(k)]}")
    else:
        for key in ("dataset_hash", "code_hash", "science_hash", "fleet_hash"):
            if not local.get(key) or local[key] != man["hashes"].get(key):
                raise ValueError(f"Base incompatible en {key}; valide dataset local y use mismo codigo/configuracion/mapa")
    mapping = fleet_mapping(cfg)
    if mapping.get(man["pc_id"]) != man.get("machine_id"):
        raise ValueError("Procedencia distinta del mapa declarado")
    root = sync_root(cfg)
    bundle = sha256_file(zip_path)
    with FileLock(root / "import.guard"):
        _apply_journal(cfg, runs_dir)
        ledger_path = root / "import_ledger.json"
        ledger = _json(ledger_path) if ledger_path.exists() else {"files": {}, "sequences": {}, "bundles": []}
        if bundle in ledger["bundles"]:
            return {"kind": man["kind"], "copied": 0, "idempotent": True, "conflicts": []}
        if man["sequence"] <= ledger["sequences"].get(man["pc_id"], -1):
            raise ValueError("Exportacion antigua/reordenada; no se retrocede un snapshot")
        own = read_binding(cfg)["pc_id"]
        if man["kind"] == "results" and own == man["pc_id"]:
            raise ValueError("No se importa un snapshot sobre su propia maquina propietaria")
        entries = []
        conflicts = []
        for e in man["files"]:
            if not _authorized(cfg, man, e["path"]):
                raise ValueError(f"Archivo fuera del alcance de su propietario: {e['path']}")
            target = _target(cfg, e["path"], runs_dir)
            old = sha256_file(target) if target.is_file() else None
            prior = ledger["files"].get(e["path"])
            # Solo puede actualizar su propia version previamente importada; nunca datos locales.
            if old is not None and old != e["sha256"] and not (
                prior and prior["pc_id"] == man["pc_id"] and prior["sha256"] == old):
                conflicts.append(e["path"])
            entries.append({**e, "old_sha256": old})
        if conflicts:
            raise RuntimeError(f"Conflictos: no se modifico ningun archivo: {conflicts}")
        staging = root / "received" / bundle
        staging.mkdir(parents=True, exist_ok=True)
        required = sum(e["size"] for e in entries)
        if shutil.disk_usage(root).free < required * 2 + (16 << 20):
            raise OSError("Espacio insuficiente para staging y publicacion de importacion")
        with zipfile.ZipFile(zip_path) as z:
            for e in entries:
                dest = staging / e["path"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(e["path"]) as inp, dest.open("wb") as out:
                    shutil.copyfileobj(inp, out)
                if sha256_file(dest) != e["sha256"]:
                    raise ValueError("ZIP cambio durante importacion")
        atomic_json(root / "import_journal.json", {
            "bundle": bundle, "pc_id": man["pc_id"], "sequence": man["sequence"],
            "created_at": utcnow(), "entries": entries})
        _apply_journal(cfg, runs_dir)
    return {"kind": man["kind"], "copied": len(entries), "idempotent": False, "conflicts": [],
            "queue_merge": "foreign evidence only; local authority preserved"}
