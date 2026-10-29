"""Pruebas aisladas de archivos/SQLite/colas/reloj. Sin CUDA, modelos ni dataset real."""
from __future__ import annotations
import json
import os
import sqlite3
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
from orchestration import exchange as ex
from orchestration.pc_registry import PCRegistry
from orchestration.study_queue import StudyQueue, COMPLETE, FAILED
from orchestration.launch import schedule_times
from utils.config_loader import load_config, FixedDecisionViolation
from utils.durable import FileLock, atomic_json
from utils.safety import read_policy

H = {k: str(i) * 64 for i, k in enumerate(("code_hash", "science_hash", "dataset_hash", "split_hash",
     "inner_folds_detection", "inner_folds_classification", "classification_manifest", "fleet_hash"), 1)}
MAP = {"PC-01": "i21315", "PC-02": "i21314", "PC-03": "i21313"}

def config_at(base):
    return load_config(None, [f"paths.artifacts_dir={base / 'artifacts'}",
        f"paths.queue_dir={base / 'artifacts/queue'}", f"paths.optuna_dir={base / 'artifacts/optuna'}",
        f"paths.environments_dir={base / 'artifacts/environments'}", f"paths.runs_dir={base / 'runs'}",
        f"paths.best_configs_dir={base / 'artifacts/best_configs'}"])

@pytest.fixture
def fleet(tmp_path, monkeypatch):
    source, dest = config_at(tmp_path / "source"), config_at(tmp_path / "dest")
    source["_test_pc"], dest["_test_pc"] = "PC-01", "PC-02"
    monkeypatch.setattr(ex, "hashes", lambda cfg: dict(H))
    monkeypatch.setattr(ex, "fleet_mapping", lambda cfg: MAP)
    monkeypatch.setattr(ex, "read_binding", lambda cfg: {"pc_id": cfg["_test_pc"], "machine_id": MAP[cfg["_test_pc"]]})
    return source, dest

def make_publication(cfg, value="first"):
    art = Path(cfg["paths"]["artifacts_dir"])
    runs = Path(cfg["paths"]["runs_dir"])
    out = runs / "yolov8n/baseline/summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, {"value": value})
    db = art / "optuna/botrytis_yolov8n.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE IF NOT EXISTS scientific (value TEXT)")
        c.execute("INSERT INTO scientific VALUES (?)", (value,))
    ex.publish_architecture(cfg, "PC-01", "yolov8n", runs, stage="baseline")
    return out, db

def pack(cfg, path):
    return ex.pack_results(cfg, path, "PC-01", Path(cfg["paths"]["runs_dir"]), with_checkpoints=True)

def import_into(cfg, path):
    return ex.unpack(cfg, path, path.parent, runs_dir=Path(cfg["paths"]["runs_dir"]), force=False)

def test_three_independent_queues_have_disjoint_ownership(tmp_path):
    allocated = []
    for pc in ("PC-01", "PC-02", "PC-03"):
        cfg = config_at(tmp_path / pc)
        q = StudyQueue.from_cfg(cfg)
        q.plan(cfg)
        jobs = []
        while True:
            job = q.claim_next(pc, kinds=["baseline"])
            if job is None:
                break
            jobs.append(job.architecture)
            q.finish(job.job_id, COMPLETE)
        allocated.append(set(jobs))
    assert set.union(*allocated) == set(PCRegistry(config_at(tmp_path)).ownership())
    assert all(not (a & b) for i, a in enumerate(allocated) for b in allocated[i + 1:])
    assert list(map(len, allocated)) == [3, 2, 2]

def test_disabled_architectures_cannot_be_reenabled():
    for architecture in ("YOLOv8m", "YOLO26n", "YOLO26s"):
        with pytest.raises(FixedDecisionViolation):
            load_config(None, [f"detector.architectures=[{architecture}]"])
    assert PCRegistry(load_config()).owner("YOLO26l") == "PC-02"
    with pytest.raises(FixedDecisionViolation, match="microbatch"):
        load_config(None, ["classifier.oom_fallback=true"])

def test_plan_empty_selection_and_bounded_retries(tmp_path):
    cfg = config_at(tmp_path)
    q = StudyQueue.from_cfg(cfg)
    assert q.plan(cfg, []) == 0
    q.plan(cfg)
    for attempt in range(3):
        job = q.claim_next("PC-01", slugs=["yolov8n"])
        assert job.attempt == attempt + 1
        q.finish(job.job_id, FAILED)
        retried = q.retry_failed()
        assert retried == (1 if attempt < 2 else 0)
    assert q.claim_next("PC-01", slugs=["yolov8n"]) is None

def test_lock_is_exclusive_and_release_is_recoverable(tmp_path):
    p = tmp_path / "test.guard"
    with FileLock(p):
        with pytest.raises(RuntimeError):
            with FileLock(p):
                pytest.fail("doble lock")
    with FileLock(p):
        pass

def test_sqlite_backup_captures_wal_and_not_uncommitted_data(tmp_path):
    src, out = tmp_path / "live.db", tmp_path / "backup.db"
    conn = sqlite3.connect(src)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE sample(value)")
    conn.execute("INSERT INTO sample VALUES ('committed')")
    conn.commit()
    conn.execute("INSERT INTO sample VALUES ('uncommitted')")
    ex.sqlite_backup(src, out)
    with sqlite3.connect(out) as c:
        assert c.execute("SELECT value FROM sample").fetchall() == [("committed",)]
        assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    conn.rollback()
    conn.close()

def test_export_uses_immutable_publication_not_active_files(fleet, tmp_path):
    src, _ = fleet
    path, db = make_publication(src)
    before = path.read_bytes()
    path.write_text("partially rewritten active output", encoding="utf-8")
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO scientific VALUES ('active-not-published')")
    out = tmp_path / "snapshot.zip"
    man = pack(src, out)
    with zipfile.ZipFile(out) as z:
        assert z.read("runs/yolov8n/baseline/summary.json") == before
        assert not any("queue.sqlite" in n or n.startswith("dataset/") or n.endswith((".cache", ".log.tmp")) for n in z.namelist())
        received_db = tmp_path / "snapshot.db"
        received_db.write_bytes(z.read("artifacts/optuna/botrytis_yolov8n.db"))
    with sqlite3.connect(received_db) as c:
        assert c.execute("SELECT value FROM scientific").fetchall() == [("first",)]
    assert man["live_epoch_checkpoint_included"] is False
    assert ex.verify_zip(out)["schema"] == 2
    with pytest.raises(FileExistsError):
        pack(src, out)

def test_import_idempotence_authoritative_updates_and_local_queue_survives(fleet, tmp_path):
    src, dst = fleet
    make_publication(src)
    q = StudyQueue.from_cfg(dst)
    q.plan(dst)
    local_job = q.claim_next("PC-02")
    out = tmp_path / "first.zip"
    pack(src, out)
    assert import_into(dst, out)["copied"] > 0
    assert import_into(dst, out)["idempotent"]
    assert q.get(local_job.job_id).status == "RUNNING"
    target = Path(dst["paths"]["runs_dir"]) / "yolov8n/baseline/summary.json"
    make_publication(src, "second")
    out2 = tmp_path / "second.zip"
    pack(src, out2)
    import_into(dst, out2)
    assert json.loads(target.read_text())["value"] == "second"
    # Otro contenido local valido no se destruye, ni siquiera aunque el origen sea correcto.
    target.write_text('{"local":true}', encoding="utf-8")
    make_publication(src, "third")
    out3 = tmp_path / "third.zip"
    pack(src, out3)
    with pytest.raises(RuntimeError, match="Conflictos"):
        import_into(dst, out3)
    assert target.read_text() == '{"local":true}'

def test_failed_import_recovers_from_journal_without_duplicate_results(fleet, tmp_path, monkeypatch):
    src, dst = fleet
    make_publication(src)
    out = tmp_path / "bundle.zip"
    pack(src, out)
    original = ex.atomic_copy
    def simulate_power_loss(source, target):
        if "received" in str(source):
            raise OSError("simulated interrupted copy")
        return original(source, target)
    monkeypatch.setattr(ex, "atomic_copy", simulate_power_loss)
    with pytest.raises(OSError):
        import_into(dst, out)
    with pytest.raises(RuntimeError, match="pendiente"):
        ex.assert_no_pending_import(dst)
    monkeypatch.setattr(ex, "atomic_copy", original)
    ex.recover_import(dst, Path(dst["paths"]["runs_dir"]))
    ex.assert_no_pending_import(dst)
    assert import_into(dst, out)["idempotent"]

@pytest.mark.parametrize("name", ["../escape", "/absolute", "artifacts/a:stream", "runs\\escape", "runs/CON.txt",
                                 "runs/foo./x", "runs//x", "runs/a/../../outside"])
def test_malicious_archive_paths_are_rejected(name):
    with pytest.raises(ValueError):
        ex._safe_name(name)

def test_missing_checksum_corruption_and_wrong_compatibility_are_rejected(fleet, tmp_path, monkeypatch):
    src, dst = fleet
    make_publication(src)
    out = tmp_path / "bundle.zip"
    pack(src, out)
    side = out.with_suffix(".zip.sha256")
    contents = side.read_bytes()
    side.unlink()
    with pytest.raises(ValueError, match="checksum"):
        ex.verify_zip(out)
    side.write_bytes(contents)
    original_hashes = ex.hashes
    monkeypatch.setattr(ex, "hashes", lambda cfg: {**H, "dataset_hash": "f" * 64} if cfg is dst else original_hashes(cfg))
    with pytest.raises(ValueError, match="incompatible"):
        import_into(dst, out)
    data = bytearray(out.read_bytes())
    data[len(data) // 2] ^= 255
    out.write_bytes(data)
    with pytest.raises(ValueError):
        ex.verify_zip(out)

def test_no_final_zip_is_visible_when_validation_fails(fleet, tmp_path, monkeypatch):
    src, _ = fleet
    make_publication(src)
    def fail(*args, **kwargs):
        raise OSError("simulated disk validation failure")
    monkeypatch.setattr(ex, "verify_zip", fail)
    out = tmp_path / "broken.zip"
    with pytest.raises(OSError):
        pack(src, out)
    assert not out.exists()
    assert list(tmp_path.glob("broken.zip.*.partial"))

@pytest.mark.parametrize("hour", [8, 21, 22, 23])
def test_schedule_always_uses_start_day_and_late_starts_do_not_roll_over(hour):
    start = datetime(2026, 10, 28, hour, 15, tzinfo=timezone(timedelta(hours=-5)))
    due, deadline = schedule_times(start)
    assert due.date() == deadline.date() == start.date()
    assert due.hour == 21 and deadline.hour == 22
    if hour >= 22:
        assert deadline < start


@pytest.mark.skipif(os.name != "nt", reason="generador de tareas de este perfil requiere Windows")
@pytest.mark.parametrize("exit_code", [0, 1])
def test_task_xml_structure_and_registration_failures_are_recorded_without_registering(tmp_path, monkeypatch, exit_code):
    import xml.etree.ElementTree as ET
    from types import SimpleNamespace
    from orchestration import launch
    atomic_json(tmp_path / "session.json", {"session_id": "SYNTHETIC", "export_at": "2026-10-28T21:00:00-05:00"})
    def fake(command, **kwargs):
        if command[0] == "whoami":
            return SimpleNamespace(stdout='"fixture","S-1-5-21-123-456-789-1000"\n')
        assert command[0] == "schtasks"
        return SimpleNamespace(returncode=exit_code, stdout="MOCK_NO_REGISTRATION", stderr="simulated" if exit_code else "")
    monkeypatch.setattr(launch.subprocess, "run", fake)
    result = launch.register_windows_task(tmp_path / "session.json")
    assert result["status"] == ("PASS" if exit_code == 0 else "WARN")
    root = ET.parse(tmp_path / "windows_task.xml").getroot()
    assert root.attrib["version"] == "1.3"
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    trigger = root.find("t:Triggers/t:TimeTrigger", ns)
    assert [e.tag.rsplit("}", 1)[-1] for e in trigger] == ["Enabled", "StartBoundary"]

def test_safety_policy_missing_does_not_assume_limits(tmp_path):
    cfg = config_at(tmp_path)
    cfg["operations"] = {"safety_policy_file": str(tmp_path / "not-verified.json")}
    with pytest.raises(FileNotFoundError):
        read_policy(cfg)


def test_scientific_fingerprint_is_portable_but_detects_effective_decisions(tmp_path):
    from utils.hashing import scientific_config_hash
    a, b = config_at(tmp_path / "one"), config_at(tmp_path / "two")
    b["compute"]["pc_id"] = "PC-03"
    b["decisions"]["confirmed_by_user"] = True
    assert scientific_config_hash(a) == scientific_config_hash(b)
    b["classes"]["fruit_class_id"] = 123
    assert scientific_config_hash(a) != scientific_config_hash(b)
    b["classes"]["fruit_class_id"] = a["classes"]["fruit_class_id"]
    b["classifier"]["workers"] += 1
    assert scientific_config_hash(a) != scientific_config_hash(b)


def test_study_identity_blocks_mixing_existing_trials():
    import optuna
    from utils.optuna_storage import assert_study_identity
    study = optuna.create_study()
    assert_study_identity(study, H, initialize=True)
    trial = study.ask()
    study.tell(trial, 0.5)  # fixture de almacenamiento, no metrica de un modelo
    assert_study_identity(study, H)
    with pytest.raises(ValueError, match="science_hash"):
        assert_study_identity(study, {**H, "science_hash": "changed"})
    legacy = optuna.create_study()
    trial = legacy.ask()
    legacy.tell(trial, 0.5)
    with pytest.raises(ValueError, match="incompatible"):
        assert_study_identity(legacy, H, initialize=True)


def test_checkpoint_provenance_and_complete_candidates_are_required(tmp_path):
    from utils.architectures import archs
    from utils.hpo_results import final_checkpoint, best_config_path, stage_json_path, require_final_checkpoint
    from utils.selection import assert_candidates_ready
    from utils.hashing import sha256_file
    from utils.checks import MethodologyViolation
    cfg = config_at(tmp_path)
    for spec in archs("classification"):
        checkpoint = final_checkpoint(cfg, spec)
        checkpoint.write_bytes(b"SYNTHETIC-NOT-A-TORCH-CHECKPOINT")
        atomic_json(checkpoint.with_name("final_meta.json"), {**H, "config_id": "fixture", "sha256": sha256_file(checkpoint)})
        from utils.semantics import semantic_fingerprint
        atomic_json(best_config_path(cfg, spec), {"config_id": "fixture", "semantic_fingerprint":semantic_fingerprint(cfg)})
        atomic_json(stage_json_path(cfg, "outer_validation", spec), {**H, "checkpoint_sha256": sha256_file(checkpoint)})
        atomic_json(stage_json_path(cfg, "inference_thresholds", spec),
                    {**H, "checkpoint_sha256": sha256_file(checkpoint), "recall": 0.5, "test_used": False})
    assert_candidates_ready(cfg, "classification", H)
    spec = archs("classification")[0]
    with pytest.raises(MethodologyViolation, match="science_hash"):
        require_final_checkpoint(cfg, spec, {**H, "science_hash": "changed"})
    stage_json_path(cfg, "inference_thresholds", spec).unlink()
    with pytest.raises(FileNotFoundError):
        assert_candidates_ready(cfg, "classification", H)


def test_freeze_detects_effective_config_change_without_importing_ml(tmp_path):
    from utils.freeze import verify_freeze
    from utils.hashing import scientific_config_hash, sha256_file
    from utils.checks import MethodologyViolation
    from test_pipeline_and_freeze import _freeze
    cfg, context, path = _freeze(tmp_path)
    from utils.io import read_json
    frozen=read_json(path)
    verify_freeze(cfg,frozen,context['hashes'],context['env'])
    cfg['classifier']['workers'] += 1
    with pytest.raises(MethodologyViolation,match='Protocolo'):
        verify_freeze(cfg,frozen,context['hashes'],context['env'])

@pytest.mark.parametrize("condition", ["gpu_hot", "ram_low", "cpu_hot", "cpu_unobservable", "wrong_device", "sensor_missing"])
def test_safety_blocks_verified_failure_conditions_without_real_hardware(tmp_path, monkeypatch, condition):
    import psutil
    from types import SimpleNamespace
    from utils import safety
    cfg = config_at(tmp_path)
    policy = {"gpu_uuid": "GPU-SYNTHETIC", "gpu_model": "SYNTHETIC",
              "max_gpu_temperature_c": 70, "gpu_limit_source": "synthetic fixture, NOT hardware recommendation",
              "min_available_ram_bytes": 10, "min_free_disk_bytes": 10,
              "max_cpu_temperature_c": 70, "cpu_thermal_protection_verified": True,
              "cpu_protection_source": "synthetic fixture", "cpu_sensor_unavailable_acknowledged": False}
    cfg["operations"] = {"safety_policy_file": str(tmp_path / "policy.json")}
    atomic_json(cfg["operations"]["safety_policy_file"], policy)
    monkeypatch.setattr(safety.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="OTHER" if condition == "wrong_device" else "GPU-SYNTHETIC"))
    def reading(uuid):
        if condition == "sensor_missing":
            raise ValueError("sensor unavailable")
        return {"name": "SYNTHETIC", "temperature.gpu": 71 if condition == "gpu_hot" else 30}
    monkeypatch.setattr(safety, "gpu_reading", reading)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=1 if condition == "ram_low" else 100))
    monkeypatch.setattr(psutil, "disk_usage", lambda path: SimpleNamespace(free=100))
    monkeypatch.setattr(psutil, "cpu_percent", lambda **kw: 0)
    monkeypatch.setattr(psutil, "sensors_temperatures",
                        lambda: {} if condition == "cpu_unobservable" else {"cpu": [SimpleNamespace(current=71 if condition == "cpu_hot" else 30)]},
                        raising=False)
    log = tmp_path / "safety.jsonl"
    with pytest.raises((RuntimeError, ValueError)):
        safety.SafetyMonitor(cfg, log).check()
    assert json.loads(log.read_text())["status"] == "BLOCKED"

@pytest.mark.parametrize("hour,expected", [(21, 0), (23, 3)])
def test_watcher_publishes_same_day_and_records_late_deadline(fleet, tmp_path, monkeypatch, hour, expected):
    import yaml
    from orchestration import launch
    src, _ = fleet
    make_publication(src)
    directory = tmp_path / "session"
    directory.mkdir()
    fixed = datetime(2026, 10, 28, hour, 0, tzinfo=timezone(timedelta(hours=-5)))
    start = fixed.replace(hour=18)
    due, deadline = schedule_times(start)
    atomic_json(directory / "session.json", {"session_id": "synthetic-session", "started_at": start.isoformat(),
        "export_at": due.isoformat(), "deadline_at": deadline.isoformat(), "machine_id": "i21315",
        "utc_offset_minutes": -300, "hashes": H})
    (directory / "effective_config.yaml").write_text(yaml.safe_dump(src), encoding="utf-8")
    monkeypatch.setattr(launch, "now", lambda: fixed)
    monkeypatch.setattr(launch, "hashes", lambda cfg: dict(H))
    monkeypatch.setattr(launch, "read_binding", lambda cfg: {"pc_id": "PC-01", "machine_id": "i21315"})
    monkeypatch.setattr(launch, "machine_paths", lambda mid: {"export_dir": tmp_path / "exports"})
    assert launch.watch(directory / "session.json") == expected
    status = json.loads((directory / "export_status.json").read_text())
    assert status["zip_valid"]
    assert status["before_deadline"] == (hour < 22)
    assert bool(status["incidents"]) == (hour >= 22)
    assert ex.verify_zip(Path(status["zip"]))["kind"] == "results"
