"""Inicio supervisado y exportacion de las 21:00 del MISMO dia de inicio.

El watcher independiente no detiene el worker. Se agrega una tarea Windows
con StartWhenAvailable y disparador al iniciar sesion cuando los permisos permiten
registrarla. No promete cumplir un plazo sin evidencia de disco/tamano.
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
import os
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config_loader import PROJECT_ROOT, cfg_path, load_config
from utils.durable import FileLock, atomic_json
from orchestration.machine import read_binding, machine_paths
from orchestration.exchange import hashes, require_hashes, pack_results, sync_root

def now():
    return datetime.now().astimezone()

def schedule_times(start):
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("Inicio sin zona horaria")
    return (start.replace(hour=21, minute=0, second=0, microsecond=0),
            start.replace(hour=22, minute=0, second=0, microsecond=0))

def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def event(directory, name, **data):
    with (directory / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": now().isoformat(), "event": name, **data}, ensure_ascii=False, default=str) + "\n")

def register_windows_task(session_file):
    if os.name != "nt":
        return {"status": "WARN", "reason": "Programador Windows no disponible"}
    import csv
    session = _read(session_file)
    sid_result = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True, text=True, timeout=10, check=True)
    sid = next(csv.reader(sid_result.stdout.strip().splitlines()))[1]
    namespace = "http://schemas.microsoft.com/windows/2004/02/mit/task"
    ET.register_namespace("", namespace)
    def child(parent, tag, text=None, **attrib):
        node = ET.SubElement(parent, "{" + namespace + "}" + tag, attrib)
        node.text = text
        return node
    task = ET.Element("{" + namespace + "}Task", {"version": "1.3"})
    triggers = child(task, "Triggers")
    trigger = child(triggers, "TimeTrigger")
    child(trigger, "Enabled", "true")
    child(trigger, "StartBoundary", datetime.fromisoformat(session["export_at"]).replace(tzinfo=None).isoformat())
    login = child(triggers, "LogonTrigger")
    child(login, "Enabled", "true")
    child(login, "UserId", sid)
    principal = child(child(task, "Principals"), "Principal", id="Author")
    child(principal, "UserId", sid)
    child(principal, "LogonType", "InteractiveToken")
    child(principal, "RunLevel", "LeastPrivilege")
    settings = child(task, "Settings")
    for tag, value in (("MultipleInstancesPolicy", "IgnoreNew"), ("DisallowStartIfOnBatteries", "false"),
                       ("StopIfGoingOnBatteries", "false"), ("StartWhenAvailable", "true"),
                       ("WakeToRun", "true"), ("ExecutionTimeLimit", "PT2H"), ("Enabled", "true")):
        child(settings, tag, value)
    action = child(child(task, "Actions", Context="Author"), "Exec")
    child(action, "Command", sys.executable)
    child(action, "Arguments", subprocess.list2cmdline([str(Path(__file__).resolve()), "watch", "--session", str(session_file)]))
    child(action, "WorkingDirectory", str(PROJECT_ROOT))
    xml = session_file.parent / "windows_task.xml"
    ET.ElementTree(task).write(xml, encoding="utf-16", xml_declaration=True)
    name = "Botrytis_Export_" + session["session_id"]
    result = subprocess.run(["schtasks", "/Create", "/TN", name, "/XML", str(xml)], capture_output=True, text=True, timeout=20)
    return {"status": "PASS" if result.returncode == 0 else "WARN", "task": name,
            "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

def watch(session_file, *, immediate=False):
    session_file = Path(session_file).resolve()
    directory = session_file.parent
    try:
        lock = FileLock(directory / "watch.guard").__enter__()
    except RuntimeError:
        return 0  # el watcher ya activo es el responsable
    try:
        session = _read(session_file)
        status_file = directory / "export_status.json"
        if status_file.exists() and _read(status_file).get("zip_valid"):
            return 0
        cfg = load_config(directory / "effective_config.yaml")
        binding = read_binding(cfg)
        if binding["machine_id"] != session["machine_id"] or hashes(cfg) != session["hashes"]:
            raise RuntimeError("Sesion incompatible con codigo/datos/vinculacion actuales")
        due, deadline = datetime.fromisoformat(session["export_at"]), datetime.fromisoformat(session["deadline_at"])
        start = datetime.fromisoformat(session["started_at"])
        atomic_json(directory / "watch_ready.json", {"pid": os.getpid(), "at": now().isoformat()})
        last_wall, last_mono = time.time(), time.monotonic()
        incidents = []
        while not immediate and now() < due:
            current = now()
            drift = (time.time() - last_wall) - (time.monotonic() - last_mono)
            offset = int(current.utcoffset().total_seconds() / 60)
            if abs(drift) > 120 or current < start - timedelta(seconds=60) or offset != session["utc_offset_minutes"]:
                incidents.append("CLOCK_ANOMALY: exportacion inmediata; reloj requiere revision")
                event(directory, "CLOCK_ANOMALY", drift_seconds=drift, current=current.isoformat())
                break
            atomic_json(status_file, {"state": "ARMED", "zip_valid": False, "due": due.isoformat(),
                                     "heartbeat_at": current.isoformat(), "pid": os.getpid()})
            time.sleep(min(5, max(0.1, (due - current).total_seconds())))
        if now() >= deadline:
            incidents.append("DEADLINE_ALREADY_MISSED: se exporta ahora, nunca se traslada al siguiente dia")
        elif now() > due + timedelta(seconds=60):
            incidents.append("LATE_START: inicio/reanudacion despues de las 21:00")
        event(directory, "EXPORT_STARTED", due=due, deadline=deadline, incidents=incidents)
        outdir = machine_paths(binding["machine_id"])["export_dir"]
        for attempt in range(1, 4):
            output = outdir / f"sync_results_{binding['machine_id']}_{now().strftime('%Y%m%d_%H%M%S_%f')}_{session['session_id'][:8]}.zip"
            begun = time.perf_counter()
            atomic_json(status_file, {"state": "EXPORTING", "zip_valid": False, "attempt": attempt,
                                     "started_at": now().isoformat(), "deadline_at": deadline.isoformat(), "incidents": incidents})
            try:
                result = pack_results(cfg, output, binding["pc_id"], cfg_path(cfg, "paths.runs_dir"),
                                      with_checkpoints=True, deadline_at=deadline.isoformat())
                end = now()
                before_deadline = end < deadline
                if not before_deadline:
                    incidents.append("DEADLINE_MISSED: ZIP valido terminado fuera del plazo")
                status = {"state": "COMPLETE" if before_deadline and not incidents else "COMPLETE_WITH_INCIDENT",
                          "zip_valid": True, "before_deadline": before_deadline, "zip": str(output),
                          "sha256": result["zip_sha256"], "bytes": result["zip_size"],
                          "seconds": time.perf_counter() - begun, "completed_at": end.isoformat(),
                          "incidents": incidents, "snapshot_receipts": result["receipts"]}
                atomic_json(status_file, status)
                event(directory, "EXPORT_CONFIRMED", **status)
                return 0 if before_deadline and not incidents else 3
            except Exception as exc:
                event(directory, "EXPORT_FAILED", attempt=attempt, error=repr(exc))
                atomic_json(status_file, {"state": "FAILED", "zip_valid": False, "attempt": attempt,
                                         "error": repr(exc), "incidents": incidents})
                if attempt >= 3 or now() >= deadline:
                    return 1
                time.sleep(10)
        return 1
    finally:
        lock.__exit__(None, None, None)

def launch(config, *, resume=False, clock_verified=False, new_session=False, retry_failed=False):
    cfg = load_config(config)
    b = read_binding(cfg)
    h = hashes(cfg)
    require_hashes(h)
    from utils.prepared import load_prepared_data
    load_prepared_data(cfg, with_folds=True)
    from utils.safety import SafetyMonitor
    SafetyMonitor(cfg, sync_root(cfg) / "prelaunch_telemetry.jsonl").check()
    from utils.manual_gates import require_gate
    require_gate(cfg,b["pc_id"],"READY_FOR_EXPERIMENT")
    current = sync_root(cfg) / "current_session.json"
    if resume:
        if not current.exists():
            raise RuntimeError("No existe sesion que reanudar")
        session_file = Path(_read(current)["session_file"])
        session = _read(session_file)
        if session["hashes"] != h:
            raise RuntimeError("La sesion original tiene otros hashes; no se reanuda silenciosamente")
    else:
        if current.exists() and not new_session:
            raise RuntimeError("Ya existe una sesion; use resume o --new-session explicitamente")
        if not clock_verified:
            raise RuntimeError("Verifique hora/fecha/zona en Windows y use --clock-verified")
        start = now()
        offset = int(start.utcoffset().total_seconds() / 60)
        expected = int(cfg.get("operations", {}).get("expected_utc_offset_minutes", -300))
        if offset != expected:
            raise RuntimeError(f"Zona horaria UTC{offset/60:+g}, esperada UTC{expected/60:+g}; revise el reloj")
        due, deadline = schedule_times(start)
        session_id = uuid.uuid4().hex
        directory = sync_root(cfg) / "sessions" / session_id
        directory.mkdir(parents=True)
        session_file = directory / "session.json"
        import yaml
        clean = {k: v for k, v in cfg.items() if k != "_meta"}
        (directory / "effective_config.yaml").write_text(yaml.safe_dump(clean, sort_keys=False, allow_unicode=True), encoding="utf-8")
        atomic_json(session_file, {"session_id": session_id, "started_at": start.isoformat(), "export_at": due.isoformat(),
                                  "deadline_at": deadline.isoformat(), "machine_id": b["machine_id"], "pc_id": b["pc_id"],
                                  "utc_offset_minutes": offset, "hashes": h, "clock_verified_by_user": True})
        atomic_json(current, {"session_file": str(session_file.resolve())})
    directory = session_file.parent
    task_result = {"status":"PENDIENTE_USUARIO", "reason":"Watcher local activo; programacion Windows opcional requiere invocacion personal de register_windows_task"}
    atomic_json(directory / "task_registration.json", task_result)
    print("Programacion Windows:", task_result["status"])
    flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
    with (directory / "watcher.log").open("a", encoding="utf-8") as log:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "watch", "--session", str(session_file)],
                         stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT, creationflags=flags,
                         start_new_session=os.name != "nt")
    ready = directory / "watch_ready.json"
    for _ in range(50):
        status = directory / "export_status.json"
        if status.exists() and _read(status).get("zip_valid"):
            break
        if ready.exists():
            import psutil
            if psutil.pid_exists(int(_read(ready)["pid"])):
                break
        time.sleep(0.1)
    else:
        raise RuntimeError("Watcher no confirmo el armado; revise watcher.log; entrenamiento NO iniciado")
    cmd = [sys.executable, str(Path(__file__).with_name("worker.py")), "--config", str(directory / "effective_config.yaml"),
           "--pc-id", b["pc_id"], "--init-plan", "--wait"]
    if resume:
        cmd += ["--force-recover"]
    if retry_failed:
        cmd += ["--retry-failed"]
    env = dict(os.environ)
    env["BOTRYTIS_SESSION_FILE"] = str(session_file)
    env["PYTHONHASHSEED"] = str(cfg["reproducibility"]["seed"])
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print("Sesion:", session_file, "\nExportacion:", _read(session_file)["export_at"])
    code = subprocess.call(cmd, cwd=PROJECT_ROOT, env=env)
    atomic_json(directory / "worker_exit.json", {"code": code, "at": now().isoformat()})
    return code

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("start", "resume", "watch", "status"))
    p.add_argument("--config", default="configs/local.yaml")
    p.add_argument("--session")
    p.add_argument("--clock-verified", action="store_true")
    p.add_argument("--new-session", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    a = p.parse_args()
    if a.action in ("start", "resume"):
        return launch(a.config, resume=a.action == "resume", clock_verified=a.clock_verified,
                      new_session=a.new_session, retry_failed=a.retry_failed)
    if a.session:
        session = Path(a.session)
    else:
        session = Path(_read(sync_root(load_config(a.config)) / "current_session.json")["session_file"])
    if a.action == "status":
        print(session.read_text(encoding="utf-8"))
        status = session.parent / "export_status.json"
        print(status.read_text(encoding="utf-8") if status.exists() else "BLOCKED: sin evidencia de watcher")
        return 0
    return watch(session)

if __name__ == "__main__":
    raise SystemExit(main())
