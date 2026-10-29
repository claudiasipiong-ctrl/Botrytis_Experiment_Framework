"""QA local por subprocesos; conserva stdout/stderr y nunca lanza el experimento oficial."""
from __future__ import annotations

# Stdlib bootstrap: records import/CLI/config failures before optional dependencies.
import sys as _boot_sys
from pathlib import Path as _BootPath
_boot_root = _BootPath(__file__).resolve().parents[1]
_boot_sys.path.insert(0, str(_boot_root / "src"))
from utils import bootstrap as _bootstrap
_bootstrap.install(__file__, _boot_root)
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True

def terminate_tree(proc):
    try:
        import psutil
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()
        _, alive = psutil.wait_procs(children + [parent], timeout=5)
        for child in alive:
            child.kill()
    except Exception:
        proc.kill()
    proc.wait(timeout=15)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/local.yaml")
    p.add_argument("--machine-id", choices=("i21315", "i21314", "i21313"))
    p.add_argument("--bash-version", default="not supplied")
    p.add_argument("--allow-cuda-selftest", action="store_true")
    p.add_argument("--allow-training-smoke", action="store_true")
    p.add_argument("--full-suite", action="store_true")
    p.add_argument("--suite-timeout", type=int, default=1800)
    a = p.parse_args()
    if not 30 <= a.suite_timeout <= 3600:
        p.error("--suite-timeout debe estar entre 30 y 3600 segundos")
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    directory = ROOT / "logs_qa" / f"{a.machine_id or platform.node()}_{stamp}_{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    env = dict(os.environ)
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
               MPLCONFIGDIR=str(directory / "matplotlib"), YOLO_CONFIG_DIR=str(directory / "ultralytics"),
               TORCH_HOME=str(directory / "torch_cache"), HF_HOME=str(directory / "huggingface_cache"),
               XDG_CACHE_HOME=str(directory / "cache"), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    metadata = {"machine_id": a.machine_id, "hostname": platform.node(), "start": datetime.now().astimezone().isoformat(),
                "python_executable": sys.executable, "python": sys.version, "os": platform.platform(),
                "bash": a.bash_version, "parameters": vars(a), "real_hardware_qa": True,
                "scientific_training_started": False}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    def record(name, status, reason, **extra):
        row = {"check": name, "status": status, "reason": reason, **extra}
        rows.append(row)
        print(f"[{status}] {name}: {reason}", flush=True)
        (directory / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    def run(name, command, timeout=180, monitored=False):
        start = time.perf_counter()
        stdout, stderr = directory / (name + ".stdout.log"), directory / (name + ".stderr.log")
        monitor = None
        if monitored:
            prerequisites = {"configuration", "hardware", "safety", "dependencies"}
            if name in {"training_smoke", "framework_full_suite"}:
                prerequisites.add("cuda_selftest")
            blocked = [r["check"] for r in rows if r["check"] in prerequisites and r["status"] != "PASS"]
            if blocked:
                record(name, "BLOCKED", f"Comprobaciones previas sin aprobar: {blocked}")
                return
            try:
                sys.path.insert(0, str(ROOT / "src"))
                from utils.config_loader import load_config
                from utils.safety import SafetyMonitor
                monitor = SafetyMonitor(load_config(a.config), directory / (name + ".telemetry.jsonl"))
                monitor.check()
            except Exception as exc:
                record(name, "BLOCKED", f"Supervision previa: {exc}")
                return
        error, rc = None, None
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            try:
                proc = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, env=env)
                while proc.poll() is None:
                    if time.perf_counter() - start > timeout:
                        error = f"Tiempo maximo {timeout}s agotado; proceso y descendientes detenidos"
                        terminate_tree(proc)
                        break
                    if monitor:
                        try:
                            monitor.check()
                        except Exception as exc:
                            error = f"SAFETY_BLOCKED: {exc}; proceso y descendientes detenidos"
                            terminate_tree(proc)
                            break
                    time.sleep(1)
                rc = proc.wait(timeout=15)
            except Exception as exc:
                error = repr(exc)
        status = {0: "PASS", 10: "WARN", 11: "SKIP", 12: "BLOCKED"}.get(rc, "FAIL")
        if error:
            status = "FAIL"
        reason = error or f"codigo {rc}; ver {stdout.name} y {stderr.name}"
        if name in {"framework_full_suite", "framework_safe_suite"} and rc == 0:
            import xml.etree.ElementTree as ET
            xml = directory / (name + ".xml")
            if xml.exists():
                skipped = sum(int(t.get("skipped", 0)) for t in ET.parse(xml).getroot().iter("testsuite"))
                if skipped:
                    status, reason = "WARN", f"{skipped} pruebas omitidas; revise XML y logs"
        record(name, status, reason, command=command, parameters={"timeout_seconds": timeout, "monitored": monitored},
               exit_code=rc, seconds=time.perf_counter() - start, stdout=str(stdout.relative_to(ROOT)), stderr=str(stderr.relative_to(ROOT)))
    for name in ("static", "dependencies", "configuration", "hardware", "dataset_zip", "dataset_validation",
                 "prepared_data", "environment", "safety", "clock", "exchange_state", "storage_io"):
        command = [sys.executable, str(ROOT / "tools/qa_checks.py"), name, "--config", a.config, "--output", str(directory)]
        if a.machine_id:
            command += ["--machine-id", a.machine_id]
        run(name, command, timeout=1800 if name.startswith("dataset") else 180)
    run("pip_check", [sys.executable, "-m", "pip", "check"])
    safe_tests = ["dev_tests", "tests/test_boxes_and_rule.py", "tests/test_config_hashing.py"]
    run("framework_safe_suite", [sys.executable, "-m", "pytest", *safe_tests, "-q",
                                "--basetemp", str(directory / "pytest_safe"), "-o", "cache_dir=" + str(directory / "pytest_cache"),
                                "--junitxml", str(directory / "framework_safe_suite.xml")], timeout=600)
    if a.allow_cuda_selftest:
        run("cuda_selftest", [sys.executable, str(ROOT / "tools/qa_checks.py"), "cuda_selftest",
                             "--config", a.config, "--output", str(directory)], timeout=60, monitored=True)
    else:
        record("cuda_selftest", "SKIP", "Requiere --allow-cuda-selftest y politica verificada")
    if a.allow_training_smoke:
        run("training_smoke", [sys.executable, str(ROOT / "tools/qa_training_smoke.py"),
                              "--config", a.config, "--output", str(directory)], timeout=600, monitored=True)
    else:
        record("training_smoke", "SKIP", "Requiere --allow-training-smoke; no se ejecuto entrenamiento")
    if a.full_suite:
        run("framework_full_suite", [sys.executable, "-m", "pytest", "tests", "-q",
                                    "--basetemp", str(directory / "pytest_full"), "-o", "cache_dir=" + str(directory / "pytest_full_cache"),
                                    "--junitxml", str(directory / "framework_full_suite.xml")],
            timeout=a.suite_timeout, monitored=True)
    else:
        record("framework_full_suite", "SKIP", "Requiere --full-suite; contiene modelos sinteticos")
    critical = {r["check"] for r in rows}
    # Warnings de anotaciones no invalidan por si solos el dataset: se conserva fail_on_warnings=false.
    # Toda omision de una prueba critica, incluso pytest skipped, sigue impidiendo acreditar preparacion.
    missing = [r["check"] for r in rows if r["check"] in critical and r["status"] != "PASS"
               and not (r["check"] == "dataset_validation" and r["status"] == "WARN")]
    ready = not missing
    h = None
    try:
        from orchestration.exchange import hashes
        from utils.config_loader import load_config
        h = hashes(load_config(a.config))
    except Exception:
        pass
    result = {"qa_manual_status":"QA_MANUAL_PENDIENTE", "ready": False, "technical_checks_complete": ready, "overall": "TECHNICAL_CHECKS_COMPLETE_QA_MANUAL_PENDIENTE" if ready else "TECHNICAL_CHECKS_BLOCKED_QA_MANUAL_PENDIENTE",
              "critical_not_passed": missing, "machine_id": a.machine_id, "hashes": h,
              "completed_at": datetime.now().astimezone().isoformat(), "evidence_dir": str(directory), "tests": rows,
              "limits": ["No acredita rendimiento sostenido ni salud termica por una prueba breve",
                         "No acredita las otras dos PCs ni transferencia fisica entre ellas",
                         "No utiliza test real para seleccionar configuraciones"]}
    (directory / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "logs_qa/latest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Informe QA local", "", f"Estado: **{result['overall']}**", f"Equipo: {a.machine_id}; host: {platform.node()}",
             f"Fecha: {result['completed_at']}", "", "| Comprobacion | Estado | Evidencia |", "|---|---|---|"]
    lines += [f"| {r['check']} | {r['status']} | {r['reason'].replace('|', '/')} |" for r in rows]
    lines += ["", "## Problemas y pendientes", ""] + [f"- {r['check']}: {r['status']} — {r['reason']}" for r in rows if r["status"] != "PASS"]
    lines += ["", "## Limites", ""] + ["- " + s for s in result["limits"]]
    (directory / "RESUMEN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "issues.json").write_text(json.dumps([r for r in rows if r["status"] != "PASS"], indent=2, ensure_ascii=False), encoding="utf-8")
    print("Evidencia:", directory, "\nEstado:", result["overall"])
    return 0 if ready else (1 if any(r["status"] == "FAIL" for r in rows) else 3)

if __name__ == "__main__":
    raise SystemExit(main())
