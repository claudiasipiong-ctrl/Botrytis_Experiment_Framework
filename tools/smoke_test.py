"""Prueba de humo de EXTREMO A EXTREMO (00 -> 18) con el perfil ``configs/smoke.yaml`` y un dataset SINTETICO.

NO produce resultados cientificos: verifica que toda la cadena (preparacion, baseline, HPO con Optuna, robustez,
entrenamiento definitivo, evaluacion, umbrales, seleccion, pipeline, freeze, test, benchmark, exportacion y
auditoria) se ejecuta, deja sus logs/hashes/figuras y respeta los controles metodologicos.

Uso::

    python tools/smoke_test.py --workdir C:/botrytis_work/smoke --n-images 120
    python tools/smoke_test.py --workdir C:/botrytis_work/smoke --workers 3      # simula las 3 PCs con 3 workers
    python tools/smoke_test.py --workdir C:/botrytis_work/smoke --from-step 12    # reanuda desde un paso

Los datos, runs y artefactos se crean en ``--workdir`` (fuera del proyecto). Escribe ``<workdir>/smoke_report.json``.
"""

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
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_synthetic_dataset import generate  # noqa: E402


def workspace_sets(ws: Path, *, seed: int = 20260930, fruit: int = 0, sign: int = 2) -> list[str]:
    """``--set`` que redirigen TODAS las rutas del experimento a ``ws`` y fijan los TODO bloqueantes de la prueba."""
    p = {
        "paths.artifacts_dir": ws / "artifacts", "paths.environments_dir": ws / "artifacts" / "environments", "paths.data_root": ws / "data",
        "paths.raw_images_dir": ws / "raw" / "images", "paths.raw_labels_dir": ws / "raw" / "labels", "paths.figures_dir": ws / "reports" / "figures",
        "paths.reports_dir": ws / "reports", "paths.tables_dir": ws / "reports" / "tables", "paths.optuna_dir": ws / "artifacts" / "optuna",
        "paths.queue_dir": ws / "artifacts" / "queue", "paths.best_configs_dir": ws / "artifacts" / "best_configs",
        "compute.reference_environment": ws / "artifacts" / "environments" / "environment_reference.json", "test.freeze_manifest": ws / "artifacts" / "freeze_manifest.json",
    }
    sets = [f"{k}={v}" for k, v in p.items()]
    return sets + [f"classes.fruit_class_id={fruit}", f"classes.sign_class_id={sign}", f"outer_split.seed={seed}"]


def main() -> int:
    """Ejecuta la cadena completa y escribe el informe."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--n-images", type=int, default=120)
    ap.add_argument("--workers", type=int, default=1, choices=(1, 2, 3), help="n.º de workers (PCs simuladas) en la fase de entrenamiento")
    ap.add_argument("--from-step", type=int, default=0, help="reanuda desde este paso (0-18)")
    ap.add_argument("--to-step", type=int, default=18)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--extra-set", action="append", default=[], help="overrides adicionales --set")
    a = ap.parse_args()
    ws = a.workdir.resolve()
    (ws / "raw").mkdir(parents=True, exist_ok=True)
    if not (ws / "raw" / "images").exists():
        print("generando dataset sintetico:", generate(ws / "raw", n=a.n_images, seed=7, fruit_class=0, sign_class=2))
    sets = workspace_sets(ws) + a.extra_set
    if a.from_step <= 15:  # permite repetir la cola final (12-18): el freeze/test de una corrida anterior se descartan (solo en la prueba de humo)
        for name in ("freeze_manifest.json", "freeze_manifest.sha256", "test_results.json", "test_results.sha256", "test_evaluation_started.json", "selection_policy.yaml",
                     "selection_policy.sha256", "selected_detector.json", "selected_classifier.json", "final_thresholds.json", "pipeline_validation.json"):
            (ws / "artifacts" / name).unlink(missing_ok=True)
    common = ["--config", str(ROOT / "configs" / "smoke.yaml"), "--runs-dir", str(ws / "runs")] + [x for s in sets for x in ("--set", s)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "YOLO_VERBOSE": "False"}
    log: list[dict[str, object]] = []

    def run(step: int, name: str, script: str, *extra: str, pc: str = "PC-01", ok_codes: tuple[int, ...] = (0,)) -> bool:
        if step < a.from_step or step > a.to_step:
            return True
        cmd = [a.python, str(SRC / script), "--pc-id", pc, *common, *extra]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=str(ws))
        dt = time.time() - t0
        ok = res.returncode in ok_codes
        log.append({"step": step, "name": name, "returncode": res.returncode, "seconds": round(dt, 1), "ok": ok})
        print(f"[{'OK ' if ok else 'FAIL'}] paso {step:02d} {name} ({dt:.0f}s)")
        if not ok:
            print((res.stdout or "")[-3500:])
            print((res.stderr or "")[-3500:])
        (ws / "smoke_report.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        return ok

    steps: list[tuple[int, str, str, list[str], dict[str, object]]] = [
        (0, "00_check_environment PC-01 (referencia)", "00_check_environment.py", ["--write-reference"], {}),
        (1, "01_validate_dataset", "01_validate_dataset.py", [], {}),
        (2, "02_create_outer_split", "02_create_outer_split.py", [], {}),
        (3, "03_create_inner_folds", "03_create_inner_folds.py", [], {}),
        (4, "04_build_detection_dataset", "04_build_detection_dataset.py", [], {}),
        (5, "05_build_classification_crops", "05_build_classification_crops.py", [], {}),
    ]
    for step, name, script, extra, kw in steps:
        if not run(step, name, script, *extra, **kw):  # type: ignore[arg-type]
            return 1
    if 0 >= a.from_step:
        for pc in ("PC-02", "PC-03"):
            if not run(0, f"00_check_environment {pc}", "00_check_environment.py", pc=pc):
                return 1

    # 6-11: entrenamiento por la cola (worker/es)
    if a.from_step <= 11 and a.to_step >= 6:
        wcmd_common = ["--wait", "--poll-seconds", "5"]
        procs: list[subprocess.Popen[str]] = []
        for i in range(a.workers):
            pc = f"PC-0{i + 1}"
            cmd = [a.python, str(SRC / "orchestration" / "worker.py"), "--pc-id", pc, *common, *wcmd_common] + (["--init-plan"] if i == 0 else [])
            procs.append(subprocess.Popen(cmd, stdout=open(ws / f"worker_{pc}.out", "w", encoding="utf-8"), stderr=subprocess.STDOUT, env=env, cwd=str(ws), text=True))
            time.sleep(3)
        t0 = time.time()
        rcs = [p.wait() for p in procs]
        ok = all(rc == 0 for rc in rcs)
        log.append({"step": "6-11", "name": f"worker x{a.workers} (baseline, HPO, robustez, final, validation, umbrales)", "returncode": rcs, "seconds": round(time.time() - t0, 1), "ok": ok})
        print(f"[{'OK ' if ok else 'FAIL'}] pasos 06-11 via cola con {a.workers} worker(s) ({time.time() - t0:.0f}s)")
        q = subprocess.run([a.python, str(SRC / "orchestration" / "worker.py"), "--pc-id", "PC-01", *common, "--status", "--skip-env-check"], capture_output=True, text=True, encoding="utf-8", env=env, cwd=str(ws))
        print((q.stdout or "")[-3000:])
        (ws / "smoke_report.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        if not ok:
            return 1

    tail: list[tuple[int, str, str, list[str]]] = [
        (12, "12_select_detector", "12_select_detector.py", []),
        (13, "13_select_classifier", "13_select_classifier.py", []),
        (14, "14_validate_integrated_pipeline", "14_validate_integrated_pipeline.py", ["--retune-classifier-threshold"]),
        (15, "15_freeze_experiment", "15_freeze_experiment.py", ["--allow-unconfirmed-decisions"]),
        (16, "16_evaluate_test", "16_evaluate_test.py", []),
        (17, "17_benchmark_latency", "17_benchmark_latency.py", []),
        (18, "18_export_master_results", "18_export_master_results.py", []),
    ]
    for step, name, script, extra in tail:
        if not run(step, name, script, *extra):
            return 1
    print("\nPRUEBA DE HUMO COMPLETA. Informe:", ws / "smoke_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
