"""Auditoria interna del modulo base (estatica + opcionalmente la suite de tests).

Comprueba, sobre ``src/``, ``tools/`` y ``configs/``:

1. sintaxis (``py_compile``) de todos los ficheros;
2. imports: cada modulo de ``src/utils`` importa y cada script (``NN_*.py``) carga sin ejecutar ``main``;
3. cada script responde a ``--help`` (exit 0);
4. ausencia de rutas hardcodeadas (``C:\\``, ``/home/``, ``/Users/``, ``D:\\`` ...);
5. type hints completos en TODA funcion/metodo y docstring en las publicas;
6. imports sin usar y nombres indefinidos (si ``pyflakes`` esta instalado);
7. finales de linea LF y UTF-8 sin BOM;
8. coherencia de rutas de ``configs/default.yaml`` (todas relativas o parametrizadas);
9. (``--with-tests``) ejecuta ``pytest``.

Escribe ``reports/audit_report.md``. Sale con codigo 1 si hay fallos.

Ejemplo::

    python tools/audit.py --with-tests
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
import ast
import importlib
import importlib.util
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

HARDCODED = re.compile(r"""(?:["'])(?:[A-Za-z]:[\\/](?!\\)|/home/|/Users/|/mnt/|/media/)""")
SKIP_DIRS = {"__pycache__", ".pytest_cache", "runs", "artifacts", "data", "tmp", ".validation_tmp", ".validation_extra", ".validation_retest", ".validation_derived", ".validation_final"}
OPTIONAL = {"torch", "torchvision", "timm", "ultralytics", "optuna", "cv2", "psutil"}


def _py_files(*dirs: Path) -> list[Path]:
    """Lista ficheros .py bajo los directorios dados (sin cachés ni salidas)."""
    out: list[Path] = []
    for d in dirs:
        out += [p for p in sorted(d.rglob("*.py")) if not (SKIP_DIRS & set(p.relative_to(ROOT).parts))]
    return out


class Report:
    """Acumula resultados PASS/FAIL para el informe."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool | None, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        """Registra una comprobacion y la imprime."""
        self.rows.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not ok else ""))

    @property
    def failed(self) -> int:
        """Numero de comprobaciones fallidas."""
        return sum(1 for _, ok, _ in self.rows if ok is False)

    def skip(self, name: str, detail: str) -> None:
        """Record a genuinely unavailable check without treating it as passed."""
        self.rows.append((name, None, detail))
        print(f"[SKIP] {name} -- {detail}")


def check_syntax(rep: Report, files: list[Path]) -> None:
    """Compila cada fichero para detectar errores de sintaxis."""
    bad = []
    for f in files:
        try:
            compile(f.read_text(encoding="utf-8"), str(f), "exec")  # no deja .pyc
        except SyntaxError as exc:
            bad.append(f"{f.name}:{exc.lineno}: {exc.msg}")
    rep.add(f"sintaxis de {len(files)} ficheros .py", not bad, "; ".join(bad))


def check_imports(rep: Report) -> None:
    """Importa utils y carga los scripts sin ejecutar main()."""
    bad = []
    skipped = 0
    mods = [p.stem for p in sorted((SRC / "utils").glob("*.py")) if p.stem != "__init__"]
    for m in mods:
        try:
            importlib.import_module(f"utils.{m}")
        except ModuleNotFoundError as exc:
            if str(exc.name).split(".")[0] in OPTIONAL:
                rep.skip(f"import utils.{m}", str(exc))
                skipped += 1
            else:
                bad.append(f"utils.{m}: {exc}")
        except Exception as exc:  # noqa: BLE001
            bad.append(f"utils.{m}: {type(exc).__name__}: {exc}")
    _group_result(rep, "imports utils", len(mods), skipped, bad)
    bad = []
    skipped = 0
    scripts = sorted(SRC.glob("[0-9][0-9]_*.py"))
    for s in scripts:
        try:
            spec = importlib.util.spec_from_file_location(f"script_{s.stem}", s)
            module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            if not callable(getattr(module, "main", None)):
                bad.append(f"{s.name}: sin main()")
        except ModuleNotFoundError as exc:
            if str(exc.name).split(".")[0] in OPTIONAL:
                rep.skip(f"load {s.name}", str(exc))
                skipped += 1
            else:
                bad.append(f"{s.name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{s.name}: {type(exc).__name__}: {exc}")
    _group_result(rep, "carga de scripts (sin ejecutar main)", len(scripts), skipped, bad)
    expected = {"00_check_environment", "01_validate_dataset", "02_create_outer_split", "03_create_inner_folds", "04_build_detection_dataset",
                "05_build_classification_crops", "06_run_baseline", "07_run_hpo_study", "08_repeat_top_configs", "09_train_outer_candidate",
                "10_evaluate_outer_validation", "11_tune_inference_thresholds", "12_select_detector", "13_select_classifier",
                "14_validate_integrated_pipeline", "15_freeze_experiment", "16_evaluate_test", "17_benchmark_latency", "18_export_master_results"}
    rep.add("los 19 scripts de la arquitectura final presentes", expected <= {s.stem for s in scripts}, str(expected - {s.stem for s in scripts}))
    required_utils = {"logging_utils", "reproducibility", "hashing", "boxes", "labels", "folds", "metrics", "metrics_detection", "metrics_classification", "visualization",
                      "io", "config_loader", "optuna_storage", "timing", "architectures", "search_spaces", "guards", "selection", "freeze", "scientific_audit"}
    rep.add("utilidades requeridas presentes (sec. 30)", required_utils <= set(mods), str(required_utils - set(mods)))
    orch = {"worker", "study_queue", "pc_registry"}
    rep.add("orchestration/{worker,study_queue,pc_registry}.py presentes", orch <= {p.stem for p in (SRC / "orchestration").glob("*.py")}, str(orch))
    bad = []
    for name in ("orchestration.worker", "orchestration.study_queue", "orchestration.pc_registry", "orchestration.sync"):
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{name}: {type(exc).__name__}: {exc}")
    rep.add("import de orchestration.*", not bad, "; ".join(bad))


def check_help(rep: Report) -> None:
    """Ejecuta `--help` en cada script."""
    bad = []
    skipped = 0
    scripts = sorted(SRC.glob("[0-9][0-9]_*.py")) + [SRC / "orchestration" / "worker.py", SRC / "orchestration" / "sync.py"]
    for s in scripts:
        r = subprocess.run([sys.executable, str(s), "--help"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        if r.returncode != 0 and any(f"No module named '{pkg}'" in r.stderr for pkg in OPTIONAL):
            rep.skip(f"--help {s.name}", "Optional ML dependency unavailable; actual CLI import not verified")
            skipped += 1
        elif r.returncode != 0 or ("--config" not in r.stdout and "--dataset" not in r.stdout):
            bad.append(f"{s.name} (exit {r.returncode})")
    _group_result(rep, "scripts ejecutables (--help exit 0)", len(scripts), skipped, bad)


def _group_result(rep: Report, name: str, total: int, skipped: int, bad: list[str]) -> None:
    """Report incomplete groups as SKIP, retaining successes and failures explicitly."""
    detail = f"{total - skipped - len(bad)}/{total} ejecutados correctamente; {skipped} omitidos; {len(bad)} fallidos"
    if bad:
        rep.add(name, False, detail + "; " + "; ".join(bad))
    elif skipped:
        rep.skip(name, detail)
    else:
        rep.add(name, True, detail)


def check_hardcoded(rep: Report, files: list[Path]) -> None:
    """Busca rutas absolutas en el codigo y en la configuracion."""
    bad = []
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
            if HARDCODED.search(line) and "re.compile" not in line and "HARDCODED" not in line:
                bad.append(f"{f.relative_to(ROOT)}:{i}")
    rep.add("sin rutas absolutas hardcodeadas en .py", not bad, ", ".join(bad[:10]))
    cfg = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    abs_cfg = [ln for ln in cfg.splitlines() if re.search(r":\s*[\"']?([A-Za-z]:[\\/]|/(home|Users|mnt)/)", ln)]
    rep.add("sin rutas absolutas en configs/default.yaml", not abs_cfg, "; ".join(abs_cfg))


def check_annotations(rep: Report, files: list[Path]) -> None:
    """Verifica type hints y docstrings mediante AST."""
    no_hint, no_doc = [], []
    n_funcs = 0
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        if not ast.get_docstring(tree):
            no_doc.append(f"{f.relative_to(ROOT)} (modulo)")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            n_funcs += 1
            args = node.args
            params = [a for a in args.posonlyargs + args.args + args.kwonlyargs if a.arg not in ("self", "cls")]
            params += [a for a in (args.vararg, args.kwarg) if a is not None]
            if node.returns is None or any(a.annotation is None for a in params):
                no_hint.append(f"{f.relative_to(ROOT)}:{node.lineno} {node.name}")
            is_public = not node.name.startswith("_")
            nested = any(isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)) and node in ast.walk(p) and p is not node for p in ast.walk(tree))
            if is_public and not nested and not ast.get_docstring(node):
                no_doc.append(f"{f.relative_to(ROOT)}:{node.lineno} {node.name}")
    rep.add(f"type hints completos ({n_funcs} funciones)", not no_hint, "; ".join(no_hint[:8]))
    rep.add("docstrings en modulos y funciones publicas", not no_doc, "; ".join(no_doc[:8]))


def check_pyflakes(rep: Report, files: list[Path]) -> None:
    """Ejecuta pyflakes si esta disponible."""
    try:
        r = subprocess.run([sys.executable, "-m", "pyflakes", *map(str, files)], capture_output=True, text=True, timeout=120)
    except Exception:  # noqa: BLE001
        rep.skip("pyflakes (imports sin usar / nombres indefinidos)", "no disponible")
        return
    if "No module named pyflakes" in r.stderr:
        rep.skip("pyflakes (imports sin usar / nombres indefinidos)", "no instalado")
        return
    out = (r.stdout + r.stderr).strip()
    rep.add("pyflakes (imports sin usar / nombres indefinidos)", r.returncode == 0, out.replace("\n", "; ")[:600])


def check_text_files(rep: Report) -> None:
    """Verifica LF y UTF-8 sin BOM."""
    bad = []
    for p in list(ROOT.rglob("*")):
        if (not p.is_file() or SKIP_DIRS & set(p.relative_to(ROOT).parts) or p.name == "desktop.ini"
                or any(part.startswith((".validation", "tmp", "_migration")) for part in p.relative_to(ROOT).parts)
                or p.suffix not in {".py", ".yaml", ".md", ".txt", ".ini"}):
            continue
        raw = p.read_bytes()
        if b"\r\n" in raw or raw.startswith(b"\xef\xbb\xbf"):
            bad.append(str(p.relative_to(ROOT)))
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            bad.append(f"{p.relative_to(ROOT)} (no UTF-8)")
    rep.add("finales de linea LF, UTF-8 sin BOM", not bad, ", ".join(bad[:10]))


def check_config(rep: Report) -> None:
    """Carga la configuracion por defecto y valida las decisiones FIJAS."""
    from utils.config_loader import list_todos, load_config

    try:
        cfg = load_config()
        rep.add("configs/default.yaml carga y respeta decisiones FIJAS (perfil official)", True)
        todos = list_todos(cfg)
        rep.add("Seed externa 67 predeclarada y sin TODO bloqueantes en default",
                cfg["outer_split"]["seed"] == 67 and not todos, "; ".join(todos))
    except Exception as exc:  # noqa: BLE001
        rep.add("configs/default.yaml carga y respeta decisiones FIJAS", False, str(exc))
    try:
        load_config(ROOT / "configs" / "smoke.yaml")
        rep.add("configs/smoke.yaml carga (perfil smoke no relaja decisiones FIJAS)", True)
    except Exception as exc:  # noqa: BLE001
        rep.add("configs/smoke.yaml carga (perfil smoke no relaja decisiones FIJAS)", False, str(exc))
    import yaml

    spaces = yaml.safe_load((ROOT / "configs" / "search_spaces.yaml").read_text(encoding="utf-8"))
    rep.add("configs/search_spaces.yaml define los 6 espacios (yolov8, yolo26, rtdetr, resnet50, vit, swin)",
            {"yolov8", "yolo26", "rtdetr", "resnet50", "vit", "swin"} <= set(spaces), str(set(spaces)))


def check_no_synthetic_in_pipeline(rep: Report) -> None:
    """Los scripts del experimento no importan generadores de datos sinteticos/demo."""
    bad = []
    for p in _py_files(SRC):
        txt = p.read_text(encoding="utf-8")
        if re.search(r"^\s*(from|import)\s+(make_synthetic_dataset|demo_figures)", txt, re.M):
            bad.append(str(p.relative_to(ROOT)))
    rep.add("src/ no importa datos sinteticos ni figuras demo (graficos con datos reales)", not bad, ", ".join(bad))


def check_test_locked_in_code(rep: Report) -> None:
    """Solo 16 (y la guardia) mencionan el split test de forma operativa: ningun otro script pasa stage='final_test'."""
    bad = []
    for p in _py_files(SRC):
        rel = p.relative_to(SRC).as_posix()
        if rel in ("16_evaluate_test.py", "utils/splits.py", "utils/scientific_audit.py", "utils/detector_data.py", "utils/classifier_data.py", "utils/freeze.py"):
            continue
        if re.search(r"""["']final_test["']""", p.read_text(encoding="utf-8")):
            bad.append(rel)
    rep.add("solo 16_evaluate_test usa la etapa 'final_test' (test bloqueado)", not bad, ", ".join(bad))


def run_tests(rep: Report) -> None:
    """Ejecuta pytest y registra el resultado."""
    import tempfile
    import xml.etree.ElementTree as ET

    base = Path(tempfile.mkdtemp(prefix="botrytis_pytest_"))  # evita permisos/colisiones del basetemp por defecto
    result_xml = base / "tests.xml"
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(base / "t"), "--junitxml", str(result_xml)], cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=3600)
    tail = "\n".join(r.stdout.strip().splitlines()[-3:])
    if not result_xml.is_file():
        rep.add("suite de tests (pytest)", False, f"exit={r.returncode}; sin JUnit XML; {tail}")
        return
    cases = ET.parse(result_xml).findall(".//testcase")
    skipped = sum(c.find("skipped") is not None for c in cases)
    failed = [c.get("name", "test") for c in cases if c.find("failure") is not None or c.find("error") is not None]
    if r.returncode and not failed:
        failed.append(f"pytest exit={r.returncode}")
    _group_result(rep, "suite de tests (pytest)", len(cases), skipped, failed)


def main() -> int:
    """CLI de la auditoria."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-tests", action="store_true", help="ejecuta tambien pytest (tarda unos minutos)")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "audit_report.md")
    a = ap.parse_args()
    rep = Report()
    code_files = _py_files(SRC, ROOT / "tools")
    all_files = _py_files(SRC, ROOT / "tools", ROOT / "tests")
    check_syntax(rep, all_files)
    check_imports(rep)
    check_help(rep)
    check_hardcoded(rep, code_files)
    check_annotations(rep, code_files)
    check_pyflakes(rep, all_files)
    check_config(rep)
    check_no_synthetic_in_pipeline(rep)
    check_test_locked_in_code(rep)
    check_text_files(rep)
    if a.with_tests:
        run_tests(rep)
    else:
        rep.skip("suite de tests (pytest)", "No solicitada en esta ejecución; usar --with-tests o consultar reports/tests.xml")
    lines = ["# Informe de auditoria interna", "", f"Generado: {datetime.now().astimezone().isoformat(timespec='seconds')}  ",
             f"Python: {sys.version.split()[0]}", "", "| Comprobacion | Resultado | Detalle |", "|---|---|---|"]
    for name, ok, detail in rep.rows:
        status = "SKIP" if ok is None else "PASS" if ok else "**FAIL**"
        lines.append(f"| {name} | {status} | {detail.replace('|', '/').replace(chr(10), ' ')[:300]} |")
    lines += ["", f"**Resumen:** {sum(ok is True for _,ok,_ in rep.rows)} PASS; {rep.failed} FAIL; {sum(ok is None for _,ok,_ in rep.rows)} SKIP."]
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\nInforme: {a.report}  |  fallos: {rep.failed}")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
