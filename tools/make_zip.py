"""Empaqueta el framework en un ZIP portable (rutas con ``/``, sin residuos).

Excluye ``__pycache__``, ``.pytest_cache``, ``desktop.ini``/``Thumbs.db`` (los inyecta OneDrive/Windows),
el contenido de ``runs/``, ``data/``, ``artifacts/``, ``reports/figures/`` y ``reports/tables/`` (solo se
conservan sus ``.gitkeep``), pesos (``*.pt``) y cualquier ``tmp*``. El ZIP contiene una carpeta raiz
``Botrytis_Experiment_Framework/`` con: codigo (``src/``), configuracion YAML, tests, herramientas,
``requirements*.txt``, ``README.md`` y ``AUDIT_REPORT.md``.

Ejemplo::

    python tools/make_zip.py --out ../Botrytis_Experiment_Framework_ADAPTADO.zip
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
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP = "Botrytis_Experiment_Framework"
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".git", ".venv", "venv", "env", "envs", ".vscode", ".codex", ".agents", ".mypy_cache", ".ruff_cache", "node_modules", "datasets", "weights", "checkpoints", "estimated_preview", "superseded", "obsolete"}
EXCLUDE_NAMES = {"desktop.ini", "Thumbs.db", ".DS_Store"}
EMPTY_DIRS = {"runs", "data", "reports/figures", "reports/tables", "artifacts"}  # solo se incluyen sus .gitkeep
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".safetensors", ".weights", ".onnx", ".engine", ".cache", ".zip", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".avif", ".gif"}


def _included(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if set(parts) & EXCLUDE_DIRS or path.name in EXCLUDE_NAMES or any(p.startswith(("tmp", ".validation", "_migration", ".venv")) for p in parts):
        return False
    if "local_image_validation" in parts:
        return False  # audit summary is preserved separately, raw runtime traces are local
    if any(p.startswith("graficos_preview") for p in parts):
        return False  # layout-only exports are deliberately kept outside the delivery archive
    if parts[:2] in (("reports", "graficos"), ("reports", "graficos_local")):
        return False  # prior semantic-version exports; current ones are in graficos_cierre
    if parts[0] == "reports" and path.name.startswith(("tests_derived.", "tests_graphics.", "tests_retest.", "tests_semantic.", "static_checks.")):
        return False  # historical attempts are retained locally, not presented as final checks
    if path.name == "figura_dataset.pdf":
        return False  # no embedded dataset photographs in the portable code archive
    posix = rel.as_posix()
    for d in EMPTY_DIRS:
        if posix.startswith(d + "/") and path.name != ".gitkeep":
            return False
    return path.suffix.lower() not in EXCLUDE_SUFFIXES


def main() -> int:
    """Crea el ZIP y muestra su SHA-256."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and _included(p))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in files:
            info = zipfile.ZipInfo(f"{TOP}/{p.relative_to(ROOT).as_posix()}", date_time=(2026, 9, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if p.suffix == ".py" and p.parent.name in {"src", "tools"} else 0o644) << 16
            zf.writestr(info, p.read_bytes())
    with zipfile.ZipFile(a.out) as zf:
        if zf.testzip() is not None:
            raise RuntimeError("ZIP CRC verification failed")
        if len(zf.namelist()) != len(set(zf.namelist())):
            raise RuntimeError("Duplicate archive members")
        required = {f"{TOP}/{p}" for p in ("README.md", "ADAPTACION_BOTRYTIS.md", "src/utils/pairing.py", "src/utils/semantics.py", "configs/adaptado.yaml")}
        if not required <= set(zf.namelist()):
            raise RuntimeError(f"Missing required deliverables: {required-set(zf.namelist())}")
    digest = hashlib.sha256(a.out.read_bytes()).hexdigest()
    a.out.with_suffix(".zip.sha256").write_text(f"{digest}  {a.out.name}\n", encoding="utf-8", newline="\n")
    print(f"{len(files)} ficheros -> {a.out} ({a.out.stat().st_size / 1024:.0f} KB)\nsha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
