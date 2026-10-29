"""Normaliza los ficheros de texto del proyecto a UTF-8 sin BOM y finales de linea LF.

Util antes de empaquetar/sincronizar entre PCs Windows y Linux. No toca ``runs/``, ``data/``
ni ``artifacts/``.

Ejemplo::

    python tools/normalize_text.py
"""

from __future__ import annotations

# Stdlib bootstrap: records import/CLI/config failures before optional dependencies.
import sys as _boot_sys
from pathlib import Path as _BootPath
_boot_root = _BootPath(__file__).resolve().parents[1]
_boot_sys.path.insert(0, str(_boot_root / "src"))
from utils import bootstrap as _bootstrap
_bootstrap.install(__file__, _boot_root)

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt", ".ini", ".cfg", ".toml", ".gitignore"}
SKIP = {"runs", "data", "artifacts", "__pycache__", ".pytest_cache", ".git"}
SKIP_NAMES = {"desktop.ini", "Thumbs.db"}  # ficheros que inyectan OneDrive/Windows


def main() -> int:
    """Reescribe los ficheros que lo necesiten y devuelve cuantos cambio."""
    changed = 0
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or SKIP & set(p.relative_to(ROOT).parts) or p.name in SKIP_NAMES:
            continue
        if p.suffix not in SUFFIXES and p.name not in {".gitignore"}:
            continue
        raw = p.read_bytes()
        new = raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw
        new = new.replace(b"\r\n", b"\n")
        if new != raw:
            p.write_bytes(new)
            changed += 1
            print(f"normalizado: {p.relative_to(ROOT)}")
    print(f"{changed} fichero(s) modificados")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
