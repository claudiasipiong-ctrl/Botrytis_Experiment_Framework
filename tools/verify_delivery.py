"""Verify archive bytes, exclusions and lightweight behavior in an isolated extraction.

Example: python tools/verify_delivery.py --archive ../Botrytis_Experiment_Framework_ADAPTADO.zip
The report is written beside the ZIP, without changing the packaged project.
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
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from make_zip import ROOT, TOP, _included


def verify(archive: Path) -> dict[str, Any]:
    """Check CRC, member hashes and independent imports/tests without ML training."""
    archive = archive.resolve()
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_suffix(".zip.sha256").read_text(encoding="utf-8").split()[0]
    if checksum != digest:
        raise ValueError("SHA-256 mismatch")
    expected = {f"{TOP}/{p.relative_to(ROOT).as_posix()}": p for p in ROOT.rglob("*") if p.is_file() and _included(p)}
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if z.testzip() is not None or len(names) != len(set(names)):
            raise ValueError("CRC failure or duplicate members")
        if set(names) != set(expected):
            raise ValueError(f"Archive/source member mismatch: {set(names) ^ set(expected)}")
        for name, path in expected.items():
            if z.read(name) != path.read_bytes():
                raise ValueError(f"Source bytes mismatch: {name}")
        temp_root = Path(tempfile.gettempdir()).resolve()
        with tempfile.TemporaryDirectory(prefix="botrytis_delivery_") as td:
            work = Path(td).resolve()
            if not work.is_relative_to(temp_root):
                raise ValueError("Extraction must remain within the system temporary directory")
            for name in names:
                if not (work / name).resolve().is_relative_to(work):
                    raise ValueError("Unsafe archive member")
            z.extractall(work)
            project = work / TOP
            for path in project.rglob("*.py"):
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            # The separately delivered script remains outside the archive/project.
            graphics = ROOT.parent / "generar_graficos_botrytis.py"
            shutil.copy2(graphics, work / graphics.name)
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONPATH"] = os.pathsep.join(p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and not Path(p).resolve().is_relative_to(ROOT))
            probe = "import sys,pathlib; sys.path.insert(0,str(pathlib.Path.cwd()/'src')); import utils.pairing as p; assert pathlib.Path(p.__file__).resolve().is_relative_to(pathlib.Path.cwd()); print(p.SEMANTICS_VERSION)"
            run = subprocess.run([sys.executable, "-c", probe], cwd=project, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
            if run.returncode:
                raise RuntimeError(run.stdout + run.stderr)
            result_xml = work / "tests.xml"
            selected = ["tests/test_semantic_pairing.py", "tests/test_config_hashing.py", "tests/test_splits_folds_metrics.py", "tests/test_external_graphics.py", "tests/test_audit_status.py"]
            run = subprocess.run([sys.executable, "-m", "pytest", *selected, "-o", "addopts=-ra", "-p", "no:cacheprovider", "--basetemp", str(work / "fixtures"), "--junitxml", str(result_xml)], cwd=project, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
            cases = ET.parse(result_xml).findall(".//testcase") if result_xml.exists() else []
            failed = [c.get("name") for c in cases if c.find("failure") is not None or c.find("error") is not None]
            skips = [{"test": c.get("name"), "reason": c.find("skipped").get("message")} for c in cases if c.find("skipped") is not None]
            if run.returncode or failed or not cases:
                raise RuntimeError(run.stdout + run.stderr)
            return {"archive": str(archive), "sha256": digest, "members": len(names), "crc": "PASS", "duplicate_entries": 0,
                    "source_member_bytes": "PASS", "exclusions": "PASS", "extracted_compilation": "PASS", "isolated_import": "PASS",
                    "portable_tests": {"pass": len(cases) - len(skips), "fail": len(failed), "skip": len(skips), "skip_reasons": skips},
                    "graphics_sha256": hashlib.sha256(graphics.read_bytes()).hexdigest(), "scope": "Functional tests only; no experimental model results"}


def main() -> int:
    """Verify the definitive ZIP and save a report outside the packaged project."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args.archive)
    report = args.report or args.archive.with_suffix(".verification.json")
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
