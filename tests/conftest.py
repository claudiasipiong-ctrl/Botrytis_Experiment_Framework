"""Fixtures compartidas: rutas del proyecto, dataset sintetico y ejecucion de scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for p in (SRC, TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from make_synthetic_dataset import generate  # noqa: E402
from utils.config_loader import load_config  # noqa: E402


@dataclass
class Workspace:
    """Directorio de trabajo aislado para ejecutar los scripts con rutas sobreescritas."""

    base: Path
    seed: int = 67
    fruit_class: int = 0
    sign_class: int = 2
    extra: list[str] = field(default_factory=list)

    @property
    def raw(self) -> Path:
        return self.base / "raw"

    @property
    def artifacts(self) -> Path:
        return self.base / "artifacts"

    @property
    def runs(self) -> Path:
        return self.base / "runs"

    @property
    def figs(self) -> Path:
        return self.base / "figs"

    @property
    def data(self) -> Path:
        return self.base / "data"

    def args(self, with_seed: bool = True, with_classes: bool = True) -> list[str]:
        a = [
            "--runs-dir", str(self.runs), "--pc-id", "PC-01",
            "--set", f"paths.artifacts_dir={self.artifacts}",
            "--set", f"paths.environments_dir={self.artifacts / 'environments'}",
            "--set", f"paths.figures_dir={self.figs}", "--set", f"paths.reports_dir={self.base / 'reports'}",
            "--set", f"paths.tables_dir={self.base / 'reports' / 'tables'}", "--set", f"paths.optuna_dir={self.artifacts / 'optuna'}",
            "--set", f"paths.queue_dir={self.artifacts / 'queue'}", "--set", f"paths.best_configs_dir={self.artifacts / 'best_configs'}",
            "--set", f"paths.data_root={self.data}",
            "--set", f"paths.raw_images_dir={self.raw / 'images'}",
            "--set", f"paths.raw_labels_dir={self.raw / 'labels'}",
            "--set", f"compute.reference_environment={self.artifacts / 'environments' / 'environment_reference.json'}",
        ]
        if with_classes:
            a += ["--set", f"classes.fruit_class_id={self.fruit_class}", "--set", f"classes.sign_class_id={self.sign_class}"]
        if with_seed:
            a += ["--set", f"outer_split.seed={self.seed}"]
        return a + self.extra

    def run(self, script: str, *extra: str, with_seed: bool = True, with_classes: bool = True) -> subprocess.CompletedProcess[str]:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        return subprocess.run(
            [sys.executable, str(SRC / f"{script}.py"), *self.args(with_seed, with_classes), *extra],
            capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=str(self.base), timeout=600,
        )

    def last_run(self, script: str) -> Path:
        runs = sorted(p for p in (self.runs / "_pipeline").glob(f"*_{script}*") if p.is_dir())
        assert runs, f"no hay runs de {script}"
        return runs[-1]


@pytest.fixture(scope="session")
def synthetic_329(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Dataset sintetico de 329 imagenes de prueba."""
    out = tmp_path_factory.mktemp("synth329")
    generate(out, n=329, seed=7, fruit_class=0, sign_class=2)
    return out


@pytest.fixture(scope="module")
def pipeline_ws(tmp_path_factory: pytest.TempPathFactory, synthetic_329: Path) -> Workspace:
    """Workspace con el dataset sintetico y los scripts 00-05 ya ejecutados."""
    import shutil

    base = tmp_path_factory.mktemp("ws_full")
    shutil.copytree(synthetic_329, base / "raw")
    ws = Workspace(base)
    for script, extra in (("00_check_environment", ["--write-reference"]), ("01_validate_dataset", []), ("02_create_outer_split", []),
                          ("03_create_inner_folds", []), ("04_build_detection_dataset", []), ("05_build_classification_crops", [])):
        res = ws.run(script, *extra)
        assert res.returncode == 0, f"{script} fallo:\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}"
    return ws


@pytest.fixture()
def small_ws(tmp_path: Path) -> Workspace:
    """Workspace con 12 imagenes (para pruebas negativas; requiere --allow-count-mismatch)."""
    generate(tmp_path / "raw", n=12, seed=3, fruit_class=0, sign_class=2)
    return Workspace(tmp_path)


@pytest.fixture(scope="module")
def ws60(tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    base = tmp_path_factory.mktemp("ws60 investigación ñandú")  # ruta CON acentos y espacios: los scripts deben funcionar igual
    generate(base / "raw", n=60, seed=11, fruit_class=0, sign_class=2)
    ws = Workspace(base, extra=["--set", "dataset.enforce_expected_count=false"])
    for script, extra in (("00_check_environment", ["--write-reference"]), ("01_validate_dataset", []), ("02_create_outer_split", []),
                          ("03_create_inner_folds", []), ("04_build_detection_dataset", []), ("05_build_classification_crops", [])):
        res = ws.run(script, *extra)
        assert res.returncode == 0, f"{script}:\n{res.stdout[-2500:]}\n{res.stderr[-1500:]}"
    # Synthetic fleet metadata, not a binding or approval of physical machines.
    shutil_source=ROOT/'configs/fleet_mapping.json'
    (ws.artifacts/'fleet_mapping.json').write_bytes(shutil_source.read_bytes())
    return ws


@pytest.fixture()
def cfg60(ws60: Workspace) -> dict:
    sets = [a for a in ws60.args() if "=" in a and not a.startswith("--")]
    return load_config(None, sets + ["dataset.enforce_expected_count=false"])


@pytest.fixture(autouse=True)
def release_test_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close('all')
