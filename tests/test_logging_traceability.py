"""Trazabilidad obligatoria: TODA ejecucion (exitosa o no) deja su run completo."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from utils.io import read_json, read_yaml
from utils.logging_utils import RUN_FILES, RunContext, run_cli

REQUIRED_META = [
    "run_id", "script", "status", "date", "start_time", "end_time", "started_at", "finished_at", "duration_seconds", "pc_id",
    "gpu", "cuda", "pytorch_version", "git_commit", "dataset_hash", "split_hash", "code_hash", "environment_hash", "seed",
    "parameters", "error",
]
REQUIRED_ENV = ["pc_id", "python_version", "gpu_model", "cuda_runtime", "torch", "git_commit", "dataset_hash", "split_hash", "environment_hash"]


def _only_run(runs: Path) -> Path:
    dirs = [p for p in (runs / "_pipeline").iterdir() if p.is_dir()]
    assert len(dirs) == 1, dirs
    return dirs[0]


def _assert_complete(run: Path) -> dict:
    for name in RUN_FILES:
        assert (run / name).is_file(), f"falta {name}"
    assert (run / "figures").is_dir()
    meta = read_json(run / "metadata.json")
    for key in REQUIRED_META:
        assert key in meta, f"metadata.json sin '{key}'"
    env = read_json(run / "environment.json")
    for key in REQUIRED_ENV:
        assert key in env, f"environment.json sin '{key}'"
    assert isinstance(read_yaml(run / "config.yaml"), dict)
    assert isinstance(read_json(run / "checkpoint_metadata.json"), dict)
    assert (run / "execution.log").read_text(encoding="utf-8").strip()
    return meta


def test_successful_run_writes_everything(tmp_path: Path) -> None:
    def main(ctx: RunContext, args: argparse.Namespace) -> None:
        ctx.logger.info("hola")
        ctx.log_metrics({"a": 1, "nested": {"b": 2}})
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        ctx.save_figure(fig, "fig_de_prueba")

    code = run_cli("t_ok", main, description="t", argv=["--runs-dir", str(tmp_path / "runs"), "--pc-id", "PC-02",
                                                       "--set", f"paths.figures_dir={tmp_path / 'rep'}", "--set", "outer_split.seed=5"])
    assert code == 0
    run = _only_run(tmp_path / "runs")
    meta = _assert_complete(run)
    assert meta["status"] == "COMPLETE" and meta["pc_id"] == "PC-02" and meta["pc_id_explicit"] is True
    assert meta["error"] is None and meta["duration_seconds"] >= 0
    assert meta["parameters"]["config_overrides"] == [f"paths.figures_dir={tmp_path / 'rep'}", "outer_split.seed=5"]
    assert read_json(run / "metrics.json") == {"a": 1, "nested": {"b": 2}}
    assert (run / "figures" / "fig_de_prueba.png").is_file() and (tmp_path / "rep" / "fig_de_prueba.png").is_file()
    assert run.name.endswith("_t_ok") and run.name[:8].isdigit()
    assert read_yaml(run / "config.yaml")["outer_split"]["seed"] == 5


def test_failed_run_keeps_full_traceback(tmp_path: Path) -> None:
    def main(ctx: RunContext, args: argparse.Namespace) -> None:
        ctx.log_metrics({"antes_del_fallo": True})
        raise ValueError("boom-xyz")

    assert run_cli("t_fail", main, description="t", argv=["--runs-dir", str(tmp_path / "runs")]) == 1
    run = _only_run(tmp_path / "runs")
    meta = _assert_complete(run)
    assert meta["status"] == "FAILED" and meta["error"]["type"] == "ValueError"
    assert "boom-xyz" in meta["error"]["message"] and "Traceback" in meta["error"]["traceback"]
    log = (run / "execution.log").read_text(encoding="utf-8")
    assert "Traceback" in log and "boom-xyz" in log and "hora_fin" in log
    assert read_json(run / "metrics.json") == {"antes_del_fallo": True}
    assert meta["errors_count"] >= 1


def test_config_error_still_creates_a_run(tmp_path: Path) -> None:
    def main(ctx: RunContext, args: argparse.Namespace) -> None:  # pragma: no cover - no debe ejecutarse
        raise AssertionError("main no debe correr con config invalida")

    code = run_cli("t_cfg", main, description="t", argv=["--runs-dir", str(tmp_path / "runs"), "--set", "classes.fruit_class_id=1"])
    assert code == 1
    meta = _assert_complete(_only_run(tmp_path / "runs"))
    assert meta["status"] == "FAILED" and meta["error"]["type"] == "FixedDecisionViolation"


def test_argument_error_still_creates_a_run(tmp_path: Path) -> None:
    def main(ctx: RunContext, args: argparse.Namespace) -> None:  # pragma: no cover
        raise AssertionError

    code = run_cli("t_args", main, description="t", argv=["--runs-dir", str(tmp_path / "runs"), "--argumento-inexistente"])
    assert code == 2
    meta = _assert_complete(_only_run(tmp_path / "runs"))
    assert meta["status"] == "FAILED" and meta["error"]["type"] == "CliArgumentError"


def test_keyboard_interrupt_and_system_exit(tmp_path: Path) -> None:
    def interrupted(ctx: RunContext, args: argparse.Namespace) -> None:
        raise KeyboardInterrupt

    assert run_cli("t_int", interrupted, description="t", argv=["--runs-dir", str(tmp_path / "runs")]) == 130
    assert read_json(next((tmp_path / "runs" / "_pipeline").iterdir()) / "metadata.json")["status"] == "INTERRUPTED"

    def exits_ok(ctx: RunContext, args: argparse.Namespace) -> None:
        raise SystemExit(0)

    assert run_cli("t_exit", exits_ok, description="t", argv=["--runs-dir", str(tmp_path / "runs2")]) == 0
    assert read_json(next((tmp_path / "runs2" / "_pipeline").iterdir()) / "metadata.json")["status"] == "COMPLETE"


def test_runs_do_not_collide(tmp_path: Path) -> None:
    def main(ctx: RunContext, args: argparse.Namespace) -> None:
        pass

    for _ in range(3):
        assert run_cli("t_same", main, description="t", argv=["--runs-dir", str(tmp_path / "runs")]) == 0
    assert len([p for p in (tmp_path / "runs" / "_pipeline").iterdir()]) == 3


def test_metadata_is_written_at_start_as_running(tmp_path: Path) -> None:
    """Si el proceso muriera a mitad, metadata.json ya existe con status RUNNING."""
    cfg = {"paths": {"runs_dir": str(tmp_path / "runs")}, "reproducibility": {"seed": 1}}
    ctx = RunContext(cfg, "t_crash")
    ctx.__enter__()
    meta = read_json(ctx.run_dir / "metadata.json")
    assert meta["status"] == "RUNNING" and meta["end_time"] is None and meta["seed"] == 1
    ctx.__exit__(None, None, None)
    assert read_json(ctx.run_dir / "metadata.json")["status"] == "COMPLETE"


def test_seed_reproducibility() -> None:
    import random

    import numpy as np

    from utils.reproducibility import set_global_seed

    set_global_seed(123)
    a = (random.random(), np.random.rand())
    set_global_seed(123)
    assert a == (random.random(), np.random.rand())


@pytest.mark.parametrize("cfg_seed", [None])
def test_no_default_seed_is_allowed(tmp_path: Path, cfg_seed: None) -> None:
    ctx = RunContext({"paths": {"runs_dir": str(tmp_path)}, "reproducibility": {"seed": cfg_seed}}, "t_noseed")
    with ctx:
        pass
    assert read_json(ctx.run_dir / "metadata.json")["seed"] is None
