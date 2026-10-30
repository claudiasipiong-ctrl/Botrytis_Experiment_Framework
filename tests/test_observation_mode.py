"""Observation-only is explicit, preserves hardware identity, and never invents OEM verification."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from utils.config_loader import load_config
from utils.durable import atomic_json
from utils.hashing import scientific_config_hash
from utils.safety import SafetyMonitor, read_policy

ROOT = Path(__file__).resolve().parents[1]


def local_config(tmp_path):
    values = {"compute": {"pc_id": "PC-01"}, "dataset": {"n_images_expected": 324},
              "paths": {"artifacts_dir": str(tmp_path / "artifacts"), "runs_dir": str(tmp_path / "runs"),
                        "raw_images_dir": str(tmp_path / "original images"),
                        "raw_labels_dir": str(tmp_path / "original labels")}}
    path = tmp_path / "local.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def observed_policy(tmp_path):
    return {"gpu_uuid": "GPU-SYNTHETIC", "gpu_model": "SYNTHETIC", "mode": "observe",
            "user_requested_observation_only": True, "configured_by": "SYNTHETIC_OPERATOR",
            "configured_at": "2026-10-30T00:00:00+00:00", "cpu_thermal_protection_verified": False}


@pytest.mark.parametrize("cpu_sensor", [True, False])
def test_observation_records_hot_hardware_and_low_resources_without_threshold_stops(tmp_path, monkeypatch, cpu_sensor):
    import psutil
    import utils.safety as safety
    cfg = load_config(local_config(tmp_path))
    cfg["operations"] = {"safety_policy_file": str(tmp_path / "policy.json")}
    atomic_json(cfg["operations"]["safety_policy_file"], observed_policy(tmp_path))
    monkeypatch.setattr(safety.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="GPU-SYNTHETIC"))
    monkeypatch.setattr(safety, "gpu_reading", lambda _: {"name": "SYNTHETIC", "temperature.gpu": 99})
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=0))
    disk_paths = []
    def windows_disk_usage(path):
        if not isinstance(path, str):
            raise TypeError("argument 1 must be str, not WindowsPath")
        disk_paths.append(path)
        return SimpleNamespace(free=0)
    monkeypatch.setattr(psutil, "disk_usage", windows_disk_usage)
    monkeypatch.setattr(psutil, "cpu_percent", lambda **kw: 100)
    monkeypatch.setattr(psutil, "sensors_temperatures", lambda: {"cpu": [SimpleNamespace(current=105)]} if cpu_sensor else {}, raising=False)
    report = SafetyMonitor(cfg, tmp_path / "telemetry.jsonl").check()
    assert report["status"] == "PASS" and report["resource_limits_enforced"] is False
    assert report["cpu_thermal_protection_verified"] is False
    assert report["gpu"]["temperature.gpu"] == 99
    assert report["available_ram_bytes"] == 0
    assert len(disk_paths) == 2
    assert report["paths.artifacts_dir.free_bytes"] == 0
    assert report["paths.runs_dir.free_bytes"] == 0
    assert report["cpu_temperatures_c"] == ([105] if cpu_sensor else [])
    assert json.loads((tmp_path / "telemetry.jsonl").read_text())["mode"] == "observe"


@pytest.mark.parametrize("field,value", [("user_requested_observation_only", False), ("configured_by", ""), ("mode", "unknown")])
def test_observation_requires_explicit_valid_choice(tmp_path, field, value):
    cfg = {"operations": {"safety_policy_file": str(tmp_path / "policy.json")}}
    policy = observed_policy(tmp_path)
    policy[field] = value
    atomic_json(cfg["operations"]["safety_policy_file"], policy)
    with pytest.raises(ValueError):
        read_policy(cfg)


def test_observation_does_not_accept_another_physical_gpu(tmp_path, monkeypatch):
    import utils.safety as safety
    cfg = load_config(local_config(tmp_path))
    cfg["operations"] = {"safety_policy_file": str(tmp_path / "policy.json")}
    atomic_json(cfg["operations"]["safety_policy_file"], observed_policy(tmp_path))
    monkeypatch.setattr(safety.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="OTHER-GPU"))
    with pytest.raises(RuntimeError, match="GPU seleccionada"):
        SafetyMonitor(cfg, tmp_path / "telemetry.jsonl").check()


def configure_in_subprocess(path, *extra):
    # Fake only the GPU query inside the child; no actual hardware or training.
    code = """import runpy, subprocess, sys
from types import SimpleNamespace
subprocess.run = lambda *a, **k: SimpleNamespace(stdout='GPU-SYNTHETIC, SYNTHETIC\\n')
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    return subprocess.run([sys.executable, "-c", code, str(ROOT / "tools/configure_safety.py"),
                           "--config", str(path), "--gpu-index", "0", *extra],
                          capture_output=True, text=True, timeout=60, cwd=ROOT)


def test_cli_observation_preserves_dataset_configuration_and_backs_up_replacements(tmp_path):
    path = local_config(tmp_path)
    before = load_config(path)
    original = path.read_bytes()
    result = configure_in_subprocess(path, "--observe-only")
    assert result.returncode == 0, result.stdout + result.stderr
    after = load_config(path)
    assert scientific_config_hash(after) == scientific_config_hash(before)
    assert after["paths"]["raw_images_dir"] == before["paths"]["raw_images_dir"]
    assert after["dataset"]["n_images_expected"] == 324
    policy = read_policy(after)
    assert policy["mode"] == "observe" and policy["cpu_thermal_protection_verified"] is False
    assert policy["max_gpu_temperature_c"] is None
    assert policy["min_available_ram_bytes"] is None
    backups = list(tmp_path.glob("local.yaml.before_*"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    policy_path = Path(after["operations"]["safety_policy_file"])
    original_policy = policy_path.read_bytes()
    refused = configure_in_subprocess(path, "--observe-only")
    assert refused.returncode != 0 and policy_path.read_bytes() == original_policy
    replaced = configure_in_subprocess(path, "--observe-only", "--replace-existing")
    assert replaced.returncode == 0, replaced.stdout + replaced.stderr
    assert any(p.read_bytes() == original_policy for p in policy_path.parent.glob("safety_policy.json.before_*"))


def test_cli_default_still_requires_verified_limits(tmp_path):
    path = local_config(tmp_path)
    original = path.read_bytes()
    result = configure_in_subprocess(path)
    assert result.returncode == 2
    assert path.read_bytes() == original
    assert not (tmp_path / "artifacts/safety_policy.json").exists()


@pytest.mark.parametrize("cuda_available,training_failure", [(True, False), (True, True), (False, False)])
def test_training_check_orchestration_records_outcome_without_real_training(tmp_path, monkeypatch, cuda_available, training_failure):
    import torch
    import utils.bootstrap as bootstrap
    import utils.guards as guards
    import utils.safety as safety
    monkeypatch.setattr(bootstrap, "install", lambda *a: None)
    spec = importlib.util.spec_from_file_location("synthetic_qa_training_entry", ROOT / "tools/qa_training_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = local_config(tmp_path)
    output = tmp_path / "qa"
    monkeypatch.setattr(sys, "argv", ["qa_training_smoke.py", "--config", str(path), "--output", str(output)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(guards, "gpu_lock_path", lambda *a: tmp_path / "local_gpu.json")
    monitor = SimpleNamespace(policy={"mode": "observe"}, check=lambda: {})
    monkeypatch.setattr(safety, "SafetyMonitor", lambda *a: monitor)
    calls = []
    def fake_training(*args):
        calls.append(True)
        if training_failure:
            raise RuntimeError("SYNTHETIC_FAILURE")
        return {"classifier_epochs": 1, "detector_checkpoint": "SYNTHETIC_CHECKPOINT"}
    monkeypatch.setattr(module, "_run_smoke", fake_training)
    if cuda_available and not training_failure:
        assert module.main() == 0
    else:
        with pytest.raises(RuntimeError):
            module.main()
    report = json.loads((output / "training_check.json").read_text())
    assert report["status"] == ("COMPLETE" if cuda_available and not training_failure else "FAILED")
    assert report["supervision_mode"] == "observe" and report["scientific_experiment"] is False
    assert bool(calls) == cuda_available
