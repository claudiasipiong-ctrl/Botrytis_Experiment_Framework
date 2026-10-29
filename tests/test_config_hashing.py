"""Configuracion (TODO bloqueantes, decisiones FIJAS, interpolacion) y hashes portables."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.config_loader import (
    BlockingTodoError, ConfigError, FixedDecisionViolation, is_todo, list_todos, load_config, require, resolve_path,
)
from utils.hashing import (
    compute_code_hash, compute_dataset_hash, read_checksum_file, sha256_text_file, verify_checksum_file, write_checksum_file,
)


def test_default_config_loads_and_has_todos() -> None:
    cfg = load_config()
    assert cfg["pairing"]["min_iou"] == 0.50
    todos = list_todos(cfg)
    assert cfg["classes"]["fruit_class_id"] == 0
    assert cfg["outer_split"]["seed"] == 67 and not todos


def test_blocking_todo_aborts() -> None:
    cfg = load_config()
    cfg["outer_split"]["seed"] = "TODO_FIXTURE"
    with pytest.raises(BlockingTodoError, match="outer_split.seed"):
        require(cfg, "outer_split.seed")
    assert require(cfg, "classes.fruit_class_id", "classes.sign_class_id") == [0, 2]


@pytest.mark.parametrize("override", [
    "classes.fruit_class_id=[0,1]", "classes.sign_class_id=1", "classes.suspicious_class_id=2",
    "outer_split.train=0.8", "outer_split.test_locked=false", "inner_hpo.unit=crop",
    "classification_dataset.margin_px=4", "compute.max_gpu_jobs_per_pc=2", "pairing.min_iou=0",
])
def test_fixed_decisions_cannot_change(override: str) -> None:
    with pytest.raises(FixedDecisionViolation):
        load_config(overrides=[override])


def test_equal_class_ids_rejected() -> None:
    with pytest.raises(FixedDecisionViolation):
        load_config(overrides=["classes.fruit_class_id=1", "classes.sign_class_id=1"])


def test_interpolation_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOTRYTIS_DATA_ROOT", str(tmp_path / "dd"))
    cfg = load_config()
    assert cfg["paths"]["detection_dir"] == str(tmp_path / "dd") + "/derived_v4/detection"
    monkeypatch.delenv("BOTRYTIS_DATA_ROOT")
    assert load_config()["paths"]["data_root"] == "data"
    with pytest.raises(ConfigError):
        load_config(overrides=["paths.x=${NO_SUCH_ENV_VAR_XYZ}"])


def test_overrides_yaml_types() -> None:
    cfg = load_config(overrides=["outer_split.seed=67", "foo.bar=[1,2]", "foo.baz=null"])
    assert cfg["outer_split"]["seed"] == 67 and cfg["foo"]["bar"] == [1, 2] and cfg["foo"]["baz"] is None
    assert is_todo("TODO_X") and not is_todo(None) and not is_todo(0)


def test_relative_paths_anchor_to_project_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    p = resolve_path("artifacts")
    assert p.name == "artifacts" and (p.parent / "src").is_dir()


def test_text_hash_is_newline_agnostic(tmp_path: Path) -> None:
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_bytes(b"0 0.5 0.5 0.1 0.1\r\n1 0.2 0.2 0.1 0.1\r\n")
    b.write_bytes(b"0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1")
    assert sha256_text_file(a) == sha256_text_file(b)


def test_dataset_hash_order_independent_and_sensitive() -> None:
    r = [("a", "1", "x"), ("b", "2", "y")]
    assert compute_dataset_hash(r) == compute_dataset_hash(list(reversed(r)))
    assert compute_dataset_hash(r) != compute_dataset_hash([("a", "1", "x"), ("b", "2", "z")])


def test_checksum_file_roundtrip(tmp_path: Path) -> None:
    f = tmp_path / "m.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8", newline="\n")
    digest = write_checksum_file(f, tmp_path / "m.sha256")
    assert read_checksum_file(tmp_path / "m.sha256") == digest
    assert verify_checksum_file(f, tmp_path / "m.sha256")
    f.write_text("a,b\n1,3\n", encoding="utf-8", newline="\n")
    assert not verify_checksum_file(f, tmp_path / "m.sha256")


def test_code_hash_ignores_line_endings(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    f = tmp_path / "src" / "x.py"
    f.write_bytes(b"print(1)\r\n")
    h1 = compute_code_hash(tmp_path)
    f.write_bytes(b"print(1)\n")
    assert compute_code_hash(tmp_path) == h1
    f.write_bytes(b"print(2)\n")
    assert compute_code_hash(tmp_path) != h1
