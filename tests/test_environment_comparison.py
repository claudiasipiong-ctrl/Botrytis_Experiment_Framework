"""Small OS RAM reservations must not hide real fleet incompatibilities."""
from copy import deepcopy

import pytest

from orchestration.pc_registry import PCRegistry
from utils.reproducibility import compare_environments


@pytest.mark.parametrize("left,right", [(261604, 261587), (261604, 261540), (32768, 32740)])
def test_small_usable_ram_difference_preserves_reported_values(left, right):
    reference, candidate = {"ram_total_mb": left}, {"ram_total_mb": right}
    originals = deepcopy((reference, candidate))
    assert compare_environments(reference, candidate, ["ram_total_mb"]) == []
    assert compare_environments(candidate, reference, ["ram_total_mb"]) == []
    assert (reference, candidate) == originals


@pytest.mark.parametrize("left,right", [
    (261604, 261539), (32768, 32730), (261604, 130802),
    (261604, None), (261604, "261587"), (261604, 0), (261604, True),
    (261604, float("nan")), (261604, float("inf")),
])
def test_large_or_invalid_ram_difference_remains_blocking(left, right):
    assert compare_environments({"ram_total_mb": left}, {"ram_total_mb": right}, ["ram_total_mb"])


@pytest.mark.parametrize("field", ["gpu_vram_mb", "torch", "code_hash", "dataset_hash", "split_hash"])
def test_other_fields_still_require_exact_equality(field):
    differences = compare_environments({field: 261604}, {field: 261587}, [field])
    assert differences == [{"field": field, "reference": 261604, "candidate": 261587}]


def test_fleet_accepts_reservation_difference_but_rejects_code_change(tmp_path):
    import json

    cfg = {"paths": {"environments_dir": str(tmp_path)}}
    reports = {"pc01": 261604, "pc02": 261604, "pc03": 261587}
    for pc, ram in reports.items():
        (tmp_path / f"environment_{pc}.json").write_text(
            json.dumps({"ram_total_mb": ram, "code_hash": "same"}), encoding="utf-8"
        )
    registry = PCRegistry(cfg)
    assert registry.missing_environments() == []
    assert registry.environment_mismatches() == []
    path = tmp_path / "environment_pc03.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["code_hash"] = "different"
    path.write_text(json.dumps(report), encoding="utf-8")
    mismatches = registry.environment_mismatches()
    assert len(mismatches) == 1
    assert mismatches[0]["field"] == "code_hash"
