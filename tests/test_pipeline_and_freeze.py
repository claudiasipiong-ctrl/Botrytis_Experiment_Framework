"""Pipeline integrado (errores E_det/E_clas/E_flujo), umbrales, freeze/test bloqueado y pairing de anotaciones separado del matching."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.checks import MethodologyViolation
from utils.config_loader import load_config
from utils.freeze import load_freeze, verify_freeze
from utils.hashing import write_checksum_file
from utils.metrics_classification import pick_threshold, threshold_sweep
from utils.metrics_detection import ap_over_iou_table, conf_sweep, counts_at, pick_conf
from utils.pipeline_eval import DET_COLUMNS, MISS_COLUMNS, gt_from_fruit_table, summarize_from_table
from utils.splits import TestSetLockedError, assert_split_allowed
from utils.semantics import semantic_fingerprint

POS, NEG = "con_signos", "sin_signos"
OP = {"det_conf": 0.25, "nms_iou": 0.7, "cls_threshold": 0.5, "match_iou": 0.5}


def _det(rows: list[tuple]) -> pd.DataFrame:
    """(image, matched_fruit_id, gt_label, p, outcome)."""
    recs = []
    for i, (img, fid, gt, p, out) in enumerate(rows):
        recs.append({"image_id": img, "det_id": i, "x1": 0, "y1": 0, "x2": 1, "y2": 1, "det_conf": 0.9, "p_con_signos": p, "matched_fruit_id": fid, "gt_label": gt,
                     "iou": 0.8 if fid is not None else None, "outcome": out, "x_norm": 0.5, "y_norm": 0.5})
    return pd.DataFrame(recs, columns=DET_COLUMNS)


def test_error_rates_follow_the_formulas_of_the_spec() -> None:
    det = _det([("a", 0, POS, 0.9, None), ("a", 1, POS, 0.2, None), ("a", 2, NEG, 0.1, None), ("a", 3, NEG, 0.8, None),    # 4 asociados: 2 aciertos, 2 EC
                ("b", None, None, 0.7, "FP_no_gt"), ("b", None, None, 0.3, "FP_duplicate")])                                # 2 FP (localizacion)
    misses = pd.DataFrame([{"image_id": "b", "fruit_id": 5, "gt_label": POS, "x1": 0, "y1": 0, "x2": 1, "y2": 1, "x_norm": 0.5, "y_norm": 0.5}], columns=MISS_COLUMNS)
    s, tab, cm = summarize_from_table(det, misses, n_images=2, pos=POS, neg=NEG, op=OP)
    assert (s["n_gt"], s["tp"], s["fp"], s["fn"], s["n_duplicate_fp"], s["n_no_gt_fp"]) == (5, 4, 2, 1, 1, 1)
    e = s["errors"]
    assert e["e_det"] == pytest.approx(1 / 5) and e["e_clas"] == pytest.approx(2 / 4) and e["e_flujo"] == pytest.approx((1 + 2) / 5)
    assert e["n_ec"] == 2 and e["n_matched"] == 4
    assert list(tab.loc[tab["matched_fruit_id"].notna(), "outcome"]) == ["TP", "EC", "TP", "EC"]
    assert cm.shape == (3, 3) and cm[:, :].sum() == 4 + 1 + 2                       # asociados + FN + FP
    assert cm[0, 2] == 1 and cm[2, 0] + cm[2, 1] == 2                                # FN de con_signos; los 2 FP caen en la fila 'fondo'
    assert s["detection"]["precision"] == pytest.approx(4 / 6) and s["detection"]["recall"] == pytest.approx(4 / 5)


def test_classifier_threshold_can_be_retuned_offline_on_predicted_crops() -> None:
    det = _det([("a", 0, POS, 0.40, None), ("a", 1, POS, 0.45, None), ("a", 2, NEG, 0.10, None), ("a", 3, NEG, 0.20, None)])
    empty = pd.DataFrame(columns=MISS_COLUMNS)
    hi, _, _ = summarize_from_table(det, empty, n_images=1, pos=POS, neg=NEG, op={**OP, "cls_threshold": 0.5})
    lo, _, _ = summarize_from_table(det, empty, n_images=1, pos=POS, neg=NEG, op={**OP, "cls_threshold": 0.35})
    assert hi["errors"]["e_clas"] == pytest.approx(0.5) and lo["errors"]["e_clas"] == 0.0
    y = (det["gt_label"] == POS).astype(int)
    thr, val = pick_threshold(threshold_sweep(y, det["p_con_signos"]), "f1")
    assert 0.2 < thr <= 0.45 and val == pytest.approx(1.0)


def test_iou_matching_is_one_to_one_and_duplicates_are_false_positives() -> None:
    gt = [np.array([[0, 0, 10, 10.0]])]
    preds = [(np.array([[0, 0, 10, 10.0], [1, 1, 10, 10.0], [50, 50, 60, 60.0]]), np.array([0.9, 0.8, 0.7]))]
    c = counts_at(preds, gt, conf=0.0, iou_thr=0.5)
    assert (c["tp"], c["fp"], c["fn"]) == (1, 2, 0)                                   # duplicado + caja sin GT = FP de localizacion
    sw = conf_sweep(preds, gt, [0.0, 0.85], 0.5)
    assert list(sw["tp"]) == [1, 1] and list(sw["fp"]) == [2, 0]
    assert pick_conf(sw, "f1")[0] == 0.85
    tab = ap_over_iou_table(preds, gt)
    assert len(tab) == 10 and tab["iou"].iloc[0] == 0.5 and tab["ap"].is_monotonic_decreasing


def test_gt_table_to_pipeline_ground_truth() -> None:
    fr = pd.DataFrame({"image_id": ["a", "a"], "fruit_id": [1, 0], "x1": [5, 0], "y1": [5, 0], "x2": [9, 4], "y2": [9, 4], "image_w": [20, 20], "image_h": [20, 20], "label": [NEG, POS]})
    g = gt_from_fruit_table(fr, ["a", "z"])
    assert g["a"]["fruit_ids"] == [0, 1] and g["a"]["labels"] == [POS, NEG] and g["z"]["boxes"].shape == (0, 4)


# ---------------------------------- test bloqueado y freeze ----------------------------------- #
def test_test_split_is_locked_until_the_freeze_manifest_exists(tmp_path: Path) -> None:
    cfg = load_config(None, [f"test.freeze_manifest={tmp_path / 'freeze_manifest.json'}"])
    for stage in ("hpo_trial", "model_selection", "threshold_tuning", "pipeline_validation"):
        with pytest.raises(TestSetLockedError):
            assert_split_allowed("test", stage, cfg)
    with pytest.raises(TestSetLockedError, match="bloqueado"):
        assert_split_allowed("test", "final_test", cfg)
    with pytest.raises(TestSetLockedError, match="trials"):
        assert_split_allowed("validation", "hpo_trial", cfg)
    assert_split_allowed("train", "hpo_trial", cfg)
    (tmp_path / "freeze_manifest.json").write_text("{}")
    with pytest.raises(TestSetLockedError, match="freeze invalido"):
        assert_split_allowed("test", "final_test", cfg)
    cfg, _, fp = _freeze(tmp_path)
    with pytest.raises(RuntimeError,match='test cerrado'):
        assert_split_allowed('test','final_test',cfg)
    from utils.durable import atomic_json
    from utils.hashing import sha256_file
    atomic_json(tmp_path/'manual_test_release.json',dict(state='AUTORIZADO_PERSONALMENTE',freeze_sha256=sha256_file(fp),operator='SYNTHETIC_UNIT_FIXTURE',evidence='SYNTHETIC_NO_REAL_TEST'))
    assert_split_allowed("test", "final_test", cfg)
    with pytest.raises(TestSetLockedError):
        assert_split_allowed("test", "threshold_tuning", cfg)                         # el freeze no autoriza ajustar con test


def _freeze(tmp_path: Path) -> tuple[dict, dict, Path]:
    cfg = load_config(None, [f"paths.artifacts_dir={tmp_path}", f"test.freeze_manifest={tmp_path / 'freeze_manifest.json'}"])
    det, cls = tmp_path / "det.pt", tmp_path / "cls.pt"
    det.write_bytes(b"detector-weights")
    cls.write_bytes(b"classifier-weights")
    from utils.hashing import sha256_file

    manifest = {
        "semantic_fingerprint": semantic_fingerprint(cfg),
        "detector": {"checkpoint": str(det), "checkpoint_hash": sha256_file(det), "confidence_threshold": 0.3, "nms_iou": 0.6},
        "classifier": {"checkpoint": str(cls), "checkpoint_hash": sha256_file(cls), "probability_threshold": 0.45},
        "data": {"dataset_hash": "D", "split_hash": "S"}, "code": {"code_hash": "C"},
        "environment": {"python_version": "3.12.1", "torch": "2.5.0", "torchvision": "0.20", "timm": "1.0", "ultralytics": "8.4.170", "optuna": "5.0"},
    }
    from utils.comparison import protocol
    from utils.hashing import sha256_json
    derived = {}
    for task in ("detection", "classification"):
        folder = tmp_path / task; folder.mkdir()
        cfg["paths"][task + "_dir"] = str(folder)
        content = folder / "content_integrity.json"; content.write_text("{}")
        derived[task] = sha256_file(content)
    validation = tmp_path / "candidate_validation.json"; validation.write_text("{}")
    manifest["comparison"] = {"protocol_sha256": sha256_json(protocol(cfg)), "derived_content_sha256": derived,
        "candidates": [{"architecture": name, "checkpoint": str(ck), "checkpoint_sha256": sha256_file(ck),
        "validation_file": str(validation), "validation_sha256": sha256_file(validation)} for name, ck in (("det", det), ("cls", cls))]}
    fp = tmp_path / "freeze_manifest.json"
    fp.write_text(json.dumps(manifest))
    write_checksum_file(fp, fp.with_suffix(".sha256"))
    (tmp_path / "final_thresholds.json").write_text(json.dumps({"detector": {"conf": 0.3, "nms_iou": 0.6}, "classifier": {"probability_threshold": 0.45}}))
    env = {k: v for k, v in manifest["environment"].items()}
    return cfg, {"hashes": {"code_hash": "C", "dataset_hash": "D", "split_hash": "S"}, "env": env, "manifest": manifest}, fp


def test_verify_freeze_passes_when_nothing_changed_and_detects_every_change(tmp_path: Path) -> None:
    cfg, ctx, fp = _freeze(tmp_path)
    fz = load_freeze(cfg)
    verify_freeze(cfg, fz, ctx["hashes"], ctx["env"])
    with pytest.raises(MethodologyViolation, match="code_hash"):
        verify_freeze(cfg, fz, {**ctx["hashes"], "code_hash": "OTRO"}, ctx["env"])
    with pytest.raises(MethodologyViolation, match="split_hash"):
        verify_freeze(cfg, fz, {**ctx["hashes"], "split_hash": "OTRO"}, ctx["env"])
    with pytest.raises(MethodologyViolation, match="torch"):
        verify_freeze(cfg, fz, ctx["hashes"], {**ctx["env"], "torch": "9.9"})
    Path(fz["detector"]["checkpoint"]).write_bytes(b"retrained")
    with pytest.raises(MethodologyViolation, match="checkpoint del detector"):
        verify_freeze(cfg, fz, ctx["hashes"], ctx["env"])
    (tmp_path / "final_thresholds.json").write_text(json.dumps({"detector": {"conf": 0.5, "nms_iou": 0.6}, "classifier": {"probability_threshold": 0.45}}))
    Path(fz["detector"]["checkpoint"]).write_bytes(b"detector-weights")
    with pytest.raises(MethodologyViolation, match="umbrales del detector"):
        verify_freeze(cfg, fz, ctx["hashes"], ctx["env"])
    fp.write_text(fp.read_text().replace("0.3", "0.35"))                              # alterar el manifiesto sellado
    with pytest.raises(MethodologyViolation, match="SHA-256"):
        load_freeze(cfg)


def test_load_freeze_requires_the_manifest(tmp_path: Path) -> None:
    with pytest.raises(MethodologyViolation, match="15_freeze_experiment"):
        load_freeze(load_config(None, [f"test.freeze_manifest={tmp_path / 'nope.json'}"]))
