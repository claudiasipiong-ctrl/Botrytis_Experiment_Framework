"""Split externo, folds internos, guardia del test y metricas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.checks import (
    MethodologyViolation, assert_crops_inherit_split, assert_folds_only_train, assert_one_split_per_image,
    assert_same_image_same_fold,
)
from utils.config_loader import load_config
from utils.folds import assign_stratified_folds, build_inner_folds, classification_strata, detection_strata
from utils.metrics import (
    average_precision_detection, average_precision_np, best_threshold, binary_classification_report, class_balance,
    confusion_matrix_binary, map_over_iou, match_detections, pipeline_confusion_matrix, pipeline_error_rates, roc_auc_np,
)
from utils.splits import (
    SPLITS, SplitIntegrityError, TestSetLockedError, allocate_counts, assert_split_allowed, assign_outer_split,
    build_split_manifest, load_split_manifest, write_split_manifest,
)

RATIOS = {"train": 0.70, "validation": 0.20, "test": 0.10}
IDS = [f"IMG_{i:04d}" for i in range(329)]


def test_allocation_of_329_images() -> None:
    assert allocate_counts(329, RATIOS) == {"train": 230, "validation": 66, "test": 33}
    for n in (10, 11, 99, 100, 327, 329, 1000):
        assert sum(allocate_counts(n, RATIOS).values()) == n


def test_split_is_deterministic_disjoint_and_complete() -> None:
    a = assign_outer_split(IDS, RATIOS, seed=1)
    b = assign_outer_split(list(reversed(IDS)), RATIOS, seed=1)
    assert a == b
    assert set(a) == set(IDS)
    assert pd.Series(a).value_counts().to_dict() == {"train": 230, "validation": 66, "test": 33}
    assert a != assign_outer_split(IDS, RATIOS, seed=2)


def test_stratified_split_keeps_global_counts_and_balances_strata() -> None:
    rng = np.random.default_rng(0)
    strata = {i: ("pos" if rng.random() < 0.27 else "neg") for i in IDS}
    a = assign_outer_split(IDS, RATIOS, seed=5, strata=strata)
    s = pd.Series(a)
    assert s.value_counts().to_dict() == {"train": 230, "validation": 66, "test": 33}
    frac = pd.crosstab(pd.Series(strata), s, normalize="columns")
    assert frac.loc["pos"].max() - frac.loc["pos"].min() < 0.06


def test_split_rejects_bad_seed_and_duplicates() -> None:
    with pytest.raises(ValueError):
        assign_outer_split(IDS, RATIOS, seed="abc")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        assign_outer_split(["a", "a", "b"], RATIOS, seed=1)


def test_manifest_is_immutable(tmp_path: Path) -> None:
    man = build_split_manifest(assign_outer_split(IDS, RATIOS, seed=3))
    csv_path, _, _ = write_split_manifest(man, tmp_path)
    assert (tmp_path / "split_manifest.sha256").is_file()
    assert load_split_manifest(tmp_path).equals(man)
    raw = csv_path.read_bytes()
    assert b"\r\n" not in raw  # hash estable Windows/Linux
    tampered = raw.replace(b",train,", b",test,", 1)
    csv_path.write_bytes(tampered)
    with pytest.raises(SplitIntegrityError, match="SHA-256"):
        load_split_manifest(tmp_path)


def test_test_split_is_locked(tmp_path: Path) -> None:
    cfg = load_config(overrides=[f"test.freeze_manifest={tmp_path / 'freeze_manifest.json'}"])
    for stage in ("hpo_trial", "threshold_tuning", "model_selection", "pipeline_validation", "final_test"):
        if stage == "final_test":
            with pytest.raises(TestSetLockedError, match="sigue bloqueado"):
                assert_split_allowed("test", stage, cfg)
        else:
            with pytest.raises(TestSetLockedError):
                assert_split_allowed("test", stage, cfg)
    (tmp_path / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(TestSetLockedError, match="freeze invalido"):
        assert_split_allowed("test", "final_test", cfg)
    assert_split_allowed("validation", "threshold_tuning", cfg)
    with pytest.raises(TestSetLockedError):
        assert_split_allowed("validation", "hpo_trial", cfg)
    assert_split_allowed("train", "hpo_trial", cfg)


def _summary(n: int = 230, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_f = rng.integers(1, 14, n)
    n_p = np.where(rng.random(n) < 0.3, rng.integers(1, 3, n), 0)
    n_p = np.minimum(n_p, n_f)
    return pd.DataFrame({"image_id": [f"T_{i:04d}" for i in range(n)], "n_fruits": n_f, "n_positive": n_p,
                         "has_positive_signs": n_p > 0, "pct_positive": 100 * n_p / n_f})


def test_inner_folds_are_balanced_and_stratified() -> None:
    cfg = load_config()
    s = _summary()
    det, cls = build_inner_folds(s, cfg, seed=11)
    for df, col in ((det, "inner_fold_det"), (cls, "inner_fold_cls")):
        sizes = df[col].value_counts()
        assert set(df[col]) == {0, 1, 2} and sizes.max() - sizes.min() <= 1
    pos = cls.groupby("inner_fold_cls")["has_positive_signs"].sum()
    assert pos.max() - pos.min() <= 1
    # determinismo
    det2, cls2 = build_inner_folds(s, cfg, seed=11)
    assert det.equals(det2) and cls.equals(cls2)
    assert not det["inner_fold_det"].equals(build_inner_folds(s, cfg, seed=12)[0]["inner_fold_det"])


def test_fold_strata_helpers() -> None:
    s = _summary(60)
    st = classification_strata(s, extra_positive_bin=True)
    assert set(st.values()) <= {"cls_neg", "cls_pos_lo", "cls_pos_hi"}
    d = detection_strata(s.set_index("image_id")["n_fruits"], 3)
    assert set(d.values()) == {"det_bin0", "det_bin1", "det_bin2"}
    f = assign_stratified_folds(list(s["image_id"]), st, 3, 0)
    assert set(f.values()) == {0, 1, 2}


def test_methodology_checks_raise() -> None:
    man = pd.DataFrame({"image_id": ["a", "b", "c"], "outer_split": ["train", "validation", "test"], "split_hash": "h"})
    assert_one_split_per_image(man)
    with pytest.raises(MethodologyViolation):
        assert_one_split_per_image(pd.concat([man, man.iloc[[0]].assign(outer_split="test")]))
    crops = pd.DataFrame({"image_id": ["a", "a"], "outer_split": ["train", "validation"], "inner_fold": [0, 0]})
    with pytest.raises(MethodologyViolation, match="split distinto"):
        assert_crops_inherit_split(crops, man)
    crops2 = pd.DataFrame({"image_id": ["a", "a"], "outer_split": ["train", "train"], "inner_fold": [0, 1]})
    with pytest.raises(MethodologyViolation, match="folds distintos"):
        assert_same_image_same_fold(crops2)
    folds = pd.DataFrame({"image_id": ["a", "b"], "inner_fold_cls": [0, 1]})
    with pytest.raises(MethodologyViolation, match="EXACTAMENTE outer train"):
        assert_folds_only_train(folds, man, "inner_fold_cls", 3)


# ------------------------------- metricas ---------------------------------- #
def test_classification_metrics_match_sklearn() -> None:
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.15).astype(int)
    score = np.clip(0.3 * y + rng.normal(0.3, 0.2, 400), 0, 1)
    score[:20] = np.round(score[:20], 1)  # empates
    assert roc_auc_np(y, score) == pytest.approx(sk.roc_auc_score(y, score))
    assert average_precision_np(y, score) == pytest.approx(sk.average_precision_score(y, score))
    rep = binary_classification_report(y, score, threshold=0.5)
    pred = (score >= 0.5).astype(int)
    assert rep["precision"] == pytest.approx(sk.precision_score(y, pred))
    assert rep["recall"] == pytest.approx(sk.recall_score(y, pred))
    assert rep["f1"] == pytest.approx(sk.f1_score(y, pred))
    assert rep["f2"] == pytest.approx(sk.fbeta_score(y, pred, beta=2))
    assert np.array_equal(confusion_matrix_binary(y, pred), sk.confusion_matrix(y, pred))
    t, v = best_threshold(y, score, "f1")
    assert v >= rep["f1"] - 1e-12 and 0 <= t <= 1


def test_degenerate_classification_cases() -> None:
    assert roc_auc_np([0, 0, 0], [0.1, 0.2, 0.3]) is None
    assert average_precision_np([0, 0], [0.1, 0.2]) is None
    rep = binary_classification_report([0, 0, 1, 1], y_pred=[0, 0, 0, 0])
    assert rep["precision"] == 0.0 and rep["recall"] == 0.0 and rep["f1"] == 0.0


def test_class_balance() -> None:
    cb = class_balance(["con_signos"] * 10 + ["sin_signos"] * 90)
    assert cb["imbalance_ratio"] == pytest.approx(9.0) and cb["pct_con_signos"] == pytest.approx(10.0)
    assert class_balance(["sin_signos"] * 5)["imbalance_ratio"] is None


def test_matching_one_to_one_and_duplicates() -> None:
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]], float)
    pred = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 29, 29], [80, 80, 90, 90]], float)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    res = match_detections(pred, scores, gt, 0.5)
    assert [(p, g) for p, g, _ in res["matches"]] == [(0, 0), (2, 1)]
    assert res["fp"] == [1, 3]      # duplicado + sin GT
    assert res["fn"] == [2]


def test_map_perfect_and_degraded() -> None:
    gts = [np.array([[0, 0, 10, 10], [20, 20, 40, 40]], float), np.array([[5, 5, 25, 25]], float)]
    perfect = [(g.copy(), np.ones(len(g))) for g in gts]
    m = map_over_iou(perfect, gts)
    assert m["map50"] == pytest.approx(1.0, abs=1e-6) and m["map50_95"] == pytest.approx(1.0, abs=1e-6)
    shifted = [(g + 2.0, np.ones(len(g))) for g in gts]
    assert average_precision_detection(shifted, gts, 0.5) > average_precision_detection(shifted, gts, 0.9)
    empty = [(np.zeros((0, 4)), np.zeros(0)) for _ in gts]
    assert average_precision_detection(empty, gts, 0.5) == 0.0


def test_pipeline_errors_and_confusion() -> None:
    r = pipeline_error_rates(n_gt=100, n_fn=10, n_ec=9, n_matched=90)
    assert r["e_det"] == pytest.approx(0.10) and r["e_clas"] == pytest.approx(0.10) and r["e_flujo"] == pytest.approx(0.19)
    cm = pipeline_confusion_matrix(["con_signos", "sin_signos", "sin_signos"], ["con_signos", "sin_signos", "con_signos"],
                                   {"con_signos": 1}, ["sin_signos"])
    assert cm.shape == (3, 3) and cm[0, 0] == 1 and cm[1, 1] == 1 and cm[1, 0] == 1 and cm[0, 2] == 1 and cm[2, 1] == 1
    assert pipeline_error_rates(0, 0, 0, 0)["e_det"] is None
    assert SPLITS == ("train", "validation", "test")
