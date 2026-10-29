"""Entrenamiento REAL de un clasificador (CPU/GPU) sobre GT crops de un workspace sintetico: folds, callback de poda, early stopping,
modo fixed_epochs (70 % completo, sin validation), reintento por OOM y evaluacion por prediccion."""

from __future__ import annotations

import pytest
import os
pytestmark = pytest.mark.skipif(os.environ.get("BOTRYTIS_ALLOW_TRAINING_TESTS") != "1", reason="Training tests require explicit BOTRYTIS_ALLOW_TRAINING_TESTS=1; task forbids training")
pytest.importorskip("torch", reason="Optional ML dependency torch unavailable; no installation or weights download")
pytest.importorskip("torchvision", reason="Optional ML dependency torchvision unavailable; no installation or weights download")
pytest.importorskip("timm", reason="Optional ML dependency timm unavailable; no installation or weights download")


from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import utils.classifier_train as ct
from conftest import Workspace
from utils.architectures import get_arch
from utils.classifier_data import crops_for, load_crop_manifest
from utils.classifier_models import load_classifier, save_classifier
from utils.classifier_train import load_fold_frames, train_classifier
from utils.config_loader import load_config
from utils.metrics import average_precision_np

HP = {"optimizer": "AdamW", "lr": 3e-4, "weight_decay": 1e-4, "warmup_epochs": 0, "label_smoothing": 0.05}


@pytest.fixture()
def tiny_cfg(ws60: Workspace) -> dict:
    sets = [a for a in ws60.args() if "=" in a and not a.startswith("--")]
    # perfil smoke: sin descargar pesos, presupuestos diminutos; las reglas metodologicas siguen activas
    return load_config(None, sets + ["dataset.enforce_expected_count=false", "project.profile=smoke", "classifier.pretrained=false", "detector.pretrained=false", "initialization.mode=scratch", "initialization.scratch_decision=SYNTHETIC_DEV_FIXTURE", "classifier.batch_size=8",
                                     "classifier.workers=0", "classifier.max_epochs=2", "classifier.early_stopping_patience=1"])


def _subset(df: pd.DataFrame, n: int) -> pd.DataFrame:
    pos, neg = df[df["y"] == 1].head(n // 2), df[df["y"] == 0].head(n - n // 2)
    return pd.concat([pos, neg]).reset_index(drop=True)


def test_training_reports_each_epoch_early_stops_and_hashes_the_best_checkpoint(tiny_cfg: dict) -> None:
    man = load_crop_manifest(tiny_cfg)
    tr, va = load_fold_frames(man, tiny_cfg, 0)
    seen: list[tuple[int, float]] = []
    run = train_classifier(get_arch("ResNet50"), HP, tiny_cfg, _subset(tr, 16), _subset(va, 16), seed=1, max_epochs=3, patience=1,
                           epoch_callback=lambda step, row: seen.append((step, row["val_ap"])), step_offset=10, keep_state=True, log=lambda m: None)
    assert [s for s, _ in seen] == list(range(10, 10 + len(seen)))                      # pasos globales consecutivos (fold * max_epochs + epoca)
    assert run.epochs_run == len(seen) and 1 <= run.epochs_run <= 3 and (run.stopped_early or run.epochs_run == 3)
    assert 0.0 <= run.score <= 1.0 and run.best_epoch >= 1 and len(run.state_sha256) == 64 and run.state is not None
    assert list(run.history.columns)[:4] == ["epoch", "train_loss", "val_loss", "val_ap"] and run.params_m > 20 and run.gflops > 4
    assert run.val_prob.shape == run.val_true.shape == run.val_index.shape
    ap = average_precision_np(run.val_true, run.val_prob)
    assert ap is None or run.score == pytest.approx(ap)                                  # el score es el AP de con_signos en el fold de validacion


def test_a_callback_can_prune_the_trial_mid_training(tiny_cfg: dict) -> None:
    class Pruned(Exception):
        pass

    def cb(step: int, row: dict) -> None:
        if step >= 1:
            raise Pruned(f"podado en {step}")

    man = load_crop_manifest(tiny_cfg)
    tr, va = load_fold_frames(man, tiny_cfg, 1)
    with pytest.raises(Pruned):
        train_classifier(get_arch("ResNet50"), HP, tiny_cfg, _subset(tr, 16), _subset(va, 16), seed=2, max_epochs=3, epoch_callback=cb, log=lambda m: None)


def test_fixed_epochs_mode_trains_on_the_full_70_percent_without_looking_at_validation(tiny_cfg: dict, tmp_path: Path) -> None:
    man = load_crop_manifest(tiny_cfg)
    tr = crops_for(man, split="train", stage="outer_train", cfg=tiny_cfg)
    empty = pd.DataFrame(columns=list(tr.columns))
    calls: list[int] = []
    spec = get_arch("ResNet50")
    run = train_classifier(spec, HP, tiny_cfg, _subset(tr, 16), empty, seed=3, max_epochs=2, fixed_epochs=True, keep_state=True, epoch_callback=lambda s, r: calls.append(s), log=lambda m: None)
    assert run.epochs_run == 2 and not run.stopped_early and calls == []                # ni poda ni early stopping en el entrenamiento definitivo
    assert run.val_true.size == 0 and run.state is not None
    from utils.classifier_models import build_classifier

    model, _ = build_classifier(spec, HP, pretrained=False)
    model.load_state_dict(run.state)
    save_classifier(tmp_path / "final.pt", model, spec, HP, run.norm, epochs=2)
    loaded, meta = load_classifier(tmp_path / "final.pt")
    assert meta["epochs"] == 2 and loaded is not None


def test_oom_retry_halves_the_micro_batch_and_keeps_the_effective_batch(tiny_cfg: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    original = ct._train_once
    seen: list[int] = []

    def flaky(*a, **kw):  # type: ignore[no-untyped-def]
        seen.append(kw["micro_batch"])
        if len(seen) == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return original(*a, **kw)

    monkeypatch.setattr(ct, "_train_once", flaky)
    man = load_crop_manifest(tiny_cfg)
    tr, va = load_fold_frames(man, tiny_cfg, 0)
    run = train_classifier(get_arch("ResNet50"), HP, tiny_cfg, _subset(tr, 16), _subset(va, 16), seed=4, max_epochs=1, log=lambda m: None)
    assert seen == [8, 4] and run.micro_batch == 4 and run.accum_steps == 2            # batch efectivo 8 = 4 x 2
    monkeypatch.setattr(ct, "_train_once", lambda *a, **kw: (_ for _ in ()).throw(ValueError("otro error")))
    with pytest.raises(ValueError, match="otro error"):
        train_classifier(get_arch("ResNet50"), HP, tiny_cfg, _subset(tr, 16), _subset(va, 16), seed=4, max_epochs=1, log=lambda m: None)


def test_same_seed_gives_identical_fold_predictions(tiny_cfg: dict) -> None:
    man = load_crop_manifest(tiny_cfg)
    tr, va = load_fold_frames(man, tiny_cfg, 2)
    outs = [train_classifier(get_arch("ResNet50"), HP, tiny_cfg, _subset(tr, 12), _subset(va, 12), seed=7, max_epochs=1, log=lambda m: None) for _ in range(2)]
    if tiny_cfg["reproducibility"]["deterministic"]:
        assert np.allclose(outs[0].val_prob, outs[1].val_prob, atol=1e-4)
