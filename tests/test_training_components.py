"""Componentes de entrenamiento sin dataset: schedule, RT-DETR (backbone lr), trasplante ImageNet, Grad-CAM, FLOPs, RunContext anidado."""

from __future__ import annotations

import pytest
pytest.importorskip("torch", reason="Optional ML dependency torch unavailable; no installation or weights download")
pytest.importorskip("torchvision", reason="Optional ML dependency torchvision unavailable; no installation or weights download")
pytest.importorskip("timm", reason="Optional ML dependency timm unavailable; no installation or weights download")
pytest.importorskip("ultralytics", reason="Optional ML dependency ultralytics unavailable; no installation or weights download")


import math
from pathlib import Path

import pytest
import torch
from torch import nn

from utils.architectures import get_arch
from utils.classifier_data import RANDOM_TRANSFORMS, assert_deterministic_transform, build_transforms, class_weights
from utils.classifier_models import build_classifier, load_classifier, save_classifier
from utils.classifier_train import build_optimizer, lr_at
from utils.config_loader import load_config
from utils.detector_train import _rtdetr_backbone_lr_callback
from utils.gradcam import gradcam, target_layer
from utils.model_stats import classifier_stats, count_gflops
from utils.pretrained import torchvision_to_ultralytics_key
from utils.training_common import is_oom, mean_sd, state_dict_sha256


def test_lr_schedule_is_linear_warmup_then_cosine_to_zero() -> None:
    base, warm, total = 1e-3, 10, 110
    assert lr_at(0, base, warm, total) == pytest.approx(base / 10) and lr_at(9, base, warm, total) == pytest.approx(base)
    assert lr_at(10, base, warm, total) == pytest.approx(base)                                   # inicio del coseno
    assert lr_at(60, base, warm, total) == pytest.approx(base / 2)                               # mitad del coseno
    assert lr_at(total, base, warm, total) == pytest.approx(0.0, abs=1e-12)
    assert all(lr_at(i, base, 0, 50) <= base for i in range(50))                                 # sin warmup


def test_eval_transform_is_deterministic_and_train_transform_is_random() -> None:
    from PIL import Image

    cfg = load_config(None, [])
    norm = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
    ev, tr = build_transforms(cfg, norm, train=False), build_transforms(cfg, norm, train=True)
    assert_deterministic_transform(ev)
    img = Image.effect_noise((90, 70), 40).convert("RGB")
    assert torch.equal(ev(img), ev(img)) and ev(img).shape == (3, 224, 224)
    outs = [tr(img) for _ in range(4)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:])
    with pytest.raises(Exception, match="Augmentation aleatoria"):
        assert_deterministic_transform(tr)
    assert {t.__name__ for t in RANDOM_TRANSFORMS} >= {"RandomResizedCrop", "RandomHorizontalFlip", "ColorJitter"}


def test_class_weights_are_inverse_to_frequency() -> None:
    import numpy as np

    w = class_weights(np.array([0] * 90 + [1] * 10), "weighted_ce")
    assert w is not None and w[1] / w[0] == pytest.approx(9.0) and class_weights(np.array([0, 1]), "none") is None
    assert class_weights(np.array([0, 0, 0]), "weighted_ce") is None                        # una clase ausente: sin pesos


def test_optimizer_groups_do_not_decay_norm_and_bias() -> None:
    cfg = load_config(None, [])
    model, _ = build_classifier(get_arch("ResNet50"), pretrained=False)
    opt = build_optimizer(model, {"optimizer": "AdamW", "lr": 1e-4, "weight_decay": 0.01}, cfg)
    assert opt.param_groups[0]["weight_decay"] == 0.01 and opt.param_groups[1]["weight_decay"] == 0.0
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(model.parameters()))
    assert type(build_optimizer(model, {"optimizer": "SGD", "lr": 1e-2, "weight_decay": 1e-4}, cfg)).__name__ == "SGD"
    with pytest.raises(ValueError):
        build_optimizer(model, {"optimizer": "Adagrad", "lr": 1e-2, "weight_decay": 0}, cfg)


@pytest.mark.parametrize("name", ["ResNet50", "ViT-B/16", "Swin-Tiny"])
def test_classifiers_have_two_logits_checkpoint_roundtrip_and_gradcam(name: str, tmp_path: Path) -> None:
    spec = get_arch(name)
    model, norm = build_classifier(spec, {"drop_path_rate": 0.1}, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    assert model(x).shape == (1, 2)
    path = tmp_path / "m.pt"
    save_classifier(path, model, spec, {"drop_path_rate": 0.1}, norm)
    m2, meta = load_classifier(path)
    assert torch.allclose(m2(x), model(x), atol=1e-5) and meta["arch"] == spec.name and meta["classes"] == ["sin_signos", "con_signos"]
    assert state_dict_sha256(model.state_dict()) == state_dict_sha256(m2.state_dict())
    cam, p = gradcam(model, spec, x)
    assert cam.shape == (224, 224) and 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-6 and 0.0 <= p <= 1.0
    assert target_layer(model, spec) is not None


def test_swin_drop_path_is_a_construction_hyperparameter() -> None:
    m, _ = build_classifier(get_arch("Swin-Tiny"), {"drop_path_rate": 0.2}, pretrained=False)
    rates = [float(b.drop_path1.drop_prob) for layer in m.layers for b in layer.blocks if hasattr(b.drop_path1, "drop_prob")]
    assert rates and max(rates) == pytest.approx(0.2)


def test_flops_and_params_use_the_same_criterion_across_families() -> None:
    m, _ = build_classifier(get_arch("ResNet50"), pretrained=False)
    s = classifier_stats(m, 224)
    assert 8.0 < s["gflops"] < 8.6 and 23.0 < s["params_m"] < 24.0                              # ResNet50: 4.1 GMACs = 8.2 GFLOPs, 23.5 M
    v, _ = build_classifier(get_arch("ViT-B/16"), pretrained=False)
    assert 33.0 < count_gflops(v, (1, 3, 224, 224)) < 37.0                                      # ViT-B/16: ~17.6 GMACs
    assert count_gflops(m, (1, 3, 448, 448)) == pytest.approx(4 * s["gflops"], rel=0.05)


# ---------------------------------- RT-DETR-R50 ---------------------------------------------- #
class _FakeRTDETR(nn.Module):
    """Modelo con la nomenclatura de Ultralytics: capas 0-4 = backbone, 5+ = cabeza."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(*[nn.Linear(4, 4) for _ in range(8)])


class _FakeTrainer:
    def __init__(self, lr: float = 1e-3) -> None:
        self.model = _FakeRTDETR()
        groups = [{"params": [p for n, p in self.model.named_parameters() if n.endswith("bias")], "param_group": "bias"},
                  {"params": [p for n, p in self.model.named_parameters() if n.endswith("weight")], "param_group": "weight", "weight_decay": 1e-4}]
        self.optimizer = torch.optim.AdamW(groups, lr=lr)
        self.epochs = 10
        self.lf = lambda x: 0.5 * (1 + math.cos(math.pi * x / self.epochs))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)


def test_rtdetr_backbone_lr_ratio_survives_the_scheduler() -> None:
    tr = _FakeTrainer(1e-3)
    n_params = sum(len(g["params"]) for g in tr.optimizer.param_groups)
    _rtdetr_backbone_lr_callback(0.1)(tr)
    groups = tr.optimizer.param_groups
    assert len(groups) == 4 and sum(len(g["params"]) for g in groups) == n_params
    assert {g["param_group"] for g in groups} == {"bias", "weight"}                                # se conserva el marcador de warmup de sesgos
    lrs = sorted({round(g["lr"], 12) for g in groups})
    assert lrs == [1e-4, 1e-3]
    backbone_ids = {id(p) for n, p in tr.model.named_parameters() if n.split(".")[1] in "01234"}
    for g in groups:
        is_bb = all(id(p) in backbone_ids for p in g["params"])
        assert g["lr"] == pytest.approx(1e-4 if is_bb else 1e-3) and g["initial_lr"] == pytest.approx(g["lr"])
    for _ in range(3):                                                                              # el planificador actualiza LOS 4 grupos
        tr.optimizer.step()
        tr.scheduler.step()
    for g in tr.optimizer.param_groups:
        is_bb = all(id(p) in backbone_ids for p in g["params"])
        base = 1e-4 if is_bb else 1e-3
        assert g["lr"] == pytest.approx(base * tr.lf(3))


def test_partial_r50_transplant_is_blocked():
    with pytest.raises(RuntimeError, match="obsoleta"):
        torchvision_to_ultralytics_key("conv1.weight")


def test_oom_detection_and_helpers() -> None:
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate")) and not is_oom(ValueError("otra cosa"))
    assert mean_sd([1.0, 3.0]) == (2.0, pytest.approx(math.sqrt(2))) and mean_sd([5.0]) == (5.0, 0.0) and mean_sd([]) == (None, None)
