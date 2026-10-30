"""Synthetic state dictionaries exercise timm compatibility without training or downloads."""
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from timm.models.swin_transformer import SwinTransformer
from utils.architectures import get_arch
from utils.classifier_models import build_classifier, load_classifier, save_classifier


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    def tiny_model(num_classes):
        return SwinTransformer(img_size=32, embed_dim=8, depths=(2, 2),
                               num_heads=(1, 2), window_size=2, num_classes=num_classes)

    source = tiny_model(1000)
    state = {k: v.clone() for k, v in source.state_dict().items()}
    derived = {k: v.clone() for k, v in source.named_buffers()
               if k.endswith(("relative_position_index", "attn_mask"))}
    assert derived and not (set(derived) & set(state))
    state.update(derived)
    path = tmp_path / "synthetic_swin.pt"

    def factory(name, pretrained, num_classes, **kwargs):
        assert pretrained is False  # never fetch official checkpoints
        return tiny_model(num_classes)

    monkeypatch.setattr(timm, "create_model", factory)
    monkeypatch.setattr("utils.weights.verified_local_weight", lambda *args: path)
    return source, state, path, derived


@pytest.mark.parametrize("include_legacy_buffers", [True, False])
def test_strict_swin_loading_preserves_backbone_and_binary_roundtrip(checkpoint, tmp_path, include_legacy_buffers):
    source, state, path, derived = checkpoint
    if not include_legacy_buffers:
        for key in derived:
            del state[key]
    torch.save(state, path)
    original = path.read_bytes()
    spec = get_arch("Swin-Tiny")
    model, norm = build_classifier(spec, pretrained=True, cfg={})
    assert model.head.fc.out_features == 2
    for name, parameter in source.named_parameters():
        if not name.startswith("head."):
            assert torch.equal(model.state_dict()[name], parameter)
    output = tmp_path / "own_checkpoint.pt"
    save_classifier(output, model, spec, {}, norm)
    restored, _ = load_classifier(output, "cpu")
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
    assert path.read_bytes() == original


@pytest.mark.parametrize("problem", ["missing", "unexpected", "shape"])
def test_swin_adapter_does_not_relax_strict_parameter_loading(checkpoint, problem):
    _, state, path, _ = checkpoint
    key = "layers.0.blocks.0.attn.qkv.weight"
    if problem == "missing":
        del state[key]
    elif problem == "unexpected":
        state["unrecognized_trainable_weight"] = torch.ones(2)
    else:
        state[key] = state[key][:-1]
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="Missing key|Unexpected key|size mismatch"):
        build_classifier(get_arch("Swin-Tiny"), pretrained=True, cfg={})
