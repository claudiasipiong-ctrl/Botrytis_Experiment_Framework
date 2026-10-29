"""Espacios de busqueda, TPE/MedianPruner por arquitectura, guardias metodologicas y politica de seleccion."""

from __future__ import annotations

from pathlib import Path

try:
    import optuna
except ModuleNotFoundError:
    optuna = None
import pandas as pd
import pytest

from utils.architectures import archs, get_arch, trials_for
from utils.checks import MethodologyViolation
from utils.config_loader import FixedDecisionViolation, load_config
from utils.guards import assert_equal_hpo_budget, assert_image_size_fixed, assert_one_selected_by_mean, assert_same_pretrained_policy
from utils.optuna_storage import budget_used, create_or_load_study, make_pruner, make_sampler, storage_url
from utils.search_spaces import (
    baseline_hyperparameters, config_id, enqueue_params_from_hp, load_search_spaces, search_space_hash, space_for, suggest_hyperparameters,
)
from utils.selection import materialize_policy, rank_candidates

if optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture()
def cfg(tmp_path: Path) -> dict:
    return load_config(None, [f"paths.optuna_dir={tmp_path / 'optuna'}", f"paths.artifacts_dir={tmp_path / 'art'}"])


def _sample(spec_name: str, cfg: dict, seed: int = 0) -> dict:
    if optuna is None:
        pytest.skip("Optuna unavailable: sampling/search study test requires optional dependency")
    spec = get_arch(spec_name)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=seed))
    captured: dict = {}

    def obj(trial: optuna.Trial) -> float:
        captured.update(suggest_hyperparameters(trial, spec, cfg))
        return 0.0

    study.optimize(obj, n_trials=1)
    return captured


@pytest.mark.parametrize("name", [a.name for a in archs()])
def test_every_architecture_has_its_own_space_inside_the_declared_ranges(name: str, cfg: dict) -> None:
    for seed in range(8):
        hp = _sample(name, cfg, seed)
        spec = get_arch(name)
        space = space_for(spec, cfg)
        assert set(hp) == set(space)
        for k, d in space.items():
            v = hp[k]
            if d["type"] in ("float", "log_float"):
                assert d["min"] <= v <= d["max"], (k, v)
            elif d["type"] == "log_float_with_zero":
                assert v == 0.0 or d["min_nonzero"] <= v <= d["max"]
            elif d["type"] == "categorical":
                assert v in d["values"]
            elif d["type"] == "fixed":
                assert v == d["value"]
            elif d["type"] == "conditional_log_float":
                lo, hi = d["ranges"][hp[d["depends_on"]]]
                assert lo <= v <= hi


def test_spec_examples_of_the_search_spaces(cfg: dict) -> None:
    sp = load_search_spaces()
    assert sp["yolov8"]["optimizer"]["values"] == ["SGD", "AdamW"] and sp["yolov8"]["lr0"]["min"] == 1e-5 and sp["yolov8"]["lr0"]["max"] == 1e-2
    assert sp["rtdetr"]["optimizer"]["value"] == "AdamW" and sp["rtdetr"]["backbone_lr_ratio"]["value"] == 0.1
    assert sp["resnet50"]["lr"]["ranges"] == {"AdamW": [1e-5, 1e-3], "SGD": [1e-4, 1e-2]}
    assert sp["swin"]["drop_path_rate"]["values"] == [0.0, 0.1, 0.2] and "drop_path_rate" not in sp["vit"]
    assert sp["yolo26"]["freeze"]["values"] == [0, 10]
    assert "mosaic" not in space_for(get_arch("YOLO26n"), cfg)                      # no se abren mas augmentations sin ampliar el presupuesto
    assert len(search_space_hash()) == 64


def test_budgets_are_15_detectors_and_20_classifiers(cfg: dict) -> None:
    assert {trials_for(a, cfg) for a in archs("detection")} == {15} and {trials_for(a, cfg) for a in archs("classification")} == {20}
    assert sum(trials_for(a, cfg) for a in archs("detection")) == 105 and sum(trials_for(a, cfg) for a in archs("classification")) == 60
    assert_equal_hpo_budget(cfg)
    with pytest.raises(MethodologyViolation, match="desigual"):
        assert_equal_hpo_budget(cfg, {"YOLOv8n": 10})
    assert_equal_hpo_budget(cfg, {"YOLOv8n": 10}, declared_override=True)


def test_official_profile_fixes_sizes_pretraining_and_seeds() -> None:
    for override in ("detector.image_size=320", "classifier.batch_size=16", "detector.trials_per_architecture=10", "robustness.seeds=[1,2,3]", "optuna.sampler.name=Random"):
        with pytest.raises(FixedDecisionViolation):
            load_config(None, [override])
    cfg = load_config(None, ["project.profile=smoke", "detector.image_size=320"])       # el perfil smoke relaja presupuestos/tamanos
    assert_image_size_fixed(cfg)
    with pytest.raises(FixedDecisionViolation):
        load_config(None, ["project.profile=smoke", "detector.pretrained=false"])


def test_tpe_and_median_pruner_configuration_and_independent_studies(cfg: dict) -> None:
    if optuna is None:pytest.skip("Optuna unavailable; TPE/MedianPruner checks pending")
    sampler, pruner = make_sampler(cfg, 1), make_pruner(cfg)
    assert isinstance(sampler, optuna.samplers.TPESampler) and isinstance(pruner, optuna.pruners.MedianPruner)
    assert sampler._multivariate and sampler._group and sampler._n_startup_trials == 5                # sec. 10.2
    assert pruner._n_startup_trials == 5 and pruner._n_warmup_steps == 8 and pruner._interval_steps == 1   # sec. 10.3
    names = set()
    for spec in (get_arch("YOLOv8n"), get_arch("ResNet50")):
        st = create_or_load_study(spec, cfg, 1)
        st.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=2)
        names.add(st.study_name)
        assert storage_url(cfg, spec).endswith(f"botrytis_{spec.slug}.db")                              # un fichero por estudio
    assert names == {"botrytis_yolov8n", "botrytis_resnet50"}
    again = create_or_load_study(get_arch("YOLOv8n"), cfg, 1)                                           # load_if_exists: se reanuda
    assert len(again.trials) == 2 and budget_used(again) == 2


def test_zombie_running_trials_are_failed_on_resume_and_do_not_consume_budget(cfg: dict) -> None:
    if optuna is None:pytest.skip("Optuna unavailable; study resume checks pending")
    pytest.importorskip("torch",reason="HPO runner requires PyTorch")
    import importlib
    import logging

    hpo = importlib.import_module("07_run_hpo_study")
    spec = get_arch("ResNet50")
    study = create_or_load_study(spec, cfg, 1)
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=2)
    zombie = study.ask()                                                      # un trial RUNNING que dejo un corte de luz
    zombie.suggest_float("x", 0, 1)
    resumed = create_or_load_study(spec, cfg, 1)
    assert hpo.fail_zombie_trials(resumed, logging.getLogger("t")) == 1
    states = [t.state.name for t in resumed.trials]
    assert states.count("FAIL") == 1 and states.count("COMPLETE") == 2 and budget_used(resumed) == 2
    from utils.optuna_storage import failed_trials, trials_table

    assert failed_trials(resumed) == 1 and set(trials_table(resumed)["state"]) == {"COMPLETE", "FAIL"}
    inter = hpo.intermediate_table(resumed)
    assert list(inter.columns) == ["trial", "step", "value"]


def test_enqueue_recipe_trial_reproduces_the_baseline_hp(cfg: dict) -> None:
    if optuna is None:pytest.skip("Optuna unavailable; enqueue checks pending")
    spec = get_arch("YOLO26n")
    base = baseline_hyperparameters(spec, cfg)
    study = optuna.create_study(direction="maximize", sampler=make_sampler(cfg, 0))
    study.enqueue_trial(enqueue_params_from_hp(spec, base, cfg))
    captured: dict = {}

    def obj(trial: optuna.Trial) -> float:
        captured.update(suggest_hyperparameters(trial, spec, cfg))
        return 0.0

    study.optimize(obj, n_trials=1)
    for k, v in base.items():
        assert captured[k] == pytest.approx(v), k


def test_config_id_is_stable_and_order_independent() -> None:
    a = {"optimizer": "AdamW", "lr": 0.0001, "weight_decay": 1e-4}
    assert config_id(a) == config_id(dict(reversed(list(a.items())))) and config_id(a).startswith("cfg_") and config_id(a) != config_id({**a, "lr": 0.0002})


def test_baselines_match_the_annex(cfg: dict) -> None:
    d = baseline_hyperparameters(get_arch("YOLOv8n"), cfg)
    assert (d["optimizer"], d["lr0"], d["weight_decay"], d["warmup_epochs"]) == ("AdamW", 0.001, 0.0005, 3)
    r = baseline_hyperparameters(get_arch("RT-DETR-R50"), cfg)
    assert r["lr0"] == 1e-3 and r["backbone_lr_ratio"] == 0.1 and r["weight_decay"] == 1e-4
    assert baseline_hyperparameters(get_arch("ViT-B/16"), cfg)["lr"] == 5e-5 and baseline_hyperparameters(get_arch("ResNet50"), cfg)["lr"] == 1e-4


def test_selection_by_the_mean_of_three_seeds_not_a_single_run() -> None:
    assert_one_selected_by_mean({17: [0.5], 42: [0.6], 73: [0.55]}, [17, 42, 73])
    with pytest.raises(MethodologyViolation, match="unico seed"):
        assert_one_selected_by_mean({42: [0.6]}, [17, 42, 73])


def test_rank_candidates_primary_metric_min_recall_and_tie_breakers() -> None:
    df = pd.DataFrame({"architecture": ["A", "B", "C"], "map50_95": [0.600, 0.598, 0.50], "recall": [0.80, 0.90, 0.95], "gflops": [100.0, 50.0, 10.0]})
    tb = [{"metric": "recall", "direction": "max"}, {"metric": "gflops", "direction": "min"}]
    ranked, winner = rank_candidates(df, primary="map50_95", min_recall=None, recall_col="recall", tie_margin=0.005, tie_breakers=tb)
    assert winner == "B" and "desempate" in ranked.loc[ranked["architecture"] == "B", "reason"].iloc[0]     # A y B empatan (<=0.005): gana mayor recall
    ranked, winner = rank_candidates(df, primary="map50_95", min_recall=0.85, recall_col="recall", tie_margin=0.0, tie_breakers=[])
    assert winner == "B" and not ranked.loc[ranked["architecture"] == "A", "eligible"].iloc[0]             # A no cumple recall >= 0.85
    _, none = rank_candidates(df, primary="map50_95", min_recall=0.99, recall_col="recall", tie_margin=0.0, tie_breakers=[])
    assert none is None
    _, w0 = rank_candidates(df, primary="map50_95", min_recall=None, recall_col="recall", tie_margin=0.0, tie_breakers=tb)
    assert w0 == "A"                                                                                        # sin margen de empate gana la primaria


def test_selection_policy_is_materialized_once_and_then_immutable(tmp_path: Path) -> None:
    cfg = load_config(None, [f"paths.artifacts_dir={tmp_path}"])
    path, sha = materialize_policy(cfg)
    assert path.is_file() and (tmp_path / "selection_policy.sha256").is_file() and len(sha) == 64
    assert materialize_policy(cfg)[1] == sha
    changed = load_config(None, [f"paths.artifacts_dir={tmp_path}", "selection.detector.minimum_recall=0.9"])
    with pytest.raises(MethodologyViolation, match="cambio"):
        materialize_policy(changed)
