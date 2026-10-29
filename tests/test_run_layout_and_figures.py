"""Layout runs/<arquitectura>/<trial>, estados PRUNED/FAILED, checkpoint_metadata, procedencia de figuras y graficos obligatorios."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from utils.architectures import archs
from utils.config_loader import load_config
from utils.figure_catalog import required_figures
from utils.io import read_json
from utils.logging_utils import RUN_FILES, RunContext
from utils.plots_experiment import (
    plot_accumulated_errors, plot_ap_vs_iou, plot_confusion_matrices, plot_detector_radar, plot_gradcam_grid, plot_hpo_curves, plot_metric_by_architecture,
    plot_metric_vs_cost, plot_optuna_distributions, plot_pareto, plot_pr_curves, plot_recall_vs_latency, plot_robustness, plot_roc_curves, plot_threshold_vs_metrics,
    plot_tp_fp_fn_counts,
)
from utils.scientific_audit import check_figures, check_logs, check_test_locked


def _cfg(tmp_path: Path) -> dict:
    return load_config(None, [f"paths.runs_dir={tmp_path / 'runs'}", f"paths.figures_dir={tmp_path / 'figs'}", f"paths.artifacts_dir={tmp_path / 'art'}"])


def _assert_complete(d: Path) -> None:
    for f in RUN_FILES:
        assert (d / f).is_file(), f
    assert (d / "figures").is_dir()


def test_architecture_trial_layout_and_mandatory_files(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with RunContext(cfg, "07_run_hpo_study", architecture="yolov8n", trial_name="hpo/trial_003", pc_id="PC-02") as ctx:
        ctx.add_metadata(stage="hpo", hp={"lr0": 0.01})
        ctx.note_split_access("train", "hpo_trial")
        ctx.record_checkpoint("fold0_best", sha256="abc", fold=0, weights_discarded=True)
        ctx.log_metrics({"value": 0.5})
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        ctx.save_figure(fig, "f", data=pd.DataFrame({"x": [1, 2, 3]}), copy_to_reports=False)
    run = tmp_path / "runs" / "yolov8n" / "hpo" / "trial_003"
    _assert_complete(run)
    meta = read_json(run / "metadata.json")
    assert "07_run_hpo_study_yolov8n" in meta["run_id"] and meta["architecture"] == "yolov8n" and meta["status"] == "COMPLETE"
    assert meta["splits_accessed"] == [{"split": "train", "stage": "hpo_trial"}] and meta["extra"]["hp"] == {"lr0": 0.01}
    ck = read_json(run / "checkpoint_metadata.json")
    assert ck["checkpoints"][0]["sha256"] == "abc" and ck["architecture"] == "yolov8n" and set(ck) >= {"dataset_hash", "code_hash", "environment_hash", "split_hash"}
    prov = read_json(run / "figures" / "f.provenance.json")
    assert prov["n_points"] == 3 and prov["has_data"] is True and prov["synthetic"] is False
    assert (run / "figures" / "f.data.csv").is_file()


def test_reuse_existing_resumes_the_same_directory(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    for _ in range(2):
        with RunContext(cfg, "06_run_baseline", architecture="resnet50", trial_name="baseline", reuse_existing=True):
            pass
    assert [p.name for p in (tmp_path / "runs" / "resnet50").iterdir()] == ["baseline"]
    with RunContext(cfg, "x", architecture="resnet50", trial_name="baseline"):                       # sin reuse: nuevo directorio
        pass
    assert sorted(p.name for p in (tmp_path / "runs" / "resnet50").iterdir()) == ["baseline", "baseline_2"]


def test_pruned_trial_is_not_a_failure_and_failed_keeps_traceback(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    class TrialPruned(Exception):
        pass

    with pytest.raises(TrialPruned):
        with RunContext(cfg, "07", architecture="a", trial_name="hpo/trial_001"):
            raise TrialPruned("podado")
    meta = read_json(tmp_path / "runs" / "a" / "hpo" / "trial_001" / "metadata.json")
    assert meta["status"] == "PRUNED" and meta["error"] is None and meta["errors_count"] == 0
    with pytest.raises(ValueError):
        with RunContext(cfg, "07", architecture="a", trial_name="hpo/trial_002"):
            raise ValueError("boom")
    meta = read_json(tmp_path / "runs" / "a" / "hpo" / "trial_002" / "metadata.json")
    assert meta["status"] == "FAILED" and "Traceback" in meta["error"]["traceback"]
    _assert_complete(tmp_path / "runs" / "a" / "hpo" / "trial_002")


def test_nested_context_reuses_the_environment_of_the_parent(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with RunContext(cfg, "07", architecture="a", trial_name="hpo/study_run") as parent:
        child = RunContext(cfg, "07", architecture="a", trial_name="hpo/trial_000", runs_dir=parent.runs_base, env_override=parent.env)
        with child:
            pass
    assert read_json(child.run_dir / "environment.json")["environment_hash"] == parent.env["environment_hash"]


def test_audit_check_logs_detects_a_run_missing_files(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with RunContext(cfg, "06_run_baseline", architecture="a", trial_name="baseline"):
        pass
    runs = [{"dir": tmp_path / "runs" / "a" / "baseline", "meta": read_json(tmp_path / "runs" / "a" / "baseline" / "metadata.json")}]
    assert check_logs(cfg, runs).status in ("PASS", "FAIL")          # hashes de dataset/split ausentes => FAIL en un run de entrenamiento sin datos
    (tmp_path / "runs" / "a" / "baseline" / "execution.log").unlink()
    assert check_logs(cfg, runs).status == "FAIL"


def test_audit_test_access_is_only_allowed_for_script_16_after_the_freeze(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    (tmp_path / "art").mkdir()
    (tmp_path / "art" / "test_lock.json").write_text("{}")
    leak = [{"dir": tmp_path, "meta": {"run_id": "r1", "script": "07_run_hpo_study", "started_at": "2026-01-01", "splits_accessed": [{"split": "test", "stage": "hpo_trial"}]}}]
    assert check_test_locked(cfg, leak).status == "FAIL"
    clean = [{"dir": tmp_path, "meta": {"run_id": "r2", "script": "07_run_hpo_study", "splits_accessed": [{"split": "train", "stage": "hpo_trial"}]}}]
    assert check_test_locked(cfg, clean).status == "PASS"
    val_in_hpo = [{"dir": tmp_path, "meta": {"run_id": "r3", "script": "07", "splits_accessed": [{"split": "validation", "stage": "hpo_trial"}]}}]
    assert check_test_locked(cfg, val_in_hpo).status == "FAIL"


def test_audit_compares_instants_across_timezones(tmp_path: Path) -> None:
    from utils.scientific_audit import _before

    assert not _before("2026-09-30T17:50:56-05:00", "2026-09-30T22:50:40+00:00")      # 22:50:56 UTC > 22:50:40 UTC: DESPUES del freeze
    assert _before("2026-09-30T17:50:30-05:00", "2026-09-30T22:50:40+00:00")
    cfg = _cfg(tmp_path)
    (tmp_path / "art").mkdir()
    (tmp_path / "art" / "test_lock.json").write_text("{}")
    (tmp_path / "art" / "freeze_manifest.json").write_text(json.dumps({"frozen_at": "2026-09-30T22:50:40+00:00"}))
    cfg["test"]["freeze_manifest"] = str(tmp_path / "art" / "freeze_manifest.json")
    ok = [{"dir": tmp_path, "meta": {"run_id": "r", "script": "16_evaluate_test", "started_at": "2026-09-30T17:50:56-05:00", "splits_accessed": [{"split": "test", "stage": "final_test"}]}}]
    assert check_test_locked(cfg, ok).status == "PASS"


def test_required_figure_catalog_covers_the_mandatory_charts() -> None:
    names = {f["name"] for f in required_figures()}
    plt.close("all")
    for must in ("det_map50_95_por_arquitectura", "det_map50_por_arquitectura", "det_recall_vs_latencia", "det_radar_detectores", "det_ap_vs_iou", "det_map_vs_flops",
                 "det_map_vs_parametros", "det_pareto_map_latencia", "det_pareto_map_vram", "cls_curva_pr", "cls_curva_roc", "cls_matriz_confusion", "cls_gradcam",
                 "cls_umbral_vs_f1", "cls_ap_positiva_por_arquitectura", "pipe_arquitectura_completa_test", "pipe_latencia_por_etapa", "pipe_errores_acumulados_test",
                 "pipe_tp_fp_fn_test"):
        assert must in names, must
    for a in archs():
        assert f"hpo_curvas_{a.slug}" in names and f"hpo_distribuciones_{a.slug}" in names
    assert len(names) == len(required_figures())


def test_check_figures_requires_real_data_provenance(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    rep = tmp_path / "figs"
    rep.mkdir()
    req = required_figures()
    for i, f in enumerate(req):
        (rep / f"{f['name']}.png").write_bytes(b"png")
        (rep / f"{f['name']}.provenance.json").write_text(json.dumps({"n_points": 5, "has_data": True, "synthetic": False}))
    assert check_figures(cfg).status == "PASS"
    (rep / f"{req[0]['name']}.provenance.json").write_text(json.dumps({"n_points": 0, "has_data": False, "synthetic": False}))
    bad = check_figures(cfg)
    assert bad.status == "FAIL" and req[0]["name"] in bad.summary
    (rep / f"{req[0]['name']}.provenance.json").write_text(json.dumps({"n_points": 9, "has_data": True, "synthetic": True}))
    assert check_figures(cfg).status == "FAIL"
    (rep / f"{req[1]['name']}.png").unlink()
    miss = check_figures(cfg)
    assert miss.status == "FAIL" and req[1]["name"] in miss.evidence["missing"]
    assert check_figures(cfg, allow_missing=True).status == "WARN"
    (rep / "DEMO_algo.png").write_bytes(b"x")
    assert "DEMO" in check_figures(cfg).summary


# ----------------------------------- funciones de graficos ------------------------------------ #
def _det_df() -> pd.DataFrame:
    n = ["YOLOv8n", "YOLOv8m", "RT-DETR-R50"]
    return pd.DataFrame({"architecture": n, "map50_95": [0.5, 0.6, 0.4], "map50": [0.8, 0.85, 0.7], "precision": [0.7, 0.8, 0.6], "recall": [0.75, 0.8, 0.6], "f1": [0.72, 0.8, 0.6],
                         "gflops": [8.0, 79.0, 136.0], "params_m": [3.0, 25.0, 42.0], "latency_ms": [5.0, 12.0, 30.0], "vram_inference_mb": [300.0, 900.0, 2000.0]})


def test_every_experiment_plot_draws_with_data_and_with_empty_inputs() -> None:
    d = _det_df()
    ap = pd.DataFrame({"architecture": ["A"] * 10, "iou": np.linspace(0.5, 0.95, 10), "ap": np.linspace(0.9, 0.1, 10)})
    trials = pd.DataFrame({"number": range(6), "state": ["COMPLETE"] * 4 + ["PRUNED"] * 2, "value": [0.4, 0.5, 0.45, 0.6, 0.3, 0.2], "lr0": [1e-4, 1e-3, 1e-2, 1e-3, 1e-5, 1e-2],
                           "optimizer": ["SGD", "AdamW"] * 3})
    inter = pd.DataFrame({"trial": [0, 0, 1, 1], "step": [0, 1, 0, 1], "value": [0.1, 0.2, 0.15, 0.3]})
    curves = {"A": {"precision": np.array([1.0, 0.8, 0.5]), "recall": np.array([0.0, 0.5, 1.0]), "ap": 0.75}}
    roc = {"A": {"fpr": np.array([0.0, 0.3, 1.0]), "tpr": np.array([0.0, 0.8, 1.0]), "auc": 0.9}}
    sweep = pd.DataFrame({"threshold": np.linspace(0, 1, 11), "f1": np.linspace(0.2, 0.8, 11), "recall": np.linspace(1, 0, 11)})
    figs = [
        plot_metric_by_architecture(d, "map50_95", "mAP", "t"), plot_recall_vs_latency(d), plot_detector_radar(d), plot_ap_vs_iou(ap), plot_metric_vs_cost(d, "gflops", "G"),
        plot_pareto(d, "latency_ms", "map50_95", "ms", "mAP", "t"), plot_hpo_curves(trials, inter), plot_optuna_distributions(trials, ["lr0", "optimizer"], {"lr0": 0.7, "optimizer": 0.3}),
        plot_robustness(pd.DataFrame({"config_id": ["a", "a", "b", "b"], "seed": [17, 42, 17, 42], "score": [0.5, 0.55, 0.4, 0.45]}),
                        pd.DataFrame({"config_id": ["a", "b"], "mean": [0.525, 0.425], "sd": [0.03, 0.03], "selected": [True, False]})),
        plot_pr_curves(curves, 0.3), plot_roc_curves(roc), plot_confusion_matrices({"A": [[5, 1], [2, 4]]}), plot_threshold_vs_metrics({"A": sweep}, {"A": 0.5}),
        plot_accumulated_errors([{"label": "val", "e_det": 0.1, "e_clas": 0.2, "e_flujo": 0.28, "n_gt": 10, "n_ec": 2}]), plot_tp_fp_fn_counts([{"label": "val", "tp": 5, "fp": 2, "fn": 1}]),
        plot_gradcam_grid({"A": [{"crop": np.zeros((8, 8, 3)), "overlay": np.zeros((8, 8, 3)), "title": "x"}]}, 2),
    ]
    empties = [plot_metric_by_architecture(pd.DataFrame(), "map50_95", "m", "t"), plot_recall_vs_latency(pd.DataFrame()), plot_detector_radar(pd.DataFrame()), plot_ap_vs_iou(pd.DataFrame()),
               plot_metric_vs_cost(pd.DataFrame(), "gflops", "G"), plot_pareto(pd.DataFrame(), "a", "b", "a", "b", "t"), plot_hpo_curves(pd.DataFrame()), plot_optuna_distributions(pd.DataFrame(), []),
               plot_robustness(pd.DataFrame(), pd.DataFrame()), plot_pr_curves({}), plot_roc_curves({}), plot_confusion_matrices({}), plot_threshold_vs_metrics({}), plot_accumulated_errors([]),
               plot_tp_fp_fn_counts([]), plot_gradcam_grid({})]
    for f in figs + empties:
        assert f.axes
        plt.close(f)
    no_latency = d.drop(columns=["latency_ms"])
    assert "Sin datos" in "".join(t.get_text() for t in plot_recall_vs_latency(no_latency).axes[0].texts)
    plt.close("all")
