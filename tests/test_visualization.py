"""Todas las funciones de graficos funcionan (con datos demo y con datos vacios)."""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from utils import visualization as viz

EXPECTED_DEMO = {
    "curvas_aprendizaje", "loss_train_val", "metricas_por_epoca", "hpo_trial_vs_score", "hiperparametros_ganadores",
    "baseline_vs_hpo", "heatmap_hiperparametros", "parallel_coordinates", "pareto_precision_latencia", "rendimiento_eficiencia",
    "parametros_flops_metrica", "tiempo_entrenamiento", "uso_vram", "diagrama_pipeline_hibrido", "latencia_descompuesta",
    "matriz_errores_pipeline", "mapa_espacial_errores", "ejemplos_tp_fp_fn",
}


def test_demo_generates_every_experiment_comparison_pipeline_figure(tmp_path: Path, synthetic_329: Path) -> None:
    from demo_figures import generate

    paths = generate(tmp_path, synthetic_329 / "images", synthetic_329 / "labels")
    names = {p.stem.removeprefix("DEMO_") for p in paths}
    assert EXPECTED_DEMO <= names
    assert all(p.stat().st_size > 5_000 for p in paths)


@pytest.mark.parametrize("fn", [
    viz.plot_dataset_distribution, viz.plot_class_balance_by_split, viz.plot_sign_coverage_distribution, viz.plot_crops_per_image,
])
def test_dataset_plots_tolerate_empty_input(fn) -> None:  # noqa: ANN001
    sig = inspect.signature(fn)
    empty = pd.DataFrame()
    args = [empty] * len([p for p in sig.parameters.values() if p.default is inspect._empty])
    fig = fn(*args)
    assert fig is not None


def test_other_plots_tolerate_empty_input() -> None:
    e = pd.DataFrame()
    for fig in (
        viz.plot_fruits_per_image(e), viz.plot_bbox_size_distribution(e), viz.plot_learning_curves(e, "a", "b", "y"),
        viz.plot_loss_curves(e), viz.plot_metrics_per_epoch(e, ["x"]), viz.plot_hpo_trials_vs_score(e),
        viz.plot_winning_hyperparameter_distribution(e), viz.plot_baseline_vs_hpo(e), viz.plot_hyperparameter_heatmap(e, "a", "b"),
        viz.plot_parallel_coordinates(e, ["a"]), viz.plot_accuracy_latency_pareto(e), viz.plot_performance_efficiency_comparison(e),
        viz.plot_params_flops_metric(e), viz.plot_training_time_by_architecture(e), viz.plot_vram_usage_by_model(e),
        viz.plot_latency_breakdown(e), viz.plot_tp_fp_fn_examples([]), viz.plot_gt_annotation_examples([]),
        viz.plot_spatial_error_map(pd.DataFrame(columns=["x_norm", "y_norm", "error_type"])), viz.plot_inner_fold_balance(e, e),
        viz.plot_annotation_class_counts({}),
    ):
        assert fig is not None


def test_static_diagrams_and_save_formats(tmp_path: Path) -> None:
    for fig, name in ((viz.plot_annotation_pairing_flow(0.50), "flow"), (viz.plot_hybrid_pipeline_diagram(), "pipe")):
        paths = viz.save_figure(fig, tmp_path, name, formats=("png", "pdf", "svg"), dpi=80)
        assert {p.suffix for p in paths} == {".png", ".pdf", ".svg"} and all(p.is_file() for p in paths)


def test_pareto_front_mask() -> None:
    df = pd.DataFrame({"lat": [1, 2, 3, 4], "acc": [0.5, 0.7, 0.6, 0.9]})
    assert viz.pareto_front(df, "lat", "acc").tolist() == [True, True, False, True]
