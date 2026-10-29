"""Graficos OBLIGATORIOS de la experimentacion (deteccion, clasificacion, pipeline).

Todas las funciones reciben ``DataFrame``/diccionarios con resultados REALES (leidos de ``runs/`` y
``artifacts/``) y devuelven la ``Figure``; se guardan con ``RunContext.save_figure(fig, nombre, data=df,
sources=[...])``, que deja el CSV de datos y el ``provenance.json`` que audita ``18_export_master_results``.
Con datos vacios dibujan "Sin datos" (``n_points = 0`` => la figura NO cuenta como evidencia).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from utils.visualization import _CYCLE, NEG_COLOR, POS_COLOR, _empty, pareto_front


def _color(i: int) -> str:
    return _CYCLE[i % len(_CYCLE)]


# =========================================================================== #
#                          DETECCION / COMPARACION                            #
# =========================================================================== #
def plot_metric_by_architecture(
    df: pd.DataFrame, metric_col: str, ylabel: str, title: str, *, label_col: str = "architecture", err_col: str | None = None,
    group_col: str | None = None, ylim: tuple[float, float] | None = None,
) -> Figure:
    """Barras de una metrica por arquitectura (p. ej. mAP@0.5:0.95, mAP@0.5, AP de ``con_signos``)."""
    fig, ax = plt.subplots(figsize=(max(6.5, 0.95 * max(len(df), 1) + 2.5), 4.6))
    d = df.dropna(subset=[metric_col]) if not df.empty and metric_col in df.columns else df.iloc[0:0]
    if d.empty:
        _empty(ax)
        ax.set_title(title)
        return fig
    x = np.arange(len(d))
    colors = [_color(0)] * len(d)
    if group_col and group_col in d.columns:
        groups = sorted(d[group_col].astype(str).unique())
        colors = [_color(groups.index(str(g))) for g in d[group_col]]
    err = d[err_col].to_numpy(float) if err_col and err_col in d.columns else None
    bars = ax.bar(x, d[metric_col].to_numpy(float), yerr=err, capsize=3, color=colors)
    for b, v in zip(bars, d[metric_col]):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", xytext=(0, 3 if err is None else 10),
                    textcoords="offset points", fontsize=8)
    ax.set_xticks(x, d[label_col].astype(str), rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    return fig


def plot_recall_vs_latency(df: pd.DataFrame, *, latency_col: str = "latency_ms", recall_col: str = "recall", label_col: str = "architecture") -> Figure:
    """Recall (validation externo) frente a la latencia por imagen de cada detector."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    d = df.dropna(subset=[latency_col, recall_col]) if not df.empty and {latency_col, recall_col} <= set(df.columns) else df.iloc[0:0]
    if d.empty:
        _empty(ax, "Sin datos de latencia (ejecute 17_benchmark_latency.py)")
        ax.set_title("Recall vs latencia")
        return fig
    for i, (_, r) in enumerate(d.iterrows()):
        ax.scatter(r[latency_col], r[recall_col], s=80, color=_color(i), zorder=3)
        ax.annotate(str(r[label_col]), (r[latency_col], r[recall_col]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Latencia [ms / imagen]")
    ax.set_ylabel("Recall")
    ax.set_title("Recall vs latencia (validation externo)")
    return fig


def plot_detector_radar(df: pd.DataFrame, *, label_col: str = "architecture", latency_col: str | None = "latency_ms") -> Figure:
    """Radar de detectores: mAP@0.5:0.95, mAP@0.5, precision, recall, F1 y eficiencia (FLOPs y, si hay, latencia).

    Los ejes de calidad son las metricas crudas en [0, 1]; los de eficiencia son ``min/valor`` (1 = el mas eficiente).
    """
    quality = [("map50_95", "mAP@0.5:0.95"), ("map50", "mAP@0.5"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    axes_def: list[tuple[str, str, bool]] = [(c, n, False) for c, n in quality if not df.empty and c in df.columns]
    if not df.empty and "gflops" in df.columns:
        axes_def.append(("gflops", "Eficiencia FLOPs", True))
    if latency_col and not df.empty and latency_col in df.columns and df[latency_col].notna().any():
        axes_def.append((latency_col, "Velocidad", True))
    fig = plt.figure(figsize=(7.2, 6.4))
    if df.empty or len(axes_def) < 3:
        ax = fig.add_subplot(111)
        _empty(ax)
        return fig
    ax = fig.add_subplot(111, projection="polar")
    n = len(axes_def)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    for i, (_, r) in enumerate(df.iterrows()):
        vals = []
        for col, _, inverse in axes_def:
            v = float(r[col]) if pd.notna(r[col]) else 0.0
            if inverse:
                pos = df[col].replace(0, np.nan).dropna()
                v = float(pos.min() / v) if v > 0 and len(pos) else 0.0
            vals.append(v)
        ax.plot(ang + ang[:1], vals + vals[:1], color=_color(i), lw=1.8, label=str(r[label_col]))
        ax.fill(ang + ang[:1], vals + vals[:1], color=_color(i), alpha=0.05)
    ax.set_xticks(ang, [n_ for _, n_, _ in axes_def], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title("Radar de detectores (validation externo)", pad=18, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    return fig


def plot_ap_vs_iou(ap_table: pd.DataFrame, *, label_col: str = "architecture") -> Figure:
    """AP a cada umbral de IoU (0.50:0.05:0.95) por detector. ``ap_table``: ``architecture, iou, ap``."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if ap_table.empty:
        _empty(ax)
        ax.set_title("AP vs umbral de IoU")
        return fig
    for i, (name, g) in enumerate(ap_table.groupby(label_col, sort=False)):
        g = g.sort_values("iou")
        ax.plot(g["iou"], g["ap"], marker="o", ms=4, color=_color(i), label=str(name))
    ax.set_xlabel("Umbral de IoU")
    ax.set_ylabel("AP")
    ax.set_ylim(0, 1.0)
    ax.set_title("AP vs umbral de IoU (validation externo)")
    ax.legend(fontsize=8, ncol=2)
    return fig


def plot_metric_vs_cost(
    df: pd.DataFrame, cost_col: str, cost_label: str, metric_col: str = "map50_95", metric_label: str = "mAP@0.5:0.95",
    *, label_col: str = "architecture", log_x: bool = True, title: str | None = None,
) -> Figure:
    """Metrica frente a un coste (FLOPs o parametros)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    d = df.dropna(subset=[cost_col, metric_col]) if not df.empty and {cost_col, metric_col} <= set(df.columns) else df.iloc[0:0]
    if d.empty:
        _empty(ax)
        ax.set_title(title or f"{metric_label} vs {cost_label}")
        return fig
    for i, (_, r) in enumerate(d.iterrows()):
        ax.scatter(r[cost_col], r[metric_col], s=80, color=_color(i), zorder=3, edgecolors="k")
        ax.annotate(str(r[label_col]), (r[cost_col], r[metric_col]), xytext=(6, 5), textcoords="offset points", fontsize=8)
    if log_x and (d[cost_col] > 0).all():
        ax.set_xscale("log")
    ax.set_xlabel(cost_label)
    ax.set_ylabel(metric_label)
    ax.set_title(title or f"{metric_label} vs {cost_label}")
    return fig


def plot_pareto(
    df: pd.DataFrame, cost_col: str, metric_col: str, cost_label: str, metric_label: str, title: str, *, label_col: str = "architecture",
) -> Figure:
    """Dispersion metrica-coste con la frontera de Pareto (menor coste, mayor metrica)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    d = df.dropna(subset=[cost_col, metric_col]).reset_index(drop=True) if not df.empty and {cost_col, metric_col} <= set(df.columns) else df.iloc[0:0]
    if d.empty:
        _empty(ax, "Sin datos (requiere 17_benchmark_latency.py)")
        ax.set_title(title)
        return fig
    mask = pareto_front(d, cost_col, metric_col)
    for i, r in d.iterrows():
        ax.scatter(r[cost_col], r[metric_col], s=80, color=_color(i), zorder=3, edgecolors="k" if mask[i] else "none", linewidths=1.6)
        ax.annotate(str(r[label_col]), (r[cost_col], r[metric_col]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    front = d[mask].sort_values(cost_col)
    ax.step(front[cost_col], front[metric_col], where="post", color="k", ls="--", lw=1.3, label="frontera de Pareto")
    ax.set_xlabel(cost_label)
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.legend()
    return fig


# =========================================================================== #
#                               HPO / OPTUNA                                  #
# =========================================================================== #
def plot_hpo_curves(trials: pd.DataFrame, intermediate: pd.DataFrame | None = None, *, title: str = "HPO", ylabel: str = "Objetivo (media de 3 folds)") -> Figure:
    """Curvas del HPO de UN estudio: score por trial + mejor acumulado, y curvas intermedias por epoca.

    Args:
        trials: ``number, value, state`` (``utils.optuna_storage.trials_table``).
        intermediate: ``trial, step, value`` (valores intermedios reportados al pruner) o ``None``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    if trials.empty or "value" not in trials.columns:
        for a in axes:
            _empty(a)
        fig.suptitle(title)
        return fig
    d = trials.sort_values("number")
    ok = d[d["state"].astype(str).str.contains("COMPLETE")]
    pr = d[d["state"].astype(str).str.contains("PRUNED")]
    ax = axes[0]
    ax.scatter(ok["number"], ok["value"], s=46, color=_color(0), label="COMPLETE", zorder=3)
    if len(pr):
        yv = pr["value"].fillna(pr["value"].min() if pr["value"].notna().any() else 0)
        ax.scatter(pr["number"], yv, s=46, facecolors="none", edgecolors="#7F8C8D", label="PRUNED", zorder=3)
    if len(ok):
        ax.step(ok["number"], ok["value"].cummax(), where="post", color=_color(1), lw=2, label="mejor acumulado")
    ax.set_xlabel("N.º de trial")
    ax.set_ylabel(ylabel)
    ax.set_title("Score por trial")
    ax.legend(fontsize=8)
    ax = axes[1]
    if intermediate is None or intermediate.empty:
        _empty(ax, "Sin valores intermedios")
    else:
        for tn, g in intermediate.groupby("trial"):
            g = g.sort_values("step")
            pruned = int(tn) in set(pr["number"].astype(int))
            ax.plot(g["step"], g["value"], color="#7F8C8D" if pruned else _color(int(tn)), alpha=0.55, lw=1.2, ls="--" if pruned else "-")
        ax.set_xlabel("Paso global (fold × epoca)")
        ax.set_ylabel("Metrica de validacion interna")
        ax.set_title("Curvas intermedias (discontinuo = podado)")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_optuna_distributions(trials: pd.DataFrame, param_cols: Sequence[str], importances: Mapping[str, float] | None = None, *, title: str = "Distribuciones Optuna", ncols: int = 3) -> Figure:
    """Distribucion de los hiperparametros muestreados frente al objetivo (slice plot) + importancias.

    Los parametros numericos se dibujan en escala log si abarcan >= 2 ordenes de magnitud; los categoricos
    como puntos por categoria. El ultimo panel muestra la importancia de cada parametro (fANOVA) si existe.
    """
    cols = [c for c in param_cols if not trials.empty and c in trials.columns and trials[c].notna().any()]
    d = trials[trials["value"].notna()] if not trials.empty and "value" in trials.columns else trials.iloc[0:0]
    n_panels = len(cols) + (1 if importances else 0) + 1
    if d.empty or not cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        _empty(ax)
        ax.set_title(title)
        return fig
    ncols = min(ncols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.4 * nrows), squeeze=False)
    flat = list(axes.ravel())
    for a in flat:
        a.axis("off")
    vmin, vmax = float(d["value"].min()), float(d["value"].max())
    for ax, c in zip(flat, cols):
        ax.axis("on")
        s = d[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            ax.scatter(s, d["value"], c=d["value"], cmap="viridis", vmin=vmin, vmax=vmax, s=34, edgecolors="k", linewidths=0.4)
            pos = s[s > 0]
            if len(pos) == len(s) and s.max() / max(s.min(), 1e-30) >= 100:
                ax.set_xscale("log")
            else:
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        else:
            cats = sorted(s.astype(str).unique())
            xs = s.astype(str).map({k: i for i, k in enumerate(cats)}) + np.random.default_rng(0).uniform(-0.12, 0.12, len(s))
            ax.scatter(xs, d["value"], c=d["value"], cmap="viridis", vmin=vmin, vmax=vmax, s=34, edgecolors="k", linewidths=0.4)
            ax.set_xticks(range(len(cats)), cats, rotation=20)
            ax.set_xlim(-0.6, len(cats) - 0.4)
        ax.set_xlabel(c)
        ax.set_ylabel("objetivo")
    k = len(cols)
    if importances:
        ax = flat[k]
        ax.axis("on")
        items = sorted(importances.items(), key=lambda kv: kv[1])
        ax.barh([i[0] for i in items], [i[1] for i in items], color=_color(2))
        ax.set_title("Importancia de parametros")
        k += 1
    ax = flat[k]
    ax.axis("on")
    ax.hist(d["value"], bins=min(10, max(3, len(d))), color=_color(0), edgecolor="k")
    ax.set_xlabel("objetivo")
    ax.set_ylabel("n.º de trials")
    ax.set_title("Distribucion del objetivo")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_robustness(per_seed: pd.DataFrame, summary: pd.DataFrame, *, title: str = "Robustez: top-3 x 3 seeds", ylabel: str = "Objetivo (media de 3 folds)") -> Figure:
    """Media +- SD (entre seeds) de cada configuracion finalista y el score de cada seed.

    Args:
        per_seed: ``config_id, seed, score`` (score = media de los 3 folds de esa seed).
        summary: ``config_id, mean, sd, selected`` (``selected`` marca la configuracion elegida por la MEDIA).
    """
    fig, ax = plt.subplots(figsize=(max(6.5, 1.8 * max(len(summary), 1) + 3), 4.6))
    if summary.empty or per_seed.empty:
        _empty(ax)
        ax.set_title(title)
        return fig
    x = np.arange(len(summary))
    colors = [_color(3) if bool(s) else _color(0) for s in summary["selected"]]
    ax.bar(x, summary["mean"], yerr=summary["sd"].fillna(0), capsize=4, color=colors, alpha=0.75)
    seeds = sorted(per_seed["seed"].unique())
    for si, s in enumerate(seeds):
        sub = per_seed[per_seed["seed"] == s].set_index("config_id")["score"]
        ax.scatter(x + (si - (len(seeds) - 1) / 2) * 0.12, [sub.get(c, np.nan) for c in summary["config_id"]], s=40, color="k", marker=["o", "s", "^", "D"][si % 4],
                   label=f"seed {s}", zorder=3)
    for xi, (m, sd) in enumerate(zip(summary["mean"], summary["sd"])):
        ax.annotate(f"{m:.3f} ± {0 if pd.isna(sd) else sd:.3f}", (xi, m), ha="center", va="bottom", xytext=(0, 12), textcoords="offset points", fontsize=8)
    ax.set_xticks(x, [f"{c}\n{'(elegida)' if s else ''}" for c, s in zip(summary["config_id"], summary["selected"])], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    return fig


# =========================================================================== #
#                               CLASIFICACION                                 #
# =========================================================================== #
def plot_pr_curves(curves: Mapping[str, Mapping[str, Any]], prevalence: float | None = None) -> Figure:
    """Curvas Precision-Recall superpuestas. ``curves[nombre] = {'precision': arr, 'recall': arr, 'ap': float}``."""
    fig, ax = plt.subplots(figsize=(6.4, 5))
    if not curves:
        _empty(ax)
        ax.set_title("Curva PR")
        return fig
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(c["recall"], c["precision"], color=_color(i), lw=2, label=f"{name} (AP = {c['ap']:.3f})")
    if prevalence is not None:
        ax.axhline(prevalence, color="#7F8C8D", ls=":", label=f"azar ({prevalence:.2f})")
    ax.set_xlabel("Recall (con_signos)")
    ax.set_ylabel("Precision (con_signos)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Curva Precision-Recall (validation externo)")
    ax.legend(loc="lower left", fontsize=8)
    return fig


def plot_roc_curves(curves: Mapping[str, Mapping[str, Any]]) -> Figure:
    """Curvas ROC superpuestas. ``curves[nombre] = {'fpr': arr, 'tpr': arr, 'auc': float}``."""
    fig, ax = plt.subplots(figsize=(6.4, 5))
    if not curves:
        _empty(ax)
        ax.set_title("Curva ROC")
        return fig
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(c["fpr"], c["tpr"], color=_color(i), lw=2, label=f"{name} (AUC = {c['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="#7F8C8D", ls=":", label="azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC (validation externo)")
    ax.legend(loc="lower right", fontsize=8)
    return fig


def plot_confusion_matrices(cms: Mapping[str, Any], labels: Sequence[str] = ("sin_signos", "con_signos"), title: str = "Matriz de confusion (validation externo)") -> Figure:
    """Una matriz de confusion 2x2 por clasificador (``[[TN, FP], [FN, TP]]``), con conteos y % por fila."""
    names = list(cms)
    if not names:
        fig, ax = plt.subplots(figsize=(4, 4))
        _empty(ax)
        ax.set_title(title)
        return fig
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        cm = np.asarray(cms[name], dtype=float)
        ax.imshow(cm, cmap="Blues")
        ax.grid(False)
        ax.set_xticks(range(2), [f"pred {x}" for x in labels], rotation=15)
        ax.set_yticks(range(2), labels)
        for i in range(2):
            for j in range(2):
                frac = cm[i, j] / cm[i].sum() if cm[i].sum() else 0
                ax.text(j, i, f"{int(cm[i, j])}\n({100 * frac:.0f} %)", ha="center", va="center", color="white" if cm[i, j] > cm.max() * 0.55 else "black")
        ax.set_title(name)
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Ground truth")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_gradcam_grid(samples_by_arch: Mapping[str, Sequence[Mapping[str, Any]]], ncols: int = 4) -> Figure:
    """Grad-CAM: una fila de ejemplos por clasificador. Cada muestra: ``{'crop': PIL, 'overlay': (H,W,3), 'title': str}``.

    Se muestran pares (crop, mapa superpuesto) de los primeros ``ncols // 2`` ejemplos.
    """
    archs = [a for a, s in samples_by_arch.items() if s]
    if not archs:
        fig, ax = plt.subplots(figsize=(5, 4))
        _empty(ax)
        ax.set_title("Grad-CAM")
        return fig
    per_row = max(1, ncols // 2)
    fig, axes = plt.subplots(len(archs), per_row * 2, figsize=(2.6 * per_row * 2, 2.9 * len(archs)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for r, name in enumerate(archs):
        for j, s in enumerate(list(samples_by_arch[name])[:per_row]):
            axes[r, 2 * j].imshow(s["crop"])
            axes[r, 2 * j].set_title(s.get("title", ""), fontsize=7)
            axes[r, 2 * j + 1].imshow(s["overlay"])
            axes[r, 2 * j + 1].set_title("Grad-CAM", fontsize=7)
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9, fontweight="bold")
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
    fig.suptitle("Grad-CAM de los clasificadores (clase con_signos)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_threshold_vs_metrics(sweeps: Mapping[str, pd.DataFrame], chosen: Mapping[str, float] | None = None, metric: str = "f1") -> Figure:
    """``threshold`` vs F1 (y precision/recall) por clasificador, marcando el umbral elegido."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if not sweeps:
        _empty(ax)
        ax.set_title("Umbral vs F1")
        return fig
    for i, (name, sw) in enumerate(sweeps.items()):
        ax.plot(sw["threshold"], sw[metric], color=_color(i), lw=2, label=f"{name} ({metric.upper()})")
        ax.plot(sw["threshold"], sw["recall"], color=_color(i), lw=1, ls=":", alpha=0.7)
        if chosen and name in chosen:
            t = chosen[name]
            ax.axvline(t, color=_color(i), ls="--", lw=1)
            ax.annotate(f"{t:.2f}", (t, 0.02), fontsize=8, color=_color(i), rotation=90, va="bottom")
    ax.set_xlabel("Umbral de probabilidad P(con_signos)")
    ax.set_ylabel(metric.upper() + "  (linea punteada: recall)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Umbral vs {metric.upper()} (validation externo; no confundir con el pairing GT)")
    ax.legend(fontsize=8)
    return fig


# =========================================================================== #
#                                  PIPELINE                                   #
# =========================================================================== #
def plot_accumulated_errors(rows: Sequence[Mapping[str, Any]]) -> Figure:
    """Errores acumulados del flujo: ``E_det`` (FN/N_GT) + ``EC/N_GT`` = ``E_flujo``, y ``E_clas`` (EC/matched).

    Args:
        rows: ``[{'label': 'validation', 'e_det': .., 'e_clas': .., 'e_flujo': .., 'n_gt': .., 'n_ec': ..}, ...]``.
    """
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    rows = [r for r in rows if r.get("e_det") is not None]
    if not rows:
        _empty(ax)
        ax.set_title("Errores acumulados del pipeline")
        return fig
    x = np.arange(len(rows))
    e_det = np.array([r["e_det"] for r in rows], float)
    ec_part = np.array([(r["n_ec"] / r["n_gt"]) if r["n_gt"] else 0.0 for r in rows], float)
    ax.bar(x, e_det, color=_color(4), label="E_det = FN / N_GT (no detectados)")
    ax.bar(x, ec_part, bottom=e_det, color=_color(1), label="EC / N_GT (errores de clasificacion)")
    for xi, r in enumerate(rows):
        tot = e_det[xi] + ec_part[xi]
        ax.annotate(f"E_flujo = {r['e_flujo']:.3f}\nE_clas = {r['e_clas']:.3f}" if r.get("e_clas") is not None else f"E_flujo = {r['e_flujo']:.3f}",
                    (xi, tot), ha="center", va="bottom", xytext=(0, 4), textcoords="offset points", fontsize=8)
    ax.set_xticks(x, [str(r["label"]) for r in rows])
    ax.set_ylabel("Tasa de error")
    ax.set_title("Errores acumulados: deteccion -> clasificacion -> flujo")
    ax.legend(fontsize=8)
    return fig


def plot_tp_fp_fn_counts(rows: Sequence[Mapping[str, Any]]) -> Figure:
    """TP / FP / FN de deteccion (matching por IoU) por conjunto evaluado."""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    rows = list(rows)
    if not rows:
        _empty(ax)
        ax.set_title("TP / FP / FN")
        return fig
    x = np.arange(len(rows))
    w = 0.26
    for k, (key, col) in enumerate((("tp", "#27AE60"), ("fp", POS_COLOR), ("fn", NEG_COLOR))):
        vals = [r[key] for r in rows]
        bars = ax.bar(x + (k - 1) * w, vals, w, color=col, label=key.upper())
        for b, v in zip(bars, vals):
            ax.annotate(str(v), (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, [str(r["label"]) for r in rows])
    ax.set_ylabel("N.º de cajas")
    ax.set_title("Detecciones: TP / FP / FN (IoU >= umbral de matching)")
    ax.legend()
    return fig
