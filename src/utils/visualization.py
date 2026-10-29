"""Generacion de graficos reutilizables (matplotlib, backend ``Agg``; sin seaborn).

Todas las funciones ``plot_*``:

* reciben ``DataFrame``/estructuras simples (nunca modelos) y devuelven la ``Figure``;
* no guardan nada por si mismas: usar :func:`save_figure` o ``RunContext.save_figure``;
* toleran datos vacios dibujando un aviso "sin datos" en lugar de fallar.

Catalogo (ver README):

* Dataset: distribucion y etiquetas, balance por split, frutos por imagen, tamanos de caja,
  cobertura de signos, crops por imagen, ejemplos GT, flujo de pairing de anotaciones.
* Experimentacion: curvas de aprendizaje, loss, metricas por epoca, HPO trial vs score,
  hiperparametros ganadores, baseline vs HPO, heatmap, coordenadas paralelas.
* Comparacion: Pareto precision-latencia, eficiencia, parametros-FLOPs-metrica, tiempo de
  entrenamiento, VRAM.
* Pipeline: diagrama hibrido, latencia descompuesta, matriz de errores, ejemplos TP/FP/FN,
  mapa espacial de errores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla: seguro en servidores/PCs de entrenamiento

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm as mpl_cm
from matplotlib import colors as mpl_colors
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from utils.io import ensure_dir, load_image_rgb

# --------------------------------------------------------------------------- #
# Estilo
# --------------------------------------------------------------------------- #
POS_COLOR = "#C0392B"      # con_signos
NEG_COLOR = "#2E86C1"      # sin_signos
SPLIT_COLORS = {"train": "#2C7FB8", "validation": "#F39C12", "test": "#7F8C8D"}
KIND_COLORS = {"suspect": "#8E44AD", "fruit": "#27AE60", "sign": "#E67E22"}
TP_COLOR, FP_COLOR, FN_COLOR = "#27AE60", "#C0392B", "#2471A3"
_CYCLE = ["#2C7FB8", "#E67E22", "#27AE60", "#8E44AD", "#C0392B", "#16A085", "#7F8C8D", "#D4AC0D", "#2E4053", "#AF7AC5"]

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.prop_cycle": matplotlib.cycler(color=_CYCLE), "legend.frameon": False,
})


def save_figure(
    fig: Figure,
    out_dir: str | Path,
    name: str,
    formats: Sequence[str] = ("png",),
    dpi: int = 200,
    close: bool = True,
) -> list[Path]:
    """Guarda ``fig`` como ``<out_dir>/<name>.<fmt>`` para cada formato.

    Args:
        close: Cierra la figura tras guardar (liberar memoria).

    Returns:
        Rutas escritas.
    """
    out = ensure_dir(out_dir)
    paths: list[Path] = []
    for fmt in formats:
        p = out / f"{name}.{fmt.lstrip('.')}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(p)
    if close:
        plt.close(fig)
    return paths


def _empty(ax: Any, msg: str = "Sin datos") -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#7F8C8D", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _bar_labels(ax: Any, bars: Any, fmt: str = "{:.0f}", pad: float = 2.0, **kw: Any) -> None:
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom",
                        xytext=(0, pad), textcoords="offset points", fontsize=8, **kw)


def _label_color(label: str, positive_label: str) -> str:
    return POS_COLOR if label == positive_label else NEG_COLOR


# =========================================================================== #
#                                   DATASET                                   #
# =========================================================================== #
def plot_annotation_class_counts(counts: Mapping[str, int], title: str = "Anotaciones por tipo") -> Figure:
    """Barras con el numero de anotaciones por tipo (p. ej. ``{'fruto': n, 'signo': m}``)."""
    fig, ax = plt.subplots(figsize=(5, 3.6))
    if not counts:
        _empty(ax)
    else:
        bars = ax.bar(list(counts), list(counts.values()), color=_CYCLE[: len(counts)])
        _bar_labels(ax, bars)
        ax.set_ylabel("N.º de cajas")
    ax.set_title(title)
    return fig


def plot_dataset_distribution(
    image_df: pd.DataFrame,
    fruit_df: pd.DataFrame,
    positive_label: str = "con_signos",
    negative_label: str = "sin_signos",
) -> Figure:
    """Distribucion del dataset y de las etiquetas (imagenes por split, frutos por clase, imagenes +/-).

    Args:
        image_df: ``image_id`` (+ ``outer_split`` opcional, + ``has_positive_signs`` opcional).
        fruit_df: ``label`` por fruto (+ ``outer_split`` opcional).
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axes[0]
    if "outer_split" in image_df.columns and len(image_df):
        c = image_df["outer_split"].value_counts().reindex(list(SPLIT_COLORS)).dropna()
        bars = ax.bar(c.index, c.values, color=[SPLIT_COLORS[s] for s in c.index])
        _bar_labels(ax, bars)
        for b, v in zip(bars, c.values):
            ax.text(b.get_x() + b.get_width() / 2, v / 2, f"{100 * v / c.sum():.1f} %", ha="center", va="center", color="white", fontsize=9)
        ax.set_ylabel("Imagenes")
        ax.set_title("Imagenes por split externo")
    elif len(image_df):
        bars = ax.bar(["imagenes"], [len(image_df)], color=_CYCLE[0])
        _bar_labels(ax, bars)
        ax.set_title("Imagenes")
    else:
        _empty(ax)

    ax = axes[1]
    if len(fruit_df):
        c = fruit_df["label"].value_counts().reindex([positive_label, negative_label]).fillna(0)
        bars = ax.bar(c.index, c.values, color=[POS_COLOR, NEG_COLOR])
        _bar_labels(ax, bars)
        for b, v in zip(bars, c.values):
            ax.text(b.get_x() + b.get_width() / 2, v / 2, f"{100 * v / c.sum():.1f} %", ha="center", va="center", color="white", fontsize=9)
        ratio = (c.max() / c.min()) if c.min() > 0 else float("inf")
        ax.set_title(f"Frutos por etiqueta (desbalance {ratio:.2f}:1)" if np.isfinite(ratio) else "Frutos por etiqueta")
        ax.set_ylabel("Frutos (crops)")
    else:
        _empty(ax)

    ax = axes[2]
    if len(image_df) and "has_positive_signs" in image_df.columns:
        n_pos = int(image_df["has_positive_signs"].sum())
        n_neg = int(len(image_df) - n_pos)
        bars = ax.bar(["con >=1 fruto\ncon_signos", "sin frutos\ncon_signos"], [n_pos, n_neg], color=[POS_COLOR, NEG_COLOR])
        _bar_labels(ax, bars)
        ax.set_title("Imagenes segun presencia de positivos")
        ax.set_ylabel("Imagenes")
    else:
        _empty(ax)
    fig.suptitle("Distribucion del dataset y de las etiquetas", fontweight="bold", y=1.03)
    fig.tight_layout()
    return fig


def plot_class_balance_by_split(
    fruit_df: pd.DataFrame, positive_label: str = "con_signos", negative_label: str = "sin_signos"
) -> Figure:
    """Balance de clases por split (conteos y porcentaje). Requiere ``outer_split`` y ``label``."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if fruit_df.empty or "outer_split" not in fruit_df.columns:
        for a in axes:
            _empty(a)
        return fig
    tab = (fruit_df.groupby(["outer_split", "label"]).size().unstack(fill_value=0)
           .reindex(index=[s for s in SPLIT_COLORS if s in set(fruit_df["outer_split"])],
                    columns=[positive_label, negative_label], fill_value=0))
    x = np.arange(len(tab))
    w = 0.38
    b1 = axes[0].bar(x - w / 2, tab[positive_label], w, color=POS_COLOR, label=positive_label)
    b2 = axes[0].bar(x + w / 2, tab[negative_label], w, color=NEG_COLOR, label=negative_label)
    _bar_labels(axes[0], b1)
    _bar_labels(axes[0], b2)
    axes[0].set_xticks(x, tab.index)
    axes[0].set_ylabel("Frutos (crops)")
    axes[0].set_title("Conteo por split")
    axes[0].legend()
    pct = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0) * 100
    axes[1].bar(x, pct[positive_label], color=POS_COLOR, label=positive_label)
    axes[1].bar(x, pct[negative_label], bottom=pct[positive_label], color=NEG_COLOR, label=negative_label)
    for xi, (p, n) in enumerate(zip(pct[positive_label], pct[negative_label])):
        if np.isfinite(p):
            axes[1].text(xi, p / 2, f"{p:.1f} %", ha="center", va="center", color="white", fontsize=9)
            axes[1].text(xi, p + n / 2, f"{n:.1f} %", ha="center", va="center", color="white", fontsize=9)
    axes[1].set_xticks(x, tab.index)
    axes[1].set_ylabel("% de frutos")
    axes[1].set_title("Proporcion por split")
    fig.suptitle("Balance de clases por split", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_fruits_per_image(image_df: pd.DataFrame, count_col: str = "n_fruits") -> Figure:
    """Histograma de frutos por imagen (apilado por split si existe ``outer_split``)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if image_df.empty:
        _empty(ax)
        return fig
    vmax = int(image_df[count_col].max())
    bins = np.arange(-0.5, vmax + 1.5, 1)
    if "outer_split" in image_df.columns:
        groups = [(s, image_df.loc[image_df["outer_split"] == s, count_col]) for s in SPLIT_COLORS if (image_df["outer_split"] == s).any()]
        ax.hist([g[1] for g in groups], bins=bins, stacked=True, color=[SPLIT_COLORS[g[0]] for g in groups], label=[g[0] for g in groups])
        ax.legend(title="split")
    else:
        ax.hist(image_df[count_col], bins=bins, color=_CYCLE[0])
    mean = image_df[count_col].mean()
    ax.axvline(mean, color="k", ls="--", lw=1)
    ax.text(mean, ax.get_ylim()[1] * 0.95, f" media = {mean:.1f}", va="top", fontsize=9)
    ax.set_xlabel("Frutos anotados por imagen")
    ax.set_ylabel("Imagenes")
    ax.set_title("Cantidad de frutos por imagen")
    return fig


def plot_bbox_size_distribution(boxes_df: pd.DataFrame) -> Figure:
    """Distribucion de tamanos de caja (lado equivalente en px, % del area de imagen, aspecto).

    Args:
        boxes_df: ``kind`` ('fruit'|'sign'), ``x1, y1, x2, y2``, ``image_w``, ``image_h`` (pixeles).
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    if boxes_df.empty:
        for a in axes:
            _empty(a)
        return fig
    d = boxes_df.copy()
    d["w_px"] = (d["x2"] - d["x1"]).clip(lower=0)
    d["h_px"] = (d["y2"] - d["y1"]).clip(lower=0)
    d["side_px"] = np.sqrt(d["w_px"] * d["h_px"])
    d["area_pct"] = 100 * d["w_px"] * d["h_px"] / (d["image_w"] * d["image_h"])
    d["aspect"] = d["w_px"] / d["h_px"].replace(0, np.nan)
    kinds = [k for k in KIND_COLORS if (d["kind"] == k).any()]
    for ax, col, xl, log in (
        (axes[0], "side_px", "Lado equivalente sqrt(w*h) [px]", False),
        (axes[1], "area_pct", "Area de la caja [% de la imagen]", True),
        (axes[2], "aspect", "Relacion de aspecto w/h", False),
    ):
        vals = d[col].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            _empty(ax)
            continue
        lo, hi = vals.min(), vals.max()
        bins = np.logspace(np.log10(max(lo, 1e-4)), np.log10(max(hi, lo * 1.01)), 35) if log else np.linspace(lo, hi, 35)
        for k in kinds:
            ax.hist(d.loc[d["kind"] == k, col].replace([np.inf, -np.inf], np.nan).dropna(), bins=bins, alpha=0.65,
                    color=KIND_COLORS[k], label=f"{k} (n={int((d['kind'] == k).sum())})")
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("Cajas")
    axes[0].legend()
    fig.suptitle("Distribucion de tamanos de bounding boxes", fontweight="bold", y=1.03)
    fig.tight_layout()
    return fig


def plot_sign_coverage_distribution(
    fruit_df: pd.DataFrame, threshold: float | None = None, positive_label: str = "con_signos", negative_label: str = "sin_signos"
) -> Figure:
    """Auxiliary sign coverage only; labels come from annotation pairing."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if fruit_df.empty:
        for a in axes:
            _empty(a)
        return fig
    cov = 100 * fruit_df["max_sign_coverage"]
    bins = np.linspace(0, 100, 41)
    for ax, sel, title in (
        (axes[0], np.ones(len(fruit_df), bool), "Todos los frutos"),
        (axes[1], (cov > 0).to_numpy(), "Solo frutos con cobertura > 0"),
    ):
        data = [cov[sel & (fruit_df["label"] == lbl).to_numpy()] for lbl in (negative_label, positive_label)]
        ax.hist(data, bins=bins, stacked=True, color=[NEG_COLOR, POS_COLOR], label=[negative_label, positive_label])
        ax.set_xlabel("max cobertura = area(signo ∩ fruto) / area(fruto)  [%]")
        ax.set_ylabel("Frutos")
        ax.set_title(title)
    n0 = int((cov == 0).sum())
    axes[0].text(0.02, 0.95, f"Cobertura = 0: {n0} ({100 * n0 / len(cov):.1f} %)", transform=axes[0].transAxes, va="top", fontsize=9)
    axes[0].legend(loc="upper center")
    fig.suptitle("Distribucion de la cobertura de signos", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_crops_per_image(
    summary_df: pd.DataFrame, positive_label: str = "con_signos", negative_label: str = "sin_signos"
) -> Figure:
    """Crops por imagen: histogramas de positivos/negativos y del % de positivos por imagen.

    Args:
        summary_df: ``labels.image_summary`` (``n_positive``, ``n_negative``, ``pct_positive``).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if summary_df.empty:
        for a in axes:
            _empty(a)
        return fig
    vmax = int(max(summary_df["n_positive"].max(), summary_df["n_negative"].max()))
    bins = np.arange(-0.5, vmax + 1.5, 1)
    axes[0].hist(summary_df["n_negative"], bins=bins, alpha=0.65, color=NEG_COLOR, label=f"{negative_label} (media {summary_df['n_negative'].mean():.1f})")
    axes[0].hist(summary_df["n_positive"], bins=bins, alpha=0.65, color=POS_COLOR, label=f"{positive_label} (media {summary_df['n_positive'].mean():.1f})")
    axes[0].set_xlabel("Crops por imagen")
    axes[0].set_ylabel("Imagenes")
    axes[0].set_title("Cantidad de crops por imagen y clase")
    axes[0].legend()
    with_fruit = summary_df[summary_df["n_fruits"] > 0]
    axes[1].hist(with_fruit["pct_positive"], bins=np.linspace(0, 100, 21), color=POS_COLOR, alpha=0.85)
    axes[1].set_xlabel(f"% de crops {positive_label} por imagen")
    axes[1].set_ylabel("Imagenes")
    axes[1].set_title("Porcentaje de crops positivos por imagen")
    fig.suptitle("Crops por imagen", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def select_gt_examples(fruit_df: pd.DataFrame, n: int = 6, seed: int = 0) -> list[str]:
    """Elige ``n`` ``image_id`` variados (con positivos, sin positivos, con signos parciales)."""
    if fruit_df.empty:
        return []
    rng = np.random.default_rng(seed)
    g = fruit_df.groupby("image_id")
    info = pd.DataFrame({
        "n": g.size(), "pos": g["label"].apply(lambda s: int((s == "con_signos").sum())),
        "maxcov": g["max_sign_coverage"].max(),
    })
    pools = [
        info[(info["pos"] > 0) & (info["pos"] < info["n"])].index.tolist(),   # mixtas
        info[info["pos"] == info["n"]].index.tolist(),                          # todas positivas
        info[info["pos"] == 0].index.tolist(),                                  # ninguna positiva
        info[(info["pos"] == 0) & (info["maxcov"] > 0)].index.tolist(),         # borde: cobertura parcial
    ]
    chosen: list[str] = []
    i = 0
    while len(chosen) < min(n, len(info)) and any(pools):
        pool = [p for p in pools[i % len(pools)] if p not in chosen]
        if pool:
            chosen.append(str(rng.choice(sorted(pool))))
        i += 1
        if i > 10 * n and len(chosen) < n:
            rest = [x for x in info.index if x not in chosen]
            chosen.extend(str(x) for x in rest[: n - len(chosen)])
            break
    return chosen[:n]


def plot_inner_fold_balance(det_folds: pd.DataFrame, cls_folds: pd.DataFrame) -> Figure:
    """Auditoria de los folds internos: imagenes/frutos por fold (deteccion) e imagenes
    positivas/negativas por fold (clasificacion).

    Args:
        det_folds: ``inner_fold_det`` y ``n_fruits`` por imagen de outer train.
        cls_folds: ``inner_fold_cls`` y ``has_positive_signs`` por imagen de outer train.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if det_folds.empty or cls_folds.empty:
        for a in axes:
            _empty(a)
        return fig
    g = det_folds.groupby("inner_fold_det").agg(imagenes=("image_id", "size"), frutos=("n_fruits", "sum"))
    x = np.arange(len(g))
    b1 = axes[0].bar(x - 0.2, g["imagenes"], 0.4, color=_CYCLE[0], label="imagenes")
    b2 = axes[0].bar(x + 0.2, g["frutos"], 0.4, color=_CYCLE[2], label="frutos")
    _bar_labels(axes[0], b1)
    _bar_labels(axes[0], b2)
    axes[0].set_xticks(x, [f"fold {i}" for i in g.index])
    axes[0].set_title("Deteccion: imagenes y frutos por fold")
    axes[0].legend()
    c = cls_folds.groupby("inner_fold_cls")["has_positive_signs"].agg(pos="sum", total="size")
    c["neg"] = c["total"] - c["pos"]
    x = np.arange(len(c))
    b1 = axes[1].bar(x - 0.2, c["pos"], 0.4, color=POS_COLOR, label="imagenes con >=1 con_signos")
    b2 = axes[1].bar(x + 0.2, c["neg"], 0.4, color=NEG_COLOR, label="imagenes sin con_signos")
    _bar_labels(axes[1], b1)
    _bar_labels(axes[1], b2)
    axes[1].set_xticks(x, [f"fold {i}" for i in c.index])
    axes[1].set_title("Clasificacion: imagenes por fold (estratificado)")
    axes[1].legend()
    fig.suptitle("Folds internos para HPO (solo outer train)", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_gt_annotation_examples(
    samples: Sequence[Mapping[str, Any]],
    positive_label: str = "con_signos",
    ncols: int = 3,
    max_side_px: int = 900,
) -> Figure:
    """Ejemplos visuales de anotaciones GT.

    Cada muestra: ``{"image_path", "title", "fruits": [{x1,y1,x2,y2,label,max_sign_coverage}],
    "signs": [(x1,y1,x2,y2)]}``. Frutos coloreados por etiqueta; signos en naranja discontinuo.
    """
    n = len(samples)
    if n == 0:
        fig, ax = plt.subplots(figsize=(5, 4))
        _empty(ax)
        return fig
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, s in zip(axes.ravel(), samples):
        img = load_image_rgb(s["image_path"])
        scale = min(1.0, max_side_px / max(img.size))
        if scale < 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        ax.imshow(img)
        for f in s.get("fruits", []):
            c = _label_color(f["label"], positive_label)
            ax.add_patch(Rectangle((f["x1"] * scale, f["y1"] * scale), (f["x2"] - f["x1"]) * scale, (f["y2"] - f["y1"]) * scale,
                                   fill=False, ec=c, lw=2))
            ax.text(f["x1"] * scale, f["y1"] * scale - 3, str(f["label"]), color="white", fontsize=7,
                    bbox={"facecolor": c, "pad": 1, "lw": 0, "alpha": 0.9}, va="bottom")
        for (x1, y1, x2, y2) in s.get("signs", []):
            ax.add_patch(Rectangle((x1 * scale, y1 * scale), (x2 - x1) * scale, (y2 - y1) * scale, fill=False, ec="#F1C40F", lw=1.5, ls="--"))
        ax.set_title(str(s.get("title", "")), fontsize=9)
        ax.axis("off")
    handles = [Rectangle((0, 0), 1, 1, fill=False, ec=POS_COLOR, lw=2), Rectangle((0, 0), 1, 1, fill=False, ec=NEG_COLOR, lw=2),
               Rectangle((0, 0), 1, 1, fill=False, ec="#F1C40F", lw=1.5, ls="--")]
    fig.legend(handles, ["fruto: con_signos (clase 1 emparejada)", "fruto: sin_signos (sin clase 1)", "caja de signo visual"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Ejemplos de anotaciones ground truth (etiqueta = pairing de anotaciones)", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


def _box(ax: Any, xy: tuple[float, float], w: float, h: float, text: str, fc: str = "#EAF2F8", ec: str = "#2C3E50", fs: float = 9, bold: bool = False) -> None:
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.3))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def _arrow(ax: Any, p0: tuple[float, float], p1: tuple[float, float], text: str = "", color: str = "#2C3E50", rad: float = 0.0) -> None:
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, lw=1.4, color=color, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.12, text, ha="center", fontsize=8, color=color)


def plot_annotation_pairing_flow(threshold: float = 0.50) -> Figure:
    """Annotation identity, strict anomaly gate, then crop inheritance."""
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.set_xlim(0,13);ax.set_ylim(0,4.4);ax.axis("off")
    boxes = [(0.2, "Imagen original\nclase 0 = fruto\nclase 1 = sospechoso"),
             (3.4, "Asignacion bipartita 1:1\nmaximizar IoU total\ndesempate por ordinal"),
             (6.6, f"Auditar candidatos/anomalias\nIoU minimo tecnico {threshold:.2f}\nAnomalias -> ABORTAR"),
             (9.8, "clase 1 emparejada: con_signos\nsin pareja: sin_signos\ncrop solo de clase 0")]
    for x, text in boxes:
        _box(ax,(x,2),2.9,1.4,text)
        if x>0.2:_arrow(ax,(x-0.3,2.7),(x,2.7))
    ax.text(.2,.8,"Clase 2: auditoria visual y Grad-CAM. No define etiquetas.\nSplit por imagen ANTES de crops; cada crop hereda split/fold.",fontsize=11)
    ax.set_title("Ground truth por pairing de anotaciones",fontweight="bold")
    return fig


def _curve(ax: Any, history: pd.DataFrame, col: str, color: str, label: str, group_col: str | None, x_col: str = "epoch", ls: str = "-") -> None:
    if col not in history.columns:
        return
    if group_col and group_col in history.columns and history[group_col].nunique() > 1:
        for _, g in history.groupby(group_col):
            ax.plot(g[x_col], g[col], color=color, alpha=0.25, lw=1, ls=ls)
        m = history.groupby(x_col)[col].agg(["mean", "std"]).fillna(0.0)
        ax.plot(m.index, m["mean"], color=color, lw=2, ls=ls, label=f"{label} (media)")
        ax.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"], color=color, alpha=0.15)
    else:
        ax.plot(history[x_col], history[col], color=color, lw=2, ls=ls, label=label)


def plot_learning_curves(
    history: pd.DataFrame, train_col: str, val_col: str, ylabel: str, *, group_col: str | None = None,
    mode: str = "max", title: str = "Curvas de aprendizaje",
) -> Figure:
    """Curvas de aprendizaje (train vs validation) de una metrica por epoca.

    Args:
        history: ``epoch`` + columnas de metrica (+ ``group_col``, p. ej. ``fold``/``seed``).
        mode: ``max`` | ``min``: marca la mejor epoca de validation.
    """
    fig, ax = plt.subplots(figsize=(7, 4.3))
    if history.empty:
        _empty(ax)
        return fig
    _curve(ax, history, train_col, _CYCLE[0], "train", group_col)
    _curve(ax, history, val_col, _CYCLE[1], "validation", group_col)
    if val_col in history.columns:
        agg = history.groupby("epoch")[val_col].mean()
        be = agg.idxmax() if mode == "max" else agg.idxmin()
        ax.axvline(be, color="k", ls=":", lw=1)
        ax.annotate(f"mejor epoca = {be}", (be, agg.loc[be]), xytext=(6, -14), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Epoca")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig


def plot_loss_curves(
    history: pd.DataFrame, train_loss_col: str = "train_loss", val_loss_col: str = "val_loss", *, group_col: str | None = None,
) -> Figure:
    """Loss de train y validation por epoca; sombrea la brecha y marca el minimo de validation."""
    fig, ax = plt.subplots(figsize=(7, 4.3))
    if history.empty:
        _empty(ax)
        return fig
    _curve(ax, history, train_loss_col, _CYCLE[0], "loss train", group_col)
    _curve(ax, history, val_loss_col, _CYCLE[1], "loss validation", group_col)
    if {train_loss_col, val_loss_col} <= set(history.columns) and history[val_loss_col].notna().any():
        m = history.groupby("epoch")[[train_loss_col, val_loss_col]].mean()
        ax.fill_between(m.index, m[train_loss_col], m[val_loss_col], color="#7F8C8D", alpha=0.12, label="brecha (generalizacion)")
        be = m[val_loss_col].idxmin()
        ax.axvline(be, color="k", ls=":", lw=1)
        ax.annotate(f"min val = {m.loc[be, val_loss_col]:.4f} (epoca {be})", (be, m.loc[be, val_loss_col]), xytext=(6, 10), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss")
    ax.set_title("Loss de entrenamiento y validation")
    ax.legend()
    return fig


def plot_metrics_per_epoch(history: pd.DataFrame, metrics: Sequence[str], *, group_col: str | None = None, ncols: int = 3) -> Figure:
    """Una celda por metrica (``metrics``) contra la epoca; ``group_col`` dibuja media +- SD."""
    cols = [m for m in metrics if m in history.columns]
    if history.empty or not cols:
        fig, ax = plt.subplots(figsize=(5, 4))
        _empty(ax)
        return fig
    ncols = min(ncols, len(cols))
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.3 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (ax, m) in enumerate(zip(axes.ravel(), cols)):
        ax.axis("on")
        _curve(ax, history, m, _CYCLE[i % len(_CYCLE)], m, group_col)
        ax.set_title(m)
        ax.set_xlabel("Epoca")
    fig.suptitle("Metricas por epoca", fontweight="bold")
    fig.tight_layout()
    return fig


def trials_dataframe_from_optuna(study: Any) -> pd.DataFrame:
    """DataFrame de trials de un ``optuna.Study`` con columnas ``number, value, state`` y un
    parametro por columna (sin el prefijo ``params_``)."""
    df = study.trials_dataframe()
    df = df.rename(columns={c: c.removeprefix("params_") for c in df.columns if c.startswith("params_")})
    df["state"] = df["state"].astype(str)
    return df


def plot_hpo_trials_vs_score(
    trials_df: pd.DataFrame, value_col: str = "value", state_col: str = "state", number_col: str = "number",
    maximize: bool = True, ylabel: str = "Score (objetivo HPO)", title: str = "HPO: trial vs score",
) -> Figure:
    """Score de cada trial (COMPLETE solido, PRUNED hueco) y mejor acumulado."""
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    if trials_df.empty:
        _empty(ax)
        return fig
    d = trials_df.sort_values(number_col)
    done = d[d[state_col].astype(str).str.contains("COMPLETE")]
    other = d[~d.index.isin(done.index)]
    ax.scatter(done[number_col], done[value_col], s=42, color=_CYCLE[0], label="COMPLETE", zorder=3)
    if len(other) and other[value_col].notna().any():
        ax.scatter(other[number_col], other[value_col], s=42, facecolors="none", edgecolors="#7F8C8D", label="PRUNED/otros", zorder=3)
    if len(done):
        best = done[value_col].cummax() if maximize else done[value_col].cummin()
        ax.step(done[number_col], best, where="post", color=_CYCLE[1], lw=2, label="mejor acumulado")
    ax.set_xlabel("N.º de trial")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig


def _is_numeric_param(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def plot_winning_hyperparameter_distribution(
    winners_df: pd.DataFrame, param_cols: Sequence[str] | None = None, group_col: str = "architecture", ncols: int = 3,
) -> Figure:
    """Distribucion de hiperparametros ganadores: numericos (puntos por grupo) y categoricos (conteos).

    Args:
        winners_df: Una fila por configuracion ganadora (p. ej. top-k por arquitectura).
        param_cols: Columnas de hiperparametros; por defecto todas salvo ``group_col``.
    """
    if winners_df.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        _empty(ax)
        return fig
    cols = list(param_cols) if param_cols else [c for c in winners_df.columns if c != group_col]
    ncols = min(ncols, len(cols))
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)
    groups = sorted(winners_df[group_col].astype(str).unique()) if group_col in winners_df.columns else ["all"]
    gcolor = {g: _CYCLE[i % len(_CYCLE)] for i, g in enumerate(groups)}
    for ax in axes.ravel():
        ax.axis("off")
    for ax, c in zip(axes.ravel(), cols):
        ax.axis("on")
        s = winners_df[c]
        if _is_numeric_param(s):
            rng = np.random.default_rng(0)
            for gi, g in enumerate(groups):
                v = winners_df.loc[winners_df[group_col].astype(str) == g, c].dropna() if group_col in winners_df.columns else s.dropna()
                ax.scatter(gi + rng.uniform(-0.12, 0.12, len(v)), v, color=gcolor[g], s=40, alpha=0.85)
            pos = s[s > 0]
            if len(pos) and pos.max() / max(pos.min(), 1e-30) >= 100:
                ax.set_yscale("log")
            ax.set_xticks(range(len(groups)), groups, rotation=30, ha="right")
        else:
            tab = winners_df.assign(_g=winners_df[group_col].astype(str) if group_col in winners_df.columns else "all").groupby([c, "_g"]).size().unstack(fill_value=0)
            bottom = np.zeros(len(tab))
            for g in groups:
                if g in tab.columns:
                    ax.bar(tab.index.astype(str), tab[g], bottom=bottom, color=gcolor[g], label=g)
                    bottom += tab[g].to_numpy()
            ax.set_ylabel("N.º de configuraciones")
        ax.set_title(c)
    handles = [Rectangle((0, 0), 1, 1, color=gcolor[g]) for g in groups]
    fig.legend(handles, groups, loc="lower center", ncol=min(len(groups), 5), bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Distribucion de hiperparametros ganadores", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig


def plot_baseline_vs_hpo(
    df: pd.DataFrame, label_col: str = "architecture", baseline_col: str = "baseline", hpo_col: str = "hpo_mean",
    hpo_sd_col: str | None = "hpo_sd", ylabel: str = "Score (objetivo)",
) -> Figure:
    """Baseline vs mejor configuracion HPO por arquitectura (media +- SD del HPO si existe)."""
    fig, ax = plt.subplots(figsize=(max(6.5, 1.3 * len(df) + 3), 4.4))
    if df.empty:
        _empty(ax)
        return fig
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2, df[baseline_col], w, color="#95A5A6", label="baseline")
    err = df[hpo_sd_col] if hpo_sd_col and hpo_sd_col in df.columns else None
    ax.bar(x + w / 2, df[hpo_col], w, yerr=err, capsize=3, color=_CYCLE[0], label="mejor HPO (media ± SD)" if err is not None else "mejor HPO")
    for xi, (b, h) in enumerate(zip(df[baseline_col], df[hpo_col])):
        if b:
            ax.annotate(f"{100 * (h - b) / abs(b):+.1f} %", (xi + w / 2, h), xytext=(0, 4 + (8 if err is not None else 0)), textcoords="offset points", ha="center", fontsize=8,
                        color="#1E8449" if h >= b else "#922B21")
    ax.set_xticks(x, df[label_col], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title("Baseline vs mejor configuracion HPO")
    ax.legend()
    return fig


def _bin_param(s: pd.Series, bins: int) -> tuple[pd.Series, list[str]]:
    """Discretiza un parametro (log si abarca >= 2 ordenes de magnitud). Devuelve (codigos, etiquetas)."""
    if _is_numeric_param(s) and s.nunique() > bins:
        pos = s[s > 0]
        use_log = len(pos) == len(s) and s.max() / s.min() >= 100
        vals = np.log10(s) if use_log else s
        edges = np.linspace(vals.min(), vals.max() + 1e-12, bins + 1)
        codes = pd.Series(np.clip(np.digitize(vals, edges[1:-1]), 0, bins - 1), index=s.index)
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(bins)]
        labels = [f"{10 ** c:.1e}" if use_log else f"{c:.3g}" for c in centers]
        return codes, labels
    cats = sorted(s.astype(str).unique())
    return s.astype(str).map({c: i for i, c in enumerate(cats)}), cats


def plot_hyperparameter_heatmap(
    trials_df: pd.DataFrame, x_param: str, y_param: str, value_col: str = "value", bins: int = 6,
    agg: str = "max", cbar_label: str = "Score",
) -> Figure:
    """Heatmap de una metrica en funcion de dos hiperparametros (mejor/media por celda).

    Numericos se discretizan en ``bins`` (escala log si aplica); categoricos usan sus valores.
    Cada celda anota ``n`` trials; las celdas sin trials quedan en blanco.
    """
    fig, ax = plt.subplots(figsize=(6.6, 5))
    if trials_df.empty or not {x_param, y_param, value_col} <= set(trials_df.columns):
        _empty(ax)
        return fig
    d = trials_df.dropna(subset=[x_param, y_param, value_col])
    if d.empty:
        _empty(ax)
        return fig
    xc, xl = _bin_param(d[x_param], bins)
    yc, yl = _bin_param(d[y_param], bins)
    grid = np.full((len(yl), len(xl)), np.nan)
    cnt = np.zeros_like(grid)
    tmp = pd.DataFrame({"x": xc, "y": yc, "v": d[value_col]})
    for (xi, yi), g in tmp.groupby(["x", "y"]):
        grid[int(yi), int(xi)] = g["v"].max() if agg == "max" else g["v"].mean()
        cnt[int(yi), int(xi)] = len(g)
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xl)), xl, rotation=30, ha="right")
    ax.set_yticks(range(len(yl)), yl)
    ax.grid(False)
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    for yi in range(grid.shape[0]):
        for xi in range(grid.shape[1]):
            if np.isfinite(grid[yi, xi]):
                dark = (grid[yi, xi] - vmin) / (vmax - vmin + 1e-12) < 0.55
                ax.text(xi, yi, f"{grid[yi, xi]:.3f}\n(n={int(cnt[yi, xi])})", ha="center", va="center", fontsize=7, color="white" if dark else "black")
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f"Heatmap {y_param} vs {x_param} ({'mejor' if agg == 'max' else 'media'} score)")
    fig.colorbar(im, ax=ax, label=cbar_label)
    return fig


def plot_parallel_coordinates(
    trials_df: pd.DataFrame, param_cols: Sequence[str], value_col: str = "value", maximize: bool = True,
    title: str = "Coordenadas paralelas de hiperparametros (HPO)",
) -> Figure:
    """Coordenadas paralelas estilo Optuna (matplotlib): una linea por trial, color = score.

    Numericos se normalizan (log si abarcan >= 2 ordenes); categoricos se codifican ordinalmente.
    """
    fig, ax = plt.subplots(figsize=(max(7, 1.7 * (len(param_cols) + 1)), 4.8))
    cols = [c for c in param_cols if c in trials_df.columns]
    if trials_df.empty or value_col not in trials_df.columns or not cols:
        _empty(ax)
        return fig
    d = trials_df.dropna(subset=[value_col]).copy()
    if d.empty:
        _empty(ax)
        return fig
    axes_def: list[tuple[str, np.ndarray, list[tuple[float, str]]]] = []
    for c in cols + [value_col]:
        s = d[c]
        if _is_numeric_param(s):
            pos = s > 0
            use_log = bool(pos.all()) and s.max() / s.min() >= 100
            v = np.log10(s.to_numpy(float)) if use_log else s.to_numpy(float)
            lo, hi = float(v.min()), float(v.max())
            norm = (v - lo) / (hi - lo) if hi > lo else np.full(len(v), 0.5)
            fmt = (lambda z: f"{10 ** z:.1e}") if use_log else (lambda z: f"{z:.3g}")
            ticks = [(0.0, fmt(lo)), (1.0, fmt(hi))]
        else:
            cats = sorted(s.astype(str).unique())
            codes = s.astype(str).map({k: i for i, k in enumerate(cats)}).to_numpy(float)
            norm = codes / max(len(cats) - 1, 1)
            ticks = [(i / max(len(cats) - 1, 1), k) for i, k in enumerate(cats)]
        axes_def.append((c, norm, ticks))
    cmap = matplotlib.colormaps["viridis" if maximize else "viridis_r"]
    vals = d[value_col].to_numpy(float)
    cn = mpl_colors.Normalize(vals.min(), vals.max() if vals.max() > vals.min() else vals.min() + 1)
    order = np.argsort(vals if maximize else -vals)
    xs = np.arange(len(axes_def))
    for i in order:
        ax.plot(xs, [a[1][i] for a in axes_def], color=cmap(cn(vals[i])), alpha=0.75, lw=1.6)
    for xi, (name, _, ticks) in zip(xs, axes_def):
        ax.axvline(xi, color="#2C3E50", lw=1.2)
        for yv, lab in ticks:
            ax.text(xi + 0.03, yv, lab, fontsize=7, va="center")
    ax.set_xticks(xs, [a[0] for a in axes_def], rotation=15)
    ax.set_yticks([])
    ax.set_ylim(-0.05, 1.05)
    ax.grid(False)
    fig.colorbar(mpl_cm.ScalarMappable(norm=cn, cmap=cmap), ax=ax, label=value_col)
    ax.set_title(title)
    return fig


# =========================================================================== #
#                                 COMPARACION                                 #
# =========================================================================== #
def pareto_front(df: pd.DataFrame, x_col: str, y_col: str, minimize_x: bool = True, maximize_y: bool = True) -> np.ndarray:
    """Mascara booleana de puntos no dominados (menor ``x`` y mayor ``y`` por defecto)."""
    x = df[x_col].to_numpy(float) * (1 if minimize_x else -1)
    y = df[y_col].to_numpy(float) * (1 if maximize_y else -1)
    mask = np.ones(len(df), bool)
    for i in range(len(df)):
        dominated = (x <= x[i]) & (y >= y[i]) & ((x < x[i]) | (y > y[i]))
        if dominated.any():
            mask[i] = False
    return mask


def plot_accuracy_latency_pareto(
    df: pd.DataFrame, latency_col: str = "latency_ms", metric_col: str = "metric", label_col: str = "architecture",
    group_col: str | None = None, metric_label: str = "Metrica", latency_label: str = "Latencia [ms]",
) -> Figure:
    """Dispersion precision-latencia con frontera de Pareto (menor latencia, mayor metrica)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if df.empty:
        _empty(ax)
        return fig
    d = df.reset_index(drop=True)
    groups = sorted(d[group_col].astype(str).unique()) if group_col else ["modelos"]
    for gi, g in enumerate(groups):
        sub = d if not group_col else d[d[group_col].astype(str) == g]
        ax.scatter(sub[latency_col], sub[metric_col], s=70, color=_CYCLE[gi % len(_CYCLE)], label=g, zorder=3)
    mask = pareto_front(d, latency_col, metric_col)
    front = d[mask].sort_values(latency_col)
    ax.step(front[latency_col], front[metric_col], where="post", color="k", ls="--", lw=1.3, label="frontera de Pareto", zorder=2)
    ax.scatter(front[latency_col], front[metric_col], s=160, facecolors="none", edgecolors="k", lw=1.5, zorder=4)
    for _, r in d.iterrows():
        ax.annotate(str(r[label_col]), (r[latency_col], r[metric_col]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel(latency_label)
    ax.set_ylabel(metric_label)
    ax.set_title("Precision vs latencia (frontera de Pareto)")
    ax.legend()
    return fig


def plot_performance_efficiency_comparison(
    df: pd.DataFrame, metric_col: str = "metric", label_col: str = "architecture", flops_col: str | None = "gflops",
    latency_col: str | None = "latency_ms", metric_label: str = "Metrica",
) -> Figure:
    """Rendimiento y eficiencia: metrica, metrica por GFLOP y metrica por ms (paneles disponibles)."""
    panels: list[tuple[str, pd.Series, str]] = []
    if not df.empty:
        panels.append((metric_label, df[metric_col], "rendimiento"))
        if flops_col and flops_col in df.columns:
            panels.append((f"{metric_label} / GFLOP", df[metric_col] / df[flops_col].replace(0, np.nan), "eficiencia computacional"))
        if latency_col and latency_col in df.columns:
            panels.append((f"{metric_label} / ms", df[metric_col] / df[latency_col].replace(0, np.nan), "eficiencia temporal"))
    if not panels:
        fig, ax = plt.subplots(figsize=(5, 4))
        _empty(ax)
        return fig
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 0.45 * len(df) + 2.6), squeeze=False)
    for ax, (xl, vals, ttl) in zip(axes[0], panels):
        order = vals.sort_values().index
        ax.barh(df.loc[order, label_col].astype(str), vals.loc[order], color=_CYCLE[0])
        for yi, v in enumerate(vals.loc[order]):
            if np.isfinite(v):
                ax.text(v, yi, f" {v:.3g}", va="center", fontsize=8)
        ax.set_xlabel(xl)
        ax.set_title(ttl.capitalize())
    fig.suptitle("Comparacion rendimiento vs eficiencia", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_params_flops_metric(
    df: pd.DataFrame, params_col: str = "params_m", flops_col: str = "gflops", metric_col: str = "metric",
    label_col: str = "architecture", metric_label: str = "Metrica",
) -> Figure:
    """Parametros (M) vs FLOPs (G) con el color/tamano codificando la metrica."""
    fig, ax = plt.subplots(figsize=(7, 5))
    if df.empty:
        _empty(ax)
        return fig
    m = df[metric_col].to_numpy(float)
    size = 60 + 400 * (m - m.min()) / (np.ptp(m) + 1e-12)
    sc = ax.scatter(df[params_col], df[flops_col], c=m, s=size, cmap="viridis", edgecolors="k", zorder=3)
    for _, r in df.iterrows():
        ax.annotate(str(r[label_col]), (r[params_col], r[flops_col]), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parametros [M]")
    ax.set_ylabel("FLOPs [G]")
    ax.set_title("Parametros vs FLOPs vs metrica")
    fig.colorbar(sc, ax=ax, label=metric_label)
    return fig


def plot_training_time_by_architecture(
    df: pd.DataFrame, label_col: str = "architecture", time_col: str = "train_minutes", err_col: str | None = None,
    group_col: str | None = "task",
) -> Figure:
    """Tiempo de entrenamiento por arquitectura (barras, coloreadas por tarea si ``group_col``)."""
    fig, ax = plt.subplots(figsize=(7, 0.45 * max(len(df), 3) + 1.8))
    if df.empty:
        _empty(ax)
        return fig
    d = df.sort_values(time_col).reset_index(drop=True)
    groups = sorted(d[group_col].astype(str).unique()) if group_col and group_col in d.columns else [None]
    colors = {g: _CYCLE[i % len(_CYCLE)] for i, g in enumerate(groups)}
    cs = [colors[str(r[group_col])] if group_col in d.columns and group_col else _CYCLE[0] for _, r in d.iterrows()]
    ax.barh(d[label_col].astype(str), d[time_col], xerr=d[err_col] if err_col and err_col in d.columns else None, color=cs, capsize=3)
    for yi, v in enumerate(d[time_col]):
        ax.text(v, yi, f" {v:.1f}", va="center", fontsize=8)
    if groups != [None]:
        ax.legend([Rectangle((0, 0), 1, 1, color=colors[g]) for g in groups], [str(g) for g in groups], loc="lower right")
    ax.set_xlabel("Tiempo de entrenamiento [min]")
    ax.set_title("Tiempo de entrenamiento por arquitectura")
    return fig


def plot_vram_usage_by_model(
    df: pd.DataFrame, label_col: str = "architecture", vram_col: str = "vram_mb", total_vram_mb: float | None = None,
) -> Figure:
    """Pico de VRAM por modelo con la capacidad total de la GPU como referencia (opcional)."""
    fig, ax = plt.subplots(figsize=(7, 0.45 * max(len(df), 3) + 1.8))
    if df.empty:
        _empty(ax)
        return fig
    d = df.sort_values(vram_col).reset_index(drop=True)
    ax.barh(d[label_col].astype(str), d[vram_col] / 1024, color=_CYCLE[2])
    for yi, v in enumerate(d[vram_col] / 1024):
        ax.text(v, yi, f" {v:.1f} GB", va="center", fontsize=8)
    if total_vram_mb:
        ax.axvline(total_vram_mb / 1024, color=POS_COLOR, ls="--", lw=1.4)
        ax.text(total_vram_mb / 1024, -0.6, f" VRAM GPU = {total_vram_mb / 1024:.0f} GB", color=POS_COLOR, fontsize=8, va="top", ha="right")
    ax.set_xlabel("VRAM pico [GB]")
    ax.set_title("Uso de VRAM por modelo")
    return fig


# =========================================================================== #
#                                  PIPELINE                                   #
# =========================================================================== #
def plot_hybrid_pipeline_diagram(detector_name: str | None = None, classifier_name: str | None = None) -> Figure:
    """Diagrama completo del pipeline hibrido: desarrollo independiente -> integracion -> test."""
    fig, ax = plt.subplots(figsize=(14, 8.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    ax.grid(False)
    det = detector_name or "detector seleccionado"
    cls = classifier_name or "clasificador seleccionado"

    _box(ax, (0.3, 6.9), 3.0, 0.9, "Imagenes RGB descubiertas\n(unidad experimental)", fc="#FDEBD0", bold=True)
    _box(ax, (4.2, 6.9), 3.6, 0.9, "Split externo unico por imagen\n70 % train / 20 % val / 10 % test (bloqueado)", fc="#FDEBD0")
    _arrow(ax, (3.3, 7.35), (4.2, 7.35))
    ax.text(0.3, 6.45, "DESARROLLO / HPO  (ramas independientes; HPO solo dentro del 70 % train, 3 folds internos)", fontsize=9, fontweight="bold", color="#555")

    _box(ax, (0.3, 4.9), 4.6, 1.3, "DETECCION\nimagen completa + cajas GT de fruto\n7 arquitecturas · 1 estudio HPO c/u (15 trials)", fc="#D5F5E3")
    _box(ax, (5.3, 4.9), 4.6, 1.3, "CLASIFICACION\nGT crops clase 0 + pairing clase 1\n3 arquitecturas · 1 estudio HPO c/u (20 trials)", fc="#D6EAF8")
    _arrow(ax, (6.0, 6.9), (2.6, 6.2))
    _arrow(ax, (6.0, 6.9), (7.6, 6.2))
    _box(ax, (0.3, 3.4), 4.6, 0.95, "Top-3 configs × 3 seeds (17, 42, 73) → media ± SD\nentrenar con 70 % completo → validation 20 %", fc="#E8F8F5", fs=8.5)
    _box(ax, (5.3, 3.4), 4.6, 0.95, "Top-3 configs × 3 seeds (17, 42, 73) → media ± SD\nentrenar con 70 % completo → validation 20 %", fc="#EBF5FB", fs=8.5)
    _arrow(ax, (2.6, 4.9), (2.6, 4.35))
    _arrow(ax, (7.6, 4.9), (7.6, 4.35))
    _box(ax, (0.3, 2.3), 4.6, 0.75, "Seleccion del detector (selection_policy.yaml)", fc="#ABEBC6", bold=True, fs=8.5)
    _box(ax, (5.3, 2.3), 4.6, 0.75, "Seleccion del clasificador (selection_policy.yaml)", fc="#AED6F1", bold=True, fs=8.5)
    _arrow(ax, (2.6, 3.4), (2.6, 3.05))
    _arrow(ax, (7.6, 3.4), (7.6, 3.05))

    ax.text(10.4, 7.75, "PIPELINE INTEGRADO (inferencia)", fontsize=9, fontweight="bold", color="#555")
    steps = [
        ("Imagen RGB completa", "#FDEBD0"), (det, "#D5F5E3"), ("Cajas predichas", "#F9E79F"),
        ("Crops desde la imagen original", "#F9E79F"), (cls, "#D6EAF8"), ("P(con_signos)  ≥  threshold", "#FCF3CF"),
        ("con_signos / sin_signos por fruto", "#F5B7B1"),
    ]
    y0, h, gap = 7.1, 0.62, 0.22
    for i, (txt, fc) in enumerate(steps):
        y = y0 - i * (h + gap)
        _box(ax, (10.3, y - h), 3.4, h, txt, fc=fc, fs=8.5, bold=i in (1, 4))
        if i:
            _arrow(ax, (12.0, y + gap), (12.0, y))
    _arrow(ax, (4.9, 2.65), (10.3, 5.85), rad=-0.15)
    _arrow(ax, (9.9, 2.65), (10.3, 3.95), rad=0.15)

    _box(ax, (0.3, 0.9), 9.6, 0.85, "Ajuste de inferencia SOLO en validation externo: detector conf/NMS · clasificador threshold (independiente del pairing GT)\n"
                                   "Validacion del pipeline con PREDICTED crops en validation (matching detector-GT)", fc="#FDFEFE", fs=8.5)
    _box(ax, (0.3, 0.05), 4.6, 0.6, "freeze_manifest.json (hashes de modelos, datos, codigo)", fc="#D7BDE2", fs=8.5, bold=True)
    _box(ax, (5.3, 0.05), 4.6, 0.6, "Test 10 %: evaluacion final UNICA", fc="#D7BDE2", fs=8.5, bold=True)
    _arrow(ax, (4.9, 0.35), (5.3, 0.35))
    _arrow(ax, (5.1, 2.3), (5.1, 1.75))
    ax.set_title("Pipeline hibrido: deteccion de frutos + clasificacion de signos compatibles con Botrytis", fontweight="bold")
    return fig


def plot_latency_breakdown(
    df: pd.DataFrame, group_col: str | None = None,
    components: Sequence[str] = ("t_detector_ms", "t_crop_preprocess_ms", "t_classifier_ms"), total_col: str = "t_total_ms",
    n_col: str = "n_detections",
) -> Figure:
    """Latencia del pipeline descompuesta (detector + crop/preproceso + clasificador).

    Args:
        df: Una fila por imagen (formato sec. 29.2): ``image_id, n_detections, t_*_ms, t_total_ms``.
        group_col: Columna para comparar varias configuraciones de pipeline.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.1, 1]})
    if df.empty:
        for a in axes:
            _empty(a)
        return fig
    labels = {"t_detector_ms": "detector", "t_crop_preprocess_ms": "crop + preproceso", "t_classifier_ms": "clasificador"}
    groups = sorted(df[group_col].astype(str).unique()) if group_col else ["pipeline"]
    means = pd.DataFrame({g: (df if not group_col else df[df[group_col].astype(str) == g])[list(components)].mean() for g in groups}).T
    bottom = np.zeros(len(means))
    for i, c in enumerate(components):
        axes[0].bar(means.index, means[c], bottom=bottom, color=_CYCLE[i], label=labels.get(c, c))
        for xi, (v, b) in enumerate(zip(means[c], bottom)):
            if v / max(means.sum(axis=1).max(), 1e-9) > 0.06:
                axes[0].text(xi, b + v / 2, f"{v:.1f}", ha="center", va="center", color="white", fontsize=8)
        bottom += means[c].to_numpy()
    for xi, tot in enumerate(bottom):
        axes[0].text(xi, tot, f"{tot:.1f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Latencia media por imagen [ms]")
    axes[0].set_title("Latencia descompuesta")
    axes[0].legend()
    if n_col in df.columns and total_col in df.columns:
        for gi, g in enumerate(groups):
            sub = df if not group_col else df[df[group_col].astype(str) == g]
            axes[1].scatter(sub[n_col], sub[total_col], s=22, alpha=0.6, color=_CYCLE[gi % len(_CYCLE)], label=g)
            if sub[n_col].nunique() > 1:
                k, b = np.polyfit(sub[n_col], sub[total_col], 1)
                xs = np.linspace(sub[n_col].min(), sub[n_col].max(), 20)
                axes[1].plot(xs, k * xs + b, color=_CYCLE[gi % len(_CYCLE)], lw=1.5)
        axes[1].set_xlabel("N.º de detecciones (crops) por imagen")
        axes[1].set_ylabel("Latencia total [ms]")
        axes[1].set_title("Latencia total vs numero de crops")
        if len(groups) > 1:
            axes[1].legend()
    else:
        _empty(axes[1])
    fig.suptitle("Latencia del pipeline: T = T_detector + T_crop/preproceso + T_clasificador", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_pipeline_error_matrix(
    cm: np.ndarray, row_labels: Sequence[str] = ("con_signos", "sin_signos", "fondo (sin fruto GT)"),
    col_labels: Sequence[str] = ("pred con_signos", "pred sin_signos", "sin deteccion"), normalize_rows: bool = False,
) -> Figure:
    """Matriz de errores del pipeline (ver ``metrics.pipeline_confusion_matrix``)."""
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    cm = np.asarray(cm, dtype=float)
    if cm.size == 0:
        _empty(ax)
        return fig
    show = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize_rows else cm
    im = ax.imshow(show, cmap="Blues")
    ax.grid(False)
    ax.set_xticks(range(cm.shape[1]), col_labels, rotation=20, ha="right")
    ax.set_yticks(range(cm.shape[0]), row_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            frac = cm[i, j] / cm[i].sum() if cm[i].sum() else 0
            txt = f"{int(cm[i, j])}\n({100 * frac:.0f} %)"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="white" if show[i, j] > show.max() * 0.55 else "black")
    ax.set_xlabel("Salida del pipeline")
    ax.set_ylabel("Ground truth")
    ax.set_title("Matriz de errores del pipeline (detector + clasificador)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return fig


def plot_tp_fp_fn_examples(samples: Sequence[Mapping[str, Any]], ncols: int = 3, max_side_px: int = 900) -> Figure:
    """Ejemplos cualitativos TP/FP/FN.

    Cada muestra: ``{"image_path", "title", "tp": [box], "fp": [box], "fn": [box]}`` con cajas
    ``(x1, y1, x2, y2)`` en pixeles de la imagen orientada. TP verde, FP rojo, FN azul discontinuo.
    """
    n = len(samples)
    if n == 0:
        fig, ax = plt.subplots(figsize=(5, 4))
        _empty(ax)
        return fig
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, s in zip(axes.ravel(), samples):
        img = load_image_rgb(s["image_path"])
        scale = min(1.0, max_side_px / max(img.size))
        if scale < 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        ax.imshow(img)
        for key, col, ls in (("tp", TP_COLOR, "-"), ("fp", FP_COLOR, "-"), ("fn", FN_COLOR, "--")):
            for (x1, y1, x2, y2) in s.get(key, []):
                ax.add_patch(Rectangle((x1 * scale, y1 * scale), (x2 - x1) * scale, (y2 - y1) * scale, fill=False, ec=col, lw=2, ls=ls))
        ax.set_title(f"{s.get('title', '')}  [TP={len(s.get('tp', []))} FP={len(s.get('fp', []))} FN={len(s.get('fn', []))}]", fontsize=9)
    fig.legend([Rectangle((0, 0), 1, 1, fill=False, ec=TP_COLOR, lw=2), Rectangle((0, 0), 1, 1, fill=False, ec=FP_COLOR, lw=2),
                Rectangle((0, 0), 1, 1, fill=False, ec=FN_COLOR, lw=2, ls="--")],
               ["TP (deteccion correcta)", "FP (deteccion sin GT valido / duplicada)", "FN (fruto GT no detectado)"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Ejemplos TP / FP / FN del pipeline", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


def plot_spatial_error_map(
    errors_df: pd.DataFrame, x_col: str = "x_norm", y_col: str = "y_norm", type_col: str = "error_type",
    types: Sequence[str] = ("FP", "FN", "EC"), bins: int = 12,
) -> Figure:
    """Mapa espacial de errores sobre un lienzo de imagen normalizado ``[0,1]^2``.

    Args:
        errors_df: Una fila por error con el centro normalizado de la caja y su tipo
            (``FP`` localizacion, ``FN`` fruto no detectado, ``EC`` error de clasificacion).
    """
    names = {"FP": "FP (localizacion)", "FN": "FN (no detectado)", "EC": "EC (error de clasificacion)", "TP": "TP"}
    present = [t for t in types if not errors_df.empty and (errors_df[type_col] == t).any()] or list(types)
    fig, axes = plt.subplots(1, len(present), figsize=(4.6 * len(present), 4.2), squeeze=False)
    for ax, t in zip(axes[0], present):
        sub = errors_df[errors_df[type_col] == t] if not errors_df.empty else errors_df
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        ax.set_title(f"{names.get(t, t)}  (n={len(sub)})")
        if len(sub) == 0:
            ax.text(0.5, 0.5, "sin errores", ha="center", va="center", color="#7F8C8D")
            continue
        h, xe, ye = np.histogram2d(sub[x_col], sub[y_col], bins=bins, range=[[0, 1], [0, 1]])
        ax.imshow(h.T, extent=(0, 1, 1, 0), cmap="Reds", alpha=0.75, aspect="equal")
        ax.scatter(sub[x_col], sub[y_col], s=12, color="k", alpha=0.6)
        ax.grid(False)
        ax.set_xlabel("x / ancho")
        ax.set_ylabel("y / alto")
    fig.suptitle("Mapa espacial de errores del pipeline", fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Conjuntos de figuras de dataset (usados por los scripts 01/05)
# --------------------------------------------------------------------------- #
def dataset_figure_catalog() -> list[str]:
    """Nombres de fichero de las figuras de dataset que generan los scripts."""
    return [
        "dataset_distribucion_etiquetas", "dataset_balance_clases_por_split", "dataset_frutos_por_imagen",
        "dataset_tamanos_bbox", "dataset_cobertura_signos", "dataset_crops_por_imagen",
        "dataset_ejemplos_gt", "dataset_flujo_pairing",
    ]
