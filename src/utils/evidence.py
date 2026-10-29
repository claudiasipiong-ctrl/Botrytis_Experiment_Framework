"""Complete split-local evidence from actual predictions, independent of architecture."""
from __future__ import annotations
import re
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from utils.boxes import crop_window, iou_matrix
from utils.checks import MethodologyViolation
from utils.config_loader import cfg_path
from utils.hashing import sha256_file, sha256_text
from utils.io import load_image_rgb, write_csv, write_json
from utils.metrics import binary_classification_report, match_detections, precision_recall_curve_np, roc_curve_np
from utils.metrics_detection import ap_over_iou_table, conf_sweep
from utils.splits import assert_split_allowed


def result_root(cfg):
    return cfg_path(cfg, "paths.results_dir")


def split_dir(cfg, spec, split):
    return result_root(cfg) / spec.task / spec.slug / split


def token(value):
    raw = str(value)
    return re.sub(r"[^A-Za-z0-9_-]", "_", raw)[:80] + "_" + sha256_text(raw)[:10]


def font(size=18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def annotated(image, lines):
    # Metadata is outside the input crop; the crop itself is not modified.
    f = font()
    widths = [int(f.getlength(str(line))) + 24 for line in lines]
    out = Image.new("RGB", (max([image.width, *widths]), image.height + 28 * len(lines) + 12), "#142032")
    out.paste(image, (0, 28 * len(lines) + 12))
    d = ImageDraw.Draw(out)
    for i, line in enumerate(lines):
        d.text((10, 6 + i * 28), str(line), font=f, fill="white")
    return out


def begin(cfg, spec, split, manifest):
    stage = "final_test" if split == "test" else "outer_validation"
    assert_split_allowed(split, stage, cfg)
    out = split_dir(cfg, spec, split)
    # Each attempt replaces its own complete evidence; other models/splits are untouched.
    if out.exists():
        from datetime import datetime,timezone
        archive=out.with_name(out.name+"_obsolete_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        out.rename(archive)
    for name in ("visualizations", "crops", "metrics", "plots"):
        (out / name).mkdir(parents=True, exist_ok=True)
    ids = sorted(manifest.loc[manifest.outer_split == split, "image_id"].astype(str))
    write_csv(pd.DataFrame({"image_id": ids, "split": split}), out / "metrics" / "split_images.csv")
    return out, ids


def save_chart(out, name, x, series, xlabel, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for label, values in series.items():
        ax.plot(x, values, label=label)
    ax.set(xlabel=xlabel, title=title)
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(out / "plots" / f"{name}.png", dpi=160)
    plt.close(fig)


def detection_evidence(cfg, spec, split, manifest, paths, gt, predictions):
    expected = sorted(manifest.loc[manifest.outer_split == split, "image_id"].astype(str))
    if sorted(paths) != expected or sorted(gt) != expected or len(predictions) != len(expected):
        raise MethodologyViolation("Evidencia detector no cubre exactamente su split")
    out, ids = begin(cfg, spec, split, manifest)
    rows, image_rows, misses = [], [], []
    for iid, (boxes, scores) in zip(ids, predictions):
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        scores = np.asarray(scores, float).reshape(-1)
        if len(boxes) != len(scores) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise MethodologyViolation("Predicciones invalidas")
        gb = np.asarray(gt[iid]["boxes"], float).reshape(-1, 4)
        mat = iou_matrix(boxes, gb)
        match = match_detections(boxes, scores, gb, .5)
        pairs = {pi: (gi, value) for pi, gi, value in match["matches"]}
        im = load_image_rgb(paths[iid]); overlay = im.copy(); draw = ImageDraw.Draw(overlay)
        f = font(max(12, min(30, im.width // 45)))
        for gi, b in enumerate(gb):
            draw.rectangle(tuple(b), outline="#28e36b", width=6)
            draw.text((b[0], b[1]), f"GT{gi}", font=f, fill="#28e36b", stroke_width=1, stroke_fill="black")
        for pi, (b, score) in enumerate(zip(boxes, scores)):
            matched = pi in pairs
            gi, overlap = pairs[pi] if matched else ((int(mat[pi].argmax()), float(mat[pi].max())) if len(gb) else (-1, 0.0))
            label = f"P{pi} {'IoU' if matched else 'best IoU'}={overlap*100:.1f}% {'TP' if matched else 'FP'}"
            draw.rectangle(tuple(b), outline="#ff972e", width=2)
            draw.text((b[0], max(0, b[3] - 22)), label, font=f, fill="#ff972e", stroke_width=1, stroke_fill="black")
            rel = f"crops/{spec.slug}__{split}__{token(iid)}__p{pi:04d}.png"
            win = crop_window(tuple(b), im.width, im.height)
            im.crop(win).save(out / rel)
            rows.append({"image_id": iid, "original_image": str(paths[iid]), "model": spec.name, "split": split, "prediction_id": pi,
                         "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "confidence": score, "gt_index": gi,
                         "iou_percent": overlap*100, "iou_role": "one_to_one_match" if matched else "best_overlap_only",
                         "outcome_iou50": "TP" if matched else "FP", "crop_path": rel,
                         "crop_window": str(win), "crop_sha256": sha256_file(out / rel)})
        for gi in match["fn"]:
            misses.append({"image_id": iid, "model": spec.name, "split": split, "gt_index": gi, "outcome": "FN"})
        vis = f"visualizations/{spec.slug}__{split}__{token(iid)}.png"
        annotated(overlay, [f"{spec.name} | {split} | {iid}", "GREEN: GT arandano_a | ORANGE: predictions | TP/FP: IoU >=50%, one-to-one",
                            "FP IoU: best overlap only; not a valid match"]).save(out / vis)
        image_rows.append({"image_id": iid, "model": spec.name, "split": split, "visualization": vis, "n_gt": len(gb), "n_predictions": len(boxes),
                           "tp": len(pairs), "fp": len(match["fp"]), "fn": len(match["fn"]), "source_sha256": sha256_file(paths[iid])})
    columns = ["image_id","original_image","model","split","prediction_id","x1","y1","x2","y2","confidence","gt_index","iou_percent","iou_role","outcome_iou50","crop_path","crop_window","crop_sha256"]
    write_csv(pd.DataFrame(rows, columns=columns), out / "metrics" / "predictions.csv")
    write_csv(pd.DataFrame(image_rows), out / "metrics" / "images.csv")
    write_csv(pd.DataFrame(misses, columns=["image_id","model","split","gt_index","outcome"]), out / "metrics" / "misses.csv")
    gts = [gt[i]["boxes"] for i in ids]
    ap = ap_over_iou_table(predictions, gts)
    sweep = conf_sweep(predictions, gts, np.linspace(0, 1, 41), .5)
    write_csv(ap, out / "metrics" / "ap_per_iou.csv")
    write_csv(sweep, out / "metrics" / "confidence_curves.csv")
    save_chart(out, "ap_vs_iou", ap.iou, {"AP": ap.ap}, "IoU", f"{spec.name} | {split}")
    save_chart(out, "precision_recall", sweep.recall, {"Precision": sweep.precision}, "Recall", f"{spec.name} | {split} | IoU=0.50")
    save_chart(out, "confidence_curves", sweep.conf, {k: sweep[k] for k in ("precision", "recall", "f1")}, "Confidence (descriptive only)", f"{spec.name} | {split}")
    return {"root": str(out), "n_images": len(ids), "n_gt": sum(len(g) for g in gts), "n_crops": len(rows), "split_ids_sha256": sha256_text("\n".join(ids))}


def classification_evidence(cfg, spec, split, manifest, crop_manifest, predictions, root, threshold=.5):
    expected = crop_manifest[crop_manifest.outer_split == split]
    keys = ["image_id", "fruit_id", "crop_path"]
    if expected.duplicated(keys).any() or predictions.duplicated(keys).any() or set(map(tuple, expected[keys].values)) != set(map(tuple, predictions[keys].values)):
        raise MethodologyViolation("Evidencia clasificador no cubre exactamente los crops evaluados")
    if "label" in expected:
        joined = predictions.merge(expected[keys + ["label"]], on=keys, validate="one_to_one")
        expected_y = (joined.label == "con_signos").astype(int)
        if not expected_y.eq(joined.y_true).all():
            raise MethodologyViolation("Etiqueta real distinta a la del crop utilizado")
    out, ids = begin(cfg, spec, split, manifest)
    if not set(predictions.image_id).issubset(set(ids)):
        raise MethodologyViolation("Crop de otro split en evidencia")
    labels = ["sin_signos", "con_signos"]
    rows = []
    for r in predictions.itertuples(index=False):
        p = float(r.p_con_signos)
        if not np.isfinite(p) or not 0 <= p <= 1 or int(r.y_true) not in (0, 1):
            raise MethodologyViolation("Prediccion de clasificacion invalida")
        yp = int(p >= threshold); correct = yp == int(r.y_true)
        basename = f"{spec.slug}__{split}__{token(r.image_id)}__f{int(r.fruit_id):04d}"
        src = Path(root) / r.crop_path
        rel = "crops/" + basename + src.suffix
        shutil.copy2(src, out / rel)
        vis = "visualizations/" + basename + ".png"
        confidence = p if yp else 1-p
        annotated(load_image_rgb(src), [f"{spec.name} | {split} | {r.image_id} | fruit {r.fruit_id}",
                  f"TRUE: {labels[int(r.y_true)]} | PRED: {labels[yp]} | {'CORRECT' if correct else 'INCORRECT'}",
                  f"P(con_signos)={p:.6f} | predicted-class probability={confidence:.6f} | threshold={threshold}"]).save(out / vis)
        rows.append({"image_id": r.image_id, "fruit_id": r.fruit_id, "model": spec.name, "split": split, "source_crop": r.crop_path,
                     "y_true": int(r.y_true), "true_label": labels[int(r.y_true)], "y_pred": yp, "predicted_label": labels[yp], "p_con_signos": p,
                     "confidence": confidence, "correct": correct, "threshold": threshold, "crop_path": rel, "visualization": vis, "crop_sha256": sha256_file(src)})
    cols = ["image_id","fruit_id","model","split","source_crop","y_true","true_label","y_pred","predicted_label","p_con_signos","confidence","correct","threshold","crop_path","visualization","crop_sha256"]
    write_csv(pd.DataFrame(rows, columns=cols), out / "metrics" / "predictions.csv")
    yt, prob = predictions.y_true.to_numpy(), predictions.p_con_signos.to_numpy()
    rep = binary_classification_report(yt, prob, threshold=threshold)
    pc, rc, _ = precision_recall_curve_np(yt, prob); fpr, tpr, _ = roc_curve_np(yt, prob)
    write_csv(pd.DataFrame({"precision": pc, "recall": rc}), out / "metrics" / "pr_curve.csv")
    write_csv(pd.DataFrame({"fpr": fpr, "tpr": tpr}), out / "metrics" / "roc_curve.csv")
    write_csv(pd.DataFrame(rep["confusion_matrix"], columns=["pred_sin", "pred_con"]), out / "metrics" / "confusion_matrix.csv")
    save_chart(out, "pr_curve", rc, {"Precision": pc}, "Recall", f"{spec.name} | {split}")
    save_chart(out, "roc_curve", fpr, {"TPR": tpr}, "FPR", f"{spec.name} | {split}")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4,4)); ax.imshow(rep["confusion_matrix"], cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(rep["confusion_matrix"][i][j]),ha="center",va="center")
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=labels,yticklabels=labels,xlabel="Predicted",ylabel="True",title=f"{spec.name} | {split}")
    fig.tight_layout(); fig.savefig(out / "plots" / "confusion_matrix.png", dpi=160); plt.close(fig)
    return {"root": str(out), "n_images": int(predictions.image_id.nunique()), "n_crops": len(predictions),
            "split_ids_sha256": sha256_text("\n".join(ids)), "crop_keys_sha256": sha256_text("\n".join(sorted(expected.crop_path)))}


def finish_evidence(payload):
    out = Path(payload["evidence"]["root"])
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics" / "metrics.json", payload)
    # No epoch curves can be claimed for a one-time held-out evaluation.
    write_json(out / "metrics" / "losses.json", {"loss": payload.get("loss"), "loss_components": payload.get("loss_components", {}),
               "status": "measured" if payload.get("loss") is not None else "not_exposed_by_backend",
               "epoch_curves": "not_applicable_single_checkpoint_evaluation"})
    artifacts = [{"path": p.relative_to(out).as_posix(), "sha256": sha256_file(p), "bytes": p.stat().st_size}
                 for p in sorted(out.rglob("*")) if p.is_file() and p.name != "artifact_index.json"]
    write_json(out / "artifact_index.json", {"model": payload["architecture"], "split": payload["split"], "artifacts": artifacts})


def training_outputs(cfg, spec, history, run_dir, prefix):
    if history is None or history.empty:
        return
    base = split_dir(cfg, spec, "train")
    internal = "fold" in history.columns
    # Folds are within outer train: never label them as outer validation.
    dst = base / "internal" / token(str(run_dir)) if internal else base
    common = [c for c in ("fold", "epoch") if c in history]
    for role in ("train", "inner_validation"):
        candidates = [c for c in history if c.startswith("train/") or c == "train_loss"] if role == "train" else [c for c in history if c.startswith(("val/", "val_", "metrics/")) or c in ("precision", "recall", "f1", "map50", "map50_95")]
        cols = [c for c in candidates if history[c].notna().any()]
        if not cols:
            continue
        out = dst / role if internal else dst
        (out / "metrics").mkdir(parents=True, exist_ok=True); (out / "plots").mkdir(exist_ok=True)
        sub = history[common + cols].copy()
        write_csv(sub, out / "metrics" / "history.csv")
        write_json(out / "metrics" / "provenance.json", {"model":spec.name,"outer_split":"train", "role":role,"source_run":str(run_dir),"prefix":prefix})
        for group, tab in sub.groupby("fold") if internal else [(None, sub)]:
            for col in cols:
                save_chart(out, token(f"fold{group}_{col}"), tab.epoch, {col: tab[col]}, "Epoch", f"{spec.name} | outer train | {role} | fold={group}")
