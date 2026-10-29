"""Classification GT from one-to-one class 0 ↔ class 1 annotation pairing."""
from __future__ import annotations
from typing import Any, Mapping
import pandas as pd
from utils.boxes import max_sign_coverage
from utils.pairing import pair_image

FRUIT_TABLE_COLUMNS = ["image_id", "fruit_id", "x1", "y1", "x2", "y2", "image_w", "image_h",
    "n_sign_boxes", "max_sign_coverage", "best_sign_ordinal", "suspect_ordinal", "pairing_iou", "pairing_status", "label"]


def build_fruit_table(annotations: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Only class 0 boxes become crops. Ambiguous images retain NA labels.

    Coverage is descriptive only. Audit rows are attached for export before abort.
    """
    rows, pairs, candidates, anomalies = [], [], [], []
    for iid, group in annotations.groupby("image_id", sort=True):
        decisions, candidate_rows, issues = pair_image(group.to_dict("records"), float(cfg["pairing"]["min_iou"]))
        pairs.extend(dict(image_id=iid, **r) for r in decisions)
        candidates.extend(dict(image_id=iid, **r) for r in candidate_rows)
        anomalies.extend(dict(image_id=iid, **r) for r in issues)
        assigned = {r["fruit_ordinal"]:r for r in decisions if r["status"] == "valid"}
        signs = group[group["class_id"] == 2].sort_values("ordinal")
        sign_boxes = signs[["x1","y1","x2","y2"]].to_numpy(float)
        for f in group[group["class_id"] == 0].sort_values("ordinal").itertuples(index=False):
            cov, idx = max_sign_coverage((f.x1,f.y1,f.x2,f.y2),sign_boxes)
            pair = assigned.get(f.ordinal)
            label = pd.NA if issues else cfg["classification_labels"]["positive_label" if pair else "negative_label"]
            rows.append(dict(image_id=str(iid),fruit_id=int(f.ordinal),x1=f.x1,y1=f.y1,x2=f.x2,y2=f.y2,
                image_w=int(f.image_w),image_h=int(f.image_h),n_sign_boxes=len(signs),max_sign_coverage=cov,
                best_sign_ordinal=int(signs.iloc[idx]["ordinal"]) if idx >= 0 else -1,
                suspect_ordinal=pair["suspect_ordinal"] if pair else -1,
                pairing_iou=pair["iou"] if pair else None,
                pairing_status="unresolved_image" if issues else "matched" if pair else "unmatched_fruit",label=label))
    table = pd.DataFrame(rows,columns=FRUIT_TABLE_COLUMNS).sort_values(["image_id","fruit_id"],kind="stable").reset_index(drop=True)
    table.attrs.update(pairing_audit=pairs,pairing_candidates=candidates,pairing_anomalies=anomalies)
    return table


def image_summary(fruit_table: pd.DataFrame, images: pd.DataFrame, annotations: pd.DataFrame, cfg: Mapping[str,Any]) -> pd.DataFrame:
    """Per-original-image statistics used before splitting; never split crops."""
    pos, neg = cfg["classification_labels"]["positive_label"], cfg["classification_labels"]["negative_label"]
    out = images[["image_id","image_w","image_h"]].copy()
    groups = {k:g for k,g in fruit_table.groupby("image_id")}
    signs = annotations[annotations["class_id"]==2].groupby("image_id").size()
    out["n_fruits"] = [len(groups.get(i,[])) for i in out["image_id"]]
    out["n_positive"] = [int((groups[i]["label"]==pos).sum()) if i in groups else 0 for i in out["image_id"]]
    out["n_negative"] = [int((groups[i]["label"]==neg).sum()) if i in groups else 0 for i in out["image_id"]]
    out["n_unresolved"] = out["n_fruits"]-out["n_positive"]-out["n_negative"]
    out["n_signs"] = out["image_id"].map(signs).fillna(0).astype(int)
    out["pct_positive"] = (100*out["n_positive"]/out["n_fruits"].where(out["n_fruits"]>0)).fillna(0.)
    out["has_positive_signs"] = out["n_positive"]>0
    return out.sort_values("image_id").reset_index(drop=True)
