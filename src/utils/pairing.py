"""Deterministic annotation pairing; class 2 never supplies a class label.

Maximum total IoU is solved over associations meeting the technical threshold.
The Hungarian objective uses exact integer encodings of float IoUs and a
lexicographic ordinal secondary objective; no epsilon can change the optimum.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

COORDS = ("x1", "y1", "x2", "y2")
SEMANTICS_VERSION = "botrytis-physical-fruit-pairing-v4"


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two xyxy boxes, returning zero for degenerate boxes."""
    if not all(math.isfinite(v) for v in (*a,*b)):
        return 0.
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    ab = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return inter / (aa + ab - inter) if aa + ab - inter > 0 else 0.


def _hungarian(cost: list[list[int]]) -> list[int]:
    """Rectangular minimum assignment, rows <= columns; integer arithmetic."""
    n, m = len(cost), len(cost[0]) if cost else 0
    u, v, p, way = [0]*(n+1), [0]*(m+1), [0]*(m+1), [0]*(m+1)
    for i in range(1, n+1):
        p[0], j0 = i, 0
        mins, used = [None]*(m+1), [False]*(m+1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], None, 0
            for j in range(1, m+1):
                if not used[j]:
                    cur = cost[i0-1][j-1] - u[i0] - v[j]
                    if mins[j] is None or cur < mins[j]:
                        mins[j], way[j] = cur, j0
                    if delta is None or mins[j] < delta:
                        delta, j1 = mins[j], j
            for j in range(m+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                elif mins[j] is not None:
                    mins[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0], j0 = p[j1], j1
    result = [-1]*n
    for j in range(1, m+1):
        if p[j]:
            result[p[j]-1] = j-1
    return result


def maximum_assignment(weights: list[list[float]], forbidden: tuple[int,int] | None = None) -> tuple[list[int],int]:
    """Maximum weight, then lexicographically smallest column sequence by row."""
    if not weights:
        return [], 0
    n, m = len(weights), len(weights[0])
    ratios = [[float(w).as_integer_ratio() for w in row] for row in weights]
    scale = max((d for row in ratios for _, d in row), default=1)
    ints = [[a*(scale//b) for a, b in row] + [0]*n for row in ratios]
    base, dominance = m+n+1, (m+n+1)**n
    costs = [[-w*dominance + j*base**(n-1-i) for j, w in enumerate(row)] for i, row in enumerate(ints)]
    # Zero overlap is absence of a candidate; dummy assignments retain the same
    # maximum IoU without misleadingly naming an unrelated physical fruit.
    for i, row in enumerate(ints):
        for j in range(m):
            if row[j] == 0:
                costs[i][j] = (n+1)*scale*dominance
    if forbidden is not None:
        i, j = forbidden
        costs[i][j] = (n+1)*scale*dominance
    assignment = _hungarian(costs)
    return assignment, sum(ints[i][j] for i, j in enumerate(assignment))


def pair_image(records: list[dict[str,Any]], min_iou: float = 0.50) -> tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    """Returns pair decisions, every candidate IoU and image-level anomalies.

    Ordinals are zero based within each class. An anomalous image cannot supply
    negative labels: callers must abort or explicitly quarantine the entire image.
    """
    if not math.isfinite(min_iou) or not 0 < min_iou <= 1:
        raise ValueError("min_iou must be in (0,1]")
    records = sorted(records, key=lambda r: (r["class_id"], r["ordinal"], *(r[k] for k in COORDS)))
    fruits = sorted((r for r in records if r["class_id"] == 0), key=lambda r: r["ordinal"])
    suspects = sorted((r for r in records if r["class_id"] == 1), key=lambda r: r["ordinal"])
    issues = []
    seen = set()
    seen_ordinals = set()
    for r in records:
        box = tuple(r[k] for k in COORDS)
        key = (r["class_id"], box)
        reason = None
        if r["class_id"] not in (0, 1, 2):
            reason = "unknown_class"
        elif not all(math.isfinite(x) for x in box) or box[2] <= box[0] or box[3] <= box[1]:
            reason = "degenerate_box"
        elif key in seen:
            reason = "duplicate_box"
        seen.add(key)
        if reason:
            issues.append({"class_id": r["class_id"], "ordinal": r["ordinal"], "reason": reason})
        ordinal_key = (r["class_id"], r["ordinal"])
        if ordinal_key in seen_ordinals:
            issues.append({"class_id": r["class_id"], "ordinal": r["ordinal"], "reason": "duplicate_ordinal"})
        seen_ordinals.add(ordinal_key)
    weights = [[box_iou(tuple(s[k] for k in COORDS), tuple(f[k] for k in COORDS)) for f in fruits] for s in suspects]
    admissible = [[w if w >= min_iou else 0. for w in row] for row in weights]
    assignment, total = maximum_assignment(admissible)
    candidates = []
    claims = Counter(j for row in weights for j, w in enumerate(row) if w >= min_iou)
    decisions = []
    for i, s in enumerate(suspects):
        j = assignment[i]
        selected = j < len(fruits)
        value = weights[i][j] if selected else 0.
        eligible = [k for k, w in enumerate(weights[i]) if w >= min_iou]
        reasons = []
        if not fruits or max(weights[i], default=0.) == 0:
            reasons.append("no_candidate")
        elif max(weights[i], default=0.) < min_iou:
            reasons.append("insufficient_iou")
        elif not selected:
            reasons.append("unassigned_competing_suspect")
        elif value < min_iou:
            reasons.append("insufficient_iou_after_assignment")
        if len(eligible) > 1:
            reasons.append("multiple_eligible_candidates")
        if any(claims[k] > 1 for k in eligible):
            reasons.append("multiple_suspects_claim_fruit")
        if selected and value > 0 and maximum_assignment(admissible, (i, j))[1] == total:
            reasons.append("global_assignment_tie")
        if reasons:
            issues.append({"class_id": 1, "ordinal": s["ordinal"], "reason": ";".join(reasons)})
        decision = {"suspect_ordinal": s["ordinal"], "suspect_line": s.get("line_no"),
                    "suspect_box": [s[k] for k in COORDS],
                    "fruit_ordinal": fruits[j]["ordinal"] if selected else None,
                    "fruit_box": [fruits[j][k] for k in COORDS] if selected else None,
                    "iou": value, "status": "anomaly" if reasons else "valid",
                    "reason": ";".join(reasons) or "assigned_maximum_total_iou"}
        best = max(range(len(fruits)), key=lambda k: weights[i][k], default=None)
        decision.update(best_fruit_ordinal=fruits[best]["ordinal"] if best is not None and weights[i][best]>0 else None,
                        best_iou=weights[i][best] if best is not None else 0.)
        decisions.append(decision)
        for k, f in enumerate(fruits):
            candidates.append({"suspect_ordinal": s["ordinal"], "fruit_ordinal": f["ordinal"],
                               "suspect_box": [s[c] for c in COORDS], "fruit_box": [f[c] for c in COORDS],
                               "iou": weights[i][k], "selected": selected and k == j,
                               "status": decision["status"] if selected and k == j else "candidate",
                               "reason": decision["reason"] if selected and k == j else "below_min_iou" if weights[i][k]<min_iou else "not_selected_by_assignment"})
    return decisions, candidates, issues
