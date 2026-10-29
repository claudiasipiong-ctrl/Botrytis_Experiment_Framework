"""Label-only audit for a YOLO directory or ZIP. Does not infer image dimensions."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence
from utils.pairing import pair_image, SEMANTICS_VERSION

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPECIAL_STEMS = ["213f3bad-IMG_6979", "2d515215-IMG_6819", "7c5a8c55-IMG_6823", "a8bbe159-IMG_6969"]


def source_files(source: str | Path, *, labels_dir: Path | None = None, images_dir: Path | None = None) -> tuple[list[tuple[str,bytes]],list[str],list[tuple[str,str]]]:
    """Read source labels and discover original images; never extract ZIP paths."""
    source = Path(source)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            bad = z.testzip()
            if bad:
                raise ValueError(f"ZIP CRC invalido: {bad}")
            labels = [(n, z.read(n)) for n in sorted(z.namelist())
                      if "/labels/" in "/"+n and n.endswith(".txt") and Path(n).name != "classes.txt"]
            images = [n for n in sorted(z.namelist()) if "/images/" in "/"+n and Path(n).suffix.lower() in IMAGE_EXTENSIONS]
            class_files = [(n, z.read(n).decode("utf-8-sig")) for n in z.namelist() if Path(n).name == "classes.txt"]
    else:
        label_paths = sorted(labels_dir.glob("*.txt")) if labels_dir else sorted(source.rglob("labels/*.txt"))
        image_paths = sorted(images_dir.glob("*")) if images_dir else sorted(source.rglob("images/*"))
        relative = lambda p: str(p.relative_to(source)) if source in p.parents else str(p)
        labels = [(relative(p), p.read_bytes()) for p in label_paths if p.name != "classes.txt"]
        images = [relative(p) for p in image_paths if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        class_files = [(str(p.relative_to(source)), p.read_text(encoding="utf-8-sig")) for p in source.rglob("classes.txt")]
    return labels, images, class_files


def _distribution(values: Sequence[float]) -> dict[str,Any]:
    values = sorted(values)
    if not values:
        return {"n": 0, "min": None, "max": None, "mean": None, "quantiles": {}}
    def quantile(p: float) -> float:
        k = (len(values)-1)*p
        lo, hi = math.floor(k), math.ceil(k)
        return values[lo] + (values[hi]-values[lo])*(k-lo)
    return {"n": len(values), "min": values[0], "max": values[-1],
            "mean": sum(values)/len(values), "quantiles": {str(p): quantile(p) for p in (0., .05, .25, .5, .75, .95, 1.)}}


def audit_source(source: str | Path, min_iou: float = .50, metadata: str | Path | None = None,
                 *, labels_dir: Path | None = None, images_dir: Path | None = None) -> tuple:
    """Recompute counts, all candidate IoUs and anomalies without reading pixels."""
    labels, images, class_files = source_files(source, labels_dir=labels_dir, images_dir=images_dir)
    counts, empties = Counter(), []
    boxes, pairs, candidates, anomalies, per_image = [], [], [], [], []
    content_groups = defaultdict(list)
    digest = hashlib.sha256()
    for name, raw in labels:
        digest.update(name.encode()+b"\0"+raw.replace(b"\r\n", b"\n")+b"\0")
        iid = Path(name).stem
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            anomalies.append(dict(image_id=iid, reason="invalid_utf8"))
            per_image.append(dict(image_id=iid,n_fruits=0,n_suspects=0,n_signs=0,valid_pairs=0,n_anomalies=1,positive_possible=None,negative_possible=None))
            continue
        content_groups[hashlib.sha256(raw).hexdigest()].append(iid)
        records, file_issues, ordinals = [], [], Counter()
        lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if not lines:
            empties.append(iid)
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                tokens = line.split()
                if len(tokens) != 5 or not re.fullmatch(r"-?\d+", tokens[0]):
                    raise ValueError("bad_format")
                cid = int(tokens[0])
                xc, yc, w, h = map(float, tokens[1:])
                ordinal = ordinals[cid]
                ordinals[cid] += 1
                counts[cid] += 1
                record = dict(image_id=iid, class_id=cid, ordinal=ordinal, line_no=line_no,
                              xc=xc, yc=yc, w=w, h=h, x1=xc-w/2, y1=yc-h/2,
                              x2=xc+w/2, y2=yc+h/2, image_w=1, image_h=1)
                if not all(math.isfinite(v) for v in (xc,yc,w,h)):
                    raise ValueError("non_finite")
                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    file_issues.append(dict(class_id=cid, ordinal=ordinal, line_no=line_no, reason="invalid_yolo_coordinates"))
                if min(record["x1"],record["y1"]) < -1e-6 or max(record["x2"],record["y2"]) > 1+1e-6:
                    file_issues.append(dict(class_id=cid, ordinal=ordinal, line_no=line_no, reason="box_outside_image"))
                records.append(record)
            except (ValueError, OverflowError) as exc:
                file_issues.append(dict(line_no=line_no, reason=str(exc)))
        decisions, all_candidates, issues = pair_image(records, min_iou)
        issues += file_issues
        pairs.extend(dict(image_id=iid, **r) for r in decisions)
        candidates.extend(dict(image_id=iid, **r) for r in all_candidates)
        anomalies.extend(dict(image_id=iid, **r) for r in issues)
        boxes.extend(records)
        valid = sum(r["status"] == "valid" for r in decisions)
        per_image.append(dict(image_id=iid, n_fruits=ordinals[0], n_suspects=ordinals[1], n_signs=ordinals[2],
                              valid_pairs=valid, n_anomalies=len(issues),
                              positive_possible=valid if not issues else None,
                              negative_possible=ordinals[0]-valid if not issues else None))
    image_stems, label_stems = {Path(n).stem for n in images}, {Path(n).stem for n, _ in labels}
    safe = [r for r in per_image if not r["n_anomalies"]]
    report = dict(semantics_version=SEMANTICS_VERSION, source=str(Path(source).resolve()),
        labels_hash=digest.hexdigest(), label_only=not bool(images), images_discovered=len(images),
        labels_total=len(labels), labels_empty=len(empties), empty_label_stems=empties,
        boxes_by_class=dict(sorted(counts.items())), classes_files=dict(class_files),
        unknown_classes={k:v for k,v in counts.items() if k not in (0,1,2)},
        images_without_label=sorted(image_stems-label_stems), labels_without_image=sorted(label_stems-image_stems),
        duplicate_image_stems=[k for k,v in Counter(Path(n).stem.casefold() for n in images).items() if v>1],
        duplicate_label_stems=[k for k,v in Counter(Path(n).stem.casefold() for n,_ in labels).items() if v>1],
        duplicate_nonempty_label_content=[ids for ids in content_groups.values() if len(ids)>1 and ids[0] not in empties],
        min_iou=min_iou, valid_pairs=sum(r["status"]=="valid" for r in pairs),
        anomalies_total=len(anomalies), anomalous_images=sorted({r["image_id"] for r in anomalies}),
        anomalies_by_reason=dict(Counter(reason for r in anomalies for reason in r["reason"].split(";"))),
        crops_possible_clean_images={"con_signos":sum(r["positive_possible"] for r in safe),
                                    "sin_signos":sum(r["negative_possible"] for r in safe)},
        crops_generated={"con_signos":0,"sin_signos":0},
        quarantined_fruits=sum(r["n_fruits"] for r in per_image if r["n_anomalies"]),
        iou_all_candidates=_distribution([r["iou"] for r in candidates]),
        iou_selected=_distribution([r["iou"] for r in pairs]),
        iou_best_per_suspect=_distribution([max((c["iou"] for c in candidates if c["image_id"]==p["image_id"] and c["suspect_ordinal"]==p["suspect_ordinal"]),default=0.) for p in pairs]),
        special_images={iid:dict(summary=next((r for r in per_image if r["image_id"]==iid),None),
            pairs=[r for r in pairs if r["image_id"]==iid], anomalies=[r for r in anomalies if r["image_id"]==iid]) for iid in SPECIAL_STEMS},
        full_image_validation="pending: images required" if not images else "pending: run 01 image validation")
    if metadata:
        text = Path(metadata).read_text(encoding="utf-8-sig")
        match = re.search(r"Total de imagenes PNG:\s*(\d+)", text)
        report["reference_metadata"] = {"source":str(metadata),"historical_image_count":int(match[1]) if match else None,
            "note":"Historical snapshot, never a replacement for images/ discovery."}
    report["invalid_yolo_coordinates"] = sum("invalid_yolo_coordinates" in r["reason"] or "box_outside_image" in r["reason"] or "non_finite" in r["reason"] or "bad_format" in r["reason"] for r in anomalies)
    report["degenerate_boxes"] = sum(r["reason"] == "degenerate_box" for r in anomalies)
    report["duplicate_boxes"] = sum(r["reason"] == "duplicate_box" for r in anomalies)
    report["positive_unambiguous"] = report["crops_possible_clean_images"]["con_signos"]
    report["negative_unambiguous"] = report["crops_possible_clean_images"]["sin_signos"]
    report["unresolved_physical_fruits"] = report["quarantined_fruits"]
    report["crops_possible_with_discovered_images"] = {
        "con_signos":sum(r["positive_possible"] for r in safe if r["image_id"] in image_stems),
        "sin_signos":sum(r["negative_possible"] for r in safe if r["image_id"] in image_stems)}
    report["orphan_label_boxes_by_class"] = dict(Counter(r["class_id"] for r in boxes if r["image_id"] not in image_stems))
    return report, boxes, pairs, candidates, anomalies, per_image


def export_audit(result: tuple, output: str | Path) -> None:
    """Export exact counts, raw boxes and full pairing audit with provenance."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report, boxes, pairs, candidates, anomalies, per_image = result
    (output/"label_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8", newline="\n")
    for name, rows, defaults in (("boxes",boxes,["image_id","class_id","ordinal"]),
        ("pairing_audit",pairs,["image_id","suspect_ordinal","fruit_ordinal","iou","status","reason"]),
        ("pairing_candidates",candidates,["image_id","suspect_ordinal","fruit_ordinal","iou","selected"]),
        ("label_anomalies",anomalies,["image_id","reason"]), ("label_per_image",per_image,["image_id","n_fruits"])):
        fields = list(dict.fromkeys(defaults+[k for r in rows for k in r]))
        with (output/f"{name}.csv").open("w",encoding="utf-8",newline="") as f:
            writer = csv.DictWriter(f, fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Auditoría exacta de labels", "", f"Fuente: `{report['source']}`", "",
        f"Imágenes: {report['images_discovered']}; labels: {report['labels_total']}; vacíos: {report['labels_empty']}.",
        f"Cajas: {report['boxes_by_class']}. Pares válidos: {report['valid_pairs']}.",
        f"Anomalías: {report['anomalies_total']}; imágenes afectadas: {len(report['anomalous_images'])}.",
        f"Crops posibles en imágenes sin anomalías: {report['crops_possible_clean_images']}; generados: 0.",
        "", "El perfil oficial aborta ante anomalías. Los conteos posibles excluyen imágenes ambiguas",
        "como diagnóstico; no autorizan su exclusión del experimento. El CSV conserva todos los candidatos.",
        "Sin imágenes no se comprueban píxeles, EXIF, hashes de imágenes, correspondencia completa ni crops."]
    (output/"label_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")


def main(argv: list[str] | None = None) -> int:
    """Label-only CLI; exit 2 indicates unresolved annotation anomalies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True,help="YOLO directory or ZIP; no extraction required")
    parser.add_argument("--output",type=Path,default=Path("reports/label_audit"))
    parser.add_argument("--min-iou",type=float,default=.50)
    parser.add_argument("--metadata",type=Path)
    args = parser.parse_args(argv)
    if not 0 < args.min_iou <= 1:
        parser.error("--min-iou must be in (0,1]")
    result = audit_source(args.dataset,args.min_iou,args.metadata)
    export_audit(result,args.output)
    print(json.dumps({k:result[0][k] for k in ("images_discovered","labels_total","labels_empty","boxes_by_class","valid_pairs","anomalies_total","crops_possible_clean_images")},indent=2))
    return 2 if result[0]["anomalies_total"] else 0
