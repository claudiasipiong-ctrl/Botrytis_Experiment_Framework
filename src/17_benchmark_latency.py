"""17 - Benchmark de latencia en la PC designada ``BENCHMARK_PC`` (Especificacion s2, sec. 29).

Protocolo IDENTICO para todos los modelos: batch 1, FP16, ``warmup_iterations`` descartadas y
``measured_iterations`` medidas con ``torch.cuda.synchronize()`` antes/despues, imagenes/crops PRE-CARGADOS en
RAM (``include_disk_io: false``). Mide: detector en ms/imagen, clasificador en ms/crop (solo forward y
con preprocesamiento) y el pipeline congelado ``T = T_detector + T_crop/preproceso + T_clasificador`` por imagen
(``image_id, n_detections, t_detector_ms, t_crop_preprocess_ms, t_classifier_ms, t_total_ms``), ademas del pico de VRAM de inferencia.

Aborta si: la PC no es ``BENCHMARK_PC``; hay otro proceso usando la GPU o un trabajo RUNNING de la cola en esta
PC (sec. 35: no medir latencia mientras la GPU entrena). Usa imagenes del validation externo (no toca el test).

Salidas (``artifacts/benchmark/``): ``latency_detection.csv``, ``latency_classification.csv``, ``latency_pipeline.csv``,
``benchmark_protocol.json``; figura ``pipe_latencia_por_etapa`` (pipeline congelado).

Ejemplo::

    python src/17_benchmark_latency.py --pc-id PC-01
"""

from __future__ import annotations

# Stdlib bootstrap: records import/CLI/config failures before optional dependencies.
import sys as _boot_sys
from pathlib import Path as _BootPath
_boot_root = _BootPath(__file__).resolve().parents[1]
_boot_sys.path.insert(0, str(_boot_root / "src"))
from utils import bootstrap as _bootstrap
_bootstrap.install(__file__, _boot_root)

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from orchestration.study_queue import StudyQueue  # noqa: E402
from utils.architectures import archs, get_arch  # noqa: E402
from utils.checks import MethodologyViolation  # noqa: E402
from utils.classifier_data import build_transforms, crops_for, load_crop_manifest  # noqa: E402
from utils.classifier_models import load_classifier  # noqa: E402
from utils.config_loader import cfg_path  # noqa: E402
from utils.freeze import load_freeze, verify_freeze  # noqa: E402
from utils.guards import GpuJobLock, assert_benchmark_pc  # noqa: E402
from utils.hpo_results import final_checkpoint, stage_json_path  # noqa: E402
from utils.inference import crops_from_boxes, load_detector, rgb_to_bgr  # noqa: E402
from utils.io import load_image_rgb, read_json, write_csv, write_json  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.pipeline_eval import _classify_timed  # noqa: E402
from utils.run_common import prepare_for_inference  # noqa: E402
from utils.timing import free_cuda_memory, gpu_compute_processes, peak_vram_mb, reset_peak_vram, summarize_ms, synchronize, time_callable  # noqa: E402
from utils.training_common import resolve_device, ultralytics_device  # noqa: E402
from utils.visualization import plot_latency_breakdown  # noqa: E402

SCRIPT = "17_benchmark_latency"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    parser.add_argument("--stage", choices=("models", "pipeline", "all"), default="all", help="modelos por separado, pipeline congelado o ambos")
    parser.add_argument("--architectures", nargs="*", default=None, help="solo estas arquitecturas (por defecto las 10)")
    parser.add_argument("--allow-busy-gpu", action="store_true", help="NO usar en el benchmark oficial: permite medir con la GPU ocupada (solo desarrollo)")
    parser.add_argument("--skip-env-check", action="store_true", help="no comparar el entorno con la referencia")


def assert_gpu_idle(ctx: RunContext, allow_busy: bool) -> None:
    """[sec. 35] No medir latencia mientras la GPU ejecuta otro trabajo."""
    cfg = ctx.cfg
    busy: list[str] = []
    # En Windows (WDDM) nvidia-smi lista tambien apps de escritorio (Office, navegador...) como usuarias de la GPU; lo que invalida
    # la medicion es OTRO trabajo de computo: procesos Python (entrenamiento/inferencia). Los procesos no Python se registran como aviso.
    procs = [p for p in gpu_compute_processes() if p["pid"] != os.getpid()]
    others = [p for p in procs if "python" in str(p["name"]).lower()]
    if others:
        busy.append(f"procesos Python en GPU: {others}")
    if procs and not others:
        ctx.logger.warning("procesos de escritorio usando la GPU (no son trabajos de computo; cierre lo que pueda): %d", len(procs))
    running = [j for j in StudyQueue.from_cfg(cfg).jobs("RUNNING") if j.pc_id == ctx.pc_id]
    if running:
        busy.append(f"jobs RUNNING de esta PC en la cola: {[j.job_id for j in running]}")
    if busy and bool(cfg["benchmark"].get("require_idle_gpu", True)):
        if not allow_busy:
            raise MethodologyViolation("La GPU de benchmark no esta libre (" + "; ".join(busy) + "). No se mide latencia mientras se entrena (sec. 29.1, 35).")
        ctx.logger.warning("--allow-busy-gpu: medicion NO valida para el benchmark oficial (%s)", busy)


def bench_detector(ctx: RunContext, spec: Any, images: list[np.ndarray], proto: dict[str, Any]) -> dict[str, Any]:
    """Latencia (ms/imagen) de un detector con el protocolo comun."""
    cfg = ctx.cfg
    free_cuda_memory()  # el pico de VRAM debe ser solo de ESTE modelo, no de los anteriores
    model = load_detector(final_checkpoint(cfg, spec), spec)
    th_path = stage_json_path(cfg, "inference_thresholds", spec)
    th = read_json(th_path) if th_path.is_file() else {}
    conf = float(th.get("conf", 0.25))
    iou = float(th["nms_iou"]) if th.get("nms_iou") is not None else float(cfg["detector"]["val_nms_iou"])
    half = str(cfg["detector"]["precision"]).lower() == "fp16" and resolve_device(cfg) != "cpu"
    bgr = [rgb_to_bgr(a) for a in images]
    reset_peak_vram()

    def call(i: int) -> Any:
        return model.predict(source=bgr[i % len(bgr)], conf=conf, iou=iou, imgsz=int(cfg["detector"]["image_size"]), half=half, device=ultralytics_device(cfg), verbose=False)

    samples = time_callable(call, warmup=proto["warmup_iterations"], iterations=proto["measured_iterations"], sync=proto["cuda_synchronization"])
    st = summarize_ms(samples)
    return {"architecture": spec.name, "slug": spec.slug, "task": "detection", "latency_mean_ms": st["mean_ms"], "latency_std_ms": st["std_ms"], "latency_median_ms": st["median_ms"],
            "latency_p95_ms": st["p95_ms"], "fps": st["fps"], "n": st["n"], "vram_inference_mb": peak_vram_mb(), "conf": conf, "nms_iou": iou, "imgsz": int(cfg["detector"]["image_size"]),
            "precision": "fp16" if half else "fp32"}


def bench_classifier(ctx: RunContext, spec: Any, crops: list[Any], proto: dict[str, Any]) -> dict[str, Any]:
    """Latencia (ms/crop) de un clasificador: forward puro y con preprocesamiento."""
    cfg = ctx.cfg
    free_cuda_memory()
    device = resolve_device(cfg)
    model, meta = load_classifier(final_checkpoint(cfg, spec), device)
    tf = build_transforms(cfg, meta["norm"], train=False)
    use_amp = device != "cpu" and str(cfg["classifier"]["precision"]).lower() == "fp16"
    tensors = [tf(c.convert("RGB")).unsqueeze(0).to(device) for c in crops]
    reset_peak_vram()

    @torch.no_grad()
    def fwd(i: int) -> Any:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(tensors[i % len(tensors)])
        return torch.softmax(out.float(), dim=1)

    @torch.no_grad()
    def full(i: int) -> Any:
        x = tf(crops[i % len(crops)].convert("RGB")).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x)
        return torch.softmax(out.float(), dim=1)

    f = summarize_ms(time_callable(fwd, warmup=proto["warmup_iterations"], iterations=proto["measured_iterations"], sync=proto["cuda_synchronization"]))
    vram = peak_vram_mb()
    g = summarize_ms(time_callable(full, warmup=proto["warmup_iterations"], iterations=proto["measured_iterations"], sync=proto["cuda_synchronization"]))
    return {"architecture": spec.name, "slug": spec.slug, "task": "classification", "latency_mean_ms": f["mean_ms"], "latency_std_ms": f["std_ms"], "latency_median_ms": f["median_ms"],
            "latency_p95_ms": f["p95_ms"], "fps": f["fps"], "n": f["n"], "latency_with_preprocess_ms": g["mean_ms"], "vram_inference_mb": vram, "imgsz": int(cfg["classifier"]["image_size"]),
            "precision": "fp16" if use_amp else "fp32"}


def bench_pipeline(ctx: RunContext, freeze: dict[str, Any], images: dict[str, np.ndarray], proto: dict[str, Any]) -> pd.DataFrame:
    """Latencia por etapa del pipeline congelado (una fila por iteracion medida)."""
    cfg = ctx.cfg
    from PIL import Image

    det_spec = get_arch(freeze["detector"]["architecture"])
    detector = load_detector(freeze["detector"]["checkpoint"], det_spec)
    classifier, meta = load_classifier(freeze["classifier"]["checkpoint"], resolve_device(cfg))
    half = str(cfg["detector"]["precision"]).lower() == "fp16" and resolve_device(cfg) != "cpu"
    conf = float(freeze["detector"]["confidence_threshold"])
    nms = float(freeze["detector"]["nms_iou"]) if freeze["detector"]["nms_iou"] is not None else float(cfg["detector"]["val_nms_iou"])
    ids = sorted(images)
    pil = {i: Image.fromarray(images[i]) for i in ids}
    bgr = {i: rgb_to_bgr(images[i]) for i in ids}
    import time

    def one(iid: str) -> dict[str, Any]:
        synchronize()
        t0 = time.perf_counter()
        r = detector.predict(source=bgr[iid], conf=conf, iou=nms, imgsz=int(cfg["detector"]["image_size"]), half=half, device=ultralytics_device(cfg), verbose=False)[0]
        synchronize()
        t_det = (time.perf_counter() - t0) * 1000.0
        boxes = r.boxes.xyxy.detach().cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        t1 = time.perf_counter()
        crops = crops_from_boxes(pil[iid], boxes, int(cfg["classification_dataset"]["margin_px"]))
        t_crop = (time.perf_counter() - t1) * 1000.0
        _, t_pre, t_cls = _classify_timed(classifier, crops, meta["norm"], cfg)
        return {"image_id": iid, "n_detections": int(len(boxes)), "t_detector_ms": t_det, "t_crop_preprocess_ms": t_crop + t_pre, "t_classifier_ms": t_cls,
                "t_total_ms": t_det + t_crop + t_pre + t_cls}

    for i in range(proto["warmup_iterations"]):
        one(ids[i % len(ids)])
    rows = [one(ids[i % len(ids)]) for i in range(proto["measured_iterations"])]
    return pd.DataFrame(rows)


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Mide la latencia de los 10 modelos y del pipeline congelado en la PC de benchmark."""
    cfg, log = ctx.cfg, ctx.logger
    assert_benchmark_pc(cfg, ctx.pc_id)
    data = prepare_for_inference(ctx, args)
    assert_gpu_idle(ctx, bool(args.allow_busy_gpu))
    b = cfg["benchmark"]
    for k in ("warmup_iterations", "measured_iterations", "cuda_synchronization", "include_disk_io"):
        if str(b[k]).upper().startswith("TODO"):
            raise MethodologyViolation(f"benchmark.{k} sin definir (sec. 29.3)")
    proto = {"benchmark_pc": b["benchmark_pc"], "warmup_iterations": int(b["warmup_iterations"]), "measured_iterations": int(b["measured_iterations"]),
             "cuda_synchronization": bool(b["cuda_synchronization"]), "include_disk_io": bool(b["include_disk_io"]), "batch_size": int(b.get("batch_size", 1)),
             "precision": "fp16", "gpu_model": ctx.env.get("gpu_model"), "nvidia_driver": ctx.env.get("nvidia_driver"), "torch": ctx.env.get("torch"),
             "ultralytics": ctx.env.get("ultralytics"), "images": "validation externo (pre-cargadas en RAM)", "environment_hash": ctx.env.get("environment_hash")}
    if proto["include_disk_io"]:
        raise MethodologyViolation("include_disk_io=true no esta soportado: las imagenes se pre-cargan en RAM (sec. 29.1)")
    ctx.note_split_access("validation", "benchmark")
    val_ids = sorted(data.manifest.loc[data.manifest["outer_split"] == "validation", "image_id"])
    paths = {r.image_id: Path(r.image_path) for r in data.images.itertuples(index=False) if r.image_id in set(val_ids)}
    images = {i: np.asarray(load_image_rgb(paths[i])) for i in val_ids}
    wanted = {get_arch(a).slug for a in args.architectures} if args.architectures else None
    out_dir = cfg_path(cfg, "paths.artifacts_dir") / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    with GpuJobLock(cfg, ctx.pc_id, job="benchmark"):
        if args.stage in ("models", "all"):
            det_rows = [bench_detector(ctx, s, [images[i] for i in val_ids], proto) for s in archs("detection") if wanted is None or s.slug in wanted]
            crops = crops_for(load_crop_manifest(cfg), split="validation", stage="outer_validation", cfg=cfg)
            root = cfg_path(cfg, "paths.classification_dir")
            from PIL import Image

            crop_imgs = [Image.open(root / p).convert("RGB") for p in crops["crop_path"].head(200)]
            cls_rows = [bench_classifier(ctx, s, crop_imgs, proto) for s in archs("classification") if wanted is None or s.slug in wanted]
            for name, rows in (("detection", det_rows), ("classification", cls_rows)):
                if rows:
                    df = pd.DataFrame(rows).assign(pc_id=ctx.pc_id)
                    write_csv(df, out_dir / f"latency_{name}.csv", float_format="%.6g")
                    ctx.register_artifact(out_dir / f"latency_{name}.csv", f"latencia {name}")
                    log.info("latencia %s:\n%s", name, df[["architecture", "latency_mean_ms", "latency_p95_ms", "fps", "vram_inference_mb"]].to_string(index=False))
        if args.stage in ("pipeline", "all"):
            freeze = load_freeze(cfg)
            verify_freeze(cfg, freeze, ctx.hashes, ctx.env)
            pipe = bench_pipeline(ctx, freeze, images, proto)
            write_csv(pipe, out_dir / "latency_pipeline.csv", float_format="%.5f")
            ctx.register_artifact(out_dir / "latency_pipeline.csv", "latencia del pipeline congelado")
            ctx.save_figure(plot_latency_breakdown(pipe), "pipe_latencia_por_etapa", data=pipe, sources=[out_dir / "latency_pipeline.csv"])
            ctx.log_metrics({"pipeline_latency_mean_ms": float(pipe["t_total_ms"].mean()), "pipeline_detector_ms": float(pipe["t_detector_ms"].mean()),
                             "pipeline_classifier_ms": float(pipe["t_classifier_ms"].mean())})
    write_json(out_dir / "benchmark_protocol.json", proto)
    ctx.log_metrics({"protocol": proto})


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments))
