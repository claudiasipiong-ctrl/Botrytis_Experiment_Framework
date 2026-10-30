"""Manual CPU-only loading of assigned, available pretrained checkpoints; no training or inference."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils import bootstrap
bootstrap.install(__file__, ROOT)

import argparse
import gc
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pc-id", required=True, choices=("PC-01", "PC-02", "PC-03"))
    args = parser.parse_args()

    import torch
    from orchestration.pc_registry import PCRegistry
    from utils.architectures import get_arch
    from utils.classifier_models import build_classifier
    from utils.config_loader import load_config, cfg_path
    from utils.durable import atomic_json
    from utils.weights import entry_for, verified_local_weight

    torch.set_num_threads(2)
    cfg = load_config(args.config)
    if cfg["compute"]["pc_id"] != args.pc_id:
        raise RuntimeError("La PC indicada no coincide con la configuracion local")
    report = cfg_path(cfg, "paths.artifacts_dir") / "model_loading" / f"{args.pc_id}_{uuid.uuid4().hex}.json"
    result = {"pc_id": args.pc_id, "scope": "available_assigned_checkpoints_cpu_only",
              "training_started": False, "inference_started": False,
              "checks": [], "blocked_architectures": []}

    for slug in PCRegistry(cfg).owned_slugs(args.pc_id):
        spec = get_arch(slug)
        catalog = entry_for(spec, cfg)
        if not catalog["compatible_pretrained"]:
            result["blocked_architectures"].append({"architecture": spec.name, "reason": catalog["block_reason"]})
            atomic_json(report, result)
            print(f"BLOQUEADO: {spec.name}; no se intenta cargar ni sustituir", flush=True)
            continue
        print(f"CARGANDO EN CPU: {spec.name}", flush=True)
        try:
            source = verified_local_weight(spec, cfg)
            if spec.is_detector:
                from utils.pretrained import configure_ultralytics
                configure_ultralytics()
                from ultralytics import YOLO
                wrapper = YOLO(str(source))
                model = wrapper.model
            else:
                model, _ = build_classifier(spec, pretrained=True, cfg=cfg)
            if any(p.device.type != "cpu" for p in model.parameters()):
                raise RuntimeError("Modelo fuera de CPU")
            count = sum(p.numel() for p in model.parameters())
            if count <= 0:
                raise RuntimeError("Modelo sin parametros")
            result["checks"].append({"architecture": spec.name, "status": "PASS", "device": "cpu",
                                     "parameters": count, "weight": str(source)})
            atomic_json(report, result)
            print(f"CARGA CORRECTA: {spec.name} / {count:,} parametros", flush=True)
            del model
            if spec.is_detector:
                del wrapper
            gc.collect()
        except Exception as exc:
            result["checks"].append({"architecture": spec.name, "status": "FAIL", "error": str(exc)})
            atomic_json(report, result)
            raise

    if not result["checks"]:
        raise RuntimeError("No se comprobaron modelos disponibles; revise el informe")
    print(f"CARGA DE PESOS DISPONIBLES COMPLETADA: {args.pc_id}", flush=True)
    print(f"INFORME: {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
