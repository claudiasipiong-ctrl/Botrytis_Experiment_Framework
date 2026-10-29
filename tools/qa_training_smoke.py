"""Prueba manual opt-in: datos sinteticos minimos, una epoca, sin pesos preentrenados.

No usa las imagenes reales. qa.py aplica timeout y supervision externa del proceso.
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
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from utils.config_loader import load_config
    from utils.safety import SafetyMonitor
    base = load_config(a.config)
    SafetyMonitor(base, a.output / "smoke_telemetry.jsonl").check()
    work = a.output / "synthetic_training"
    work.mkdir()
    import numpy as np
    import pandas as pd
    from PIL import Image
    generator = np.random.default_rng(20261028)  # seed de QA exclusivamente
    rows = []
    for i in range(8):
        path = work / f"synthetic_{i}.png"
        Image.fromarray(generator.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)
        rows.append({"crop_path": path.name, "y": i % 2, "image_id": f"qa_{i}"})
    cfg = load_config("configs/smoke.yaml", [f"paths.classification_dir={work}", "classifier.pretrained=false", "detector.pretrained=false", "initialization.mode=scratch", "initialization.scratch_decision=SYNTHETIC_QA_PERSONAL",
                      "classifier.batch_size=2", "classifier.workers=0", "classifier.image_size=64",
                      "classifier.max_epochs=1", "classifier.oom_fallback=false"])
    cfg["compute"]["device"] = base["compute"]["device"]
    from utils.architectures import get_arch
    from utils.classifier_train import train_classifier
    run = train_classifier(get_arch("ResNet50"), {"optimizer": "AdamW", "lr": 1e-4, "weight_decay": 1e-4},
                           cfg, pd.DataFrame(rows[:4]), pd.DataFrame(rows[4:]), seed=20261028, max_epochs=1,
                           keep_state=False)
    if run.epochs_run != 1 or not 0 <= run.score <= 1:
        raise RuntimeError("Salida clasificador sintetico invalida")
    # Un detector activo verifica API Ultralytics y backward sin descargar checkpoints.
    from utils.pretrained import configure_ultralytics
    configure_ultralytics()
    from ultralytics import YOLO
    detect = work / "detection"
    for split in ("train", "val"):
        (detect / "images" / split).mkdir(parents=True)
        (detect / "labels" / split).mkdir(parents=True)
        for i in range(2):
            source = work / f"synthetic_{i}.png"
            import shutil
            shutil.copyfile(source, detect / "images" / split / source.name)
            (detect / "labels" / split / f"synthetic_{i}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    import yaml
    data_file = work / "data.yaml"
    data_file.write_text(yaml.safe_dump({"path": str(detect), "train": "images/train", "val": "images/val",
                                        "names": {0: "fruit"}}), encoding="utf-8")
    device = 0 if str(cfg["compute"]["device"]) == "auto" else cfg["compute"]["device"]
    model = YOLO("yolov8n.yaml")
    model.train(data=str(data_file), epochs=1, imgsz=64, batch=2, workers=0, device=device,
                pretrained=False, amp=False, plots=False, save=True, project=str(work), name="detector",
                deterministic=True, seed=20261028, exist_ok=False)
    final = work / "detector/weights/last.pt"
    if not final.is_file():
        raise RuntimeError("Ultralytics no publico last.pt en la prueba sintetica")
    print("PASS: forward/backward sinteticos clasificador+detector; sin precision ni velocidad cientificas")
    print("Limitacion: no prueba todas las arquitecturas, AMP de todos los modelos ni carga sostenida")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
