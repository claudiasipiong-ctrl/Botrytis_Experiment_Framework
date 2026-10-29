"""Compatibilidad de API: solo pesos locales verificados, sin red implicita."""
from pathlib import Path
from typing import Any, Mapping
from utils.architectures import ArchSpec
from utils.config_loader import cfg_path
from utils.hashing import sha256_file
RTDETR_R50_CKPT = "rtdetr_r50_verified.pt"
def pretrained_dir(cfg):
    return cfg_path(cfg, "paths.artifacts_dir") / "pretrained"
def configure_ultralytics():
    import os
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    from ultralytics import settings
    settings.update({"sync": False})
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import ultralytics.utils.downloads as downloads
    def local_asset(file, *args, **kwargs):
        path=Path(file)
        if not path.is_file():raise FileNotFoundError("Descarga implicita bloqueada: "+str(file)+". Adquirir manualmente con manage_weights.py")
        return str(path)
    def no_network(*args, **kwargs):
        raise RuntimeError("Descarga implicita bloqueada. Use adquisicion manual explicita")
    downloads.attempt_download_asset=local_asset
    downloads.safe_download=no_network
    import ultralytics.nn.tasks as tasks
    if hasattr(tasks,"attempt_download_asset"):tasks.attempt_download_asset=local_asset
def ensure_ultralytics_weights(spec: ArchSpec, cfg: Mapping[str, Any]) -> Path:
    from utils.weights import verified_local_weight
    return verified_local_weight(spec, cfg)
def weights_sha256(path):
    return sha256_file(path)  # se recalcula para no aceptar un archivo modificado
def build_rtdetr_r50_checkpoint(dest):
    raise RuntimeError("BLOQUEADO: trasplante ImageNet parcial no equivale a RT-DETR-R50 preentrenado. Vea docs/PESOS.md")
def torchvision_to_ultralytics_key(key):
    raise RuntimeError("Conversion parcial obsoleta; no usar como preentrenamiento")
