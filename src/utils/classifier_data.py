"""Datos del clasificador: GT crops (sec. 8), folds internos heredados de la imagen y transformaciones.

* El clasificador SOLO se entrena con GT crops (``data/classification``), nunca con crops de un detector.
* Los folds salen de ``inner_fold`` del manifiesto (heredado de la imagen original: todos los crops de
  una imagen caen en el mismo fold; sec. 6.1).
* Augmentation aleatoria SOLO en train. validation/test usan ``Resize`` determinista (sec. 35).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from utils.checks import MethodologyViolation
from utils.classifier_models import CLASS_NAMES
from utils.config_loader import cfg_path, get_nested
from utils.io import read_csv
from utils.reproducibility import make_torch_generator, seed_worker
from utils.splits import assert_split_allowed

#: Clases de ``torchvision.transforms`` con comportamiento aleatorio (prohibidas en validation/test).
RANDOM_TRANSFORMS: tuple[type, ...] = (
    T.RandomResizedCrop, T.RandomHorizontalFlip, T.RandomVerticalFlip, T.RandomRotation, T.ColorJitter,
    T.RandomAffine, T.RandomCrop, T.RandomPerspective, T.RandomErasing, T.AutoAugment, T.RandAugment, T.TrivialAugmentWide,
)


def load_crop_manifest(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Manifiesto de crops GT (``data/classification/manifest.csv``)."""
    root = cfg_path(cfg, "paths.classification_dir")
    from utils.semantics import require_stamp
    require_stamp(root, cfg)
    table = read_csv(root / str(cfg["classification_dataset"]["manifest_name"]))
    from utils.isolation import validate_derived_manifest
    validate_derived_manifest(cfg, table, "classification")
    return table


def label_to_index(label: str, cfg: Mapping[str, Any]) -> int:
    """``sin_signos`` -> 0, ``con_signos`` -> 1 (la clase positiva es fija)."""
    pos = cfg["classification_labels"]["positive_label"]
    if label not in CLASS_NAMES:
        raise ValueError(f"Etiqueta desconocida {label!r}")
    return 1 if label == pos else 0


def crops_for(manifest: pd.DataFrame, *, split: str, stage: str, cfg: Mapping[str, Any], exclude_fold: int | None = None,
              only_fold: int | None = None) -> pd.DataFrame:
    """Filas de crops de un split con las guardias de acceso de la metodologia.

    Args:
        split: ``train`` | ``validation`` | ``test`` (``test`` exige ``freeze_manifest.json`` y ``stage='final_test'``).
        stage: Etapa (``hpo_trial``, ``outer_train``, ``outer_validation``, ``threshold_tuning``, ``final_test``...).
        exclude_fold: Para ENTRENAR un fold interno: todas las filas de train salvo las de este fold.
        only_fold: Para VALIDAR un fold interno: solo las filas de este fold.
    """
    assert_split_allowed(split, stage, cfg)
    df = manifest[manifest["outer_split"] == split].copy()
    if exclude_fold is not None or only_fold is not None:
        if split != "train":
            raise MethodologyViolation("Los folds internos solo existen dentro del outer train")
        if df["inner_fold"].isna().any():
            raise MethodologyViolation("Hay crops de train sin inner_fold")
        if only_fold is not None:
            df = df[df["inner_fold"].astype(int) == int(only_fold)]
        if exclude_fold is not None:
            df = df[df["inner_fold"].astype(int) != int(exclude_fold)]
    df["y"] = [label_to_index(v, cfg) for v in df["label"]]
    return df.reset_index(drop=True)


def build_transforms(cfg: Mapping[str, Any], norm: Mapping[str, Sequence[float]], *, train: bool) -> Callable[[Image.Image], torch.Tensor]:
    """Transformacion de train (con augmentation moderada) o de evaluacion (determinista)."""
    size = int(cfg["classifier"]["image_size"])
    normalize = T.Normalize(mean=list(norm["mean"]), std=list(norm["std"]))
    if not train:
        return T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), normalize])
    a = cfg["classifier"]["augmentation"]
    return T.Compose([
        T.RandomResizedCrop(size, scale=tuple(a["rrc_scale"]), ratio=tuple(a["rrc_ratio"]), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(float(a["hflip"])),
        T.RandomRotation(float(a["rotation_deg"])),
        T.ColorJitter(brightness=float(a["color_jitter"]), contrast=float(a["color_jitter"]), saturation=float(a["color_jitter"])),
        T.ToTensor(), normalize,
    ])


def assert_deterministic_transform(transform: Any) -> None:
    """[sec. 35] Aborta si una transformacion de validation/test contiene augmentation aleatoria."""
    items = getattr(transform, "transforms", [transform])
    for t in items:
        if isinstance(t, RANDOM_TRANSFORMS):
            raise MethodologyViolation(f"Augmentation aleatoria ({type(t).__name__}) en validation/test (sec. 35)")


class CropDataset(Dataset):
    """Crops GT en memoria (dataset pequeno) con transformacion por muestra."""

    def __init__(self, df: pd.DataFrame, root: Path, transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.images: list[Image.Image] = []
        for rel in self.df["crop_path"]:
            with Image.open(root / rel) as im:
                self.images.append(im.convert("RGB").copy())
        self.y = self.df["y"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        return self.transform(self.images[i]), int(self.y[i]), i


def make_loader(ds: Dataset, *, batch_size: int, shuffle: bool, seed: int, workers: int, drop_last: bool = False) -> DataLoader:
    """``DataLoader`` reproducible (generador y ``worker_init_fn`` sembrados)."""
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=drop_last,
        generator=make_torch_generator(seed), worker_init_fn=seed_worker if workers > 0 else None,
        persistent_workers=workers > 0, pin_memory=torch.cuda.is_available(),
    )


def class_weights(y: np.ndarray, strategy: str) -> torch.Tensor | None:
    """Pesos de clase para la CE (``weighted_ce``: inversamente proporcionales a la frecuencia)."""
    if strategy == "none":
        return None
    if strategy != "weighted_ce":
        raise ValueError(f"imbalance.strategy desconocido: {strategy!r}")
    counts = np.bincount(y, minlength=2).astype(float)
    if (counts == 0).any():
        return None
    return torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32)


def eval_workers(cfg: Mapping[str, Any]) -> int:
    """Workers del DataLoader de clasificacion."""
    return int(get_nested(cfg, "classifier.workers", default=0))
