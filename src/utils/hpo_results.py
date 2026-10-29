"""Resultados del HPO y de la robustez: top-k configuraciones, mejor configuracion congelada por arquitectura."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from utils.architectures import ArchSpec
from utils.config_loader import cfg_path
from utils.io import ensure_dir, read_json
from utils.search_spaces import config_id


def top_configs(study: Any, k: int) -> list[dict[str, Any]]:
    """Top-``k`` configuraciones DISTINTAS de un estudio por valor (desempate: menor n.º de trial).

    Returns:
        ``[{'config_id', 'hp', 'trial_number', 'value'}]`` ordenado de mejor a peor.
    """
    done = [t for t in study.get_trials(deepcopy=False) if t.state.name == "COMPLETE" and t.value is not None and t.user_attrs.get("hp")]
    done.sort(key=lambda t: (-float(t.value), t.number))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for t in done:
        cid = config_id(t.user_attrs["hp"])
        if cid in seen:
            continue
        seen.add(cid)
        out.append({"config_id": cid, "hp": dict(t.user_attrs["hp"]), "trial_number": int(t.number), "value": float(t.value)})
        if len(out) >= k:
            break
    return out


def best_config_path(cfg: Mapping[str, Any], spec: ArchSpec) -> Path:
    """``artifacts/best_configs/<slug>.json`` (configuracion congelada de la arquitectura)."""
    return ensure_dir(cfg_path(cfg, "paths.best_configs_dir")) / f"{spec.slug}.json"


def load_best_config(cfg: Mapping[str, Any], spec: ArchSpec) -> dict[str, Any]:
    """Carga la mejor configuracion (resultado de ``08_repeat_top_configs.py``).

    Raises:
        FileNotFoundError: Si la robustez de la arquitectura aun no se agrego.
    """
    p = best_config_path(cfg, spec)
    if not p.is_file():
        raise FileNotFoundError(f"No existe {p}: ejecute 08_repeat_top_configs.py (--aggregate-only) para {spec.name} primero.")
    result = dict(read_json(p))
    from utils.semantics import semantic_fingerprint
    if result.get("semantic_fingerprint") != semantic_fingerprint(cfg):
        raise RuntimeError("Mejor configuracion obsoleta/no trazable; repetir HPO/robustez v4")
    return result


def stage_json_path(cfg: Mapping[str, Any], stage: str, spec: ArchSpec) -> Path:
    """``artifacts/<stage>/<slug>.json`` (p. ej. ``outer_validation``, ``inference_thresholds``)."""
    return ensure_dir(cfg_path(cfg, "paths.artifacts_dir") / stage) / f"{spec.slug}.json"


def predictions_path(cfg: Mapping[str, Any], spec: ArchSpec) -> Path:
    """Predicciones de validation de ``spec`` en ESTA PC (``artifacts/outer_validation/<slug>_predictions.csv``).

    Se ignora la ruta absoluta guardada dentro del JSON: los resultados viajan entre PCs por ZIP y la ruta original
    (de otra PC) no existe aqui.
    """
    return stage_json_path(cfg, "outer_validation", spec).with_name(f"{spec.slug}_predictions.csv")


def checkpoint_dir(cfg: Mapping[str, Any], spec: ArchSpec) -> Path:
    """``artifacts/checkpoints/<slug>`` (checkpoint definitivo, 70 % completo)."""
    return ensure_dir(cfg_path(cfg, "paths.artifacts_dir") / "checkpoints" / spec.slug)


def final_checkpoint(cfg: Mapping[str, Any], spec: ArchSpec) -> Path:
    """Ruta del checkpoint definitivo (``final.pt``)."""
    return checkpoint_dir(cfg, spec) / "final.pt"
