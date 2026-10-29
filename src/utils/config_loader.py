"""Carga, validacion y resolucion de la configuracion YAML del experimento.

Responsabilidades
-----------------
* Cargar ``configs/default.yaml`` (y opcionalmente un YAML de override).
* Interpolar ``${ENV}``, ``${ENV:-default}`` y ``${a.b.c}``.
* Aplicar overrides de linea de comandos (``--set a.b=valor``).
* Detectar valores ``TODO*`` (decisiones BLOQUEANTES) y abortar con un mensaje claro.
* Verificar que las decisiones FIJAS de la metodologia no se hayan alterado.
* Resolver rutas relativas contra la raiz del proyecto (independiente del cwd).
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import yaml

#: Raiz del proyecto: ``<root>/src/utils/config_loader.py`` -> ``<root>``.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "default.yaml"

_INTERP_RE = re.compile(r"\$\{([^${}]+)\}")
_MAX_INTERP_DEPTH = 10


class ConfigError(Exception):
    """Error generico de configuracion."""


class BlockingTodoError(ConfigError):
    """Se intento usar un valor marcado como TODO BLOQUEANTE en la especificacion."""


class FixedDecisionViolation(ConfigError):
    """Se altero una decision marcada como FIJO en la metodologia."""


# --------------------------------------------------------------------------- #
# Utilidades de acceso
# --------------------------------------------------------------------------- #
def is_todo(value: Any) -> bool:
    """Indica si ``value`` es un marcador de decision pendiente.

    Solo las cadenas que empiezan por ``TODO`` cuentan como pendientes; ``null`` es un
    valor opcional legitimo y no bloquea.
    """
    return isinstance(value, str) and value.strip().upper().startswith("TODO")


def get_nested(cfg: Mapping[str, Any], dotted_key: str, default: Any = ...) -> Any:
    """Obtiene ``cfg['a']['b']`` desde ``'a.b'``.

    Args:
        cfg: Configuracion.
        dotted_key: Clave con puntos.
        default: Valor por defecto; si se omite y la clave no existe, lanza ``KeyError``.
    """
    node: Any = cfg
    for part in dotted_key.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        else:
            if default is ...:
                raise KeyError(f"Clave de configuracion inexistente: '{dotted_key}'")
            return default
    return node


def set_nested(cfg: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    """Asigna ``cfg['a']['b'] = value`` creando niveles intermedios si hace falta."""
    parts = dotted_key.split(".")
    node: MutableMapping[str, Any] = cfg
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def require(cfg: Mapping[str, Any], *dotted_keys: str, spec_ref: str = "") -> Any:
    """Devuelve el valor de la(s) clave(s) o aborta si es TODO / None / vacio.

    Args:
        cfg: Configuracion.
        *dotted_keys: Una o mas claves con puntos.
        spec_ref: Referencia a la especificacion para el mensaje de error.

    Returns:
        El valor (una clave) o una lista de valores (varias claves).

    Raises:
        BlockingTodoError: Si alguna clave esta sin definir.
    """
    values: list[Any] = []
    missing: list[str] = []
    for key in dotted_keys:
        value = get_nested(cfg, key, default=None)
        if value is None or is_todo(value) or (isinstance(value, str) and not value.strip()):
            missing.append(f"{key} = {value!r}")
        values.append(value)
    if missing:
        ref = f" (ver {spec_ref})" if spec_ref else ""
        raise BlockingTodoError(
            "TODO BLOQUEANTE sin resolver" + ref + ": " + "; ".join(missing)
            + ". Defina el valor en configs/default.yaml o con --set clave=valor ANTES de continuar."
        )
    return values[0] if len(values) == 1 else values


# --------------------------------------------------------------------------- #
# Interpolacion
# --------------------------------------------------------------------------- #
def _resolve_expression(expr: str, root: Mapping[str, Any]) -> str:
    """Resuelve una expresion ``ENV``, ``ENV:-default`` o ``a.b.c``."""
    default: str | None = None
    name = expr
    if ":-" in expr:
        name, default = expr.split(":-", 1)
    name = name.strip()
    if name in os.environ:
        return os.environ[name]
    if "." in name:
        ref = get_nested(root, name, default=None)
        if ref is not None:
            return str(ref)
    if default is not None:
        return default
    raise ConfigError(f"No se pudo resolver la interpolacion '${{{expr}}}' (ni entorno ni clave de config).")


def _interpolate(node: Any, root: Mapping[str, Any], depth: int = 0) -> Any:
    if depth > _MAX_INTERP_DEPTH:
        raise ConfigError("Interpolacion recursiva excede la profundidad maxima (referencia circular?).")
    if isinstance(node, str):
        result = node
        for _ in range(_MAX_INTERP_DEPTH):
            new = _INTERP_RE.sub(lambda m: _resolve_expression(m.group(1), root), result)
            if new == result:
                return new
            result = new
        raise ConfigError(f"Interpolacion no converge para: {node!r}")
    if isinstance(node, dict):
        return {k: _interpolate(v, root, depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate(v, root, depth + 1) for v in node]
    return node


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #
def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def parse_override(text: str) -> tuple[str, Any]:
    """Convierte ``'a.b=valor'`` en ``('a.b', valor)`` interpretando el valor como YAML."""
    if "=" not in text:
        raise ConfigError(f"Override invalido '{text}': use clave=valor")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ConfigError(f"Override invalido '{text}': clave vacia")
    return key, yaml.safe_load(raw)


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: Iterable[str] | None = None,
    *,
    validate_fixed: bool = True,
) -> dict[str, Any]:
    """Carga la configuracion resuelta.

    Args:
        path: YAML a cargar. Si es ``None`` se usa ``configs/default.yaml``. Si es distinto del
            por defecto, se aplica como *override* profundo sobre el por defecto.
        overrides: Lista de ``clave=valor`` (prioridad maxima).
        validate_fixed: Si es ``True`` verifica las decisiones FIJAS.

    Returns:
        Diccionario de configuracion con interpolaciones resueltas. Incluye la clave
        ``_meta`` con la ruta de los ficheros usados.

    Raises:
        ConfigError: YAML ausente o invalido.
        FixedDecisionViolation: Si se altera una decision FIJA.
    """
    if not DEFAULT_CONFIG_PATH.is_file():
        raise ConfigError(f"No existe la configuracion por defecto: {DEFAULT_CONFIG_PATH}")
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}
    used = [str(DEFAULT_CONFIG_PATH)]

    if path is not None:
        p = resolve_path(path)
        if p.resolve() != DEFAULT_CONFIG_PATH.resolve():
            if not p.is_file():
                raise ConfigError(f"No existe el fichero de configuracion: {p}")
            with p.open("r", encoding="utf-8") as fh:
                _deep_merge(cfg, yaml.safe_load(fh) or {})
            used.append(str(p))

    for item in overrides or []:
        key, value = parse_override(item)
        set_nested(cfg, key, value)

    cfg = _interpolate(cfg, cfg)
    cfg["_meta"] = {"config_files": used, "overrides": list(overrides or []), "project_root": str(PROJECT_ROOT)}
    from utils.bootstrap import record_config
    record_config(cfg,"CONFIG_LOADED_PENDING_VALIDATION")
    if validate_fixed:
        validate_fixed_decisions(cfg)
    record_config(cfg,"CONFIG_EFFECTIVE")
    return cfg


def resolve_path(path: str | os.PathLike[str], base: Path | None = None) -> Path:
    """Resuelve una ruta: absoluta se respeta; relativa se ancla a la raiz del proyecto.

    Se expande ``~``. No exige que exista.
    """
    raw_path = str(path)
    if raw_path.startswith(("\\\\", "//")) or "://" in raw_path:
        raise ConfigError("Rutas remotas/UNC no permitidas; use una copia local transferida personalmente")
    p = Path(os.path.expanduser(str(path)))
    if p.is_absolute():
        return p
    return ((base or PROJECT_ROOT) / p).resolve()


def cfg_path(cfg: Mapping[str, Any], dotted_key: str) -> Path:
    """Atajo: ``resolve_path(cfg[dotted_key])`` con error claro si falta."""
    value = get_nested(cfg, dotted_key, default=None)
    if value is None or is_todo(value) or value == "":
        raise BlockingTodoError(f"Ruta sin definir: {dotted_key} = {value!r}")
    return resolve_path(value)


# --------------------------------------------------------------------------- #
# Decisiones FIJAS
# --------------------------------------------------------------------------- #
_FIXED_EXPECTATIONS: dict[str, tuple[Any, str]] = {
    "classes.fruit_class_id": (0, "physical fruit only"),
    "classes.suspicious_class_id": (1, "additional positive annotation"),
    "classes.sign_class_id": (2, "auxiliary visual region"),
    "pairing.method": ("maximum_total_iou_ordinal", "deterministic bipartite matching"),
    "classification_labels.positive_label": ("con_signos", "positive class"),
    "classification_labels.negative_label": ("sin_signos", "negative class"),
    "outer_split.unit": ("original_image", "sec. 5.1: la unidad del split es la imagen original"),
    "outer_split.train": (0.70, "sec. 5: 70 % train"),
    "outer_split.validation": (0.20, "sec. 5: 20 % validation"),
    "outer_split.test": (0.10, "sec. 5: 10 % test"),
    "outer_split.test_locked": (True, "sec. 5.1: test bloqueado hasta el final"),
    "inner_hpo.unit": ("original_image", "sec. 6.3: nunca folds sobre crops"),
    "inner_hpo.group_key": ("original_image_id", "sec. 6.1: todos los crops de una imagen -> mismo fold"),
    "classifier.positive_class": ("con_signos", "sec. 13.2: clase positiva fija"),
    "compute.max_gpu_jobs_per_pc": (1, "sec. 16.1: un trabajo GPU por PC"),
    "compute.max_gpu_jobs_global": (3, "sec. 16.1: maximo 3 trabajos GPU globales"),
    "classification_dataset.margin_px": (0, "sec. 8: sin margen adicional alrededor del fruto"),
    "test.allowed_during_hpo": (False, "sec. 35: test no se usa durante HPO"),
    "test.allowed_during_model_selection": (False, "sec. 35: test no se usa en seleccion"),
    "test.allowed_during_threshold_tuning": (False, "sec. 35: test no se usa para thresholds"),
}


#: Decisiones FIJAS del perfil ``official`` (el perfil ``smoke`` solo relaja presupuestos y tamanos, nunca la logica).
_OFFICIAL_EXPECTATIONS: dict[str, tuple[Any, str]] = {
    "detector.image_size": (640, "sec. 14.1: image_size 640 fijo en la comparacion principal"),
    "classifier.image_size": (224, "sec. 14.2: image_size 224 fijo"),
    "detector.batch_size": (16, "sec. 14.1: batch 16 (effective batch comparable via nbs)"),
    "classifier.batch_size": (32, "sec. 14.2: batch 32"),
    "detector.trials_per_architecture": (15, "sec. 12.1: 15 trials por detector"),
    "classifier.trials_per_architecture": (20, "sec. 12.2: 20 trials por clasificador"),
    "outer_split.seed": (67, "split predeclarado distinto de semillas de robustez"),
    "inner_hpo.folds": (3, "tres folds internos"),
    "detector.precision": ("fp16", "sec. 14: precision fp16"),
    "classifier.precision": ("fp16", "sec. 14: precision fp16"),
    "robustness.top_k_configs": (3, "sec. 12.3: top 3 configuraciones"),
    "robustness.seeds": ([17, 42, 73], "sec. 12.3: seeds predeclaradas 17, 42, 73"),
    "optuna.sampler.name": ("TPE", "sec. 10.2: sampler TPE"),
    "optuna.pruner.name": ("MedianPruner", "sec. 10.3: MedianPruner"),
    "compute.intra_study_parallel_trials": (1, "sec. 16.3: un trial a la vez dentro de un estudio"),
}


def validate_fixed_decisions(cfg: Mapping[str, Any]) -> None:
    """Verifica las decisiones FIJAS.

    Raises:
        FixedDecisionViolation: Si alguna difiere del valor metodologico.
    """
    errors: list[str] = []
    expectations = dict(_FIXED_EXPECTATIONS)
    if str(get_nested(cfg, "project.profile", default="official")) == "official":
        expectations.update(_OFFICIAL_EXPECTATIONS)
    for key, (expected, why) in expectations.items():
        actual = get_nested(cfg, key, default=None)
        if isinstance(expected, float) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
            ok = abs(float(actual) - expected) < 1e-12
        else:
            ok = actual == expected and type(actual) is type(expected)
        if not ok:
            errors.append(f"{key}: esperado {expected!r}, encontrado {actual!r} [{why}]")

    total = sum(float(get_nested(cfg, f"outer_split.{s}", default=0)) for s in ("train", "validation", "test"))
    if abs(total - 1.0) > 1e-9:
        errors.append(f"outer_split: las proporciones suman {total}, deben sumar 1.0")

    fruit = get_nested(cfg, "classes.fruit_class_id", default=None)
    sign = get_nested(cfg, "classes.sign_class_id", default=None)
    fruit_list = list(fruit) if isinstance(fruit, (list, tuple)) else [fruit]
    if isinstance(sign, int) and sign in fruit_list:
        errors.append("classes: sign_class_id no puede estar entre los fruit_class_id")
    if len(fruit_list) != len(set(map(repr, fruit_list))):
        errors.append("classes: fruit_class_id tiene IDs repetidos")

    threshold = get_nested(cfg, "pairing.min_iou", default=None)
    if not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
        errors.append("pairing.min_iou must be in (0,1]")
    if get_nested(cfg, "pairing.anomaly_policy") not in ("error", "quarantine_images"):
        errors.append("pairing.anomaly_policy must be error or quarantine_images")
    if "label_rule" in cfg:
        errors.append("Obsolete label_rule configuration; migrate to annotation pairing")
    init = cfg.get("initialization", {})
    mode = init.get("mode")
    if mode not in ("pretrained", "scratch"):
        errors.append("initialization.mode debe ser pretrained o scratch")
    elif any(cfg.get(s, {}).get("pretrained") is not (mode == "pretrained") for s in ("detector", "classifier")):
        errors.append("detector/classifier.pretrained deben coincidir con initialization.mode")
    if mode == "scratch" and not str(init.get("scratch_decision") or "").strip():
        errors.append("Scratch requiere initialization.scratch_decision predeclarada; experimento separado")
    if errors:
        raise FixedDecisionViolation(
            "Se alteraron decisiones FIJAS de la metodologia:\n  - " + "\n  - ".join(errors)
        )


def list_todos(cfg: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Lista todas las claves con valor TODO* (util para el informe de pendientes)."""
    found: list[str] = []
    for key, value in cfg.items():
        if key == "_meta":
            continue
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping):
            found.extend(list_todos(value, dotted + "."))
        elif isinstance(value, list):
            found.extend(f"{dotted}[{i}]" for i, v in enumerate(value) if is_todo(v))
        elif is_todo(value):
            found.append(f"{dotted} = {value}")
    return found


def dump_config_yaml(cfg: Mapping[str, Any], path: Path) -> None:
    """Escribe la configuracion efectiva (``config.yaml`` de cada run) en UTF-8 con ``\\n``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(__import__("utils.bootstrap", fromlist=["redact"]).redact(dict(cfg)), fh, sort_keys=False, allow_unicode=True, default_flow_style=False)
