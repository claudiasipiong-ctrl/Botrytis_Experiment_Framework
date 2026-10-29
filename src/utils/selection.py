"""Politica de seleccion del detector y del clasificador (sec. 22): ``selection_policy.yaml``.

* Metrica primaria: mAP@0.5:0.95 (detector) / AP de ``con_signos`` (clasificador) sobre el validation externo.
* Restriccion opcional de recall minimo (``maximizar primaria sujeto a recall >= R_min``).
* Empates (diferencia <= ``tie_margin``) se resuelven con ``tie_breakers`` en orden (metrica, direccion).
* La politica se materializa UNA vez en ``artifacts/selection_policy.yaml`` (+ SHA-256) y despues no puede cambiar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from utils.checks import MethodologyViolation
from utils.config_loader import cfg_path
from utils.hashing import sha256_text_file, write_checksum_file


def policy_from_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Politica de seleccion declarada en la configuracion."""
    return {"version": 1, "detector": dict(cfg["selection"]["detector"]), "classifier": dict(cfg["selection"]["classifier"])}


def materialize_policy(cfg: Mapping[str, Any]) -> tuple[Path, str]:
    """Escribe ``artifacts/selection_policy.yaml`` la primera vez; despues exige que no haya cambiado.

    Raises:
        MethodologyViolation: Si el fichero existe y difiere de la configuracion actual.
    """
    path = cfg_path(cfg, "paths.artifacts_dir") / "selection_policy.yaml"
    text = yaml.safe_dump(policy_from_cfg(cfg), sort_keys=False, allow_unicode=True)
    if path.is_file():
        if yaml.safe_load(path.read_text(encoding="utf-8")) != yaml.safe_load(text):
            raise MethodologyViolation("La politica de seleccion cambio despues de materializarse (selection_policy.yaml): debe predeclararse ANTES de ver resultados (sec. 22.3)")
    else:
        path.write_text(text, encoding="utf-8", newline="\n")
        write_checksum_file(path, path.with_suffix(".sha256"))
    return path, sha256_text_file(path)


def rank_candidates(
    df: pd.DataFrame, *, primary: str, min_recall: float | None, recall_col: str, tie_margin: float, tie_breakers: Sequence[Mapping[str, str]],
) -> tuple[pd.DataFrame, str | None]:
    """Ordena candidatos y devuelve ``(tabla con rank/elegible/motivo, nombre ganador)``.

    Args:
        df: Una fila por arquitectura con ``architecture`` y las metricas.
        primary: Columna de la metrica primaria (mayor es mejor).
        min_recall: Recall minimo exigido sobre ``recall_col`` (``None`` = sin restriccion).
        tie_margin: Candidatos a ``tie_margin`` o menos de la mejor primaria se consideran empatados.
        tie_breakers: ``[{'metric': 'recall', 'direction': 'max'}, ...]`` aplicados en orden.
    """
    t = df.copy().reset_index(drop=True)
    t["eligible"] = True
    t["reason"] = ""
    if min_recall is not None:
        bad = t[recall_col].isna() | (t[recall_col] < float(min_recall))
        t.loc[bad, "eligible"] = False
        t.loc[bad, "reason"] = f"{recall_col} < {min_recall}"
    elig = t[t["eligible"]]
    if elig.empty:
        t["rank"] = range(1, len(t) + 1)
        return t, None
    top = float(elig[primary].max())
    tied = elig[elig[primary] >= top - float(tie_margin)].copy()
    for tb in tie_breakers:
        if len(tied) <= 1:
            break
        col, direction = tb["metric"], tb.get("direction", "max")
        vals = tied[col].astype(float)
        if vals.isna().all():
            continue
        best = vals.max() if direction == "max" else vals.min()
        tied = tied[vals == best]
    winner_row = tied.sort_values([primary], ascending=False).iloc[0]
    t.loc[t["architecture"] == winner_row["architecture"], "reason"] = "elegido" + (" (por desempate)" if len(elig[elig[primary] >= top - float(tie_margin)]) > 1 else " (mejor metrica primaria)")
    t = (t.assign(_w=(t["architecture"] == winner_row["architecture"]).astype(int))
         .sort_values(["eligible", "_w", primary], ascending=[False, False, False], kind="stable").reset_index(drop=True).drop(columns="_w"))
    t["rank"] = range(1, len(t) + 1)
    return t, str(winner_row["architecture"])


def assert_candidates_ready(cfg, task, expected):
    from utils.architectures import archs
    from utils.hpo_results import require_final_checkpoint,stage_json_path,load_best_config
    from utils.io import read_json
    for spec in archs(task):
        require_final_checkpoint(cfg,spec,expected)
        load_best_config(cfg,spec)
        for stage in ('outer_validation','inference_thresholds'):
            path=stage_json_path(cfg,stage,spec)
            if not path.is_file():raise FileNotFoundError(path)
            row=read_json(path)
            if any(row.get(k)!=v for k,v in expected.items()):raise MethodologyViolation('Candidato incompatible: '+spec.slug)
