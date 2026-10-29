"""00 - Verificacion del entorno de computo (una vez por PC, antes de cualquier otra etapa).

Que hace (Especificacion s2, sec. 16.6, 31, 41.4):

* identifica la PC y registra GPU/CPU/RAM, driver, CUDA, PyTorch y versiones de paquetes;
* comprueba hash del codigo y hash del dataset (si ya fue validado);
* escribe ``artifacts/environments/environment_<pc>.json`` (p. ej. ``environment_pc01.json``);
* compara los campos criticos con el entorno de referencia y con las demas PCs;
* **aborta (exit 1)** si la configuracion critica difiere, salvo ``--no-strict``;
* lista los TODO BLOQUEANTES aun pendientes en la configuracion.

Ejemplos::

    python src/00_check_environment.py --pc-id PC-01 --write-reference
    python src/00_check_environment.py --pc-id PC-02            # compara contra la referencia
    python src/00_check_environment.py --pc-id PC-03 --require-gpu --gpu-selftest
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
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config_loader import cfg_path, get_nested, list_todos, resolve_path  # noqa: E402
from utils.dataset import compute_dataset_hash, recorded_dataset_hash  # noqa: E402
from utils.io import ensure_dir, read_json, write_json  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.reproducibility import compare_environments  # noqa: E402

SCRIPT = "00_check_environment"
CORE_PACKAGES = ("numpy", "pandas", "pillow", "pyyaml", "matplotlib")
ML_PACKAGES = ("torch", "torchvision", "timm", "ultralytics", "optuna")
MIN_FREE_GB = 5.0
HASH_FIELDS = ("git_commit", "code_hash", "dataset_hash", "split_hash")


class EnvironmentCheckFailed(RuntimeError):
    """Hay comprobaciones criticas fallidas o el entorno difiere del de referencia."""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    g = parser.add_argument_group("00_check_environment")
    g.add_argument("--write-reference", action="store_true", help="guarda este entorno como referencia para las demas PCs")
    g.add_argument("--no-compare", action="store_true", help="no comparar con la referencia ni con otras PCs")
    g.add_argument("--no-strict", action="store_true", help="las diferencias criticas son WARNING en vez de error")
    g.add_argument("--require-gpu", action="store_true", help="falla si no hay GPU CUDA utilizable")
    g.add_argument("--require-ml", action="store_true", help="falla si faltan torch/torchvision/timm/ultralytics/optuna")
    g.add_argument("--gpu-selftest", action="store_true", help="ejecuta una multiplicacion de matrices FP16 en la GPU")
    g.add_argument("--skip-dataset-hash", action="store_true", help="no recalcular el hash del dataset")


def _pc_file_token(pc_id: str) -> str:
    """``PC-01`` -> ``pc01`` (nombre de fichero ``environment_pc01.json``)."""
    return re.sub(r"[^a-z0-9]", "", pc_id.lower()) or "unknown"


class _Checks:
    """Acumulador de comprobaciones PASS/WARN/FAIL."""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.rows: list[dict[str, str]] = []

    def add(self, name: str, status: str, detail: str) -> None:
        """Registra una comprobacion (``PASS`` | ``WARN`` | ``FAIL``) y la escribe en el log."""
        self.rows.append({"check": name, "status": status, "detail": detail})
        log = {"PASS": self.ctx.logger.info, "WARN": self.ctx.logger.warning, "FAIL": self.ctx.logger.error}[status]
        log("[%s] %s: %s", status, name, detail)

    def count(self, status: str) -> int:
        """Numero de comprobaciones con ese estado."""
        return sum(1 for r in self.rows if r["status"] == status)


def _writable(path: Path) -> bool:
    try:
        ensure_dir(path)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _gpu_selftest(ctx: RunContext, checks: _Checks) -> None:
    try:
        import torch  # type: ignore[import-not-found]

        if not torch.cuda.is_available():
            checks.add("gpu_selftest", "FAIL", "torch.cuda.is_available() = False")
            return
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        c = a @ b
        torch.cuda.synchronize()
        ok = bool(torch.isfinite(c).all().item())
        checks.add("gpu_selftest", "PASS" if ok else "FAIL", f"matmul FP16 2048x2048 {'finito' if ok else 'con NaN/inf'}")
    except Exception as exc:
        checks.add("gpu_selftest", "FAIL", f"{type(exc).__name__}: {exc}")


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Ejecuta todas las comprobaciones y escribe los ficheros de entorno."""
    cfg, env, log = ctx.cfg, ctx.env, ctx.logger
    checks = _Checks(ctx)
    strict = not args.no_strict
    critical = list(get_nested(cfg, "compute.critical_environment_fields", default=[]))
    env_dir = ensure_dir(cfg_path(cfg, "paths.environments_dir"))
    require_gpu = bool(args.require_gpu or get_nested(cfg, "compute.require_gpu", default=False))

    # -- software base -------------------------------------------------------------------
    py = tuple(int(x) for x in str(env.get("python_version", "0.0")).split(".")[:2])
    checks.add("python", "PASS" if py >= (3, 10) else "FAIL", f"Python {env.get('python_version')} (minimo 3.10)")
    pkgs = env.get("packages", {})
    for name in CORE_PACKAGES:
        checks.add(f"paquete:{name}", "PASS" if pkgs.get(name) else "FAIL", f"version={pkgs.get(name)}")
    for name in ML_PACKAGES:
        present = bool(pkgs.get(name))
        status = "PASS" if present else ("FAIL" if args.require_ml else "WARN")
        checks.add(f"paquete:{name}", status, f"version={pkgs.get(name)}" + ("" if present else " (no instalado; requerido desde 06)"))

    # -- identificacion ------------------------------------------------------------------
    checks.add("pc_id", "PASS" if ctx.pc_id_explicit else "WARN",
               f"{ctx.pc_id}" + ("" if ctx.pc_id_explicit else " (usa --pc-id o BOTRYTIS_PC_ID; las 3 PCs deben tener ID propio)"))

    # -- GPU / CUDA ---------------------------------------------------------------------
    n_gpu = int(env.get("gpu_count") or 0)
    if n_gpu > 0:
        checks.add("gpu", "PASS", f"{env.get('gpu_model')} | {env.get('gpu_vram_mb')} MB | driver {env.get('nvidia_driver')} | CUDA driver API {env.get('cuda_driver_api')}")
        if n_gpu > 1:
            checks.add("gpu_count", "WARN", f"{n_gpu} GPUs detectadas; el protocolo asume 1 GPU por PC (sec. 2 Estrategia 3 PCs)")
    else:
        checks.add("gpu", "FAIL" if require_gpu else "WARN", "no se detecto GPU NVIDIA (nvidia-smi)")
    if env.get("torch"):
        cuda_ok = bool(env.get("cuda_available_torch"))
        status = "PASS" if cuda_ok else ("FAIL" if require_gpu else "WARN")
        checks.add("cuda_torch", status, f"torch {env.get('torch')} | CUDA runtime {env.get('cuda_runtime')} | cuDNN {env.get('cudnn')} | disponible={cuda_ok}")
    if args.gpu_selftest:
        _gpu_selftest(ctx, checks)

    # -- git / codigo --------------------------------------------------------------------
    if env.get("git_commit"):
        checks.add("git_commit", "WARN" if env.get("git_dirty") else "PASS",
                   f"{env.get('git_commit')} (rama {env.get('git_branch')})" + (" con cambios sin commitear" if env.get("git_dirty") else ""))
    else:
        checks.add("git_commit", "WARN", "sin commit git; la trazabilidad de codigo usa code_hash")
    checks.add("code_hash", "PASS" if ctx.hashes.get("code_hash") else "FAIL", str(ctx.hashes.get("code_hash")))

    # -- rutas y espacio -----------------------------------------------------------------
    writable_targets = {
        "paths.artifacts_dir": cfg_path(cfg, "paths.artifacts_dir"),
        "runs_dir (efectivo)": ctx.run_dir.parent,
        "paths.figures_dir": cfg_path(cfg, "paths.figures_dir"),
        "paths.data_root": cfg_path(cfg, "paths.data_root"),
    }
    for key, p in writable_targets.items():
        checks.add(f"escritura:{key}", "PASS" if _writable(p) else "FAIL", str(p))
    free_gb = shutil.disk_usage(cfg_path(cfg, "paths.data_root")).free / 1024 ** 3
    checks.add("espacio_disco", "PASS" if free_gb >= MIN_FREE_GB else "WARN", f"{free_gb:.1f} GB libres (minimo recomendado {MIN_FREE_GB} GB)")

    # -- dataset -------------------------------------------------------------------------
    recorded = recorded_dataset_hash(cfg_path(cfg, "paths.artifacts_dir"))
    if args.skip_dataset_hash:
        checks.add("dataset_hash", "WARN", "omitido (--skip-dataset-hash)")
    elif recorded is None:
        checks.add("dataset_hash", "WARN", "dataset aun no validado (ejecute 01_validate_dataset.py)")
    else:
        try:
            current, _ = compute_dataset_hash(cfg)
            if current == recorded:
                checks.add("dataset_hash", "PASS", current)
            else:
                checks.add("dataset_hash", "FAIL", f"el dataset local ({current[:12]}...) difiere del validado ({recorded[:12]}...)")
        except Exception as exc:
            checks.add("dataset_hash", "FAIL", f"no se pudo recalcular: {type(exc).__name__}: {exc}")
    if env.get("split_hash"):
        checks.add("split_hash", "PASS", str(env["split_hash"]))
    else:
        checks.add("split_hash", "WARN", "split externo aun no creado (02_create_outer_split.py)")

    # -- TODO bloqueantes pendientes -----------------------------------------------------
    todos = list_todos(cfg)
    checks.add("todos_bloqueantes", "WARN" if todos else "PASS", f"{len(todos)} pendientes en la configuracion")
    for t in todos:
        log.info("   TODO pendiente: %s", t)

    # -- escritura de environment_<pc>.json ----------------------------------------------
    env_file = env_dir / f"environment_{_pc_file_token(ctx.pc_id)}.json"
    write_json(env_file, env)
    ctx.register_artifact(env_file, "environment de esta PC")
    log.info("entorno de esta PC escrito en %s", env_file)

    ref_path = resolve_path(get_nested(cfg, "compute.reference_environment", default="artifacts/environments/environment_reference.json"))
    if args.write_reference:
        write_json(ref_path, env)
        ctx.register_artifact(ref_path, "entorno de referencia")
        checks.add("referencia", "PASS", f"entorno de referencia escrito en {ref_path}")

    # -- comparacion entre PCs -----------------------------------------------------------
    mismatches: list[dict[str, Any]] = []
    if not args.no_compare:
        if ref_path.is_file() and not args.write_reference:
            ref = read_json(ref_path)
            # un hash que la referencia aun no tenia (p. ej. split creado despues de escribirla) no es una discrepancia
            diffs = [d for d in compare_environments(ref, env, critical) if ref.get(d["field"]) is not None or d["field"] not in HASH_FIELDS]
            for d in diffs:
                d["against"] = "reference"
            mismatches += diffs
            checks.add("comparacion_referencia", "PASS" if not diffs else "FAIL",
                       "coincide en todos los campos criticos" if not diffs else f"{len(diffs)} campos difieren: {[d['field'] for d in diffs]}")
        elif not args.write_reference:
            checks.add("comparacion_referencia", "WARN", f"no existe {ref_path}; ejecute una PC con --write-reference")
        peers = sorted(p for p in env_dir.glob("environment_pc*.json") if p != env_file)
        pc_fields = [f for f in critical if f not in ("git_commit", "code_hash", "dataset_hash", "split_hash")]
        for peer in peers:
            other = read_json(peer)
            diffs = compare_environments(other, env, pc_fields)
            for d in diffs:
                d["against"] = peer.name
            mismatches += diffs
            checks.add(f"comparacion:{peer.name}", "PASS" if not diffs else "FAIL",
                       "coincide" if not diffs else f"difiere en {[d['field'] for d in diffs]}")

    for m in mismatches:
        log.error("DIFERENCIA critica vs %s: %s  referencia=%r  esta_pc=%r", m["against"], m["field"], m["reference"], m["candidate"])

    # -- resumen -------------------------------------------------------------------------
    n_fail = checks.count("FAIL")
    if not strict:
        for r in checks.rows:
            if r["status"] == "FAIL" and r["check"].startswith(("comparacion", "dataset_hash")):
                r["status"] = "WARN"
        n_fail = checks.count("FAIL")
    ctx.log_metrics({
        "n_pass": checks.count("PASS"), "n_warn": checks.count("WARN"), "n_fail": n_fail,
        "checks": checks.rows, "environment_mismatches": mismatches,
        "environment_hash": env.get("environment_hash"), "pending_todos": todos, "strict": strict,
    })
    if n_fail:
        raise EnvironmentCheckFailed(
            f"{n_fail} comprobacion(es) critica(s) fallida(s): "
            + ", ".join(r["check"] for r in checks.rows if r["status"] == "FAIL")
            + ". No lance HPO/entrenamiento desde esta PC hasta resolverlo."
        )
    log.info("Entorno OK: %d PASS, %d WARN, 0 FAIL", checks.count("PASS"), checks.count("WARN"))


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments))
