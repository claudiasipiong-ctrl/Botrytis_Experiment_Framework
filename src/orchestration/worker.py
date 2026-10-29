"""Worker de una PC: toma de la cola el siguiente trabajo y lo ejecuta (UN trabajo GPU a la vez).

Uso (una terminal por PC)::

    python src/orchestration/worker.py --pc-id PC-01 --init-plan --wait
    python src/orchestration/worker.py --pc-id PC-02 --wait
    python src/orchestration/worker.py --pc-id PC-03 --wait

* Cada trabajo se ejecuta en un SUBPROCESO (aisla la memoria de GPU entre trabajos y deja su propio run).
* Paralelismo ENTRE arquitecturas/seeds; NUNCA tres trials simultaneos del mismo estudio (un estudio = un job).
* Tras un ``hpo`` COMPLETE se crean los jobs de robustez (top-3 x seeds), agregacion, entrenamiento
  externo, evaluacion y umbrales de esa arquitectura.
* Estados: PENDING / RUNNING / COMPLETE / FAILED (los trials PRUNED se registran en la tabla ``trials``).
* Sincronizacion sin servidor: ``--export-state`` / ``--import-state`` (ver ``orchestration/sync.py``).
"""

from __future__ import annotations

# Stdlib bootstrap: records import/CLI/config failures before optional dependencies.
import sys as _boot_sys
from pathlib import Path as _BootPath
_boot_root = _BootPath(__file__).resolve().parents[2]
_boot_sys.path.insert(0, str(_boot_root / "src"))
from utils import bootstrap as _bootstrap
_bootstrap.install(__file__, _boot_root)

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from orchestration.pc_registry import PCRegistry  # noqa: E402
from orchestration.study_queue import FAILED, KINDS, PENDING, RUNNING, Job, StudyQueue  # noqa: E402
from utils.architectures import get_arch  # noqa: E402
from utils.config_loader import cfg_path  # noqa: E402
from utils.dataset import recorded_dataset_hash  # noqa: E402
from utils.hashing import compute_code_hash  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.reproducibility import get_git_info  # noqa: E402
from utils.run_common import check_environment_matches_reference  # noqa: E402
from utils.splits import current_split_hash  # noqa: E402

SCRIPT = "worker"
SCRIPT_FOR_KIND = {
    "baseline": "06_run_baseline.py", "hpo": "07_run_hpo_study.py", "repeat": "08_repeat_top_configs.py", "aggregate": "08_repeat_top_configs.py",
    "outer_train": "09_train_outer_candidate.py", "outer_eval": "10_evaluate_outer_validation.py", "thresholds": "11_tune_inference_thresholds.py",
}
HEARTBEAT_SECONDS = 30.0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del worker."""
    g = parser.add_argument_group("worker")
    g.add_argument("--init-plan", action="store_true", help="crea baseline -> hpo solo para arquitecturas activas propias filtradas (idempotente)")
    g.add_argument("--wait", action="store_true", help="si no hay job listo pero otras PCs trabajan, espera en vez de salir")
    g.add_argument("--once", action="store_true", help="ejecuta a lo sumo un job")
    g.add_argument("--max-jobs", type=int, default=None, help="maximo de jobs a ejecutar")
    g.add_argument("--poll-seconds", type=float, default=60.0, help="espera entre consultas con --wait")
    g.add_argument("--kinds", nargs="*", choices=KINDS, default=None, help="solo estos tipos de job")
    g.add_argument("--architectures", nargs="*", default=None, help="solo estas arquitecturas")
    g.add_argument("--dry-run", action="store_true", help="solo muestra la cola y el siguiente job (no reclama ni ejecuta)")
    g.add_argument("--status", action="store_true", help="imprime el estado de la cola y termina")
    g.add_argument("--retry-failed", action="store_true", help="FAILED -> PENDING")
    g.add_argument("--recover-stale", action="store_true", help="RUNNING sin latido -> PENDING")
    g.add_argument("--force-recover", action="store_true", help="RUNNING de ESTA PC -> PENDING (tras un corte; verifique que no hay procesos vivos)")
    g.add_argument("--export-state", default=None, metavar="JSON", help="vuelca el estado de la cola a JSON")
    g.add_argument("--import-state", nargs="*", default=None, metavar="JSON", help="fusiona estados exportados por otras PCs")
    g.add_argument("--skip-env-check", action="store_true", help="no exigir entorno identico (solo desarrollo)")


def build_command(job: Job, pc_id: str, config: str, sets: Sequence[str], python: str | None = None, runs_dir: str | None = None) -> list[str]:
    """Linea de comandos del job (mismas ``--config``/``--set`` que el worker)."""
    cmd = [python or sys.executable, str(SRC / SCRIPT_FOR_KIND[job.kind]), "--architecture", job.architecture, "--pc-id", pc_id, "--config", config]
    for s in sets:
        cmd += ["--set", s]
    if runs_dir:
        cmd += ["--runs-dir", runs_dir]
    if job.kind == "repeat":
        cmd += ["--config-id", str(job.payload["config_id"]), "--seed", str(job.payload["seed"])]
    if job.kind == "aggregate":
        cmd += ["--aggregate-only"]
    return cmd


def top_config_ids(cfg: Mapping[str, Any], slug: str) -> list[str]:
    """``config_id`` de las top-k configuraciones del estudio (por valor, desempate por n.º de trial)."""
    from utils.hpo_results import top_configs
    from utils.optuna_storage import load_study

    study = load_study(get_arch(slug), cfg)
    return [c["config_id"] for c in top_configs(study, int(cfg["robustness"]["top_k_configs"]))]


def local_inputs_ready(cfg: Mapping[str, Any], job: Job, runs_base: Path) -> bool:
    """``True`` si los INSUMOS LOCALES del job estan en esta PC (sin servidor central los ficheros viajan en ZIPs; ver sync.py)."""
    from utils.hpo_results import best_config_path, final_checkpoint, stage_json_path
    from utils.optuna_storage import study_db_path

    spec = get_arch(job.architecture)
    from utils.weights import entry_for
    if cfg.get('initialization',{}).get('mode') == 'pretrained' and not entry_for(spec,cfg)['compatible_pretrained']:
        return False
    if job.kind == "repeat":
        return study_db_path(cfg, spec).is_file()
    if job.kind == "aggregate":
        if not study_db_path(cfg, spec).is_file():
            return False
        for dep in job.depends_on:
            _, _, cid, seed = dep.split(":")
            if not (runs_base / spec.slug / "robustness" / cid / f"seed_{seed[1:]}" / "result.json").is_file():
                return False
        return True
    if job.kind == "outer_train":
        return best_config_path(cfg, spec).is_file()
    if job.kind == "outer_eval":
        return final_checkpoint(cfg, spec).is_file()
    if job.kind == "thresholds":
        return stage_json_path(cfg, "outer_validation", spec).is_file()
    return True


def _provenance(cfg: Mapping[str, Any]) -> dict[str, Any]:
    root = SRC.parent
    art = cfg_path(cfg, "paths.artifacts_dir")
    git = get_git_info(root)
    return {"git_commit": git.get("commit"), "code_hash": compute_code_hash(root), "dataset_hash": recorded_dataset_hash(art),
            "split_hash": current_split_hash(art, str(cfg["outer_split"]["manifest_name"]))}


def _run_job(ctx: RunContext, queue: StudyQueue, job: Job, args: argparse.Namespace, log_dir: Path) -> int:
    """Ejecuta el job en un subproceso con latido; devuelve el codigo de salida."""
    cmd = build_command(job, ctx.pc_id, str(args.config), args.set, runs_dir=args.runs_dir)
    log_path = log_dir / (job.job_id.replace(":", "_").replace("/", "_") + f"_a{job.attempt}.log")
    ctx.logger.info("job %s (intento %d) -> %s", job.job_id, job.attempt, " ".join(cmd))
    with log_path.open("w", encoding="utf-8", errors="replace") as fh:
        env=_child_env()
        env['BOTRYTIS_WORKER_JOB_ID']=job.job_id
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        while True:
            try:
                rc = proc.wait(timeout=HEARTBEAT_SECONDS)
                break
            except subprocess.TimeoutExpired:
                queue.heartbeat(job.job_id)
    partial = False
    if rc != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
        queue.finish(job.job_id, FAILED, error=f"exit code {rc}\n{tail}", log_path=str(log_path))
    else:
        if job.kind == 'hpo':
            from utils.optuna_storage import load_study, budget_used
            from utils.architectures import trials_for
            if budget_used(load_study(get_arch(job.architecture),ctx.cfg)) < trials_for(get_arch(job.architecture),ctx.cfg):
                queue.mark_partial_block(job.job_id)
                partial = True
        # Computation is durable independently of publication; never report published on failure.
        queue.computed(job.job_id, published=False)
    if rc == 0:
        from orchestration.exchange import publish_architecture
        try:
            publish_architecture(ctx.cfg,ctx.pc_id,job.architecture,ctx.runs_base,stage=job.kind,complete=not partial)
        except Exception as exc:
            ctx.logger.exception('Computo terminado, PUBLICATION_PENDING; no repetir entrenamiento: %s', exc)
            return 4
        queue.computed(job.job_id, published=True)
    return 10 if partial else rc


def _child_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("YOLO_VERBOSE", "False")
    return env


def _main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Bucle del worker."""
    cfg, log = ctx.cfg, ctx.logger
    registry = PCRegistry(cfg)
    info = registry.validate(ctx.pc_id)
    queue = StudyQueue.from_cfg(cfg)
    log_dir = cfg_path(cfg, "paths.queue_dir") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.import_state:
        for p in args.import_state:
            res = queue.merge_state(p)
            log.info("estado importado de %s: %s", p, res)
            if res["conflicts"]:
                log.warning("CONFLICTOS (mismo job RUNNING en dos PCs): %s", res["conflicts"])
    if args.export_state:
        log.info("estado exportado a %s", queue.export_state(args.export_state))
    if args.retry_failed:
        log.info("FAILED -> PENDING: %d jobs", queue.retry_failed())
    if args.recover_stale or args.force_recover:
        rec = queue.recover_stale(minutes=float(cfg["pc_registry"]["stale_running_minutes"]), force=args.force_recover, pc_id=ctx.pc_id)
        log.info("jobs RUNNING huerfanos recuperados: %s", rec)
    selected = [get_arch(a) for a in args.architectures] if args.architectures is not None else [get_arch(s) for s in registry.owned_slugs(ctx.pc_id)]
    target = cfg.get('experiment',{}).get('architecture')
    if target: selected = [a for a in selected if a.slug == target]
    if any(registry.owner(a.slug) != ctx.pc_id for a in selected):
        raise RuntimeError('Arquitectura pertenece a otra PC o esta en transferencia')
    if args.init_plan:
        log.info("jobs nuevos en el plan: %d", queue.plan(cfg, selected))
    if args.status or args.dry_run:
        table = queue.table()
        log.info("cola de trabajos:\n%s", table.to_string(index=False) if len(table) else "(vacia)")
        ctx.log_metrics({"queue_counts": queue.counts()})
        if args.status:
            return

    if not args.skip_env_check and not args.dry_run:
        check_environment_matches_reference(ctx, strict=True)
        mism = registry.environment_mismatches()
        if mism:
            raise RuntimeError(f"Los entornos de las PCs difieren en campos criticos: {mism}. Resuelvalo antes de lanzar (sec. 16.6).")
        missing = registry.missing_environments()
        if missing:
            log.warning("PCs sin environment_pc*.json (aun no ejecutaron 00_check_environment.py): %s", missing)

    done = 0
    max_global = int(cfg["compute"]["max_gpu_jobs_global"])
    if args.dry_run: return
    from utils.manual_gates import require_gate
    require_gate(cfg, ctx.pc_id, "READY_FOR_EXPERIMENT")
    assigned = set(registry.owned_slugs(ctx.pc_id))
    archs_filter = [a.slug for a in selected]
    if not set(archs_filter) <= assigned:
        raise RuntimeError("Arquitectura asignada a otra PC; transferencia de propiedad requiere decision manual predeclarada")
    while args.max_jobs is None or done < args.max_jobs:
        if args.dry_run:
            break
        job = queue.claim_next(ctx.pc_id, preferred_slugs=info.preferred_slugs, kinds=args.kinds, slugs=archs_filter,
                               max_global_running=max_global, provenance=_provenance(cfg), ready_check=lambda j: local_inputs_ready(cfg, j, ctx.runs_base))
        if job is None:
            if args.wait and queue.counts()[RUNNING] > 0:  # otra PC (o este mismo ciclo) aun puede liberar dependencias
                log.info("sin job listo (dependencias/otras PCs en ejecucion); espero %.0fs...", args.poll_seconds)
                time.sleep(args.poll_seconds)
                continue
            blocked = [j.job_id for j in queue.jobs(PENDING)]
            log.info("no quedan jobs ejecutables para %s%s", ctx.pc_id, f" ({len(blocked)} PENDING sin insumos locales o bloqueados por FAILED: descomprima los ZIP de resultados de las otras PCs con sync.py unpack, o use --retry-failed)" if blocked else "")
            break
        rc = _run_job(ctx, queue, job, args, log_dir)
        done += 1
        if rc == 0 and job.kind == "hpo":
            ids = top_config_ids(cfg, job.architecture)
            log.info("HPO de %s completo; top configuraciones %s -> %d jobs nuevos", job.architecture, ids,
                     queue.expand_after_hpo(cfg, get_arch(job.architecture), ids))
        elif rc == 10:
            log.info('Bloque HPO terminado y pendiente de revision; presupuesto total intacto')
        elif rc != 0:
            log.error("job %s FALLO (codigo %d); ver %s", job.job_id, rc, log_dir)
        queue.export_state(cfg_path(cfg, "paths.queue_dir") / f"state_{ctx.pc_id.lower().replace('-', '')}.json")
        if args.once:
            break
    ctx.log_metrics({"jobs_ejecutados": done, "queue_counts": queue.counts()})


def main(ctx, args):
    from utils.durable import FileLock
    from utils.guards import gpu_lock_path
    if args.status or args.dry_run:
        return _main(ctx,args)
    with FileLock(gpu_lock_path(ctx.cfg,ctx.pc_id).with_suffix('.worker.guard')):
        return _main(ctx,args)


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments))
