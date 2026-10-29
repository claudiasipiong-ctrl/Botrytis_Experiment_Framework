"""Logging cientifico y trazabilidad obligatoria de TODA ejecucion.

Cada ejecucion crea ``runs/<architecture>/<trial>/`` (para los scripts que no pertenecen a una
arquitectura: ``runs/_pipeline/<YYYYmmdd_HHMMSS>_<script>/``) con::

    execution.log           log completo (UTF-8), incluye tracebacks completos
    config.yaml             configuracion efectiva (tras overrides e interpolacion)
    environment.json        PC, GPU, CUDA, PyTorch, versiones, git, hashes
    metrics.json            metricas acumuladas con ``ctx.log_metrics``
    checkpoint_metadata.json  checkpoints producidos (ruta, sha256, tamano) + los 4 hashes
    metadata.json           fechas/horas, PC ID, seed, hashes, parametros, estado, error
    figures/                figuras de la ejecucion (+ ``<figura>.provenance.json`` y ``.data.csv``)

``metadata.json`` se escribe ya al INICIO con ``status = RUNNING`` y se reescribe al final
(``COMPLETE`` / ``FAILED`` / ``PRUNED`` / ``INTERRUPTED``), de modo que incluso un corte abrupto
deja rastro. Incluso los fallos de argumentos o de configuracion (antes de que exista ``cfg``)
generan run.

Uso tipico en un script::

    def main(ctx: RunContext, args: argparse.Namespace) -> None:
        ctx.logger.info("hola")
        ctx.log_metrics({"n_images": len(images)})

    if __name__ == "__main__":
        raise SystemExit(run_cli("01_validate_dataset", main, description="..."))
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
import traceback
import json
import uuid
import threading
import contextvars
ACTIVE_CONFIG = contextvars.ContextVar("botrytis_active_config", default=None)
ACTIVE_RUN_DIR = contextvars.ContextVar("botrytis_active_run_dir", default=None)
ACTIVE_CONTEXT = contextvars.ContextVar("botrytis_active_context", default=None)
def log_epoch(metrics, **tags):
    ctx=ACTIVE_CONTEXT.get()
    if ctx is not None:ctx.emit_event("epoch_metrics",metrics=dict(metrics),**tags)
    else:bootstrap_event("epoch_metrics",metrics=dict(metrics),**tags)
from utils.bootstrap import redact, event as bootstrap_event, set_exit
from utils.durable import FileLock
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from utils.config_loader import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    ConfigError,
    dump_config_yaml,
    get_nested,
    load_config,
    resolve_path,
)
from utils.hashing import compute_code_hash, sha256_json
from utils.semantics import semantic_fingerprint
from utils.io import ensure_dir, write_json
from utils.reproducibility import collect_environment, resolve_pc_id, set_global_seed

RUN_FILES: tuple[str, ...] = (
    "execution.log", "events.jsonl", "config.yaml", "environment.json", "metrics.json", "checkpoint_metadata.json", "metadata.json",
)
#: Directorio (bajo ``runs/``) de los runs que no pertenecen a ninguna arquitectura.
PIPELINE_DIR = "_pipeline"
#: Nombres de clase de excepcion que se registran como estado ``PRUNED`` (no son errores).
_PRUNED_EXCEPTIONS = {"TrialPruned"}
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class CliArgumentError(Exception):
    """Error de argumentos de linea de comandos (capturado para dejar traza en ``runs/``)."""


class _TraceableParser(argparse.ArgumentParser):
    """``ArgumentParser`` que lanza excepcion en vez de ``sys.exit(2)`` ante argumentos invalidos."""

    def error(self, message: str) -> None:  # type: ignore[override]
        """Lanza :class:`CliArgumentError` en lugar de terminar el proceso."""
        raise CliArgumentError(f"{self.prog}: error de argumentos: {message}")


class _CountingHandler(logging.Handler):
    """Cuenta registros WARNING/ERROR (para ``metadata.json``)."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.warnings = 0
        self.errors = 0

    def emit(self, record: logging.LogRecord) -> None:
        """Incrementa el contador correspondiente al nivel del registro."""
        if record.levelno >= logging.ERROR:
            self.errors += 1
        else:
            self.warnings += 1


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "run"


def _deep_update(base: MutableMapping[str, Any], new: Mapping[str, Any]) -> None:
    for k, v in new.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), MutableMapping):
            _deep_update(base[k], v)
        else:
            base[k] = v


class RunContext:
    """Contexto de una ejecucion: directorio, logger, metricas y metadatos.

    Se usa como gestor de contexto (``with RunContext(...) as ctx``). Toda la
    informacion obligatoria queda escrita aunque la ejecucion falle.

    Attributes:
        cfg: Configuracion efectiva.
        run_dir: Directorio del run.
        figures_dir: ``run_dir / 'figures'``.
        logger: Logger del run (consola + ``execution.log``).
        metrics: Metricas acumuladas.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
        script_name: str,
        *,
        cli_args: Mapping[str, Any] | None = None,
        argv: Sequence[str] | None = None,
        experiment_name: str | None = None,
        pc_id: str | None = None,
        runs_dir: str | os.PathLike[str] | None = None,
        architecture: str | None = None,
        trial_name: str | None = None,
        reuse_existing: bool = False,
        env_override: Mapping[str, Any] | None = None,
    ) -> None:
        """Crea el run.

        Args:
            architecture: Slug de arquitectura (``yolov8n``...). ``None`` => ``runs/_pipeline``.
            trial_name: Subruta dentro de la arquitectura (``hpo/trial_003``, ``baseline``...). Si se
                omite, se usa ``<timestamp>_<experimento>``.
            reuse_existing: Reutiliza ``trial_name`` si ya existe (reanudacion tras un corte); el log
                se anexa y el resto de ficheros se reescribe.
            env_override: ``environment.json`` ya recolectado por el proceso padre (evita repetir
                nvidia-smi/pip en cada trial).
        """
        self.cfg: dict[str, Any] = dict(cfg)
        self.script_name = script_name
        self.experiment_name = _slug(experiment_name or script_name)
        self.architecture = architecture
        self.cli_args: dict[str, Any] = dict(cli_args or {})
        self.argv: list[str] = list(argv if argv is not None else sys.argv)
        self.metrics: dict[str, Any] = {}
        from utils.experiments import identity
        self.extra_metadata: dict[str, Any] = {'experiment': cfg.get('experiment',{}), 'experiment_identity': identity(cfg) if 'classes' in cfg and cfg.get('paths',{}).get('artifacts_dir') else {'state':'CONFIGURATION_INCOMPLETE'}}
        self.artifacts: list[dict[str, str]] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.checkpoint_extra: dict[str, Any] = {}
        self.splits_accessed: list[dict[str, str]] = []
        self.status = "RUNNING"
        self.status_override: str | None = None
        self.error: dict[str, Any] | None = None
        self.seed: int | None = None
        self.seed_info: dict[str, Any] | None = None
        self._env_override = dict(env_override) if env_override is not None else None

        configured_pc = pc_id or get_nested(self.cfg, "compute.pc_id", default=None)
        self.pc_id, self.pc_id_explicit = resolve_pc_id(configured_pc)

        self._t0 = time.perf_counter()
        self.started_at = datetime.now(timezone.utc)
        self.execution_uuid = uuid.uuid4().hex
        self.finished_at: datetime | None = None

        base = resolve_path(runs_dir) if runs_dir else resolve_path(get_nested(self.cfg, "paths.runs_dir", default="runs"))
        self.runs_base: Path = base
        self.run_dir = self._create_run_dir(base, trial_name, reuse_existing)
        self.run_id = "/".join(self.run_dir.relative_to(base).parts) if self.run_dir.is_relative_to(base) else self.run_dir.name
        self.run_id = self.started_at.strftime("%Y%m%dT%H%M%S.%fZ") + "_" + _slug(self.pc_id) + "_" + _slug(script_name) + "_" + _slug(architecture or "pipeline") + "_" + _slug(trial_name or "none") + "_" + self.execution_uuid
        self._run_lock = FileLock(self.run_dir / ".execution.lock")
        self._run_lock.__enter__()
        # Reusing a stable artifact location preserves immutable snapshots of all previous evidence.
        old_files = [p for p in self.run_dir.iterdir() if p.name not in (".execution.lock", "previous_attempts")]
        if old_files:
            import shutil
            previous = ensure_dir(self.run_dir / "previous_attempts" / self.execution_uuid)
            for p in old_files:
                if p.is_dir(): shutil.copytree(p, previous / p.name)
                else: shutil.copy2(p, previous / p.name)
        self.figures_dir = ensure_dir(self.run_dir / "figures")
        for sub in get_nested(self.cfg, "logging.run_subdirs", default=[]) or []:
            ensure_dir(self.run_dir / str(sub))

        self._counter = _CountingHandler()
        self.logger = self._setup_logger()
        self.env: dict[str, Any] = {}
        self.hashes: dict[str, str | None] = {"dataset_hash": None, "split_hash": None, "code_hash": None}
        self._started = False
        self._event_lock = threading.RLock()
        self.emit_event("run_created", script=script_name, pc_id=self.pc_id, architecture=architecture, trial=trial_name, command=self.argv)
        write_json(self.run_dir / "metadata.json", {"run_id":self.run_id,"status":"RUNNING","started_at_utc":self.started_at.isoformat()})

    def emit_event(self, kind, **data):
        with self._event_lock:
            with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(redact({"utc":datetime.now(timezone.utc).isoformat(), "run_id":self.run_id, "event":kind, **data}), default=str, ensure_ascii=False) + "\n")
                f.flush()
        bootstrap_event(kind, context_run=self.run_id, **data)

    # ------------------------------------------------------------------ setup
    def _create_run_dir(self, base: Path, trial_name: str | None, reuse_existing: bool) -> Path:
        parent = ensure_dir(base / (self.architecture or PIPELINE_DIR))
        if trial_name:
            target = parent / trial_name
            if reuse_existing:
                return ensure_dir(target)
            name = trial_name
        else:
            name = f"{self.started_at.strftime('%Y%m%d_%H%M%S')}_{self.experiment_name}"
        for i in range(0, 1000):
            candidate = parent / (name if i == 0 else f"{name}_{i + 1}")
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            except FileExistsError:
                continue
        raise RuntimeError(f"No se pudo crear un directorio de run unico bajo {parent}")

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("botrytis." + self.run_id.replace("/", "."))
        logger.setLevel(getattr(logging, str(get_nested(self.cfg, "logging.level", default="INFO")).upper(), logging.INFO))
        logger.propagate = False
        for h in list(logger.handlers):
            logger.removeHandler(h)
        class RedactFilter(logging.Filter):
            def filter(self, rec):
                rec.msg = redact(rec.getMessage()); rec.args=()
                return True
        class EventHandler(logging.Handler):
            def emit(handler, rec):
                self.emit_event("log",level=rec.levelname,message=redact(rec.getMessage()))
        logger.addFilter(RedactFilter())
        logger.addHandler(EventHandler())
        fmt = logging.Formatter(_LOG_FORMAT)
        fh = logging.FileHandler(self.run_dir / "execution.log", mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        if get_nested(self.cfg, "logging.console", default=True):
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            logger.addHandler(sh)
        logger.addHandler(self._counter)
        logging.captureWarnings(True)
        warn_logger = logging.getLogger("py.warnings")
        if fh not in warn_logger.handlers:
            warn_logger.addHandler(fh)
        return logger

    def _detect_hashes(self) -> None:
        from utils.dataset import recorded_dataset_hash
        from utils.splits import current_split_hash

        artifacts = resolve_path(get_nested(self.cfg, "paths.artifacts_dir", default="artifacts"))
        manifest_name = get_nested(self.cfg, "outer_split.manifest_name", default="split_manifest.csv")
        try:
            self.hashes["code_hash"] = compute_code_hash(PROJECT_ROOT)
        except Exception as exc:  # no bloquea la ejecucion, pero queda registrado
            self.logger.warning("No se pudo calcular code_hash: %s", exc)
        if self.hashes["dataset_hash"] is None:
            try:
                self.hashes["dataset_hash"] = recorded_dataset_hash(artifacts)
            except Exception as exc:
                self.logger.warning("No se pudo leer dataset_hash registrado: %s", exc)
        if self.hashes["split_hash"] is None:
            try:
                self.hashes["split_hash"] = current_split_hash(artifacts, str(manifest_name))
            except Exception as exc:
                self.logger.warning("No se pudo leer split_hash registrado: %s", exc)

    def __enter__(self) -> "RunContext":
        self._started = True
        self._cfg_token=ACTIVE_CONFIG.set(self.cfg)
        self._dir_token=ACTIVE_RUN_DIR.set(self.run_dir)
        self._context_token=ACTIVE_CONTEXT.set(self)
        import contextlib
        ctx = self
        class ContextStream:
            def __init__(self, stream): self.stream=stream;self.pending=""
            def write(self, text):
                self.pending+=str(text)
                while "\n" in self.pending:
                    line,self.pending=self.pending.split("\n",1)
                    clean=redact(line+"\n")
                    with ctx._event_lock:
                        with (ctx.run_dir / "execution.log").open("a", encoding="utf-8") as f:f.write(clean);f.flush()
                    self.stream.write(clean)
                return len(text)
            def flush(self):
                if self.pending:
                    clean=redact(self.pending);self.pending=""
                    with (ctx.run_dir / "execution.log").open("a",encoding="utf-8") as f:f.write(clean);f.flush()
                    self.stream.write(clean)
                self.stream.flush()
            def __getattr__(self,n):return getattr(self.stream,n)
        self._stdout_scope=contextlib.redirect_stdout(ContextStream(sys.stdout))
        self._stderr_scope=contextlib.redirect_stderr(ContextStream(sys.stderr))
        self._stdout_scope.__enter__();self._stderr_scope.__enter__()
        # 1) config.yaml primero: si algo falla despues, la config ya esta guardada.
        try:
            dump_config_yaml(self.cfg, self.run_dir / "config.yaml")
        except Exception:
            self.logger.exception("No se pudo escribir config.yaml")
        # 2) entorno + hashes
        self._detect_hashes()
        if self._env_override is not None:
            self.env = {**self._env_override, "pc_id": self.pc_id, "dataset_hash": self.hashes["dataset_hash"],
                        "split_hash": self.hashes["split_hash"], "code_hash": self.hashes["code_hash"]}
        else:
            try:
                self.env = collect_environment(
                    self.cfg, PROJECT_ROOT, pc_id=self.pc_id, dataset_hash=self.hashes["dataset_hash"],
                    split_hash=self.hashes["split_hash"], code_hash=self.hashes["code_hash"],
                )
            except Exception:
                self.logger.exception("No se pudo recolectar el entorno completo")
                self.env = {"pc_id": self.pc_id, "collection_error": traceback.format_exc()}
        # 3) seed global por defecto (los scripts pueden sobreescribirla con ctx.set_seed)
        default_seed = get_nested(self.cfg, "reproducibility.seed", default=None)
        if isinstance(default_seed, int) and not isinstance(default_seed, bool):
            self.set_seed(default_seed)
        self._log_banner()
        self._write_all(final=False)
        return self

    def _log_banner(self) -> None:
        env, log = self.env, self.logger
        log.info("=" * 78)
        log.info("RUN %s", self.run_id)
        log.info("script=%s  experimento=%s", self.script_name, self.experiment_name)
        log.info("fecha=%s  hora_inicio=%s", self.started_at.strftime("%Y-%m-%d"), self.started_at.strftime("%H:%M:%S%z"))
        log.info("PC ID=%s%s", self.pc_id, "" if self.pc_id_explicit else "  (NO definido: usando hostname; fije BOTRYTIS_PC_ID o compute.pc_id)")
        log.info("GPU=%s  VRAM_MB=%s  driver=%s", env.get("gpu_model"), env.get("gpu_vram_mb"), env.get("nvidia_driver"))
        log.info("CUDA runtime(torch)=%s  CUDA driver API=%s  cuDNN=%s", env.get("cuda_runtime"), env.get("cuda_driver_api"), env.get("cudnn"))
        log.info("PyTorch=%s  torchvision=%s  timm=%s  ultralytics=%s  optuna=%s",
                 env.get("torch"), env.get("torchvision"), env.get("timm"), env.get("ultralytics"), env.get("optuna"))
        log.info("Python=%s  SO=%s %s", env.get("python_version"), env.get("os_system"), env.get("os_release"))
        log.info("git_commit=%s  dirty=%s", env.get("git_commit"), env.get("git_dirty"))
        log.info("code_hash=%s", self.hashes.get("code_hash"))
        log.info("dataset_hash=%s", self.hashes.get("dataset_hash"))
        log.info("split_hash=%s", self.hashes.get("split_hash"))
        log.info("seed=%s", self.seed)
        log.info("environment_hash=%s", env.get("environment_hash"))
        log.info("config_files=%s  overrides=%s", get_nested(self.cfg, "_meta.config_files", default=None),
                 get_nested(self.cfg, "_meta.overrides", default=None))
        log.info("parametros CLI=%s", self.cli_args)
        log.info("=" * 78)
        if not self.env.get("git_commit"):
            log.warning("Sin commit git disponible (el proyecto no es un repositorio propio); la trazabilidad de codigo usa code_hash.")
        elif self.env.get("git_dirty"):
            log.warning("El arbol git tiene cambios sin commitear: el commit NO describe exactamente el codigo ejecutado.")

    # --------------------------------------------------------------- API publica
    def set_seed(self, seed: int) -> None:
        """Aplica y registra la semilla del run (Python, NumPy, PyTorch, CUDA)."""
        rep = self.cfg.get("reproducibility", {}) or {}
        self.seed = int(seed)
        self.seed_info = set_global_seed(
            int(seed),
            deterministic=bool(rep.get("deterministic", True)),
            cudnn_deterministic=bool(rep.get("cudnn_deterministic", True)),
            cudnn_benchmark=bool(rep.get("cudnn_benchmark", False)),
        )
        self.logger.info("seed aplicada=%s  detalle=%s", self.seed, self.seed_info)

    def set_hashes(self, *, dataset_hash: str | None = None, split_hash: str | None = None) -> None:
        """Actualiza los hashes registrados (p. ej. tras crear el split)."""
        if dataset_hash is not None:
            self.hashes["dataset_hash"] = dataset_hash
            self.env["dataset_hash"] = dataset_hash
        if split_hash is not None:
            self.hashes["split_hash"] = split_hash
            self.env["split_hash"] = split_hash
        self.logger.info("hashes registrados: dataset_hash=%s split_hash=%s", self.hashes["dataset_hash"], self.hashes["split_hash"])

    def log_metrics(self, metrics: Mapping[str, Any], *, prefix: str | None = None) -> None:
        """Acumula metricas (fusion profunda) y las vuelca a ``metrics.json`` de inmediato."""
        payload: Mapping[str, Any] = {prefix: dict(metrics)} if prefix else metrics
        _deep_update(self.metrics, payload)
        text = str(dict(payload))
        self.logger.info("metricas: %s", text if len(text) <= 600 else text[:600] + f"... [+{len(text) - 600} caracteres; ver metrics.json]")
        try:
            write_json(self.run_dir / "metrics.json", self.metrics)
        except Exception:
            self.logger.exception("No se pudo escribir metrics.json")

    def add_metadata(self, **kwargs: Any) -> None:
        """Agrega campos extra a ``metadata.json`` (bajo ``extra``)."""
        self.extra_metadata.update(kwargs)

    def register_artifact(self, path: str | os.PathLike[str], description: str = "") -> None:
        """Registra un fichero producido (queda listado en ``metadata.json``)."""
        self.artifacts.append({"path": str(path), "description": description})

    def save_figure(
        self,
        fig: Any,
        name: str,
        *,
        data: Any = None,
        sources: Sequence[str | os.PathLike[str]] | None = None,
        copy_to_reports: bool | None = None,
    ) -> list[Path]:
        """Guarda una figura en ``run_dir/figures`` y, si se configura, en ``reports/figures``.

        Args:
            data: ``DataFrame`` con los datos que se dibujan. Se guarda como ``<name>.data.csv`` y
                su tamano queda en ``<name>.provenance.json`` (``n_points``): asi se puede AUDITAR
                que la figura procede de resultados reales y no esta vacia.
            sources: Ficheros de resultados de los que proceden los datos (se registra su SHA-256).
            copy_to_reports: Fuerza/evita la copia en ``reports/figures`` (defecto: ``figures.also_save_to_reports``).
        """
        from utils.hashing import sha256_file
        from utils.visualization import save_figure

        fmts = list(get_nested(self.cfg, "figures.formats", default=["png"]))
        dpi = int(get_nested(self.cfg, "figures.dpi", default=200))
        import matplotlib.pyplot as plt
        to_reports = get_nested(self.cfg, "figures.also_save_to_reports", default=True) if copy_to_reports is None else copy_to_reports
        try:
            paths = save_figure(fig, self.figures_dir, name, formats=fmts, dpi=dpi, close=False)
            if to_reports:
                rep = resolve_path(get_nested(self.cfg, "paths.figures_dir", default="reports/figures"))
                paths += save_figure(fig, rep, name, formats=fmts, dpi=dpi, close=False)
        finally:
            plt.close(fig)
        n_points: int | None = None
        sidecar_dirs = [self.figures_dir]
        if to_reports:
            sidecar_dirs.append(resolve_path(get_nested(self.cfg, "paths.figures_dir", default="reports/figures")))
        if data is not None:
            try:
                n_points = int(len(data))
                for d in sidecar_dirs:
                    data.to_csv(d / f"{name}.data.csv", index=False, lineterminator="\n")
                self.register_artifact(self.figures_dir / f"{name}.data.csv", "datos de la figura")
            except Exception:
                self.logger.exception("No se pudieron guardar los datos de la figura %s", name)
        src_info: list[dict[str, Any]] = []
        for s in sources or []:
            sp = Path(s)
            src_info.append({"path": str(sp), "sha256": sha256_file(sp) if sp.is_file() else None})
        prov = {
            "semantic_fingerprint": semantic_fingerprint(self.cfg),
            "profile": self.cfg["project"]["profile"],
            "figure": name, "run_id": self.run_id, "created_at": datetime.now().astimezone().isoformat(),
            "n_points": n_points, "has_data": bool(n_points) if n_points is not None else None,
            "synthetic": False, "sources": src_info, "code_hash": self.hashes.get("code_hash"),
            "dataset_hash": self.hashes.get("dataset_hash"), "split_hash": self.hashes.get("split_hash"),
        }
        for d in sidecar_dirs:
            write_json(d / f"{name}.provenance.json", prov)
        if n_points == 0:
            self.logger.warning("figura %s generada SIN datos (n_points=0): no cuenta como evidencia", name)
        for p in paths:
            self.register_artifact(p, "figura")
        self.logger.info("figura guardada: %s", ", ".join(str(p) for p in paths))
        return paths

    def set_checkpoint(self, path: str | os.PathLike[str] | None, *, role: str = "best", **info: Any) -> dict[str, Any]:
        """Registra un checkpoint (ruta relativa al run si es posible, SHA-256 y tamano)."""
        from utils.hashing import sha256_file

        entry: dict[str, Any] = {"role": role, **info}
        if path is not None and Path(path).is_file():
            p = Path(path)
            entry.update(path=str(p), sha256=sha256_file(p), size_bytes=p.stat().st_size)
        else:
            entry.update(path=str(path) if path else None, sha256=None, size_bytes=None)
        self.checkpoints.append(entry)
        self._write_checkpoint_metadata()
        return entry

    def record_checkpoint(self, role: str, *, sha256: str | None, path: str | os.PathLike[str] | None = None, **info: Any) -> None:
        """Registra un checkpoint por su hash (p. ej. pesos de un fold que se descartan tras medirlos)."""
        self.checkpoints.append({"role": role, "path": str(path) if path else None, "sha256": sha256, "size_bytes": None, **info})
        self._write_checkpoint_metadata()

    def add_checkpoint_info(self, **info: Any) -> None:
        """Agrega campos libres a ``checkpoint_metadata.json`` (epocas, pesos descartados, etc.)."""
        self.checkpoint_extra.update(info)
        self._write_checkpoint_metadata()

    def note_split_access(self, split: str, stage: str) -> None:
        """Registra que este run ley el ``split`` en la etapa ``stage`` (auditoria de fuga)."""
        self.splits_accessed.append({"split": split, "stage": stage})

    def _write_checkpoint_metadata(self) -> None:
        payload = {
            "run_id": self.run_id, "architecture": self.architecture, "script": self.script_name,
            "status": self.status, "checkpoints": self.checkpoints, **self.checkpoint_extra,
            "dataset_hash": self.hashes.get("dataset_hash"), "split_hash": self.hashes.get("split_hash"),
            "code_hash": self.hashes.get("code_hash"), "environment_hash": self.env.get("environment_hash"),
            "git_commit": self.env.get("git_commit"), "seed": self.seed, "pc_id": self.pc_id,
            "note": None if self.checkpoints else "este run no produjo checkpoints (o fueron descartados)",
        }
        try:
            write_json(self.run_dir / "checkpoint_metadata.json", redact(payload))
        except Exception:
            self.logger.exception("No se pudo escribir checkpoint_metadata.json")

    # ------------------------------------------------------------- finalizacion
    def _metadata(self, final: bool) -> dict[str, Any]:
        end = self.finished_at
        env = self.env
        try:
            fingerprint = semantic_fingerprint(self.cfg)
        except KeyError:
            # Bootstrap failures and minimal logging contexts still need a trace.
            fingerprint = None
        return {
            "run_id": self.run_id,
            "architecture": self.architecture,
            "script": self.script_name,
            "experiment_name": self.experiment_name,
            "status": self.status,
            "splits_accessed": self.splits_accessed,
            "date": self.started_at.strftime("%Y-%m-%d"),
            "start_time": self.started_at.strftime("%H:%M:%S"),
            "end_time": end.strftime("%H:%M:%S") if end else None,
            "started_at": self.started_at.isoformat(),
            "finished_at": end.isoformat() if end else None,
            "started_at_utc": self.started_at.astimezone(timezone.utc).isoformat(),
            "finished_at_utc": end.astimezone(timezone.utc).isoformat() if end else None,
            "duration_seconds": round(time.perf_counter() - self._t0, 3) if final else None,
            "pc_id": self.pc_id,
            "pc_id_explicit": self.pc_id_explicit,
            "hostname": env.get("hostname"),
            "gpu": {"model": env.get("gpu_model"), "vram_mb": env.get("gpu_vram_mb"), "driver": env.get("nvidia_driver")},
            "cuda": {"runtime": env.get("cuda_runtime"), "driver_api": env.get("cuda_driver_api"),
                     "cudnn": env.get("cudnn"), "available": env.get("cuda_available_torch")},
            "pytorch_version": env.get("torch"),
            "python_version": env.get("python_version"),
            "git_commit": env.get("git_commit"),
            "git_dirty": env.get("git_dirty"),
            "code_hash": self.hashes.get("code_hash"),
            "dataset_hash": self.hashes.get("dataset_hash"),
            "semantic_fingerprint": fingerprint,
            "split_hash": self.hashes.get("split_hash"),
            "environment_hash": env.get("environment_hash"),
            "config_hash": sha256_json({k: v for k, v in self.cfg.items() if k != "_meta"}),
            "seed": self.seed,
            "seed_info": self.seed_info,
            "parameters": {
                "cli_args": self.cli_args,
                "argv": self.argv,
                "config_files": get_nested(self.cfg, "_meta.config_files", default=None),
                "config_overrides": get_nested(self.cfg, "_meta.overrides", default=None),
            },
            "artifacts": self.artifacts,
            "warnings_count": self._counter.warnings,
            "errors_count": self._counter.errors,
            "error": self.error,
            "extra": self.extra_metadata,
        }

    def _write_all(self, final: bool) -> None:
        """Escribe environment/metrics/metadata; nunca lanza (la traza no debe fallar)."""
        for fname, payload in (
            ("environment.json", self.env),
            ("metrics.json", self.metrics),
            ("metadata.json", self._metadata(final)),
        ):
            try:
                write_json(self.run_dir / fname, redact(payload))
            except Exception:
                self.logger.exception("No se pudo escribir %s", fname)
        self._write_checkpoint_metadata()

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        self.finished_at = datetime.now().astimezone()
        if exc is None or (isinstance(exc, SystemExit) and exc.code in (0, None)):
            self.status = self.status_override or "COMPLETE"
        elif exc_type is not None and exc_type.__name__ in _PRUNED_EXCEPTIONS:
            # Un trial podado NO es un error (sec. 34): se registra como PRUNED sin traceback de fallo.
            self.status = "PRUNED"
            self.logger.info("trial PRUNED por el pruner: %s", exc)
        else:
            self.status = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
            full_tb = "".join(traceback.format_exception(exc_type, exc, tb))
            self.error = {"type": exc_type.__name__ if exc_type else None, "message": str(exc), "traceback": full_tb}
            self.logger.error("EJECUCION %s: %s: %s\n%s", self.status, self.error["type"], exc, full_tb)
        self.logger.info("hora_fin=%s  duracion=%.2fs  estado=%s",
                         self.finished_at.strftime("%H:%M:%S%z"), time.perf_counter() - self._t0, self.status)
        self.logger.info("run guardado en: %s", self.run_dir)
        self._write_all(final=True)
        for h in list(self.logger.handlers):
            try:
                h.flush()
                h.close()
            except Exception:
                pass
            self.logger.removeHandler(h)
        self.emit_event("run_finished",status=self.status,reason=self.error)
        if hasattr(self,"_stdout_scope"):
            sys.stdout.flush();sys.stderr.flush()
            self._stderr_scope.__exit__(None,None,None);self._stdout_scope.__exit__(None,None,None)
        self._run_lock.__exit__(None,None,None)
        if hasattr(self,"_cfg_token"): ACTIVE_CONFIG.reset(self._cfg_token)
        if hasattr(self,"_dir_token"): ACTIVE_RUN_DIR.reset(self._dir_token)
        if hasattr(self,"_context_token"): ACTIVE_CONTEXT.reset(self._context_token)
        return False  # nunca suprime la excepcion


# --------------------------------------------------------------------------- #
# Punto de entrada comun de los scripts
# --------------------------------------------------------------------------- #
def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos comunes a todos los scripts."""
    g = parser.add_argument_group("comunes")
    g.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML de configuracion (override sobre default.yaml)")
    g.add_argument("--set", dest="set", action="append", default=[], metavar="CLAVE=VALOR",
                   help="override de configuracion, p. ej. --set outer_split.seed=2026 (repetible)")
    g.add_argument("--pc-id", default=None, help="ID de la PC (PC-01|PC-02|PC-03); por defecto BOTRYTIS_PC_ID")
    g.add_argument("--runs-dir", default=None, help="directorio raiz de runs (por defecto paths.runs_dir)")
    g.add_argument("--experiment-name", default=None, help="sufijo del directorio del run (por defecto el nombre del script)")


def _configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass


def run_cli(
    script_name: str,
    main: Callable[[RunContext, argparse.Namespace], None],
    *,
    description: str,
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
    argv: Sequence[str] | None = None,
    run_layout: Callable[[argparse.Namespace], tuple[Any, ...]] | None = None,
) -> int:
    """Ejecuta ``main`` dentro de un :class:`RunContext`, garantizando trazabilidad.

    Incluso si fallan el parseo de argumentos o la carga de la configuracion se crea un run
    con ``status = FAILED`` y el traceback completo.

    Args:
        run_layout: ``args -> (architecture_slug, trial_name)`` para ubicar el run en
            ``runs/<architecture>/<trial_name>`` (por defecto ``runs/_pipeline/<ts>_<script>``).

    Returns:
        Codigo de salida: 0 OK, 1 error, 2 argumentos invalidos, 130 interrupcion.
    """
    _configure_console()
    parser = _TraceableParser(prog=script_name, description=description,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    if add_arguments is not None:
        add_arguments(parser)

    raw_argv = list(sys.argv if argv is None else [script_name, *argv])
    bootstrap_error: BaseException | None = None
    exit_code = 1
    try:
        args = parser.parse_args(None if argv is None else list(argv))
    except CliArgumentError as exc:
        bootstrap_error, exit_code = exc, 2
        # Recuperar (de forma tolerante) los argumentos comunes para que el run quede donde se pidio.
        lenient = _TraceableParser(prog=script_name, add_help=False)
        add_common_arguments(lenient)
        try:
            args, _ = lenient.parse_known_args(None if argv is None else list(argv))
        except Exception:
            args = argparse.Namespace(config=None, set=[], pc_id=None, runs_dir=None, experiment_name=None)

    cfg: dict[str, Any] | None = None
    if bootstrap_error is None:
        try:
            cfg = load_config(args.config, args.set)
        except (ConfigError, OSError, ValueError, KeyError) as exc:
            bootstrap_error = exc
        except Exception as exc:  # YAML mal formado, etc.
            bootstrap_error = exc
    if cfg is None:
        cfg = {
            "paths": {"runs_dir": args.runs_dir or os.environ.get("BOTRYTIS_RUNS_DIR", "runs")},
            "_bootstrap_failure": {"type": type(bootstrap_error).__name__, "message": str(bootstrap_error)},
            "_meta": {"config_files": [str(args.config)], "overrides": list(args.set or [])},
        }

    cli_args = {k: v for k, v in vars(args).items()}
    architecture: str | None = None
    trial_name: str | None = None
    reuse = False
    if run_layout is not None and bootstrap_error is None:
        try:
            layout = run_layout(args)
            architecture, trial_name = layout[0], layout[1]
            reuse = bool(layout[2]) if len(layout) > 2 else False
        except Exception as exc:  # un argumento invalido no debe impedir dejar traza
            bootstrap_error, exit_code = exc, 1
    ctx = RunContext(
        cfg, script_name, cli_args=cli_args, argv=raw_argv,
        experiment_name=args.experiment_name, pc_id=args.pc_id, runs_dir=args.runs_dir,
        architecture=architecture, trial_name=trial_name, reuse_existing=reuse,
    )
    try:
        with ctx:
            if bootstrap_error is not None:
                raise bootstrap_error
            # Operational acceptance belongs exclusively to the person at this PC.
            if script_name[:2].isdigit() and 6 <= int(script_name[:2]) <= 17 and cfg.get("project",{}).get("profile")=="official":
                from utils.manual_gates import require_gate
                require_gate(cfg, ctx.pc_id, "READY_FOR_EXPERIMENT")
            main(ctx, args)
            from utils.experiments import finish_direct
            finish_direct(ctx,success=True)
            if script_name in ('10_evaluate_outer_validation','11_tune_inference_thresholds') and cfg.get('project',{}).get('profile')=='official':
                from utils.experiments import finish_external
                finish_external(cfg,'outer_validation' if script_name.startswith('10_') else 'threshold_tuning')
        set_exit(0)
        return 0
    except KeyboardInterrupt:
        set_exit(130)
        return 130
    except BaseException as exc:  # el traceback ya quedo en execution.log/metadata.json
        from utils.experiments import finish_direct
        if getattr(ctx,'_direct_stage_job',None):
            from orchestration.study_queue import StudyQueue,PUBLICATION_PENDING
            q=StudyQueue.from_cfg(ctx.cfg)
            if q.get(ctx._direct_stage_job).status!=PUBLICATION_PENDING:
                finish_direct(ctx,success=False)
        set_exit(1)
        print(f"[{script_name}] ERROR: {type(exc).__name__}: {exc}\n  Detalle completo en {ctx.run_dir / 'execution.log'}",
              file=sys.stderr)
        if isinstance(exc, SystemExit):
            return int(exc.code) if isinstance(exc.code, int) else 1
        return exit_code if bootstrap_error is not None and exit_code == 2 else 1
