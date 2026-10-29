"""Guardias metodologicas de las fases de entrenamiento (sec. 35): hacen FALLAR el programa.

Complementan ``utils.checks`` (datos/split/folds) con los controles de presupuesto, preentrenamiento,
tamano de imagen, trabajos GPU por PC, seleccion por seeds y estado del test/freeze.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.architectures import archs, get_arch, trials_for
from utils.checks import MethodologyViolation
from utils.config_loader import get_nested, resolve_path

OFFICIAL = "official"


def is_official(cfg: Mapping[str, Any]) -> bool:
    """``True`` si el perfil es ``official`` (los perfiles ``smoke`` relajan presupuestos, no la logica)."""
    return str(get_nested(cfg, "project.profile", default=OFFICIAL)) == OFFICIAL


def assert_equal_hpo_budget(cfg: Mapping[str, Any], requested: Mapping[str, int] | None = None, *, declared_override: bool = False) -> None:
    """[sec. 35] Todas las arquitecturas de un grupo reciben el mismo presupuesto HPO (15 / 20) salvo que se declare.

    Args:
        requested: ``{arquitectura: n_trials}`` efectivamente solicitado (p. ej. por ``--n-trials``).
        declared_override: ``True`` si el usuario paso ``--allow-budget-override`` (queda registrado).
    """
    for task in ("detection", "classification"):
        group = archs(task)
        budgets = {a.name: trials_for(a, cfg) for a in group}
        if requested:
            budgets.update({k: v for k, v in requested.items() if get_arch(k).task == task})
        if len(set(budgets.values())) > 1 and not declared_override:
            raise MethodologyViolation(f"Presupuesto HPO desigual sin declarar en {task}: {budgets}")
    if is_official(cfg):
        if int(cfg["detector"]["trials_per_architecture"]) != 15 or int(cfg["classifier"]["trials_per_architecture"]) != 20:
            if not declared_override:
                raise MethodologyViolation("El presupuesto oficial es 15 trials (detector) / 20 (clasificador) (sec. 12)")


def assert_same_pretrained_policy(cfg: Mapping[str, Any]) -> None:
    """El protocolo primario requiere pretrained; scratch es una alternativa predeclarada."""
    mode = cfg.get("initialization", {}).get("mode")
    if mode not in ("pretrained", "scratch"):
        raise MethodologyViolation("Declare initialization.mode")
    if any(cfg[s]["pretrained"] is not (mode == "pretrained") for s in ("detector", "classifier")):
        raise MethodologyViolation("Inicializaciones incompatibles; no mezclar pretrained y scratch")
    if mode == "scratch" and not cfg["initialization"].get("scratch_decision"):
        raise MethodologyViolation("Scratch requiere decision manual documentada y nuevo experimento")


def assert_image_size_fixed(cfg: Mapping[str, Any]) -> None:
    """[sec. 35] ``image_size`` fijo durante la comparacion principal: 640 (detector) / 224 (clasificador)."""
    if not is_official(cfg):
        return
    if int(cfg["detector"]["image_size"]) != 640 or int(cfg["classifier"]["image_size"]) != 224:
        raise MethodologyViolation("image_size es FIJO en la comparacion principal: detector 640, clasificador 224 (sec. 14)")


def assert_one_selected_by_mean(scores_by_seed: Mapping[int, Sequence[float]], expected_seeds: Sequence[int]) -> None:
    """[sec. 35] Una configuracion se elige por la MEDIA de las 3 seeds, nunca por una sola."""
    if sorted(scores_by_seed) != sorted(int(s) for s in expected_seeds):
        raise MethodologyViolation(f"Seleccion por un unico seed/seeds incompletas: {sorted(scores_by_seed)} != {sorted(expected_seeds)}")


def gpu_lock_path(cfg, pc_id):
    """Machine-wide local lock, shared by all checkouts and experiment rounds."""
    import tempfile
    return Path(tempfile.gettempdir()) / 'botrytis_gpu_locks' / (str(pc_id) + '.json')


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception:
        if os.name == "nt":
            # os.kill(pid, 0) is not a safe Windows existence probe. Query the
            # process without signalling it; access denied must not reclaim a lock.
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x1000, False, pid)
            if not handle:
                return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER = no such PID
            try:
                code = wintypes.DWORD()
                return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
            finally:
                kernel.CloseHandle(handle)
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


class GpuJobLock:
    """Bloqueo de GPU: como maximo UN trabajo GPU pesado por PC (sec. 16.1, 35).

    Uso: ``with GpuJobLock(cfg, 'PC-01', job='hpo:yolov8n'): ...``. Si otro proceso VIVO ya lo tiene,
    lanza :class:`MethodologyViolation`. Un lock huerfano (proceso muerto) se recupera.
    """

    def __init__(self, cfg: Mapping[str, Any], pc_id: str, job: str = "") -> None:
        self.path = gpu_lock_path(cfg, pc_id)
        self.pc_id = pc_id
        self.job = job
        self._held = False

    def __enter__(self) -> "GpuJobLock":
        import json
        from utils.durable import FileLock
        self._lock = FileLock(self.path.with_suffix('.guard'))
        try:
            self._lock.__enter__()
        except RuntimeError as exc:
            raise MethodologyViolation(f'{self.pc_id}: otro trabajo GPU local mantiene el lock') from exc
        self.path.write_text(json.dumps({'pid': os.getpid(), 'pc_id': self.pc_id, 'job': self.job}))
        self._held = True
        return self

    def __exit__(self, *exc):
        if self._held:
            self.path.unlink(missing_ok=True)
            self._lock.__exit__(*exc)
            self._held = False


def assert_not_frozen(cfg: Mapping[str, Any], action: str) -> None:
    """[sec. 26] Despues de ``freeze_manifest.json`` no se puede reentrenar/seleccionar/ajustar nada."""
    freeze = resolve_path(get_nested(cfg, "test.freeze_manifest", default="artifacts/freeze_manifest.json"))
    if freeze.is_file():
        raise MethodologyViolation(f"El experimento esta CONGELADO ({freeze.name}): no se permite '{action}' (sec. 26).")


def assert_benchmark_pc(cfg: Mapping[str, Any], pc_id: str) -> None:
    """[sec. 29.1] La latencia se mide SOLO en la PC designada (``benchmark.benchmark_pc``)."""
    bench = str(get_nested(cfg, "benchmark.benchmark_pc", default="PC-01"))
    if pc_id != bench:
        raise MethodologyViolation(f"La latencia solo se mide en {bench}; esta PC es {pc_id} (sec. 29.1)")
