"""Reproducibilidad: semillas, deteccion de hardware/software, git y comparacion entre PCs.

Ninguna funcion de este modulo exige ``torch``: si no esta instalado, los campos relacionados
se registran como ``None`` (asi ``00_check_environment.py`` funciona antes de instalar el stack ML).
"""

from __future__ import annotations

import ctypes
import os
import platform
import random
import re
import socket
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utils.hashing import sha256_json

#: Paquetes cuya version interesa registrar explicitamente (sec. 16.6 / 41.4).
TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy", "pandas", "pillow", "pyyaml", "matplotlib", "psutil",
    "torch", "torchvision", "timm", "ultralytics", "optuna", "scikit-learn", "opencv-python",
)

#: Campos del entorno que NO forman parte del ``environment_hash`` (son datos/codigo, no entorno).
_NON_ENV_FIELDS = {"git_commit", "code_hash", "dataset_hash", "split_hash"}


# --------------------------------------------------------------------------- #
# Semillas
# --------------------------------------------------------------------------- #
def set_global_seed(
    seed: int,
    *,
    deterministic: bool = True,
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False,
) -> dict[str, Any]:
    """Fija las semillas de Python, NumPy y (si existe) PyTorch/CUDA.

    Args:
        seed: Semilla entera.
        deterministic: Activa ``torch.use_deterministic_algorithms(True, warn_only=True)``.
        cudnn_deterministic: ``torch.backends.cudnn.deterministic``.
        cudnn_benchmark: ``torch.backends.cudnn.benchmark``.

    Returns:
        Diccionario con lo realmente aplicado (se guarda en ``metadata.json``).
    """
    seed = int(seed)
    applied: dict[str, Any] = {"seed": seed, "python": True, "numpy": True, "torch": False}
    os.environ["PYTHONHASHSEED"] = str(seed)  # solo afecta a subprocesos lanzados despues
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        applied.update(
            torch=True,
            cuda=bool(torch.cuda.is_available()),
            deterministic_algorithms=bool(deterministic),
            cudnn_deterministic=bool(cudnn_deterministic),
            cudnn_benchmark=bool(cudnn_benchmark),
        )
    except Exception:  # torch ausente o CUDA defectuosa: no es un error para el modulo base
        pass
    return applied


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` para ``DataLoader``: deriva semillas de ``torch.initial_seed()``."""
    import torch  # type: ignore[import-not-found]

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int) -> Any:
    """``torch.Generator`` sembrado (para ``DataLoader(generator=...)``)."""
    import torch  # type: ignore[import-not-found]

    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


# --------------------------------------------------------------------------- #
# Identificacion de PC, git, hardware
# --------------------------------------------------------------------------- #
def resolve_pc_id(configured: str | None = None) -> tuple[str, bool]:
    """Devuelve ``(pc_id, es_explicito)``.

    Prioridad: valor configurado/CLI > variable ``BOTRYTIS_PC_ID`` > hostname.
    """
    value = (configured or os.environ.get("BOTRYTIS_PC_ID") or "").strip()
    if value:
        return value, True
    return f"HOST-{socket.gethostname()}", False


def _run(cmd: Sequence[str], cwd: Path | None = None, timeout: float = 15.0) -> str | None:
    try:
        out = subprocess.run(
            list(cmd), cwd=str(cwd) if cwd else None, capture_output=True, text=True,
            timeout=timeout, check=False, encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def get_git_info(project_root: Path) -> dict[str, Any]:
    """Informacion de git del proyecto (``commit=None`` si ``project_root`` no es un repo propio)."""
    info: dict[str, Any] = {"available": False, "commit": None, "branch": None, "dirty": None}
    top = _run(["git", "rev-parse", "--show-toplevel"], cwd=project_root)
    if not top:
        return info
    try:
        if Path(top).resolve() != project_root.resolve():
            info["note"] = f"git toplevel ({top}) != raiz del proyecto; se ignora"
            return info
    except OSError:
        return info
    info["available"] = True
    info["commit"] = _run(["git", "rev-parse", "HEAD"], cwd=project_root)
    info["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=project_root)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=project_root)
    info["dirty"] = bool(status) if status is not None else None
    return info


def _cpu_name() -> str:
    system = platform.system()
    try:
        if system == "Windows":
            import winreg  # type: ignore[import-not-found]

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if system == "Darwin":
            out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            if out:
                return out
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"


def _ram_total_mb() -> int | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.virtual_memory().total // (1024 * 1024))
    except Exception:
        pass
    try:
        if platform.system() == "Windows":
            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return int(stat.ullTotalPhys // (1024 * 1024))
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None


def _nvidia_smi() -> dict[str, Any]:
    """GPUs, driver y version CUDA del driver via ``nvidia-smi`` (sin depender de torch)."""
    result: dict[str, Any] = {"gpus": [], "driver": None, "cuda_driver_api": None}
    out = _run([
        "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,uuid",
        "--format=csv,noheader,nounits",
    ])
    if out:
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                result["gpus"].append({
                    "index": int(parts[0]), "name": parts[1],
                    "vram_mb": int(float(parts[2])), "uuid": parts[4],
                })
                result["driver"] = parts[3]
    header = _run(["nvidia-smi"])
    if header:
        m = re.search(r"CUDA Version:\s*([0-9.]+)", header)
        if m:
            result["cuda_driver_api"] = m.group(1)
    return result


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _all_installed_packages() -> dict[str, str]:
    pkgs: dict[str, str] = {}
    for dist in importlib_metadata.distributions():
        name = dist.metadata["Name"] if dist.metadata else None
        if name:
            pkgs[name.lower()] = dist.version
    return dict(sorted(pkgs.items()))


def _torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch": None, "cuda_runtime": None, "cudnn": None, "cuda_available": False,
        "torch_gpu_name": None, "torch_gpu_vram_mb": None, "gpu_count": 0, "bf16_supported": None,
    }
    try:
        import torch  # type: ignore[import-not-found]

        info["torch"] = str(torch.__version__)
        info["cuda_runtime"] = torch.version.cuda
        cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info["cudnn"] = str(cudnn) if cudnn is not None else None
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            info["gpu_count"] = int(torch.cuda.device_count())
            info["torch_gpu_name"] = props.name
            info["torch_gpu_vram_mb"] = int(props.total_memory // (1024 * 1024))
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception as exc:  # torch no instalado o roto
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info


def collect_environment(
    cfg: Mapping[str, Any] | None = None,
    project_root: Path | None = None,
    *,
    pc_id: str | None = None,
    dataset_hash: str | None = None,
    split_hash: str | None = None,
    code_hash: str | None = None,
    include_all_packages: bool = True,
) -> dict[str, Any]:
    """Construye el contenido de ``environment.json``.

    Incluye PC, SO, CPU/RAM, GPU, driver, CUDA, versiones de paquetes, git, hashes y un
    ``environment_hash`` calculado sobre los campos criticos de entorno.
    """
    from utils.config_loader import PROJECT_ROOT, get_nested

    root = project_root or PROJECT_ROOT
    smi = _nvidia_smi()
    tinfo = _torch_info()
    pkgs = _package_versions()
    git = get_git_info(root)
    gpu0 = smi["gpus"][0] if smi["gpus"] else None

    precision = None
    if cfg is not None:
        det_p = get_nested(cfg, "detector.precision", default=None)
        cls_p = get_nested(cfg, "classifier.precision", default=None)
        precision = det_p if det_p == cls_p else f"detector={det_p};classifier={cls_p}"

    env: dict[str, Any] = {
        "pc_id": pc_id,
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "os_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "cpu_model": _cpu_name(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_total_mb": _ram_total_mb(),
        "gpu_model": (gpu0["name"] if gpu0 else tinfo.get("torch_gpu_name")),
        "gpu_vram_mb": (gpu0["vram_mb"] if gpu0 else tinfo.get("torch_gpu_vram_mb")),
        "gpu_count": len(smi["gpus"]) or tinfo.get("gpu_count", 0),
        "gpus": smi["gpus"],
        "nvidia_driver": smi["driver"],
        "cuda_driver_api": smi["cuda_driver_api"],
        "cuda_runtime": tinfo["cuda_runtime"],
        "cudnn": tinfo["cudnn"],
        "cuda_available_torch": tinfo["cuda_available"],
        "bf16_supported": tinfo["bf16_supported"],
        "torch": tinfo["torch"] or pkgs.get("torch"),
        "torchvision": pkgs.get("torchvision"),
        "timm": pkgs.get("timm"),
        "ultralytics": pkgs.get("ultralytics"),
        "optuna": pkgs.get("optuna"),
        "numpy": pkgs.get("numpy"),
        "precision_mode": precision,
        "packages": pkgs,
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("dirty"),
        "git_info": git,
        "code_hash": code_hash,
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
    }
    if "torch_error" in tinfo:
        env["torch_error"] = tinfo["torch_error"]
    if include_all_packages:
        env["all_packages"] = _all_installed_packages()

    fields = _environment_hash_fields(cfg)
    env["environment_hash"] = environment_hash(env, fields)
    return env


def _environment_hash_fields(cfg: Mapping[str, Any] | None) -> list[str]:
    from utils.config_loader import get_nested

    fields = get_nested(cfg, "compute.critical_environment_fields", default=None) if cfg else None
    if not fields:
        fields = [
            "python_version", "os_system", "gpu_model", "gpu_vram_mb", "nvidia_driver", "cuda_runtime",
            "cudnn", "torch", "torchvision", "timm", "ultralytics", "optuna", "numpy", "precision_mode",
        ]
    return [f for f in fields if f not in _NON_ENV_FIELDS]


def environment_hash(env: Mapping[str, Any], fields: Sequence[str]) -> str:
    """SHA-256 de los campos criticos de entorno (excluye git/codigo/dataset)."""
    return sha256_json({f: env.get(f) for f in fields if f not in _NON_ENV_FIELDS})


def compare_environments(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Devuelve las diferencias en ``fields`` entre dos ``environment.json``.

    Cada elemento: ``{"field", "reference", "candidate"}``. Una lista vacia => comparables.
    Se ignoran campos ausentes en AMBOS lados.
    """
    diffs: list[dict[str, Any]] = []
    for f in fields:
        a, b = reference.get(f), candidate.get(f)
        if a is None and b is None:
            continue
        if a != b:
            diffs.append({"field": f, "reference": a, "candidate": b})
    return diffs
