"""Configura limites verificados o solo registro por decision explicita del operador."""
from __future__ import annotations

# Stdlib bootstrap: records import/CLI/config failures before optional dependencies.
import sys as _boot_sys
from pathlib import Path as _BootPath
_boot_root = _BootPath(__file__).resolve().parents[1]
_boot_sys.path.insert(0, str(_boot_root / "src"))
from utils import bootstrap as _bootstrap
_bootstrap.install(__file__, _boot_root)
import argparse
import csv
import getpass
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-index", type=int, required=True)
    p.add_argument("--observe-only", action="store_true", help="solo registro, sin paradas por temperatura o reservas de RAM/disco")
    p.add_argument("--config", type=Path, help="conecta esta configuracion local con la politica creada")
    p.add_argument("--replace-existing", action="store_true", help="respalda la politica existente antes de reemplazarla")
    p.add_argument("--max-gpu-temperature-c", type=float)
    p.add_argument("--gpu-limit-source", help="especificacion del fabricante + margen verificado")
    p.add_argument("--min-available-ram-gib", type=float)
    p.add_argument("--min-free-disk-gib", type=float)
    p.add_argument("--cpu-protection-source")
    p.add_argument("--cpu-thermal-protection-verified", action="store_true")
    p.add_argument("--cpu-sensor-unavailable-acknowledged", action="store_true")
    p.add_argument("--max-cpu-temperature-c", type=float)
    a = p.parse_args()
    limits = (a.max_gpu_temperature_c, a.min_available_ram_gib, a.min_free_disk_gib)
    if not a.observe_only and (any(v is None or v <= 0 for v in limits)
                              or not all((a.gpu_limit_source, a.cpu_protection_source, a.cpu_thermal_protection_verified))):
        p.error("Modo con paradas requiere limites positivos, fuentes y protecciones CPU verificadas")
    import yaml
    from utils.config_loader import load_config, cfg_path, resolve_path
    from utils.durable import atomic_json, atomic_bytes
    dest = ROOT / "artifacts/v4/safety_policy.json"
    local = None
    if a.config:
        a.config = resolve_path(a.config)
        effective = load_config(a.config)
        local = yaml.safe_load(a.config.read_text(encoding="utf-8-sig"))
        if not isinstance(local, dict) or not isinstance(local.get("operations", {}), dict):
            raise ValueError("Configuracion local invalida")
        dest = cfg_path(effective, "paths.artifacts_dir") / "safety_policy.json"
    if dest.exists() and not a.replace_existing:
        raise FileExistsError("Ya hay una politica. --replace-existing conserva un respaldo antes de reemplazarla")
    result = subprocess.run(["nvidia-smi", "-i", str(a.gpu_index), "--query-gpu=uuid,name",
                             "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=10)
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 2:
        raise RuntimeError("GPU ambigua")
    data = {"gpu_uuid": rows[0][0].strip(), "gpu_model": rows[0][1].strip(),
            "mode": "observe" if a.observe_only else "enforce",
            "user_requested_observation_only": a.observe_only,
            "max_gpu_temperature_c": a.max_gpu_temperature_c, "gpu_limit_source": a.gpu_limit_source,
            "min_available_ram_bytes": int(a.min_available_ram_gib * 1024**3) if a.min_available_ram_gib is not None else None,
            "min_free_disk_bytes": int(a.min_free_disk_gib * 1024**3) if a.min_free_disk_gib is not None else None,
            "cpu_protection_source": a.cpu_protection_source,
            "cpu_thermal_protection_verified": a.cpu_thermal_protection_verified,
            "cpu_sensor_unavailable_acknowledged": a.cpu_sensor_unavailable_acknowledged,
            "max_cpu_temperature_c": a.max_cpu_temperature_c,
            "configured_by": getpass.getuser(), "configured_at": datetime.now().astimezone().isoformat()}
    suffix = ".before_" + uuid.uuid4().hex
    if dest.exists():
        atomic_bytes(dest.with_name(dest.name + suffix), dest.read_bytes())
    atomic_json(dest, data)
    if a.config:
        atomic_bytes(a.config.with_name(a.config.name + suffix), a.config.read_bytes())
        local.setdefault("operations", {})["safety_policy_file"] = dest.relative_to(ROOT).as_posix() if dest.is_relative_to(ROOT) else str(dest)
        atomic_bytes(a.config, yaml.safe_dump(local, allow_unicode=True, sort_keys=False).encode("utf-8"))
    print("Politica registrada:", dest)
    print("Modo:", data["mode"])
    if a.observe_only:
        print("Solo registro: sin paradas por temperatura, RAM o disco. No modifica controles del hardware ni verifica protecciones OEM.")
    else:
        print("No se ejecuto una prueba de carga ni se certifico seguridad por ausencia de errores.")

if __name__ == "__main__":
    main()
