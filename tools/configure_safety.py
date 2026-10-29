"""Guarda limites verificados por el operador; no deriva limites universales del hardware."""
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
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-index", type=int, required=True)
    p.add_argument("--max-gpu-temperature-c", type=float, required=True)
    p.add_argument("--gpu-limit-source", required=True, help="especificacion del fabricante + margen verificado")
    p.add_argument("--min-available-ram-gib", type=float, required=True)
    p.add_argument("--min-free-disk-gib", type=float, required=True)
    p.add_argument("--cpu-protection-source", required=True)
    p.add_argument("--cpu-thermal-protection-verified", action="store_true", required=True)
    p.add_argument("--cpu-sensor-unavailable-acknowledged", action="store_true")
    p.add_argument("--max-cpu-temperature-c", type=float)
    a = p.parse_args()
    result = subprocess.run(["nvidia-smi", "-i", str(a.gpu_index), "--query-gpu=uuid,name",
                             "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=10)
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 2:
        raise RuntimeError("GPU ambigua")
    data = {"gpu_uuid": rows[0][0].strip(), "gpu_model": rows[0][1].strip(),
            "max_gpu_temperature_c": a.max_gpu_temperature_c, "gpu_limit_source": a.gpu_limit_source,
            "min_available_ram_bytes": int(a.min_available_ram_gib * 1024**3),
            "min_free_disk_bytes": int(a.min_free_disk_gib * 1024**3),
            "cpu_protection_source": a.cpu_protection_source, "cpu_thermal_protection_verified": True,
            "cpu_sensor_unavailable_acknowledged": a.cpu_sensor_unavailable_acknowledged,
            "max_cpu_temperature_c": a.max_cpu_temperature_c,
            "verified_by": getpass.getuser(), "verified_at": datetime.now().astimezone().isoformat()}
    if any(data[k] <= 0 for k in ("max_gpu_temperature_c", "min_available_ram_bytes", "min_free_disk_bytes")):
        raise ValueError("Los limites deben ser positivos y haber sido verificados")
    from utils.durable import atomic_json
    dest = ROOT / "artifacts/v4/safety_policy.json"
    if dest.exists():
        raise FileExistsError("Ya hay una politica. Preserve y revise la existente antes de reemplazarla explicitamente")
    atomic_json(dest, data)
    print("Politica registrada:", dest)
    print("No se ejecuto una prueba de carga ni se certifico seguridad por ausencia de errores.")

if __name__ == "__main__":
    main()
