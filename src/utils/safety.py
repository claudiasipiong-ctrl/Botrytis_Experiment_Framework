"""Supervision configurable; no modifica potencia, reloj, voltaje ni protecciones del hardware."""
from __future__ import annotations
import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from utils.config_loader import cfg_path, resolve_path

class SafetyStop(BaseException):
    """Sale del job sin que Optuna lo transforme en reintentos."""

def check_stop(cfg):
    p = cfg_path(cfg, "paths.queue_dir") / f"stop_{os.getpid()}.json"
    if p.exists():
        raise SafetyStop(p.read_text(encoding="utf-8"))

def read_policy(cfg):
    p = resolve_path(cfg.get("operations", {}).get("safety_policy_file", "artifacts/safety_policy.json"))
    if not p.is_file():
        raise FileNotFoundError(f"Falta politica verificada: {p}; vea GUIA_EJECUCION_3PCs.md")
    policy = json.loads(p.read_text(encoding="utf-8"))
    mode = policy.get("mode", "enforce")
    if mode not in ("enforce", "observe"):
        raise ValueError("Modo de supervision desconocido")
    if mode == "observe":
        if not all(policy.get(k) for k in ("gpu_uuid", "gpu_model", "configured_by", "configured_at")):
            raise ValueError("Politica de observacion incompleta")
        if policy.get("user_requested_observation_only") is not True:
            raise ValueError("Solo registro requiere una decision explicita del usuario")
        return policy
    required = ("gpu_uuid", "gpu_model", "max_gpu_temperature_c", "gpu_limit_source",
                "min_available_ram_bytes", "min_free_disk_bytes", "cpu_protection_source")
    if any(policy.get(k) in (None, "", 0) for k in required):
        raise ValueError(f"Politica incompleta: {required}")
    if policy.get("cpu_thermal_protection_verified") is not True:
        raise ValueError("Falta confirmar las protecciones termicas CPU del fabricante")
    for k in ("max_gpu_temperature_c", "min_available_ram_bytes", "min_free_disk_bytes"):
        if type(policy[k]) not in (float, int) or policy[k] <= 0:
            raise ValueError(f"Limite sin verificar: {k}")
    return policy

def gpu_reading(uuid):
    fields = ["uuid", "name", "temperature.gpu", "utilization.gpu", "memory.total",
              "memory.used", "power.draw", "power.limit"]
    result = subprocess.run(["nvidia-smi", "-i", uuid, "--query-gpu=" + ",".join(fields),
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=5, check=True)
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != len(fields):
        raise RuntimeError("Telemetria GPU ambigua/incompleta")
    row = dict(zip(fields, (x.strip() for x in rows[0])))
    if row["uuid"] != uuid:
        raise RuntimeError("UUID GPU distinto de la politica")
    row["temperature.gpu"] = float(row["temperature.gpu"])
    return row

class SafetyMonitor:
    def __init__(self, cfg, log_path):
        self.cfg, self.log_path = cfg, Path(log_path)
        self.policy = read_policy(cfg)

    def check(self):
        import psutil
        policy = self.policy
        observe = policy.get("mode", "enforce") == "observe"
        result = {"time": datetime.now(timezone.utc).isoformat(), "status": "PASS",
                  "mode": "observe" if observe else "enforce", "resource_limits_enforced": not observe,
                  "cpu_thermal_protection_verified": policy.get("cpu_thermal_protection_verified") is True}
        try:
            selected = str(self.cfg.get("compute", {}).get("device", "auto"))
            selected = "0" if selected == "auto" else selected
            identity = subprocess.run(["nvidia-smi", "-i", selected, "--query-gpu=uuid", "--format=csv,noheader"],
                                      capture_output=True, text=True, timeout=5, check=True).stdout.strip()
            if identity != policy["gpu_uuid"]:
                raise RuntimeError("La GPU seleccionada para entrenar no coincide con la politica")
            gpu = gpu_reading(policy["gpu_uuid"])
            result["gpu"] = gpu
            if gpu["name"] != policy["gpu_model"]:
                raise RuntimeError("Modelo GPU distinto de la politica")
            if not observe and gpu["temperature.gpu"] >= policy["max_gpu_temperature_c"]:
                raise RuntimeError("Temperatura GPU alcanzo el limite verificado")
            vm = psutil.virtual_memory()
            result["available_ram_bytes"] = vm.available
            result["cpu_utilization_percent"] = psutil.cpu_percent(interval=None)
            if not observe and vm.available < policy["min_available_ram_bytes"]:
                raise RuntimeError("RAM disponible por debajo de la reserva verificada")
            for name in ("paths.artifacts_dir", "paths.runs_dir"):
                p = cfg_path(self.cfg, name)
                while not p.exists():
                    p = p.parent
                free = psutil.disk_usage(p).free
                result[name + ".free_bytes"] = free
                if not observe and free < policy["min_free_disk_bytes"]:
                    raise RuntimeError("Disco por debajo de la reserva verificada")
            temperatures = getattr(psutil, "sensors_temperatures", lambda: {})()
            cpu = [r.current for key, rows in temperatures.items()
                   if any(n in key.lower() for n in ("coretemp", "k10temp", "cpu"))
                   for r in rows if r.current is not None]
            result["cpu_temperatures_c"] = cpu
            if observe:
                result["cpu_temperature_status"] = (
                    "OBSERVACION: temperaturas registradas sin umbral de parada"
                    if cpu else "NO OBSERVABLE: protecciones OEM no verificadas por esta lectura"
                )
            elif cpu:
                limit = policy.get("max_cpu_temperature_c")
                if not isinstance(limit, (float, int)) or limit <= 0:
                    raise RuntimeError("CPU con sensores sin limite configurado")
                if max(cpu) >= limit:
                    raise RuntimeError("CPU alcanzo el limite termico verificado")
            elif policy.get("cpu_sensor_unavailable_acknowledged") is not True:
                raise RuntimeError("Sin sensor CPU: requiere verificacion OEM y reconocimiento explicito")
            else:
                result["cpu_temperature_status"] = "WARN: no observable; protecciones OEM verificadas por usuario"
        except BaseException as exc:
            result.update(status="BLOCKED", error=str(exc))
            raise
        finally:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result
