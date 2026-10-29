"""Vinculacion explicita de los identificadores de usuario con PC-01/02/03.

El hostname se registra como evidencia y jamas se usa para deducir la asignacion.
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
import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config_loader import PROJECT_ROOT, cfg_path, load_config
from utils.durable import atomic_json
from orchestration.pc_registry import PCRegistry

MACHINES = ("i21315", "i21314", "i21313")
DATASET_NAME = "dataset-botrytis_2026-10-01-20261007T231204Z-1-001.zip"

def machine_paths(machine_id):
    if machine_id not in MACHINES:
        raise ValueError(f"Identificador invalido: {machine_id}")
    base = Path(f"C:/Users/{machine_id}/Documents/s2_no_borrar")
    return {"base": base, "dataset_zip": base / DATASET_NAME,
            "export_dir": base / "Exportar", "import_dir": base / "Importar"}

def binding_path(cfg):
    return cfg_path(cfg, "paths.artifacts_dir") / "machine_binding.json"

def fleet_mapping(cfg):
    p = cfg_path(cfg, "paths.artifacts_dir") / "fleet_mapping.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    if data != {"PC-01":"i21315", "PC-02":"i21314", "PC-03":"i21313"}:
        raise ValueError("fleet_mapping.json debe ser una biyeccion PC-01/02/03 -> i21315/14/13")
    return data

def read_binding(cfg, *, check_host=True):
    p = binding_path(cfg)
    if not p.is_file():
        raise FileNotFoundError("Falta vinculacion explicita: ejecute machine.py bind --machine-id ... --pc-id ...")
    b = json.loads(p.read_text(encoding="utf-8"))
    PCRegistry(cfg).validate(b["pc_id"])
    machine_paths(b["machine_id"])
    if fleet_mapping(cfg)[b["pc_id"]] != b["machine_id"]:
        raise RuntimeError("La vinculacion no coincide con fleet_mapping.json")
    if check_host and b["hostname"] != socket.gethostname():
        raise RuntimeError("La vinculacion fue creada en otro hostname; vincule esta copia local")
    return b

def bind(cfg, machine_id, pc_id):
    PCRegistry(cfg).validate(pc_id)
    paths = machine_paths(machine_id)
    if fleet_mapping(cfg)[pc_id] != machine_id:
        raise ValueError("Asignacion distinta de fleet_mapping.json")
    if os.name != "nt":
        raise RuntimeError("Vinculacion de estas rutas requiere Python nativo Windows")
    if not cfg_path(cfg,"paths.raw_images_dir").is_dir():
        raise FileNotFoundError("Dataset activo local ausente; ejecute tools/prepare_machine.py")
    data = {"schema": 1, "machine_id": machine_id, "pc_id": pc_id, "hostname": socket.gethostname(),
            "created_at": datetime.now().astimezone().isoformat(),
            "dataset_zip": str(paths["dataset_zip"]), "project_root": str(PROJECT_ROOT)}
    p = binding_path(cfg)
    if p.exists():
        old = read_binding(cfg)
        if any(old.get(k) != data[k] for k in ("machine_id", "pc_id", "hostname")):
            raise RuntimeError("Existe otra vinculacion; no se reemplaza automaticamente")
        return old
    for key in ("export_dir", "import_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    atomic_json(p, data)
    return data

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("bind", "show"))
    p.add_argument("--machine-id", choices=MACHINES)
    p.add_argument("--pc-id", choices=("PC-01", "PC-02", "PC-03"))
    p.add_argument("--config")
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.action == "bind" and (not a.machine_id or not a.pc_id):
        p.error("bind requiere --machine-id y --pc-id; la correspondencia la decide el investigador")
    print(json.dumps(bind(cfg, a.machine_id, a.pc_id) if a.action == "bind" else read_binding(cfg), indent=2))

if __name__ == "__main__":
    main()
