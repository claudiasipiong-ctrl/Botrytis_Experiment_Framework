"""Comprobaciones QA invocadas por qa.py. Salidas: 0 PASS, 10 WARN, 11 SKIP, 12 BLOCKED, 1 FAIL."""
from __future__ import annotations

# Stdlib bootstrap: records import/CLI/config failures before optional dependencies.
import sys as _boot_sys
from pathlib import Path as _BootPath
_boot_root = _BootPath(__file__).resolve().parents[1]
_boot_sys.path.insert(0, str(_boot_root / "src"))
from utils import bootstrap as _bootstrap
_bootstrap.install(__file__, _boot_root)
import argparse
import ast
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("check")
    p.add_argument("--config", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--machine-id")
    a = p.parse_args()
    if a.check == "static":
        files = [*ROOT.glob("src/**/*.py"), *ROOT.glob("tools/*.py"), *ROOT.glob("tests/*.py")]
        for file in files:
            ast.parse(file.read_text(encoding="utf-8-sig"), filename=str(file))
        print(f"PASS: AST de {len(files)} archivos; no se importaron modelos")
        return 0
    if a.check == "dependencies":
        from packaging.requirements import Requirement
        rows = []
        for filename in ("requirements.txt", "requirements-dev.txt"):
            for line in (ROOT / filename).read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                req = Requirement(line)
                try:
                    version = importlib.metadata.version(req.name)
                except importlib.metadata.PackageNotFoundError:
                    version = None
                rows.append({"name": req.name, "required": str(req.specifier), "installed": version,
                             "ok": version is not None and version in req.specifier})
        for name in ("torch", "torchvision"):
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                version = None
            rows.append({"name": name, "installed": version, "ok": version is not None,
                         "required": "pareja compatible oficial + misma en tres PCs; QA importa ambas"})
        dump(a.output / "dependency_versions.json", rows)
        if sys.version_info < (3, 10):
            return 1
        return 0 if all(r["ok"] for r in rows) else 12
    from utils.config_loader import load_config, cfg_path, require
    cfg = load_config(a.config)
    from orchestration.machine import read_binding, machine_paths
    from utils.hashing import sha256_file
    if a.check == "configuration":
        from orchestration.pc_registry import PCRegistry
        require(cfg, "classes.fruit_class_id", "classes.sign_class_id", "outer_split.seed")
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            print("BLOCKED: CUDA_VISIBLE_DEVICES heredado remapea dispositivos; verifique correspondencia UUID antes de utilizar este perfil")
            return 12
        b = read_binding(cfg)
        if a.machine_id != b["machine_id"]:
            raise ValueError("--machine-id es obligatorio y debe coincidir con la vinculacion")
        owners = PCRegistry(cfg).ownership()
        dump(a.output / "configuration.json", {"binding": b, "owners": owners, "profile": cfg["project"]["profile"],
                                              "raw_images_dir": cfg["paths"]["raw_images_dir"], "test_locked": cfg["outer_split"]["test_locked"]})
        return 0
    if a.check == "hardware":
        import psutil
        from utils.reproducibility import collect_environment
        env = collect_environment(cfg=cfg, project_root=ROOT)
        env["ram_available_bytes"] = psutil.virtual_memory().available
        env["disk"] = {k: psutil.disk_usage(str(ROOT))._asdict() for k in ["project_volume"]}
        env["cpu_count_logical"] = psutil.cpu_count()
        env["cpu_count_physical"] = psutil.cpu_count(logical=False)
        dump(a.output / "hardware.json", env)
        import torch, torchvision, ultralytics, timm, optuna
        print("Imports ML:", torch.__version__, torchvision.__version__, ultralytics.__version__, timm.__version__, optuna.__version__)
        detail = subprocess.run(["nvidia-smi", "-q", "-d", "TEMPERATURE,POWER"], capture_output=True, text=True, timeout=15)
        (a.output / "nvidia_limits.txt").write_text(detail.stdout + "\n" + detail.stderr, encoding="utf-8")
        if os.name != "nt":
            print("BLOCKED: estos perfiles requieren Python nativo Windows, no un entorno WSL accidental")
            return 12
        if not torch.cuda.is_available():
            print("FAIL: CUDA no disponible")
            return 1
        print("Dispositivos:", [(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory)
                                for i in range(torch.cuda.device_count())])
        if torch.cuda.device_count() != 1:
            print("BLOCKED: el protocolo y fixtures actuales requieren una GPU visible por PC; varias GPUs necesitan validar aislamiento y correspondencia UUID antes de ejecutar cargas")
            return 12
        return 0
    if a.check == "dataset_zip":
        b = read_binding(cfg)
        zip_path = Path(b["dataset_zip"]) if b.get("dataset_zip") else None
        if zip_path is None:
            images=cfg_path(cfg,"paths.raw_images_dir")
            labels=cfg_path(cfg,"paths.raw_labels_dir")
            names=[p for p in images.iterdir() if p.is_file() and p.suffix.lower() in set(cfg["dataset"]["image_extensions"])]
            labs=[p for p in labels.glob("*.txt") if p.name!="classes.txt"]
            im={p.stem.casefold() for p in names};la={p.stem.casefold() for p in labs}
            dump(a.output/"dataset_source.json",{"source":"directorio activo local, ZIP no requerido", "images":len(names),"labels":len(labs),"without_label":sorted(im-la),"without_image":sorted(la-im),"expected_reference":cfg["dataset"]["n_images_expected"]})
            if im!=la:return 1
            return 10 if len(names)!=int(cfg["dataset"]["n_images_expected"]) else 0
        from orchestration.exchange import _safe_name
        details = {"zip": str(zip_path), "zip_sha256": sha256_file(zip_path)}
        errors = []
        with zipfile.ZipFile(zip_path) as z:
            infos = z.infolist()
            for info in infos:
                _safe_name(info.filename.rstrip("/"))
            if len(infos) != len({i.filename.casefold() for i in infos}):
                errors.append("miembros duplicados")
            bad = z.testzip()
            if bad:
                errors.append("CRC: " + bad)
            roots = {i.filename.rsplit("/images/", 1)[0] for i in infos if "/images/" in i.filename
                     and not i.is_dir() and not any(part.startswith("_") for part in Path(i.filename).parts)}
            if len(roots) != 1:
                raise ValueError(f"Raiz images activa ambigua: {roots}")
            prefix = roots.pop() + "/"
            extensions = set(cfg["dataset"]["image_extensions"])
            images = {i.filename[len(prefix + "images/"):]: i for i in infos
                      if i.filename.startswith(prefix + "images/") and not i.is_dir() and Path(i.filename).suffix.lower() in extensions}
            labels = {i.filename[len(prefix + "labels/"):]: i for i in infos
                      if i.filename.startswith(prefix + "labels/") and not i.is_dir() and i.filename.endswith(".txt") and Path(i.filename).name != "classes.txt"}
            details.update(active_images=len(images), active_labels=len(labels),
                           expected_images=cfg["dataset"]["n_images_expected"],
                           excluded_images=sum(1 for i in infos if "/_off/images/" in i.filename))
            tree_file = ROOT / "arbol_dataset.md"
            if tree_file.is_file():
                section, listed = None, []
                for line in tree_file.read_text(encoding="utf-8-sig").splitlines():
                    if line.startswith("  - "):
                        section = line.strip()[2:]
                    elif section == "images/" and line.startswith("    - "):
                        listed.append(line.strip()[2:])
                details["tree_reference"] = {"listed_active_images": len(listed),
                    "only_in_zip": sorted(set(images) - set(listed)), "only_in_tree": sorted(set(listed) - set(images)),
                    "note": "arbol documental, no prueba de integridad"}
            if len(images) != int(cfg["dataset"]["n_images_expected"]):
                errors.append("Conteo real distinto del esperado; no se cambia automaticamente")
            image_stems = {Path(n).stem for n in images}
            label_stems = {Path(n).stem for n in labels}
            if image_stems != label_stems:
                errors.append(f"Sin label: {sorted(image_stems-label_stems)}; labels sin imagen: {sorted(label_stems-image_stems)}")
            for names, key in ((images, "paths.raw_images_dir"), (labels, "paths.raw_labels_dir")):
                local = cfg_path(cfg, key)
                import hashlib
                for name, info in names.items():
                    h = hashlib.sha256()
                    with z.open(info) as f:
                        for chunk in iter(lambda: f.read(1 << 20), b""):
                            h.update(chunk)
                    target = local / name
                    if not target.is_file() or sha256_file(target) != h.hexdigest():
                        errors.append("Extraido ausente/distinto del ZIP: " + str(target))
            details["errors"] = errors
        dump(a.output / "dataset_zip.json", details)
        return 1 if errors else 0
    if a.check == "dataset_validation":
        dest = a.output / "dataset_validation"
        dest.mkdir()
        overrides = {"paths.artifacts_dir": dest / "artifacts", "paths.environments_dir": dest / "environments",
                     "paths.figures_dir": dest / "figures", "paths.reports_dir": dest / "reports",
                     "paths.tables_dir": dest / "tables"}
        command = [sys.executable, str(ROOT / "src/01_validate_dataset.py"), "--config", str(Path(a.config).resolve()),
                   "--pc-id", read_binding(cfg)["pc_id"], "--runs-dir", str(dest / "runs"), "--no-figures"]
        for k, value in overrides.items():
            command += ["--set", f"{k}={value}"]
        print("Comando:", command, flush=True)
        code = subprocess.call(command, cwd=ROOT)
        if code:
            return 1
        report = json.loads((dest / "artifacts/dataset_validation.json").read_text(encoding="utf-8"))
        print(report)
        return 10 if report.get("n_warnings", 0) else 0
    if a.check == "prepared_data":
        from utils.prepared import load_prepared_data
        from utils.scientific_audit import check_no_leakage, check_detector_fruit_only, check_gt_crops, check_test_locked
        data = load_prepared_data(cfg, with_folds=True)
        runs = []
        for metadata in cfg_path(cfg, "paths.runs_dir").rglob("metadata.json"):
            runs.append({"dir": metadata.parent, "meta": json.loads(metadata.read_text(encoding="utf-8"))})
        reports = [check_no_leakage(cfg, data), check_detector_fruit_only(cfg, data),
                   check_gt_crops(cfg, data, runs), check_test_locked(cfg, runs)]
        dump(a.output / "prepared_integrity.json", [r.as_dict() for r in reports])
        if any(r.status == "FAIL" for r in reports):
            return 1
        return 10 if any(r.status not in {"PASS", "N/A"} for r in reports) else 0
    if a.check == "environment":
        from orchestration.pc_registry import PCRegistry
        from utils.reproducibility import collect_environment, compare_environments
        from utils.io import read_json
        registry = PCRegistry(cfg)
        current = collect_environment(cfg=cfg, project_root=ROOT)
        reference = Path(cfg["compute"]["reference_environment"])
        if not reference.is_absolute():
            reference = ROOT / reference
        ref = read_json(reference)
        fields = [k for k in cfg["compute"]["critical_environment_fields"] if k not in {"git_commit", "code_hash", "dataset_hash", "split_hash"}]
        missing_fields = [f for f in fields if current.get(f) is None or ref.get(f) is None]
        differences = compare_environments(ref, current, fields) + registry.environment_mismatches()
        missing_peers = registry.missing_environments()
        dump(a.output / "environment_comparison.json", {"differences": differences, "missing_fields": missing_fields, "missing_peers": missing_peers})
        if differences:
            return 1
        return 12 if missing_fields or missing_peers else 0
    if a.check == "safety":
        from utils.safety import SafetyMonitor
        result = SafetyMonitor(cfg, a.output / "safety_telemetry.jsonl").check()
        print(result)
        # El sensor CPU ausente esta reconocido explicitamente en la politica; se informa como limitacion.
        return 0
    if a.check == "clock":
        current = datetime.now().astimezone()
        minutes = int(current.utcoffset().total_seconds() / 60)
        expected = cfg.get("operations", {}).get("expected_utc_offset_minutes", -300)
        result = {"time": current.isoformat(), "utc_offset_minutes": minutes, "expected": expected,
                  "clock_accuracy": "requiere verificacion humana antes de --clock-verified"}
        code = 0 if minutes == expected else 1
        if os.name == "nt":
            status = subprocess.run(["w32tm", "/query", "/status"], capture_output=True, text=True, timeout=15)
            result.update(w32tm_code=status.returncode, stdout=status.stdout, stderr=status.stderr)
            if status.returncode:
                code = 10 if code == 0 else code
        else:
            code = 12
        dump(a.output / "clock.json", result)
        return code
    if a.check == "exchange_state":
        from orchestration.exchange import assert_no_pending_import, sync_root, verify_zip
        assert_no_pending_import(cfg)
        errors = []
        for receipt in (sync_root(cfg) / "published").glob("*.json"):
            meta = json.loads(receipt.read_text(encoding="utf-8"))
            for e in meta["files"]:
                obj = sync_root(cfg) / "objects" / e["sha256"]
                if not obj.is_file() or sha256_file(obj) != e["sha256"]:
                    errors.append(str(obj))
        paths = machine_paths(read_binding(cfg)["machine_id"])
        zips = []
        for folder in ("import_dir", "export_dir"):
            paths[folder].mkdir(parents=True, exist_ok=True)
            for path in paths[folder].glob("*.zip"):
                verify_zip(path)
                zips.append(str(path))
        dump(a.output / "exchange_integrity.json", {"bad_objects": errors, "verified_zips": zips,
                                                   "real_transfer_end_to_end": "pendiente de evidencia de las tres PCs"})
        return 1 if errors else 0
    if a.check == "storage_io":
        # Acotado a 16 MiB dentro de logs_qa. No estima lectura fria ni velocidad de entrenamiento.
        target = a.output / "io_probe.bin"
        payload = bytes(range(256)) * 4096
        start = time.perf_counter()
        with target.open("xb") as f:
            for _ in range(16):
                f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        write_seconds = time.perf_counter() - start
        start = time.perf_counter()
        digest = sha256_file(target)
        read_seconds = time.perf_counter() - start
        dump(a.output / "storage_io.json", {"bytes": target.stat().st_size, "write_fsync_seconds": write_seconds,
             "read_hash_seconds": read_seconds, "sha256": digest, "free_bytes": shutil.disk_usage(a.output).free,
             "limit": "lectura posiblemente en cache; no acredita rendimiento sostenido ni plazo de exportacion"})
        return 0
    if a.check == "cuda_selftest":
        from utils.safety import SafetyMonitor
        monitor = SafetyMonitor(cfg, a.output / "cuda_telemetry.jsonl")
        monitor.check()
        import torch
        device = int(cfg["compute"]["device"]) if str(cfg["compute"]["device"]).isdigit() else 0
        actual_uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
        if actual_uuid is not None:
            normalize = lambda value: str(value).lower().removeprefix("gpu-").strip("{}")
            if normalize(actual_uuid) != normalize(monitor.policy["gpu_uuid"]):
                raise RuntimeError("UUID CUDA no corresponde a la GPU supervisada")
        with torch.cuda.device(device):
            x = torch.eye(128, device="cuda", dtype=torch.float16)
            y = torch.ones((128, 128), device="cuda", dtype=torch.float16)
            z = x @ y
            torch.cuda.synchronize()
            if not torch.isfinite(z).all().item() or not torch.equal(z, y):
                raise RuntimeError("Operacion CUDA FP16 no coincide con identidad x unos")
            print({"shape": [128, 128], "iterations": 1, "allocated_bytes": torch.cuda.max_memory_allocated(device),
                   "cuda_runtime": torch.version.cuda, "device": torch.cuda.get_device_name(device)})
            del x, y, z
            torch.cuda.empty_cache()
        monitor.check()
        return 0
    raise ValueError("Comprobacion desconocida: " + a.check)

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(12)
    except Exception as exc:
        # Una entrada previa ausente impide ejecutar el check; un hash falso/corrupcion si es FAIL.
        blocked = type(exc).__name__ == "BlockingTodoError" or any(
            text in str(exc) for text in ("No existe la configuracion", "No existe ", "Falta "))
        import traceback
        traceback.print_exc()
        print(("BLOCKED" if blocked else "FAIL") + ": " + str(exc), file=sys.stderr)
        raise SystemExit(12 if blocked else 1)
