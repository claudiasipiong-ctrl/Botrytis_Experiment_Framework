"""Registro manual de gates; solo el usuario debe ejecutar record/release-test en persona."""
from __future__ import annotations

import sys as _boot_sys
from pathlib import Path as _BootPath
_boot_root=_BootPath(__file__).resolve().parents[1]
_boot_sys.path.insert(0,str(_boot_root/"src"))
from utils import bootstrap as _bootstrap
_bootstrap.install(__file__,_boot_root)
import argparse,json,sys,socket
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from utils.manual_gates import GATES,ledger_path,require_gate
from utils.config_loader import load_config,cfg_path,PROJECT_ROOT
from utils.durable import atomic_json
from utils.weights import digest
from utils.hashing import compute_code_hash
from utils.semantics import semantic_fingerprint
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('status','record','release-test'));p.add_argument('--config',default='configs/local.yaml')
    p.add_argument('--pc-id',required=True,choices=('PC-01','PC-02','PC-03'));p.add_argument('--gate',choices=GATES)
    p.add_argument('--operator');p.add_argument('--evidence');p.add_argument('--confirm-personal-execution',action='store_true')
    a=p.parse_args();cfg=load_config(a.config);path=ledger_path(cfg,a.pc_id)
    if cfg.get("compute",{}).get("pc_id") not in (a.pc_id, "", None):raise RuntimeError("--pc-id no coincide con la configuracion local; no transferir aprobaciones de otra PC")
    data=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {'pc_id':a.pc_id,'gates':{g:{'state':'NO EJECUTADO'} for g in GATES}}
    if a.action=='status':print(json.dumps(data,ensure_ascii=False,indent=2));return 0
    if not all((a.operator,a.evidence,a.confirm_personal_execution)):p.error('Requiere --operator, --evidence y --confirm-personal-execution; no reemplaza el checklist')
    row=dict(state='APROBADO_POR_USUARIO',operator=a.operator,evidence=a.evidence,utc=datetime.now(timezone.utc).isoformat(),hostname=socket.gethostname(),code_hash=compute_code_hash(PROJECT_ROOT),semantic_fingerprint=semantic_fingerprint(cfg))
    if a.action=='record':
        if not a.gate:p.error('--gate requerido')
        if a.gate=='READY_FOR_EXPERIMENT':
            for g in GATES[:8]:require_gate(cfg,a.pc_id,g)
        data['gates'][a.gate]=row;atomic_json(path,data)
    else:
        require_gate(cfg,a.pc_id,'FREEZE')
        from utils.freeze import load_freeze
        load_freeze(cfg)  # verifica la integridad antes de autorizar acceso al test
        f=Path(cfg['test']['freeze_manifest']);f=f if f.is_absolute() else ROOT/f
        row.update(state='AUTORIZADO_PERSONALMENTE',freeze_sha256=digest(f))
        atomic_json(cfg_path(cfg,'paths.artifacts_dir')/'manual_test_release.json',row)
    print(json.dumps(row,ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
