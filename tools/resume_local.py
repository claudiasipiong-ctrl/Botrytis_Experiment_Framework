"""Prepare a concrete local resume config from an existing checkpoint; never invent one."""
from __future__ import annotations

import sys as _boot_sys
from pathlib import Path as _BootPath
_boot_root=_BootPath(__file__).resolve().parents[1]
_boot_sys.path.insert(0,str(_boot_root/"src"))
from utils import bootstrap as _bootstrap
_bootstrap.install(__file__,_boot_root)
import argparse,json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/local.yaml');p.add_argument('--architecture',required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--entry',required=True,choices=('src/06_run_baseline.py','src/07_run_hpo_study.py','src/08_repeat_top_configs.py','src/09_train_outer_candidate.py'));p.add_argument('--execute-personally',action='store_true');p.add_argument('--extra',nargs=argparse.REMAINDER,default=[])
    a=p.parse_args();from utils.config_loader import load_config
    from utils.architectures import get_arch
    from utils.resume import require_checkpoint
    from utils.manual_gates import require_gate
    cfg=load_config(a.config);s=get_arch(a.architecture);path=require_checkpoint(a.checkpoint,s,cfg)
    key='detector' if s.is_detector else 'classifier'
    cmd=[sys.executable,str(ROOT/a.entry),'--config',a.config,'--architecture',s.name,'--set',key+'.resume_checkpoint='+str(path),*a.extra]
    print(json.dumps({'command_argv':cmd,'state':'PREPARADO_PENDIENTE_USUARIO','note':'Reutilice exactamente seed/fold/config del checkpoint. Solo reanuda el entrenamiento especifico, no el estudio completo.'},indent=2))
    if a.execute_personally:
        require_gate(cfg,cfg['compute']['pc_id'],'READY_FOR_EXPERIMENT')
        return subprocess.call(cmd,cwd=ROOT)
    return 0
if __name__=='__main__':raise SystemExit(main())
