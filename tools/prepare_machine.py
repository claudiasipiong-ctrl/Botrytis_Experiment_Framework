"""Prepare local paths ONLY. User runs physically; never connects to another machine."""
from __future__ import annotations

import sys as _boot_sys
from pathlib import Path as _BootPath
_boot_root=_BootPath(__file__).resolve().parents[1]
_boot_sys.path.insert(0,str(_boot_root/"src"))
from utils import bootstrap as _bootstrap
_bootstrap.install(__file__,_boot_root)
import argparse,json,sys,socket
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from utils.durable import atomic_json
MAPPING={'PC-01':'i21315','PC-02':'i21314','PC-03':'i21313'}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pc-id',required=True,choices=MAPPING);p.add_argument('--machine-id',required=True,choices=MAPPING.values())
    p.add_argument('--dataset-dir',required=True,type=Path);p.add_argument('--config-output',type=Path,default=ROOT/'configs/local.yaml')
    a=p.parse_args()
    if MAPPING[a.pc_id]!=a.machine_id:p.error('Mapa acordado incompatible: PC-01=i21315, PC-02=i21314, PC-03=i21313')
    if str(a.dataset_dir).startswith(("\\\\","//")):raise ValueError("Use una copia del dataset en disco local")
    d=a.dataset_dir.resolve()
    if not (d/'images').is_dir() or not (d/'labels').is_dir():p.error('Seleccione raiz ACTIVA con images/ y labels/; no _off ni respaldos')
    classes=d/'classes.txt'
    if not classes.is_file():classes=d/'labels/classes.txt'
    names=classes.read_text(encoding='utf-8-sig').strip().splitlines()
    if [s.strip() for s in names]!=['arandano_a','arandano_sospechoso','signo_sospechoso_botrytis']:raise RuntimeError('Clases incompatibles, conservar originales y revisar personalmente: '+repr(names))
    cfg={'paths':{'raw_images_dir':str(d/'images'),'raw_labels_dir':str(d/'labels')},'outer_split':{'seed':67},'compute':{'pc_id':a.pc_id,'require_gpu':True}}
    import yaml
    if a.config_output.exists():
        old=yaml.safe_load(a.config_output.read_text(encoding='utf-8-sig'))
        if old!=cfg:raise FileExistsError('Config local distinta preservada; revise y elija --config-output nuevo')
    else:
        a.config_output.parent.mkdir(parents=True,exist_ok=True);a.config_output.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    art=ROOT/'artifacts/v4'
    for name,value in [('fleet_mapping.json',MAPPING),('machine_binding.json',{'schema':2,'pc_id':a.pc_id,'machine_id':a.machine_id,'hostname':socket.gethostname(),'dataset_root':str(d),'project_root':str(ROOT)})]:
        q=art/name
        if q.exists() and json.loads(q.read_text(encoding='utf-8'))!=value:raise FileExistsError(str(q)+' previo distinto preservado')
        atomic_json(q,value)
    print('PREPARADO; QA_MANUAL_PENDIENTE. No se leyeron pixeles, no se crearon splits ni crops.');return 0
if __name__=='__main__':raise SystemExit(main())
