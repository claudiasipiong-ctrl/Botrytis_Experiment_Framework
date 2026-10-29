"""Manual architecture inspection on CPU, without pretrained downloads or metrics."""
from __future__ import annotations

import sys as _boot_sys
from pathlib import Path as _BootPath
_boot_root=_BootPath(__file__).resolve().parents[1]
_boot_sys.path.insert(0,str(_boot_root/"src"))
from utils import bootstrap as _bootstrap
_bootstrap.install(__file__,_boot_root)
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--architecture',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();from utils.architectures import get_arch
    s=get_arch(a.architecture)
    if s.is_detector:
        from utils.pretrained import configure_ultralytics
        configure_ultralytics()
        from ultralytics import YOLO,RTDETR
        wrapper=RTDETR('rtdetr-resnet50.yaml') if s.family=='rtdetr' else YOLO(s.slug+'.yaml')
        model=wrapper.model
    else:
        from utils.classifier_models import build_classifier
        model,_=build_classifier(s,pretrained=False)
    modules=[{'path':n,'module':type(m).__name__} for n,m in model.named_modules()]
    result={'architecture':s.name,'initialization':'scratch_architecture_inspection_only','trained':False,'modules':modules,'CIB_present':any(r['module']=='CIB' for r in modules),'C2PSA_present':any(r['module']=='C2PSA' for r in modules),'attention_modules':[r for r in modules if any(t in r['module'].lower() for t in ('attn','attention','aifi','psa'))]}
    from utils.durable import atomic_json
    atomic_json(a.output,result);print('Grafo guardado; no es resultado entrenado.');return 0
if __name__=='__main__':raise SystemExit(main())
