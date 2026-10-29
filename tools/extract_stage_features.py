"""User-only stage responses from explicit equivalent layers and predefined non-test inputs."""
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/local.yaml');p.add_argument('--architecture',required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True,help='CSV image_id,image_path,split,image_sha256 predeclarado');p.add_argument('--layers',type=Path,required=True,help='JSON stage -> {path,equivalence_rationale}');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();import torch,pandas as pd
    from PIL import Image
    from utils.architectures import get_arch
    from utils.config_loader import load_config
    from utils.manual_gates import require_gate
    from utils.splits import assert_split_allowed
    from utils.weights import digest
    cfg=load_config(a.config);require_gate(cfg,cfg['compute']['pc_id'],'READY_FOR_EXPERIMENT');s=get_arch(a.architecture)
    if not a.checkpoint.is_file():raise FileNotFoundError('PENDIENTE: checkpoint propio aun no generado')
    finalmeta=a.checkpoint.with_name("final_meta.json")
    if not finalmeta.is_file():raise RuntimeError("PENDIENTE: final_meta.json propio con SHA256 requerido para activaciones")
    provenance=json.loads(finalmeta.read_text(encoding="utf-8"))
    if provenance.get("sha256")!=digest(a.checkpoint):raise RuntimeError("Checkpoint final alterado/no propio")
    if s.is_detector:
        from utils.pretrained import configure_ultralytics
        configure_ultralytics();from ultralytics import YOLO,RTDETR
        model=(RTDETR(str(a.checkpoint)) if s.family=='rtdetr' else YOLO(str(a.checkpoint))).model.cpu().eval()
    else:
        from utils.classifier_models import load_classifier
        model,meta=load_classifier(a.checkpoint,'cpu')
    inputs=pd.read_csv(a.inputs);layers=json.loads(a.layers.read_text(encoding='utf-8'));modules=dict(model.named_modules());rows=[]
    for stage,spec in layers.items():
        if not spec.get('equivalence_rationale') or spec['path'] not in modules:raise ValueError('Capa inexistente o equivalencia no justificada')
    # Validate every split and byte hash before any pixel decoding.
    for r in inputs.itertuples():
        assert_split_allowed(r.split,'outer_validation',cfg)
        if r.split=='test':raise RuntimeError('Test no permitido para elegir/inspeccionar features')
        if digest(r.image_path)!=r.image_sha256:raise RuntimeError('Input alterado')
    for r in inputs.itertuples():
        captured={};handles=[]
        for stage,spec in layers.items():
            def hook(m,i,o,key=stage):
                if not torch.is_tensor(o):raise ValueError('Punto no tensorial; seleccione submodulo equivalente')
                captured[key]=float(o.detach().abs().mean().cpu())
            handles.append(modules[spec['path']].register_forward_hook(hook))
        try:
            if s.is_detector:
                from torchvision.transforms.functional import to_tensor
                x=to_tensor(Image.open(r.image_path).convert('RGB').resize((640,640))).unsqueeze(0)
            else:
                from utils.classifier_data import build_transforms
                x=build_transforms(cfg,meta['norm'],train=False)(Image.open(r.image_path).convert('RGB')).unsqueeze(0)
            with torch.no_grad():model(x)
            for stage,value in captured.items():rows.append({'architecture':s.name,'stage':stage,'response':value,'input_id':r.image_id,'model_sha256':digest(a.checkpoint)})
        finally:
            for h in handles:h.remove()
    a.output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(a.output,index=False)
    from utils.durable import atomic_json
    atomic_json(a.output.with_suffix('.provenance.json'),{'input_sha256':digest(a.inputs),'layers_sha256':digest(a.layers),'checkpoint_sha256':digest(a.checkpoint),'response_definition':'mean absolute activation; units are model-specific, not accuracy','layers':layers,'normalization':'backend input transform recorded by model/task'})
    return 0
if __name__=='__main__':raise SystemExit(main())
