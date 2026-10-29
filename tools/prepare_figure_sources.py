"""Adapt v4 artefacts to the catalogue; never opens raw pixels or final test."""
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/local.yaml');p.add_argument('--output',type=Path,default=ROOT/'reports/v4/figure_sources.json');a=p.parse_args()
    import pandas as pd
    from utils.config_loader import load_config,cfg_path
    from utils.semantics import semantic_fingerprint
    from utils.pairing import SEMANTICS_VERSION
    from utils.weights import digest
    from utils.durable import atomic_json
    cfg=load_config(a.config);art=cfg_path(cfg,'paths.artifacts_dir');out=a.output.parent/'figure_data';out.mkdir(parents=True,exist_ok=True);index={}
    def save(name,df,split,phase,metric,unit,denominator,sources):
        if df.empty:return
        f=out/(name+'.csv');df.to_csv(f,index=False)
        index[name]={'path':str(f.resolve()),'source_sha256':digest(f),'dataset':str(json.loads((art/'dataset_validation.json').read_text(encoding='utf-8'))['dataset_hash']) if (art/'dataset_validation.json').is_file() else 'PENDIENTE','split':split,'phase':phase,'metric':metric,'unit':unit,'denominator':denominator,'configuration':semantic_fingerprint(cfg),'semantic_version':SEMANTICS_VERSION,'upstream':[{'path':str(p),'sha256':digest(p)} for p in sources]}
    def read(path):return pd.read_csv(path) if path.is_file() else pd.DataFrame()
    summary_path=art/'classification_per_image.csv';summary=read(summary_path)
    if len(summary):
        # Inventory is descriptive and frozen before modeling. No test scores are consumed.
        save('image_summary',summary,'dataset_inventory','preparation','n_fruits','frutos/imagen',len(summary),[summary_path])
    crop_path=art/'classification_manifest.csv';crop=read(crop_path)
    if len(crop):
        if 'semantic_version' not in crop or not (crop.semantic_version==SEMANTICS_VERSION).all():raise RuntimeError('Manifiesto de crops obsoleto')
        save('class_balance',crop.groupby(['outer_split','label']).size().rename('count').reset_index().rename(columns={'outer_split':'split'}),'dataset_inventory','preparation','count','frutos',len(crop),[crop_path])
        save('crop_counts',crop.groupby('image_id').size().rename('n_crops').reset_index(),'dataset_inventory','preparation','n_crops','crops/imagen',crop.image_id.nunique(),[crop_path])
        if 'max_sign_coverage' in crop:save('coverage',crop.assign(coverage_percent=100*crop.max_sign_coverage),'dataset_inventory','preparation','coverage descriptive','%',len(crop),[crop_path])
        save('bbox_sizes',crop.assign(width_px=crop.x2-crop.x1,height_px=crop.y2-crop.y1,class_name='arandano_a'),'dataset_inventory','preparation','bbox','px',len(crop),[crop_path])
    pair=read(art/'pairing_audit.csv')
    if len(pair):save('pairing',pair,'dataset_inventory','annotation_audit','IoU 0↔1','IoU',len(pair),[art/'pairing_audit.csv'])
    for name,file,phase,split in [('comparison','validation_detectors.csv','outer_validation','validation'),('classifier_comparison','validation_classifiers.csv','outer_validation','validation'),('trials','hpo_trials_all.csv','hpo','inner_validation'),('robustness','robustness_summary.csv','robustness','inner_validation')]:
        path=cfg_path(cfg,'paths.tables_dir')/file;df=read(path)
        if len(df):
            df=df.rename(columns={"number":"trial","value":"score","vram_train_peak_mb":"vram_peak_mb"})
        if len(df):save(name,df,split,phase,'map50_95' if name=='comparison' else 'source metric','fraccion',len(df),[path])
    # The selected cohort comes from actual robustness CSVs; external validation never becomes an epoch curve.
    from utils.architectures import archs,get_arch
    from utils.pairing import SEMANTICS_VERSION
    from utils.hashing import sha256_json
    runroot=cfg_path(cfg,"paths.runs_dir")
    detector_seed_rows=[];detector_seed_sources=[];baseline_rows=[];baseline_sources=[]
    for s in archs():
        per_path=runroot/s.slug/"robustness/per_seed.csv";per=read(per_path)
        if s.is_detector and len(per):
            detector_seed_rows.append(per.assign(architecture=s.name,map50_95=per.score));detector_seed_sources.append(per_path)
        best_path=cfg_path(cfg,"paths.best_configs_dir")/(s.slug+".json")
        base_path=runroot/s.slug/"baseline/summary.json"
        if s.is_detector and best_path.is_file() and base_path.is_file():
            best=json.loads(best_path.read_text(encoding="utf-8"));base=json.loads(base_path.read_text(encoding="utf-8"))
            if best.get("semantic_fingerprint")==semantic_fingerprint(cfg) and base.get("semantic_fingerprint")==semantic_fingerprint(cfg):
                seed=str(cfg["detector"]["baseline"]["seed"]);matched=best.get("per_seed_scores",{}).get(seed)
                if matched is not None and base.get("score_mean") is not None:
                    baseline_rows.append({"architecture":s.name,"baseline":base["score_mean"],"hpo":matched,"seed":int(seed)})
                    baseline_sources.extend([best_path,base_path])
    if detector_seed_rows:save("robustness",pd.concat(detector_seed_rows,ignore_index=True),"inner_validation","robustness","map50_95","fraccion","seeds; cada score es media de folds internos",detector_seed_sources)
    if baseline_rows:save("baseline_hpo",pd.DataFrame(baseline_rows),"inner_validation","baseline_vs_robustness_matched_seed","map50_95","fraccion","misma seed42 y folds internos",baseline_sources)
    for task,key in (("detector","selected_detector_history"),("classifier","selected_classifier_history")):
        selected=art/("selected_"+task+".json")
        if not selected.is_file():continue
        choice=json.loads(selected.read_text(encoding="utf-8"))
        if choice.get("semantic_fingerprint")!=semantic_fingerprint(cfg):continue
        spec=get_arch(choice["architecture"]);best_path=cfg_path(cfg,"paths.best_configs_dir")/(spec.slug+".json")
        if not best_path.is_file():continue
        best=json.loads(best_path.read_text(encoding="utf-8"));cid=best["config_id"];histories=[];sources=[selected,best_path]
        for seed in cfg["robustness"]["seeds"]:
            path=runroot/spec.slug/"robustness"/cid/("seed_"+str(seed))/"history.csv"
            meta=path.with_name("metadata.json")
            if not path.is_file() or not meta.is_file():continue
            m=json.loads(meta.read_text(encoding="utf-8"))
            if m.get("semantic_fingerprint")!=semantic_fingerprint(cfg) or m.get("status")!="COMPLETE":continue
            h=read(path).rename(columns={"val_ap":"ap_positive","val_accuracy":"accuracy","val_f1":"f1"})
            if "run_id" not in h:continue
            histories.append(h);sources.append(path)
        if histories:
            history=pd.concat(histories,ignore_index=True)
            save(key,history,"inner_validation","robustness_selected_configuration",spec.name+" selected internal objective","fraccion/loss",str(history.run_id.nunique())+" fold-seed runs; N por epoca",sources)
            # General per-epoch plots refer to the same explicit comparable cohort.
            alias="history" if task=="detector" else "classifier_history"
            index[alias]=dict(index[key])
    # Direct per-IoU AP table is available without reading models/pixels.
    ap_rows=[];ap_sources=[]
    for s in archs("detection"):
        p=art/"outer_validation"/(s.slug+".json")
        if not p.is_file():continue
        value=json.loads(p.read_text(encoding="utf-8"))
        if value.get("semantic_fingerprint")!=semantic_fingerprint(cfg):continue
        values=value.get("ap_per_iou") or []
        if len(values)==10:
            ap_rows.extend({"architecture":s.name,"iou":round(.5+.05*i,2),"ap":v} for i,v in enumerate(values));ap_sources.append(p)
    if ap_rows:save("ap_iou",pd.DataFrame(ap_rows),"validation","outer_validation","AP by evaluation IoU","fraccion","GT de validacion externa",ap_sources)
    # Preserve manually supplied contracts; generated adapters update only their own current sources.
    if a.output.exists():
        previous=json.loads(a.output.read_text(encoding="utf-8"));index={**previous,**index}
        from datetime import datetime,timezone
        import shutil
        shutil.copy2(a.output,a.output.with_name(a.output.stem+"_previous_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json"))
    atomic_json(a.output,index);print('Fuentes preparadas:',len(index),'Sin pixeles/modelos/test. Otras fuentes requieren contratos explicitos.');return 0
if __name__=='__main__':raise SystemExit(main())
