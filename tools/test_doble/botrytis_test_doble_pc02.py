"""Dos sistemas finales en PC-02. Extension explicita del protocolo original de siete candidatos.
No entrena ni reajusta; conserva fuentes y publica A/B por separado. CLI --ejecutar-dos-test
registra la autorizacion personal de ejecutar los dos sistemas ya fijados tras el freeze.
"""
from pathlib import Path
import argparse,copy,csv,hashlib,html,json,os,shutil,sys,zipfile

EXPECTED={'yolo26l':'cb0dcf3b09570b1c55967c0c6de46630a7969f409fb4bf2d8183ab3658fb94bb',
          'resnet50':'f9aff5c2b9fe0036236fa0651488f563955eaa18d1e0e5fb4cf08113090f2c00',
          'swin_tiny':'7509ec5c02784e8ec5704dc13bd01a929b8b337c3a9caf68c70ecdb1cc4b6019'}
PAIRS=[('A','resnet50',.69),('B','swin_tiny',.85)]
CODE='e11f772d86f45e5dc3687fce3faa23443a6fcf16146484b3d5f38534cc7695df'
COMMIT='b706f176c8e308969fd3a885d66045ffea027c7e'
DATA='f9cc08785271eecb6566669e1eff7fc1dcb821b8a8a407592a6153fe248018ed'
SPLIT='617a3fc380835763c6f43f84e770df586d96d06aab7de9101a10c44b072ee64b'
NOTICE='Dos sistemas finales; profesor escogera tras ver TEST. No afirmar evaluacion independiente del ganador. Sin reajustes posteriores.'

def emit(event,**kw):print(event+' '+json.dumps(kw,ensure_ascii=False,default=str),flush=True)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,value):
    from utils.durable import atomic_json
    atomic_json(Path(p),value)
def pin_file(p,value):
    from utils.hashing import write_checksum_file
    if p.exists():
        if read(p)!=value:raise RuntimeError('Freeze existente distinto; no se cambia: '+str(p))
    else:write(p,value)
    if p.with_suffix('.sha256').exists():
        if p.with_suffix('.sha256').read_text().split()[0]!=sha(p):raise RuntimeError('Sello freeze distinto')
    else:write_checksum_file(p,p.with_suffix('.sha256'))

def install_resnet(home,dest):
    source=home/'Downloads/ResNet50_para_PC2.zip';side=source.with_suffix('.zip.sha256.txt')
    if sha(source)!=side.read_text().split()[0]:raise RuntimeError('ZIP ResNet50 no coincide con SHA256')
    names=['config.yaml','manifest.json','artifacts/checkpoints/resnet50/final.pt','artifacts/checkpoints/resnet50/final_meta.json',
           'artifacts/best_configs/resnet50.json','artifacts/outer_validation/resnet50.json','artifacts/outer_validation/resnet50_predictions.csv']
    with zipfile.ZipFile(source) as z:
        if sorted(z.namelist())!=sorted(names):raise RuntimeError('Contenido inesperado del paquete ResNet50')
        for name in names:
            p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():
                with z.open(name) as f:
                    h=hashlib.file_digest(f,'sha256').hexdigest()
                if sha(p)!=h:raise RuntimeError('Copia ResNet existente distinta; se conserva')
            else:
                with z.open(name) as f,p.open('xb') as out:shutil.copyfileobj(f,out)
    return dest

def candidates(home,work):
    from utils.config_loader import load_config
    from utils.semantics import semantic_fingerprint
    roots={'yolo26l':home/'bt-hoy-20261103/registro/botrytis_v4/yolo26l-20261103-cb706f17',
           'swin_tiny':home/'bn/botrytis_v4/n_swin_tiny','resnet50':install_resnet(home,work/'fuentes/resnet50')}
    result={}
    for slug,root in roots.items():
        ck=root/f'artifacts/checkpoints/{slug}/final.pt';meta=ck.with_name('final_meta.json');v=root/f'artifacts/outer_validation/{slug}.json';b=root/f'artifacts/best_configs/{slug}.json'
        cfg=load_config(root/'config.yaml');vm=read(v);fm=read(meta);best=read(b)
        if sha(ck)!=EXPECTED[slug] or fm.get('sha256')!=EXPECTED[slug] or vm.get('checkpoint_sha256')!=EXPECTED[slug]:raise RuntimeError('Checkpoint no coincide con validado: '+slug)
        for row in (fm,vm,best):
            if row.get('dataset_hash')!=DATA or row.get('split_hash')!=SPLIT or row.get('code_hash')!=CODE:raise RuntimeError('Procedencia incompatible: '+slug)
        if vm.get('config_id')!=best.get('config_id') or vm.get('semantic_fingerprint')!=semantic_fingerprint(cfg):raise RuntimeError('Configuracion/semantica incompatible: '+slug)
        result[slug]=dict(root=root,checkpoint=ck,validation=v,meta=fm,best=best,cfg=cfg,validation_metrics=vm)
    if len({semantic_fingerprint(x['cfg']) for x in result.values()})!=1:raise RuntimeError('Semanticas diferentes entre candidatos')
    return result

def positive_counts(det,misses):
    """End-to-end positive errors; includes missed positives and unmatched positive predictions."""
    matched=det[det['matched_fruit_id'].notna()];unmatched=det[det['matched_fruit_id'].isna()]
    tp=int(((matched.gt_label=='con_signos')&(matched.pred_label=='con_signos')).sum())
    fn_cls=int(((matched.gt_label=='con_signos')&(matched.pred_label=='sin_signos')).sum())
    fn_det=int((misses.gt_label=='con_signos').sum())
    fp_cls=int(((matched.gt_label=='sin_signos')&(matched.pred_label=='con_signos')).sum())
    fp_det=int((unmatched.pred_label=='con_signos').sum())
    fn=fn_cls+fn_det;fp=fp_cls+fp_det
    p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else None
    return dict(tp_positive=tp,fp_positive=fp,fn_positive=fn,fn_positive_detector=fn_det,fn_positive_classifier=fn_cls,
                fp_positive_matched_negative=fp_cls,fp_positive_unmatched=fp_det,n_gt_positive=tp+fn,
                precision_positive=p,recall_positive=r,f1_positive=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                f2_positive=5*tp/(5*tp+fp+4*fn) if 5*tp+fp+4*fn else 0.)

def save_comparison(work):
    rows=[]
    for name,slug,t in PAIRS:
        p=work/name/'resultado.json'
        if p.is_file():
            v=read(p);rows.append(dict(system=name,detector='yolo26l',classifier=slug,det_conf=.325,classifier_threshold=t,**v['positive'],**v.get('theoretical_cost',{}),
                                       detection_tp=v['summary']['tp'],detection_fp=v['summary']['fp'],detection_fn=v['summary']['fn']))
    if rows:
        with (work/'comparacion_TEST.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        keys=list(rows[0]);table='<table border="1"><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v[k]))+'</td>' for k in keys)+'</tr>' for v in rows)+'</table>'
        (work/'INFORME_TEST.html').write_text('<meta charset="utf-8"><h1>Dos sistemas finales: TEST</h1><p>'+NOTICE+'</p>'+table+'<p>FP positivos incluyen predicciones con_signos sin GT valido. FN positivos incluyen omisiones del detector y errores del clasificador. Tiempos por imagen son descriptivos de esta ejecucion.</p>',encoding='utf-8')

def execute(project,home,work,run_test,retry):
    from utils.config_loader import load_config,cfg_path
    from utils.hashing import compute_code_hash,sha256_json
    from utils.reproducibility import get_git_info
    from utils.logging_utils import RunContext
    from utils.prepared import load_prepared_data
    from utils.run_common import check_environment_matches_reference
    from utils.semantics import semantic_fingerprint
    from utils.manual_gates import require_gate
    from orchestration.machine import read_binding
    from utils.guards import GpuJobLock
    from utils.durable import FileLock
    from utils.freeze import verify_freeze,load_freeze
    from utils.splits import assert_split_allowed
    from utils.comparison import protocol
    from utils.timing import gpu_compute_processes,free_cuda_memory
    info=get_git_info(project)
    if info.get('commit')!=COMMIT or compute_code_hash(project)!=CODE:raise RuntimeError('Codigo nativo distinto del validado; se conserva')
    local=load_config(project/'artifacts/v4/local.yaml');binding=read_binding(local)
    if local['compute']['pc_id']!='PC-02' or binding['pc_id']!='PC-02' or binding['machine_id'].casefold()!='i21314':raise RuntimeError('Vinculacion PC-02 incompatible')
    with FileLock(work/'control.guard'):
        busy=[p for p in gpu_compute_processes() if p['pid']!=os.getpid() and 'python' in str(p['name']).lower()]
        if busy:raise RuntimeError('Otra ejecucion GPU activa; no se interrumpe: '+str(busy))
        cs=candidates(home,work);source=cs['yolo26l']['cfg']
        require_gate(source,'PC-02','READY_FOR_EXPERIMENT')
        # All audits write only to this new scope, never into the training round.
        prep=copy.deepcopy(source);prep['paths']=dict(prep['paths'],artifacts_dir=str(work/'preparacion'))
        art=Path(prep['paths']['artifacts_dir']);art.mkdir(parents=True,exist_ok=True)
        src=cfg_path(source,'paths.artifacts_dir')
        if (src/'test_results.json').exists() or (src/'test_evaluation_started.json').exists():raise RuntimeError('TEST previo en ronda fuente: revisar antes de comparar')
        for f in src.iterdir():
            if f.is_file() and f.suffix.lower() in ('.csv','.json','.sha256'):
                target=art/f.name
                if not target.exists():shutil.copy2(f,target)
        prep['paths'].update(runs_dir=str(work/'runs'),results_dir=str(work/'resultados'),reports_dir=str(work/'reports'),figures_dir=str(work/'reports/figures'),tables_dir=str(work/'reports/tables'))
        with RunContext(prep,'preparar_dos_sistemas',pc_id='PC-02') as ctx:
            data=load_prepared_data(prep,with_folds=True)
            if data.dataset_hash!=DATA or data.split_hash!=SPLIT:raise RuntimeError('Dataset/split distinto del validado')
            check_environment_matches_reference(ctx)
            from utils.isolation import require_near_duplicate_review
            require_near_duplicate_review(prep,data)
            env=ctx.env
        expected=dict(code_hash=CODE,dataset_hash=DATA,split_hash=SPLIT)
        seals={str(p):sha(p) for p in [Path(__file__),project/'src/utils/pipeline_eval.py',art/'split_manifest.csv',art/'split_manifest.sha256']}
        records=[dict(architecture=slug,checkpoint=str(x['checkpoint']),checkpoint_sha256=EXPECTED[slug],validation_file=str(x['validation']),validation_sha256=sha(x['validation'])) for slug,x in cs.items()]
        comparison=dict(candidates=records,protocol=protocol(prep),protocol_sha256=sha256_json(protocol(prep)),
                        review_sha256=sha(art/'near_duplicate_review.json') if (art/'near_duplicate_review.json').exists() else None,
                        derived_content_sha256={task:sha(cfg_path(prep,'paths.'+task+'_dir')/'content_integrity.json') for task in ['detection','classification']},purpose=NOTICE)
        fixed=dict(scope='DOS_SISTEMAS_FINALES_EXTENSION_EXPLICITA_3_CANDIDATOS',notice=NOTICE,code_sha256=sha(Path(__file__)),sources=records,seals=seals,
                   dataset_hash=DATA,split_hash=SPLIT,det_conf=.325,match_iou=.5,pairs=[list(p) for p in PAIRS],threshold_origin='VAL previo, F2 exploratorio: detector .325, ResNet .69, Swin .85; no reajustar en TEST',
                   exception='No es cohorte original de siete arquitecturas; RT-DETR excluido y seleccion detector por rendimiento/costo autorizados por usuario.')
        # Freeze both variants before selecting any test image or running inference.
        pin_file(work/'protocolo_doble.json',fixed)
        configs=[]
        for name,slug,t in PAIRS:
            cfg=copy.deepcopy(prep);pair=work/name;pair.mkdir(exist_ok=True)
            cfg['test']['freeze_manifest']=str(pair/'freeze_manifest.json')
            freeze=dict(semantic_fingerprint=semantic_fingerprint(cfg),comparison=comparison,code={'code_hash':CODE},data={'dataset_hash':DATA,'split_hash':SPLIT},environment={k:env.get(k) for k in ['python_version','torch','torchvision','timm','ultralytics','optuna']},
                        detector={'checkpoint':str(cs['yolo26l']['checkpoint']),'checkpoint_hash':EXPECTED['yolo26l'],'confidence_threshold':.325,'nms_iou':None},
                        classifier={'checkpoint':str(cs[slug]['checkpoint']),'checkpoint_hash':EXPECTED[slug],'probability_threshold':t},
                        extension_protocol_sha256=sha(work/'protocolo_doble.json'),scope=fixed['scope'],notice=NOTICE)
            pin_file(pair/'freeze_manifest.json',freeze);configs.append((name,slug,t,cfg,freeze))
        emit('DOS_SISTEMAS_CONGELADOS',protocol=str(work/'protocolo_doble.json'),sha256=sha(work/'protocolo_doble.json'),test_inference_started=False)
        if not run_test:return
        # The actual command invoked personally by the operator is the release evidence.
        for name,slug,t,cfg,freeze in configs:
            release={'state':'AUTORIZADO_PERSONALMENTE','operator':os.environ['USERNAME'],'evidence':'Operador ejecuto --ejecutar-dos-test para los dos sistemas congelados; seleccion posterior por profesor declarada.', 'freeze_sha256':sha(Path(cfg['test']['freeze_manifest'])),'scope':fixed['scope']}
            pair=work/name
            # Per-system metadata directory prevents replacing the other release receipt.
            own=pair/'artifacts';own.mkdir(exist_ok=True)
            cfg['paths']['artifacts_dir']=str(own)
            for f in art.iterdir():
                if f.is_file() and not (own/f.name).exists():shutil.copy2(f,own/f.name)
            write(own/'manual_test_release.json',release)
            verify_freeze(cfg,load_freeze(cfg),expected,env)
            assert_split_allowed('test','final_test',cfg)
            if (pair/'resultado.json').exists():
                old=read(pair/'resultado.json')
                if old['freeze_sha256']!=sha(pair/'freeze_manifest.json'):raise RuntimeError('Resultado previo de otro freeze')
                emit('TEST_COMPLETO_REUTILIZADO',system=name);continue
            if (pair/'TEST_INICIADO.json').exists() and not retry:raise RuntimeError('TEST interrumpido; conservar evidencia y reanudar con --reanudar-interrumpido')
            for path,digest in seals.items():
                if sha(Path(path))!=digest:raise RuntimeError('Archivo congelado alterado: '+path)
            test_ids=sorted(data.manifest.loc[data.manifest.outer_split=='test','image_id'])
            paths={r.image_id:Path(r.image_path) for r in data.images.itertuples(index=False) if r.image_id in test_ids}
            if len(paths)!=len(test_ids):raise RuntimeError('TEST no cubierto por imagenes')
            from utils.pipeline_eval import gt_from_fruit_table,evaluate_pipeline
            from utils.inference import load_detector
            from utils.classifier_models import load_classifier
            from utils.architectures import get_arch
            from utils.training_common import resolve_device
            from utils.reporting import save_pipeline_figures
            from utils.io import write_csv
            gt=gt_from_fruit_table(data.fruits,test_ids)
            write(pair/'TEST_INICIADO.json',dict(freeze_sha256=sha(pair/'freeze_manifest.json'),system=name,retry=retry))
            with GpuJobLock(cfg,'PC-02',job='dos_test_finales_'+name),RunContext(cfg,'test_doble_'+name,pc_id='PC-02',trial_name=name,env_override=env) as run:
                run.set_hashes(dataset_hash=DATA,split_hash=SPLIT);run.set_seed(42);run.note_split_access('test','final_test')
                free_cuda_memory()
                detector=load_detector(cs['yolo26l']['checkpoint'],get_arch('yolo26l'))
                classifier,cm=load_classifier(cs[slug]['checkpoint'],resolve_device(cfg))
                emit('EJECUTANDO_TEST_FINAL',system=name,classifier=slug,images=len(test_ids),classifier_threshold=t)
                res=evaluate_pipeline(cfg,detector=detector,classifier=classifier,norm=cm['norm'],image_paths=paths,gt=gt,conf=.325,nms_iou=float(cfg['detector']['val_nms_iou']),cls_threshold=t,match_iou=.5,max_examples=6)
                positive=positive_counts(res.detections,res.misses)
                if positive['n_gt_positive']!=sum(l=='con_signos' for g in gt.values() for l in g['labels']):raise RuntimeError('Conteo positivo no cubre todos los GT')
                for filename,table in [('detecciones.csv',res.detections),('omisiones.csv',res.misses),('tiempos_por_imagen.csv',res.per_image),('errores.csv',res.errors)]:write_csv(table,pair/filename,float_format='%.10g')
                write(pair/'inferencia_completa.json',dict(summary=res.summary,positive=positive,freeze_sha256=sha(pair/'freeze_manifest.json')))
                save_pipeline_figures(run,res,'YOLO26l',get_arch(slug).name,suffix='test_'+name,sources=[pair/'detecciones.csv',pair/'omisiones.csv'],n_examples=6)
                output=dict(system=name,classifier=slug,summary=res.summary,positive=positive,freeze_sha256=sha(pair/'freeze_manifest.json'),protocol_sha256=sha(work/'protocolo_doble.json'),notice=NOTICE,theoretical_cost={'detector_params_M':cs['yolo26l']['validation_metrics']['params_m'],'classifier_params_M':cs[slug]['validation_metrics']['params_m'],'detector_GFLOPs_per_image':cs['yolo26l']['validation_metrics']['gflops'],'classifier_GFLOPs_per_crop':cs[slug]['validation_metrics']['gflops']},figures=str(run.figures_dir),test_used=True,thresholds_changed_after_test=False)
                run.log_metrics(output);write(pair/'resultado.json',output)
                emit('TEST_FINAL_COMPLETO',system=name,positive=positive,result=str(pair/'resultado.json'))
                del classifier,detector;free_cuda_memory()
            save_comparison(work)
        save_comparison(work)
        emit('DOS_TEST_FINALES_COMPLETOS',html=str(work/'INFORME_TEST.html'),csv=str(work/'comparacion_TEST.csv'),directory=str(work))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ejecutar-dos-test',action='store_true');p.add_argument('--reanudar-interrumpido',action='store_true');args=p.parse_args()
    if os.name!='nt' or os.environ.get('USERNAME','').casefold()!='i21314':raise RuntimeError('Ejecutar exclusivamente Windows PC-02/i21314')
    home=Path.home();project=home/'Documents/s2_no_borrar/Entrenamiento/Botrytis_Experiment_Framework_git';py=project/'.venv/Scripts/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():raise RuntimeError('Usar Python .venv del proyecto')
    os.chdir(project);sys.path.insert(0,str(project/'src'));os.environ.update(PYTHONUTF8='1',MPLBACKEND='Agg',YOLO_OFFLINE='true',HF_HUB_OFFLINE='1')
    work=home/'bd/dos_test_finales_v1';work.mkdir(parents=True,exist_ok=True)
    execute(project,home,work,args.ejecutar_dos_test,args.reanudar_interrumpido)
if __name__=='__main__':
    try:main()
    except Exception as exc:emit('TEST_DOBLE_DETENIDO',error=str(exc),results_preserved=True);raise
