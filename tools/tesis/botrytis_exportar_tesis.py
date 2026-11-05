"""Exportacion local de evidencias para tesis. No entrena, evalua, aprueba ni abre TEST.
Un ZIP por PC; datos de entrada TRAIN/VAL solo en PC-01 para evitar triplicarlos.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,os,shutil,sqlite3,subprocess,sys,tempfile,uuid,zipfile
from pathlib import Path
from datetime import datetime,timezone

USERS={'i21315':'PC-01','i21314':'PC-02','i21313':'PC-03'}
ROOTS={'bf':'03_RESULTADOS','bn':'02_RONDAS','bte':'02_RONDAS','bt-hoy-20261103':'02_RONDAS',
       'bg':'04_AUDITORIA','bp':'04_AUDITORIA','br':'04_AUDITORIA','bv':'04_AUDITORIA','btlog':'04_AUDITORIA','r50':'03_RESULTADOS'}
TEXT={'.csv','.json','.jsonl','.yaml','.yml','.txt','.log','.md','.html','.sha256','.svg','.pdf','.py','.cmd','.toml','.ini','.lock','.bib','.tex','.docx','.rtf','.rst'}
IMAGES={'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}
SKIPDIR={'.git','.venv','__pycache__','.pytest_cache','node_modules','off','_off','test','testing','final_test','final_testing','test_results','test_evaluation','secrets'}
SKIPNAMES={'.env','.env.local','credentials.json','token.json','id_rsa','id_ed25519'}


def emit(event,**kw):print(event+' '+json.dumps(kw,ensure_ascii=False,default=str),flush=True)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def blocked(path):
    return any(x.casefold() in SKIPDIR for x in path.parts) or path.name.casefold() in SKIPNAMES or path.name.casefold().startswith(('test_predictions','test_evaluation','final_test'))
def files(root):
    if not root.is_dir():return
    for base,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d.casefold() not in SKIPDIR and not (Path(base)/d).is_symlink())
        for n in sorted(names):
            p=Path(base)/n
            if not p.is_symlink() and not blocked(p.relative_to(root)):yield p


def idle():
    import psutil
    me=psutil.Process();excluded={me.pid,*[p.pid for p in me.parents()]};found=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        if p.pid in excluded:continue
        if (p.info['name'] or '').casefold() in ('python.exe','pythonw.exe','python','python3'):
            found.append(dict(pid=p.pid,command=(p.info['cmdline'] or [])[:2]))
    if found:raise RuntimeError('Hay otro Python activo. Exporte cuando termine; no se interrumpe ningun proceso. '+json.dumps(found))


def read_split(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows or not {'image_id','outer_split'}<=set(rows[0]):raise ValueError('Manifiesto sin image_id/outer_split: '+str(path))
    result={}
    for row in rows:
        iid=row['image_id'].casefold();split=row['outer_split'].casefold()
        if split not in ('train','validation','test'):raise ValueError('Split desconocido')
        if iid in result and result[iid]!=split:raise ValueError('ID con splits contradictorios')
        result[iid]=split
    return result


def export_csv(path,test_ids):
    """No source edits. Strip held-out rows; public manifest carries IDs/splits only."""
    raw=path.read_bytes()
    text=raw.decode('utf-8-sig')
    rows=list(csv.DictReader(io.StringIO(text),strict=True));header=next(csv.reader(io.StringIO(text)),[])
    if not header:return raw,0,'original'
    if len(header)!=len(set(header)) or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError('CSV con columnas duplicadas o registros incompletos; revisar fuente')
    if {'image_id','outer_split'}<=set(header) and path.name=='split_manifest.csv':
        keep=['image_id','outer_split'];out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=keep,lineterminator='\n');w.writeheader()
        w.writerows({k:r[k] for k in keep} for r in rows)
        return out.getvalue().encode(),0,'manifest_ids_splits_only'
    def forbidden(r):
        if any((r.get(k) or '').casefold() in ('test','final_test') for k in ('outer_split','split','evaluated_on')):return True
        iid=(r.get('image_id') or r.get('stem') or '').casefold()
        return iid in test_ids
    kept=[r for r in rows if not forbidden(r)]
    if len(kept)==len(rows):return raw,0,'original'
    out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=header,lineterminator='\n');w.writeheader();w.writerows(kept)
    return out.getvalue().encode(),len(rows)-len(kept),'test_rows_removed_export_only'


def export_json(path,test_ids):
    raw=path.read_bytes();obj=json.loads(raw.decode('utf-8-sig'));removed=0;drop=object()
    def clean(value):
        nonlocal removed
        if isinstance(value,dict):
            if (str(value.get('image_id','')).casefold() in test_ids or
                any(str(value.get(k,'')).casefold() in ('test','final_test') for k in ('outer_split','split','evaluated_on'))):
                removed+=1;return drop
            out={}
            for k,v in value.items():
                v=clean(v)
                if v is not drop:out[k]=v
            return out
        if isinstance(value,list):return [v for v in (clean(v) for v in value) if v is not drop]
        return value
    result=clean(obj)
    if result is drop:return None,removed,'test_only_document_excluded'
    return ((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode() if removed else raw),removed,('test_records_removed_export_only' if removed else 'original')


def snapshot(db,dest):
    with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True,timeout=15) as src:
        with sqlite3.connect(dest) as out:src.backup(out)
    with sqlite3.connect(dest) as check:
        if check.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise RuntimeError('Snapshot SQLite no integro')


class Packet:
    def __init__(self,path,pc,test_ids):
        self.z=zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True)
        self.pc=pc;self.test_ids=test_ids;self.rows=[];self.excluded=[];self.errors=[];self.names=set()
    def add(self,path,arc,*,code=False,raw_input=False):
        path=Path(path);arc=str(arc).replace('\\','/')
        requested=arc
        if Path(arc).is_absolute() or '..' in Path(arc).parts:raise ValueError('Ruta ZIP no segura')
        if len(arc)>190:arc='06_RUTAS_LARGAS/'+hashlib.sha256(arc.encode()).hexdigest()[:24]+path.suffix
        if arc in self.names:return
        if path.is_symlink() or (not code and blocked(Path(arc))):return
        ext=path.suffix.casefold()
        if ext in ('.pt','.pth','.ckpt','.safetensors') and path.name!='final.pt':
            self.excluded.append(dict(source=str(path),reason='intermediate_or_pretrained_weight',size=path.stat().st_size));return
        if not raw_input and ext not in TEXT|IMAGES|{'.pt','.sqlite','.db','.sqlite3'} and path.name not in ('requirements.txt','LICENSE','Dockerfile'):return
        before=path.stat();operation='original';removed=0
        with tempfile.TemporaryDirectory() as td:
            source=path;payload=None
            if ext in ('.sqlite','.sqlite3','.db'):
                source=Path(td)/'snapshot.sqlite';snapshot(path,source);operation='consistent_sqlite_snapshot'
            elif ext=='.csv' and not code:
                payload,removed,operation=export_csv(path,self.test_ids)
            elif ext=='.json' and not code:
                payload,removed,operation=export_json(path,self.test_ids)
                if payload is None:
                    self.excluded.append(dict(source=str(path),reason=operation));return
            if payload is not None:
                source_hash=digest(path);export_hash=hashlib.sha256(payload).hexdigest()
                if path.stat().st_size!=before.st_size or path.stat().st_mtime_ns!=before.st_mtime_ns:raise RuntimeError('Archivo cambio durante exportacion')
                self.z.writestr(arc,payload)
            else:
                source_hash=None if operation=='consistent_sqlite_snapshot' else digest(path)
                export_hash=digest(source) if source_hash is None else source_hash
                if operation!='consistent_sqlite_snapshot' and (path.stat().st_size!=before.st_size or path.stat().st_mtime_ns!=before.st_mtime_ns):raise RuntimeError('Archivo cambio durante exportacion')
                method=zipfile.ZIP_STORED if ext in IMAGES|{'.pt','.pdf'} else zipfile.ZIP_DEFLATED
                self.z.write(source,arc,compress_type=method)
                if operation!='consistent_sqlite_snapshot' and (path.stat().st_size!=before.st_size or path.stat().st_mtime_ns!=before.st_mtime_ns):
                    raise RuntimeError('Archivo cambio al escribir ZIP; exportacion invalida')
            self.names.add(arc);self.rows.append(dict(source=str(path),archive=arc,requested_archive=requested,source_size=before.st_size,
                source_sha256=source_hash,export_sha256=export_hash,operation=operation,test_rows_removed=removed))
            if len(self.rows)%100==0:emit('EXPORTANDO',files=len(self.rows),current=arc)
    def tree(self,root,arc,*,code=False):
        for p in files(root):
            try:self.add(p,Path(arc)/p.relative_to(root),code=code)
            except Exception as exc:self.errors.append(dict(source=str(p),error=str(exc)))


def configs(project,home):
    sys.path.insert(0,str(project/'src'))
    from utils.config_loader import load_config,cfg_path
    index_path=home/'bt-hoy-20261103/rondas.json'
    index=json.loads(index_path.read_text(encoding='utf-8-sig')) if index_path.is_file() else {}
    configs=[];problems=[]
    for model,path in index.items():
        try:
            cfg=load_config(path);configs.append((model,Path(path),cfg))
        except Exception as exc:problems.append(dict(model=model,config=str(path),error=str(exc)))
    if not configs:raise RuntimeError('No hay configuraciones legibles en rondas.json; no se genera un ZIP vacio')
    return configs,problems,cfg_path


def build(project,home,pc,output,*,include_inputs):
    cfgs,problems,cfg_path=configs(project,home)
    split={};splits=[]
    for model,path,cfg in cfgs:
        p=cfg_path(cfg,'paths.artifacts_dir')/str(cfg['outer_split']['manifest_name'])
        current=read_split(p)
        if split and current!=split:raise RuntimeError('Rondas actuales con particiones distintas; revisar antes de combinar evidencias')
        split=current;splits.append(str(p))
    test_ids={i for i,s in split.items() if s=='test'};allowed={i for i,s in split.items() if s in ('train','validation')}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');folder=output/(pc+'_'+stamp+'_'+uuid.uuid4().hex[:6]);folder.mkdir(parents=True)
    target=folder/('BOTRYTIS_TESIS_'+pc+'_'+stamp+'.zip');partial=target.with_suffix('.zip.partial')
    packet=Packet(partial,pc,test_ids);packet.errors.extend(problems)
    roots={};missing=[]
    for name,category in ROOTS.items():
        root=home/name
        if root.is_dir():roots[root]=Path(category)/name
    # Include actual paths referenced by config, even if stored outside known roots.
    for model,path,cfg in cfgs:
        if not any(path.is_relative_to(root) for root in roots):roots[path.parent]=Path('02_RONDAS')/'referenciadas'/model
        for key in ('artifacts_dir','runs_dir','results_dir','reports_dir','classification_dir','detection_dir'):
            root=cfg_path(cfg,'paths.'+key)
            # Derived inputs: tables only, never unfiltered crop/image collections.
            if key in ('classification_dir','detection_dir'):
                if root.is_dir():
                    for p in files(root):
                        if p.suffix.lower() in ('.csv','.json','.yaml','.yml'):
                            try:packet.add(p,Path('05_DATOS_TRAIN_VAL')/model/key/p.relative_to(root))
                            except Exception as exc:packet.errors.append(dict(source=str(p),error=str(exc)))
                else:missing.append(str(root))
            elif root.is_dir() and not any(root.is_relative_to(r) for r in roots):roots[root]=Path('03_RESULTADOS')/'referenciados'/model/key
        for relative in ('baseline/summary.json','hpo/study_summary.json','robustness/summary.json','final/train_evaluation.json','outer_validation/metrics.json'):
            # Effective short path is computed with exactly the helper's algorithm.
            key=hashlib.sha256(str(path.resolve()).casefold().encode()).hexdigest()[:8]
            run=(home/'r50/r') if model=='rtdetr_r50' else home/'bf'/key/'r'
            p=run/model/relative
            if not p.is_file():missing.append(dict(model=model,expected=str(p),note='Ausencia no significa fallo ni autoriza reentrenar'))
    # Avoid copying nested source roots twice.
    for root,arc in sorted(roots.items(),key=lambda x:len(x[0].parts)):
        if any(root!=other and root.is_relative_to(other) for other in roots):continue
        emit('REUNIENDO_CARPETA',source=str(root));packet.tree(root,arc)
    for root in ('src','configs','docs','reports','logs_qa'):
        packet.tree(project/root,Path('01_PROTOCOLO_CODIGO')/root,code=root in ('src','configs','docs'))
    for root in ('artifacts','runs','results'):
        if not any((project/root).is_relative_to(r) for r in roots):
            packet.tree(project/root,Path('03_RESULTADOS/proyecto')/root)
    for glob in ('README*','LICENSE*','requirements*','pyproject.toml','*.lock'):
        for p in project.glob(glob):
            if p.is_file():packet.add(p,Path('01_PROTOCOLO_CODIGO')/p.name,code=True)
    for p in (home/'Downloads').glob('botrytis*'):
        if p.is_symlink():continue
        if p.resolve()==output.resolve():continue
        if p.is_dir():
            if not any(p.glob('botrytis*.py')):continue
            for f in files(p):
                if f.suffix.lower() in ('.py','.cmd','.json','.md'):
                    packet.add(f,Path('01_PROTOCOLO_CODIGO/ayudantes')/p.name/f.relative_to(p),code=True)
        elif p.suffix.lower()=='.py':packet.add(p,Path('01_PROTOCOLO_CODIGO/ayudantes')/p.name,code=True)
    input_ids={'raw_images_dir':set(),'raw_labels_dir':set()}
    if include_inputs:
        seen=set()
        for _,_,cfg in cfgs:
            for key in ('raw_images_dir','raw_labels_dir'):
                root=cfg_path(cfg,'paths.'+key)
                for p in files(root):
                    if p.stem.casefold() not in allowed or p in seen:continue
                    if key=='raw_images_dir' and p.suffix.lower() not in IMAGES:continue
                    if key=='raw_labels_dir' and p.suffix.lower() not in ('.txt','.json'):continue
                    packet.add(p,Path('05_DATOS_TRAIN_VAL/entradas')/key/p.name,raw_input=True);seen.add(p);input_ids[key].add(p.stem.casefold())
        for key,found in input_ids.items():
            if allowed-found:packet.errors.append(dict(source=key,error='Entradas TRAIN/VAL ausentes del paquete',missing_ids=sorted(allowed-found)))
    git={}
    for name,args in [('commit',['rev-parse','HEAD']),('status',['status','--short']),('diff',['diff','--','src','configs','tools','docs'])]:
        result=subprocess.run(['git','-C',str(project),*args],capture_output=True,text=True,encoding='utf-8',errors='replace')
        git[name]=result.stdout if result.returncode==0 else 'No disponible'
    readme='''# Paquete de evidencias Botrytis para tesis\n\nNo es un analisis ni una validacion nueva. No entrena, aprueba, calibra ni usa TEST.\n\n01_PROTOCOLO_CODIGO: codigo actual, protocolo, configuraciones, dependencias y ayudantes. Git commit/status/diff en 00_INDICES/GIT.json.\n02_RONDAS: configuraciones originales, manifests, particiones, entornos, colas/estudios SQLite consistentes, revisiones y decisiones.\n03_RESULTADOS: baseline, HPO, robustez, final TRAIN y VAL; CSV/JSON, predicciones, curvas y figuras PNG/SVG/PDF, evidencias visuales, pesos final.pt.\n04_AUDITORIA: informes, recuperaciones, aprobaciones, errores y cierres; se conservan anteriores y parciales sin llamarlos completos.\n05_DATOS_TRAIN_VAL: tablas derivadas filtradas. Entradas originales TRAIN/VAL solo se incluyen por defecto desde PC-01. OFF/_off y entradas TEST excluidas.\n\nPara Experimentacion: protocolo, hardware/software, semillas, folds/splits, HPO/search spaces, entrenamiento/augmentations, pesos/pretraining y procedencia, presupuestos, reglas de seleccion, QA y desviaciones.\nPara Resultados/graficos: histories, fold_metrics, trial/study summaries, CSV de predicciones con verdad/probabilidad o cajas/scores, reportes de clasificacion, matrices, PR/ROC/AP-IoU, curvas de confianza, evaluacion TRAIN/VAL y benchmarking SI EXISTE.\nNo hay curvas VAL externo por epoca porque el final usa epocas fijas. FPS/latencia no medidos no se inventan. Los clasificadores de VAL usan GT crops hasta que se evalua el hibrido.\n\n00_INDICES/INVENTARIO.csv mapea ruta original a ZIP y hashes. CSV y registros JSON de TEST se exportan filtrados, nunca se modifica la fuente: ambas huellas y la transformacion quedan registradas. El manifiesto publico conserva solo ID/split. Las rutas ZIP muy largas se abrevian en 06_RUTAS_LARGAS y se documentan en requested_archive. Extraiga en una ruta corta, por ejemplo C:\\bt_tesis\\PC01. Los indices/recibos originales pueden referir archivos excluidos o hashes de fuentes: use este inventario para verificar el ZIP. Los SQLite son snapshots por archivo, no una instantanea transaccional de toda la PC.\n00_INDICES/FALTANTES.json distingue archivos aun no generados. EXCLUIDOS.json inventaria pesos intermedios/preentrenados grandes no incluidos. Se conservan los pesos finales disponibles; otros checkpoints NO se borran. ERRORES.json debe revisarse si el paquete queda INCOMPLETO.\n\nEstructura recomendada en Drive: Botrytis_Tesis/01_Paquetes_originales/PC-01|PC-02|PC-03/; 02_Datos_para_analisis/; 03_Graficos_editables/; 04_Tablas_tesis/; 05_Texto_experimentacion_resultados/. Guarde cada ZIP original con su SHA256. Extraiga las tres PCs en carpetas distintas; no mezcle CSV con el mismo nombre ni elimine rondas historicas. Las rutas Windows dentro de JSON son originales; el inventario muestra su ubicacion portable.\n'''
    packet.z.writestr('LEEME.md',readme)
    packet.z.writestr('00_INDICES/GIT.json',json.dumps(git,indent=2,ensure_ascii=False))
    packet.z.writestr('00_INDICES/FALTANTES.json',json.dumps(missing,indent=2,ensure_ascii=False))
    packet.z.writestr('00_INDICES/EXCLUIDOS.json',json.dumps(packet.excluded,indent=2,ensure_ascii=False))
    packet.z.writestr('00_INDICES/ERRORES.json',json.dumps(packet.errors,indent=2,ensure_ascii=False))
    manifest=dict(pc=pc,created_utc=stamp,status='INCOMPLETO_REVISAR_ERRORES' if packet.errors else 'EXPORTACION_COMPLETADA',
        files=len(packet.rows),code_exporter_sha256=digest(Path(__file__)),split_sources=splits,
        inputs_train_val_included=include_inputs,input_counts={k:len(v) for k,v in input_ids.items()},test_inputs_exported=False,test_inference_executed=False,
        incomplete_or_absent_stages=missing)
    packet.z.writestr('00_INDICES/RESUMEN_EXPORTACION.json',json.dumps(manifest,indent=2,ensure_ascii=False))
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=['source','archive','requested_archive','source_size','source_sha256','export_sha256','operation','test_rows_removed'],lineterminator='\n');w.writeheader();w.writerows(packet.rows)
    packet.z.writestr('00_INDICES/INVENTARIO.csv',buf.getvalue());packet.z.close()
    partial.replace(target)
    sha=digest(target);target.with_suffix('.zip.sha256.txt').write_text(sha+'  '+target.name+'\n')
    (folder/'LEEME.md').write_text(readme,encoding='utf-8')
    emit('ZIP_TESIS_CREADO',path=str(target),GB=round(target.stat().st_size/1024**3,3),sha256=sha,files=len(packet.rows),errors=len(packet.errors),status=manifest['status'])
    return target,manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--no-inputs',action='store_true');args=p.parse_args()
    if os.name!='nt':raise RuntimeError('Ejecutar en Windows de las PCs del proyecto; tests CPU separados')
    home=Path.home();pc=USERS.get(os.environ.get('USERNAME','').casefold())
    if not pc:raise RuntimeError('Usuario no reconocido; no se inspecciona el perfil')
    project=home/'Documents/s2_no_borrar/Entrenamiento/Botrytis_Experiment_Framework_git'
    idle()
    if shutil.disk_usage(home).free<2*1024**3:raise RuntimeError('Menos de 2GB libres; liberar espacio antes de exportar')
    target,manifest=build(project,home,pc,home/'Downloads/Botrytis_Tesis',include_inputs=pc=='PC-01' and not args.no_inputs)
    return 2 if manifest['status']!='EXPORTACION_COMPLETADA' else 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:emit('EXPORTACION_DETENIDA',error=str(e),training_started=False,results_modified=False);sys.exit(2)
