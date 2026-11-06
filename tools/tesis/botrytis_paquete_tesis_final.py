"""Empaqueta documentos de todas las etapas, modelos y TEST existentes. No ejecuta modelos."""
from pathlib import Path,PureWindowsPath
from datetime import datetime,timezone
import argparse,contextlib,csv,hashlib,io,json,os,re,shutil,sqlite3,subprocess,sys,tempfile,time,uuid,zipfile
USERS={'i21315':'PC-01','i21314':'PC-02','i21313':'PC-03'}
MODELS={'PC-01':['yolov8n','yolov8l','resnet50'],'PC-02':['yolo26l','swin_tiny'],'PC-03':['vit_b16','rtdetr_r50']}
ALL=['rtdetr_r50','resnet50','swin_tiny','vit_b16','yolo26l','yolov8n','yolov8l']
ROOTS={'bf','bg','bn','bp','br','bte','bt-hoy-20261103','btlog','bv','r50','bd','Downloads','Documents'}
DOC={'.csv','.json','.jsonl','.yaml','.yml','.txt','.log','.md','.html','.sha256','.svg','.pdf','.py','.cmd','.ps1','.toml','.ini','.lock','.bib','.tex','.docx','.rtf','.rst','.xml','.pkl','.joblib','.sqlite','.sqlite3','.db','.cfg','.bat','.patch','.diff','.bundle','.bak','.guard','.tag'}
IMAGE={'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}
NAMED={'license','dockerfile','makefile','copying','notice'}
SKIP={'.git','.venv','__pycache__','.pytest_cache','node_modules'}
SHORT={'10ee5011':'yolov8n','671d23cb':'yolov8l','17ee1aeb':'resnet50','52c7988b':'yolo26l','0346b429':'swin_tiny','9b5af1a1':'vit_b16'}

def emit(event,**kw):print(event+' '+json.dumps(kw,ensure_ascii=False,default=str),flush=True)
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def linked(p):return p.is_symlink() or (hasattr(p,'is_junction') and p.is_junction())
def same(a,b):return str(a).replace('\\','/').casefold()==str(b).replace('\\','/').casefold()
def snapshot(path,out):
 deadline=time.monotonic()+45
 def progress(*a):
  if time.monotonic()>deadline:raise TimeoutError('SQLite no disponible en 45 segundos; no se altera fuente')
 with contextlib.closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=10)) as src:
  with contextlib.closing(sqlite3.connect(out)) as dest:src.backup(dest,pages=256,progress=progress,sleep=.1);dest.commit()
 with contextlib.closing(sqlite3.connect(out)) as con:
  if con.execute('PRAGMA integrity_check').fetchall()!=[('ok',)]:raise RuntimeError('Snapshot SQLite no integro')

def classify(relative,pc):
 parts=relative.replace('\\','/').split('/');low='/'.join(parts).casefold()
 if parts[0].casefold()=='bd' and 'dos_test_finales_v1' in low:
  idx=next(i for i,p in enumerate(parts) if p=='dos_test_finales_v1');tail=parts[idx+1:]
  if tail and tail[0]=='A':return pc+'/02_SISTEMAS_TEST/A_YOLO26l_ResNet50/'+'/'.join(tail[1:])
  if tail and tail[0]=='B':return pc+'/02_SISTEMAS_TEST/B_YOLO26l_SwinTiny/'+'/'.join(tail[1:])
  return pc+'/02_SISTEMAS_TEST/00_Protocolo_comparacion_y_evidencias/'+'/'.join(tail)
 project='Botrytis_Experiment_Framework_git/'
 if project.casefold() in low:
  start=low.index(project.casefold())+len(project);tail='/'.join(parts)[start:]
  if tail.split('/')[0] in ('src','configs','tools','docs','weights') or '/' not in tail:return pc+'/00_PROYECTO_Y_EQUIPO/'+tail
 if parts[0].casefold()=='downloads':return pc+'/00_PROYECTO_Y_EQUIPO/ayudantes_y_antecedentes/'+'/'.join(parts[1:])
 model=next((m for m in ALL if m in low),None)
 if not model:model=next((m for key,m in SHORT.items() if key in low),None)
 if parts[0].casefold()=='r50':model='rtdetr_r50'
 if not model:return pc+'/03_AUDITORIA_Y_ANTECEDENTES/'+relative.replace('\\','/')
 if 'checkpoints' in low or '/weights/' in low:stage='07_Pesos_y_procedencia'
 elif 'outer_validation' in low or '/validation/' in low:stage='06_VAL'
 elif 'robustness' in low or 'robustez' in low or 'best_configs' in low:stage='04_Robustez'
 elif 'baseline' in low:stage='02_Baseline'
 elif 'hpo' in low or 'optuna' in low:stage='03_HPO'
 elif '/final/' in low or 'outer_train' in low:stage='05_Entrenamiento_final'
 elif any(x in low for x in ['audit','review','logs','queue','control','recovery']):stage='08_Auditoria_y_recuperaciones'
 else:stage='01_Configuracion_preparacion_y_otros'
 return pc+'/01_MODELOS/'+model+'/'+stage+'/'+relative.replace('\\','/')

def allowed_source(path,home,project):
 try:rel=path.resolve().relative_to(home.resolve())
 except ValueError:return False
 if not rel.parts or rel.parts[0].casefold() not in {r.casefold() for r in ROOTS}:return False
 if any(p.casefold() in SKIP for p in rel.parts):return False
 if rel.parts[0].casefold()=='documents':return path.resolve().is_relative_to(project.resolve())
 if rel.parts[0].casefold()=='downloads':return len(rel.parts)>1 and rel.parts[1].casefold().startswith('botrytis')
 return True

class Packet:
 def __init__(self,target,pc):
  self.z=zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=2,allowZip64=True);self.pc=pc;self.rows=[];self.errors=[];self.warnings=[];self.excluded=[];self.seen=set();self.arcs=set();self.bytes=0
 def add(self,p,arc,reason='original'):
  p=Path(p)
  if str(p).casefold() in self.seen:return
  if linked(p):raise RuntimeError('Enlace/junction no seguido')
  requested=arc.replace('\\','/');parts=requested.split('/')
  if requested.startswith('/') or '..' in parts or ':' in requested:raise ValueError('Ruta ZIP invalida')
  if len(requested)>185:
   prefix='/'.join(parts[:4]);arc=prefix+'/_rutas_cortas/'+hashlib.sha256(requested.encode()).hexdigest()[:20]+'_'+p.name[-65:]
  else:arc=requested
  if arc.casefold() in self.arcs:raise ValueError('Destino ZIP duplicado: '+arc)
  before=p.stat()
  with tempfile.TemporaryDirectory() as td:
   source=p;op=reason
   if p.suffix.lower() in {'.sqlite','.db','.sqlite3'}:
    source=Path(td)/'snapshot.sqlite';snapshot(p,source);op='snapshot_sqlite_consistente'
   elif p.suffix.lower()=='.csv':
    try:
     with p.open(encoding='utf-8-sig',newline='') as f:
      reader=csv.reader(f,strict=True);header=next(reader,[]);bad=[]
      for i,row in enumerate(reader,2):
       if len(row)!=len(header) and len(bad)<12:bad.append(dict(line=i,columns=len(row)))
     if len(header)!=len(set(header)) or bad:self.warnings.append(dict(source=str(p),warning='CSV original conservado sin reparar; revisar estructura',header_columns=len(header),irregular_rows=bad));op='csv_original_con_advertencia'
    except (UnicodeError,csv.Error) as exc:self.warnings.append(dict(source=str(p),warning='CSV original conservado; '+str(exc)));op='csv_original_con_advertencia'
   h=digest(source);size=source.stat().st_size
   if reason=='checkpoint_parcial_con_recibo':
    receipt=json.loads(p.with_suffix(p.suffix+'.resume.json').read_text(encoding='utf-8-sig'))
    if receipt.get('sha256')!=h or receipt.get('architecture')!='rtdetr_r50':raise RuntimeError('Recibo RT-DETR no coincide con checkpoint; se conserva fuente')
   self.z.write(source,arc,compress_type=zipfile.ZIP_STORED if p.suffix.lower() in IMAGE|{'.pt','.pth','.safetensors','.pdf'} else zipfile.ZIP_DEFLATED)
   self.arcs.add(arc.casefold())
   after=p.stat()
   if op!='snapshot_sqlite_consistente' and (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Fuente cambio durante exportacion; revisar ZIP')
  self.seen.add(str(p).casefold());self.bytes+=size
  self.rows.append(dict(source=str(p),archive=arc,requested_archive=requested,bytes=size,source_sha256=None if op=='snapshot_sqlite_consistente' else h,archive_sha256=h,operation=op,source_modified_utc=datetime.fromtimestamp(before.st_mtime,timezone.utc).isoformat()))
  if len(self.rows)%200==0:emit('EMPAQUETANDO_TESIS',files=len(self.rows),GB=round(self.bytes/1024**3,2),current=arc)
 def attempt(self,p,arc,reason='original'):
  try:self.add(p,arc,reason)
  except Exception as exc:self.errors.append(dict(source=str(p),error=str(exc)))
 def exclude(self,p,why,size):self.excluded.append(dict(source=str(p),reason=why,bytes=size))
 def generated(self,name,value):
  if name.casefold() in self.arcs:raise ValueError('Destino ZIP duplicado: '+name)
  self.z.writestr(name,value if isinstance(value,(str,bytes)) else json.dumps(value,ensure_ascii=False,indent=2)+'\n');self.arcs.add(name.casefold())

def read_inventory(path,pc):
 with zipfile.ZipFile(path) as z:
  summary=json.loads(z.read('RESUMEN.json'))
  if summary['pc']!=pc:raise RuntimeError('Inventario de otra PC')
  rows=list(csv.DictReader(io.StringIO(z.read('ARCHIVOS.csv').decode('utf-8-sig'))))
 return rows,summary

def choose_partial(rows):
 # Last RT-DETR trial with an epoch checkpoint and matching resume receipt.
 paths={r['path'].casefold():r for r in rows if r['kind']=='FILE'};found=[]
 for r in rows:
  low=re.sub('/+','/',r['path'].replace('\\','/')).casefold();m=re.search(r'/r50/r/rtdetr_r50/hpo/trial_(\d+)/fold(\d+)/cp/e(\d+)\.pt$',low)
  if m and r['path'].casefold()+'.resume.json' in paths:found.append((int(m[1]),int(m[3]),r['path']))
 if not found:return set()
 last=max(t for t,e,p in found)
 return {max([x for x in found if x[0]==last],key=lambda x:x[1])[2].casefold()}

def input_data(packet,home,project,pc):
 from utils.config_loader import load_config,cfg_path
 idx=home/'bt-hoy-20261103/rondas.json';index=json.loads(idx.read_text(encoding='utf-8-sig'));configs=[]
 for name,path in index.items():
  try:configs.append((name,load_config(path)))
  except Exception as exc:packet.errors.append(dict(source=str(path),error='Configuracion no legible para datos: '+str(exc)))
 if not configs:raise RuntimeError('No hay configuraciones actuales legibles para respaldar datos')
 seen=set()
 for name,cfg in configs:
  for key in ['detection_dir','classification_dir']:
   root=cfg_path(cfg,'paths.'+key)
   if not root.is_dir():packet.errors.append(dict(source=str(root),error='Derivados declarados ausentes'));continue
   for base,dirs,files in os.walk(root):
    dirs[:]=[d for d in dirs if d.casefold() not in SKIP|{'images','labels','crops','visualizations'} and not linked(Path(base)/d)]
    for n in files:
     p=Path(base)/n
     if p.suffix.lower() in {'.csv','.json','.yaml','.yml','.sha256','.txt'} and str(p).casefold() not in seen:
      namespace=hashlib.sha256(str(root.resolve()).casefold().encode()).hexdigest()[:10]
      packet.attempt(p,pc+'/00_PROYECTO_Y_EQUIPO/derivados_y_sellos/'+key+'/'+namespace+'/'+str(p.relative_to(root)).replace('\\','/'));seen.add(str(p).casefold())
  if pc!='PC-01':continue
  for key in ['raw_images_dir','raw_labels_dir']:
   root=cfg_path(cfg,'paths.'+key)
   if not root.is_dir():packet.errors.append(dict(source=str(root),error='Datos originales ausentes'));continue
   found=0
   for base,dirs,files in os.walk(root):
    dirs[:]=[d for d in dirs if d.casefold() not in SKIP and not linked(Path(base)/d)]
    for n in files:
     p=Path(base)/n
     if p.suffix.lower() in IMAGE|{'.txt','.json'}:found+=1
     if p.suffix.lower() in IMAGE|{'.txt','.json'} and str(p).casefold() not in seen:
      namespace=hashlib.sha256(str(root.resolve()).casefold().encode()).hexdigest()[:10]
      packet.attempt(p,pc+'/00_DATOS_COMPARTIDOS/'+key+'/'+namespace+'/'+str(p.relative_to(root)).replace('\\','/'),'datos_originales_sin_inferencia');seen.add(str(p).casefold())
   if not found:packet.errors.append(dict(source=str(root),error='Directorio original vacio: '+key))
 return index

def build(home,pc,inventory):
 project=home/'Documents/s2_no_borrar/Entrenamiento/Botrytis_Experiment_Framework_git'
 rows,summary=read_inventory(inventory,pc);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 folder=home/'Downloads/Botrytis_Tesis_Final'/('TESIS_'+pc+'_'+stamp+'_'+uuid.uuid4().hex[:6]);folder.mkdir(parents=True)
 target=folder/('TESIS_'+pc+'.zip');packet=Packet(target.with_suffix('.zip.partial'),pc);partial=choose_partial(rows)
 for r in rows:
  if r['kind']!='FILE':continue
  p=Path(r['path']);ext=p.suffix.lower();size=int(r['bytes']);low=str(p).replace('\\','/').casefold()
  if not allowed_source(p,home,project):packet.exclude(p,'fuera_de_raices_documentales',size);continue
  keep_weight=p.name=='final.pt' or str(p).casefold() in partial or p.is_relative_to(project/'weights') or p.is_relative_to(home/'r50/weights')
  if ext in {'.pt','.pth','.safetensors','.bin'} and not keep_weight:packet.exclude(p,'peso_intermedio_o_copia_no_necesario_para_redaccion; recibos y resultados incluidos',size);continue
  if ext=='.zip':packet.exclude(p,'archivo_zip_anterior; fuentes reales se incluyen por separado',size);continue
  if not ext and p.name.casefold() not in NAMED:packet.exclude(p,'objeto_interno_sin_extension; fuentes y manifiestos incluidos',size);continue
  if ext not in DOC|IMAGE|{'.pt','.pth','.safetensors','.bin'} and p.name.casefold() not in NAMED and not (size<8*1024**2 and any(x in DOC for x in p.suffixes)):packet.exclude(p,'formato_auxiliar_no_documental',size);continue
  packet.attempt(p,classify(str(p.relative_to(home)),pc),'checkpoint_parcial_con_recibo' if str(p).casefold() in partial else 'original')
 # Include a small reproducible sample of actual visual evidence; CSVs retain ALL cases.
 for r in rows:
  if r['kind']=='MASS_IMAGE_DIRECTORY' and Path(r['path']).name.casefold()=='visualizations':
   root=Path(r['path'])
   if not allowed_source(root,home,project):continue
   try:
    selected=sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE and not linked(p))[:6]
    for p in selected:packet.attempt(p,classify(str(p.relative_to(home)),pc),'muestra_visual_6_primeras_por_nombre')
    packet.warnings.append(dict(source=str(root),warning='Se incluyeron hasta seis ejemplos por nombre; CSV conserva todos los casos. Recortes masivos se regeneran con datos y codigo.'))
   except OSError as exc:packet.errors.append(dict(source=str(root),error=str(exc)))
 sys.path.insert(0,str(project/'src'))
 try:index=input_data(packet,home,project,pc)
 except Exception as exc:index={};packet.errors.append(dict(source='datos_y_configuraciones',error=str(exc)))
 packet.attempt(Path(__file__),pc+'/00_PROYECTO_Y_EQUIPO/exportador/'+Path(__file__).name)
 packet.attempt(inventory,pc+'/00_INDICES/'+inventory.name)
 support=Path(__file__).with_name('botrytis_documentos_apoyo_tesis.zip')
 if support.is_file():packet.attempt(support,pc+'/04_APOYO_PARA_REDACCION/'+support.name)
 else:packet.warnings.append(dict(source=str(support),warning='No se encontraron las tablas de apoyo de la conversacion'))
 git={}
 for key,args in [('commit',['rev-parse','HEAD']),('status',['status','--short']),('diff',['diff','--','src','configs','tools']),('diff_staged',['diff','--cached','--','src','configs','tools'])]:
  try:
   proc=subprocess.run(['git','-C',str(project),*args],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
   git[key]=dict(returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr)
  except (OSError,subprocess.SubprocessError) as exc:packet.warnings.append(dict(source='git '+key,warning=str(exc)))
 packet.generated(pc+'/00_INDICES/GIT_AL_EXPORTAR.json',git)
 # Critical final files: all available trained models and BOTH TEST results must survive export.
 for model in MODELS[pc]:
  if model=='rtdetr_r50':continue
  for suffix in [f'checkpoints/{model}/final.pt',f'outer_validation/{model}.json']:
   if not any(suffix in r['source'].replace('\\','/').casefold() for r in packet.rows):packet.errors.append(dict(source=model,error='No se exporto archivo final requerido: '+suffix))
 if pc=='PC-02':
  for pair in ['A','B']:
   p=home/f'bd/dos_test_finales_v1/{pair}/resultado.json'
   if str(p).casefold() not in packet.seen:packet.errors.append(dict(source=str(p),error='Resultado TEST final no exportado'))
 packet.generated(pc+'/00_INDICES/INVENTARIO_ORIGINAL_RESUMEN.json',summary)
 packet.generated(pc+'/00_INDICES/RONDAS.json',index)
 packet.generated(pc+'/00_INDICES/ERRORES.json',packet.errors)
 packet.generated(pc+'/00_INDICES/ADVERTENCIAS.json',packet.warnings)
 packet.generated(pc+'/00_INDICES/EXCLUIDOS.json',packet.excluded)
 buf=io.StringIO(newline='');fields=['source','archive','requested_archive','bytes','source_sha256','archive_sha256','operation','source_modified_utc'];w=csv.DictWriter(buf,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(packet.rows);packet.generated(pc+'/00_INDICES/INVENTARIO.csv',buf.getvalue())
 packet.generated(pc+'/00_INDICES/COBERTURA.json',{m:dict(files=sum('/'+m+'/' in r['archive'] for r in packet.rows),stages=sorted({r['archive'].split('/')[3] for r in packet.rows if '/01_MODELOS/'+m+'/' in r['archive']}),note='RT-DETR parcial; no se inventan robustez/final/VAL' if m=='rtdetr_r50' else 'Evidencia existente; no se ejecutan etapas') for m in MODELS[pc]})
 readme=GUIDE
 packet.generated(pc+'/LEEME.md',readme)
 packet.generated(pc+'/BITACORA_INICIAL.md',BITACORA)
 result=dict(pc=pc,status='INCOMPLETO_REVISAR_ERRORES' if packet.errors else 'PAQUETE_CREADO_CON_ADVERTENCIAS' if packet.warnings else 'PAQUETE_CREADO',files=len(packet.rows),uncompressed_bytes=packet.bytes,errors=len(packet.errors),warnings=len(packet.warnings),inventory_source=str(inventory),inventory_sha256=digest(inventory),exporter_sha256=digest(Path(__file__)),partial_checkpoint_included=sorted(r['source'] for r in packet.rows if r['operation']=='checkpoint_parcial_con_recibo'),original_data_included=any(r['operation']=='datos_originales_sin_inferencia' for r in packet.rows),test_reexecuted=False,training_started=False)
 packet.generated(pc+'/00_INDICES/RESUMEN.json',result);packet.z.close();Path(packet.z.filename).rename(target)
 emit('VERIFICANDO_ZIP_TESIS',path=str(target))
 with zipfile.ZipFile(target) as z:
  bad=z.testzip()
  if bad:raise RuntimeError('CRC incorrecto en '+bad)
 h=digest(target);target.with_suffix('.zip.sha256.txt').write_text(h+'  '+target.name+'\n',encoding='utf-8');(folder/'LEEME.md').write_text(readme,encoding='utf-8')
 emit('PAQUETE_TESIS_FINAL_CREADO',path=str(target),sha256=h,GB=round(target.stat().st_size/1024**3,3),**result)
 return target,result

GUIDE='''# Evidencias completas para tesis

Un paquete por maquina, y dentro por modelo/etapa. No es una nueva ejecucion ni un analisis de resultados.
00_INDICES/INVENTARIO.csv mapea TODAS las copias incluidas con origen, ruta ZIP, operacion y hash.
Las rutas largas se acortan y quedan en requested_archive. Extraer en ruta corta como C:\\tesis.
Conservar cada ZIP y .sha256.txt. Los tres paquetes se extraen en una carpeta comun; cada uno contiene PC-XX.

00_PROYECTO_Y_EQUIPO: codigo nativo/local, helpers y antecedentes, configuraciones, dependencia/entornos,
QA, derivados/sellos y registros. Las referencias Windows en JSON son originales; usar el inventario portable.
01_MODELOS/<modelo>/: configuracion, baseline, HPO, robustez, entrenamiento final, VAL, pesos y auditorias.
Todos los resultados de modelos descartados y de intentos anteriores se conservan como tales.
02_SISTEMAS_TEST en PC2: A=YOLO26l+ResNet50; B=YOLO26l+Swin; freezes, liberaciones y protocolo,
resultados, cajas, probabilidades, omisiones, errores, tiempos descriptivos y figuras. TEST no se reejecuta.
04_APOYO_PARA_REDACCION: ZIP de tablas/graficos comparativos VAL y scripts preparados en la conversacion.
Son analisis derivados, no sustituyen los resultados originales; sus rutas de entrada deben adaptarse.
03_AUDITORIA_Y_ANTECEDENTES: documentos no atribuibles de forma univoca a un modelo; no se inventa atribucion.
00_DATOS_COMPARTIDOS solo PC1: copia de entradas originales, incluidos los archivos que el inventario
identifica como excluidos/OFF. Que se respalden no significa que se utilizaran. Los manifiestos determinan
las entradas activas y los splits; no modificar ni mover etiquetas/datos para reconstruir el experimento.

Para metodologia/experimentacion: datos/etiquetado/pairing, splits/folds, semillas, inicializacion,
resolucion, aumentos, normalizacion, batch, perdidas/optimizador/scheduler, budgets, parada temprana,
estudios/estados de Optuna, parametros elegidos, hardware/software, comandos y logs, decisiones/desviaciones.
Para resultados/graficos: metrics.json, histories, summary, fold/trial metrics, predicciones con GT/probabilidades
 o cajas, PR/ROC/AP-IoU, matrices FP/FN, robustez por semillas/folds, resultados del sistema integrado.
Conservar etapas distintas: metricas GT crops no equivalen a pipeline con crops predichos; VAL no equivale a TEST.
Los mAP de VAL no se denominan mAP de TEST. Los tiempos del pipeline son descriptivos, no benchmark controlado.

Se incluyen los final.pt disponibles y pesos iniciales locales de weights/. Para RT-DETR se conserva el
ultimo checkpoint con recibo disponible del ultimo trial observado; sus documentos/estudio parcial se incluyen.
EXCLUIDOS.json inventaria los otros pesos intermedios, ZIP antiguos y objetos internos grandes: no es backup
byte a byte para reanudar todos los intentos. Incluye estados de muestreador .pkl y bases SQLite consistentes.
No abrir .pkl de fuentes desconocidas. Cada SQLite es una instantanea consistente individual; no una transaccion
global de toda la PC. Los CSV se copian byte por byte, incluso si tienen filas irregulares, con advertencia.
Hasta seis visualizaciones por carpeta; todas las predicciones se conservan en tablas. Recortes masivos se
regeneran con datos y codigo, con sus sellos de contenido. Las figuras existentes si se incluyen.

ERRORES.json debe estar vacio para considerar completo el exportado. ADVERTENCIAS requiere lectura;
COBERTURA enumera lo encontrado, no certifica fases terminadas. Artefactos de fuentes historicas pueden
referir pesos/ZIP excluidos: consultar el inventario. La bitacora inicial procede de la conversacion y
requiere contrastarse con logs; no inventa tiempos, causas tecnicas no comprobadas o metricas faltantes.
'''
BITACORA='''# Bitacora inicial para contrastar con evidencias

Fuente: conversacion y salidas compartidas por el equipo; fechas exactas deben citarse desde logs.
- Reanudacion R50: error de resta entre strings; version siguiente verifico historiales.
- R50: ausencia de recibo .resume.json detuvo recuperacion; se uso checkpoint previamente publicado,
  conservando un archivo posterior sin recibo. Conservar auditorias y planes de recuperacion.
- Clasificadores: hubo ajustes de recuperacion/RNG en ayudantes; conservar las versiones efectivamente usadas.
- ResNet50: primer instalador no copio SHA256SUMS.json; se corrigio la instalacion, sin borrar resultados.
- ResNet50: robustez directa fue bloqueada por modo supervisado; se continuo mediante jobs/worker.
- Exportacion anterior: WinError32 al cerrar temporales SQLite; snapshots ya archivados se verificaron
  con hash/integridad y se generaron complementos. Se corrigio cierre explicito de conexiones.
- CSV de entrenamiento final YOLO26l: ultima fila irregular. Se conservo original con advertencia; no imputar.
- RT-DETR fue pausado por decision del equipo ante el tiempo/costo de ejecucion; no afirmar HPO completo.
  El checkpoint mas reciente del inventario puede ser posterior a las primeras capturas compartidas.
- Detector elegido por equilibrio rendimiento/costo teorico: YOLO26l; distinta prioridad del desempate original.
- Para clasificacion se priorizo sensibilidad/FN positivos y se analizo F2 en VAL.
- Se declararon dos sistemas finales A/B y umbrales antes de TEST. El profesor elegira tras consultar TEST;
  no presentar el resultado del ganador como evaluacion independiente de esa seleccion.
- Worker silenciaba mensajes Speed (YOLO_VERBOSE=False); paso10 no persistia res.speed.
  Conservar tiempo de entrenamiento y tiempos descriptivos de evaluacion; no inventar benchmark17.

Completar manualmente decisiones y observaciones que no quedaron registradas en archivos locales.
'''

def main():
 if os.name!='nt':raise RuntimeError('Ejecutar en las PCs Windows del proyecto')
 pc=USERS.get(os.environ.get('USERNAME','').casefold())
 if not pc:raise RuntimeError('Usuario no reconocido')
 home=Path.home();project=home/'Documents/s2_no_borrar/Entrenamiento/Botrytis_Experiment_Framework_git'
 os.chdir(project)
 import psutil
 me=psutil.Process();exempt={me.pid,*[p.pid for p in me.parents()]}
 for p in psutil.process_iter(['pid','name','cmdline']):
  if p.pid not in exempt and 'python' in (p.info['name'] or '').lower() and 'botrytis' in ' '.join(p.info['cmdline'] or []).lower():raise RuntimeError('Otro Python de Botrytis activo; esperar su cierre sin interrumpirlo')
 inventories=sorted((home/'Downloads').glob('Inventario_Tesis_'+pc+'_*.zip'),key=lambda p:p.stat().st_mtime_ns)
 if not inventories:raise RuntimeError('No se encuentra inventario de esta PC en Downloads')
 if shutil.disk_usage(home).free<8*1024**3:raise RuntimeError('Se requieren al menos 8 GiB libres para este paquete')
 _,result=build(home,pc,inventories[-1]);return 2 if result['errors'] else 0
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as exc:emit('EMPAQUETADO_DETENIDO',error=str(exc),sources_modified=False);sys.exit(2)
