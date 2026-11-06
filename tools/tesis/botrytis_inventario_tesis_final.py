"""Inventario documental local para organizar la tesis. No ejecuta modelos ni abre bases de datos."""
from pathlib import Path
from datetime import datetime,timezone
import csv,io,json,os,subprocess,sys,uuid,zipfile
USERS={'i21315':'PC-01','i21314':'PC-02','i21313':'PC-03'}
MODELS={'PC-01':['yolov8n','yolov8l','resnet50'],'PC-02':['yolo26l','swin_tiny','TEST A/B en bd'],'PC-03':['vit_b16','rtdetr_r50 (parcial)']}
ROOTS=['bf','bg','bn','bp','br','bte','bt-hoy-20261103','btlog','bv','r50','bd']
SKIP={'.git','.venv','venv','__pycache__','.pytest_cache','node_modules','.cache'}
MASS={'crops','images','labels','visualizations'}
PRIVATE={'.env','.env.local','credentials.json','token.json','id_rsa','id_ed25519'}

def linked(path):return path.is_symlink() or (hasattr(path,'is_junction') and path.is_junction())

def gather(home,pc):
    project=home/'Documents/s2_no_borrar/Entrenamiento/Botrytis_Experiment_Framework_git'
    roots=[home/n for n in ROOTS if (home/n).is_dir()]
    roots += [project/n for n in ['src','configs','docs','tools','reports','logs_qa','artifacts','runs','results','weights'] if (project/n).is_dir()]
    downloads=home/'Downloads'
    if downloads.is_dir():roots += [p for p in downloads.glob('botrytis*') if p.is_dir() and not linked(p)]
    rows=[];errors=[]
    def add(path,kind,root,reason=''):
        try:
            stat=path.stat()
            rows.append(dict(pc=pc,root=str(root),path=str(path),relative=str(path.relative_to(root)) if path!=root else '.',kind=kind,bytes=stat.st_size if path.is_file() else '',modified_utc=datetime.fromtimestamp(stat.st_mtime,timezone.utc).isoformat(),note=reason))
        except OSError as exc:errors.append(dict(path=str(path),error=str(exc)))
    for root in roots:
        if linked(root):
            add(root,'OMITTED_LINK',root);continue
        add(root,'DIRECTORY',root)
        for base,dirs,names in os.walk(root,followlinks=False,onerror=lambda exc:errors.append(dict(path=exc.filename,error=str(exc)))):
            parent=Path(base);kept=[]
            for name in sorted(dirs):
                child=parent/name
                if name.casefold() in SKIP or linked(child):continue
                if name.casefold() in MASS:
                    add(child,'MASS_IMAGE_DIRECTORY',root,'Carpeta registrada; fotografias/recortes individuales no enumerados. Seleccionar ejemplos al empaquetar.')
                else:kept.append(name);add(child,'DIRECTORY',root)
            dirs[:]=kept
            for name in sorted(names):
                path=parent/name
                if name.casefold() in PRIVATE or linked(path):continue
                add(path,'FILE',root)
    if project.is_dir():
        for pattern in ['README*','requirements*','pyproject.toml','LICENSE*','*.lock']:
            for p in project.glob(pattern):
                if p.is_file() and not linked(p):add(p,'FILE',project)
    if downloads.is_dir():
        for p in downloads.glob('botrytis*'):
            if p.is_file() and p.suffix.casefold() in ('.py','.cmd','.json','.md') and not linked(p):add(p,'FILE',downloads)
    git={}
    if project.is_dir():
        for key,args in [('commit',['rev-parse','HEAD']),('status',['status','--short'])]:
            try:
                result=subprocess.run(['git','-C',str(project),*args],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)
                git[key]=result.stdout if result.returncode==0 else result.stderr
            except (OSError,subprocess.SubprocessError) as exc:git[key]=str(exc)
    return rows,errors,git

def build(home,pc):
    rows,errors,git=gather(home,pc)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=home/'Downloads'/('Inventario_Tesis_'+pc+'_'+stamp+'_'+uuid.uuid4().hex[:6]);out.mkdir(parents=True)
    fields=['pc','root','path','relative','kind','bytes','modified_utc','note'];buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    (out/'ARCHIVOS.csv').write_text(buf.getvalue(),encoding='utf-8-sig')
    lines=['INVENTARIO DOCUMENTAL '+pc,'Modelos: '+', '.join(MODELS[pc]),'Solo nombres, tamanos y fechas. No contiene resultados ni pesos; no ejecuta modelos.','Carpetas masivas de imagenes se registran sin enumerar fotografias individuales.','']
    previous=None
    for row in rows:
        if row['root']!=previous:lines+=['','RAIZ: '+row['root']];previous=row['root']
        depth=len(Path(row['relative']).parts)-1
        lines.append('  '*max(depth,0)+('[/] ' if row['kind']=='DIRECTORY' else '[IMAGENES] ' if row['kind']=='MASS_IMAGE_DIRECTORY' else '')+row['relative']+((' | '+str(row['bytes'])+' bytes') if row['bytes']!='' else ''))
    (out/'ARBOL.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    (out/'RESUMEN.json').write_text(json.dumps(dict(pc=pc,models=MODELS[pc],created_utc=stamp,rows=len(rows),files=sum(r['kind']=='FILE' for r in rows),roots=list(dict.fromkeys(r['root'] for r in rows)),errors=errors,git=git,scope='Inventario de documentos; no contenidos. Incluye TEST ya ejecutado en bd y rondas parciales de r50 si existen.',training_started=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    target=out.with_suffix('.zip')
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():z.write(p,p.name)
    print('INVENTARIO_TESIS_CREADO '+json.dumps(dict(pc=pc,path=str(target),files=sum(r['kind']=='FILE' for r in rows),errors=len(errors)),ensure_ascii=False),flush=True)
    return target

def main():
    if os.name!='nt':raise RuntimeError('Ejecutar en Windows de las PCs del proyecto')
    pc=USERS.get(os.environ.get('USERNAME','').casefold())
    if not pc:raise RuntimeError('Usuario no reconocido; no se inspecciona el perfil')
    build(Path.home(),pc)
if __name__=='__main__':main()
