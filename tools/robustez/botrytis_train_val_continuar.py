"""Comprueba robustez existente -> APROBAR real -> TRAIN70 fijo -> VAL20. TEST cerrado.
Solo 09 y 10. Nunca inicia baseline/HPO/repeat/aggregate/thresholds ni RT-DETR.
Ejecutar fuera del checkout junto al paquete SHA256 fijado.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import uuid

SELF=Path(__file__).resolve();HERE=SELF.parent
SHA=hashlib.sha256(SELF.read_bytes()).hexdigest()
ROBUST_SHA='ae39692639db6cd33e7e7057e28efeb9079b507043c6363cb0a1f6f57312ae79'
p=HERE/'botrytis_robustez_reanudar.py'
if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=ROBUST_SHA:
    raise RuntimeError('Ejecutor de robustez fijado ausente/alterado')
s=importlib.util.spec_from_file_location('_bt_outer_verified_robust',p)
r=importlib.util.module_from_spec(s);sys.modules[s.name]=r;s.loader.exec_module(r)
bt=r.bt
bt.__file__=str(SELF);bt.EXECUTOR_SHA256=SHA
bt.POST_KINDS=('outer_train','outer_eval')
bt.POST_FILES={'outer_train':'09_train_outer_candidate.py','outer_eval':'10_evaluate_outer_validation.py'}
MODELS={'PC-01':['yolov8n','yolov8l'],'PC-02':['yolo26l','swin_tiny'],'PC-03':['vit_b16']}

# The proven native final-training/recovery wrapper remains in use. Apply the
# classifier RNG compatibility fix also to its native outer-training call.
_install_outer=bt.install_outer_recovery

def install_outer(module,cfg,spec,state,hooks):
    _install_outer(module,cfg,spec,state,hooks)
    if not spec.is_detector:
        train=module.train_classifier
        def classifier(*a,**kw):
            with bt.classifier_rng_restore(hooks.event):return train(*a,**kw)
        module.train_classifier=classifier
bt.install_outer_recovery=install_outer


class NotReady(bt.Stop):pass


def restore_outer_pauses(database,model,paths,*,apply=False):
    """Only undo this model's original PENDING->BLOCKED priority pause for 09/10."""
    permitted=[]
    for path in paths:
        old=bt.read(path)
        if old.get('architecture')!=model or Path(old['database']).resolve()!=Path(database).resolve():continue
        marker='POSPUESTO_POR_PRIORIDAD '+str(path)
        if old.get('marker')!=marker:raise bt.Stop('Recibo de pausa incompatible: '+str(path))
        with sqlite3.connect(database,timeout=30) as con:
            con.row_factory=sqlite3.Row
            if apply:con.execute('BEGIN IMMEDIATE')
            for row in old.get('postponed',[]):
                if row['kind'] not in bt.POST_KINDS:continue
                cur=con.execute('SELECT * FROM jobs WHERE job_id=?',(row['job_id'],)).fetchone()
                if cur is None:raise bt.Stop('Job de pausa ausente')
                if cur['status']!='BLOCKED' or cur['error']!=marker:continue
                if row['status']!='PENDING' or cur['architecture']!=model:raise bt.Stop('Pausa no proviene del PENDING propio')
                for key in ('kind','architecture','payload','depends_on','attempt'):
                    a,b=cur[key],row[key]
                    if key in ('payload','depends_on'):
                        a=json.loads(a) if isinstance(a,str) else a;b=json.loads(b) if isinstance(b,str) else b
                    if a!=b:raise bt.Stop('Identidad de pausa modificada: '+row['job_id'])
                permitted.append(row['job_id'])
                if apply:
                    con.execute("UPDATE jobs SET status='PENDING',error=?,updated_at=? WHERE job_id=? AND status='BLOCKED' AND error=?",
                                (row.get('error'),bt.utc(),row['job_id'],marker))
    return permitted


def ledger_entry(cfg):
    from utils.experiments import external_ledger,fingerprint
    path=external_ledger(cfg);row=bt.read(path).get(cfg['experiment']['architecture']) if path.is_file() else None
    if row is not None and row.get('fingerprint')!=fingerprint(cfg):
        raise bt.Stop('VAL20 predeclarado para otra ronda; no se reutiliza ni se sobrescribe')
    return row


def verify_external_state(cfg,spec,jobs):
    from utils.hpo_results import stage_json_path
    row=ledger_entry(cfg);use=(row or {}).get('uses',{}).get('outer_validation')
    output=stage_json_path(cfg,'outer_validation',spec)
    if use is None:
        if output.exists():raise bt.Stop('Salida VAL20 sin predeclaracion de uso; no se aprueba retroactivamente')
        return row
    final=bt.validate_final(cfg,spec)
    if use.get('checkpoint_sha256')!=final['sha256']:raise bt.Stop('Uso VAL20 de otro checkpoint; no se repite')
    if use.get('state')=='COMPLETE' or output.exists():bt.validate_external(cfg,spec,'outer_validation')
    elif use.get('state')!='STARTED' or bt.one_job(jobs,'outer_eval')['status'] not in ('RUNNING','FAILED'):
        raise bt.Stop('Uso VAL20 previo sin cierre ni job interrumpido recuperable; no se ejecuta otra evaluacion')
    return row


def verify_model(project,pc,model,index,out):
    from utils.config_loader import load_config,cfg_path
    from utils.architectures import get_arch
    from utils.experiments import verify_round,fingerprint
    from utils.manual_gates import require_gate
    from utils.optuna_storage import study_db_path,load_study
    from utils.hpo_results import top_configs,best_config_path
    from orchestration.pc_registry import PCRegistry
    from orchestration.study_queue import StudyQueue
    if model not in MODELS[pc] or model not in index:raise NotReady('Modelo fuera de alcance o sin ronda: '+model)
    config=Path(index[model]);source=load_config(config);spec=get_arch(model)
    short,runs,results,state=bt.make_short_paths(config,Path.home()/'bt-hoy-20261103')
    sets=['paths.runs_dir='+str(runs),'paths.results_dir='+str(results)];cfg=load_config(config,sets)
    verify_round(cfg,architecture=model)
    if fingerprint(cfg)!=fingerprint(source):raise bt.Stop('Rutas cortas cambiaron la identidad experimental')
    if PCRegistry(cfg).owner(model)!=pc or cfg['compute']['pc_id']!=pc:raise bt.Stop('Ownership/configuracion incompatibles')
    require_gate(cfg,pc,'READY_FOR_EXPERIMENT')
    if cfg['robustness']['seeds']!=[17,42,73] or cfg['robustness']['top_k_configs']!=3 or cfg['inner_hpo']['folds']!=3:
        raise bt.Stop('Protocolo distinto al original')
    evidence=out/model;evidence.mkdir(parents=True,exist_ok=True)
    code=r.audit.readonly_code(project)
    if code['code_hash']!=r.CODE or code['git_commit']!=r.COMMIT:raise bt.Stop('Codigo distinto: no se cambia el checkout')
    scratch,data=r.audit.readonly_prepared(cfg,evidence/'prep')
    if data.dataset_hash!=r.DATA or data.split_hash!=r.SPLIT:raise bt.Stop('Datos/splits incompatibles')
    queue=StudyQueue.from_cfg(cfg);jobs=list(map(bt.pure_job,bt.jobs_read(cfg,model)))
    for kind in ('baseline','hpo'):
        j=bt.one_job(jobs,kind)
        if j['status']!='COMPLETE':raise NotReady(kind+' no aprobado/COMPLETE; no se ejecuta: '+j['status'])
    if not study_db_path(cfg,spec).is_file():raise bt.Stop('Falta SQLite HPO; no se crea estudio')
    study=load_study(spec,cfg);cands=top_configs(study,3)
    bt.validate_queue_plan(cfg,spec,cands,jobs)
    repeats=[j for j in jobs if j['kind']=='repeat']
    pending=[dict(job=j['job_id'],status=j['status'],attempt=j['attempt']) for j in repeats if j['status']!='COMPLETE']
    if len(repeats)!=9 or pending:
        bt.emit('ROBUSTEZ_PENDIENTE',model=model,expected_repeats=9,found=len(repeats),jobs=pending)
        raise NotReady('Robustez incompleta: '+json.dumps(pending,ensure_ascii=False))
    agg=bt.one_job(jobs,'aggregate')
    if agg['status'] not in ('COMPLETE','PENDING_REVIEW'):raise NotReady('Agregacion no lista: '+agg['status'])
    # No training is run by these validators; all 27 signed fold histories must agree.
    hpo=r.verify_closed_hpo(study,cfg,spec,bt.one_job(jobs,'hpo'),evidence)
    aggregation=bt.validate_aggregate(cfg,spec,cands,jobs)
    baseline=r.baseline_evidence(source,cfg,spec,r.CODE,r.COMMIT)
    isolation=r.audit.partition_evidence(scratch,spec)
    if isolation['confirmed_relevant_conflicts']:raise bt.Stop('Fuga confirmada: no se autoriza continuar')
    best=aggregation['selected'];bp=best_config_path(cfg,spec)
    bt.protect_existing_final(cfg,spec)
    row=verify_external_state(cfg,spec,jobs)
    pause_paths=list((Path.home()/'bp').glob('*/'+model+'/pause.json'))
    allowed=restore_outer_pauses(queue.path,model,pause_paths)
    for j in jobs:
        if j['kind'] in bt.POST_KINDS and j['status']=='BLOCKED' and j['job_id'] not in allowed:
            raise bt.Stop('Bloqueo externo ajeno a pausa: '+j['job_id']+' '+str(j.get('error')))
    warnings=[dict(path=curve['path'],flags=curve.get('flags',[])) for check in aggregation['repeat_evidence']
              for curve in check['curves'] if curve.get('flags')]
    record=dict(model=model,config=str(config),fingerprint=fingerprint(cfg),best_config_path=str(bp),
        best_config_sha256=bt.digest(bp),selected=best,baseline=baseline,hpo_best=study.best_value,
        hpo_states=hpo['trials'],robustness_repeats=9,robustness_folds=27,aggregate_status=agg['status'],
        isolation=isolation,curve_warnings=warnings,source_commit=code['git_commit'],code_hash=code['code_hash'],
        dataset_hash=data.dataset_hash,split_hash=data.split_hash,seed=cfg['reproducibility']['seed'],
        executor_sha256=SHA,test_opened=False,results_dir=str(results),runs_dir=str(runs))
    bt.atomic_json(evidence/'revision.json',record)
    bt.snapshot_sqlite(queue.path,evidence/'queue-before.sqlite')
    bt.emit('ROBUSTEZ_VERIFICADA',model=model,repeats=9,folds=27,config_id=best['config_id'],
            mean=best['score_mean'],sd=best['score_sd'],final_epochs=best['final_epochs'],report=str(evidence/'revision.json'))
    return dict(project=project,pc=pc,model=model,config=config,cfg=cfg,spec=spec,queue=queue,jobs=jobs,cands=cands,
                sets=sets,state=state/'post',evidence=evidence,record=record,pause_paths=pause_paths,ledger=row)


def authorize_ready(items,out):
    """One actual APROBAR covers the displayed candidates and predeclared VAL20."""
    from utils.experiments import authorize_external
    need=any(bt.one_job(x['jobs'],'aggregate')['status']=='PENDING_REVIEW' or x['ledger'] is None for x in items)
    scope=[dict(model=x['model'],config_id=x['record']['selected']['config_id'],
                final_epochs=x['record']['selected']['final_epochs'],fingerprint=x['record']['fingerprint'],
                revision=str(x['evidence']/'revision.json'),revision_sha256=bt.digest(x['evidence']/'revision.json')) for x in items]
    if need:
        bt.emit('REVISION_PRESENCIAL_REQUERIDA',candidates=scope,
                operation='Aprobar agregaciones mostradas, entrenar con TRAIN70 a epocas fijas y evaluar una vez en VAL20. Sin TEST ni HPO.',
                limitation='No certifica ausencia de sobreajuste ni independencia biologica; las advertencias constan en los informes.')
        for x in items:
            bt.emit('RESUMEN_REVISION',model=x['model'],baseline=x['record']['baseline']['mean'],
                    hpo=x['record']['hpo_best'],robustness=x['record']['selected']['score_mean'],
                    warnings=x['record']['curve_warnings'],isolation=x['record']['isolation']['status'])
        if not sys.stdin.isatty():raise bt.Stop('Se requiere operador presente; informes listos, no se inventa aprobacion')
        if input('Revise los informes. Escriba APROBAR para esos candidatos TRAIN70 -> VAL20: ').strip()!='APROBAR':
            raise bt.Stop('Sin aprobacion: no se inicia entrenamiento ni VAL20')
        for x in items:
            if bt.digest(x['record']['best_config_path'])!=x['record']['best_config_sha256']:
                raise bt.Stop('Candidato cambio durante la revision; no se aprueba ni se autoriza VAL20')
        acceptance=out/'aprobacion.json'
        bt.atomic_json(acceptance,dict(decision='APROBAR',operator=os.environ['USERNAME'],utc=bt.utc(),
            candidates=scope,scope='aggregate approval + fixed TRAIN70 + predeclared VAL20; TEST closed',
            human_visual_inspection_claimed=False,executor_sha256=SHA))
        for x in items:
            q=x['queue'];agg=q.get(bt.one_job(x['jobs'],'aggregate')['job_id'])
            if agg.status=='PENDING_REVIEW':q.approve(agg.job_id,operator=os.environ['USERNAME'],evidence=str(acceptance))
            if ledger_entry(x['cfg']) is None:authorize_external(x['cfg'],operator=os.environ['USERNAME'],evidence=str(acceptance))
    for x in items:
        restore_outer_pauses(x['queue'].path,x['model'],x['pause_paths'],apply=True)


def complete_model(x):
    from utils.hpo_results import best_config_path
    bt.emit('CONTINUACION_TRAIN_VAL',model=x['model'],stages=list(bt.POST_KINDS))
    cfg,spec,q=x['cfg'],x['spec'],x['queue'];x['state'].mkdir(parents=True,exist_ok=True)
    for kind in bt.POST_KINDS:
        if bt.digest(best_config_path(cfg,spec))!=x['record']['best_config_sha256']:raise bt.Stop('Candidato cambio desde revision; no se continua')
        jobs=list(map(bt.pure_job,bt.jobs_read(cfg,spec.slug)));job=bt.one_job(jobs,kind)
        byid={j['job_id']:j for j in jobs}
        if not all(byid[d]['status']=='COMPLETE' for d in job['depends_on']):raise NotReady('Dependencias no aprobadas de '+job['job_id'])
        if job['status']=='COMPLETE':
            bt.technical_job_result(cfg,spec,job,x['cands'],jobs)
            bt.emit('ETAPA_EXISTENTE_VERIFICADA',model=spec.slug,kind=kind,training_repeated=False)
            continue
        if bt.python_processes():raise bt.Stop('Otro Python activo; no se duplica entrenamiento')
        if kind=='outer_train':bt.ensure_partial_resume(cfg,spec,job)
        if bt.recovery_or_reuse(q,cfg,spec,job,x['cands'],jobs,x['state'],x['evidence']):continue
        if job['status'] not in ('PENDING','RUNNING','FAILED'):
            raise bt.Stop('Etapa no habilitada: '+job['status'])
        bt.emit('TRABAJO_ACTUAL',model=spec.slug,kind=kind,job=job['job_id'],configuration=x['record']['selected']['config_id'])
        bt.launch_visible(x['project'],cfg,spec,x['config'],x['sets'],kind,x['state'],x['evidence'],target_job=job['job_id'])
        updated=bt.pure_job(q.get(job['job_id']))
        if updated['status'] not in ('COMPLETE','PUBLICATION_PENDING'):raise bt.Stop('Trabajo sin cierre: '+str(updated))
        bt.technical_job_result(cfg,spec,updated,x['cands'],list(map(bt.pure_job,bt.jobs_read(cfg,spec.slug))))
        if updated['status']=='PUBLICATION_PENDING':
            bt.recovery_or_reuse(q,cfg,spec,updated,x['cands'],jobs,x['state'],x['evidence'])
    final=bt.technical_job_result(cfg,spec,bt.one_job(list(map(bt.pure_job,bt.jobs_read(cfg,spec.slug))),'outer_train'),x['cands'],[])
    val=bt.validate_external(cfg,spec,'outer_validation');train=final['train_evaluation']
    use=(ledger_entry(cfg) or {}).get('uses',{}).get('outer_validation',{})
    if use.get('state')=='STARTED':
        # Only after verifying the published output and the same checkpoint.
        from utils.experiments import finish_external
        finish_external(cfg,'outer_validation')
    key='map50_95' if spec.is_detector else 'average_precision_positive'
    train_value=train.get(key);val_value=val['metrics'][key]
    gap=float(train_value)-float(val_value) if train_value is not None and math.isfinite(float(train_value)) else None
    record=dict(x['record'],status='TRAIN_VAL_COMPLETOS',final=final,outer_validation=val,
                objective=key,train_value=train_value,validation_value=val_value,gap_train_validation=gap,
                conclusion='Brecha descriptiva del mismo checkpoint, no diagnostico definitivo de sobreajuste. No se declara ganador global.',
                baseline_comparison='Baseline/HPO/robustez son CV interna; no son una evaluacion externa comparable del baseline.',
                final_validation_per_epoch=False,test_opened=False,thresholds_launched=False)
    bt.atomic_json(x['evidence']/'resultado.json',record)
    bt.emit('TRAIN_VAL_COMPLETOS',model=spec.slug,objective=key,train=train_value,validation=val_value,
            report=str(x['evidence']/'resultado.json'),validation_file=val['path'],test_opened=False)
    return record


def write_report(out,pc,verified,completed,errors):
    report=dict(pc=pc,verified=[x['record'] for x in verified],completed=completed,errors=errors,
                scope=MODELS[pc],rtdetr_deferred=True,test_opened=False,baseline_hpo_robustness_repeated=False,
                next='Calibracion/seleccion e integracion del hibrido segun protocolo; TEST requiere freeze y autorizacion aparte.')
    bt.atomic_json(out/'resumen.json',report)
    rows=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
        (x['model'],x['objective'],x['train_value'],x['validation_value'],x['gap_train_validation']))+'</tr>' for x in completed)
    page='<!doctype html><meta charset="utf-8"><title>Botrytis TRAIN-VAL</title><style>body{font:16px system-ui;margin:30px;max-width:1100px}td,th{padding:10px;border:1px solid #aaa}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>TRAIN70 y VAL20 — '+html.escape(pc)+'</h1><p>TEST10 cerrado. No se declara ganador global ni ausencia de sobreajuste. Las curvas de CV son internas; el entrenamiento final usa epocas fijas, sin curvas VAL20 por epoca. Cada brecha compara el mismo checkpoint.</p><table><tr><th>Modelo</th><th>Metrica</th><th>TRAIN</th><th>VAL20</th><th>Brecha</th></tr>'+rows+'</table><h2>Evidencias y limitaciones</h2><pre>'+html.escape(json.dumps(report,ensure_ascii=False,indent=2,default=str))+'</pre>'
    bt.atomic_bytes(out/'resumen.html',page.encode('utf-8'))
    bt.emit('FIN_TRAIN_VAL',report=str(out/'resumen.html'),json=str(out/'resumen.json'),errors=errors,test_opened=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-pc',choices=list(MODELS));p.add_argument('--project')
    p.add_argument('--only-model',choices=sorted({m for v in MODELS.values() for m in v}))
    p.add_argument('--check-only',action='store_true',help='Solo comprobar robustez, sin aprobaciones ni entrenamiento')
    p.add_argument('--_stage',choices=['worker','outer_train','outer_eval'],help=argparse.SUPPRESS)
    p.add_argument('--state-dir',help=argparse.SUPPRESS);p.add_argument('--executor-sha',help=argparse.SUPPRESS)
    p.add_argument('--target-job',help=argparse.SUPPRESS)
    args,rest=p.parse_known_args()
    if args._stage:
        if args.executor_sha!=SHA:raise bt.Stop('SHA del hijo distinto')
        native=rest[1:] if rest[:1]==['--'] else rest
        child=argparse.ArgumentParser(add_help=False)
        child.add_argument('--pc-id');child.add_argument('--architecture');child.add_argument('--architectures',nargs='+')
        child.add_argument('--kinds',nargs='+')
        ca,_=child.parse_known_args(native)
        models=ca.architectures if args._stage=='worker' else [ca.architecture]
        if not models or any(m not in MODELS.get(ca.pc_id,[]) for m in models):raise bt.Stop('Modelo/PC del hijo fuera de alcance; RT-DETR permanece pospuesto')
        if args._stage=='worker' and (not ca.kinds or any(k not in bt.POST_KINDS for k in ca.kinds)):
            raise bt.Stop('Worker fuera de alcance: solo outer_train y outer_eval')
        bt.TARGET_JOB_ID=args.target_job
        return bt.child_entry(args._stage,Path(args.project),native,Path(args.state_dir))
    if rest:p.error('Argumentos desconocidos')
    if os.name!='nt':raise bt.Stop('Operacion exclusiva Windows; pruebas CPU separadas')
    project=bt.find_project(args.project);py=project/'.venv/Scripts/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():return bt.subprocess.call([str(py),'-u',str(SELF),*sys.argv[1:]])
    os.chdir(project);sys.path.insert(0,str(project/'src'))
    os.environ.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',MPLBACKEND='Agg',BOTRYTIS_BOOTSTRAP_DIR=str(Path.home()/'btlog'))
    from utils.config_loader import load_config
    from orchestration.machine import read_binding
    cfg=load_config(project/'artifacts/v4/local.yaml');pc=cfg['compute']['pc_id'];binding=read_binding(cfg)
    if pc!=args.expected_pc or os.environ.get('USERNAME','').casefold()!=r.USERS.get(pc) or binding['pc_id']!=pc or binding['machine_id'].casefold()!=r.USERS[pc]:
        raise bt.Stop('Usuario, binding, configuracion y PC esperada no coinciden')
    if args.only_model and args.only_model not in MODELS[pc]:raise bt.Stop('Modelo no pertenece a la PC o fuera de alcance')
    models=[args.only_model] if args.only_model else MODELS[pc]
    index=bt.read(Path.home()/'bt-hoy-20261103/rondas.json')
    out=Path.home()/'bv'/uuid.uuid4().hex[:8];out.mkdir(parents=True)
    ready=[];completed=[];errors=[]
    with r.controller_lock(cfg,pc),r.wait_idle(cfg,pc):
        try:
            for model in models:
                try:ready.append(verify_model(project,pc,model,index,out))
                except (Exception,bt.RecoveryStop) as exc:
                    errors.append(dict(model=model,error=str(exc)));bt.emit('MODELO_NO_HABILITADO',model=model,error=str(exc),results_preserved=True)
            if ready and not args.check_only:
                try:authorize_ready(ready,out)
                except (Exception,bt.RecoveryStop) as exc:
                    errors.append(dict(phase='approval',error=str(exc)))
                    raise
                for x in ready:
                    try:completed.append(complete_model(x))
                    except (Exception,bt.RecoveryStop) as exc:
                        errors.append(dict(model=x['model'],error=str(exc)));bt.emit('MODELO_DETENIDO',model=x['model'],error=str(exc),results_preserved=True)
        finally:write_report(out,pc,ready,completed,errors)
    return 2 if errors else 0


if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:bt.emit('INTERRUMPIDO_RESULTADOS_CONSERVADOS');sys.exit(130)
    except (Exception,bt.RecoveryStop) as exc:bt.emit('ERROR_CRITICO',error=str(exc),results_preserved=True);sys.exit(2)
