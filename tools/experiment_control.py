"""Control humano de rondas y trabajos locales. Ninguna accion entrena ni aprueba QA."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from utils import bootstrap
bootstrap.install(__file__,ROOT)
import argparse
import json
from utils.config_loader import load_config, cfg_path
from utils.experiments import create_round, verify_round, now
from utils.durable import atomic_json
from orchestration.study_queue import StudyQueue, PUBLICATION_PENDING
from orchestration.pc_registry import PCRegistry


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/default.yaml')
    p.add_argument('--set',action='append',default=[])
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('status')
    a=sub.add_parser('mirror-round');a.add_argument('--source-config',type=Path,required=True);a.add_argument('--source-manifest',type=Path,required=True);a.add_argument('--operator',required=True);a.add_argument('--reason',required=True)
    for action in ('new-round','repeat'):
        a=sub.add_parser(action)
        if action=='new-round':a.add_argument('--mode',choices=['supervised','automatic'])
        a.add_argument('--round-id',required=True)
        a.add_argument('--architecture',required=True)
        a.add_argument('--operator',required=True)
        a.add_argument('--reason',required=True)
        a.add_argument('--change',action='append',default=[])
    for action in ('approve','publish','retry','resume'):
        a=sub.add_parser(action); a.add_argument('--job-id',required=True)
        a.add_argument('--operator',required=True);a.add_argument('--evidence',required=True)
    a=sub.add_parser('assemble');a.add_argument('--configs',nargs='+',required=True);a.add_argument('--round-id',required=True);a.add_argument('--operator',required=True);a.add_argument('--reason',required=True);a.add_argument('--accept-non-equivalent',action='store_true')
    a=sub.add_parser('compare');a.add_argument('--configs',nargs='+',required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('retry-external');a.add_argument('--stage',choices=['outer_validation','threshold_tuning'],required=True);a.add_argument('--operator',required=True);a.add_argument('--evidence',required=True)
    a=sub.add_parser('authorize-external');a.add_argument('--operator',required=True);a.add_argument('--evidence',required=True)
    for action in ('transfer-offer','transfer-receive','transfer-confirm','transfer-activate'):
        a=sub.add_parser(action);a.add_argument('--architecture',required=True)
        a.add_argument('--operator',required=True)
        if action=='transfer-offer':
            a.add_argument('--pc-id',required=True);a.add_argument('--target',required=True)
            a.add_argument('--bundle',type=Path,required=True);a.add_argument('--evidence',required=True)
        elif action=='transfer-receive':
            a.add_argument('--pc-id',required=True);a.add_argument('--bundle',type=Path,required=True)
            a.add_argument('--offer',type=Path,required=True);a.add_argument('--receipt',type=Path,required=True)
        else:
            a.add_argument('--receipt',type=Path,required=True);a.add_argument('--confirmation',type=Path,required=True)
    return p


def compare(paths,output):
    import csv
    from utils.architectures import get_arch
    from utils.hashing import sha256_json,sha256_file
    from utils.experiments import identity
    rows=[]
    for path in paths:
        c=load_config(path)
        m=json.loads((cfg_path(c,'paths.artifacts_dir').parent/'manifest.json').read_text(encoding='utf-8'))
        exp=c['experiment'];spec=get_arch(exp['architecture'])
        row=dict(round_id=exp['round_id'],architecture=spec.name,task=spec.task,
                 fingerprint=sha256_json(m['identity']),protocol=m['identity']['protocol'],
                 search_space=m['identity']['search_space'],changes=m['changes'],
                 runtime_compatible=m['identity']==identity(c),
                 configuration_file_intact=m['config_sha256']==sha256_file(path))
        row['changes_json']=json.dumps(m['changes'],ensure_ascii=False,sort_keys=True)
        base=cfg_path(c,'paths.runs_dir')/exp['architecture']
        for stage,filename in [('baseline','summary.json'),('hpo','study_summary.json')]:
            p=base/stage/filename
            if p.is_file():
                data=json.loads(p.read_text());row[stage]=data
                row['objective']=data.get('objective',row.get('objective'))
                for key in ('score_mean','score_sd','train_seconds','vram_peak_mb'):
                    row[stage+'_'+key]=data.get(key)
                if stage=='hpo':row['hpo_best_value']=(data.get('best') or {}).get('value')
        best=cfg_path(c,'paths.best_configs_dir')/(spec.slug+'.json')
        if best.is_file():
            data=json.loads(best.read_text());row['robustness']=data
            for key in ('score_mean','score_sd','train_seconds','vram_peak_mb'):
                row['robustness_'+key]=data.get(key)
        rows.append(row)
    if (output/'comparison.json').exists():raise FileExistsError('Comparacion existente; elija carpeta nueva')
    output.mkdir(parents=True,exist_ok=True)
    equivalent=len({(r['protocol'],r['search_space'],r['task'],r.get('objective')) for r in rows})==1
    atomic_json(output/'comparison.json',dict(rows=rows,methodologically_equivalent=equivalent,
                warning=None if equivalent else 'Protocolos/espacios/objetivos no equivalentes; no interpretar AP y mAP como la misma metrica'))
    fields=['round_id','architecture','task','objective','protocol','search_space','baseline_score_mean',
            'baseline_score_sd','baseline_train_seconds','baseline_vram_peak_mb','hpo_best_value',
            'robustness_score_mean','robustness_score_sd','robustness_train_seconds','robustness_vram_peak_mb',
            'runtime_compatible','configuration_file_intact','changes_json']
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    import matplotlib.pyplot as plt
    valid=[r for r in rows if r.get('baseline_score_mean') is not None]
    groups=sorted({(r['task'],str(r.get('objective') or 'Objetivo no especificado')) for r in valid})
    fig,axes=plt.subplots(1,max(1,len(groups)),figsize=(7*max(1,len(groups)),5),squeeze=False)
    if not valid:axes[0,0].text(.5,.5,'PENDIENTE: sin resultados; no se estiman metricas',ha='center',transform=axes[0,0].transAxes)
    for ax,(task,objective) in zip(axes[0],groups):
        selected=[r for r in valid if r['task']==task and str(r.get('objective') or 'Objetivo no especificado')==objective]
        ax.bar([r['architecture']+'/'+r['round_id'] for r in selected],
               [r['baseline_score_mean'] for r in selected],yerr=[r.get('baseline_score_sd') or 0 for r in selected])
        ax.set_title(task+' / '+objective);ax.tick_params(axis='x',rotation=25)
    fig.suptitle('Baseline: revisar equivalencia y procedencia en comparison.json')
    fig.savefig(output/'comparison.png',bbox_inches='tight');plt.close(fig)
    return dict(rows=len(rows),output=str(output),methodologically_equivalent=equivalent)


def main(argv=None):
    a=parser().parse_args(argv);cfg=load_config(a.config,a.set)
    if a.action=='mirror-round':
        from utils.experiments import mirror_round
        return str(mirror_round(cfg,a.source_config,a.source_manifest,operator=a.operator,reason=a.reason))
    if a.action in ('new-round','repeat'):
        if a.action=='repeat' and a.change:raise ValueError('Repeat conserva configuracion; cambios requieren new-round')
        return str(create_round(cfg,round_id=a.round_id,architecture=a.architecture,operator=a.operator,reason=a.reason,parent=cfg['experiment'].get('round_id'),overrides=a.change,mode=getattr(a,'mode',None)))
    if a.action=='compare':return compare(a.configs,a.output)
    if a.action=='assemble':
        from utils.experiments import assemble_cohort
        return str(assemble_cohort(a.configs,cohort_id=a.round_id,operator=a.operator,reason=a.reason,accept_non_equivalent=a.accept_non_equivalent))
    q=StudyQueue.from_cfg(cfg)
    if a.action=='status':
        from utils.weights import registry
        return dict(blocked_architectures={r['slug']:r['block_reason'] for r in registry(cfg)['models'] if not r['compatible_pretrained']},ownership=PCRegistry(cfg).ownership(),jobs=[j.as_json() for j in q.jobs()],counts=q.counts(),round=cfg['experiment'])
    verify_round(cfg)
    if a.action=='retry-external':
        from utils.experiments import retry_external
        return retry_external(cfg,a.stage,operator=a.operator,evidence=a.evidence)
    if a.action=='authorize-external':
        from utils.experiments import authorize_external
        return authorize_external(cfg,operator=a.operator,evidence=a.evidence)
    if a.action.startswith('transfer-'):
        from orchestration import transfer
        from utils.architectures import get_arch
        slug=get_arch(a.architecture).slug
        if a.action=='transfer-offer':return transfer.offer(cfg,slug,a.pc_id,a.target,a.bundle,operator=a.operator,evidence=a.evidence)
        if a.action=='transfer-receive':return transfer.receive(cfg,a.pc_id,a.bundle,a.offer,a.receipt,operator=a.operator)
        if a.action=='transfer-confirm':return transfer.confirm(cfg,slug,a.receipt,a.confirmation,operator=a.operator)
        return transfer.activate(cfg,slug,a.receipt,a.confirmation,operator=a.operator)
    job=q.get(a.job_id)
    if not job:raise ValueError('Job inexistente; copie el ID completo de status')
    if a.action=='approve':q.approve(job.job_id,operator=a.operator,evidence=a.evidence)
    elif a.action=='publish':
        if job.status!=PUBLICATION_PENDING:raise ValueError('El job no esta PUBLICATION_PENDING')
        from orchestration.exchange import publish_architecture
        publish_architecture(cfg,job.pc_id,job.architecture,cfg_path(cfg,'paths.runs_dir'),stage=job.kind)
        q.computed(job.job_id,published=True)
    elif a.action=='retry':
        if job.status!='FAILED':raise ValueError('Retry solo FAILED')
        q.finish(job.job_id,'PENDING')
    elif a.action=='resume':
        if job.status!='RUNNING':raise ValueError('Resume solo RUNNING interrumpido')
        from utils.guards import gpu_lock_path
        from utils.durable import FileLock
        with FileLock(gpu_lock_path(cfg,job.pc_id).with_suffix('.worker.guard')):
            with FileLock(gpu_lock_path(cfg,job.pc_id).with_suffix('.guard')):
                q.finish(job.job_id,'PENDING')
    atomic_json(cfg_path(cfg,'paths.queue_dir')/'control'/((now().replace(':','-'))+'.json'),dict(action=a.action,job_id=job.job_id,operator=a.operator,evidence=a.evidence,at=now()))
    return q.get(job.job_id).as_json()

if __name__=='__main__':
    print(json.dumps(main(),ensure_ascii=False,indent=2,default=str))
