"""Immutable experiment rounds and audited, explicitly human review decisions.

Local manifests are authority only on the owning machine. ZIP transfers require
an operator handshake; no distributed SQLite exclusion is claimed.
"""
from __future__ import annotations
import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from utils.config_loader import PROJECT_ROOT, cfg_path, load_config
from utils.durable import FileLock, atomic_json, atomic_copy
from utils.hashing import compute_code_hash, scientific_config_hash, sha256_json, sha256_file


def token(value):
    value = str(value)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value):
        raise ValueError('ID invalido: use letras, numeros, guion o guion bajo (1-80)')
    return value


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(cfg):
    """Portable identity, including actual source, manifests and HPO search content."""
    from utils.search_spaces import search_space_hash
    from utils.semantics import semantic_fingerprint
    from utils.weights import registry
    claimed_weights=cfg.get('experiment',{}).get('initial_weights',{})
    art = cfg_path(cfg, 'paths.artifacts_dir')
    seals = {}
    for name in ('split_manifest.sha256',
                 'inner_folds_detection.sha256', 'inner_folds_classification.sha256',
                 'classification_manifest.sha256'):
        p = art / name
        if p.is_file(): seals[name] = sha256_file(p)
    from utils.dataset import recorded_dataset_hash
    from utils.reproducibility import get_git_info
    return dict(git_commit=get_git_info(PROJECT_ROOT).get('commit'),protocol=scientific_config_hash(cfg), semantics=semantic_fingerprint(cfg),
                search_space=search_space_hash(), code=compute_code_hash(PROJECT_ROOT),
                dataset=recorded_dataset_hash(art), seals=seals,
                weight_catalog=sha256_json(registry(cfg)['models']), verified_weight_claims=claimed_weights,
                experiment_id=cfg.get('experiment', {}).get('experiment_id', 'botrytis_v4'),
                round_id=cfg.get('experiment', {}).get('round_id'), initial_ownership=cfg.get('pc_registry',{}).get('first_batch'), execution_mode=cfg.get('experiment',{}).get('mode','supervised'), cohort=cfg.get('experiment',{}).get('cohort'))


def fingerprint(cfg):
    return sha256_json(identity(cfg))


def round_root(cfg):
    return cfg_path(cfg, 'paths.artifacts_dir').parent


def verify_round(cfg, *, architecture=None):
    exp = cfg.get('experiment', {})
    if not exp.get('round_id'):
        raise RuntimeError('Ronda explicita obligatoria: tools/experiment_control.py new-round')
    path = round_root(cfg) / 'manifest.json'
    if not path.is_file(): raise RuntimeError('Manifiesto de ronda ausente')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['identity'] != identity(cfg):
        raise RuntimeError('Ronda incompatible: codigo, datos, split, espacio HPO o protocolo cambiaron; cree new-round')
    if architecture and exp.get('architecture') not in (None, architecture):
        raise RuntimeError('La ronda pertenece a otra arquitectura')
    return manifest


def create_round(cfg, *, round_id, architecture, operator, reason, parent=None, overrides=(), root=None, mode=None, initial_weights=None):
    from utils.architectures import get_arch
    from utils.io import write_yaml
    from utils.config_loader import set_nested, parse_override, validate_fixed_decisions
    if not str(operator).strip() or not str(reason).strip(): raise ValueError('Operador y motivo obligatorios')
    spec = get_arch(architecture)
    experiment_id = token(cfg.get('experiment', {}).get('experiment_id', 'botrytis_v4'))
    rid = token(round_id)
    base = Path(root) if root else cfg_path(cfg, 'paths.artifacts_dir') / 'experiments'
    # A child round retains its original experiment registry, rather than nesting recursively.
    base = Path(cfg.get('experiment', {}).get('registry_root') or base)
    dest = base / experiment_id / rid
    new = copy.deepcopy(cfg)
    new.pop('_meta', None)
    if mode is not None:
        if mode not in ('automatic','supervised'):raise ValueError('Modo invalido')
        new['experiment']['mode']=mode
    new['experiment'].update(experiment_id=experiment_id, round_id=rid, architecture=spec.slug,
                             operator=operator, reason=reason, parent_round=parent,
                             registry_root=str(base.resolve()))
    for item in overrides:
        key, value = parse_override(item)
        if key.startswith(('paths.', 'experiment.', 'manual_qa.', 'operations.')):
            raise ValueError('Cambios de rutas/identidad/gates no son overrides cientificos de una ronda')
        set_nested(new, key, value)
    validate_fixed_decisions(new)
    from utils.weights import registry,weight_path
    claims={}
    if new.get('initialization',{}).get('mode')=='pretrained':
        if initial_weights is not None:
            claims=dict(initial_weights)
        else:
            row=next(r for r in registry(new)['models'] if r['slug']==spec.slug)
            side=weight_path(row,new).with_suffix(Path(row['filename']).suffix+'.provenance.json')
            if side.is_file():claims[spec.slug]=json.loads(side.read_text(encoding='utf-8')).get('sha256')
    new['experiment']['initial_weights']=claims
    old_art = cfg_path(cfg, 'paths.artifacts_dir')
    with FileLock(base / (experiment_id + '.guard')):
        if dest.exists(): raise FileExistsError('Ronda inmutable ya existente: ' + str(dest))
        dest.mkdir(parents=True)
        for key, rel in dict(artifacts_dir='artifacts', runs_dir='runs', queue_dir='artifacts/queue',
                             optuna_dir='artifacts/optuna', best_configs_dir='artifacts/best_configs',
                             environments_dir='artifacts/environments', reports_dir='reports',
                             figures_dir='reports/figures', tables_dir='reports/tables', results_dir='results').items():
            new['paths'][key] = str((dest / rel).resolve())
        new['test']['freeze_manifest'] = str((dest/'artifacts/freeze_manifest.json').resolve())
        # Signed preparation only; never copy mutable training DBs, checkpoints or previous decisions.
        names = ('dataset_inventory.csv', 'dataset_inventory.sha256', 'dataset_hash.json',
                 'dataset_validation.json', 'dataset_validation_summary.json', 'labels_parsed.csv',
                 'fruit_labels.csv', 'dataset_hash.sha256', 'dataset_metadata.json',
                 'split_manifest.csv', 'split_manifest.sha256', 'split_metadata.json', 'test_lock.json',
                 'inner_folds_detection.csv', 'inner_folds_detection.sha256',
                 'inner_folds_classification.csv', 'inner_folds_classification.sha256', 'inner_folds_metadata.json',
                 'image_manifest.csv', 'classification_manifest.csv', 'classification_manifest.sha256',
                 'classification_stats.json', 'fleet_mapping.json', 'machine_binding.json', 'near_duplicate_review.json')
        for name in names:
            src = old_art/name
            if src.is_file(): atomic_copy(src, dest/'artifacts'/name)
        for src in (old_art/'environments').glob('*.json'):
            atomic_copy(src, dest/'artifacts/environments'/src.name)
        # Preserve all other small signed preparation records without arbitrary globbing.
        for src in old_art.glob('*'):
            if src.is_file() and src.name.startswith(('dataset_', 'pairing_', 'annotation_', 'labels_')) and src.suffix in ('.json','.csv','.sha256'):
                atomic_copy(src, dest/'artifacts'/src.name)
        reference = cfg_path(cfg, 'compute.reference_environment')
        if reference.is_file():
            atomic_copy(reference, dest/'artifacts/environments/environment_reference.json')
        new['compute']['reference_environment'] = str((dest/'artifacts/environments/environment_reference.json').resolve())
        path = dest/'config.yaml'
        write_yaml(path, new)
        effective = load_config(path)
        old_science = {k:v for k,v in cfg.items() if k not in ('_meta','paths','experiment','compute','operations','manual_qa','test')}
        new_science = {k:v for k,v in new.items() if k in old_science}
        differences = {k:dict(before=old_science.get(k), after=v) for k,v in new_science.items() if v != old_science.get(k)}
        atomic_json(dest/'manifest.json', dict(schema=1, created_at=now(), operator=operator,
                    reason=reason, architecture=spec.slug, parent_round=parent, changes=differences,
                    identity=identity(effective), config_sha256=sha256_file(path)))
    return path


def external_ledger(cfg):
    exp=cfg['experiment']
    return Path(exp['registry_root'])/exp['experiment_id']/'external_validation.json'


def authorize_external(cfg, *, operator, evidence):
    verify_round(cfg)
    if not operator.strip() or not evidence.strip():raise ValueError('Operador y evidencia obligatorios')
    slug=cfg['experiment']['architecture']
    if not slug:raise ValueError('Autorizar una arquitectura/ronda concreta')
    path=external_ledger(cfg)
    with FileLock(path.with_suffix('.guard')):
        ledger=json.loads(path.read_text()) if path.exists() else {}
        if slug in ledger:raise RuntimeError('Validacion externa ya predeclarada para esta arquitectura; no se reutiliza para explorar rondas')
        row=dict(round_id=cfg['experiment']['round_id'],fingerprint=fingerprint(cfg),operator=operator,evidence=evidence,at=now(),uses={})
        ledger[slug]=row;atomic_json(path,ledger)
    return row


def use_external(cfg,stage,checkpoint):
    """Consume a predeclared validation/calibration operation once, before inference."""
    verify_round(cfg)
    path=external_ledger(cfg);slug=cfg['experiment']['architecture']
    with FileLock(path.with_suffix('.guard')):
        ledger=json.loads(path.read_text()) if path.exists() else {}
        row=ledger.get(slug,{})
        if row.get('fingerprint')!=fingerprint(cfg):raise RuntimeError('Validacion externa no autorizada para esta ronda; iterar solo en folds internos')
        if stage in row['uses']:raise RuntimeError('Operacion externa ya consumida; no repetir optimizacion exploratoria')
        row['uses'][stage]=dict(at=now(),checkpoint_sha256=sha256_file(checkpoint),state='STARTED')
        atomic_json(path,ledger)


def stage_job(ctx,args):
    """Validate/claim a direct script exactly like the worker; reviews cannot be bypassed."""
    from orchestration.study_queue import StudyQueue, RUNNING
    from orchestration.pc_registry import PCRegistry
    from utils.architectures import get_arch
    from utils.guards import is_official
    if not is_official(ctx.cfg):return None
    spec=get_arch(args.architecture)
    verify_round(ctx.cfg,architecture=spec.slug)
    if PCRegistry(ctx.cfg).owner(spec.slug)!=ctx.pc_id:raise RuntimeError('Otra PC es propietaria o transferencia pendiente')
    q=StudyQueue.from_cfg(ctx.cfg)
    number=int(ctx.script_name[:2])
    kind={6:'baseline',7:'hpo',8:'aggregate' if getattr(args,'aggregate_only',False) else 'repeat',9:'outer_train'}.get(number)
    if not kind:return None
    if number==8 and not getattr(args,'aggregate_only',False) and not getattr(args,'config_id',None):
        raise RuntimeError('Modo supervisado: ejecutar robustez por jobs/worker, luego aprobar aggregate')
    if kind=='baseline':q.plan(ctx.cfg,[spec])
    candidates=[j for j in q.jobs() if j.kind==kind and j.architecture==spec.slug]
    if kind=='repeat':candidates=[j for j in candidates if j.payload.get('config_id')==args.config_id and j.payload.get('seed')==args.seed]
    if len(candidates)!=1:raise RuntimeError('Job de etapa ausente/ambiguo; planifique y apruebe dependencias')
    job=candidates[0]
    import os
    worker_job=os.environ.get('BOTRYTIS_WORKER_JOB_ID')
    if worker_job:
        if worker_job!=job.job_id or job.status!=RUNNING:raise RuntimeError('Job del worker incompatible')
        return None
    if job.status=='COMPLETE':return None
    from utils.guards import gpu_lock_path
    with FileLock(gpu_lock_path(ctx.cfg,ctx.pc_id).with_suffix('.worker.guard')):
        if PCRegistry(ctx.cfg).owner(spec.slug)!=ctx.pc_id:raise RuntimeError('Ownership cambio durante claim')
        claimed=q.claim_next(ctx.pc_id,kinds=[kind],slugs=[spec.slug],ready_check=lambda j:j.job_id==job.job_id)
    if not claimed:raise RuntimeError('Etapa no aprobada, publicada, bloqueada o RUNNING')
    ctx._direct_stage_job=job.job_id
    return job.job_id


def finish_direct(ctx, *, success):
    job_id=getattr(ctx,'_direct_stage_job',None)
    if not job_id:return
    from orchestration.study_queue import StudyQueue
    from orchestration.exchange import publish_architecture
    from utils.architectures import get_arch,trials_for
    q=StudyQueue.from_cfg(ctx.cfg);job=q.get(job_id)
    if not success:
        q.finish(job_id,'FAILED',error='Ver execution.log del script directo');return
    partial=False
    if job.kind=='hpo':
        from utils.optuna_storage import budget_used,load_study
        if budget_used(load_study(get_arch(job.architecture),ctx.cfg))<trials_for(get_arch(job.architecture),ctx.cfg):
            q.mark_partial_block(job_id);partial=True
    q.computed(job_id,published=False)
    publish_architecture(ctx.cfg,ctx.pc_id,job.architecture,ctx.runs_base,stage=job.kind,complete=not partial)
    q.computed(job_id,published=True)
    if job.kind=='hpo' and not partial:
        from orchestration.worker import top_config_ids
        q.expand_after_hpo(ctx.cfg,get_arch(job.architecture),top_config_ids(ctx.cfg,job.architecture))


def assemble_cohort(paths, *, cohort_id, operator, reason, accept_non_equivalent=False):
    """Explicitly select one immutable validated round per active architecture.

    Copies local verified artifacts into a new cohort; never evaluates, retunes,
    merges an Optuna study or modifies a source round.
    """
    from utils.architectures import archs,get_arch
    from utils.hpo_results import final_checkpoint,stage_json_path,load_best_config
    from utils.evidence import split_dir
    import shutil
    sources=[load_config(path) for path in paths]
    manifests=[verify_round(c) for c in sources]
    slugs=[c['experiment']['architecture'] for c in sources]
    if set(slugs)!={a.slug for a in archs()} or len(slugs)!=len(set(slugs)):
        raise ValueError('Cohorte requiere exactamente una ronda por cada una de las siete arquitecturas activas')
    shared=[{k:m['identity'][k] for k in ('code','git_commit','dataset','seals','semantics')} for m in manifests]
    if any(row!=shared[0] for row in shared[1:]):raise ValueError('Cohorte con codigo/dataset/split/folds/semantica diferentes')
    equivalent=len({(m['identity']['protocol'],m['identity']['search_space']) for m in manifests})==1
    if not equivalent and not accept_non_equivalent:raise ValueError('Rondas no equivalentes: revisar comparacion y declarar --accept-non-equivalent')
    # Validate all candidates before creating any cohort output.
    records={}
    for c,m in zip(sources,manifests):
        spec=get_arch(c['experiment']['architecture'])
        ck=final_checkpoint(c,spec);meta=ck.with_name('final_meta.json')
        load_best_config(c,spec)
        if not ck.is_file() or not meta.is_file() or json.loads(meta.read_text()).get('sha256')!=sha256_file(ck):raise ValueError('Checkpoint candidato incompleto/alterado: '+spec.slug)
        val=stage_json_path(c,'outer_validation',spec);threshold=stage_json_path(c,'inference_thresholds',spec)
        if not val.is_file() or not threshold.is_file():raise ValueError('Falta validation/thresholds: '+spec.slug)
        if json.loads(val.read_text()).get('checkpoint_sha256')!=sha256_file(ck):raise ValueError('Validacion de otro checkpoint')
        records[spec.slug]=dict(round_id=c['experiment']['round_id'],fingerprint=fingerprint(c),checkpoint_sha256=sha256_file(ck),best_sha256=sha256_file(cfg_path(c,'paths.best_configs_dir')/(spec.slug+'.json')),source_manifest_sha256=sha256_file(round_root(c)/'manifest.json'))
    first=sources[0]
    path=create_round(first,round_id=cohort_id,architecture=slugs[0],operator=operator,reason=reason,parent=None)
    cohort=load_config(path);cohort.pop('_meta',None)
    cohort['experiment'].update(architecture=None,cohort=records,cohort_equivalent=equivalent,initial_weights={k:v for c in sources for k,v in c['experiment'].get('initial_weights',{}).items()})
    from utils.io import write_yaml
    write_yaml(path,cohort)
    for c in sources:
        slug=c['experiment']['architecture'];source=cfg_path(c,'paths.artifacts_dir');dest=cfg_path(cohort,'paths.artifacts_dir')
        for directory in ('best_configs','outer_validation','inference_thresholds'):
            for p in (source/directory).glob(slug+'*'):
                if p.is_file():atomic_copy(p,dest/directory/p.name)
        src=source/'checkpoints'/slug
        shutil.copytree(src,dest/'checkpoints'/slug)
        ev=split_dir(c,get_arch(slug),'validation')
        if ev.is_dir():shutil.copytree(ev,split_dir(cohort,get_arch(slug),'validation'))
    atomic_json(round_root(cohort)/'manifest.json',dict(schema=1,operator=operator,reason=reason,created_at=now(),changes={},identity=identity(cohort),config_sha256=sha256_file(path),cohort=records,methodologically_equivalent=equivalent))
    return path


def finish_external(cfg,stage):
    path=external_ledger(cfg);slug=cfg['experiment']['architecture']
    with FileLock(path.with_suffix('.guard')):
        ledger=json.loads(path.read_text());row=ledger[slug]['uses'][stage]
        row.update(state='COMPLETE',finished_at=now());atomic_json(path,ledger)


def retry_external(cfg,stage,*,operator,evidence):
    """Only an interrupted operation without published results may be retried."""
    if not operator.strip() or not evidence.strip():raise ValueError('Operador y evidencia obligatorios')
    slug=cfg['experiment']['architecture'];path=external_ledger(cfg)
    output_stage='outer_validation' if stage=='outer_validation' else 'inference_thresholds'
    if (cfg_path(cfg,'paths.artifacts_dir')/output_stage/(slug+'.json')).exists():
        raise RuntimeError('Ya hay resultado externo; no se autoriza una nueva optimizacion')
    with FileLock(path.with_suffix('.guard')):
        ledger=json.loads(path.read_text());row=ledger[slug]
        if row['fingerprint']!=fingerprint(cfg) or row['uses'].get(stage,{}).get('state')!='STARTED':raise RuntimeError('No hay operacion interrumpida compatible')
        previous=row['uses'].pop(stage)
        row.setdefault('recovery_audit',[]).append(dict(stage=stage,previous=previous,operator=operator,evidence=evidence,at=now()))
        atomic_json(path,ledger)
    return row


def mirror_round(local_cfg,source_config,source_manifest,*,operator,reason):
    """Install portable round metadata against locally validated paths and binding."""
    source=load_config(source_config)
    manifest=json.loads(Path(source_manifest).read_text(encoding='utf-8'))
    if manifest.get('config_sha256')!=sha256_file(source_config):raise ValueError('Config fuente alterada')
    merged=copy.deepcopy(source)
    for key in ('paths','compute','operations','manual_qa'):
        if key in local_cfg:merged[key]=copy.deepcopy(local_cfg[key])
    merged['experiment']['registry_root']=str(cfg_path(local_cfg,'paths.artifacts_dir')/'experiments')
    # Keep the declared weight hashes without requiring copies of pretrained files for import-only mirrors.
    if identity(merged)!=manifest['identity']:raise ValueError('Espejo incompatible con codigo, datos, split/folds, protocolo o pesos declarados')
    return create_round(merged,round_id=source['experiment']['round_id'],architecture=source['experiment']['architecture'],operator=operator,reason=reason,parent=source['experiment'].get('parent_round'),mode=source['experiment']['mode'],initial_weights=source['experiment'].get('initial_weights',{}))
