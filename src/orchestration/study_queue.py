"""Cola de trabajos de las 3 PCs (SQLite) con estados PENDING / RUNNING / COMPLETE / FAILED / PRUNED.

Modelo (sec. 17.3, Estrategia_3PCs sec. 10):

* Un *job* = una etapa de UNA arquitectura (``baseline``, ``hpo``, ``repeat``, ``aggregate``,
  ``outer_train``, ``outer_eval``, ``thresholds``). Un estudio HPO es un solo job: sus trials son
  SECUENCIALES (``optuna_n_jobs = 1``); el paralelismo fisico ocurre ENTRE arquitecturas y entre seeds.
* ``claim_next`` es atomico (``BEGIN IMMEDIATE``): nunca se inicia un job RUNNING existente; una PC no
  toma un segundo job si ya tiene uno RUNNING; no hay mas de ``max_gpu_jobs_global`` RUNNING a la vez.
* Cada trial queda en la tabla ``trials`` (estado COMPLETE/PRUNED/FAILED/RUNNING + hashes + PC).
* Sin servidor central (Estrategia_3PCs): el estado se sincroniza por ZIP con ``export_state`` /
  ``merge_state`` (gana el estado de mayor rango: COMPLETE > PRUNED > FAILED > RUNNING > PENDING).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.architectures import ArchSpec, archs  # noqa: E402
from utils.config_loader import cfg_path  # noqa: E402

PENDING, RUNNING, COMPLETE, FAILED, PRUNED = "PENDING", "RUNNING", "COMPLETE", "FAILED", "PRUNED"
PENDING_REVIEW, PUBLICATION_PENDING, BLOCKED = "PENDING_REVIEW", "PUBLICATION_PENDING", "BLOCKED"
STATUSES = (PENDING, RUNNING, COMPLETE, FAILED, PRUNED, PENDING_REVIEW, PUBLICATION_PENDING, BLOCKED)
RANK = {PENDING_REVIEW: 2, PUBLICATION_PENDING: 2, BLOCKED: 1, COMPLETE: 5, PRUNED: 4, FAILED: 3, RUNNING: 2, PENDING: 1}
KINDS = ("baseline", "hpo", "repeat", "aggregate", "outer_train", "outer_eval", "thresholds")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, architecture TEXT NOT NULL, task TEXT NOT NULL, study_name TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL DEFAULT 'PENDING', pc_id TEXT, depends_on TEXT NOT NULL DEFAULT '[]',
    priority INTEGER NOT NULL DEFAULT 50, created_at TEXT, started_at TEXT, finished_at TEXT, heartbeat_at TEXT, updated_at TEXT,
    git_commit TEXT, code_hash TEXT, dataset_hash TEXT, split_hash TEXT, log_path TEXT, error TEXT, attempt INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS trials (
    study_name TEXT NOT NULL, trial_number INTEGER NOT NULL, job_id TEXT, status TEXT NOT NULL, value REAL, pc_id TEXT,
    started_at TEXT, finished_at TEXT, run_dir TEXT, git_commit TEXT, dataset_hash TEXT, split_hash TEXT, updated_at TEXT,
    PRIMARY KEY (study_name, trial_number)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    """Un trabajo de la cola."""

    job_id: str
    kind: str
    architecture: str  # slug
    task: str
    study_name: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = PENDING
    pc_id: str | None = None
    depends_on: list[str] = field(default_factory=list)
    priority: int = 50
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    git_commit: str | None = None
    code_hash: str | None = None
    dataset_hash: str | None = None
    split_hash: str | None = None
    log_path: str | None = None
    error: str | None = None
    attempt: int = 0

    def as_json(self) -> dict[str, Any]:
        """Representacion del job con los campos de la sec. 17.3."""
        d = asdict(self)
        d["trial_number"] = None
        return d


def job_id_for(kind: str, slug: str, **extra: Any) -> str:
    """Identificador estable: ``hpo:yolov8n`` o ``repeat:yolov8n:cfg_ab12cd34:s42``."""
    if kind == "repeat":
        return f"repeat:{slug}:{extra['config_id']}:s{extra['seed']}"
    return f"{kind}:{slug}"


class StudyQueue:
    """Cola persistente en SQLite (un fichero por PC; se fusiona por ZIP)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)
            c.execute('CREATE TABLE IF NOT EXISTS context (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY, job_id TEXT, operator TEXT, evidence TEXT, decision TEXT, at TEXT)')
            row = c.execute("SELECT value FROM context WHERE key='config'").fetchone()
            self.cfg = json.loads(row[0]) if row else None

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> "StudyQueue":
        """Cola en ``paths.queue_dir/queue.sqlite``."""
        queue = cls(cfg_path(cfg, "paths.queue_dir") / "queue.sqlite")
        queue.bind(cfg)
        return queue

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------ altas
    def add_job(self, job: Job) -> bool:
        """Inserta el job si no existe (idempotente). Devuelve ``True`` si lo creo."""
        now = _now()
        exp = (self.cfg or {}).get('experiment',{})
        if exp.get('round_id'):
            prefix = exp['experiment_id'] + '/' + exp['round_id'] + '/'
            job.job_id = job.job_id if job.job_id.startswith(prefix) else prefix + job.job_id
            job.depends_on = [d if d.startswith(prefix) else prefix+d for d in job.depends_on]
            from utils.experiments import fingerprint
            job.payload.update(round_id=exp['round_id'], experiment_id=exp['experiment_id'], fingerprint=fingerprint(self.cfg))
        with self._conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO jobs (job_id, kind, architecture, task, study_name, payload, status, depends_on, priority, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (job.job_id, job.kind, job.architecture, job.task, job.study_name, json.dumps(job.payload, sort_keys=True), PENDING,
                 json.dumps(job.depends_on), job.priority, now, now),
            )
            return cur.rowcount > 0

    def plan(self, cfg: Mapping[str, Any], selected: Sequence[ArchSpec] | None = None) -> int:
        """Crea ``baseline`` -> ``hpo`` de cada arquitectura (las etapas posteriores se crean tras el HPO)."""
        self.bind(cfg)
        from utils.optuna_storage import study_name
        n = 0
        for spec in archs() if selected is None else selected:
            base = job_id_for("baseline", spec.slug)
            n += self.add_job(Job(base, "baseline", spec.slug, spec.task, study_name(spec,cfg), priority=10))
            n += self.add_job(Job(job_id_for("hpo", spec.slug), "hpo", spec.slug, spec.task, study_name(spec,cfg), depends_on=[base], priority=20))
        return n

    def expand_after_hpo(self, cfg: Mapping[str, Any], spec: ArchSpec, config_ids: Sequence[str]) -> int:
        """Crea repeat (top-k x seeds), aggregate, outer_train, outer_eval y thresholds de ``spec``."""
        from utils.optuna_storage import study_name
        study = study_name(spec,cfg)
        seeds = [int(s) for s in cfg["robustness"]["seeds"]]
        n = 0
        repeat_ids: list[str] = []
        for cid in config_ids:
            for s in seeds:
                jid = job_id_for("repeat", spec.slug, config_id=cid, seed=s)
                repeat_ids.append(jid)
                n += self.add_job(Job(jid, "repeat", spec.slug, spec.task, study, payload={"config_id": cid, "seed": s},
                                      depends_on=[job_id_for("hpo", spec.slug)], priority=30))
        chain = [("aggregate", repeat_ids, 40), ("outer_train", [job_id_for("aggregate", spec.slug)], 50),
                 ("outer_eval", [job_id_for("outer_train", spec.slug)], 60), ("thresholds", [job_id_for("outer_eval", spec.slug)], 70)]
        for kind, deps, prio in chain:
            n += self.add_job(Job(job_id_for(kind, spec.slug), kind, spec.slug, spec.task, study, depends_on=deps, priority=prio))
        return n

    def bind(self, cfg):
        from utils.experiments import fingerprint
        stamp = fingerprint(cfg)
        with self._conn() as c:
            old = c.execute("SELECT value FROM context WHERE key='fingerprint'").fetchone()
            if old and old[0] != stamp:
                raise RuntimeError('Cola de ronda incompatible; use new-round')
            c.execute("INSERT OR REPLACE INTO context VALUES ('fingerprint',?)", (stamp,))
            c.execute("INSERT OR REPLACE INTO context VALUES ('config',?)", (json.dumps(cfg),))
        self.cfg = dict(cfg)

    def approve(self, job_id, *, operator, evidence):
        if not operator.strip() or not evidence.strip(): raise ValueError('Operador y evidencia obligatorios')
        with self._conn() as c:
            c.execute('BEGIN IMMEDIATE')
            row = c.execute('SELECT status,payload FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            if not row or row[0] != PENDING_REVIEW: raise ValueError('Solo se aprueba PENDING_REVIEW')
            c.execute('INSERT INTO reviews(job_id,operator,evidence,decision,at) VALUES (?,?,?,?,?)',
                      (job_id,operator,evidence,'APPROVED',_now()))
            payload=json.loads(row[1]);partial=payload.pop('partial_budget',False)
            c.execute('UPDATE jobs SET status=?, payload=?, updated_at=? WHERE job_id=?',
                      (PENDING if partial else COMPLETE,json.dumps(payload),_now(),job_id))
            c.execute('COMMIT')

    def mark_partial_block(self, job_id):
        job=self.get(job_id)
        if job.kind!='hpo':raise ValueError('Solo HPO tiene bloques')
        payload=dict(job.payload,partial_budget=True)
        with self._conn() as c:
            c.execute('UPDATE jobs SET payload=? WHERE job_id=?',(json.dumps(payload),job_id))

    def computed(self, job_id, *, published=False):
        job = self.get(job_id)
        supervised = (self.cfg or {}).get('experiment',{}).get('mode','supervised') == 'supervised'
        state = PUBLICATION_PENDING if not published else (
            PENDING_REVIEW if job.payload.get('partial_budget') or (supervised and job.kind in ('baseline','hpo','aggregate')) else COMPLETE)
        self.finish(job_id, state)
        return state

    # --------------------------------------------------------------- consultas
    @staticmethod
    def _row_to_job(r: sqlite3.Row) -> Job:
        return Job(
            job_id=r["job_id"], kind=r["kind"], architecture=r["architecture"], task=r["task"], study_name=r["study_name"],
            payload=json.loads(r["payload"] or "{}"), status=r["status"], pc_id=r["pc_id"], depends_on=json.loads(r["depends_on"] or "[]"),
            priority=r["priority"], created_at=r["created_at"], started_at=r["started_at"], finished_at=r["finished_at"], git_commit=r["git_commit"],
            code_hash=r["code_hash"], dataset_hash=r["dataset_hash"], split_hash=r["split_hash"], log_path=r["log_path"], error=r["error"], attempt=r["attempt"],
        )

    def get(self, job_id: str) -> Job | None:
        """Job por id (``None`` si no existe)."""
        with self._conn() as c:
            r = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._row_to_job(r) if r else None

    def jobs(self, status: str | None = None) -> list[Job]:
        """Todos los jobs (opcionalmente filtrados por estado)."""
        with self._conn() as c:
            rows = c.execute("SELECT * FROM jobs" + (" WHERE status=?" if status else "") + " ORDER BY priority, created_at, job_id",
                             (status,) if status else ()).fetchall()
        return [self._row_to_job(r) for r in rows]

    def table(self) -> pd.DataFrame:
        """Tabla de jobs para informes."""
        with self._conn() as c:
            return pd.read_sql_query("SELECT job_id, kind, architecture, status, pc_id, attempt, started_at, finished_at, error FROM jobs ORDER BY priority, job_id", c)

    def counts(self) -> dict[str, int]:
        """Numero de jobs por estado."""
        with self._conn() as c:
            rows = c.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status").fetchall()
        return {s: 0 for s in STATUSES} | {r["status"]: r["n"] for r in rows}

    def unfinished(self) -> bool:
        """``True`` si quedan jobs PENDING o RUNNING."""
        c = self.counts()
        return bool(c[PENDING] or c[RUNNING])

    # ---------------------------------------------------------------- reclamo
    def claim_next(
        self, pc_id: str, *, preferred_slugs: Sequence[str] = (), kinds: Sequence[str] | None = None, slugs: Sequence[str] | None = None,
        max_global_running: int = 3, provenance: Mapping[str, Any] | None = None, ready_check: Callable[["Job"], bool] | None = None,
    ) -> Job | None:
        """Reclama atomicamente el siguiente job listo (dependencias COMPLETE) para ``pc_id``.

        Reglas: ``pc_id`` sin otro job RUNNING (1 GPU por PC); RUNNING global < ``max_global_running``; se prefieren
        las arquitecturas de la primera tanda de la PC y las que esta PC ya trabajo (continuidad de la cadena: los ficheros
        intermedios estan en su disco); nunca se devuelve un job que ya este RUNNING.

        Args:
            ready_check: Predicado opcional ``Job -> bool`` que verifica que los INSUMOS LOCALES del job existen en esta PC
                (p. ej. los resultados de robustez antes de agregarlos). Sin servidor central, los ficheros viajan en ZIPs: un job
                cuyas entradas no estan aqui se deja PENDING para la PC que las tenga.
        """
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                running = c.execute("SELECT pc_id FROM jobs WHERE status='RUNNING'").fetchall()
                if any(r["pc_id"] == pc_id for r in running) or len(running) >= max_global_running:
                    c.execute("ROLLBACK")
                    return None
                status = {r["job_id"]: r["status"] for r in c.execute("SELECT job_id, status FROM jobs").fetchall()}
                affinity = {row["architecture"] for row in c.execute("SELECT DISTINCT architecture FROM jobs WHERE pc_id=? AND status=?", (pc_id, COMPLETE)).fetchall()}
                cands = []
                for r in c.execute("SELECT * FROM jobs WHERE status='PENDING'").fetchall():
                    deps = json.loads(r["depends_on"] or "[]")
                    if kinds and r["kind"] not in kinds:
                        continue
                    if self.cfg:
                        from orchestration.pc_registry import PCRegistry
                        if PCRegistry(self.cfg).owner(r['architecture']) != pc_id:
                            continue
                    if slugs is not None and r["architecture"] not in slugs:
                        continue
                    if all(status.get(d) == COMPLETE for d in deps) and (ready_check is None or ready_check(self._row_to_job(r))):
                        cands.append(r)
                if not cands:
                    c.execute("ROLLBACK")
                    return None
                cands.sort(key=lambda r: (0 if (r["architecture"] in preferred_slugs or r["architecture"] in affinity) else 1, r["priority"], r["created_at"] or "", r["job_id"]))
                pick = cands[0]
                prov = dict(provenance or {})
                now = _now()
                c.execute(
                    "UPDATE jobs SET status='RUNNING', pc_id=?, started_at=?, heartbeat_at=?, updated_at=?, attempt=attempt+1, error=NULL, finished_at=NULL, "
                    "git_commit=?, code_hash=?, dataset_hash=?, split_hash=? WHERE job_id=? AND status='PENDING'",
                    (pc_id, now, now, now, prov.get("git_commit"), prov.get("code_hash"), prov.get("dataset_hash"), prov.get("split_hash"), pick["job_id"]),
                )
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise
        return self.get(pick["job_id"])

    # ------------------------------------------------------------ transiciones
    def heartbeat(self, job_id: str) -> None:
        """Latido de un job RUNNING (detecta PCs caidas)."""
        with self._conn() as c:
            c.execute("UPDATE jobs SET heartbeat_at=?, updated_at=? WHERE job_id=? AND status='RUNNING'", (_now(), _now(), job_id))

    def finish(self, job_id: str, status: str, *, error: str | None = None, log_path: str | None = None) -> None:
        """Cierra un job como COMPLETE o FAILED."""
        if status not in (PENDING, COMPLETE, FAILED, PENDING_REVIEW, PUBLICATION_PENDING, BLOCKED):
            raise ValueError("finish solo admite COMPLETE o FAILED")
        with self._conn() as c:
            c.execute("UPDATE jobs SET status=?, finished_at=?, updated_at=?, error=?, log_path=COALESCE(?, log_path) WHERE job_id=?",
                      (status, _now(), _now(), error, log_path, job_id))

    def retry_failed(self) -> int:
        """FAILED -> PENDING (conserva ``attempt``)."""
        with self._conn() as c:
            cur = c.execute("UPDATE jobs SET status='PENDING', pc_id=NULL, updated_at=? WHERE status='FAILED' AND attempt < 3", (_now(),))
            return cur.rowcount

    def recover_stale(self, *, minutes: float, force: bool = False, pc_id: str | None = None) -> list[str]:
        """RUNNING huerfanos (sin latido desde hace ``minutes``, o todos los de ``pc_id`` con ``force``) -> PENDING."""
        raise RuntimeError('Recuperacion automatica deshabilitada: use experiment_control resume con operador/evidencia y locks locales libres')

    def record_trial(self, study_name: str, number: int, status: str, *, job_id: str | None = None, value: float | None = None, pc_id: str | None = None,
                     run_dir: str | None = None, provenance: Mapping[str, Any] | None = None, started_at: str | None = None, finished: bool = False) -> None:
        """Inserta/actualiza el estado de un trial del estudio."""
        if status not in STATUSES:
            raise ValueError(status)
        prov = dict(provenance or {})
        now = _now()
        with self._conn() as c:
            c.execute(
                "INSERT INTO trials (study_name, trial_number, job_id, status, value, pc_id, started_at, finished_at, run_dir, git_commit, dataset_hash, split_hash, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(study_name, trial_number) DO UPDATE SET status=excluded.status, "
                "value=COALESCE(excluded.value, trials.value), finished_at=COALESCE(excluded.finished_at, trials.finished_at), "
                "run_dir=COALESCE(excluded.run_dir, trials.run_dir), updated_at=excluded.updated_at",
                (study_name, number, job_id, status, value, pc_id, started_at or now, now if finished else None, run_dir, prov.get("git_commit"),
                 prov.get("dataset_hash"), prov.get("split_hash"), now),
            )

    def trials_table(self) -> pd.DataFrame:
        """Tabla de trials registrados."""
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM trials ORDER BY study_name, trial_number", c)

    # -------------------------------------------------------------- sincronia
    def export_state(self, path: str | Path) -> Path:
        """Vuelca jobs y trials a JSON (para el ZIP de resultados)."""
        with self._conn() as c:
            jobs = [dict(r) for r in c.execute("SELECT * FROM jobs").fetchall()]
            trials = [dict(r) for r in c.execute("SELECT * FROM trials").fetchall()]
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"exported_at": _now(), "jobs": jobs, "trials": trials}, indent=2), encoding="utf-8")
        return p

    def merge_state(self, path: str | Path) -> dict[str, Any]:
        """Fusiona el estado exportado por otra PC. Gana el de mayor rango (COMPLETE > PRUNED > FAILED > RUNNING > PENDING).

        Returns:
            ``{'inserted', 'updated', 'conflicts'}``; un conflicto es el mismo job RUNNING en dos PCs distintas
            (se conserva el que empezo antes y se avisa).
        """
        from utils.durable import atomic_json
        from utils.hashing import sha256_json
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        atomic_json(self.path.parent/'foreign_evidence'/(sha256_json(data)+'.json'), data)
        conflicts = []
        for remote in data.get('jobs',[]):
            local = self.get(remote['job_id'])
            if local and (local.architecture != remote.get('architecture') or local.payload != json.loads(remote.get('payload','{}')) or (local.status == RUNNING and remote.get('status') == RUNNING and local.pc_id != remote.get('pc_id'))):
                conflicts.append({'job_id':local.job_id,'reason':'incompatible identity or concurrent foreign authority'})
        return dict(inserted=0,updated=0,conflicts=conflicts,authority='foreign evidence only')

    def transfer_snapshot(self,slug):
        with self._conn() as c:
            c.execute('BEGIN')
            jobs=[dict(r) for r in c.execute('SELECT * FROM jobs WHERE architecture=?',(slug,))]
            studies={j['study_name'] for j in jobs}
            trials=[dict(r) for r in c.execute('SELECT * FROM trials') if r['study_name'] in studies]
            ids={j['job_id'] for j in jobs}
            reviews=[dict(r) for r in c.execute('SELECT * FROM reviews') if r['job_id'] in ids]
            c.execute('COMMIT')
        return dict(jobs=jobs,trials=trials,reviews=reviews)

    def adopt_transfer(self,slug,snapshot):
        """Only called for an explicit, verified ownership offer; never ordinary ZIP imports."""
        incoming=snapshot['jobs']
        if any(j['architecture']!=slug or j['status']==RUNNING for j in incoming):raise ValueError('Transferencia con trabajo RUNNING/otra arquitectura')
        with self._conn() as c:
            c.execute('BEGIN IMMEDIATE')
            local=c.execute('SELECT * FROM jobs WHERE architecture=?',(slug,)).fetchall()
            if sorted([dict(j) for j in local],key=lambda j:j['job_id'])==sorted(incoming,key=lambda j:j['job_id']):
                c.execute('COMMIT');return
            if any(j['status']!=PENDING or j['attempt'] for j in local):raise ValueError('Cola receptora tiene trabajo local; no se sobrescribe')
            for row in incoming:
                if self.cfg and self.cfg.get('experiment',{}).get('round_id'):
                    prefix=self.cfg['experiment']['experiment_id']+'/'+self.cfg['experiment']['round_id']+'/'
                    if not row['job_id'].startswith(prefix):raise ValueError('Job de otra ronda')
                columns=[r[1] for r in c.execute('PRAGMA table_info(jobs)')]
                if set(row)!=set(columns):raise ValueError('Schema de job incompatible')
                c.execute('DELETE FROM jobs WHERE job_id=? AND architecture=?',(row['job_id'],slug))
                c.execute('INSERT INTO jobs ('+','.join(columns)+') VALUES ('+','.join('?' for _ in columns)+')',[row[k] for k in columns])
            studies={j['study_name'] for j in incoming};ids={j['job_id'] for j in incoming}
            for row in snapshot.get('trials',[]):
                if row['study_name'] not in studies:raise ValueError('Trial de otro estudio')
                columns=[r[1] for r in c.execute('PRAGMA table_info(trials)')]
                if set(row)!=set(columns):raise ValueError('Schema de trial incompatible')
                if c.execute('SELECT 1 FROM trials WHERE study_name=? AND trial_number=?',(row['study_name'],row['trial_number'])).fetchone():raise ValueError('Conflicto de trial local')
                c.execute('INSERT INTO trials ('+','.join(columns)+') VALUES ('+','.join('?' for _ in columns)+')',[row[k] for k in columns])
            for row in snapshot.get('reviews',[]):
                if row['job_id'] not in ids:raise ValueError('Revision de otro job')
                c.execute('INSERT INTO reviews(job_id,operator,evidence,decision,at) VALUES (?,?,?,?,?)',
                          [row[k] for k in ('job_id','operator','evidence','decision','at')])
            c.execute('COMMIT')
