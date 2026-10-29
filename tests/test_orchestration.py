"""Orquestacion de las 3 PCs: cola de estudios, estados, dependencias, 1 GPU por PC, fusion de estados y lock de GPU."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from orchestration.pc_registry import PCRegistry, UnknownPC
from orchestration.study_queue import COMPLETE, FAILED, PENDING, RUNNING, Job, StudyQueue, job_id_for
from orchestration.worker import build_command
from utils.architectures import archs, get_arch
from utils.checks import MethodologyViolation
from utils.config_loader import load_config
from utils.guards import GpuJobLock


@pytest.fixture()
def cfg(tmp_path: Path) -> dict:
    return load_config(None, [f"paths.queue_dir={tmp_path / 'queue'}", f"paths.artifacts_dir={tmp_path / 'art'}", f"paths.environments_dir={tmp_path / 'art' / 'env'}"])


@pytest.fixture()
def queue(tmp_path: Path, cfg: dict) -> StudyQueue:
    q = StudyQueue(tmp_path / "q.sqlite")
    q.plan(cfg)
    return q


def test_plan_creates_one_baseline_and_one_hpo_per_architecture(queue: StudyQueue) -> None:
    jobs = queue.jobs()
    assert len(jobs) == 20 and {j.kind for j in jobs} == {"baseline", "hpo"}
    names = {j.study_name for j in jobs}
    assert len(names) == 10 and "botrytis_yolov8n" in names and "botrytis_rtdetr_r50" in names
    assert not any("global" in n for n in names)            # un estudio por arquitectura, nunca study_global
    hpo = queue.get(job_id_for("hpo", "yolov8n"))
    assert hpo is not None and hpo.depends_on == [job_id_for("baseline", "yolov8n")]
    assert queue.plan(load_config(None, [])) == 0           # idempotente


def test_claim_respects_dependencies_one_gpu_per_pc_and_global_cap(queue: StudyQueue) -> None:
    j1 = queue.claim_next("PC-01", preferred_slugs=["yolov8n"])
    assert j1 is not None and j1.kind == "baseline" and j1.architecture == "yolov8n" and j1.status == RUNNING and j1.pc_id == "PC-01"
    assert queue.claim_next("PC-01") is None                  # una PC = un trabajo GPU
    j2 = queue.claim_next("PC-02", preferred_slugs=["yolov8l"])
    j3 = queue.claim_next("PC-03", preferred_slugs=["resnet50"])
    assert {j2.architecture, j3.architecture} == {"yolov8l", "resnet50"}      # type: ignore[union-attr]
    assert queue.claim_next("PC-04") is None                  # maximo 3 trabajos globales
    assert queue.counts()[RUNNING] == 3
    # el HPO de yolov8n no puede empezar mientras su baseline no este COMPLETE
    queue.finish(j2.job_id, COMPLETE)                          # type: ignore[union-attr]
    nxt = queue.claim_next("PC-02", preferred_slugs=["yolov8n"])
    assert nxt is not None and nxt.job_id != job_id_for("hpo", "yolov8n")
    queue.finish(j1.job_id, COMPLETE)
    queue.finish(nxt.job_id, COMPLETE)
    assert queue.claim_next("PC-01", preferred_slugs=["yolov8n"], kinds=["hpo"]).job_id == job_id_for("hpo", "yolov8n")  # type: ignore[union-attr]


def test_a_running_job_is_never_started_twice(queue: StudyQueue) -> None:
    a = queue.claim_next("PC-01", preferred_slugs=["yolov8n"])
    b = queue.claim_next("PC-02", preferred_slugs=["yolov8n"])
    assert a is not None and b is not None and a.job_id != b.job_id


def test_failed_pruned_states_and_trials_table(queue: StudyQueue) -> None:
    j = queue.claim_next("PC-01")
    assert j is not None
    queue.finish(j.job_id, FAILED, error="boom")
    assert queue.get(j.job_id).status == FAILED and "boom" in queue.get(j.job_id).error     # type: ignore[union-attr]
    queue.record_trial("botrytis_yolov8n", 0, "RUNNING", pc_id="PC-01", provenance={"git_commit": "abc", "dataset_hash": "d", "split_hash": "s"})
    queue.record_trial("botrytis_yolov8n", 0, "PRUNED", value=0.12, pc_id="PC-01", finished=True)
    queue.record_trial("botrytis_yolov8n", 1, "COMPLETE", value=0.4, pc_id="PC-01", finished=True)
    t = queue.trials_table()
    assert list(t["status"]) == ["PRUNED", "COMPLETE"] and t.loc[0, "git_commit"] == "abc"
    assert queue.retry_failed() == 1 and queue.get(j.job_id).status == PENDING          # type: ignore[union-attr]


def test_expand_after_hpo_builds_the_robustness_chain(queue: StudyQueue, cfg: dict) -> None:
    spec = get_arch("YOLOv8n")
    n = queue.expand_after_hpo(cfg, spec, ["cfg_aaaaaaaa", "cfg_bbbbbbbb", "cfg_cccccccc"])
    assert n == 9 + 4                                           # 3 configs x 3 seeds + aggregate/outer_train/outer_eval/thresholds
    rep = queue.get("repeat:yolov8n:cfg_aaaaaaaa:s17")
    assert rep is not None and rep.payload == {"config_id": "cfg_aaaaaaaa", "seed": 17}
    agg = queue.get(job_id_for("aggregate", "yolov8n"))
    assert agg is not None and len(agg.depends_on) == 9
    assert queue.get(job_id_for("thresholds", "yolov8n")).depends_on == [job_id_for("outer_eval", "yolov8n")]        # type: ignore[union-attr]
    assert queue.expand_after_hpo(cfg, spec, ["cfg_aaaaaaaa", "cfg_bbbbbbbb", "cfg_cccccccc"]) == 0


def test_seeds_can_run_in_parallel_on_three_pcs(queue: StudyQueue, cfg: dict) -> None:
    spec = get_arch("ResNet50")
    for kind in ("baseline", "hpo"):
        jid = job_id_for(kind, spec.slug)
        queue.claim_next("PC-01", slugs=[spec.slug], kinds=[kind])
        queue.finish(jid, COMPLETE)
    queue.expand_after_hpo(cfg, spec, ["cfg_aaaaaaaa"])
    got = [queue.claim_next(pc, slugs=[spec.slug], kinds=["repeat"]) for pc in ("PC-01", "PC-02", "PC-03")]
    assert all(g is not None for g in got) and {g.payload["seed"] for g in got} == {17, 42, 73}     # type: ignore[union-attr]


def test_jobs_without_local_inputs_stay_pending_and_chains_stay_on_the_same_pc(queue: StudyQueue, cfg: dict, tmp_path: Path) -> None:
    from orchestration.worker import local_inputs_ready

    spec = get_arch("ResNet50")
    cfg["paths"]["optuna_dir"] = str(tmp_path / "optuna")
    cfg["paths"]["best_configs_dir"] = str(tmp_path / "best")
    cfg["paths"]["artifacts_dir"] = str(tmp_path / "art")
    for kind in ("baseline", "hpo"):
        assert queue.claim_next("PC-03", slugs=[spec.slug], kinds=[kind]) is not None
        queue.finish(job_id_for(kind, spec.slug), COMPLETE)
    queue.expand_after_hpo(cfg, spec, ["cfg_aaaaaaaa"])
    runs = tmp_path / "runs"
    # PC-01 no tiene el estudio Optuna (se ejecuto en PC-03): no debe reclamar la robustez
    ready = lambda j: local_inputs_ready(cfg, j, runs)  # noqa: E731
    assert queue.claim_next("PC-01", slugs=[spec.slug], kinds=["repeat"], ready_check=ready) is None
    (tmp_path / "optuna").mkdir(exist_ok=True)
    (tmp_path / "optuna" / "botrytis_resnet50.db").write_bytes(b"x")                         # tras desempaquetar el ZIP de PC-03
    got = queue.claim_next("PC-01", slugs=[spec.slug], kinds=["repeat"], ready_check=ready)
    assert got is not None and got.kind == "repeat"
    queue.finish(got.job_id, FAILED)
    # agregar exige los result.json de los 3 seeds en el disco local
    assert not local_inputs_ready(cfg, queue.get(job_id_for("aggregate", spec.slug)), runs)       # type: ignore[arg-type]
    for s in (17, 42, 73):
        d = runs / spec.slug / "robustness" / "cfg_aaaaaaaa" / f"seed_{s}"
        d.mkdir(parents=True)
        (d / "result.json").write_text("{}")
    assert local_inputs_ready(cfg, queue.get(job_id_for("aggregate", spec.slug)), runs)           # type: ignore[arg-type]
    assert not local_inputs_ready(cfg, queue.get(job_id_for("outer_train", spec.slug)), runs)     # type: ignore[arg-type]
    assert not local_inputs_ready(cfg, queue.get(job_id_for("outer_eval", spec.slug)), runs)      # type: ignore[arg-type]
    assert local_inputs_ready(cfg, queue.get(job_id_for("hpo", spec.slug)), runs)                 # type: ignore[arg-type]
    # continuidad: la PC que ya trabajo una arquitectura prefiere sus siguientes etapas frente a una baseline ajena
    nxt = queue.claim_next("PC-03", kinds=["baseline", "repeat"])
    assert nxt is not None and nxt.architecture == spec.slug


def test_recover_stale_and_merge_states(tmp_path: Path, cfg: dict) -> None:
    a, b = StudyQueue(tmp_path / "a.sqlite"), StudyQueue(tmp_path / "b.sqlite")
    a.plan(cfg)
    b.plan(cfg)
    ja = a.claim_next("PC-01", preferred_slugs=["yolov8n"])
    assert ja is not None
    a.finish(ja.job_id, COMPLETE)
    jb = b.claim_next("PC-02", preferred_slugs=["yolov8l"])
    assert jb is not None
    state_a = a.export_state(tmp_path / "state_a.json")
    res = b.merge_state(state_a)
    assert res["updated"] >= 1 and b.get(ja.job_id).status == COMPLETE and b.get(jb.job_id).status == RUNNING         # type: ignore[union-attr]
    # un RUNNING huerfano (sin latido) se recupera; el que esta vivo no
    assert b.recover_stale(minutes=10_000) == []
    assert b.recover_stale(minutes=0, force=True, pc_id="PC-02") == [jb.job_id]
    assert b.get(jb.job_id).status == PENDING                                                                           # type: ignore[union-attr]


def test_merge_flags_same_job_running_on_two_pcs(tmp_path: Path, cfg: dict) -> None:
    a, b = StudyQueue(tmp_path / "a.sqlite"), StudyQueue(tmp_path / "b.sqlite")
    a.plan(cfg)
    b.plan(cfg)
    a.claim_next("PC-01", preferred_slugs=["yolov8n"])
    b.claim_next("PC-02", preferred_slugs=["yolov8n"])
    res = b.merge_state(a.export_state(tmp_path / "s.json"))
    assert res["conflicts"] and res["conflicts"][0]["job_id"] == job_id_for("baseline", "yolov8n")


def test_build_command_passes_config_and_overrides() -> None:
    job = Job("repeat:yolov8n:cfg_x:s42", "repeat", "yolov8n", "detection", "botrytis_yolov8n", payload={"config_id": "cfg_x", "seed": 42})
    cmd = build_command(job, "PC-02", "cfg.yaml", ["a=1", "b=2"], python="py", runs_dir="runs")
    assert cmd[0] == "py" and cmd[1].endswith("08_repeat_top_configs.py")
    assert "--architecture" in cmd and "yolov8n" in cmd and cmd[cmd.index("--pc-id") + 1] == "PC-02"
    assert cmd.count("--set") == 2 and cmd[cmd.index("--seed") + 1] == "42" and cmd[cmd.index("--config-id") + 1] == "cfg_x"
    agg = build_command(Job("aggregate:yolov8n", "aggregate", "yolov8n", "detection", "s"), "PC-01", "c.yaml", [])
    assert "--aggregate-only" in agg


def test_pc_registry_identity_and_first_batch(cfg: dict) -> None:
    reg = PCRegistry(cfg)
    assert reg.validate("PC-01").is_benchmark and not reg.validate("PC-02").is_benchmark
    assert reg.is_preferred("PC-01", "yolov8m") and reg.is_preferred("PC-03", "vit_b16") and reg.is_preferred("PC-02", "yolo26l")
    with pytest.raises(UnknownPC):
        reg.validate("PC-99")
    assert reg.missing_environments() == ["PC-01", "PC-02", "PC-03"]


def test_environment_mismatch_between_pcs_is_detected(cfg: dict) -> None:
    reg = PCRegistry(cfg)
    env_dir = Path(cfg["paths"]["environments_dir"])
    env_dir.mkdir(parents=True, exist_ok=True)
    base = {"gpu_model": "RTX 6000 Ada", "torch": "2.5.0", "python_version": "3.12.1", "code_hash": "h1"}
    (env_dir / "environment_pc01.json").write_text(json.dumps(base))
    (env_dir / "environment_pc02.json").write_text(json.dumps({**base, "torch": "2.6.0"}))
    (env_dir / "environment_pc03.json").write_text(json.dumps({**base, "code_hash": None}))
    fields = {m["field"] for m in reg.environment_mismatches()}
    assert "torch" in fields and "code_hash" not in fields      # hash ausente en una PC no es discrepancia


def test_gpu_lock_allows_a_single_heavy_job_per_pc(cfg: dict) -> None:
    lock_a = GpuJobLock(cfg, "PC-01", "a")
    with lock_a:
        # un segundo trabajo GPU de OTRO proceso vivo en la misma PC debe rechazarse (simulado con el pid del padre)
        lock_a.path.write_text(json.dumps({"pid": os.getppid(), "job": "otro", "pc_id": "PC-01"}))
        with pytest.raises(MethodologyViolation, match="trabajo GPU"):
            with GpuJobLock(cfg, "PC-01", "b"):
                pass
    # otra PC no se ve afectada
    with GpuJobLock(cfg, "PC-02", "c"):
        pass
    # un lock huerfano (pid inexistente) se recupera
    lock_a.path.parent.mkdir(parents=True, exist_ok=True)
    lock_a.path.write_text(json.dumps({"pid": 99999999, "job": "muerto", "pc_id": "PC-01"}))
    with GpuJobLock(cfg, "PC-01", "d"):
        pass


def test_all_ten_architectures_registered() -> None:
    assert [a.name for a in archs("detection")] == ["YOLOv8n", "YOLOv8m", "YOLOv8l", "YOLO26n", "YOLO26s", "YOLO26l", "RT-DETR-R50"]
    assert [a.name for a in archs("classification")] == ["ResNet50", "ViT-B/16", "Swin-Tiny"]
    with pytest.raises(KeyError, match="U-Net"):
        get_arch("U-Net")
