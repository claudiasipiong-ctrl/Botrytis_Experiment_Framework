# Trazabilidad requisito → archivo → prueba → figura

| Requisito | Archivos | Dev test / pendiente | Figura |
|---|---|---|---|
| Clases0/1/2, GT y negativos | src/utils/labels.py;pairing.py;dataset.py | MatchingTests | gt_matching;pairing_iou |
| Matching global exacto, empates, muchos-a-uno | src/utils/pairing.py | test_global_bruteforce;test_tie_deterministic;test_many_to_one | pairing_iou |
| Clase2 auxiliar, sin signos | src/utils/boxes.py;gradcam.py;spatial_audit.py | test_no_sign_not_negative;test_clean_negative | sign_coverage;gradcam |
| Anomalias y bloqueo auditable | src/utils/label_audit.py;prepared.py | test_unmatched_positive_blocks_negative | dataset_distribution;gt_examples |
| Split70/20/10 seed67 tresfolds | configs/default.yaml;src/utils/splits.py;folds.py | test_config_contract;test_split_leakage | balance_split |
| No reutilizar semantica anterior | src/utils/semantics.py;prepared.py;optuna_storage.py | test_no_pretrained_scratch_mix;test_source_hash_and_semantics | all |
| Registro diezarquitecturas reparto | src/utils/architectures.py;orchestration/pc_registry.py | test_ten_models;test_owner_mapping | map_arch |
| Pesos cero y RT-DETR autentico | configs/weights_registry.json;src/utils/weights.py | test_no_weights_assumed;test_r50_block | stage_features |
| Modalidades e inicializacion offline | src/utils/pretrained.py;classifier_models.py;detector_train.py | test_no_pretrained_scratch_mix | all |
| Adquisicion/verificacion manual | tools/manage_weights.py | DEV static + QA_MANUAL_PENDIENTE | all |
| Presupuesto15/20,APclasificador | src/utils/guards.py;search_spaces.py;hpo_runner.py | test_config_contract | optuna_trials;hpo_folds |
| Metricas y error global | src/utils/metrics.py;pipeline_eval.py | test_global_error | pipeline_errors;pipeline_matrix |
| Test bloqueado,freeze y release | src/utils/splits.py;manual_gates.py;freeze.py | test_test_closed;test_release_missing | all test sources pending |
| Logs arranque y subprocesos | src/utils/bootstrap.py;logging_utils.py;scripts/Invoke-Botrytis.ps1 | test_bootstrap_failure_persists;test_subprocess_and_returncode | all provenance |
| Redaccion,locks,append,reanudacion | src/utils/bootstrap.py;durable.py;resume.py | test_redaction;test_runcontext_exception_and_reuse | all provenance |
| Sin remoto,manual exchange | src/orchestration/worker.py;exchange.py;sync.py | DEV static;QA_MANUAL_PENDIENTE | pipeline_horizontal |
| Catalogo60items+aliases | configs/figure_catalog.json;src/utils/figures_all.py | test_catalog_states_traceability | all60 |
| Dos figuras principales2paneles | src/utils/figures_all.py;detector_train.py | DEV renderer synthetic;QA_MANUAL_PENDIENTE | det_final_two_panel;cls_final_two_panel |
| Grafos y stagefeatures | tools/inspect_model_graph.py;extract_stage_features.py | DEV syntax;QA_MANUAL_PENDIENTE | stage_features;cib_reference;mhsa_reference |
| HTMLoffline y comandosPS5.1 | configs/workflow.json;docs/GUIA_EJECUCION_3PCs_ACTUALIZADA.md | DEV HTML links/render + PS parser | pipeline_horizontal |
| QA personal sin aprobaciones automaticas | src/utils/manual_gates.py;tools/manual_gates.py | test_manual_gate_pending | checklist48gates |
| ZIP CRC/rutas/hash/manifiesto | tools/make_zip.py;MANIFEST_ENTREGA_SHA256.json | DEV integridad paquete | catalog manifest |
