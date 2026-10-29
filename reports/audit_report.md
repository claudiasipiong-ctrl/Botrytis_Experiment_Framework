# Informe de auditoria interna

Generado: 2026-10-03T20:59:11-05:00  
Python: 3.12.14

| Comprobacion | Resultado | Detalle |
|---|---|---|
| sintaxis de 96 ficheros .py | PASS |  |
| import utils.classifier_data | SKIP | No module named 'torch' |
| import utils.classifier_models | SKIP | No module named 'torch' |
| import utils.classifier_train | SKIP | No module named 'torch' |
| import utils.detector_train | SKIP | No module named 'torch' |
| import utils.gradcam | SKIP | No module named 'torch' |
| import utils.hpo_runner | SKIP | No module named 'torch' |
| import utils.inference | SKIP | No module named 'torch' |
| import utils.model_stats | SKIP | No module named 'torch' |
| import utils.reporting | SKIP | No module named 'torch' |
| imports utils | SKIP | 37/46 ejecutados correctamente; 9 omitidos; 0 fallidos |
| load 08_repeat_top_configs.py | SKIP | No module named 'torch' |
| load 09_train_outer_candidate.py | SKIP | No module named 'torch' |
| load 10_evaluate_outer_validation.py | SKIP | No module named 'torch' |
| load 11_tune_inference_thresholds.py | SKIP | No module named 'torch' |
| load 12_select_detector.py | SKIP | No module named 'torch' |
| load 13_select_classifier.py | SKIP | No module named 'torch' |
| load 14_validate_integrated_pipeline.py | SKIP | No module named 'torch' |
| load 16_evaluate_test.py | SKIP | No module named 'torch' |
| load 17_benchmark_latency.py | SKIP | No module named 'torch' |
| load 18_export_master_results.py | SKIP | No module named 'torch' |
| carga de scripts (sin ejecutar main) | SKIP | 10/20 ejecutados correctamente; 10 omitidos; 0 fallidos |
| los 19 scripts de la arquitectura final presentes | PASS | set() |
| utilidades requeridas presentes (sec. 30) | PASS | set() |
| orchestration/{worker,study_queue,pc_registry}.py presentes | PASS | {'pc_registry', 'study_queue', 'worker'} |
| import de orchestration.* | PASS |  |
| --help 08_repeat_top_configs.py | SKIP | Optional ML dependency unavailable; actual CLI import not verified |
| --help 09_train_outer_candidate.py | SKIP | Optional ML dependency unavailable; actual CLI import not verified |
| --help 10_evaluate_outer_validation.py | SKIP | Optional ML dependency unavailable; actual CLI import not verified |
| --help 11_tune_inference_thresholds.py | SKIP | Optional ML dependency unavailable; actual CLI import not verified |
| --help 12_select_detector.py | SKIP | Optional ML dependency unavailable; actual CLI import not verified |
| --help 13_select_classifier.py | SKIP | Optional ML dependency unavailable; actual CLI import not verified |
| --help 14_validate_integrated_pipeline.py | SKIP | Optional ML dependency unavailable; actual CLI import not verified |
| --help 16_evaluate_test.py | SKIP | Optional ML dependency unavailable; actual CLI import not verified |
| --help 17_benchmark_latency.py | SKIP | Optional ML dependency unavailable; actual CLI import not verified |
| --help 18_export_master_results.py | SKIP | Optional ML dependency unavailable; actual CLI import not verified |
| scripts ejecutables (--help exit 0) | SKIP | 12/22 ejecutados correctamente; 10 omitidos; 0 fallidos |
| sin rutas absolutas hardcodeadas en .py | PASS |  |
| sin rutas absolutas en configs/default.yaml | PASS |  |
| type hints completos (578 funciones) | PASS |  |
| docstrings en modulos y funciones publicas | PASS |  |
| pyflakes (imports sin usar / nombres indefinidos) | PASS |  |
| configs/default.yaml carga y respeta decisiones FIJAS (perfil official) | PASS |  |
| TODO BLOQUEANTES restantes = seed externa por predeclarar en default (adaptado ya la fija) | PASS | outer_split.seed = TODO_PREDECLARE |
| configs/smoke.yaml carga (perfil smoke no relaja decisiones FIJAS) | PASS |  |
| configs/search_spaces.yaml define los 6 espacios (yolov8, yolo26, rtdetr, resnet50, vit, swin) | PASS | {'resnet50', 'yolo26', 'rtdetr', 'vit', 'yolov8', 'version', 'swin'} |
| src/ no importa datos sinteticos ni figuras demo (graficos con datos reales) | PASS |  |
| solo 16_evaluate_test usa la etapa 'final_test' (test bloqueado) | PASS |  |
| finales de linea LF, UTF-8 sin BOM | PASS |  |
| suite de tests (pytest) | SKIP | No solicitada en esta ejecución; usar --with-tests o consultar reports/tests.xml |

**Resumen:** 17 PASS; 0 FAIL; 33 SKIP.
