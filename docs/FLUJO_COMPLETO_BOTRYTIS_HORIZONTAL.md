# Flujo derivado del codigo final

Fuente comun: configs/workflow.json. HTML externo: [Flujo](../../entregables_botrytis/FLUJO_COMPLETO_BOTRYTIS_HORIZONTAL.html).

Inventario e identidad local → Entorno, CUDA y seguridad personales → Gestion de pesos: inicialmente cero → QA manual del usuario, gate cerrado → Clases y matching GT; resolver anomalias → Split externo y tres folds internos → Detector una clase y crops exclusivamente clase 0 → Baselines, HPO, robustez y candidatos → Intercambio manual e importacion segura → Seleccion externa de detector y clasificador → Umbrales con el 20% externo → Validar pipeline integrado → Freeze: sellar decisiones y checkpoints → Desbloqueo exclusivamente personal del test → Test final una vez tras freeze y release → Benchmark comun y eficiencia → Figuras, fuentes y exportaciones

- 01: `tools/prepare_machine.py`; entrada ZIP completo, proyecto previo, carpeta activa; mapa acordado; salida configs/local.yaml y vinculacion explicita; originales intactos; gate IDENTITY = NO EJECUTADO.
- 02: `src/00_check_environment.py`; entrada Python/venv locales y versiones acordadas; salida environment.json + evidencia CUDA local; gate ENVIRONMENT_CUDA = NO EJECUTADO.
- 03: `tools/manage_weights.py`; entrada Catalogo oficial fijado; conexion/licencia cuando el usuario decida; salida weights/ + .provenance.json con SHA256 y tamano; gate WEIGHTS = NO EJECUTADO.
- 04: `tools/qa.py`; entrada Checklist; equipo fisico; datos/pesos listos segun prueba; salida Logs tecnicos; checklist firmado por la persona; gate LOGGING_RECOVERY = NO EJECUTADO.
- 05: `src/01_validate_dataset.py`; entrada 0=arandano_a; 1=arandano_sospechoso; 2=signo_sospechoso_botrytis; salida dataset_inventory, issues, pairing audit/candidates/anomalies; hash; gate ANNOTATION_REVIEW = NO EJECUTADO.
- 06: `src/02_create_outer_split.py`; entrada Auditoria resuelta; imagenes originales; seed 67; salida split 70/20/10; inner folds; test_lock; hashes; gate SPLIT_FOLDS = NO EJECUTADO.
- 07: `src/04_build_detection_dataset.py`; entrada Split/folds firmados; bbox clase 0; etiquetas por matching; salida data/derived_v4/detection y classification; manifests + sellos; gate DERIVED_DATA = NO EJECUTADO.
- 08: `src/orchestration/worker.py`; entrada READY_FOR_EXPERIMENT aprobado personalmente; datos/modelos verificados; salida 15 trials/detector, 20/clasificador; top3 × seeds17/42/73 × folds3; candidatos entrenados70%; gate READY_FOR_EXPERIMENT = NO EJECUTADO.
- 09: `src/orchestration/sync.py`; entrada Snapshots inmutables, propietario, hashes de codigo/datos/split; salida ZIP + sidecar SHA256, import_ledger/journal, evidencias por PC; gate LOGGING_RECOVERY = NO EJECUTADO.
- 10: `src/12_select_detector.py`; entrada Metricas del 20% externo de los diez candidatos; sin test; salida selected_detector.json, selected_classifier.json, policy hash; gate SELECTION = NO EJECUTADO.
- 11: `src/11_tune_inference_thresholds.py`; entrada Modelos seleccionados; predicciones externas; salida inference_thresholds; umbral prob, conf, NMS; gate THRESHOLDS_PIPELINE = NO EJECUTADO.
- 12: `src/14_validate_integrated_pipeline.py`; entrada Detector→crops predichos→clasificador; GTauditado; salida pipeline_validation; tablas detecciones/misses; matriz con no_detectado; gate THRESHOLDS_PIPELINE = NO EJECUTADO.
- 13: `src/15_freeze_experiment.py`; entrada Seleccion y umbrales revisados; metadatos y todos los candidatos; salida freeze_manifest.json y SHA; test aun bloqueado; gate FREEZE = NO EJECUTADO.
- 14: `tools/manual_gates.py`; entrada Gate FREEZE registrado y freeze integro; salida manual_test_release.json ligado al SHA256 del freeze; gate TEST_RELEASE = NO EJECUTADO.
- 15: `src/16_evaluate_test.py`; entrada 10% externo sellado; freeze y autorizacion personal; salida test_results.json + hashes + errores cualitativos; gate TEST_FINAL = NO EJECUTADO.
- 16: `src/17_benchmark_latency.py`; entrada Modelos congelados; PC-01 libre; salida latencia por etapas; VRAM; protocolo de benchmark; gate BENCHMARK = NO EJECUTADO.
- 17: `tools/generate_all_figures.py`; entrada CSV/JSON historiales reales y descriptores trazables; salida PNG300dpi + SVG/PDF; figure_manifest.csv/json y pendientes; gate FIGURES = NO EJECUTADO.
