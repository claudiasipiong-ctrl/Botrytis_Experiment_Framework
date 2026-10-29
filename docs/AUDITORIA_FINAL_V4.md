# AUDITORIA FINAL · Botrytis · 2026-10-29

**Estado: DEV_CHECK_PASS en las comprobaciones indicadas; QA_MANUAL_PENDIENTE; EXPERIMENTO_PENDIENTE.**
La implementacion y el paquete estan preparados con bloqueos explicitos. No se certifica un experimento terminado.
En particular RT-DETR-R50 pretrained no puede ejecutarse como si existiera un checkpoint compatible del backend heredado.

## Entradas realmente inspeccionadas y proteccion de cambios

Proyecto editable: `C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework`.
ZIP obligatorio: `C:\Users\Erick Obradovich\Documents\Codex\2026-10-28\objetivo-auditor-a-optimizaci-n-y\outputs\Botrytis_Framework_Auditado_2026-10-28.zip`; SHA256 `89a343a679e956f3e2f03d33f44aa98285b860b52ebb5e493a5035c608f3d17d`; lectura/CRC verificados.
Comparacion previa: {'MODIFICADO_LOCAL': 82, 'SOLO_ZIP': 37, 'SOLO_LOCAL': 104, 'IDENTICO': 23}. Se guardaron hashes por archivo ANTES de escribir y respaldo textual previo.
Se integraron capacidades ausentes del ZIP sin reemplazar el arbol local por el ZIP. Las modificaciones locales anteriores
se conservaron como base de trabajo y en RESPALDO_CODIGO_PREVIO.zip. Los datos originales permanecen sin modificar.
El lanzador anterior externo tambien queda en ese respaldo (external/generar_graficos_botrytis.py).
Contexto metodologico real: Downloads/Texto pegado(20261029-111929).txt; ruta confirmada por el usuario e inspeccionada.
Guia real: Downloads/GUIA_EJECUCION_3PCs_CUDA_OK_CONTINUAR.html. No se encontro la variante (1); se uso el adjunto disponible.
Sus SHA256 estan en evidencia_desarrollo/context_inputs.json. Las instrucciones actuales prevalecieron sobre texto anterior.
No se utilizaron otros chats ni se buscaron credenciales. No se realizaron conexiones a otras PCs.

## Hallazgos criticos y cambios

1. El proyecto local ya tenia matching v3 y diez arquitecturas; el ZIP traia 8 specs: YOLOv8n, YOLOv8l, YOLO26n, YOLO26l, RT-DETR-R50, ResNet50, ViT-B/16, Swin-Tiny.
   Los dos detectores ausentes en ZIP eran YOLO26n y YOLO26s. Se conservaron en local y se propagaron a inventario,
   reparto, HPO, catalogo, guias y contratos. No se afirma haberlos creado de nuevo donde ya existian.
2. Se fijo outer_split.seed=67, separada de robustez17/42/73, con70/20/10 por imagen y3folds internos.
   No se genero, reordeno ni abrio ningun split real. Namespacesv4 y huellas invalidan derivados antiguos sin borrarlos.
3. Clase0 es unico fruto/target; clase1 aporta positivos por matching global maxIoU>=0.50 configurable; clase2 es auxiliar.
   La auditoria de duplicados ahora es estable al invertir el orden de entrada, defecto encontrado y corregido por unit tests.
   Conflictos conservan NA/bloqueo, nunca negativos silenciosos. Cobertura solo descriptiva; no hay umbral de cobertura activo.
4. Se retiraron descargas implicitas y el trasplante parcial ImageNet que se presentaba como RT-DETR-R50 preentrenado.
   Existe R50-vd oficial, pero no se demostro compatibilidad con el grafo ResNet50-SiLU Ultralytics. El checkpoint completo
   compatible para ese backend carece de fuente fiable demostrada y se bloquea. La adopcion/validacion del backend oficial
   o una variante scratch separada requiere decision y QA personales; no se implemento un adaptador oficial validado.
   RT-DETR-L no se usa como R50. Este bloqueo impide cerrar la comparacion primaria de siete detectores y su freeze.
5. Las diez entradas de pesos empiezan NO_DISPONIBLE. Solo se consultaron documentacion/metadatos oficiales;
   cero pesos descargados. URLs/version/licencias/tamanos/checksums publicados y estados locales estan en el CSV/JSON.
   ResNet50 tiene prefijo oficial SHA11ad3fa6, no hash completo publicado en esta entrega; tamano/completo quedan para verificacion manual.
   ViT usa ImageNet21k→1k y los otros clasificadores1k: diferencia declarada, no condicion identica inventada.
6. Listado exclusivamente de nombres/metadatos del dataset activo: **324 imagenes y 324 labels**, referencia329.
   Diferencia imagenes: -5. No se decodificaron pixeles ni se ejecutaron matching/auditoria/QA sobre datos reales.
   Stems, labels vacios, clases, GT, positivos/negativos, duplicados, EXIF, integridad y crops reales quedan PENDIENTE_USUARIO.
   No se tomo el arbol Markdown ni los informes antiguos de328 como dataset autoritativo; no se movio material desde _off.
7. Bootstrap antes de imports/config, RunContext conUUIDUTC/PC/fase, eventsJSONL, salidas/tracebacks/subprocesos,
   configuracion efectiva redactada, hashes, locks, snapshots de intentos y estados de salida. Las epocas generan eventos.
   Hay mecanismos de checkpoints/recibos y resume posterior; su ejecucion con ML/GPU real sigue pendiente.
   Red/puertos bloqueados en comandos locales, salvo adquisicion de pesos invocada personalmente con flags explicitos.
   Orquestacion y transferencias son locales/manuales; mapa PC-01=i21315,PC-02=i21314,PC-03=i21313, sin deducirlo del host.
8. Se eliminaron sumas arbitrarias de lossYOLO y ceros por ausencia de objetivo HPO. Se preservan componentes autenticos,
   AP positiva para clasificacion y mAP50:95 para deteccion. Error integrado GT=(FN+EC)/N_GT conFP/duplicados aparte.
9. Catalogo de60items y8aliases; generadores reales desde CSV trazables, adapters de historiales/HPO y contratos de fuentes.
   Estado actual: **2 esquemas metodologicos +2 referencias arquitectonicas SVG;56 figuras pendientes por datos insuficientes**.
   Cero curvas/resultados experimentales nuevos. CIB/MHSA conceptuales no se presentan como modulos medidos.
   No se produjeron estimaciones. El CLI demo solo permite Codigo/estimaciones_exploratorias, rotuladasILUSTRATIVO.
10. Ambos HTML offline derivan de configs/workflow.json:17etapas, comandos reales,48gates editables por guia/checklist.
    Todos comienzan NO EJECUTADO. CUDA reportada el28/10 en i21315 es historica, nunca verificacion actual.

## Autocomprobaciones de desarrollo realmente ejecutadas

| Comprobacion | Resultado | Alcance |
|---|---|---|
| compileall src/tools/tests/dev_tests/lanzador | DEV_CHECK_PASS | Sintaxis; sin imports de modelos |
| Pytest selectivo con fixtures sinteticos | 90 passed, 0 fallos, 0 errores | 1 test de datos reales deliberadamente deseleccionado; no QA |
| Regresion posterior de metadata de esquemas |35 unitarias OK | Datos sinteticos; sin GPU/dataset real |
| Contratos AST de imports locales | 1088 simbolos,0errores | No importa modelos |
| Pyflakes |0nombres indefinidos/errores sintacticos | Persisten avisos de imports/variables no usados; log completo adjunto |
| --help de17entradas | DEV_CHECK_PASS | Solo ayuda/parseo, nunca main operativo |
| Parser PowerShell 5.1.19041.6456 | 22 archivos/bloques,0errores | ParseFile, sin ejecutar los bloques |
| HTML headless local |6renders OK |1440y390px,0peticionesHTTP,0erroresJS,0links/IDs rotos,48gates iniciales cerrados |
| Integridad paquete/manifest | Ver informe INTEGRIDAD_ENTREGA.json | CRC/rutas/SHA, no QA manual |

Se instalaron bibliotecas de desarrollo aisladas en work/devdeps (logs/pins observados); no se altero el entorno ML del proyecto.
Los primeros intentos fallidos y reparaciones se preservan: determinismo de duplicados, actualizacion de tests heredados,
contratos de funciones, errores de importacion/sintaxis y redaccion de stdout por fragmentos. No se convierten esos fallos en PASS retroactivo.
Una invocacion inicial del archivo auxiliar ParseFile.ps1 fue bloqueada por politica; no se cambio la politica.
La comprobacion posterior uso directamente la API estatica Parser.ParseFile en PowerShell5.1, sin cargar/ejecutar ese archivo ni los scripts del framework.
Los renders fueron de HTML locales en un navegador headless por file://; no hubo capturas de escritorio ni control de raton/teclado/equipos remotos.

## Pendientes y limites reales

- QA manual en cadaPC: identidad, CUDA actual, seguridad OEM/recursos, integridad y reconciliacion del dataset,
  auditoria de anotaciones/IoU, splits/folds/crops, pesos/licencias y pruebas de fallo/recuperacion.
- ML no instalado en este entorno de desarrollo: torch,torchvision,timm,ultralytics,optuna. Imports/componentes
  que los requieren, grafo instalado de cada modelo, entrenamiento y resume real son PENDIENTE_USUARIO.
  Un intento de import de reporting fallo por torch ausente; no se presenta como comprobacion superada.
- RT-DETR-R50 completo: integrar/validar backend oficial o decidir alternativa de forma explicita; no sustituirlo silenciosamente.
- Test externo, smoke integral, entrenamiento, HPO, robustez, seleccion, thresholds, freeze, benchmark y curvas reales:
  NO EJECUTADOS en este encargo. No se ejecutaron Invoke-QA.ps1,qa.sh,qa.py,qa_checks ni QA_MANUAL.
- El paquete es preparacion de codigo/documentacion; no certifica reproducibilidad numerica, equivalencia arquitectonica
  ni rendimiento de las tresPCs. Ningun gate fue aprobado por Codex.

## Secuencia personal del usuario

Abrir GUIA_EJECUCION_3PCs_ACTUALIZADA.html → identificar cadaPC y preparar entorno → revisar fuentes/descargar o
transferir pesos personalmente → ejecutar QA y reconciliar dataset → resolver anomalias GT → crear/importar split/folds
→ generar detector/crops → firmar gates → baselines/HPO/robustez y candidatos → intercambio manual → seleccion externa
→ umbrales/pipeline → freeze → autorizacion personal test → test una vez → benchmark PC-01 → fuentes/figuras/exportaciones.
Detenerse en cualquier gate BLOQUEADO. No ejecutar etapas posteriores para ocultar dependencias faltantes.

## Archivos cambiados

119 archivos de codigo/config/tests/documentacion agregados o modificados frente al inventario local inicial.
El detalle de hashes anteriores/finales esta en Botrytis_Experiment_Framework/docs/CHANGES_V4.json y en la tabla siguiente.
Las evidencias historicas permanecen sin reescribir en origen y tienen indice de obsolescencia.

| Archivo | Accion | Motivo |
|---|---|---|
| ADAPTACION_BOTRYTIS.md | MODIFICADO | figuras/documentacion/tests/integridad |
| arbol_dataset.md | AGREGADO_O_RESTAURADO_DESDE_ZIP | metodologia/manifiestos |
| AUDIT_REPORT.md | MODIFICADO | figuras/documentacion/tests/integridad |
| GUIA_EJECUCION_3PCs.md | MODIFICADO | figuras/documentacion/tests/integridad |
| README.md | MODIFICADO | figuras/documentacion/tests/integridad |
| SEPARACION_Y_EVIDENCIAS.md | MODIFICADO | figuras/documentacion/tests/integridad |
| configs/adaptado.yaml | MODIFICADO | figuras/documentacion/tests/integridad |
| configs/default.yaml | MODIFICADO | figuras/documentacion/tests/integridad |
| configs/figure_catalog.json | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| configs/fleet_mapping.json | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| configs/operations.yaml | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| configs/weights_registry.json | AGREGADO_O_RESTAURADO_DESDE_ZIP | arquitecturas/pesos/HPO |
| configs/workflow.json | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| dev_tests/test_revision.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| docs/FIGURAS.md | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| docs/FLUJO_COMPLETO_BOTRYTIS_HORIZONTAL.md | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| docs/GUIA_EJECUCION_3PCs_ACTUALIZADA.md | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| docs/HIPERPARAMETROS.md | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| docs/HISTORICAL_ARTIFACTS.json | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| docs/LOGGING.md | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| docs/LOGGING_ENTRYPOINTS.json | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| docs/METODOLOGIA.md | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| docs/PESOS.md | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| docs/SEMANTIC_SCAN_FINAL.json | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| docs/SOURCE_METADATA.json | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| docs/TRAZABILIDAD.md | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| scripts/Invoke-Botrytis.ps1 | AGREGADO_O_RESTAURADO_DESDE_ZIP | logging/operacion local |
| scripts/Invoke-QA.ps1 | AGREGADO_O_RESTAURADO_DESDE_ZIP | logging/operacion local |
| scripts/Prepare-Environment.ps1 | AGREGADO_O_RESTAURADO_DESDE_ZIP | logging/operacion local |
| scripts/qa.sh | AGREGADO_O_RESTAURADO_DESDE_ZIP | logging/operacion local |
| src/00_check_environment.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/01_validate_dataset.py | MODIFICADO | metodologia/manifiestos |
| src/02_create_outer_split.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/03_create_inner_folds.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/04_build_detection_dataset.py | MODIFICADO | metodologia/manifiestos |
| src/05_build_classification_crops.py | MODIFICADO | metodologia/manifiestos |
| src/06_run_baseline.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/07_run_hpo_study.py | MODIFICADO | arquitecturas/pesos/HPO |
| src/08_repeat_top_configs.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/09_train_outer_candidate.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/10_evaluate_outer_validation.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/11_tune_inference_thresholds.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/12_select_detector.py | MODIFICADO | arquitecturas/pesos/HPO |
| src/13_select_classifier.py | MODIFICADO | arquitecturas/pesos/HPO |
| src/14_validate_integrated_pipeline.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/15_freeze_experiment.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/16_evaluate_test.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/17_benchmark_latency.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/18_export_master_results.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/19_audit_labels.py | MODIFICADO | metodologia/manifiestos |
| src/20_audit_split_evidence.py | MODIFICADO | figuras/documentacion/tests/integridad |
| tests/conftest.py | MODIFICADO | figuras/documentacion/tests/integridad |
| tests/test_classifier_training.py | MODIFICADO | arquitecturas/pesos/HPO |
| tests/test_config_hashing.py | MODIFICADO | figuras/documentacion/tests/integridad |
| tests/test_e2e_pipeline.py | MODIFICADO | figuras/documentacion/tests/integridad |
| tests/test_exchange_safety.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| tests/test_external_graphics.py | MODIFICADO | figuras/documentacion/tests/integridad |
| tests/test_hpo_and_selection.py | MODIFICADO | arquitecturas/pesos/HPO |
| tests/test_semantic_pairing.py | MODIFICADO | metodologia/manifiestos |
| tests/test_training_components.py | MODIFICADO | figuras/documentacion/tests/integridad |
| tools/audit.py | MODIFICADO | figuras/documentacion/tests/integridad |
| tools/configure_safety.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| tools/demo_figures.py | MODIFICADO | figuras/documentacion/tests/integridad |
| tools/extract_stage_features.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| tools/generate_all_figures.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| tools/inspect_model_graph.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| tools/make_synthetic_dataset.py | MODIFICADO | metodologia/manifiestos |
| tools/make_zip.py | MODIFICADO | figuras/documentacion/tests/integridad |
| tools/manage_weights.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | arquitecturas/pesos/HPO |
| tools/manual_gates.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | logging/operacion local |
| tools/normalize_text.py | MODIFICADO | figuras/documentacion/tests/integridad |
| tools/prepare_figure_sources.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| tools/prepare_machine.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| tools/qa.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| tools/qa_checks.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| tools/qa_training_smoke.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| tools/resume_local.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | logging/operacion local |
| tools/smoke_test.py | MODIFICADO | figuras/documentacion/tests/integridad |
| tools/verify_delivery.py | MODIFICADO | figuras/documentacion/tests/integridad |
| docs/historical_pre_v4/ADAPTACION_BOTRYTIS.md | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| docs/historical_pre_v4/AUDIT_REPORT.md | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| docs/historical_pre_v4/GUIA_EJECUCION_3PCs.md | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| docs/historical_pre_v4/README.md | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| docs/historical_pre_v4/SEPARACION_Y_EVIDENCIAS.md | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| docs/historical_pre_v4/test_external_graphics.py.txt | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| src/orchestration/exchange.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | logging/operacion local |
| src/orchestration/launch.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | logging/operacion local |
| src/orchestration/machine.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | logging/operacion local |
| src/orchestration/pc_registry.py | MODIFICADO | logging/operacion local |
| src/orchestration/sync.py | MODIFICADO | logging/operacion local |
| src/orchestration/worker.py | MODIFICADO | logging/operacion local |
| src/utils/bootstrap.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | logging/operacion local |
| src/utils/boxes.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/utils/classifier_models.py | MODIFICADO | arquitecturas/pesos/HPO |
| src/utils/classifier_train.py | MODIFICADO | arquitecturas/pesos/HPO |
| src/utils/config_loader.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/utils/detector_train.py | MODIFICADO | arquitecturas/pesos/HPO |
| src/utils/durable.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| src/utils/evidence.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/utils/figures_all.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| src/utils/figure_catalog.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/utils/guards.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/utils/hashing.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/utils/hpo_results.py | MODIFICADO | arquitecturas/pesos/HPO |
| src/utils/hpo_runner.py | MODIFICADO | arquitecturas/pesos/HPO |
| src/utils/io.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/utils/logging_utils.py | MODIFICADO | logging/operacion local |
| src/utils/manual_gates.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | logging/operacion local |
| src/utils/metrics.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/utils/optuna_storage.py | MODIFICADO | arquitecturas/pesos/HPO |
| src/utils/pairing.py | MODIFICADO | metodologia/manifiestos |
| src/utils/pretrained.py | MODIFICADO | arquitecturas/pesos/HPO |
| src/utils/reporting.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/utils/resume.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | logging/operacion local |
| src/utils/safety.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | figuras/documentacion/tests/integridad |
| src/utils/scientific_audit.py | MODIFICADO | figuras/documentacion/tests/integridad |
| src/utils/semantics.py | MODIFICADO | metodologia/manifiestos |
| src/utils/splits.py | MODIFICADO | metodologia/manifiestos |
| src/utils/weights.py | AGREGADO_O_RESTAURADO_DESDE_ZIP | arquitecturas/pesos/HPO |

## Busqueda final

Ver evidencia_desarrollo/SEMANTIC_SCAN_FINAL.json. Las apariciones numericas legitimas son AP@IoU=.75, cuantiles,
augmentation, geometria/estilo y fixtures. label_rule solo rechaza configuraciones obsoletas. Los nombres/reglas
previos en historicos se conservan como OBSOLETOS; no generan etiquetas ni resultados nuevos.
