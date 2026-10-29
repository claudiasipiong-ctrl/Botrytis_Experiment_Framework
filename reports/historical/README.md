# Botrytis Experiment Framework — versión FINAL

Framework experimental **ejecutable y auditable** del modelo híbrido *detección de frutos → clasificación
`con_signos` / `sin_signos`* de arándanos con signos compatibles con *Botrytis cinerea*.
Implementa la metodología de `Especificacion_programacion_experimentacion_Botrytis_s2.md` (*Spec*) y la
estrategia de `Estrategia_entrenamiento_3PCs_Botrytis.md` (*Estrategia*): **scripts 00–18, orquestación de 3 PCs,
HPO con Optuna por arquitectura, benchmark, freeze, test único y auditoría científica automática**.

Resultado de integrar y auditar `botrytis_experiment_base.zip` (Cursor: módulos 00–05 + utilidades + 96 tests) con los
módulos que faltaban (06–18, `orchestration/`, entrenamiento/HPO, figuras de experimentación). Ver
[`AUDIT_REPORT.md`](AUDIT_REPORT.md) para los problemas encontrados, las correcciones, las decisiones técnicas y las pruebas.

> **Qué NO contiene:** resultados científicos. El dataset real (328 imágenes) no estaba disponible; todo se verificó con un
> dataset **sintético** (`tools/make_synthetic_dataset.py`) y el perfil `configs/smoke.yaml`. Los números de esas pruebas no
> significan nada; sirven para demostrar que la cadena completa funciona y respeta los controles metodológicos.

---

## 1. Estructura

```
Botrytis_Experiment_Framework/
├── configs/
│   ├── default.yaml          # configuración consolidada (Spec §32): FIJO / RECOMENDADO / ANEXO / TODO
│   ├── search_spaces.yaml    # espacios HPO versionados (Spec §15): yolov8, yolo26, rtdetr, resnet50, vit, swin
│   └── smoke.yaml            # perfil de PRUEBA DE HUMO (presupuestos diminutos; nunca resultados científicos)
├── src/
│   ├── 00_check_environment.py        ├── 10_evaluate_outer_validation.py
│   ├── 01_validate_dataset.py         ├── 11_tune_inference_thresholds.py
│   ├── 02_create_outer_split.py       ├── 12_select_detector.py
│   ├── 03_create_inner_folds.py       ├── 13_select_classifier.py
│   ├── 04_build_detection_dataset.py  ├── 14_validate_integrated_pipeline.py
│   ├── 05_build_classification_crops.py ├── 15_freeze_experiment.py
│   ├── 06_run_baseline.py             ├── 16_evaluate_test.py
│   ├── 07_run_hpo_study.py            ├── 17_benchmark_latency.py
│   ├── 08_repeat_top_configs.py       └── 18_export_master_results.py
│   ├── 09_train_outer_candidate.py
│   ├── orchestration/  worker.py  study_queue.py  pc_registry.py  sync.py
│   └── utils/  (boxes, labels, folds, splits, checks, guards, freeze, selection, architectures, search_spaces,
│                optuna_storage, hpo_runner, classifier_{models,data,train}, detector_{data,train}, pretrained, inference,
│                pipeline_eval, gradcam, metrics{,_detection,_classification}, model_stats, timing, hashing, io,
│                reproducibility, logging_utils, config_loader, visualization, plots_experiment, reporting,
│                figure_catalog, scientific_audit, ...)
├── tests/   tools/   requirements.txt   requirements-dev.txt   AUDIT_REPORT.md   README.md
├── data/ runs/ artifacts/ reports/        # se generan al ejecutar (no se versionan)
```

Modelos: **detectores** YOLOv8n/m/l, YOLO26n/s/l, RT-DETR-R50 (Ultralytics) · **clasificadores** ResNet50 (torchvision),
ViT-B/16 y Swin-Tiny (timm) · **HPO** Optuna (TPE multivariate+group, `MedianPruner`). U-Net **no** se programa (Spec §2.3).

## 2. Instalación (las 3 PCs con versiones idénticas)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126   # wheel CUDA de SU driver
pip install -r requirements.txt
pip install -r requirements-dev.txt                       # opcional: tests y auditoría
python src/00_check_environment.py --pc-id PC-01 --require-gpu --require-ml --gpu-selftest
```

Probado en Windows 10 + Python 3.12 + CUDA 12.6. Use una ruta de proyecto **sin caracteres especiales** si puede (las rutas con
acentos funcionan: las listas de imágenes de Ultralytics se escriben con la codificación con la que las leerá el proceso).
Todas las rutas son relativas a la raíz del proyecto y se redefinen por `--set paths.<clave>=...` o variables de entorno
(`BOTRYTIS_DATA_ROOT`, `BOTRYTIS_IMAGES_DIR`, `BOTRYTIS_LABELS_DIR`, `BOTRYTIS_RUNS_DIR`, `BOTRYTIS_PC_ID`, `OPTUNA_DATABASE_URL`).

## 3. Decisiones: qué está fijo, qué debe rellenar usted

| Tipo | Clave (`configs/default.yaml`) | Estado |
|---|---|---|
| **TODO BLOQUEANTE** (no se puede inferir; el código aborta) | `classes.fruit_class_id`, `classes.sign_class_id`, `outer_split.seed` | **Rellénelos usted.** El `Dataset_Botrytis.zip` del escritorio (75 imágenes, una sola clase `signo_sospechoso_botrytis`) **no** es el dataset final de 328 imágenes con 2 clases; los IDs no pueden deducirse de él. La seed del split debe predeclararse **antes** de ver resultados (Spec §5.3). |
| **FIJO** (validado al cargar; alterarlo lanza `FixedDecisionViolation`) | regla `> 0.75` con denominador = área del fruto, split 70/20/10 por imagen, test bloqueado, 328 imágenes, clase positiva `con_signos`, 1 trabajo GPU por PC / 3 globales, `image_size` 640/224, batch 16/32, 15/20 trials, `pretrained`, fp16, seeds `[17,42,73]`, top-3, TPE + MedianPruner, `n_jobs=1` por estudio | `project.profile: official` |
| **ANEXO → a confirmar** (los TODO de la Spec que los anexos propusieron; están en `default.yaml`, sección `decisions`) | patience clasificador 15 · métrica del umbral F1 · matching IoU ≥ 0.5 one-to-one por confianza (duplicados = FP de localización) · tie-breakers (recall → GFLOPs / F1) · benchmark 10 warm-up + 100 medidas, sync CUDA, sin I/O de disco · `weighted_ce` ante desbalance | Se usan por defecto. **`15_freeze_experiment.py` exige `decisions.confirmed_by_user: true`**: usted confirma por escrito antes de congelar. |

Las decisiones del anexo **no** son "óptimos demostrados": son recomendaciones derivadas y quedan registradas como tales en el
`freeze_manifest.json` (`decisions_adopted_from_annex`).

## 4. Ejecución (orden exacto de la Spec §36)

### Fase A — preparación (una PC, p. ej. PC-01)

```bash
# rellene antes en configs/default.yaml:  classes.fruit_class_id, classes.sign_class_id, outer_split.seed, paths.raw_*_dir
python src/01_validate_dataset.py                 # 328 imágenes + anotaciones YOLO -> dataset_hash
python src/02_create_outer_split.py               # 70/20/10 por imagen, UNA vez; manifiesto sellado; test bloqueado
python src/03_create_inner_folds.py               # 3 folds internos, solo outer train
python src/04_build_detection_dataset.py          # 1 clase (fruit); las cajas de signo NO son targets
python src/05_build_classification_crops.py       # regla GT intersección/área(fruto) > 0.75; crops GT; 8 figuras de dataset
python src/00_check_environment.py --pc-id PC-01 --write-reference
python src/orchestration/sync.py pack-base --out sync_001_base.zip      # dataset crudo + configs + src + manifiestos + hashes
```

Suba `sync_001_base.zip` (+ `.zip.sha256`) a Drive. En **PC-02 y PC-03**: descomprima, repita `04`/`05` (deterministas; el SHA-256
de `classification_manifest` debe coincidir con `hashes/hashes.json`) y ejecute `00_check_environment.py --pc-id PC-0N`
(aborta si hardware/versiones/hashes difieren de la referencia).

### Fases B–E — baseline, HPO, robustez, comparación externa (cola de trabajos, 3 PCs)

```bash
# una terminal por PC; reparto ESTATICO (cada PC tiene su cola LOCAL: artifacts/queue/queue.sqlite)
python src/orchestration/worker.py --pc-id PC-01 --init-plan --wait --architectures yolov8n yolov8m yolo26n yolo26s
python src/orchestration/worker.py --pc-id PC-02 --init-plan --wait --architectures yolov8l yolo26l
python src/orchestration/worker.py --pc-id PC-03 --init-plan --wait --architectures rtdetr_r50 resnet50 vit_b16 swin_tiny
```

Cada worker toma de SU cola local el siguiente trabajo **listo** y lo ejecuta en un subproceso (**un trabajo GPU por PC**):
`baseline` (06) → `hpo` (07) → `repeat` (08: top-3 × seeds 17/42/73) → `aggregate` (08) → `outer_train` (09) →
`outer_eval` (10) → `thresholds` (11). El reparto de `pc_registry.first_batch` (PC-01 YOLOv8n/m + YOLO26n/s · PC-02 YOLOv8l/YOLO26l ·
PC-03 RT-DETR-R50/ResNet50/ViT-B/16/Swin-Tiny) es **estático**: como no hay servidor central, las colas no se comparten entre PCs, y
`--architectures` evita que dos PCs entrenen la misma arquitectura. Reasignar una arquitectura a otra PC es manual (ver la guía §8).
**Paralelismo entre arquitecturas, nunca tres trials del mismo estudio.**
Estados de job: `PENDING / RUNNING / COMPLETE / FAILED` (los trials podados se registran como `PRUNED`).

> **Guía paso a paso (VS Code) para las 3 PCs: [`GUIA_EJECUCION_3PCs.md`](GUIA_EJECUCION_3PCs.md).** El proyecto incluye
> `.vscode/tasks.json` con una tarea por paso (Ctrl+Shift+P → *Tasks: Run Task*).

```bash
python src/orchestration/worker.py --pc-id PC-01 --status                # tabla de la cola
python src/orchestration/worker.py --pc-id PC-01 --retry-failed          # FAILED -> PENDING
python src/orchestration/worker.py --pc-id PC-01 --recover-stale         # RUNNING sin latido -> PENDING (PC caída)
python src/orchestration/worker.py --pc-id PC-01 --force-recover         # RUNNING de ESTA PC -> PENDING (tras un corte; verifique que no hay procesos vivos)
```

Cada script también se puede lanzar a mano (p. ej. `python src/07_run_hpo_study.py --architecture ResNet50 --pc-id PC-03`).

**Sincronización sin servidor** (ZIP + Drive, Estrategia §6/§10): al terminar un lote,
`python src/orchestration/sync.py pack-results --pc-id PC-02 --out sync_results_20261001_pc02.zip [--with-checkpoints]`;
en la PC de consolidación: `python src/orchestration/sync.py unpack --pc-id PC-01 --zip sync_results_20261001_pc02.zip`
(una PC nueva se arranca con `pack-base` en PC-01 y `unpack --zip sync_001_base.zip --force` en PC-02/03; la cola local nunca viaja en los ZIP). Se verifican SHA-256 del ZIP, tamaño y
SHA-256 de cada fichero, y que `dataset_hash`/`split_hash` coincidan; un ZIP corrupto **no se usa**; el estado de la cola se fusiona
(gana COMPLETE > PRUNED > FAILED > RUNNING > PENDING; el mismo job RUNNING en dos PCs se avisa como conflicto).

### Fases F–K — selección, pipeline, freeze, test, latencia, exportación (PC de consolidación)

```bash
python src/12_select_detector.py                          # politica de seleccion predeclarada (artifacts/selection_policy.yaml, inmutable)
python src/13_select_classifier.py
python src/14_validate_integrated_pipeline.py --retune-classifier-threshold   # PREDICTED crops sobre validation externo (obligatorio)
#  >>> confirme decisions.confirmed_by_user: true en configs/default.yaml <<<
python src/15_freeze_experiment.py                        # freeze_manifest.json: desbloquea el test y prohibe cambiar nada
python src/16_evaluate_test.py                            # UNA sola evaluación; test_results.json es de escritura única
python src/17_benchmark_latency.py --pc-id PC-01          # solo BENCHMARK_PC, GPU libre
python src/18_export_master_results.py                    # master_results.csv + tablas + figuras + auditoría científica (exit 1 si falla)
```

`16_evaluate_test.py --reproduce` recalcula y **compara** con los resultados oficiales sin sobrescribirlos.

## 5. Qué produce cada ejecución (logging absoluto)

```
runs/
  _pipeline/<YYYYmmdd_HHMMSS>_<script>/        # scripts que no pertenecen a una arquitectura (00-05, 12-18, worker, sync)
  <architecture>/                              # yolov8n, yolov8m, ..., rtdetr_r50, resnet50, vit_b16, swin_tiny
    baseline/
    hpo/  trial_000/ … trial_014/  study_run/  study_summary.json  trials.csv  intermediate_values.csv
    robustness/  <config_id>/seed_17|42|73/  aggregate/  summary.json  summary.csv  per_seed.csv
    final/           outer_validation/           thresholds/
```

**Cada run** (incluidos los fallidos y los podados) contiene siempre:

| Fichero | Contenido |
|---|---|
| `execution.log` | log completo con *tracebacks* |
| `metrics.json` | métricas acumuladas (hiperparámetros, scores por fold, objetivo, tiempo, **VRAM**, **FLOPs**, **parámetros**) |
| `config.yaml` | configuración efectiva (+ bloque `trial`: hiperparámetros, seed, folds, n.º de trial) |
| `environment.json` | PC, GPU/driver/CUDA/cuDNN, versiones de todos los paquetes, git, hashes |
| `checkpoint_metadata.json` | checkpoints (ruta, SHA-256, tamaño; los de fold se descartan y se guarda su hash) + **dataset/code/environment/split hash** |
| `metadata.json` | estado (`COMPLETE/FAILED/PRUNED/INTERRUPTED`), inicio/fin, PC, seed, parámetros CLI, **splits accedidos**, error completo |
| `figures/` | figuras del run + `<figura>.data.csv` (datos dibujados) + `<figura>.provenance.json` (fuentes con SHA-256, `n_points`, `synthetic:false`) |

Los 4 hashes obligatorios (`dataset_hash`, `code_hash`, `environment_hash`, `split_hash`) se guardan **siempre**. `metadata.json` se escribe al
inicio con `RUNNING` y se reescribe al final, de modo que un corte abrupto también deja rastro.

Artefactos globales en `artifacts/`: `split_manifest.csv/.sha256`, `inner_folds_*.csv`, `classification_manifest.csv`, `optuna/botrytis_<slug>.db`
(un SQLite por estudio), `best_configs/<slug>.json`, `checkpoints/<slug>/final.pt`, `outer_validation/`, `inference_thresholds/`, `selection_policy.yaml`,
`selected_{detector,classifier}.json`, `final_thresholds.json`, `pipeline_validation.json`, `freeze_manifest.json`, `test_results.json`,
`benchmark/`, `master_results.csv`, `provenance.json`, `queue/queue.sqlite`.

## 6. Gráficos obligatorios (datos reales; ver `src/utils/figure_catalog.py`)

| Grupo | Figura (`reports/figures/…`) | La genera |
|---|---|---|
| Detección | `det_map50_95_por_arquitectura`, `det_map50_por_arquitectura`, `det_radar_detectores`, `det_ap_vs_iou`, `det_map_vs_flops`, `det_map_vs_parametros` | 12 y 18 |
| | `det_recall_vs_latencia`, `det_pareto_map_latencia`, `det_pareto_map_vram` | 18 (requiere 17) |
| | `hpo_curvas_<slug>`, `hpo_distribuciones_<slug>` (10 arquitecturas: curvas HPO, distribuciones y fANOVA de Optuna) | 07 y 18 |
| Clasificación | `cls_curva_pr`, `cls_curva_roc`, `cls_matriz_confusion`, `cls_gradcam`, `cls_umbral_vs_f1`, `cls_ap_positiva_por_arquitectura` | 13 (y 10/11 por arquitectura) |
| Pipeline | `pipe_arquitectura_completa_{validacion,test}`, `pipe_errores_acumulados_*`, `pipe_tp_fp_fn_*` (+ `_conteos_*`), `pipe_matriz_errores_*`, `pipe_mapa_espacial_errores_*` | 14 y 16 |
| | `pipe_latencia_por_etapa` | 17 |
| Extra | dataset (8 figuras), curvas de aprendizaje/loss por epoca, robustez, baseline vs HPO, hiperparámetros ganadores, eficiencia, tiempo y VRAM de entrenamiento | 02–09, 18 |

`18_export_master_results.py` **verifica** que cada figura obligatoria exista, tenga `n_points > 0`, `synthetic: false` y fuentes con SHA-256;
las figuras de `tools/demo_figures.py` (datos sintéticos, prefijo `DEMO_`) jamás cuentan: su presencia en `reports/figures` falla la auditoría.

## 7. Controles metodológicos que hacen fallar el programa (Spec §35) — dónde viven

| Control | Implementación |
|---|---|
| imagen en >1 split · crop con split distinto · crops de una imagen en folds distintos | `utils/checks.py`, `utils/prepared.py` (se re-verifican al cargar) |
| test usado en HPO / selección / umbrales · validation externo dentro de trials | `utils.splits.assert_split_allowed` (todas las lecturas de datos pasan por ahí; `ctx.note_split_access` deja evidencia) |
| cajas de signo como targets del detector | `assert_detection_labels_fruit_only` + auditoría 3 |
| clasificador con predicted crops | sin imports de inferencia en `classifier_*`; `train_source: ground_truth_crops` + SHA del manifiesto en cada run; auditoría 1 |
| augmentation aleatoria en validation/test | `assert_deterministic_transform` |
| `area(signo)` como denominador, `>= 0.75`, IoU o suma de coberturas | `config_loader` (FIJO), `assert_label_rule_consistent`, auditoría 2 (recalcula etiquetas desde anotaciones crudas) |
| presupuesto HPO desigual · pretrained mixto · `image_size` libre | `utils/guards.py` + validación FIJA del perfil `official` |
| selección por un único seed | `assert_one_selected_by_mean` (8) |
| 2 entrenamientos GPU simultáneos en una PC | `GpuJobLock` (un lock por PC, recuperable si el proceso murió) + cola |
| medir latencia mientras la GPU entrena · fuera de BENCHMARK_PC | `17_benchmark_latency.py` (`nvidia-smi` + cola + `assert_benchmark_pc`) |
| modificar algo tras `freeze_manifest.json` | `assert_not_frozen` en 06–14; `16` verifica sello, hashes de código/datos/split, checkpoints, umbrales y versiones |
| reevaluar el test | `test_results.json` de escritura única; `--reproduce` no sobrescribe |

## 8. Reproducibilidad

* **Hashes** portables Windows/Linux (`\n`, saltos de línea normalizados en texto): `dataset_hash`, `split_hash`, `code_hash`
  (`src/**/*.py`, `configs/*.yaml`, `requirements*.txt`), `environment_hash`, SHA-256 de checkpoints, manifiestos, figuras y fuentes.
* **Seeds**: seed global 42 (scripts, baseline, HPO, entrenamiento definitivo), seeds de robustez `[17, 42, 73]`, seed del split = `outer_split.seed`
  (usted). `DataLoader` y workers sembrados; `torch.use_deterministic_algorithms(warn_only=True)`, `cudnn.deterministic`.
* **Métricas recomputables**: la auditoría recalcula AP/ROC/F1 del clasificador, el mAP independiente del detector y los errores del pipeline
  desde `*_predictions.csv` / `pipeline_*_detections.csv` (sin modelos). La evaluación oficial del detector es la de Ultralytics (la misma del objetivo del
  HPO); se contrasta con una implementación propia (mAP COCO-101) y la diferencia queda registrada.
* **Epocas definitivas**: el entrenamiento con el 70 % usa las épocas fijadas por los folds internos, **sin** early stopping ni elegir época con el validation externo.

## 9. Verificación

```bash
python tools/audit.py --with-tests                     # estática (sintaxis, imports, --help, type hints, pyflakes, LF/UTF-8, config) + pytest
python -m pytest -q                                    # 176 tests (~10 min; construye datasets sintéticos)
python tools/smoke_test.py --workdir C:/tmp/botrytis_smoke --n-images 120 --workers 3
        # 00 -> 18 de extremo a extremo con configs/smoke.yaml y 3 "PCs" (3 workers) sobre un dataset sintético
```

`tools/smoke_test.py` ejecuta **todas** las etapas con presupuestos diminutos (3 trials, 3 épocas, imagen 320 para detectores) y termina con la
auditoría científica de `18_export_master_results.py`. Los resultados del perfil `smoke` se marcan (`project.profile: smoke`) y **no son científicos**.

## 10. Notas técnicas y limitaciones

* **RT-DETR-R50**: Ultralytics no publica `rtdetr-resnet50.pt` (solo `-l`/`-x`). Se usa `rtdetr-resnet50.yaml` con el backbone ResNet50 inicializado con **ImageNet**
  (torchvision `IMAGENET1K_V2`, 318 tensores trasplantados y verificados); encoder/decoder parten de cero. `backbone_lr_ratio = 0.1` se implementa separando los
  grupos de parámetros del backbone (probado que el planificador los actualiza). Espere que RT-DETR rinda peor con 230 imágenes: es un resultado, no un error.
* **Optuna storage**: SQLite local, un fichero por estudio (Estrategia §12 descarta el servidor central); `optuna.storage.backend: postgresql` + `OPTUNA_DATABASE_URL`
  está soportado pero no se probó contra un servidor real.
* **Batch efectivo**: micro-batch 16 (detector) / 32 (clasificador) con `nbs=64` de Ultralytics; ante OOM se reduce el micro-batch (y se acumula en el clasificador) conservando el efectivo.
* **Poda**: paso global = `fold × max_epochs + época`; el `MedianPruner` (5 trials de arranque, 8 pasos de calentamiento) compara la mejor métrica del trial con la mediana de los demás en ese paso.
  Si < 3 trials quedan COMPLETE, el top-k se reduce (la auditoría lo marca `WARN`).
* **Tie-breakers sin latencia**: la selección se hace antes del benchmark (Spec §36), por lo que usa GFLOPs, no latencia.
* **Linux/macOS**: los hashes y la base se verificaron en Windows y Ubuntu; los módulos 06–18 solo se ejecutaron en Windows 10.
* **3 PCs físicas**: la orquestación se probó con 3 workers simultáneos (PC-01/02/03) en una sola máquina y una sola GPU.
