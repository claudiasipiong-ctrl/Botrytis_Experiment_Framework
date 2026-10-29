# Ejecución del framework adaptado en tres PCs

Use el mismo ZIP adaptado, configuración `configs/adaptado.yaml`, dataset y versiones de dependencias en las tres PCs. Verifique el SHA-256 del ZIP antes de extraerlo. La cola SQLite y los estudios Optuna son locales; una arquitectura debe tener un único propietario. No se realizaron instalaciones ni entrenamientos durante esta adaptación.

| PC | Arquitecturas | Responsabilidad adicional |
|---|---|---|
| PC-01 | YOLOv8n, YOLOv8m, YOLO26n, YOLO26s | Preparación, consolidación, freeze, test y latencia |
| PC-02 | YOLOv8l, YOLO26l | Entrenamiento asignado |
| PC-03 | RT-DETR-R50, ResNet50, ViT-B/16, Swin-Tiny | Entrenamiento asignado |

## Preparación en PC-01

Coloque el dataset completo en `data/raw/images/` y `data/raw/labels/`. El framework descubre los archivos; la captura esperada contiene 329 imágenes. La carpeta local auditada en esta entrega tiene 328 y 17 labels sin imagen: debe reconciliarse antes de la ejecución oficial. No borre labels ni cambie el control para ocultar la discrepancia.

Clases fijadas: 0 `arandano_a` (fruto físico), 1 `arandano_sospechoso` (positividad por pairing), 2 `signo_sospechoso_botrytis` (auxiliar). No edite esos IDs. `configs/adaptado.yaml` predeclara la seed externa 20261003. El pairing usa IoU mínimo 0.50 y aborta ante anomalías pendientes.

```powershell
python src/19_audit_labels.py --dataset ../dataset-botrytis_2026-10-01.zip --output reports/label_audit
python src/00_check_environment.py --config configs/adaptado.yaml --pc-id PC-01 --require-ml --require-gpu --gpu-selftest
python src/01_validate_dataset.py --config configs/adaptado.yaml --pc-id PC-01
python src/02_create_outer_split.py --config configs/adaptado.yaml --pc-id PC-01
python src/03_create_inner_folds.py --config configs/adaptado.yaml --pc-id PC-01
python src/04_build_detection_dataset.py --config configs/adaptado.yaml --pc-id PC-01 --clean
python src/05_build_classification_crops.py --config configs/adaptado.yaml --pc-id PC-01 --clean
python src/00_check_environment.py --config configs/adaptado.yaml --pc-id PC-01 --write-reference
python src/orchestration/sync.py pack-base --config configs/adaptado.yaml --pc-id PC-01 --out ../sync_001_base.zip
Get-FileHash ../sync_001_base.zip -Algorithm SHA256
```

01 debe aprobar antes de continuar. 02 fija el split por imagen 70/20/10 antes de crear crops; 03 crea folds exclusivamente en train. 04 exporta clase única `fruit`; 05 recorta sólo cajas clase 0 y etiqueta por pairing. Si hay anomalías, revise los CSV y corrija las anotaciones en una nueva versión del dataset o declare explícitamente `pairing.anomaly_policy: quarantine_images`, justificando las exclusiones. Cambiar esa política invalida todos los derivados.

## Recibir la base en PC-02 y PC-03

Compruebe el hash y cambie PC-02 por PC-03 donde corresponda:

```powershell
python src/orchestration/sync.py verify --config configs/adaptado.yaml --pc-id PC-02 --zip ../sync_001_base.zip
python src/orchestration/sync.py unpack --config configs/adaptado.yaml --pc-id PC-02 --zip ../sync_001_base.zip --force
python src/01_validate_dataset.py --config configs/adaptado.yaml --pc-id PC-02
python src/04_build_detection_dataset.py --config configs/adaptado.yaml --pc-id PC-02 --clean
python src/05_build_classification_crops.py --config configs/adaptado.yaml --pc-id PC-02 --clean
python src/00_check_environment.py --config configs/adaptado.yaml --pc-id PC-02 --require-ml --require-gpu --gpu-selftest
Get-FileHash artifacts/classification_manifest.csv -Algorithm SHA256
```

No regenere 02/03 en las PCs receptoras. Dataset, split, folds, manifest y fingerprint deben coincidir. Un cambio de código/configuración se distribuye a todas las PCs antes de continuar.

## Workers

Cada PC inicializa su plan local. Primero inspeccione con `--dry-run`; después ejecute un worker por PC:

```powershell
python src/orchestration/worker.py --config configs/adaptado.yaml --pc-id PC-01 --init-plan --dry-run
python src/orchestration/worker.py --config configs/adaptado.yaml --pc-id PC-01 --init-plan --wait --architectures YOLOv8n YOLOv8m YOLO26n YOLO26s
python src/orchestration/worker.py --config configs/adaptado.yaml --pc-id PC-02 --init-plan --wait --architectures YOLOv8l YOLO26l
python src/orchestration/worker.py --config configs/adaptado.yaml --pc-id PC-03 --init-plan --wait --architectures RT-DETR-R50 ResNet50 'ViT-B/16' Swin-Tiny
python src/orchestration/worker.py --config configs/adaptado.yaml --pc-id PC-01 --status
```

Se conservan baseline, HPO, repetición con seeds 17/42/73, entrenamiento outer y evaluación/ajuste con validation. Test permanece bloqueado. Weighted loss y sampling afectan exclusivamente train. Tras un fallo revise el log antes de `--retry-failed`; para recuperar huérfanos compruebe primero que el proceso anterior terminó y revise `--force-recover --dry-run`.

## Consolidación y cierre

Empaquete resultados en PC-02/03 con checkpoints, verifique SHA-256 y desempaquete en PC-01. La cola local no se sobrescribe:

```powershell
python src/orchestration/sync.py pack-results --config configs/adaptado.yaml --pc-id PC-02 --with-checkpoints --out ../sync_results_PC-02.zip
python src/orchestration/sync.py verify --config configs/adaptado.yaml --pc-id PC-01 --zip ../sync_results_PC-02.zip
python src/orchestration/sync.py unpack --config configs/adaptado.yaml --pc-id PC-01 --zip ../sync_results_PC-02.zip
python src/orchestration/sync.py unpack --config configs/adaptado.yaml --pc-id PC-01 --zip ../sync_results_PC-03.zip
python src/12_select_detector.py --config configs/adaptado.yaml --pc-id PC-01
python src/13_select_classifier.py --config configs/adaptado.yaml --pc-id PC-01
python src/14_validate_integrated_pipeline.py --config configs/adaptado.yaml --pc-id PC-01 --retune-classifier-threshold
```

Revise resultados y decisiones metodológicas existentes antes del freeze. El guard original `decisions.confirmed_by_user` exige una confirmación explícita del responsable experimental; esta adaptación no la declara en su nombre. Una vez confirmadas en la configuración compartida:

```powershell
python src/15_freeze_experiment.py --config configs/adaptado.yaml --pc-id PC-01
python src/16_evaluate_test.py --config configs/adaptado.yaml --pc-id PC-01
python src/17_benchmark_latency.py --config configs/adaptado.yaml --pc-id PC-01
python src/18_export_master_results.py --config configs/adaptado.yaml --pc-id PC-01
python ../generar_graficos_botrytis.py --project-root . --dataset data/raw --config configs/adaptado.yaml --output reports/graficos
```

Después del freeze no cambie código, configuración ni datos. Test se evalúa una sola vez. La selección del detector prioriza mAP@0.50:0.95. La descomposición del error global contabiliza misses y errores de clasificación entre matches válidos; FP y duplicados se reportan aparte. Ejecute el benchmark sin procesos GPU ajenos.
