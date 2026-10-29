# Actualización: separación y evidencias por modelo

Consultar [SEPARACION_Y_EVIDENCIAS.md](SEPARACION_Y_EVIDENCIAS.md) antes de ejecutar esta versión. La auditoría actual y las pruebas están en `reports/separation_audit/`. Las evidencias nuevas se guardan en `results/` y el test compara únicamente candidatos sellados antes de abrirlo.

# Botrytis Experiment Framework

Framework Python para comparar siete detectores y tres clasificadores, con split externo por imagen, folds internos, HPO, test bloqueado y ejecución distribuida en tres PCs.

La adaptación terminada, la auditoría y los pendientes se documentan en [ADAPTACION_BOTRYTIS.md](ADAPTACION_BOTRYTIS.md). No se entrenaron modelos ni se descargaron pesos durante esta adaptación.

## Ground truth

| Anotación YOLO | Uso |
|---|---|
| `0 = arandano_a` | Todos los frutos físicos. Único target del detector, remapeado a `0: fruit`. Única fuente de bboxes para crops. |
| `1 = arandano_sospechoso` | Anotación adicional de un fruto. Un pairing válido 1:1 con clase 0 define `con_signos`. |
| `2 = signo_sospechoso_botrytis` | Región visual auxiliar para auditoría, figuras y métricas espaciales Grad-CAM. |

El pairing maximiza la suma de IoU por imagen mediante asignación bipartita determinista. Los empates se resuelven por ordinal; se registran como anomalías. `pairing.min_iou: 0.50` es un umbral técnico de identidad entre anotaciones, sin interpretación biológica. La cobertura signo/fruto es exclusivamente descriptiva.

Un fruto sin clase 1 asociada es `sin_signos` sólo cuando su imagen carece de anomalías pendientes. El perfil oficial aborta ante anomalías; no convierte una anotación sospechosa defectuosa en un negativo. `--set pairing.anomaly_policy=quarantine_images` habilita una exclusión explícita de imágenes completas, registrada en `artifacts/quarantine_manifest.json`, y obliga a regenerar validación/splits/folds/derivados. No se aplica por defecto.

## Protocolo conservado

- Split externo 70/20/10 por imagen original, antes de los crops; todos sus frutos heredan partición y fold.
- Test bloqueado durante HPO, selección y ajuste de umbrales. Folds internos exclusivamente sobre outer train.
- Seeds finales `[17, 42, 73]`; `configs/adaptado.yaml` predeclara seed externa `20261003` antes de resultados de modelos.
- Detectores: YOLOv8n/m/l, YOLO26n/s/l, RT-DETR-R50. Selección principalmente por `mAP@0.50:0.95`.
- Clasificadores: ResNet50, ViT-B/16 y Swin-Tiny. AP positiva como objetivo. Weighted CE y augmentation afectan sólo train; validation/test conservan su distribución.
- Hashes de datos, código, split y entorno; versionado semántico de artefactos; cola y sincronización entre PCs.

## Preparación

Coloque el dataset completo en `data/raw/images/` y `data/raw/labels/`, o configure `BOTRYTIS_IMAGES_DIR` y `BOTRYTIS_LABELS_DIR`. Se descubre el número real de archivos; 329 es el control esperado configurable, no una constante para crear imágenes o splits. Los labels vacíos son válidos como fondo.

```powershell
# Auditoría del ZIP, sin extraerlo y sin imágenes. Exit 2 = anomalías pendientes.
python src/19_audit_labels.py --dataset ../dataset-botrytis_2026-10-01.zip --output reports/label_audit --metadata ../metadatos_repetidos_project-24_iPhone13.txt
# Alias dentro del validador:
python src/01_validate_dataset.py --label-only --dataset ../dataset-botrytis_2026-10-01.zip --output reports/label_audit

# Con dataset completo y anomalías resueltas:
python src/00_check_environment.py --config configs/adaptado.yaml --pc-id PC-01 --write-reference
python src/01_validate_dataset.py --config configs/adaptado.yaml --pc-id PC-01
python src/02_create_outer_split.py --config configs/adaptado.yaml --pc-id PC-01
python src/03_create_inner_folds.py --config configs/adaptado.yaml --pc-id PC-01
python src/04_build_detection_dataset.py --config configs/adaptado.yaml --pc-id PC-01 --clean
python src/05_build_classification_crops.py --config configs/adaptado.yaml --pc-id PC-01 --clean
```

Si cambia dataset, taxonomía, pairing, política de anomalías, semántica, seed o protocolo de split/folds, se rechazan los artefactos anteriores. Tras una modificación deliberada, vuelva a ejecutar 01; use `--force` en 02/03 para sustituir manifiestos incompatibles y `--clean` en 04/05. Conservar el split anterior sólo por coincidir sus asignaciones no está permitido si cambia su fingerprint.

## Entrenamiento y HPO

Los siguientes comandos se entregan para ejecución futura; no se ejecutaron aquí. Requieren el stack ML, dataset validado y los mismos entornos en las tres PCs.

```powershell
python src/06_run_baseline.py --config configs/adaptado.yaml --architecture YOLOv8n --pc-id PC-01
python src/06_run_baseline.py --config configs/adaptado.yaml --architecture ResNet50 --pc-id PC-03
python src/07_run_hpo_study.py --config configs/adaptado.yaml --architecture YOLOv8n --pc-id PC-01
python src/07_run_hpo_study.py --config configs/adaptado.yaml --architecture ResNet50 --pc-id PC-03

# Plan y workers distribuidos; el flag dry-run inspecciona sin entrenar.
python src/orchestration/worker.py --config configs/adaptado.yaml --pc-id PC-01 --init-plan --dry-run
python src/orchestration/worker.py --config configs/adaptado.yaml --pc-id PC-01 --architectures YOLOv8n YOLOv8m YOLO26n YOLO26s --wait
python src/orchestration/worker.py --config configs/adaptado.yaml --pc-id PC-02 --architectures YOLOv8l YOLO26l --wait
python src/orchestration/worker.py --config configs/adaptado.yaml --pc-id PC-03 --architectures RT-DETR-R50 ResNet50 'ViT-B/16' Swin-Tiny --wait
```

Pasos 08–18: repetir configuraciones con seeds finales, entrenar candidatos outer, evaluar validation, ajustar inferencia, seleccionar, validar el pipeline, congelar, evaluar test una vez, medir latencia y exportar resultados. Los scripts conservan sus guards originales y añaden rechazo de derivados incompatibles. Consulte `--help` en cada script y la [guía de tres PCs](GUIA_EJECUCION_3PCs.md).

## Evaluación integrada

Se asocian detecciones con frutos GT clase 0 mediante IoU `>= matching.threshold` (inicial 0.5), confianza descendente y asignación greedy 1:1. En empate se conserva el orden de detección y el ordinal GT. Este matching de predicciones es distinto del pairing de anotaciones.

`N_GT = N_miss + N_match` y `N_cls_error <= N_match`. Se calcula:

```text
E_det      = N_miss / N_GT
E_cls_cond = N_cls_error / N_match
E_cls_abs  = N_cls_error / N_GT
E_global   = (N_miss + N_cls_error) / N_GT
E_global   = E_det + (1 - E_det) * E_cls_cond
```

Las detecciones sin GT y los duplicados se reportan como FP aparte. No se suman al error global. Sin GT las tasas son `NA`; sin matches el error condicional es `NA`, el absoluto es cero y el global coincide con el de detección. Se rechazan tablas que cuentan el mismo fruto o detección dos veces.

Detección: precision, recall, F1, mAP@0.50, mAP@0.50:0.95, FP/FN y latencia. Clasificación: accuracy, precision, recall, F1, AP positiva, ROC-AUC, ROC, matriz de confusión, classification report y CSV/ejemplos FP/FN. Un positivo sin caja de signo sigue siendo positivo y tiene métricas espaciales `NA`.

## Gráficos y validación

El script externo se entrega junto al ZIP, fuera de él. Descubre configs, historiales, manifests y resultados; exporta datos e inventario con fuentes y pendientes.

```powershell
python ../generar_graficos_botrytis.py --help
python ../generar_graficos_botrytis.py --project-root . --dataset ../dataset-botrytis_2026-10-01.zip --config configs/adaptado.yaml --output reports/graficos
python ../generar_graficos_botrytis.py --project-root . --dataset ../dataset-botrytis_2026-10-01 --config configs/adaptado.yaml --output reports/graficos_local
python ../generar_graficos_botrytis.py --project-root . --dataset ../dataset-botrytis_2026-10-01.zip --config configs/adaptado.yaml --output reports/graficos_preview --allow-estimates
python -m pytest --junitxml=reports/tests.xml
python tools/audit.py
python tools/make_zip.py --out ../Botrytis_Experiment_Framework_ADAPTADO.zip
```

`paper/` contiene sólo convergencia de detectores/clasificadores y ejemplos reales del dataset cuando las fuentes lo permiten. Las ejecuciones se agrupan sólo con arquitectura, HP, dataset, split y etapa comparables; la DE es muestral. `estimated_preview/` contiene exclusivamente curvas analíticas de maquetación, marcadas `is_estimated=true`, y nunca se copia a `paper/`. No se inventan imágenes, ROC, matrices ni Grad-CAM. No existe curva de entrenamiento del híbrido.

Los tests de entrenamiento real están desactivados salvo `BOTRYTIS_ALLOW_TRAINING_TESTS=1`. Las dependencias ML ausentes y las comprobaciones que necesitan datos/checkpoints reales se registran `SKIPPED`; las fixtures sintéticas verifican software y nunca constituyen evidencia científica.
