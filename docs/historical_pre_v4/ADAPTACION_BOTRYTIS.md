# Adaptación Botrytis: semántica, auditoría y entrega

Fecha: 2026-10-03. El proyecto se modificó directamente. No se entrenaron modelos, no se descargaron pesos y no se instalaron dependencias pesadas. Se reutilizó Python 3.12 y el stack de análisis disponible.

## Implementación

Sólo `0 = arandano_a` representa frutos físicos y produce targets `0: fruit` o crops. `1 = arandano_sospechoso` es una segunda anotación del mismo fruto y define `con_signos` cuando se acepta su pairing 1:1. `2 = signo_sospechoso_botrytis` es auxiliar y nunca define la etiqueta.

`src/utils/pairing.py` resuelve por imagen una asignación bipartita de máxima suma de IoU entre asociaciones admisibles (IoU ≥ umbral técnico inicial 0.50). Incluye nodos sin pareja; las asociaciones bajo el umbral no compiten por frutos. Un objetivo secundario lexicográfico por ordinal se codifica con enteros exactos sobre los valores IoU de coma flotante, sin epsilon que altere el óptimo primario. Se exportan todos los candidatos, la asignación, el mejor candidato de cada sospechoso, IoU, cajas, estados y motivos. Se detectan duplicados, ordinals repetidos, degeneración, clases desconocidas, empates globales, varios candidatos elegibles y competencia de anotaciones sospechosas por un fruto.

La tabla de frutos asigna `NA` a todos los frutos de una imagen ambigua. La validación oficial y la preparación de crops abortan; no se convierte una anomalía en negativo. La opción `--set pairing.anomaly_policy=quarantine_images` aplica una cuarentena explícita por imagen completa antes del split, exporta `quarantine_manifest.json` y queda incluida en el fingerprint. No se activó sobre el dataset real.

La lógica de cobertura generadora de etiquetas se eliminó de funciones, asserts, configuración, pruebas, manifests, diagramas y documentación. La cobertura auxiliar permanece como estadística descriptiva. Se preservaron valores legítimos como mAP@0.75, cuantiles y augmentation.

## Auditoría exacta del ZIP suministrado

Fuente: `../dataset-botrytis_2026-10-01.zip`. CRC verificado. Auditoría sin extraer imágenes ni sustituirlas por metadata histórica.

| Concepto | Resultado |
|---|---:|
| Imágenes en el ZIP | 0 |
| Labels | 345 |
| Labels vacíos | 54 |
| Cajas clase 0 | 1,941 |
| Cajas clase 1 | 225 |
| Cajas clase 2 | 261 |
| Coordenadas YOLO inválidas / cajas degeneradas | 0 / 0 |
| Clases desconocidas | 0 |
| Pares candidatos IoU completos | 1,316 |
| Pares válidos como decisiones individuales | 218 |
| Incidencias de anomalía / imágenes afectadas | 11 / 7 |
| Cajas duplicadas exactas, por clase e imagen | 4 |
| Frutos de imágenes afectadas | 42 |
| Positivos inequívocos en imágenes sin anomalías | 203 |
| Negativos inequívocos en imágenes sin anomalías | 1,696 |
| Crops reales generados | 0 |

Los 218 pares individuales no son 218 crops autorizados: algunas imágenes tienen otra anomalía. Los 203/1,696 son posibilidades label-only de imágenes limpias, incluyendo labels sin imagen disponible; no constituyen una exclusión aplicada ni un dataset experimental aprobado. Los 42 frutos pendientes conservan etiqueta indeterminada.

Distribución IoU: 1,316 candidatos, media 0.180813635966, mediana 0, p75 0.049414868630, p95 1. Sobre las 225 decisiones sospechosas, media 0.980334743642, mediana 1, p05 0.988402008403, mínimo 0 y máximo 1. Todos los valores están exportados; no se seleccionó un umbral para ocultar casos anómalos. El mejor candidato y la asignación global pueden diferir por competencia 1:1.

Fuentes terminadas: `reports/label_audit/label_audit.json`, `label_audit.md`, `boxes.csv`, `pairing_audit.csv`, `pairing_candidates.csv`, `label_anomalies.csv`, `label_per_image.csv`.

### Anomalías sin corrección automática

| Imagen | Hallazgo |
|---|---|
| `213f3bad-IMG_6979` | Un sospechoso no solapa ningún fruto; IoU máximo 0. |
| `2d515215-IMG_6819` | Un sospechoso no solapa ningún fruto; IoU máximo 0. |
| `7c5a8c55-IMG_6823` | Clase 1 duplicada, competencia por el mismo fruto, empate global y una anotación sin asignación. |
| `a8bbe159-IMG_6969` | Mejor IoU de un sospechoso 0.148597155938; candidato ya ocupado por su propia anotación clase 1 exacta. |
| `04a501fd-IMG_6797` | Clase 0 duplicada. |
| `d928c2d0-IMG_6622` | Clase 0 duplicada, varios candidatos elegibles y empate global. |
| `fc4ef1be-IMG_6986` | Clase 0 duplicada, varios candidatos elegibles y empate global; además falta su imagen en la carpeta local. |

Los CSV detallan ordinals de base cero y número de línea de las anotaciones sospechosas. Son incidencias, no once archivos ni once frutos nuevos.

## Contraste con la carpeta local descubierta

Además del ZIP, se encontró `../dataset-botrytis_2026-10-01/images/` con **328** archivos. El dataset completo solicitado debería descubrir **329**. La captura histórica del TXT también dice 328; no demuestra que esa carpeta esté completa. No se eliminó, renombró ni inventó ningún archivo del dataset.

La carpeta local tiene los mismos 345 labels y las mismas cajas por clase que el ZIP; la comparación por stem y SHA-256 se registra en el cierre. Hay **0 imágenes sin label** y **17 labels sin imagen**. Los labels huérfanos contienen 92 cajas clase 0, 26 clase 1 y 2 clase 2. Sus stems completos están en `reports/local_dataset_crosscheck/label_audit.json`.

Se intentó validación de píxeles, EXIF, hashes y duplicados sobre las 328 imágenes locales. Resultado bloqueante esperado: **2 errores** (conteo 328 frente a 329 y anomalías de anotación) y **74 warnings** (54 labels vacíos, 3 duplicados con imagen, 17 labels huérfanos). No se detectaron imágenes corruptas, duplicados de contenido ni casi duplicados con el criterio dHash configurado. En estas imágenes existen 1,849 frutos clase 0, 199 sospechosos y 259 signos. Los resultados no autorizan entrenamiento.

En imágenes locales sin anomalías son posibles 181 crops positivos y 1,633 negativos; no se generaron crops reales. El informe íntegro local permanece en `reports/local_image_validation/`; su resumen portable está en `reports/local_image_validation_summary.json`. Las capturas y trazas pesadas locales se excluyen del ZIP.

## Reproducibilidad y bloqueo de artefactos

Versión: `botrytis-physical-fruit-pairing-v3`. El fingerprint incluye clases, nombres, semántica, parámetros/política de pairing, labels binarios y configuración del split/folds. El dataset hash incluye contenido de imágenes y **todos** los labels, incluidos huérfanos y vacíos. Cambiar un archivo o cualquiera de esas decisiones obliga a regenerar.

Validación, split, folds, datasets derivados y freeze verifican compatibilidad. Detección y clasificación exigen `semantic_stamp.json` con hashes de dataset/split. Los loaders de entrenamiento rechazan derivados sin stamp o incompatibles. La sincronización de las tres PCs incluye CSV de pairing y manifest de cuarentena. Los informes smoke anteriores fueron eliminados.

El split por imagen 70/20/10 sigue precediendo a cualquier crop; folds internos usan sólo outer train. Validation/test no usan sampling ni weighted loss. Se conservaron arquitecturas, objetivos de selección, HPO, seeds finales, test bloqueado y reparto de tres PCs. `configs/adaptado.yaml` declara seed externa 20261003 antes de resultados de entrenamiento; `default.yaml` conserva un TODO de seed para quien no use esa configuración.

## Evaluación y Grad-CAM

`N_GT` cuenta frutos físicos clase 0. El matching detección↔GT usa IoU ≥0.5 inicial, confianza descendente, orden estable y greedy 1:1. `N_GT = N_miss + N_match`, `N_cls_error <= N_match`. Las tablas repetidas o con un GT simultáneamente missed/matched se rechazan.

`E_det=N_miss/N_GT`, `E_cls_cond=N_cls_error/N_match`, `E_cls_abs=N_cls_error/N_GT`, `E_global=(N_miss+N_cls_error)/N_GT`. Los tests verifican `E_global = E_det + (1-E_det)*E_cls_cond`. Si no hay matches, el condicional es NA y el global coincide con detección. FP de localización, duplicados y detecciones sin GT se reportan aparte. Las claves antiguas `e_clas` y `e_flujo` son aliases de reporte de las tasas, no fuentes de etiquetas.

El classification report incluye ambas clases, macro y weighted averages; ROC, ROC-AUC y AP positiva se calculan desde predicciones guardadas. Los CSV FP/FN se basan en predicciones reales. Grad-CAM proyecta cajas auxiliares clase 2 al grid del crop entero y calcula energía en su unión y pertenencia del máximo; evita doble conteo de regiones superpuestas. Sin región auxiliar en el crop, esas estadísticas son NA aun si el fruto es positivo. El etiquetado TP/TN/FP/FN de ejemplos usa el umbral documentado.

## Script gráfico externo

`../generar_graficos_botrytis.py` acepta `--project-root`, `--dataset` (ZIP o directorio), `--output`, `--config`, `--allow-estimates` y `--help`.

Descubre YAML/CSV/JSON e historiales, conserva fuentes y exporta datos. Con historiales oficiales comparables genera `figura_5_1_detector_convergencia_<arquitectura/fingerprint>` y `figura_5_2_clasificador_convergencia_<arquitectura/fingerprint>`. Detección muestra sus pérdidas propias; clasificación muestra loss y AP positiva, con F1 cuando existe. Réplicas comparables comparten arquitectura, HP, dataset, split y etapa; la desviación es muestral y queda NA con una sola observación. No duplica historiales por fold si existe el agregado del run.

Con las imágenes locales se generó `reports/graficos_cierre/paper/figura_dataset.png` y PDF con ocho ejemplos reales. Se generaron gráficos exactos de clases e IoU y CSV de soporte. Las curvas de entrenamiento, comparaciones de modelos, ROC, matrices, errores, Grad-CAM y latencia están pendientes por falta de resultados/checkpoints. No se generó curva del híbrido.

El preview probado sólo produce curvas analíticas arbitrarias de maquetación en `estimated_preview/` con `is_estimated=true` y supuestos. No inventa imágenes, Grad-CAM, ROC ni matrices, ni mezcla estimados con `paper/`. `reports/graficos*/INVENTARIO.md` especifica fuente, estado real/estimated/pending/unverified y datos faltantes. Las imágenes y previews estimados se excluyen del ZIP.

## Archivos modificados

- Semántica/auditoría: `src/utils/{pairing,label_audit,labels,boxes,checks,dataset,prepared,semantics,scientific_audit,spatial_audit}.py`; `src/01_validate_dataset.py`, `src/19_audit_labels.py`.
- Derivados/trazabilidad: `src/02_create_outer_split.py`, `03_create_inner_folds.py`, `04_build_detection_dataset.py`, `05_build_classification_crops.py`; `utils/{config_loader,classifier_data,detector_data,logging_utils,freeze}.py`; `15_freeze_experiment.py`; `orchestration/sync.py`.
- Métricas/figuras: `utils/{metrics,metrics_classification,metrics_detection,pipeline_eval,reporting,visualization,plots_experiment,detector_train}.py`; `10_evaluate_outer_validation.py`, `11_tune_inference_thresholds.py`, `14_validate_integrated_pipeline.py` y comentarios relacionados.
- Configs: `default.yaml`, `smoke.yaml`, nuevo `adaptado.yaml`. Herramientas: `make_synthetic_dataset.py`, `smoke_test.py`, `demo_figures.py`, `audit.py`, `make_zip.py`.
- Tests actualizados y nuevos: `test_semantic_pairing.py`, `test_external_graphics.py`, fixtures, geometría, e2e, configuración, splits/folds, freeze, auditoría y selección.
- Documentos e informes: `README.md`, esta adaptación, `AUDIT_REPORT.md`, `GUIA_EJECUCION_3PCs.md`, auditorías label-only/local y evidencias de tests/estática.
- Script entregado fuera del ZIP: `../generar_graficos_botrytis.py`.

## Validaciones de cierre

La suite completa final está documentada en `reports/tests_final_cierre.log` y `reports/tests_final_cierre.xml`; resumen machine-readable en `reports/validation_summary.json`. La auditoría estática está en `reports/audit_report.md`; pyflakes en `reports/pyflakes.log`. Se ejecutan los tests posibles sin entrenar; las pruebas que requieren stack ML ausente o dataset/checkpoints completos se registran SKIPPED con motivo real.

El ZIP excluye entornos, cachés, datasets, imágenes, pesos, runs, ZIPs de entrada, previews estimados y artefactos smoke obsoletos. `tools/make_zip.py` comprueba CRC, miembros duplicados y archivos obligatorios. El cierre verifica además SHA-256 por miembro, exclusiones y compilación del código del ZIP.

Pendientes reales: reconciliar el dataset completo de 329 imágenes y los 17 labels huérfanos; resolver las anomalías en las fuentes o declarar una cuarentena trazable; volver a ejecutar 01–05; completar entrenamientos/HPO/inferencia/Grad-CAM/latencia en las tres PCs con checkpoints reales. Los comandos exactos de preparación, baseline/HPO y gráficos están en README y abajo.

## Comandos reproducibles

Ejecutar desde la raíz del proyecto con Python y las dependencias de análisis disponibles:

En este equipo el alias `python.exe` de WindowsApps no se pudo ejecutar. Se comprobó el intérprete y el stack ligero existentes, sin instalaciones:

```powershell
$BotrytisPython = Join-Path $env:APPDATA 'uv/python/cpython-3.12.14-windows-x86_64-none/python.exe'
$env:PYTHONPATH = Join-Path $env:LOCALAPPDATA 'Packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/LocalCache/local-packages/Python312/site-packages'
$env:PYTHONIOENCODING = 'utf-8'
```

Usar `& $BotrytisPython` en lugar de `python` en los comandos siguientes. En las tres PCs de entrenamiento se utiliza el intérprete de su entorno ML documentado. Baseline/HPO se comprobaron mediante `--help`; entrenar sigue pendiente y bloqueado por dataset/recursos.

```powershell
python src/19_audit_labels.py --dataset ../dataset-botrytis_2026-10-01.zip --output reports/label_audit --metadata ../metadatos_repetidos_project-24_iPhone13.txt
python src/01_validate_dataset.py --label-only --dataset ../dataset-botrytis_2026-10-01.zip --output reports/label_audit

# Rutas externas, una vez reconciliado el dataset completo:
$env:BOTRYTIS_IMAGES_DIR = '../dataset-botrytis_2026-10-01/images'
$env:BOTRYTIS_LABELS_DIR = '../dataset-botrytis_2026-10-01/labels'
python src/01_validate_dataset.py --config configs/adaptado.yaml --pc-id PC-01
python src/02_create_outer_split.py --config configs/adaptado.yaml --pc-id PC-01
python src/03_create_inner_folds.py --config configs/adaptado.yaml --pc-id PC-01
python src/04_build_detection_dataset.py --config configs/adaptado.yaml --pc-id PC-01 --clean
python src/05_build_classification_crops.py --config configs/adaptado.yaml --pc-id PC-01 --clean

# Iniciar sólo después de validación aprobada y stack ML disponible:
python src/06_run_baseline.py --config configs/adaptado.yaml --architecture YOLOv8n --pc-id PC-01
python src/06_run_baseline.py --config configs/adaptado.yaml --architecture ResNet50 --pc-id PC-03
python src/07_run_hpo_study.py --config configs/adaptado.yaml --architecture YOLOv8n --pc-id PC-01
python src/07_run_hpo_study.py --config configs/adaptado.yaml --architecture ResNet50 --pc-id PC-03

# Reales desde ZIP label-only o desde imágenes locales:
$BotrytisSalida = Join-Path 'reports' ('graficos_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
python ../generar_graficos_botrytis.py --project-root . --dataset ../dataset-botrytis_2026-10-01.zip --config configs/adaptado.yaml --output ($BotrytisSalida + '_zip')
python ../generar_graficos_botrytis.py --project-root . --dataset ../dataset-botrytis_2026-10-01 --config configs/adaptado.yaml --output $BotrytisSalida
# Preview, separado de paper/:
python ../generar_graficos_botrytis.py --project-root . --dataset ../dataset-botrytis_2026-10-01.zip --config configs/adaptado.yaml --output ($BotrytisSalida + '_preview') --allow-estimates
python ../generar_graficos_botrytis.py --help
python -m pytest -o addopts='-ra' --basetemp "$env:TEMP/botrytis_revalidacion" --junitxml=reports/tests_final_cierre.xml
python tools/audit.py
python tools/make_zip.py --out ../Botrytis_Experiment_Framework_ADAPTADO.zip
python tools/verify_delivery.py --archive ../Botrytis_Experiment_Framework_ADAPTADO.zip
Get-FileHash ../Botrytis_Experiment_Framework_ADAPTADO.zip -Algorithm SHA256
```

Cuando la regeneración sea intencional y existan derivados incompatibles: añadir `--force` a 02/03 después de revalidar, y `--clean` a 04/05. La cuarentena requiere el mismo `--set pairing.anomaly_policy=quarantine_images` en todas las etapas desde 01; el manifest lista cada imagen, motivo y número de frutos excluidos.

## Cierre de la reanudación

Se conservaron los cambios correctos y las funciones opcionales de revisión del documento del script externo. La entrada predeterminada del script genera `paper/`, `support/`, datos e inventario; sólo `--document` usa la exportación de pendientes del documento. Los previews necesitan `--allow-estimates`, tienen estilo distinto, rótulo explícito y CSV con `is_estimated`, método, supuestos y fuente. Una salida no vacía se rechaza para evitar mezcla con resultados anteriores.

Correcciones comprobadas: optimización del pairing sobre aristas admisibles, versión semántica v3 que invalida derivados anteriores; trazabilidad de configuraciones incompletas sin perder el error original; imports de entrenamiento diferidos en baseline/HPO para permitir `--help`; resultados de auditoría parcialmente ejecutados resumidos como SKIP con conteos; eliminación de la copia repetida de todos los candidatos en pandas después de exportar los CSV de auditoría. Se conservaron asserts, labels y recursos del usuario.

`reports/dataset_source_crosscheck.json` compara los 348 archivos del ZIP con la carpeta local: cero faltantes, cero diferencias. Los únicos archivos adicionales no fotográficos son `desktop.ini` de Windows. Las imágenes proceden de la carpeta local; el ZIP sigue siendo label-only. `reports/local_image_validation_summary.json` contiene la nueva validación completa de píxeles/EXIF/hashes/dHash: 2 errores y 74 warnings, sin corrupción ni duplicados de contenido observados.

Se descubren 1.849 frutos con imagen. En imágenes locales limpias son determinables 181 positivos y 1.633 negativos; 35 frutos locales quedan indeterminados por anomalías. Los 42 frutos pendientes del total de labels incluyen 7 sin imagen. Ninguno de estos conteos autoriza cuarentena ni crops: el número de crops oficiales generados es cero. La figura de ocho fotografías reales se inspeccionó visualmente; seis paneles llevan anotaciones. El dataset real no tiene split/folds/crops experimentales aprobados.

La búsqueda semántica final está en `reports/semantic_scan_final.json`: ninguna regla activa antigua. Las dos menciones de `label_rule` son guardias que rechazan configuraciones obsoletas. La cobertura auxiliar, mAP@0.75 y augmentation conservan sus valores legítimos.

Las figuras 5.1/5.2 y las métricas experimentales siguen pendientes de selección e historiales reales. El entrenamiento final de clasificación de 09 no registra validation/AP por época; con ese historial no puede generarse 5.2. El script admite historiales de robustez de la arquitectura/configuración seleccionada; mantiene separados los folds y etapas, y sólo agrega semillas sobre el mismo fold. Cada figura identifica su etapa. No se cambió el diseño experimental para obtener una figura.

La suite final obtuvo **172 PASS, 0 FAIL, 20 SKIP** (215,30 s). Tras el último ajuste del script externo, sus cuatro tests afectados volvieron a pasar. Los SKIP corresponden a PyTorch/Ultralytics, Optuna y datos oficiales preparados/checkpoints ausentes; los píxeles locales se validaron por separado. Las dos advertencias de matplotlib proceden de pruebas que abren muchas figuras; no son resultados experimentales. La auditoría estática obtuvo **17 PASS, 0 FAIL, 33 SKIP**, incluidos tres resúmenes parciales y la suite no solicitada dentro de esa ejecución. `reports/validation_summary.json` reúne los conteos y motivos. No se declara validación experimental completa.

`tools/verify_delivery.py` comprueba SHA-256, CRC, miembros únicos, igualdad byte a byte con los archivos incluidos, exclusiones, compilación e imports y pruebas ligeras en una extracción temporal. Copia el script gráfico externo junto al proyecto extraído, manteniéndolo fuera del ZIP. Escribe su evidencia junto al ZIP (`.verification.json`), sin cambiar el proyecto empaquetado. El ZIP y su `.sha256` se generan después del último cambio; fotos, pesos, datasets, runs, caches y evidencias históricas obsoletas se conservan localmente y se excluyen de la entrega portable.
