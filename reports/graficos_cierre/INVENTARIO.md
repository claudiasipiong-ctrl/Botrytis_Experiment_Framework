# Inventario de gráficos Botrytis

Proyecto: `C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework`
Fingerprint vigente: `7d24470184771ede32d8c0be1369db8cb6df29f8dfe5108d73a52d3884e9ac4b`

Real = fuente observada con procedencia; pending = datos faltantes; unverified = fuente no aceptada.
Conteos label-only son exactos. Crops posibles no son crops generados. No existe curva de entrenamiento del híbrido.
Media ±1 DE usa desviación muestral y sólo réplicas con arquitectura, HP, dataset, split y etapa iguales.
Estimados sólo en estimated_preview/, sin imágenes, Grad-CAM, ROC ni matrices inventadas.

| Artefacto | Estado | Fuente | Datos faltantes / supuestos |
|---|---|---|---|
| fuentes_descubiertas.csv | real | C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework |  |
| support/distribucion_clases | real | ..\dataset-botrytis_2026-10-01 |  |
| crops_posibles_label_only.csv | real | ..\dataset-botrytis_2026-10-01 | Generados pendientes de imágenes y revisión de anomalías |
| support/distribucion_iou | real | ..\dataset-botrytis_2026-10-01 |  |
| paper/figura_dataset | real | ..\dataset-botrytis_2026-10-01 |  |
| hiperparametros_iniciales.csv | real | configs\adaptado.yaml |  |
| paper/figura_5_1_detector_convergencia | pending |  | Sin historial oficial comparable con pérdidas y métrica por época |
| paper/figura_5_2_clasificador_convergencia | pending |  | Sin historial oficial comparable con pérdidas y métrica por época |
| support/comparacion_detectores | pending |  | Faltan resultados/predicciones/figuras verificables; imágenes y checkpoints según corresponda |
| support/comparacion_clasificadores | pending |  | Faltan resultados/predicciones/figuras verificables; imágenes y checkpoints según corresponda |
| support/tabla_errores | pending |  | Faltan resultados/predicciones/figuras verificables; imágenes y checkpoints según corresponda |
| support/roc | pending |  | Faltan resultados/predicciones/figuras verificables; imágenes y checkpoints según corresponda |
| support/matriz_confusion | pending |  | Faltan resultados/predicciones/figuras verificables; imágenes y checkpoints según corresponda |
| support/fp_fn_visuales | pending |  | Faltan resultados/predicciones/figuras verificables; imágenes y checkpoints según corresponda |
| support/gradcam | pending |  | Faltan resultados/predicciones/figuras verificables; imágenes y checkpoints según corresponda |
| support/descomposicion_error_global | pending |  | Faltan resultados/predicciones/figuras verificables; imágenes y checkpoints según corresponda |
| support/latencia | pending |  | Faltan resultados/predicciones/figuras verificables; imágenes y checkpoints según corresponda |
| support/hiperparametros_finales | pending |  | Faltan resultados/predicciones/figuras verificables; imágenes y checkpoints según corresponda |

## Comandos para producir los datos pendientes

Requieren dataset reconciliado, validación aprobada, stack ML y checkpoints. No se ejecutan entrenamientos desde este script.
Las curvas 5.1/5.2 requieren selección respaldada y pérdidas train/validation observadas por época; si un entrenamiento no registró validation por época, la figura queda pendiente.

```powershell
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\01_validate_dataset.py' --config 'configs\adaptado.yaml' --pc-id PC-01
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\02_create_outer_split.py' --config 'configs\adaptado.yaml' --pc-id PC-01
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\03_create_inner_folds.py' --config 'configs\adaptado.yaml' --pc-id PC-01
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\04_build_detection_dataset.py' --config 'configs\adaptado.yaml' --pc-id PC-01 --clean
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\05_build_classification_crops.py' --config 'configs\adaptado.yaml' --pc-id PC-01 --clean
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\06_run_baseline.py' --config 'configs\adaptado.yaml' --architecture YOLOv8n --pc-id PC-01
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\07_run_hpo_study.py' --config 'configs\adaptado.yaml' --architecture YOLOv8n --pc-id PC-01
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\08_repeat_top_configs.py' --config 'configs\adaptado.yaml' --architecture YOLOv8n --pc-id PC-01
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\09_train_outer_candidate.py' --config 'configs\adaptado.yaml' --architecture YOLOv8n --pc-id PC-01
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\10_evaluate_outer_validation.py' --config 'configs\adaptado.yaml' --architecture YOLOv8n --pc-id PC-01
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\11_tune_inference_thresholds.py' --config 'configs\adaptado.yaml' --architecture YOLOv8n --pc-id PC-01
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\06_run_baseline.py' --config 'configs\adaptado.yaml' --architecture ResNet50 --pc-id PC-03
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\07_run_hpo_study.py' --config 'configs\adaptado.yaml' --architecture ResNet50 --pc-id PC-03
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\08_repeat_top_configs.py' --config 'configs\adaptado.yaml' --architecture ResNet50 --pc-id PC-03
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\09_train_outer_candidate.py' --config 'configs\adaptado.yaml' --architecture ResNet50 --pc-id PC-03
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\10_evaluate_outer_validation.py' --config 'configs\adaptado.yaml' --architecture ResNet50 --pc-id PC-03
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\11_tune_inference_thresholds.py' --config 'configs\adaptado.yaml' --architecture ResNet50 --pc-id PC-03
# Selección: completar las siete arquitecturas detectoras y las tres clasificadoras antes de 12/13.
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\12_select_detector.py' --config 'configs\adaptado.yaml' --pc-id PC-01
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\13_select_classifier.py' --config 'configs\adaptado.yaml' --pc-id PC-03
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\14_validate_integrated_pipeline.py' --config 'configs\adaptado.yaml' --pc-id PC-01
& 'C:\Users\Erick Obradovich\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe' 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código\Botrytis_Experiment_Framework\src\17_benchmark_latency.py' --config 'configs\adaptado.yaml' --pc-id PC-01 --stage models
```
