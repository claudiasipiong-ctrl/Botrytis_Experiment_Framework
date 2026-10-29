# Auditoría de la adaptación Botrytis

Este informe sustituye la evidencia smoke del framework anterior. Esa evidencia utilizaba una semántica incompatible y no acredita esta adaptación. No se entrenaron modelos, instalaron dependencias ML ni descargaron pesos.

El detalle metodológico, conteos recalculados, anomalías y comandos está en [ADAPTACION_BOTRYTIS.md](ADAPTACION_BOTRYTIS.md). La evidencia ejecutable se conserva en:

- `reports/label_audit/`: auditoría completa del ZIP sin imágenes, incluidos labels vacíos y huérfanos, todas las parejas IoU y anomalías.
- `reports/local_dataset_crosscheck/`: auditoría de la carpeta adyacente descubierta con 328 imágenes; faltan imágenes respecto al control de 329 y hay 17 labels sin imagen.
- `reports/dataset_source_crosscheck.json`: comparación SHA-256 de los labels del ZIP con los de la carpeta local.
- `reports/local_image_validation_summary.json`: validación real de píxeles, EXIF y duplicados. Resultado bloqueante esperado por cantidad de imágenes y anotaciones ambiguas.
- `reports/tests.xml` y `reports/tests.log`: última ejecución completa de pytest. Los skips explican dependencias o datos/checkpoints ausentes; las fixtures sólo verifican software.
- `reports/audit_report.md`: sintaxis, imports, CLI, configuración, tipos, pyflakes y formato de texto; los imports ML no disponibles se registran SKIP.
- `reports/validation_summary.json`: resumen final de PASS/FAIL/SKIP y limitaciones.
- `reports/graficos*/INVENTARIO.md`: datos exactos y figuras disponibles o pendientes. Las previsualizaciones estimadas permanecen separadas y no se empaquetan.

La generación oficial de crops se bloquea hasta resolver las anomalías. La cuarentena exige configuración explícita y genera un manifest; no hay conversión silenciosa de frutos ambiguos en negativos. La clase 0 es el único GT detector; la clase 1 define positividad mediante pairing 1:1; la clase 2 aporta únicamente evidencia espacial auxiliar.

Quedan pendientes entrenamiento, HPO, selección empírica, checkpoints, métricas predictivas, Grad-CAM y benchmark. También queda pendiente la ejecución física en tres PCs. Los tests de orquestación y sincronización verifican el software localmente.
