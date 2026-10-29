# Auditoría de separación, resultados y evidencias

Fecha: 7 de octubre de 2026. Fuente: el ZIP adjunto de esta conversación. El informe de uso completo es `SEPARACION_Y_EVIDENCIAS.md`.

## Conclusión

Se corrigió el flujo para que entrenamiento, validación interna, validation externo y test tengan identidad explícita y salidas separadas. La comparación individual final cubre todos los candidatos entrenados y validados que se sellan antes del test. La selección de detector/clasificador y los umbrales del pipeline siguen dependiendo de validation, nunca de test. La implementación de evidencias recibe predicciones normalizadas y funciona para todos los detectores del registro, incluyendo RT-DETR.

No se puede certificar ausencia de fuga en el dataset real ni informar métricas científicas con el material suministrado: faltan imágenes/crops de entrada, split ejecutado y checkpoints. Los controles implementados abortan ante cruces exactos, herencias incorrectas, derivados alterados, evidencias incompletas, freeze inválido y alertas de similitud sin revisión documentada. La posible dependencia entre tomas del mismo fruto/planta/sesión necesita metadatos de captura que no se proporcionaron.

## Evidencia de código

- `src/utils/splits.py`: test requiere etapa final, manifiesto sellado y candidatos/protocolo íntegros.
- `src/utils/freeze.py`, `comparison.py`, `15_freeze_experiment.py`: pesos seleccionados y no seleccionados, protocolo de evaluación y contenido derivado quedan sellados; se rechazan cambios posteriores.
- `src/utils/detector_train.py`, `09_train_outer_candidate.py`: la etapa de entrenamiento final solo recibe train, desactiva ambas rutas de validación del motor y conserva el último checkpoint. La evaluación posterior de train se registra como resustitución, sin selección.
- `src/utils/classifier_train.py`: se conserva el entrenamiento final sin validation externo; los folds HPO salen exclusivamente de train.
- `src/utils/isolation.py`, `prepared.py`, `04/05`: hashes de originales/crops, revisión del contenido en disco y verificación de los manifiestos derivados.
- `src/utils/evidence.py`, `10_evaluate_outer_validation.py`: exportaciones exhaustivas por modelo y split, con visualizaciones/crops/registro individual.
- `16_evaluate_test.py`: una apertura oficial, lista de modelos predeclarada, reproducción separada y pipeline seleccionado conservado.
- `src/utils/split_summary.py`, `20_audit_split_evidence.py`: conteos exactos, métricas, rutas e integridad de todos los artefactos; comparaciones separadas por tarea y split.
- `src/orchestration/sync.py`: incluye la nueva estructura completa al transferir resultados entre PCs.

## Qué se comprobó con pruebas

La suite heredada cubre creación del split, folds, preparación del dataset, etiquetas, métricas, trazabilidad, congelación y sincronización. Las pruebas añadidas verifican duplicados con distinto formato pero iguales píxeles, crops cruzados entre splits, archivos modificados/ausentes/adicionales, exportación de imágenes sin predicciones, duplicados de detección con IoU 100 % que siguen siendo FP, exportador genérico para YOLO/RT-DETR/otra familia, registro de cada crop de clasificación, pesos no seleccionados alterados, protocolo cambiado, revisión de pares similares, conteos/estado de modelos en el resumen y alteración de una evidencia exportada.

Las comprobaciones de separación y exportación usan datos y predicciones sintéticos. El callback que desactiva validación se probó con un entrenador simulado; no se ejecutó entrenamiento nativo de Ultralytics ni PyTorch. La compilación de Python se verificó. Los resultados definitivos de pruebas están en `final_tests.xml`/`.log` y las comprobaciones focalizadas de guardias/evidencia en `targeted_final.xml`/`.log`.

Las pruebas omitidas se identifican individualmente en esos registros. No están disponibles PyTorch, torchvision, timm, Ultralytics ni Optuna en este entorno, ni el paquete externo de gráficos ni las imágenes originales. Las dependencias y sus versiones fijadas se conservaron; su instalación y ejecución nativa siguen pendientes de verificar en las PCs del experimento.

## Cantidades verificables del material entregado

| Elemento | Cantidad/estado |
|---|---|
| Arquitecturas registradas | 10: 7 detectores y 3 clasificadores |
| Imágenes originales incluidas en `data/raw/images` | 0 |
| Labels de entrada incluidos en `data/raw/labels` | 0 |
| Crops de entrada | 0 |
| Checkpoints entrenados | 0 |
| Modelos realmente evaluados aquí | 0 |
| Cantidades reales por split | No disponibles: no existe manifiesto ejecutado dentro del ZIP |
| Métricas reales por modelo | No disponibles |

Los CSV y PDF heredados bajo `reports/` son antecedentes de otra adaptación; no sustituyen el dataset ni constituyen resultados de entrenamiento/validation/test de esta entrega. No se extrapolan cantidades de esos antecedentes.

## Uso

Regenerar 04/05 para crear los nuevos sellos de contenido; conservar el split válido existente. Completar 06–10 para cada modelo, resolver alertas de similitud con evidencia y ejecutar 11–15 antes de 16. El paso 16 utiliza únicamente los candidatos sellados. Ejecutar 20 para obtener un resumen sin inferencia; después de 16 el resumen se genera automáticamente. Si faltan candidatos, el resumen los identifica como pendientes.

`changes.diff` y `changed_files.json` permiten revisar el delta contra el ZIP original. `DELIVERY_SUMMARY.json` registra el material disponible y los resultados de pruebas, sin métricas inventadas.
