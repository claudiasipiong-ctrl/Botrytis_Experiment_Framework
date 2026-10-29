# Separación y evidencias por modelo

Esta versión audita y amplía el framework adjunto. Conserva split externo 70/20/10 por imagen, estratificación existente, folds internos dentro de train, HPO, semillas, selección con validation y pipeline detector + clasificador. No aplica nuevas reglas de cobertura ni cambia las etiquetas.

## Hallazgos y correcciones

| Punto auditado | Estado previo | Corrección |
|---|---|---|
| Split por imagen y crops heredados | Ya implementado | Se conserva y se verifica al cargar derivados |
| HPO | Folds únicamente en outer train | Se conserva; se identifica `inner_validation` como parte de train |
| Entrenamiento final del detector | YAML incluía validation; el motor podía validar al cierre pese a `val=False`; selección de fichero prefería `best.pt` | YAML solo de train, callbacks deshabilitan validación de cierre y última época; checkpoint obligatorio `last.pt` |
| Entrenamiento final del clasificador | Sin validation externo por época | Se conserva |
| Integridad de derivados | Sello semántico, sin verificación completa de bytes de cada crop/imagen/label | Inventario sellado y SHA-256 de todos los derivados, revisión de archivos adicionales y herencia del split |
| Duplicados | SHA-256 de originales y alertas dHash | También comparación de píxeles RGB orientados entre splits, en originales y crops |
| Imágenes similares | Alertas sin resolver posibles cruces | Listado de pares entre splits y bloqueo antes de entrenar/evaluar si su revisión no está documentada |
| Evidencias de detección | Solo muestras del pipeline seleccionado | Todas las imágenes y predicciones por detector, incluido RT-DETR |
| Clasificación de test | Solo crops asociados del pipeline seleccionado | Evaluación independiente de todos los clasificadores sobre los mismos GT crops de test; pipeline permanece aparte |
| Comparación en test | Solo pareja seleccionada | Lista de candidatos entrenados/validados sellada antes de abrir test; comparación descriptiva sin reselección |
| Gráficos | Curvas de pérdidas train/validación interna en una figura | Exportaciones y figuras separadas por rol, sin confundir fold interno con validation externo |
| Métricas de train | Pérdidas de entrenamiento; sin evaluación independiente uniforme | Evaluación del checkpoint final en train, sin augmentación en clasificación, sin selección posterior |
| Ejemplos de preparación | Podían incluir test | Solo ejemplos de train |
| Sincronización entre PCs | No incluía esta nueva estructura | ZIP de resultados incluye `results/` completo |

## Salidas

| Ruta | Contenido |
|---|---|
| `results/detection/<slug>/train/` | Historial, pérdidas por época y métricas de evaluación en train del checkpoint final |
| `results/classification/<slug>/train/` | Equivalente para clasificadores |
| `results/<tarea>/<slug>/train/internal/<run>/train/` | Curvas y pérdidas de entrenamiento de folds |
| `results/<tarea>/<slug>/train/internal/<run>/inner_validation/` | Curvas y métricas de validación interna, siempre dentro de outer train |
| `results/detection/<slug>/validation/` y `test/` | `visualizations/`, `crops/`, `metrics/`, `plots/`, `artifact_index.json` |
| `results/classification/<slug>/validation/` y `test/` | Misma estructura, solo crops realmente evaluados |
| `results/comparisons/<tarea>/<split>/` | Tabla y gráficos entre modelos usando un solo split |
| `results/summary.json` | Conteos exactos, balance de clases, métricas por modelo/split, estado de cada modelo y todas las rutas creadas |
| `results/SUMMARY.md` | Resumen legible |
| `results/source_content_audit.csv`, `crop_content_audit.csv` | Identidad, split y hashes de archivos/píxeles |
| `artifacts/near_duplicate_cross_split.csv` | Pares similares entre splits pendientes de revisión |

Los 7 detectores del registro son YOLOv8n/m/l, YOLO26n/s/l y RT-DETR-R50. Los 3 clasificadores son ResNet50, ViT-B/16 y Swin-Tiny. El exportador visual recibe cajas/scores, no nombres de familias. Una futura arquitectura que implemente el contrato de inferencia del framework reutiliza el exportador; esta entrega no incorpora nuevos backends de entrenamiento.

Detección: GT de `arandano_a` en verde grueso y predicciones en naranja fino, conservando ambos contornos cuando coinciden. Una imagen sin predicciones también se exporta. Cada predicción genera un crop de la imagen original, sin anotaciones ni margen, con nombre que contiene modelo, split, imagen y número de predicción. Los CSV conservan coordenadas, confianza, IoU porcentual, asociación GT, ventana de recorte y hashes. TP/FP/FN se describen a IoU 0.50, con matching uno a uno por confianza. Para un FP, `best IoU` es solo el mejor solapamiento: una predicción duplicada puede tener 100 % y seguir siendo FP. No se confunde IoU con confianza ni con cobertura de signos.

Clasificación: copia exacta de cada crop de entrada y visualización con etiqueta real, clase predicha, P(con_signos), probabilidad de la clase predicha, acierto/error, modelo, split e identificador de imagen/fruto. No se evalúan imágenes completas. Se usa 0.5 en la comparación individual, igual que en el paso 10 existente; el pipeline seleccionado conserva su umbral congelado y sus métricas sobre crops predichos/asociados, reportadas aparte en sus artefactos originales.

`metrics.json` contiene las métricas. `losses.json` declara la pérdida medida o su ausencia. Ultralytics `model.val` puede no exponer pérdidas de evaluación de un checkpoint: no se inventan. Las pérdidas de train son las reales por época. Validation/test tienen curvas PR/ROC/AP y de confianza según tarea; no se inventan curvas de pérdida por época para una única evaluación de un checkpoint. El barrido de confianza de test es descriptivo y no elige umbrales. Las métricas oficiales de detección a máximo F1 se identifican como estadísticas descriptivas del backend; el punto operativo del pipeline viene congelado desde validation.

## Ejecución

1. Resolver las decisiones pendientes de la configuración existente y las rutas reales. Usar la misma configuración y entorno en todas las PCs. Esta revisión no declara ni modifica semillas en nombre del equipo.
2. Ejecutar 01–05. Si había derivados de la versión anterior, regenerar 04 y 05 con `--clean`: los nuevos inventarios de contenido son obligatorios. El manifiesto externo válido se mantiene.
3. Si hay pares en `near_duplicate_cross_split.csv`, revisar originales y registros de captura antes de entrenar. Documentar solo conclusiones verificadas en `artifacts/near_duplicate_review.json` con esta estructura:

```json
{
  "dataset_hash": "HASH_REAL_DEL_DATASET",
  "pairs": [
    {
      "image_a": "ID_REAL_A",
      "image_b": "ID_REAL_B",
      "decision": "distinct_sources",
      "evidence": "Descripción verificable de las capturas/escenas que demuestra fuentes distintas"
    }
  ]
}
```

Si son la misma fuente o no puede comprobarse su independencia, la ejecución permanece bloqueada. No marcar `distinct_sources` para saltar el control. La revisión debe ser de integridad de datos, sin mirar resultados del modelo en test. No se vuelve a sortear el split automáticamente: un cambio deliberado de agrupación/dataset invalida los derivados y resultados correspondientes.

4. Ejecutar 06–09 para cada arquitectura, manteniendo el HPO y la robustez originales. El paso 09 registra pérdidas de entrenamiento y evalúa el checkpoint final solo sobre train; esta evaluación no selecciona ni cambia pesos.
5. Ejecutar 10 para cada arquitectura. Genera todas las evidencias de validation y actualiza el resumen.
6. Ejecutar 11–14 como antes. Seleccionar y ajustar exclusivamente con validation.
7. Ejecutar 15. Audita evidencias, sella todos los candidatos que tienen checkpoint final y validation completos, pesos, datos, protocolo y revisión de similitud. Un candidato entrenado sin validation completa impide congelar. Arquitecturas sin entrenar se declaran pendientes y no reciben resultados inventados.
8. Ejecutar 16 una sola vez. Evalúa cada candidato sellado sobre los mismos datos de test; después evalúa el pipeline seleccionado. No reselecciona ni reajusta. No se pueden añadir candidatos una vez abierto el test.
9. Ejecutar 17 y 18 como antes. El benchmark sigue usando validation, nunca test para decidir. Para una auditoría de contenido/resumen sin ejecutar modelos:

```bash
python src/20_audit_split_evidence.py --config configs/adaptado.yaml
```

Añadir los mismos `--set` usados para rutas y decisiones en los pasos anteriores. `paths.results_dir` permite ubicar la nueva salida. El paso 20 requiere los datos reales y derivados completos de 01–05. El paso 18 actualiza también el resumen. Para actualizarlo inmediatamente después de train, ejecutar 20.

## Reanudación y límites

`16 --reproduce` exige resultados oficiales existentes; crea evidencia en un directorio de reproducción y compara métricas, sin sobrescribir test oficial. `--retry-after-crash` mantiene checkpoints y protocolo fijos y registra el reintento. Conservar `results/`, `runs/`, artefactos sellados, derivados y pesos al transferir el experimento. Freeze y test deben ejecutarse en la misma PC coordinadora: las referencias absolutas de checkpoints ya eran parte del diseño original. No mezclar carpetas de experimentos con hashes distintos.

Los hashes prueban exclusión de identidades y contenido exacto; dHash es una señal de revisión, no una prueba completa de independencia biológica. Para descartar tomas repetidas del mismo fruto/planta/escena que dHash no detecte, hacen falta metadatos de recolección. La unidad por imagen se conserva por solicitud; no se afirma independencia por planta o sesión sin esos datos.

El ZIP adjunto no contiene imágenes originales, crops de entrada, manifiesto externo ejecutado ni checkpoints entrenados. Por tanto, esta entrega no presenta resultados de modelos ni conteos reales de train/validation/test. Los archivos de `reports/` heredados son antecedentes de la adaptación, no resultados de esta ejecución. Las pruebas nuevas usan datos sintéticos para comprobar comportamiento; no son evidencia de desempeño científico.
