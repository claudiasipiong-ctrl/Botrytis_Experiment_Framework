# Planificación de siete modelos y tres RTX 6000 Ada

**No hay tiempos GPU medidos.** El hardware declarado por el usuario es Windows 10, Xeon w7-2495X 2,50 GHz, 256 GB RAM y una RTX 6000 Ada por PC (~47 GB VRAM disponible). No se inspeccionaron esas estaciones. Este archivo es una plantilla de planificación, no un benchmark ni una fecha prometida.

## Reparto inicial y cifras exclusivamente ilustrativas

Las cifras provienen de los supuestos de `ESTIMACION_TIEMPOS.md` de la auditoría adjunta. No se ejecutaron entrenamientos ni calibración para obtenerlas.

| Modelo | Propietario inicial | Total ilustrativo GPU-h |
| --- | --- | ---: |
| YOLOv8n | PC-01 / i21315 | 7,1 |
| YOLOv8l | PC-01 / i21315 | 20,1 |
| ResNet50 | PC-01 / i21315 | 3,9 |
| YOLO26l | PC-02 / i21314 | 23,2 |
| Swin-Tiny | PC-02 / i21314 | 7,8 |
| RT-DETR-R50 | PC-03 / i21313 | 26,2 — bloqueado |
| ViT-B/16 | PC-03 / i21313 | 7,0 |

Sumas ilustrativas secuenciales por estación: **31,1 / 31,0 / 33,2 h**. La suma redondeada es 95,3 GPU-h; el máximo sería 33,2 h de cómputo continuo únicamente bajo esos supuestos y si se desbloquea RT-DETR-R50. No incluye espera de revisiones, adquisición de pesos, QA, datos, transferencias, fallos, disponibilidad del laboratorio o nuevas rondas. Actualmente no puede estimarse una finalización oficial con RT-DETR bloqueado.

## Fórmula que debe sustituirse con mediciones

Por arquitectura, modelar por separado:

- Baseline: `3 × épocas_baseline × segundos_por_época` (100 épocas detector / 50 clasificador, con la paciencia ya declarada).
- HPO: `trials × 3 folds × épocas_máximas × fracción_ejecutada × segundos_por_época`. Trials 15/20, máximos 60/40 según configuración vigente. Registrar pruning y early stopping reales; no suponer siempre que se ejecuta el máximo.
- Robustez: `3 configuraciones × 3 seeds × 3 folds × épocas_máximas × fracción_ejecutada × segundos_por_época`.
- Candidato: épocas derivadas del criterio predeclarado de robustez, con todo el 70 %. No fijar artificialmente una fracción del máximo para acelerar.
- Preparación, validación, escritura de checkpoints, figuras, sincronización e inferencia: medir por separado; sumarlas, no deducirlas de FPS de COCO.

La auditoría anterior usó una fracción 0,65, un margen 15 %, épocas finales ilustradas como 75 % del máximo y tasas arbitrarias de 7/20/23/26/5/9/10 s por época para los siete modelos anteriores. **No son parámetros del framework ni recomendaciones para reducir el experimento.** Las nuevas rondas añaden coste y no sustituyen resultados anteriores.

## Calibración que el usuario debe ejecutar personalmente

1. Verificar seguridad térmica y sensores con fuentes oficiales del equipo, identidad/UUID, driver, CUDA y versiones exactamente iguales. No inventar límites térmicos.
2. Revisar licencias y pesos locales verificados; RT-DETR-R50 no entra hasta resolver la decisión de backend/representación. No sustituir por L.
3. Usar un experimento de QA identificado, separado del científico, dentro del 70 % y sin abrir validation externo ni test. Medir con la resolución y batch oficiales; no extrapolar de una resolución reducida sin etiquetar esa diferencia.
4. Medir tiempo de inicialización, carga de datos, warm-up y tiempos por época/batch por separado. Registrar PC, modelo exacto, commit, hashes de datos/folds/espacio, seed, AMP, transformaciones, batch nominal, microbatch/acumulación efectivos, workers/persistent_workers/pinned memory, RSS del proceso y descendientes, VRAM pico, temperatura y estado de seguridad.
5. Comparar workers 0/2/4 y persistencia de workers en QA separado. No adoptar una variante oficial hasta verificar corrección, determinismo/comparabilidad y consumo; guardar evidencia. Registrar RAM total de árbol de procesos Windows, no solo del padre.
6. Registrar filas reales en una tabla local `pc_id,architecture,phase,fold,seed,epoch,seconds,batches,vram_peak_mb,ram_tree_mb,commit,configuration_hash`. Datos ausentes permanecen vacíos, nunca cero inventado.
7. Reemplazar tasas/fracciones por medianas y dispersión observadas, separando las tres PCs y fases. Construir escenarios con cuantiles o rangos medidos, no porcentajes de confianza ficticios.
8. Actualizar el reparto mediante transferencia explícita de trabajos pendientes, conservando protocolo y dueño único. Mover una arquitectura no cambia GPU-horas necesarias; puede mejorar utilización del calendario, pero debe medirse.

Usar `tools/qa.py --help` para opciones de pruebas opt-in y `tools/qa_training_smoke.py --help` para su alcance reducido; estos comandos no fueron ejecutados con entrenamiento por el agente. El smoke sintético de QA no mide el rendimiento científico de todas las arquitecturas. Las mediciones reales y sus resultados pesados se conservan localmente/ZIP, fuera de Git.
