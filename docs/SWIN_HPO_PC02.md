# Continuar Swin-Tiny en PC-02

Ejecutor externo completo para la ronda existente, sin checkout, pull, reset, reinstalaciones ni nuevas rondas. No se invoca 06 (baseline), 08 (robustez), ni test. La entrega utiliza Git fetch y extracción de un único archivo a Downloads; el commit aprobado del proyecto no cambia.

## Estado aportado y verificación local

El registro adjunto muestra la ronda `n_swin_tiny`, fingerprint `cdbbf3341a038047d4872e24cc57d046a387bc620f8251de03af281a4920b3cc`, baseline `PENDING_REVIEW` y HPO `PENDING`, con dependencia exclusiva del baseline. AP positiva: 0.8976507275555425; SD muestral: 0.04985536544305446; folds 0.9507574684849984, 0.8903408758529654, 0.8518538383286638. Son métricas de clasificación, no mAP de detección. Estos valores del chat no sustituyen la lectura de evidencias locales.

El script exige usuario I21314, binding PC-02, configuración PC-02 y propiedad de Swin-Tiny. Lee la ruta exacta de `bt-hoy-20261103/rondas.json` y reproduce las dos rutas cortas usadas por `botrytis_prioridad.py`. No crea un experimento ni una cola si no existen. Verifica la ronda inmutable, su configuración, identidad de los jobs, código, datos, sellos y folds originales. Diferencias reales de Git bloquean; un ajuste del índice solo se permite tras demostrar que el filtro Git da exactamente el blob HEAD (LF/CRLF), sin cambiar fuentes.

Reutiliza la aprobación READY_FOR_EXPERIMENT real y verifica actualmente CUDA, entorno, pesos de inicialización, datos preparados y ambas clases en cada fold. No repite la QA completa. Verifica cierre COMPLETE de la ejecución baseline, tres recibos, hashes de historiales, épocas consecutivas, protocolo, scores, media/SD y metadatos de checkpoints. Los pesos que el baseline original declaró descartados se distinguen de checkpoints físicos conservados; nunca se inventa un best.pt ausente. Se comprueban los recibos de checkpoints existentes.

Se genera REVISION.html y revision.json en una carpeta nueva por intento. Se incluyen las curvas nativas existentes, cambios de pérdidas, métricas del mismo best epoch, AP por fold, dispersión y limitaciones. La brecha de AP train-validación queda ND cuando no hay inferencia del mismo checkpoint sobre train. Las pérdidas de train incluyen augmentation: no equivalen directamente a pérdidas de validación. Una métrica alta o una señal tardía no demuestra por sí sola sobreajuste.

Se comprueba herencia imagen->crop de split/fold, contenido exacto y similitud perceptual entre particiones internas/externas. Metadatos de planta/hoja/sesión solo se usan si existen explícitamente. Conflictos confirmados o pares perceptuales relevantes sin evidencia de fuentes distintas bloquean; APROBAR no falsifica una inspección visual. OFF y _off se respetan mediante los lectores originales. No se regeneran datos, hashes ni splits. No se usa test para HPO.

## Aprobación y HPO

Tras superar controles se abre el informe y se muestra en CMD el resumen y sus advertencias. La persona debe escribir exactamente **APROBAR** en una consola interactiva. Antes de registrar la decisión se vuelve a contrastar que no cambiaron las evidencias del baseline. La decisión queda vinculada por hashes al informe, a su evidencia JSON y al fingerprint experimental, mediante StudyQueue.approve real. Una respuesta distinta deja el baseline sin aprobar. Una aprobación válida previa se conserva.

Se usa exclusivamente `07_run_hpo_study` real con `--architecture swin_tiny --pc-id PC-02 --trial-block-size 20` y las rutas existentes. El presupuesto total sigue siendo **20 COMPLETE+PRUNED**, **tres folds**, AP positiva, TPE y MedianPruner nativos, semilla y espacio de búsqueda aprobados, máximo 40 épocas y paciencia vigente. FAIL no consume presupuesto y conserva su historial, con los límites originales del framework.

El hook externo conserva best.pt y los checkpoints de optimizador que el trainer original escribe cada época. Conserva los hashes del estado y del archivo por separado. Se guardan y verifican sampler/pruner locales, versión de Optuna/Python, máquina e identidad. Un estado de sampler anterior a una propuesta no guardada exige reconstrucción exacta de las propuestas previas; si no coincide se detiene, no se reinicia TPE silenciosamente.

Al reanudar se exige propietario local demostrado muerto (PID y creación), locks libres, ausencia de otro Python y respaldo SQLite íntegro. No se elimina ningún lock ni se mata proceso alguno. Un trial interrumpido permanece FAIL como intento histórico; se encolan una sola vez sus mismos parámetros con un vínculo al intento original. Se copian folds firmados completos al nuevo intento sin volver a entrenarlos; los parciales continúan desde el checkpoint verificado y RNG/optimizer/scaler/dataloader del trainer nativo. Si la última época o parada temprana ya se completó, se recupera el resultado del checkpoint sin otra época. Evidencia crítica ausente o corrupta bloquea.

Las interrupciones del epoch actual pueden perder el trabajo no guardado de esa época; las épocas guardadas y los trials COMPLETE/PRUNED no se repiten. No se promete determinismo bit a bit de todos los kernels CUDA.

Se muestran trial/configuración, fold, épocas, AP, pérdidas, errores y recuperación en CMD. Al finalizar el framework genera los 13 jobs posteriores; **este ejecutor no los lanza ni aprueba el HPO**. Se mantiene PENDING_REVIEW. Se conserva baseline e HPO: 50 épocas máximas en baseline frente a 40 en HPO impiden interpretar su diferencia de score como superioridad causal sin una comparación posterior.

## Rutas de PC-02

- Baseline preservado: `C:\Users\I21314\bf\0346b429\r\swin_tiny\baseline`.
- HPO: `C:\Users\I21314\bf\0346b429\r\swin_tiny\hpo`.
- Revisión: `C:\Users\I21314\bf\0346b429\swin_review\a\<intento>\REVISION.html`.
- Cierre HPO: `C:\Users\I21314\bf\0346b429\swin_review\hpo-final.json`.
- Sampler y eventos de recuperación: `...\swin_review\recovery`.

El ejecutor deriva el directorio corto de la ruta real registrada; no cambia a estas rutas a ciegas. YOLO26l y su robustez pospuesta, así como PC-01 y PC-03, permanecen fuera de la cola Swin.

## Pruebas y límites

Las pruebas adjuntas incluyen integridad del baseline, rechazos de historiales/objetivos corruptos, aprobación manual y conservación de estados, protección contra duplicados, conservación del sampler y reanudación del trainer nativo en CPU con un modelo sintético pequeño. También se prueba el 07 real con Optuna y tres folds usando un trainer diminuto explícitamente sustituido: valida interfaces, presupuesto, resultados y no repetición; no demuestra rendimiento real de Swin.

No se ejecutaron entrenamientos reales de GPU en PC-02 ni se inspeccionó remotamente su SQLite actual. La inspección definitiva de sus archivos se realiza en esa PC al ejecutar el comando.
