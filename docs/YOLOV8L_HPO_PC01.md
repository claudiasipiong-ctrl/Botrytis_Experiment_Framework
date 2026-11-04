# YOLOv8l: revisión real del baseline recuperado y HPO en PC-01

El registro aportado confirma baseline COMPLETE, cola PENDING_REVIEW, cero folds reentrenados y HPO no iniciado. mAP50-95 por fold: 0.85742, 0.83619, 0.82303; media 0.83888 y SD muestral 0.017352092092886053. Mejores épocas: 94, 92, 86. Estos datos del chat son referencias; el ejecutor comprueba los archivos locales vigentes antes de aprobar o entrenar.

## Una sola ejecución

El archivo completo es externo al checkout y exclusivo de PC-01/I21315/YOLOv8l. Lee la ronda de `bt-hoy-20261103/rondas.json`, verifica la configuración inmutable y reproduce las rutas cortas del ejecutor de prioridad. No crea una ronda nueva, no instala paquetes y no invoca baseline ni robustez. La propiedad, binding, usuario, configuración, código, datos, sellos, folds y pesos COCO se comprueban con funciones reales del framework.

La QA READY_FOR_EXPERIMENT aprobada se reutiliza; se comprueban actualmente los insumos críticos y el entorno CUDA, sin repetir la batería completa. La revisión de baseline verifica cierre COMPLETE, firmas, hashes de los tres historiales, épocas, métricas, protocolo, media/SD y metadatos de checkpoints. Las pérdidas de entrenamiento por componentes son obligatorias. El agregado train_loss vacío no se convierte a float, no se sustituye por cero y no impide revisar las componentes auténticas. Las pérdidas auxiliares de validación ausentes o NaN/inf se diferencian y se muestran como ND o anomalías; métricas objetivo no finitas, pérdidas de entrenamiento inválidas y CSV corruptos bloquean.

Se revisan la herencia de splits/folds a partir de la imagen, contenido exacto, dHash y metadatos de planta/hoja/sesión si existen. Los conflictos confirmados bloquean. Se mantiene obligatoria la revisión nativa de similitud entre splits externos. Las sospechas perceptuales limitadas a folds internos se muestran expresamente, con imágenes locales cuando están disponibles; no se transforman en fuentes distintas ni se declara independencia biológica. APROBAR autoriza HPO con las limitaciones documentadas y no certifica una inspección visual realizada por el ejecutor. Si se identifica la misma fuente, se debe conservar el estado sin aprobar y revisar el protocolo, sin regenerar splits a ciegas.

REVISION.html se abre localmente. La persona debe escribir exactamente APROBAR en CMD interactivo. Se revalida que no cambió el baseline durante la decisión. La aprobación se registra con StudyQueue.approve y copias inmutables del informe y JSON aprobados, con hashes; el informe final puede actualizarse sin alterar esas copias. Una respuesta distinta conserva los estados. Una revisión previa real registrada se conserva al reanudar.

Después se ejecuta solo el 07 real con arquitectura yolov8l, PC-01 y trial-block-size 15. El presupuesto nativo sigue siendo **15 COMPLETE+PRUNED, tres folds, máximo 60 épocas por fold y paciencia configurada**, TPE y MedianPruner originales, misma semilla, mismo espacio de búsqueda y datos preparados. No se utiliza validation externo o test para HPO. Se muestran trial, configuración, fold, época, métricas, recuperación y errores en CMD.

## Preservación y recuperación

Se conservan best.pt y snapshots firmados del optimizador anteriores al strip de Ultralytics. Se guarda sampler/pruner local con identidad, versión, máquina y hashes. No se reinicia TPE silenciosamente: si falta un estado compatible, solo se acepta reconstrucción exacta de las propuestas anteriores.

Locks ocupados u otro Python activo impiden una ejecución duplicada. Un RUNNING solo se recupera cuando el PID y creación del propietario demuestran que terminó. No se mata ningún proceso ni se borran locks. Se respaldan los SQLite antes de recuperar. Un trial interrumpido permanece FAIL como intento histórico; se encolan una sola vez sus parámetros originales con trazabilidad. Los folds firmados completos se reutilizan; los parciales se reanudan con el optimizador verificado. Se archiva el directorio parcial previo y se conserva el CSV completo. Evidencia incompatible, truncada o sin optimizador bloquea en lugar de reiniciar el entrenamiento.

Si se interrumpe después de completar las épocas pero antes de obtener un recibo de fold, el caso requiere recuperación de exportación; este ejecutor se detiene para evitar repetir las épocas. Se conserva el trabajo no perdido. La época en curso sin checkpoint puede tener que repetirse; no se promete reproducibilidad bit a bit para todos los kernels CUDA.

Al completar HPO, la cola nativa genera los 13 jobs posteriores y deja HPO PENDING_REVIEW. **El ejecutor termina ahí: no inicia robustez, otros modelos ni evaluaciones externas**. Baseline y HPO se conservan como candidatos; 100 épocas máximas del baseline y 60 del HPO no permiten atribuir una diferencia de score exclusivamente a los hiperparámetros. No se declara ganador definitivo.

## Archivos locales de PC-01

- Baseline preservado: `C:\Users\i21315\bf\671d23cb\r\yolov8l\baseline`.
- HPO nuevo: `C:\Users\i21315\bf\671d23cb\r\yolov8l\hpo`.
- Informe: `...\bf\671d23cb\yolov8l_review\a\<intento>\REVISION.html`.
- Copias aprobadas: `revision_aprobada.html`, `revision_aprobada.json`, `aprobacion.json` en ese intento.
- Resumen de HPO: `...\yolov8l_review\hpo-final.json` y el study_summary.json nativo.

Las rutas se derivan de la configuración real; no se escriben a ciegas si el estado registrado difiere. OFF/_off se respetan con los lectores originales. Los estudios de YOLOv8n y de las otras PCs no se modifican.

## Verificación

Se prueba el 07 real con Optuna/cola/caches de folds y un entrenador sintético de detector: 15 trials, 45 folds, pesos y snapshots conservados, 13 jobs posteriores sin lanzar y segunda ejecución sin entrenamiento. Hay pruebas de corrupción del baseline, agregados vacíos, diagnósticos opcionales ausentes/no finitos, aprobación humana, inmutabilidad de la aprobación, bloqueo de duplicados, estado del sampler y recuperación de un checkpoint parcial con archivo previo preservado. Son pruebas en CPU; no son entrenamiento real de GPU ni inspección remota del SQLite de PC-01.
