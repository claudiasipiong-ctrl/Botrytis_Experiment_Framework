# Recuperación del cierre de YOLOv8l en PC-01

El adaptador original de YOLO expone `train/box_loss`, `train/cls_loss` y `train/dfl_loss`; deja `train_loss=None` porque la API no proporciona un agregado auténtico. El 06 intentaba convertir ese None a float al resumir convergencia, después de completar los folds. Es un fallo de finalización; no es un OOM ni demuestra un entrenamiento inválido.

Este ejecutor externo es exclusivo de PC-01/I21315 y de la ronda YOLOv8l ya registrada en `bt-hoy-20261103/rondas.json`. Reproduce las rutas cortas originales; en el registro aportado corresponden a `C:\Users\i21315\bf\671d23cb\r\yolov8l\baseline`.

Verifica código y configuración inmutables, binding, propiedad, aprobación QA real, ausencia de otro Python, locks, los tres recibos de folds, hashes, épocas, best epoch, objetivos, pérdidas y concordancia con los CSV originales. Verifica los hashes de checkpoints registrados y los archivos que se declararon conservados. Una ausencia de pesos declarada por el backend no se transforma en un archivo inventado. Cualquier dato crítico incompleto/alterado bloquea la recuperación.

Solo admite inicialmente el TypeError concreto del 06 sobre train_loss/NoneType, respaldado por metadata.json original. Conserva el error, snapshot SQLite y verificación antes de reactivar el job. El 06 real archiva sus metadatos anteriores en previous_attempts antes de producir un nuevo cierre. Los CSV por fold y sus recibos se contrastan otra vez tras la recuperación.

La corrección de convergencia se aplica exclusivamente en memoria sobre el 06: se comprueban las tres componentes reales, sin rellenar pérdidas agregadas. Se conserva el tiempo original de entrenamiento por fold y los hashes de checkpoints en los nuevos metadatos. El SHA del ejecutor y del archivo 06 original, así como el alcance del cambio, quedan declarados. No cambia el código versionado ni la identidad científica de los folds entrenados.

El evaluador usa los caches firmados nativos; los trainers de detector y clasificador se sustituyen por una función que lanza PROHIBIDO_ENTRENAR. Incluso si desapareciera un fold después de verificarlo, la recuperación falla antes de entrenar. No se usan --force, resets, borrados de locks, cambio de splits o nuevos trials.

Al terminar muestra BASELINE_RECUPERADO, folds_retrained=0 y PENDING_REVIEW. El HPO permanece pendiente de revisión humana. No se aprueba ni ejecuta HPO, robustez u otra arquitectura. PC-02 y PC-03 no se modifican. Repetir la orden después de un cierre válido conserva el resultado.

Pruebas: reproducción del float(None) original y recuperación con el 06/evaluador/cache reales usando datos sintéticos; agregados vacíos, componentes inválidas, métricas no finitas, recibos/CSV corruptos, best epoch incompatible, pérdida de cache, preservación de archivos y cola con revisión obligatoria. No se ejecutó entrenamiento GPU ni se leyó remotamente la PC del usuario.
