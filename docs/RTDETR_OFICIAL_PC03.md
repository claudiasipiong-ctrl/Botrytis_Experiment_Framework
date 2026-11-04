# RT-DETR-R50-vd oficial para PC-03

Este adaptador externo implementa la corrección del backend, no un cambio aislado del flag `compatible_pretrained`. Usa el grafo PyTorch del repositorio oficial y el checkpoint completo COCO R50-vd. No usa RT-DETR-L, HGNetv2, un backbone ImageNet parcial ni scratch. Se ejecuta fuera del checkout aprobado: **no hacer checkout de esta rama para usarlo**. La entrega por Git permite obtener solo el archivo Python, conservando el commit, código, configuración, estudios y resultados de las tres arquitecturas anteriores.

## Evidencia y dependencias

- Backend oficial fijado: `lyuwenyu/RT-DETR`, commit `df2a96c8c3a098567a1149bf0248da6efe6d495d`.
- Archivo fuente: https://codeload.github.com/lyuwenyu/RT-DETR/zip/df2a96c8c3a098567a1149bf0248da6efe6d495d
- SHA256 del archivo fuente verificado: `e6014ae32d2d384799e353a7a6a511aaef1dad77763492738dfef70e714d6564`.
- Checkpoint: https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth
- Tamaño: 172373587 bytes; SHA256 verificado durante esta integración: `dfaafa5a3d29958a360c35f1fad553a6f287bfae9967e8ee007e65cff21ddf45`.

Estos hashes se calcularon y comprobaron al desarrollar la integración; **no se afirma que sean checksums publicados por upstream**. Se vuelve a comprobar el hash completo de cada descarga y de los pesos antes de cargarlos. Los archivos fuente extraídos se contrastan contra el ZIP verificado, no únicamente contra un manifiesto local editable. Los archivos existentes incompatibles se conservan y bloquean la operación.

Se comprobó una carga **estricta** de todas las claves del modelo COCO de 80 clases, un forward real del grafo completo a 640x640 y un backward real de prueba en CPU. Después de la carga solo se sustituyen `enc_score_head`, `dec_score_head` y la tabla de clases de denoising para `fruit=0`. Las demás tensores, incluyendo las cabezas de cajas, se mantienen idénticos al checkpoint oficial.

No se instala `requirements.txt` del repositorio oficial: su pin Torch 2.0.1 no sirve para el Python 3.12 del proyecto. Se importan únicamente sus módulos originales de modelo, matcher y criterio; se evitan el solver, el dataset COCO y las dependencias de ONNX/pycocotools. Se utiliza el Torch/torchvision/SciPy ya presentes, más datos y métricas de **Ultralytics 8.4.170**, que se verificó por código de esa versión exacta. No se reinstalan paquetes ni se cambia `.venv`.

La comprobación de CPU se hizo con Torch 2.6.0+cpu/torchvision 0.21.0. La GPU, AMP y el Torch instalado en Windows no se probaron remotamente. Antes de gastar GPU en el baseline, el ejecutor exige localmente CUDA, realiza un forward y backward sintéticos con el modelo oficial a 640x640, comprueba pérdidas y gradientes finitos y guarda `backend_verified.json`. No aplica pasos de optimizador en esta autoprueba y no la contabiliza como experimento. Reutiliza esa evidencia solo si coinciden el adaptador, Torch, runtime CUDA y modelo de GPU; no repite la QA completa del proyecto.

## Protocolo preservado y diferencias declaradas

Mantiene los hiperparámetros del baseline que realmente entrega `baseline_hyperparameters` del framework: AdamW, LR de cabeza 1e-3, ratio de backbone 0.1, weight decay 1e-4, semilla 42, máximo 100 épocas y paciencia configurada. Mantiene micro-batch 16, nominal 64 y la acumulación/warmup/coseno de la versión fijada del trainer. Las 15 pruebas HPO siguen declaradas en la configuración, pero **esta entrega inicia solamente el baseline**. No ejecuta los HPO ya terminados ni inicia automáticamente uno nuevo.

Usa las mismas listas de folds, imágenes activas y labels derivados `fruit` del proyecto. `OFF`/`_off`, hashes y particiones no se modifican. Las transformaciones y el collate proceden del `RTDETRDataset` de Ultralytics 8.4.170 con los parámetros existentes, no de una receta COCO distinta. Se evita escribir caches en el dataset. Se rechazan listas duplicadas, imágenes omitidas por corrupción y cruces train/validation. Se mantiene tamaño fijo 640, desactivando el multi-scale de la receta COCO oficial para respetar la decisión fija del framework.

El backbone conserva la normalización congelada propia del modelo oficial. `freeze_at=-1` evita congelar capas entrenables cuando el protocolo no solicita hacerlo. El cambio del grafo Ultralytics no compatible al grafo oficial y sus pérdidas queda declarado; no se presenta como equivalencia numérica con el grafo anterior. El matcher y `SetCriterion` oficiales calculan VFL/L1/GIoU y las pérdidas auxiliares/denoising. Las pérdidas se calculan en FP32 fuera del autocast; la red utiliza AMP FP16 según la configuración. Las pérdidas primarias de train y validation se reportan por separado del total usado para optimizar, que incluye términos auxiliares.

El objetivo interno usa `DetMetrics` y el matching de Ultralytics 8.4.170, igual que los detectores anteriores: mAP50, mAP50-95, precisión y recall. No se sustituye silenciosamente ese AP por la interpolación COCO propia de otras funciones del framework. La inferencia oficial convierte cxcywh normalizado a xyxy de la imagen original sin NMS. Las curvas y métricas son reales; no se rellena una pérdida ausente para declarar convergencia.

## Ronda, cola y preservación

Se verifica PC-03/I21313, binding, propietario y la aprobación `READY_FOR_EXPERIMENT` ya existente. No se registra ninguna aprobación humana ficticia. Se espera a que no haya otros Python y se adquieren los locks locales. Si sigue activo el controlador de prioridad, se detiene para evitar que dos controladores se esperen mutuamente. No se mata ningún proceso ni se borra un lock para eludirlo.

Se crea un catálogo **separado** en `%USERPROFILE%\r50\catalog.json`. El catálogo original del checkout permanece bloqueado e intacto. La entrada nueva identifica el backend oficial, su revisión y el SHA256 del adaptador. El fingerprint nativo incluye esta entrada, la identidad de los pesos y los manifiestos originales. Se crea una ronda supervisada `r50vd_01`; los directorios de ViT, YOLO y sus bases Optuna no se reutilizan ni se modifican. Se añade la nueva entrada R50 al índice local conservando una copia previa de `rondas.json`; las entradas anteriores permanecen. Si aparece otra ronda R50 con entrenamiento previo, se detiene antes de sustituirla.

Se ejecuta el **script 06 real** con un hook acotado al entrenamiento de R50 y al registro de sus checkpoints. El controlador no invoca 07 ni 08. El resumen, los recibos de folds y la cola son los nativos. Al terminar queda `PENDING_REVIEW`; HPO requiere la revisión posterior del baseline. No se elimina el bloqueo metodológico de una comparación global o del test.

Los resultados se guardan en:

```
C:\Users\I21313\r50\r\rtdetr_r50\baseline\
```

Allí están `summary.json`, `fold_metrics.csv`, `history.csv`, figuras y tres carpetas `fold0`, `fold1`, `fold2`. En cada fold se conserva `best.pt`, su recibo y las dos últimas generaciones completas del optimizador en `cp`. Se conservan modelo, EMA, optimizer, scaler, generador del dataloader, RNG de Python/NumPy/Torch/CUDA y gradientes todavía acumulados al cierre de época. Solo se retiran generaciones antiguas de ese buffer nuevo y firmado; nunca se retiran checkpoints históricos de ViT/YOLO. `results.csv` se puede recuperar del historial contenido en el checkpoint al finalizar sin volver a entrenar la última época.

Después de una interrupción se usa la misma orden. Los folds completos los reutiliza el framework; el adaptador reconstruye también sus entradas de checkpoint al recuperar una exportación. Un checkpoint más reciente corrupto o una falta de estado de optimizador bloquean el proceso: no hay fallback silencioso a COCO. Para recuperar un job RUNNING/FAILED se exige que el intento y su propietario registrado coincidan y que ese proceso ya no exista. No se utiliza la antigüedad del heartbeat como prueba. Los intentos están limitados; los metadatos anteriores y la cola se respaldan.

La reanudación fue contrastada en CPU con una ejecución seguida, incluyendo el caso con gradientes pendientes de acumulación. No se promete igualdad bit a bit de kernels CUDA no deterministas. Se utiliza la política determinista del proyecto y se fija el workspace de cuBLAS antes de inicializar CUDA.

## Informes y límite de comparación

`%USERPROFILE%\r50\summary.json` registra el cierre y la cola. `methodology.json` describe el cambio de backend. `backend_verified.json` contiene la autoprueba local. `attempts` conserva metadatos y snapshots de los intentos anteriores. Los datos preparados y las referencias originales no se regeneran.

La ronda incorpora un catálogo/backend distinto del placeholder de las rondas anteriores. **No se afirma que sus fingerprints de cohorte coincidan automáticamente.** La futura selección conjunta deberá revisar y documentar esa diferencia mediante el procedimiento del framework. No se usa `--allow-partial`, no se abre test, no se declara un ganador y no se da por completada la robustez de ViT.

## Pruebas

Se adjunta `test_r50.py`/`dev_tests/test_r50_official_adapter.py`. Usa el backend y checkpoint oficiales descargados en las rutas indicadas por `R50_TEST_SOURCE`, `R50_TEST_WEIGHT` y `R50_TEST_ARCHIVE` (por defecto la cache local `r50`). Las pruebas de orquestación sustituyen explícitamente el trainer costoso por uno sintético; las de carga, criterio, forward/backward y datos usan las implementaciones reales. No se simuló una mejora de mAP ni entrenamiento real del dataset.

La ejecución operativa necesita conectividad con los dos enlaces oficiales durante la primera adquisición y aceptación explícita mediante `--allow-network --accept-license`, como el gestor original. Una vez adquiridos y verificados se reutilizan los archivos. No se necesitan tokens en los comandos ni acceso remoto a Drive.

## Correccion de adquisicion 2026-11-04

La version inicial intentaba descargar despues de importar el framework, cuyo bootstrap impide sockets durante ejecuciones locales. La adquisicion ahora sucede expresamente antes de cualquier import del framework y requiere ambos flags de red/licencia. No se elimina, parchea ni desactiva ningun audit hook. Una prueba de HTTP local en un proceso nuevo demuestra descarga correcta y el rechazo posterior de conexiones por el bootstrap real. Se conserva el ejecutor inicial usando un nombre nuevo al entregar esta version. El entrenamiento y sus hashes no han cambiado salvo la identidad declarada del ejecutor corregido.
