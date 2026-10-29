# Entrenamiento iterativo supervisado

Esta guía describe operaciones futuras del usuario. No contiene resultados de entrenamiento ni mejoras medidas. Ver comandos completos en [COMANDOS_POWERSHELL_3PCS.md](COMANDOS_POWERSHELL_3PCS.md).

## Caso de aceptación

1. Fijar el mismo commit aprobado, dependencias compatibles y base validada en las tres estaciones. Conservar sus bindings locales y aprobar QA personalmente. No hay pesos ni imágenes reales incluidos.
2. Crear `yolov8n-r01` en PC-01, `yolo26l-r01` en PC-02 y `vit_b16-r01` en PC-03. El manifiesto de cada ronda firma protocolo, inicialización, catálogo de pesos, código/commit, dataset, split/folds y espacio HPO. Las rutas locales y el operador no alteran la identidad científica.
3. Lanzar el worker de cada PC con `--init-plan --architectures` de su modelo y `--kinds baseline --once`. Son tres procesos locales independientes. No se ha demostrado aquí ejecución sobre tres GPUs: solo se simula la orquestación con fixtures.
4. Al terminar YOLOv8n, su baseline pasa a `PENDING_REVIEW`, sin habilitar HPO. Otra arquitectura de PC-01, por ejemplo ResNet50 en su propia ronda, puede trabajar mientras la anterior espera revisión. La GPU local sigue teniendo un solo trabajo a la vez.
5. Revisar `runs/<slug>/baseline/summary.json`, `fold_metrics.csv`, `history.csv`, logs y figuras **dentro de esa ronda**. Comprobar curvas, errores, AP/mAP, dispersión entre folds, recursos y datos. No descartar una arquitectura por un baseline aislado.
6. Si se encuentra un problema, registrar diagnóstico y crear `yolov8n-r02` con `new-round --change ...`. Elegir una modificación justificada del bloque de detector; no tocar seeds, folds, trials, resolución, clases o regla de pairing para acelerar. El manifiesto conserva motivo y diferencias. No se modifica r01 ni se reentrenan los otros seis modelos.
7. Ejecutar solo el baseline de r02 y comparar ambas rondas. No hay cifras de ejemplo que puedan confundirse con métricas observadas: las tablas y gráficos se rellenan solo a partir de archivos existentes. La comparación marca protocolos/espacios diferentes como no equivalentes. También puede comparar evidencia histórica sin afirmar que sea compatible para reanudar con código nuevo.
8. Cuando el usuario aprueba el baseline elegido, continuar HPO. Mantener 15 trials por detector o 20 por clasificador, TPE + MedianPruner y tres folds internos. `trial_block_size=5` permite revisar bloques sin reducir el total. La aprobación de un bloque parcial devuelve el job a `PENDING`; únicamente completar todo el estudio y aprobar su revisión habilita robustez. Los trials fallidos no se maquillan como completos ni se mezclan con otra ronda.
9. Ejecutar top-3 configuraciones distintas × semillas `[17,42,73]` × tres folds. Revisar media y dispersión; aprobar `aggregate` antes de entrenar el candidato con el 70 %. No seleccionar la mejor seed individual. Las semillas de una arquitectura permanecen en su propietario; paralelismo entre arquitecturas, no copias concurrentes del mismo estudio.
10. Elegir personalmente una ronda por arquitectura antes de abrir el 20 %. `authorize-external` predeclara esa elección. Validación y calibración se consumen una vez por arquitectura/experimento; una nueva ronda no libera otra exploración externa. Una operación fallida sin resultado publicado admite recuperación explícita auditada; un resultado existente bloquea retuning.
11. Importar snapshots verificados, ensamblar una cohorte de siete candidatos completos y ejecutar selección e integración. `assemble` no ejecuta inferencia ni entrenamiento. Conserva fingerprints de origen, hashes de checkpoints y evidencia; rechaza duplicados, faltantes o datos/splits/código diferentes. Los cambios de protocolo requieren una aceptación explícita de no equivalencia, sin convertirla en equivalencia.
12. Revisar integración con crops de bounding boxes **predichas**; freeze íntegro y autorización personal posterior; recién entonces test final del 10 % una vez. No hay atajo por un fixture aprobado, un ZIP importado o la existencia de `freeze_manifest.json`.

## Qué significa cada acción

| Acción | Uso | Historial y límites |
| --- | --- | --- |
| Resume | Continuar un job RUNNING interrumpido y compatible | Verifica identidad y locks locales; no recupera procesos vivos ni cambia HP. Los checkpoints propios necesitan recibo SHA; GPU/AMP/RNG de workers Windows requieren QA. |
| Retry | Repetir un intento FAILED de la misma configuración | Intento nuevo, logs previos conservados; folds completos verificados se reutilizan. No confundir con cambiar configuración. |
| Repeat | Repetir una configuración completa | Nueva ronda/ID, conserva ambas ejecuciones; no admite `--change`. |
| New round | Cambiar configuración o espacio de búsqueda | Directorios y SQLite independientes, motivo/operador/diferencias firmados; código/espacio/configuración incompatibles no se reanudan. |
| Publish | Reintentar la publicación de un cómputo terminado | `PUBLICATION_PENDING` no ejecuta el entrenamiento otra vez. |
| Approve | Aprobar una transición de una etapa revisada | Operador y evidencia persistentes; no aprueba QA físico, freeze ni test. |

## Estados que no deben confundirse

`PENDING` → `RUNNING` → `PUBLICATION_PENDING` → `PENDING_REVIEW` → `COMPLETE` (cuando corresponda aprobación). Un fallo de cómputo produce `FAILED`; un bloqueo de pesos o insumos deja el trabajo sin reclamar. `COMPLETE` no significa «ZIP exportado y recibido»: la exportación tiene checksum/recibo y el import tiene ledger/journal propios. Nunca se infiere una aprobación de un código de salida.

El modo `automatic` se declara explícitamente con `new-round --mode automatic`; omite las revisiones entre fases completas, pero conserva integridad, ownership, presupuesto, gates y test cerrado. Cambiar de modo no autoriza por sí solo un gate científico. El modo predeterminado es `supervised`.

## Reasignar sin servidor

No existe una cola global transaccional. Una oferta bloquea el origen, entrega snapshot + SHA + estado autorizado de la ronda, la receptora queda bloqueada hasta recibo y confirmación, y solo después activa ownership. El origen pierde autoridad y conserva su histórico. Nunca se transfiere RUNNING. Importar un ZIP ordinario no concede autoridad, no fusiona estados extranjeros sobre jobs locales y no actualiza una SQLite activa.

Una copia antigua de disco puede contener un ownership anterior: no restaurarlo para forzar avance. Conservar ofertas, recibos, confirmaciones y sus hashes, verificar físicamente ausencia de procesos y reconciliar manualmente ante pérdida de máquina. El protocolo no puede garantizar exclusión distribuida frente a restauraciones no coordinadas o ejecución fuera del framework.

## Qué puede y qué no puede cambiarse

Los datos derivados mantienen el sello semántico v4: fruto físico clase 0, sospecha clase 1, auxiliar clase 2, pairing IoU 0,50. Cambiar configuración de entrenamiento no regenera el split. Copiar los manifiestos firmados a una ronda nueva conserva sus bytes. Cambios de datos/split/folds quedan bloqueados por hashes; la diferencia histórica 324/328/329 necesita inventario y decisión personal, no ajuste automático.

Las optimizaciones computacionales que afecten números deben predeclararse, poder desactivarse y someterse a QA. Los datasets de crops cargan rutas en lugar de multiplicar cachés PIL por worker Windows. Se mantiene el batch nominal; fallback por OOM queda desactivado por defecto hasta contar con `compute.microbatch_qa_evidence`, porque acumulación y BatchNorm no garantizan equivalencia numérica. El contrato registra microbatch/acumulación/LR efectivos expuestos por el backend. No se midió ahorro de tiempo o RAM.
