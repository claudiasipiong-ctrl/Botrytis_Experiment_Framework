# Auditoria tecnica — 28 de octubre de 2026

## Alcance y evidencia disponible

Se inspecciono el ZIP de framework de 332396 bytes, se inventariaron sus 111 archivos (87 Python) y se reconstruyeron las entradas/salidas de las etapas 00–18, sus utilidades, colas, configuraciones, scripts, tests y documentacion. El inventario original con hashes, imports, funciones y guardias esta en source_inventory_original.json. El original se conservo; los documentos heredados estan identificados como historicos. Sus afirmaciones previas sobre una laptop, CUDA o pruebas no acreditan las tres computadoras.

El enlace de Drive aporto arbol_dataset.md. Se recupero como archivo de referencia; no se recibio el ZIP de imagenes en este entorno. No hay evidencia real de hardware, rendimiento, temperatura, versiones instaladas, checkpoints, conectividad o correspondencia entre identidades. No se usaron herramientas de escritorio remoto ni se ejecuto entrenamiento real, QA de las PCs, CUDA o benchmarks.

**Hallazgo de datos:** el arbol enumera 324 imagenes activas y 325 labels activos. El stem b1602e5f-IMG_6569 tiene label activo y su imagen aparece en _off. El framework exige 328 imagenes. Es una discrepancia documental confirmada, no una prueba de corrupcion del ZIP real. No se modifico el conteo, no se recupero material excluido ni se cambio el split. prepare_machine y QA verificaran los archivos reales.

## Flujo reconstruido

| Etapas | Entradas y salidas relevantes | Dependencia y proteccion |
|---|---|---|
| 00 | Entorno local y referencia, hashes/versiones | Comparacion de PCs; no demuestra carga sostenida |
| 01 | Imagenes/labels activos → inventario y dataset_validation.json | Formatos, clases, cajas, duplicados, conteos, hashes |
| 02 | Dataset validado + seed predeclarada → split 70/20/10 y test_lock | Unidad imagen original; split sellado e inmutable |
| 03 | Split + dataset → folds internos y image_manifest | Folds solo dentro de train; agrupacion por imagen |
| 04–05 | Manifiestos → detector de fruto y crops GT etiquetados | Regla estricta cobertura >0.75, sin mezcla con detecciones |
| 06 | Baseline por arquitectura en folds internos | No selecciona con test |
| 07 | Estudio Optuna por arquitectura; TPE/MedianPruner | 15 trials detector /20 clasificador, n_jobs=1 |
| 08 | Top-3 distintos ×3 seeds ×3 folds → best_config | Seleccion por media; no por mejor seed aislada |
| 09 | best_config + train externo completo → final.pt/meta | Numero de epocas fijado por folds; sin elegir con validation externo |
| 10–11 | Checkpoint final → validacion, predicciones y umbrales | Solo validation; hashes de pesos/datos/configuracion |
| 12–14 | Candidatos completos → seleccion y pipeline | Cinco detectores y tres clasificadores; test protegido |
| 15–18 | Decisiones confirmadas → freeze, test, benchmark y tablas | Guardias heredadas de test/freeze se conservan |

La orquestacion original usa trabajos independientes por arquitectura, no DDP de un modelo. SQLite es local por estudio/cola. Los equipos no tienen un estado compartido en tiempo real. Un lock en una base local no puede excluir a otra base desconectada.

## Defectos confirmados, causa y correccion

| Defecto observado | Causa e impacto | Cambio y verificacion |
|---|---|---|
| Duplicacion entre colas independientes | first_batch era preferencia; todas las colas podian reclamar cualquier baseline | Propietario unico, validacion de cobertura y filtros obligatorios; test con tres bases separadas |
| Arquitecturas retiradas aun operativas | Registro, YAML, CLI, tests y tareas IDE incluian modelos excluidos | Registro activo de ocho; validacion rechaza su reactivacion y mantiene YOLO26n |
| Lock GPU susceptible a carreras | Leer PID y escribir archivo no era exclusion atomica | Lock del SO con descriptor retenido; guards compartidos bajo s2_no_borrar para copias vinculadas |
| Importacion destructiva | Se sobrescribian runs, reports y Optuna aun sin force | Preflight de todos los conflictos; actualizacion solo de version extranjera importada intacta |
| Cola ajena tomada como autoridad | Rango COMPLETE > otros estados podia dar por hecha una dependencia sin archivos | JSON extranjero solo como evidencia; merge_state deshabilitado |
| Backup Optuna no consistente | Copia directa de .db activo podia omitir WAL/transacciones | API SQLite backup, integridad, timeout y cierre explicito |
| ZIP final visible antes de validarse | Escritura directa y hash previo de archivos mutables | Objetos confirmados, .partial, verificacion SHA/CRC al leer, fsync y publicacion final |
| Entradas ZIP inseguras | No se validaban traversal, duplicados/colisiones Windows ni destinos enlazados | Rutas permitidas, nombres Windows/ADS/symlinks/miembros extra rechazados |
| No habia exportacion horaria automatica | pack-results manual y sin sesion de inicio | Lanzador, watcher independiente y tarea Windows con recuperacion al iniciar sesion |
| Perdida de expansion HPO tras corte | COMPLETE podia quedar persistido antes de crear hijos | Reconstruccion idempotente de jobs de robustez al reiniciar |
| Reintentos indefinidos/manuales sin tope | FAILED podia volver siempre a PENDING | Hasta tres intentos de job; limite Optuna heredado y salidas con detalle |
| Resultados antiguos reutilizados por existencia | Repeticiones/final solo comprobaban parte de su identidad | Verificacion de hp/seed/config/dataset/split/codigo y hash de checkpoint |
| Memoria de crops replicada en Windows | Todos los crops decodificados dentro del Dataset se serializaban a workers spawn | Lectura a demanda con misma conversion RGB; equivalencia con transformaciones pendiente de QA ML |
| Fallback OOM se describia como equivalente | Microbatch cambia BN, particion de muestras, ponderacion y recorrido de augmentations | Oficial detiene OOM. No se afirma equivalencia por nbs/acumulacion; queda decision cientifica |
| Detector fijo podia publicar best.pt | La eleccion de archivo usaba best aun con fixed_epochs | El modo fijo publica last.pt; implementacion de ultima epoca requiere QA con version fijada |
| Seleccion podia mezclar recalls distintos | reporting usa metricas de 10 si falta 11 | Se exigen umbrales/checkpoint/procedencia completos antes de comparar |
| Configuracion efectiva sin huella portable | Rutas distintas no deben invalidar codigo comun, pero overrides cientificos deben sellarse | science_hash en estudio, run, best_config, checkpoint, evaluacion y freeze; tests negativos de cambios efectivos |
| Predicciones alteradas podian afinar umbrales | No habia hash de predicciones enlazado a 10 | 10 publica SHA; 11 lo exige junto a identidad de datos/pesos |

Durante la verificacion aislada aparecieron dos incompatibilidades Windows corregidas: el contexto sqlite3 confirma/revierte, pero no cierra la conexion; y fsync requiere descriptor con acceso de escritura en esta ruta de Windows. Las pruebas ejercitan backup WAL, publicacion y recuperacion real de archivos temporales.

## Optimizaciones implementadas y evidencia

| Cambio | Documentado | Observado en codigo | Verificado aqui | Hipotesis o limite pendiente |
|---|---|---|---|---|
| Crops a demanda | Spawn puede replicar objetos del Dataset; pinning/prefetch son opciones del loader | Lista completa de PIL.Image mas workers persistentes | Revision de ruta de lectura y AST | RAM/throughput/equivalencia del pipeline ML en cada PC |
| Snapshots por contenido | Copias inmutables y backup SQLite permiten lectura consistente | Exportacion original leia archivos activos | ZIP/import/WAL/cortes en fixtures aisladas | Coste de snapshots y plazo con archivos reales |
| Menos I/O redundante | Sin necesidad de transferir caches/dataset | Se incluian runs/figuras y opcionalmente pesos | Tests excluyen dataset/cola DB y conservan dependencias | Tamano real de checkpoints y tiempos de disco |
| Propiedad exclusiva | Sin coordinacion online se necesita particion o protocolo de handoff | Preferencias locales no excluian duplicados | Tres colas independientes sin interseccion | Balance de duraciones con hardware real |
| Reuso de folds terminados | Persistencia atomica de resultados confirmados | Baseline/repeticiones repetian folds tras corte | Identidad/CSV/SHA revisados; ruta ML sin ejecutar | Equivalencia numérica y recuperacion integrada con ML local |

Se conservan batch oficial, precision fp16, seeds, augmentations, espacios HPO, trials, split y test. Los workers/prefetch/pinning/persistent_workers mantienen el comportamiento configurado; no se elige un numero optimo sin medir. No se habilitan compile, TF32, cudnn.benchmark, BF16, caches RAM, hardlinks ni cambios de batch automaticamente. Pueden cambiar reproducibilidad, capacidad o E/S; su conveniencia depende de QA. El ahorro de RAM no demuestra una ganancia de velocidad en disco lento.

No se hizo overclock, cambio de voltaje/BIOS, ajuste de potencia ni desactivacion de protecciones. El monitor exige limites y fuentes verificados. Ante peligro bloquea/detiene; no intenta mantener utilizacion arbitraria. Sin sensor CPU requiere declaracion explicita de limitacion y verificacion OEM. La seguridad fisica y la estabilidad sostenida no se certifican con un smoke.

## Distribucion y limites del paralelismo

Se conserva la asignacion existente despues de retirar los dos modelos: PC-01 tiene YOLOv8n/YOLO26n; PC-02 YOLOv8l/YOLO26l; PC-03 RT-DETR-R50/ResNet50/ViT-B/16/Swin-Tiny. El mapa entre esas PCs e i21315/i21314/i21313 se crea por decision explicita del operador y queda sellado por hash. No se usa el hostname para inferirlo.

Cada propietario agota sus trabajos disponibles y sus dependencias locales. La exportacion no detiene el worker. Importar resultados ajenos no convierte al receptor en propietario de un estudio. Si falta un archivo, se registra ruta/dependencia; se siguen otros trabajos. Hay cinco consultas de pendientes como maximo antes de salir.

Una PC que termina su asignacion puede quedar libre mientras otra sigue trabajando. No se implemento robo de trabajo offline: sin una cesion exclusiva confirmada se arriesgaria duplicacion. No existe evidencia para redistribuir cargas con criterio de rendimiento. Ese criterio de aprovechamiento global sigue pendiente de mediciones y, si procede, un handoff seguro autorizado/planificado. No se afirma utilizacion maxima de las tres. El QA inventaria todas las GPUs pero bloquea cargas si hay varias visibles o un remapeo CUDA heredado: el protocolo/fixtures actuales requieren validar primero ese aislamiento y su correspondencia UUID; no se supone una topologia.

## Exportacion, consistencia y contenido

Los entrenadores publican snapshots entre jobs y, para HPO, despues de confirmar un trial. Los objetos de contenido son inmutables y se deduplican. Cada publicacion captura resultados de la arquitectura, DB Optuna respaldada, configuracion congelada, validacion/umbrales y final.pt/meta cuando son coherentes. Los archivos son seleccionados por dependencias reales; el dataset, derivados, caches, figuras regenerables y pesos de folds activos quedan fuera.

El timer lee las ultimas publicaciones confirmadas y agrega evidencia de cola/entorno/configuracion. No copia DB de cola ni un checkpoint mientras se escribe. El JSON de cola puede mostrar un job activo y los recibos una etapa anterior: eso representa explicitamente progreso confirmado frente a trabajo incompleto, no un supuesto checkpoint de ese job.

La sesion conserva 21:00 y 22:00 del dia de inicio. El watcher independiente sigue aunque el worker termine. Se intenta registrar tarea limitada Windows con inicio a las 21:00 y al iniciar sesion, sin credenciales guardadas. Fallos de registro, reloj/zona, inicio tardio, espacio, permisos, locks, validacion y plazo se registran. Un inicio posterior a 22:00 intenta preservar un ZIP valido y marca incumplimiento.

Se publica nombre final solo tras validacion de cada miembro y SHA del ZIP, sin sobrescribir nombres previos. Progreso/estimacion de copia y riesgo de plazo quedan en last_export_progress.json. Esa estimacion excluye verificacion final y no garantiza el plazo. Los apagados, disco lento/fallido, falta de sesion interactiva, bloqueo del SO y tamanos reales pueden impedirlo. El registro/ejecucion reales del Programador de tareas siguen pendientes.

La importacion valida formato, SHA externo/individual, nombres, propietario, mapa y hashes de ciencia/codigo/datos/splits/folds. Hace staging completo y preflight de conflictos antes de escribir. Un journal permite continuar una publicacion parcial tras corte sin repetir resultados. El origen puede actualizar solo su ultima copia importada intacta; una modificacion local diferente bloquea. Seleccion/lectura bloquean journals pendientes. No existe sobrescritura force en v2.

Los hashes detectan corrupcion, no autentican criptograficamente a una persona o equipo. Se usa procedencia declarada en el mapa y transporte manual de confianza. Los ZIP antiguos no se importan automaticamente. No se incluyen sus datasets de nuevo. Recibos/objetos previos no se borran automaticamente; hay que prever espacio para conservar versiones y evidencia.

## Recuperacion y exactitud cientifica

Se conservan trials COMPLETE/PRUNED y su numeracion Optuna. Un trial RUNNING abandonado se marca FAIL mediante API publica, bajo exclusividad local, y se repone dentro del presupuesto segun politica existente. No se prometen los mismos siguientes puntos TPE tras reiniciar: el framework recrea el sampler con seed y no persiste su estado RNG interno. Ese limite heredado se documenta y debe considerarse al interpretar reproducibilidad tras cortes.

Baseline y robustez reutilizan folds completamente confirmados si coinciden identidad de codigo/datos/hp/seed/presupuesto y hash del historial. No reutilizan folds de otro trial podable. Repeticiones/resultados finales incompatibles se bloquean. Los jobs se expanden idempotentemente y tienen limite de intentos.

No hay reanudacion exacta de mitad de epoca del clasificador ni del trial interrumpido: faltan estados completos de optimizer/scaler/RNG/loaders y contrato de resume en el loop original. Ultralytics tiene resume, pero el framework no encadena automaticamente un checkpoint activo con un trial recuperado. Su integracion exacta requiere cambios/versiones y validacion especifica. Se preserva lo confirmado y se informa la unidad incompleta que se repite.

La huella cientifica excluye rutas, hardware local y el acto administrativo confirmed_by_user; incluye clases, seeds, augmentations, workers, presupuestos y decisiones concretas. Un freeze sin esa huella o con configuracion efectiva diferente se bloquea. No se confirman decisions.adopted_from_annex ni se aplica una nueva particion. Se bloquea top-k incompleto en vez de reducirlo o incrementar trials. Correcciones de procedencia/checkpoints/seleccion hacen cumplir la metodologia declarada. Cualquier resultado anterior debe conservarse como una version distinta, no compararse como si hubiera sido producido con este codigo.

## Evaluacion de YOLO26n

**Recomendacion: mantenerlo por ahora; exclusion pendiente de autorizacion.** Conserva la comparacion entre familias en escala n, junto a YOLOv8n, y una escala l por familia. Eliminarlo dejaria la familia YOLO26 representada solo por l. Eso cambia la pregunta experimental y reduce la comparacion por escala. No hay metricas ni tiempos reales con que demostrar redundancia o coste innecesario en este dataset.

Con la configuracion nominal: baseline 3 fits; HPO hasta 15×3=45 fits si todos los trials llegan a los tres folds; robustez 3×3×3=27 fits; candidato externo 1 fit. Total nominal hasta 76 fits y hasta 300+2700+1620+60=4680 epocas de fold/candidato, excluyendo fallos/reintentos y con reducciones por poda/early stopping. No son horas, resultados ni trabajo ya realizado. El numero final de epocas externo se deriva de los folds.

Sus 15 trials son 15/135 del presupuesto HPO activo total (75 detector +60 clasificador); no equivalen a esa proporcion de tiempo. Antes de excluirlo, compare tiempos/VRAM observados y el valor cientifico de la comparacion n contra n. No se agrego ninguna arquitectura.

## Fuentes oficiales consultadas

- [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html): funcionamiento de workers, spawn, pinning, prefetch y persistencia. Sustenta investigar replicacion de memoria y conservar control de RNG; no ofrece un numero universal de workers para estas PCs.
- [PyTorch carga de datos](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html): opciones de E/S y procesos; su beneficio necesita medicion local.
- [PyTorch reproducibilidad](https://docs.pytorch.org/docs/2.14/notes/randomness.html): seeds y determinismo requieren cuidar operaciones y entorno; no se presupone equivalencia entre hardware/versiones.
- [Ultralytics entrenamiento](https://docs.ultralytics.com/modes/train/) y [trainer](https://docs.ultralytics.com/reference/engine/trainer): contrato de checkpoint/resume y argumentos. La documentacion actual puede diferir del pin; QA debe comprobar la version instalada y la ruta real de entrenamiento.
- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/): familia y escalas disponibles. No se trasladan sus benchmarks externos a este dataset/equipos.
- [Optuna FAQ](https://optuna.readthedocs.io/en/stable/faq.html) y [RDBStorage](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html): cautelas de SQLite y paralelismo/heartbeat. Se mantiene un escritor por estudio local; no se fusionan estudios ni se usa un SQLite de red.
- [sqlite3 backup](https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.backup): respaldo coherente mediante API, con cierre explicito de conexiones.
- [NVIDIA nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html): telemetria y limites dependen del producto; valores no soportados no se convierten en limites inventados.
- [WSL oficial](https://learn.microsoft.com/en-us/windows/wsl/install) y [Git for Windows](https://git-scm.com/install/windows): Bash requiere instalacion/entorno reales. Se entrega deteccion, no se da por instalado.

- [Esquema oficial del Programador de tareas](https://learn.microsoft.com/en-us/windows/win32/taskschd/task-scheduler-schema) y [StartWhenAvailable](https://learn.microsoft.com/en-us/windows/win32/taskschd/taskschedulerschema-startwhenavailable-settingstype-element): se valido estaticamente el XML generado contra el XSD. El registro real no se ejecuto.

Consulta: 2026-10-28. Se distingue documentacion, codigo observado, tests aislados y pendientes locales en este informe. Ninguna fuente general reemplaza la comprobacion de las tres PCs.
