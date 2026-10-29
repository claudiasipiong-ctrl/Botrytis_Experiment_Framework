# Protocolo y contratos v4

La solicitud del 29/10 prevalece sobre texto/HTML previos. QA_MANUAL_PENDIENTE y EXPERIMENTO_PENDIENTE.

## GT y auditoria

0=arandano_a representa TODOS los frutos fisicos y es el unico GT del detector de una clase fruit.
1=arandano_sospechoso es anotacion adicional sobre el mismo fruto. 2=signo_sospechoso_botrytis es auxiliar.
El clasificador recibe exclusivamente recortes de la bbox0. Positivo con_signos iff matching valido 0↔1;
negativo sin_signos solo si no existe par y la imagen pasa auditoria. Ninguna cobertura determina la etiqueta.

utils/pairing.py resuelve asignacion bipartita de maxima suma de IoU con dummies para no-asignacion.
Las IoU float se codifican exactamente como enteros de denominador comun. El objetivo secundario ordinal
es lexicografico y no puede cambiar el optimo de IoU. Umbral tecnico inicial >=0.50 configurable en pairing.min_iou.
Identificadores ordinales por clase/imagen; el desempate no depende del orden de entrada.
Todos los pares candidatos, IoU, cajas, seleccion, duplicados, positivos sin par, conflictos y empates se exportan.
La politica por defecto bloquea la imagen/etiquetas ante anomalias. quarantine_images es una decision explicita
que conserva decisiones y denominadores; exige revision humana, nunca vuelve negativos los casos conflictivos.
Clase2 ausente en un positivo no es un error automatico. Cobertura individual maxima se conserva solo como descripcion.
Revisar personalmente histogramas de IoU y candidatos cerca de 0.50 antes de aceptar el GT. No optimizar ese umbral con test.

La auditoria completa del usuario verifica imagenes reales por stem, labels vacios, rangos, duplicados y hashes;
src/19_audit_labels.py puede leer labels sin pixeles; src/01_validate_dataset.py decodifica y audita cuando el usuario lo ejecuta.
En este encargo solo se listaron nombres/metadatos de la carpeta activa: 324 imagenes y 324 labels; referencia329.
Esos numeros no son auditoria de aceptacion. No se modificaron datos ni se abrio el test.

## Split, HPO y prevencion de fuga

Split externo 70/20/10 por imagen original ANTES de crops, seed67. El numero real de imagenes manda;
329 es control configurable que genera advertencia de discrepancia, no constante del algoritmo.
assign_outer_split ordena IDs y estratos, baraja con random.Random(67), usa cuotas globales por mayor resto
y reparte grupos pequenos primero con capacidad restante. Estrato has_positive_signs significa positivos por matching,
no existencia de clase2. Con 329 hipoteticas las cuotas son230/66/33; no se genero ese split real aqui.
Revisar soporte por estrato/clase en cada particion; estratos escasos pueden no estar en las tres particiones.
Tres folds internos solo en el70%, agrupados por original_image_id; todos los crops heredan split/fold.
Detector balancea cuantiles de frutos por imagen; clasificador presencia positiva (parametros en inner_hpo).
Casi-duplicados deben revisarse para evitar fuga; no se deducen condiciones visuales por opinion del modelo.

HPO independiente por arquitectura,15 trials/detector y20/clasificador; COMPLETE/PRUNED consumen presupuesto,
FAIL se reintenta con limite explicito y permanece en evidencia. Baselines y todos los trials/folds tienen configuracion,
tiempos, memoria y metricas. Top3 configuraciones se repiten con seeds17/42/73 sobre folds internos.
El mejor HP y numero de epocas derivan del procedimiento interno. El candidato final se entrena sobre70% a epocas fijas,
sin mirar20% por epoca. El20% se usa para seleccion y umbrales; nunca para HPO.
No se deben inferir curvas val externas por epoca para el candidato final: las dos figuras principales vienen de
historiales internos comparables de la configuracion seleccionada, conservando N por epoca y censura por early stopping.
La media±DE es dispersion observada, no intervalo de confianza. No mezclar fases/splits/configuraciones/pretraining.

## Inicializacion y arquitecturas

7 detectores: YOLOv8n/m/l, YOLO26n/s/l, RT-DETR-R50;3 clasificadores: ResNet50,ViT-B/16,Swin-Tiny.
El primario es pretrained. Inicializacion desde cero solo en experimento alternativo predeclarado y namespaces nuevos.
Ningun loader descarga pesos. R50 completo compatible esta BLOQUEADO; consulte PESOS.md, no use RT-DETR-L.
Los graficos CIB/MHSA entregados son referencias conceptuales, no componentes medidos. YOLO26 usa C2PSA en su YAML oficial;
CIB no es C2PSA. ViT usa autoatencion; RT-DETR usa AIFI/decoder; Swin usa ventanas desplazadas.
La comprobacion de named_modules de las versiones instaladas se hace con inspect_model_graph.py; queda pendiente.
Stage-wise requiere entradas predeclaradas no-test, checkpoints autenticos, capas y justificacion de equivalencia.
Las magnitudes de activacion dependen de normalizacion y arquitectura; no equivalen a precision.

## Evaluacion y fronteras

Detector: precision/recall/F1/FP/FN, mAP@0.5 y mAP@0.5:0.95 principal, AP por IoU0.50...0.95 y latencia.
AP de entrenamiento del backend se identifica como tal; la implementacion NumPy usa interpolacion de101puntos,
por lo que no se mezcla sin identificar con el integrador Ultralytics. Clasificador: accuracy,precision,recall,F1,
AP positiva (objetivo HPO), ROC-AUC, PR/ROC y classification report. AP escalonada no es identica al area trapezoidal PR.

Matching detecciones↔GT se hace1:1 greedy por confianza con desempate estable e IoU matching.threshold=0.5;
es distinto del matching global de identidad de anotaciones y del IoU de NMS. Cada GT esta asociado o perdido una vez.
E_det=FN_GT/N_GT; E_cls=GT_detectados_mal_clasificados/N_GT_detectados;
E_global=(FN_GT+GT_detectados_mal_clasificados)/N_GT = E_det+(1-E_det)*E_cls.
Si N_GT=0 la medida es no definida; si no hay matches E_cls no definida y E_global=1 cuando N_GT>0.
FP y duplicados se reportan aparte y en fila fondo; no se suman de nuevo al error orientado a GT.
Matriz integrada: filas con_signos/sin_signos/fondo, columnas predcon_signos/predsin_signos/no_detectado.
No hay loss del hibrido: la inferencia es secuencial y no se entrena conjuntamente.

## Bloqueo e invalidacion

Semantica botrytis-physical-fruit-pairing-v4 y huella de clases/pairing/split/folds/inicializacion.
Nuevos artifacts/results/runs/reports y derivados usan namespacesv4. Los historicos permanecen intactos como obsoletos.
Los consumidores de manifiestos verifican sellos de semantica/datos/split y hashes de contenido. Los estudios Optuna y
mejores configuraciones exigen huella compatible. No se regenera ni reordena un split valido automaticamente;
--force de02 conserva superseded solo bajo cambio documentado por el usuario.
Test requiere freeze integro, candidatos sellados y manual_test_release ligado al SHA256 del freeze.
Los gates operativos son aprobaciones personales por PC; pruebas de desarrollo o logs antiguos no los aprueban.
