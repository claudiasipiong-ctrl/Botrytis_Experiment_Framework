# Catalogo regenerable y contrato de fuentes

60items canonicos y8aliases en configs/figure_catalog.json. El generador real es tools/generate_all_figures.py;
lanzador independiente ../generar_graficos_botrytis.py. No necesita pesos para graficar CSV reales ya registrados.
Los plot_* existentes siguen disponibles; el generador central usa contratos explicitos para cubrir catalogo completo.
Sin observaciones se escribe PENDIENTE:datos insuficientes con artefacto/campo/comando requerido, sin archivos de
lineas ficticias. Los cuatro esquemas entregados son diagramas, no resultados medidos.

## Flujo de fuentes

1. Usuario ejecuta su auditoria/experimentos y conserva CSV/JSON/Optuna/historias con huellasv4.
2. tools/prepare_figure_sources.py --config configs/local.yaml normaliza inventarios, crops, pares y tablas disponibles.
3. Para fuentes restantes registre un descriptor en reports/v4/figure_sources.json; la clave corresponde a table
   de cada item del catalogo. Es la misma tabla para aliases y graficos relacionados. No se mezclan datos de fases.
4. tools/generate_all_figures.py --source-index reports/v4/figure_sources.json. Para ejemplos visuales el usuario
   agrega --allow-pixels; para resultados test requiere --allow-final-test y freeze/release validos. Por defecto no se lee test.

Descriptor por tabla (todos obligatorios):
```json
{"path":"ruta/tabla.csv","source_sha256":"SHA256_REAL_64_HEX","dataset":"hash_dataset_real","split":"inner_validation",
 "phase":"robustness","metric":"map50_95","unit":"fraction","denominator":"numero_real_de_GT_o_runs",
 "configuration":"config_id_real_y_huella","semantic_version":"botrytis-physical-fruit-pairing-v4"}
```
El placeholder es documentacion, no un resultado. sources deben proceder de observaciones reales.
Si un CSV contiene varias fases/splits/configuraciones el generador rechaza la agregacion; separar cohortes.
Inventarios pueden mostrar split como eje descriptivo sin agregar metricas de entrenamiento/evaluacion.
Para tablas no producidas directamente, derive CSV con un script y guarde upstream con rutas/hashes del JSON,
Optuna o predictions. Se registran source_sha256 y .data.csv; los PNG/SVG/PDF reciben hashes en figure_manifest.

## Columnas por visualizacion

El catalogo contiene table,x,y,group,unit,kind y required_artifact. Es el contrato preciso para barras,
histogramas,scatter,burbujas,Pareto,radar,boxplots,intervalos,heatmaps,coordenadas paralelas y matrices.
comparison: architecture,map50_95,map50,precision,recall,f1,latency_ms,params_m,gflops,model_size_mb,
train_seconds,vram_peak_mb,accuracy (solo si se compara accuracy de clasificacion; no confundir con mAP).
Las tablas detector/clasificador se mantienen separadas; use la fuente adecuada para cada catalogo/estimador.
trials: architecture,trial,score; fold_metrics:architecture,fold,map50_95; robustness:architecture,config_id,seed,score,map50_95.
hyperparameters:trial,parameter,value (codificaciones categoricas predeclaradas en descriptor; no inferidas).
baseline_hpo:architecture,baseline,hpo, ambos del mismo estimador interno. No restar HPOmax a mAP externa.
Histories:run_id,epoch,fold,seed y metricas; classifier_history usa ap_positive,accuracy,f1 (aliases de val_ap,val_accuracy,val_f1).
selected_detector_history conserva componentes autenticos train/box_loss,val/box_loss,train/cls_loss,etc y map50_95.
selected_classifier_history:train_loss,val_loss,ap_positive. Debe ser configuracion seleccionada y validacion interna.
No existen curves nuevas del hibrido. La media±DE se hace por epoca sobre runs comparables; N real debe mantenerse.
Early stopping cambia N: conservar .data.csv y definir criterio de comparabilidad; no interpolar epocas ausentes.
APporIoU:architecture,iou,ap. ROC:fpr,tpr; PR:recall,precision. Thresholds:threshold,f1,recall/precision.
Matrices:actual,predicted,count, con todos los ceros estructurales explicitamente registrados; pipeline incluye no_detectado.
pipeline_errors:component,error_rate con denominadorGT y FP por separado. pipeline_latency:stage,latency_ms.

## Fuentes visuales, arquitectura y estratos

Ejemplos: image_id,image_path,image_sha256,annotation_json (lista de {xyxy:[x1,y1,x2,y2],class_id,label}).
Grad-CAM debe apuntar a overlays ya calculados por utils/gradcam/evidence, con regionesclase2 cuando existan.
No se abren pixeles sin --allow-pixels. Las fuentes/preprocesamiento/checkpoint/capa deben quedar en descriptor.
Stage-features:architecture,stage,response,input_id y model_sha256; tools/extract_stage_features.py usa inputs
predeclarados no-test y mapping decapas con equivalence_rationale. La respuesta media absoluta no es accuracy.
tools/inspect_model_graph.py inventaria named_modules de arquitectura scratch, sin confundirla con un modelo entrenado.
CIB/MHSA son referencias conceptuales explicitamente rotuladas; no afirman presencia o equivalencia CIB/C2PSA/AIFI.
Fuentes conceptuales:https://arxiv.org/abs/2405.14458 y https://arxiv.org/abs/1706.03762.
YOLO26yaml oficial:https://github.com/ultralytics/ultralytics/blob/v8.4.170/ultralytics/cfg/models/26/yolo26.yaml

Estratos difficulty,density,visual_conditions exigen strata_definition_source y origen de metadatos fiable.
Predeclarar categorias/cortes antes de mirar metricas. La densidad se calcula del GT; iluminacion/oclusion no
se deducen subjetivamente. Sin metadatos: PENDIENTE, nunca barras inventadas.
Latencias exigen latency_protocol:{hardware,batch,resolution,warmup,cuda_sync:true,precision,include_disk_io}.
Hardware distinto o falta de sincronizacion bloquea el grafico. Los tiempos incidentales de otrasPCs no son benchmark.

## Salidas y limites actuales

PNG300dpi,SVG,PDF cuando hay datos y matplotlib disponible; .data.csv y manifiestoJSON/CSV con ID,titulo,
fuente/hash,split,fase,metrica,unidad,denominador,configuracion,fecha,estado,ruta/hash. Los esquemasv4 tienenSVG
y se exportan PNG/PDF si hay backend de render disponible. Plotly es opcional; no se requiere Internet para leer salidas.
Las fuentes reales de curvas/latencias/errores/activaciones permanecen PENDIENTE por falta de entrenamiento propio.
Los fixtures sinteticos de dev_tests no se incorporan a resultados ni al catalogo oficial.
No se generaron estimaciones exploratorias. Si se solicitan, deben ir solo en ../estimaciones_exploratorias/,
rotuladas ESTIMADO/ILUSTRATIVO con supuestos y sin mezclarlas en este catalogo oficial.
