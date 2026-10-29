# Pesos, fuentes y decisiones manuales

Estado inicial: cero pesos. Los diez registros comienzan NO_DISPONIBLE. El gestor distingue disponibilidad de fuente
(DESCARGABLE / NO_EXISTE_FUENTE_FIABLE) y estado local (NO_DISPONIBLE / DESCARGADO_VERIFICADO /
GENERADO_POR_ENTRENAMIENTO). Nunca toma caches ni nombres .pt como prueba de preentrenamiento.
INVENTARIO_PESOS_Y_FUENTES.csv y configs/weights_registry.json fijan modelo, backend/version, URL, licencia,
tamano/checksum publicado y ruta. docs/SOURCE_METADATA.json conserva respuestas de metadatos consultadas el29/10;
no se descargaron archivos de pesos. El tamano de ResNet50 queda pendiente hasta adquisicion/verificacion manual;
su nombre oficial publica el prefijo SHA25611ad3fa6. Eso es menos fuerte que un hash oficial completo: se declara
y se conserva el SHA256 completo calculado, sin presentarlo como hash completo publicado.

## RT-DETR-R50: BLOQUEADO en pretrained

Existe checkpoint autentico R50-vd en el repositorio oficial:
https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch
https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth
172373587bytes segun API de release, Apache-2.0 original. No se verifico aqui su contenido ni se descargo.
El backend heredado usa ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml, ResNet con activacionesSiLU;
no es el mismo backboneR50-vd original. El checkpoint oficial no es directamente compatible con ese loader.
No existe una fuente fiable demostrada de checkpoint COMPLETO de ese backend. Un backboneImageNet con
encoder/decoder aleatorios es pretraining parcial diferente de COCO y ya no se acepta como equivalente.
RT-DETR-L tiene HGNetv2 y no es R50. No se utiliza rtdetr-l.pt como sustituto.

Alternativas para decision manual, sin ejecutar ninguna ahora:
1. Adoptar backend oficial R50-vd y validar un adaptador de entrenamiento/evaluacion/metricas antes de comparar.
   El backend oficial tiene entorno y configuraciones propios; no se prometio compatibilidad con el Python local.
2. Predeclarar variante Ultralytics ResNet50 scratch en experimento separado, con initialization.scratch_decision
   e initialization.rtdetr_variant_decision. Compararla solo bajo un protocolo de scratch comun/documentado;
   no llamarla reproduccion exacta de R50-vd COCO. No altera silenciosamente la definicion primaria.
Mientras se decide, la seleccion oficial de los cuatro detectores y el freeze completo permanecen bloqueados.

## Clasificadores

ResNet50: torchvisionResNetV1.5 IMAGENET1K_V2. ViT-B/16:timm vit_base_patch16_224.augreg_in21k_ft_in1k,
revision fija; ImageNet21k→1k. Swin-Tiny:timm swin_tiny_patch4_window7_224.ms_in1k, revision fija;ImageNet1k.
La diferencia de datasets pretraining se documenta al comparar; no se presenta como condicion identica.
Se carga state_dict estricto con cabeza1000 antes de reemplazarla por2, preservando transformaciones declaradas.
Nunca se permite una llamada pretrained=True que descargue implicitamente.

## Comandos personales

tools/manage_weights.py inventory: no descarga. acquire requiere --allow-network y --accept-license,
valida URL fijada, tamano y SHA oficial completo cuando existe. Descarga a .partial, conserva fallos y nunca
sobrescribe un peso existente. ResNet usa prefijo oficial y registra SHA completo local con ese origen limitado.
verify valida transferencia local y crea procedencia. generated registra checkpoints posteriores de un run COMPLETE.
Los loaders re-calculan el hash, no usan cache de hash. Nunca calculan metricas con pesos ausentes.
Los comandos listos estan en la GUIA/COMANDOS_POWERSHELL. Sin red/pesos: PENDIENTE_USUARIO.
La aceptacion de licencias y permisos de la organizacion corresponde al usuario; no se incluyen tokens.

Resume: tools/resume_local.py requiere checkpoint generado y recibo compatible .resume.json.
Detector usa estado nativo Ultralytics; clasificador conserva optimizer/scaler/RNG/generador al limite de epoca.
Reanudar no significa reiniciar desde pesos COCO ni cambiar de fold/seed. Retome el job exacto, con el mismo entorno,
workers y batch; la equivalencia numerica real es QA manual pendiente. Se conservan logs de cada intento.
