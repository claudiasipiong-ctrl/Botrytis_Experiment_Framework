# Paquete final de evidencias para tesis

Usar el comando fijado al commit de entrega en las tres PCs. Utiliza su inventario `Inventario_Tesis_PC-XX_*.zip` más reciente en Downloads y la misma `.venv` del proyecto. El script identifica el usuario y crea un ZIP nuevo, sin sustituir exportaciones anteriores. No entrena ni ejecuta inferencias. No cambia configuraciones, pesos ni estados de selección.

Distribución real revisada:

| PC | Modelos | Evidencia adicional |
|---|---|---|
| PC-01 | YOLOv8n, YOLOv8l, ResNet50 | Entradas originales compartidas, copiadas una sola vez |
| PC-02 | YOLO26l, Swin-Tiny | TEST A: YOLO26l+ResNet50; TEST B: YOLO26l+Swin-Tiny |
| PC-03 | ViT-B/16, RT-DETR-R50 | RT-DETR parcial; incluye el checkpoint con recibo más reciente del último trial observado |

Dentro de cada ZIP:

```
PC-XX/
  00_INDICES/                         Inventario de copias, hash, errores, advertencias y exclusiones
  00_PROYECTO_Y_EQUIPO/               Código real, configuraciones, ayudantes, datos derivados/sellos
  00_DATOS_COMPARTIDOS/               Sólo PC1: imágenes y etiquetas originales
  01_MODELOS/<modelo>/
    01_Configuracion_preparacion_y_otros/
    02_Baseline/
    03_HPO/
    04_Robustez/
    05_Entrenamiento_final/
    06_VAL/
    07_Pesos_y_procedencia/
    08_Auditoria_y_recuperaciones/
  02_SISTEMAS_TEST/                   Sólo PC2: ambos sistemas y sus fuentes/protocolo
  03_AUDITORIA_Y_ANTECEDENTES/         Registros que no se atribuyen inequívocamente a un modelo
  04_APOYO_PARA_REDACCION/             Tablas, Excel, figuras y scripts de las comparaciones VAL
  BITACORA_INICIAL.md
  LEEME.md
```

Se crean únicamente etapas para las que existen documentos. Cada etapa retiene las rutas relativas de origen, que distinguen rondas, trials, folds, semillas, intentos y archivos con el mismo nombre. Las rutas excesivamente largas se acortan conservando la correspondencia en el inventario. No hay que interpretar cada copia histórica como el resultado definitivo: configuración, hash del checkpoint, protocolo y artifacts finales identifican el resultado utilizado.

| Para escribir | Evidencias incluidas |
|---|---|
| Datos y diseño experimental | Imágenes y etiquetas originales; inventarios, pairing/QA, exclusiones, manifest de splits/folds, clases y semillas |
| Reproducibilidad | Código y ayudantes locales, YAML, dependencias/entornos registrados, hardware/software, comandos, commits/diffs, procedencia y pesos iniciales/finales disponibles |
| Baseline e HPO | Historias y métricas por época/fold/trial, estados COMPLETE/PRUNED/FAILED, espacios de búsqueda, Optuna SQLite y estados `.pkl`, tiempos y figuras |
| Robustez | Resultados por configuración, semilla y fold, agregación, selección y épocas definitivas |
| Entrenamiento y VAL | Historias, métricas, predicciones/GT, probabilidades/cajas, matrices y curvas, figuras y pesos finales |
| Sistema integrado y TEST | Protocolo y freezes, resultados A/B, detecciones, omisiones, errores, tiempos por imagen y figuras ya existentes |
| Contratiempos y decisiones | Logs, trazas de error, QA, auditorías, planes de recuperación y bitácora inicial que debe contrastarse con esos registros |

Los originales CSV se preservan sin imputar registros. Las bases SQLite se copian mediante backup consistente y se comprueba su integridad cerrando expresamente todas las conexiones. Es una instantánea individual por base, no una transacción global de toda la máquina.

Para evitar cientos de GB de copias, no se incluyen todos los checkpoints intermedios, los antiguos ZIP ni los objetos internos sin extensión. Se inventarían como exclusiones; sus resultados/documentos sí se conservan. Se incluyen todos los `final.pt` inventariados, los pesos iniciales de `weights/` y el último checkpoint con recibo de RT-DETR, comprobando el hash contra el recibo. No es una copia para reanudar cualquier intento histórico. Se conservan todas las tablas de predicciones y hasta seis ejemplos por carpeta de visualizaciones masivas; las figuras documentales sí se incluyen. Los recortes pueden reconstruirse con las entradas originales y el código.

La bitácora inicial distingue observaciones comunicadas y evidencia a corroborar. No completa fases inexistentes de RT-DETR ni inventa latencia de benchmark. Las tablas de apoyo son de VAL; los archivos originales y los protocolos TEST prevalecen para describir lo realmente ejecutado. Documentar que el profesor consultará los dos TEST para elegir un sistema.

Salida: `Downloads/Botrytis_Tesis_Final/TESIS_PC-XX_<fecha>_<id>/TESIS_PC-XX.zip`, su `.zip.sha256.txt` y una guía. Copiar las tres carpetas de salida a Drive. Conservar los ZIP y hashes. Extraer en una ruta corta si se desea, reuniendo las tres carpetas PC-XX bajo un directorio común. Las rutas Windows internas originales se relacionan con las rutas portables mediante `00_INDICES/INVENTARIO.csv`.

Al terminar, `errors` debe ser 0. Las advertencias indican, por ejemplo, la selección de ejemplos visuales o irregularidades de CSV conservadas sin alterar. Si aparece `INCOMPLETO_REVISAR_ERRORES`, revisar `00_INDICES/ERRORES.json`; no tomarlo por un respaldo completo.

Validación: pruebas sintéticas de clasificación/rutas, instantánea SQLite con WAL, CSV irregular conservado byte a byte, rechazo de colisiones, hash de cada archivo y del ZIP, copia única de entradas originales, ambos TEST, pesos finales y RT-DETR parcial. Auditoría adicional de los tres inventarios reales: sin colisiones de destinos ni ausencia de los archivos finales requeridos. No se ha ejecutado el empaquetado sobre el contenido completo de las PCs desde este entorno; esa verificación se realiza al correr el comando local y queda en sus índices.
