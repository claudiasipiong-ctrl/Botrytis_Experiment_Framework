# Botrytis Experiment Framework · siete arquitecturas, rondas supervisadas

Código para validación personal en tres estaciones Windows 10 con una RTX 6000 Ada cada una. **QA manual y experimento científico pendientes.** No contiene datasets ni pesos; no se ejecutaron entrenamientos, HPO ni test científico durante estas correcciones. RT-DETR-R50 preentrenado permanece bloqueado por incompatibilidad de backend/checkpoint, sin sustitución por RT-DETR-L.

| Estación | Identidad | Arquitecturas iniciales |
| --- | --- | --- |
| PC-01 | i21315 | YOLOv8n, YOLOv8l, ResNet50 |
| PC-02 | i21314 | YOLO26l, Swin-Tiny |
| PC-03 | i21313 | RT-DETR-R50 (bloqueado), ViT-B/16 |

Un trabajo GPU secuencial por estación; hasta tres arquitecturas independientes en paralelo. SQLite local por ronda y ZIP + SHA-256 manual, sin servidor, escritorio remoto ni SQLite compartida. La transferencia de ownership exige oferta, recibo, confirmación y activación: no se presume exclusión global transaccional entre equipos desconectados.

## Documentación vigente

- [Guía práctica HTML](docs/GUIA_EJECUCION_3PCS_FINAL.html) y [diagrama horizontal](docs/DIAGRAMA_FLUJO_3PCS_HORIZONTAL.html).
- [Comandos PowerShell por fase y PC](docs/COMANDOS_POWERSHELL_3PCS.md).
- [Entrenamiento iterativo](docs/GUIA_ENTRENAMIENTO_ITERATIVO.md): revisión después de baseline, HPO/bloques y robustez; rondas sin sobrescritura.
- [Hallazgos y correcciones](docs/CAMBIOS_Y_HALLAZGOS.md), [pruebas/QA pendiente](docs/PRUEBAS_Y_QA_PENDIENTE.md) y [tiempos ilustrativos/calibración](docs/ESTIMACION_TIEMPOS_ACTUALIZADA.md).

## Protocolo preservado

70/20/10 por imagen original; split seed 67; seed global 42; tres folds internos; top-3 × semillas `[17,42,73]` × tres folds; 15 trials/detector, 20/clasificador; TPE + MedianPruner. HPO en el 70 %; selección/calibración predeclaradas en el 20 %; test 10 % sellado hasta freeze íntegro y autorización personal posterior. Una nueva ronda no habilita exploración repetida del validation externo.

Clase 0 = fruto físico, clase 1 = sospecha positiva por matching bipartito IoU, clase 2 = auxiliar. IoU técnico 0,50 no es porcentaje de área enferma. No se restauró la regla descartada del 75 %. Las referencias históricas 324/328/329 imágenes requieren inventario personal; no se inventaron muestras, cambiaron etiquetas ni regeneraron splits reales.

## Desarrollo CPU

Python 3.12. Instalar la pareja PyTorch/torchvision de `requirements-ml.txt` desde el índice oficial CPU o CUDA apropiado, después `requirements.txt` y `requirements-dev.txt`. No se descargan pesos al construir modelos. En este entorno cloud: `source /workspace/botrytis-onboarding/activate.sh`.

Las pruebas de control y componentes usan fixtures sintéticos. Las cinco pruebas de entrenamiento son opt-in y no se habilitan para esta entrega. Los resultados actuales, omisiones y limitaciones están documentados en el informe de QA, sin reutilizar PASS históricos.

Los informes de `reports/historical`, `docs/historical_pre_v4` y otros artefactos marcados históricos se conservan como procedencia y no acreditan esta versión. Las guías anteriores de diez modelos quedan sustituidas para ejecución por los documentos anteriores.
