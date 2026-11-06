# Dos sistemas finales en PC-02

Extension expresa para la decision del usuario: A=YOLO26l+ResNet50 y B=YOLO26l+Swin-Tiny.
No es la cohorte original de siete candidatos. El profesor elegira despues de consultar TEST;
no se presenta el ganador como una evaluacion independiente de esa seleccion.

Requiere Windows PC-02/i21314, .venv del proyecto b706f176/e11f772d, datos/split originales,
QA READY_FOR_EXPERIMENT vigente, modelos YOLO26l/Swin locales y ResNet50_para_PC2.zip + huella
en Downloads. Extrae exclusivamente los siete archivos permitidos del paquete; comprueba
los tres checkpoints contra los hashes de las evaluaciones VAL ya recibidas.

Detector: confianza .325, sin NMS para YOLO26l. Matching uno a uno IoU>=.5.
Clasificador A: .69; B: .85. Valores del analisis previo de VAL, F2 exploratorio.
No entrena, reajusta ni elige umbrales usando TEST. Ninguna ronda original se modifica.

Por defecto prepara y congela sin inferencia TEST. El argumento --ejecutar-dos-test es la
orden personal del operador que autoriza ambas pruebas y se registra ligado a cada freeze.
Se utiliza un directorio de artefactos separado por sistema, los verificadores nativos de
freeze/candidatos/split y el recibo de liberacion TEST ligado por SHA256. No se marcan como
completadas gates manuales no realizadas ni se cambia el perfil a smoke.

El evaluador es utils.pipeline_eval.evaluate_pipeline nativo, conservando normalizacion,
recortes, etiquetas y matching. No ejecuta src/16 dos veces sobre la misma ronda.
Se guardan CSV de detecciones, omisiones, errores y tiempos por imagen; matrices/figuras
nativas; FN positivos por deteccion y por clasificacion; FP positivos tanto sobre negativos
GT como detecciones sin GT valido. Los tiempos son de esta ejecucion, no benchmark17.

Salida estable: %USERPROFILE%\bd\dos_test_finales_v1\
A/resultado.json y B/resultado.json; comparacion_TEST.csv e INFORME_TEST.html.
Logs y figuras: runs/ (las rutas se indican dentro de cada resultado).
Si A ya termino, se verifica su freeze y se conserva al volver a invocar; no se reevalua.
Una interrupcion sin cierre requiere --reanudar-interrumpido y mantiene pesos/umbrales
fijos; las evidencias del run anterior quedan guardadas en sus propios directorios.
Si hubo un fallo, conservar todo y revisar el motivo antes de reintentar.

Validacion de entrega: pruebas CPU sinteticas en Linux, nunca datos TEST reales ni GPU Windows.
No se accedio a las PCs; la ejecucion real ocurre cuando el operador invoca el comando.
