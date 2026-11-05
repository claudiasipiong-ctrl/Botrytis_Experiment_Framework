# Continuacion segura: comprobar robustez -> APROBAR -> TRAIN70 -> VAL20

El ejecutor completo es `botrytis_train_val_continuar.py`, junto con los cinco ayudantes inmutables listados en SHA256SUMS.json. Se extraen fuera del checkout. No se cambia el commit instalado b706 ni el dataset, splits, pesos preentrenados o presupuestos anteriores.

## Alcance por computadora

- PC-01: yolov8n, despues yolov8l.
- PC-02: yolo26l, despues swin_tiny.
- PC-03: exclusivamente vit_b16. RT-DETR-R50 queda pospuesto; no se consulta ni modifica su cola, HPO o checkpoints.

La identificacion exige USERNAME, binding y configuracion coherentes con --expected-pc. El entorno virtual debe ser el del proyecto. Se conservan los controles de procesos y locks. No iniciar otra orden si ya se esta entrenando. En PC-03 esperar primero a que termine su robustez de ViT.

## Comprobacion local y revision presencial

Antes de autorizar entrenamientos comprueba baseline/HPO aprobados, identidad de codigo y ronda, dataset/splits, la cola original, nueve repeticiones COMPLETE, sus 27 folds con firmas e historiales, pesos y agregacion nativa. Reutiliza la interpretacion demostrada del callback final de Ultralytics. No repite QA completa, baseline, HPO ni robustez. OFF y _off permanecen excluidos segun la preparacion existente.

Muestra `ROBUSTEZ_VERIFICADA` por cada modelo comprobado, con candidato, media, SD y epocas. Si faltan repeticiones muestra `ROBUSTEZ_PENDIENTE` y los job_id/estados exactos, sin fabricar un cierre ni entrenar ese modelo. Un modelo incompleto no impide continuar otro que este correctamente habilitado.

Genera informes locales bajo `%USERPROFILE%\bv\<id>`. Presenta candidatos, advertencias y limitaciones. Una decision APROBAR real acepta las agregaciones mostradas y predeclara expresamente entrenamiento fijo TRAIN70 y evaluacion VAL20 de esos candidatos. Registra operador, fecha, hashes de informes y candidatos; nunca afirma inspeccion visual humana ni ausencia de sobreajuste. Si ya hay aprobacion/predeclaracion compatible se reutilizan. Sin APROBAR no se inicia la etapa nueva.

Solo restaura outer_train/outer_eval que estaban PENDING antes de una pausa propia por prioridad con recibo exacto. No elimina otros bloqueos, no marca etapas omitidas COMPLETE ni restaura thresholds. No amplia limites de intentos.

## Ejecucion nativa

1. `09_train_outer_candidate.py`: candidato de agregacion, TRAIN70 completo, seed42 y numero fijo de epocas derivado de la robustez. No hay seleccion de epoca con VAL20. Guarda final.pt, final_meta.json, history.csv y train_evaluation.json.
2. `10_evaluate_outer_validation.py`: evalua ese mismo checkpoint en el 20 % reservado. Guarda metricas, predicciones y evidencias nativas de deteccion o clasificacion. No altera hiperparametros ni repite una evaluacion externa completada.

Se usan el worker real y sus dependencias. Los procesos hijos solo admiten outer_train/outer_eval y modelos asignados a la PC. El progreso se muestra en CMD con las lineas reales del worker y del entrenamiento. Se conserva la recuperacion de checkpoints propia del framework y la correccion de RNG de clasificacion en CPU. Estados incompletos/corruptos o una publicacion de checkpoint no recuperable se detienen con evidencias conservadas; no se reinicia un entrenamiento a escondidas.

La validacion externa consume una operacion predeclarada del framework; un resultado completo compatible se reutiliza, no se ejecuta de nuevo. Un uso externo interrumpido solo se reintenta mediante la recuperacion nativa y comprobacion del mismo checkpoint. No se autorizan retrospectivamente resultados sin predeclaracion.

Al terminar muestra TRAIN_VAL_COMPLETOS y FIN_TRAIN_VAL; guarda resumen.html/resumen.json y las rutas de los resultados. Compara TRAIN y VAL del mismo checkpoint y describe las brechas; no diagnostica sobreajuste solo por una cifra alta. Baseline/HPO/robustez siguen siendo CV interna, no una evaluacion externa comparable del baseline. No declara ganador global ni ejecuta calibracion, seleccion global, freeze o TEST10.

`--check-only` permite consultar sin aprobaciones ni entrenamientos. `--only-model yolov8n` limita la consulta/continuacion a ese modelo. No hacen falta esas opciones en el comando normal: comprueba los modelos de la PC y continua solo los listos.

## Verificacion

Pruebas CPU y dobles explicitos de ejecucion neuronal. Incluyen cola SQLite nativa, aprobacion real simulada solo en fixtures, ausencia/rechazo de aprobacion, identidad del candidato, restauracion selectiva de pausas, gate de nueve repeticiones, un modelo incompleto independiente de otro listo, limites de etapas/modelos en hijos, idempotencia y ejecucion de los main reales 09/10 con backend neuronal falso. No se ejecutaron GPU ni Windows reales en la nube. Los discos de las PCs se verifican localmente al ejecutar.
