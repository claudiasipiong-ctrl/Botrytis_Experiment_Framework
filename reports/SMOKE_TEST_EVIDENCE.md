> HISTORICO RECIBIDO: las pruebas y el hardware descritos aqui pertenecen al ZIP original. No son evidencia de la revision de 2026-10-28 ni de las tres PCs. Consulte reports/VALIDACION_2026-10-28.md.

# Evidencia de la prueba de humo (SINTETICA, no cientifica)

Dataset sintetico de 120 imagenes; perfil `configs/smoke.yaml`; 1 laptop (RTX 3070 Ti 8 GB); 10 arquitecturas.

## Pasos

| paso | etapa | resultado | segundos |
|---|---|---|---|
| 18 | 18_export_master_results | OK | 18.4 |

# Auditoria cientifica automatica

| # | Comprobacion | Estado | Resumen |
|---|---|---|---|
| 1 | clasificador con GT crops | **PASS** | crops = cajas GT del fruto; entrenamiento sin predicciones del detector |
| 2 | la regla 75 % solo crea etiquetas | **PASS** | etiquetas recalculadas desde anotaciones crudas = manifiesto; la regla no aparece en entrenamiento/inferencia/seleccion |
| 3 | el detector solo detecta frutos | **PASS** | todas las cajas de entrenamiento son clase 0 (fruit); nc=1 |
| 4 | test bloqueado hasta el freeze | **PASS** | solo 16_evaluate_test abrio el test, despues del freeze |
| 5 | sin fuga de datos | **PASS** | split por imagen; crops y folds heredados; sin duplicados entre splits |
| 6 | metricas reproducibles | **PASS** | 11 conjuntos de metricas recalculados desde predicciones guardadas: coinciden |
| 7 | graficos con datos reales | **PASS** | 44/44 figuras obligatorias OK |
| 8 | cada ejecucion deja logs y hashes | **PASS** | 158 runs con execution.log, config.yaml, environment.json, metrics.json, checkpoint_metadata.json y hashes |
| 9 | protocolo HPO | **PASS** | 10 estudios independientes con el presupuesto oficial y robustez top-k x seeds predeclaradas |


Fuente: C:/botrytis_work/smoke (runs/, artifacts/, reports/). Los numeros de metricas de esta prueba no tienen valor cientifico.
Nota: la tabla de pasos corresponde a la ULTIMA invocacion de tools/smoke_test.py (el informe se sobrescribe); la cadena 00-18 se ejecuto en varias invocaciones (la cola de entrenamiento con workers, luego 12-18 y una repeticion 6-18 tras corregir la auditoria). La auditoria de arriba se ejecuto sobre los 158 runs finales.
