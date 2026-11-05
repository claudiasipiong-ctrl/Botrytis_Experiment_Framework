# Reanudar la robustez original

Ejecutor externo para PC-01 (YOLOv8n/YOLOv8l), PC-02 (YOLO26l/Swin-Tiny) y PC-03 (ViT-B/16/RT-DETR-R50-vd).

Conservar juntos los cinco archivos Python. Ejecutar solo botrytis_robustez_reanudar.py desde el .venv original, con --expected-pc PC-01, PC-02 o PC-03. El archivo CMD entregado extrae y verifica los cinco archivos desde un commit fijado; no cambia el checkout del proyecto.

## Alcance

Mantiene las tres mejores configuraciones HPO, semillas 17/42/73 y tres folds por semilla. Solo habilita repeat y aggregate del script nativo 08. No ejecuta baseline/HPO ni entrenamiento final, validacion externa, thresholds, freeze o TEST. No crea rondas ni inicia ResNet50, que sigue pendiente de preparacion/entrenamiento por separado.

Espera a que terminen otros procesos Python y a que se liberen los bloqueos originales. Durante la espera no envia senales, mata procesos ni borra locks. Un segundo controlador de robustez se rechaza mediante bloqueo del sistema operativo. Mientras espera no altera colas. Una vez libre, comprueba identidad, codigo b706f176, config inmutable, datos/splits, preparacion, evidencia baseline y HPO y separacion de particiones. La revision de HPO PENDING_REVIEW se presenta al operador para escribir APROBAR. Rechazar conserva el estado. Esto no afirma inspeccion visual ni ausencia de sobreajuste.

Restaura exclusivamente repeat/aggregate BLOCKED cuyo marcador y recibo original demuestran POSPUESTO_POR_PRIORIDAD; conserva bloqueos ajenos y estados activos. Hace backup SQLite previo. Los trabajos posteriores siguen pendientes o bloqueados; no se marcan completados. La agregacion permanece PENDING_REVIEW cuando asi lo exige la cola.

Reutiliza folds firmados y resultados completos. Puede reutilizar el calculo HPO de semilla42 solo cuando la firma coincide exactamente con arquitectura, ronda, HP, semilla, fold, epocas y paciencia y el historial tiene SHA valido. Esto representa la misma ejecucion de semilla42, no una repeticion independiente adicional. Las semillas17 y73 se entrenan o reanudan desde evidencia valida. No cambia metricas, presupuesto ni regla de seleccion. Se preservan checkpoints y los optimizadores recuperables mediante los hooks existentes. Si hay parcial sin checkpoint firmado compatible, se conserva y se detiene esa arquitectura. RUNNING solo se recupera con propietario local documentado y proceso demostrado muerto; no se deduce por antiguedad.

RT-DETR usa exactamente el adaptador oficial con SHA e62e47fdc4cdbe6648d0830a8efac9701ab04ae82b77757d4a0880ec73381df7; no cae al grafo YOLO/Ultralytics incompatible. Los archivos r50_hpo incluidos son dependencias de lectura/validacion; el controlador no llama su main ni su funcion run_hpo.

## Salidas

Se mantienen las rutas BF originales por ronda, y r50\r para RT-DETR. Robustez: <runs>/<modelo>/robustness/<config>/seed_<semilla>/foldN. Resumen nativo: <runs>/<modelo>/robustness/summary.json. Seleccion nativa: artifacts/best_configs/<modelo>.json dentro de cada ronda. El controlador imprime FIN_ROBUSTEZ y la direccion completa de %USERPROFILE%\bg\<id>\resumen.json; cada subcarpeta de modelo contiene revision.json y resumen.json.

ROBUSTEZ_COMPLETA acredita la matriz completa y la agregacion verificada, no un ganador global ni validacion externa. Luego corresponde revisar la agregacion, entrenar el candidato final con TRAIN y evaluarlo en VAL externo20%. TEST10% permanece cerrado hasta la congelacion y autorizacion finales del sistema hibrido.

## Validacion del entregable

72 tests CPU aprobados, incluidos fixtures del 08 nativo para 27 combinaciones, cache idempotente, agregacion, rechazo de matriz incompleta, restauracion selectiva/transaccional, HPO activo, aprobacion presencial, backend oficial, firmas y CSV opcionales/no finitos/corruptos. 36 subtests adicionales en la suite CSV. No se entrenaron modelos en GPU ni se verificaron en vivo los discos de las PCs Windows. La evidencia real se comprueba localmente al ejecutar.

Comando de pruebas (desde un checkout del framework, con sus dependencias): PYTHONPATH=src python -m pytest tools/robustez/test_robustez.py -q

Correccion del comparador YOLO y epocas reales: ver CORRECCION_CSV_V2.md.

PC-03: la version v3 corrige la comparacion de ap_per_iou serializado; ver CORRECCION_R50_CSV_V3.md y VALIDACION.json. Las pruebas de 72 casos corresponden a la entrega v2; esta correccion tiene 49 pruebas CPU y 2 integraciones adicionales.

PC-03: v4 comprueba los pares checkpoint/recibo y el linaje antes de recuperar; ver CORRECCION_R50_RECOVERY_V4.md. No repetir intentos si falta una evidencia critica.
