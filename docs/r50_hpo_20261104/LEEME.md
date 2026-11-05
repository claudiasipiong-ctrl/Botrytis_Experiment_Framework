# PC-03: aprobación y HPO RT-DETR-R50-vd

El usuario autorizó expresamente iniciar y aprobar el HPO original. Ejecutar el comando publicado con `--aprobar-original-hpo` registra esa autorización únicamente después de superar los controles técnicos. No pide otra aprobación en consola y no afirma una inspección visual humana ni ausencia de sobreajuste.

## Alcance

- PC-03 / I21313, ronda `botrytis_v4/r50vd_01`, fingerprint `8aa24ae8da9a178a3d864f3e447667f52be6bbda55691abc585dff3d8c654352`.
- Mantiene 15 trials COMPLETE+PRUNED, tres folds, mAP50–95, máximo60 épocas, paciencia20, seed42, FP16 y el espacio de búsqueda TPE/MedianPruner original.
- Verifica identidad, propietario y configuración; reutiliza la auditoría local B compatible o ejecuta la auditoría CPU antes de aprobar.
- Recalcula los controles críticos actuales de dataset/splits/derivados, listas originales y conflictos entre particiones antes de registrar la aprobación. Reutiliza READY_FOR_EXPERIMENT y la autoprueba GPU compatibles; no repite toda la QA.
- Registra la aprobación real en `reviews` mediante `StudyQueue.approve`; eso cambia solo el baseline PENDING_REVIEW a COMPLETE y habilita su HPO dependiente.
- Instala el adaptador oficial exacto SHA256 `e62e47fdc4cdbe6648d0830a8efac9701ab04ae82b77757d4a0880ec73381df7` y llama a `07_run_hpo_study` con sus argumentos reales. No utiliza el backend R50 nativo de Ultralytics como sustituto.
- No llama a baseline06, robustez08, selección global, validación externa ni TEST. Los trece trabajos posteriores se planifican nativamente, pero permanecen pendientes.
- Actúa únicamente en PC-03, con su entorno virtual. Los archivos externos descargados no cambian HEAD, `src`, configuración científica ni código aprobado.

## Evidencia y límites

El baseline aportado tiene media0,47245137 y SD muestral0,32536641; fold0 es deficiente. Los números concuerdan con su historial, pero no permiten certificar ausencia de sobreajuste, independencia biológica ni rendimiento en imágenes nuevas.

Dos folds mejoraron después de la época60. El HPO conserva el presupuesto acordado de60; sus resultados no se comparan causalmente con el baseline100 como si fueran presupuestos iguales. El resumen no declara un ganador ni validación definitiva.

## Conservación y recuperación

El baseline queda fuera de las rutas de entrenamiento y su inventario de hashes se compara al terminar. Los trials COMPLETE y PRUNED se verifican y no se vuelven a entrenar. Si falta SQLite pero existen resultados HPO previos, se detiene; no crea un estudio nuevo.

El estado de TPE/pruner se guarda localmente después de cada propuesta y trial. Si falta un estado previo se exige replay determinista exacto de las propuestas; no se reinicia silenciosamente el RNG. Solo se recuperan procesos propios demostrablemente muertos y se conserva el historial de FAIL. Los reintentos usan los mismos HP, enlazan el trial original y copian sus folds/estados a una carpeta nueva; los originales se conservan. La política nativa del adaptador retiene `best.pt` y dos generaciones rolling de optimizador por fold nuevo. No promete guardar todas las épocas históricas. Se comprueba el espacio requerido antes del trabajo GPU.

Una segunda ejecución detecta HPO activo o terminado y evita duplicarlo. Si terminó el cómputo y falló la publicación/exportación, se recupera esa fase sin volver a entrenar trials. No se borran locks para forzar acceso.

Las diferencias Git de LF/CRLF solo actualizan el índice cuando se ha probado igualdad exacta con HEAD mediante los filtros de Git; no se resetea ni reemplaza ningún archivo modificado.

## Progreso y archivos

CMD muestra identificación, reutilización/auditoría, aprobación registrada, trial/configuración/HP, fold, época, mAP50–95/mAP50/P/R/loss y errores. El adaptador guarda optimizador/EMA/scaler/RNG/gradientes y recupera desde la última época completa; una época interrumpida antes de guardarse puede recalcularse.

- Baseline conservado: `%USERPROFILE%\r50\r\rtdetr_r50\baseline`.
- HPO: `%USERPROFILE%\r50\r\rtdetr_r50\hpo`.
- Aprobación y controles: `%USERPROFILE%\r50\hpo_control\a\<id>\`.
- Estado de reanudación: `%USERPROFILE%\r50\hpo_control\recovery`.
- Resumen final: `%USERPROFILE%\r50\hpo_control\latest.json` apunta al `summary.json` de ese intento.
- Errores: `error.json` y logs nativos; una reejecución del mismo comando recupera únicamente trabajo compatible.

Éxito computacional: `HPO_RTDETR_TERMINADO`, COMPLETE+PRUNED=15, al menos un COMPLETE y cola HPO `PENDING_REVIEW`. El resumen mantiene `final_validation: false`, `overfitting: NO_DESCARTADO`, `winner: NO_DECLARADO`. Revisar los resultados es una etapa posterior.

## Pruebas y alcance de verificación

Pruebas del controlador con la cola real, Optuna y el código nativo07, empleando un backend explícitamente simulado: 15 trials/45 folds, aprobación condicional, idempotencia, recuperación de folds y estados, conservación de originales, restore del sampler, valores no finitos, ausencia de SQLite con pesos previos, fallos de integridad y bloqueo de procesos vivos. No se entrenan modelos GPU en estas pruebas. La auditoría fijada conserva además sus 35 pruebas previas, incluida carga estricta CPU del modelo COCO real. La ejecución completa en Windows debe comprobarse en PC-03; no se afirma haber accedido a ella desde la nube.
