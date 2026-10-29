# Validacion y criterios de aceptacion — 2026-10-28

## Lo verificado realmente

Entorno de desarrollo separado de i21315/i21314/i21313: Python del runtime local y venv de trabajo sin torch/torchvision/timm/Ultralytics instalados. No se ejecutaron cargas, CUDA ni diagnosticos sobre las tres PCs. Se instalaron dependencias de analisis/tests solamente en el venv de trabajo; no se modifica su entorno de entrenamiento.

Se ejecutaron pruebas de algoritmos puros, configuracion, Optuna sin modelos, SQLite, filesystem, locks, snapshots, importacion y reloj simulado. No se ejecutaron fixtures que consultan hardware, generan un experimento completo ni entrenan modelos. En el modulo HPO se excluyo la prueba zombie que importa el stack de entrenamiento; su verificacion queda para QA local. La suite completa ML permanece pendiente.

La evidencia final esta en development_tests.xml y development_validation.json. Se verifican AST Python, pyflakes, sintaxis Bash mediante bash -n y parseo estatico PowerShell. No se ejecuto el Bash integral. Las pruebas con reloj simulado no registraron una tarea Windows ni certifican el horario real.

Resultado final aislado: **99 pruebas ejecutadas, 0 fallos, 0 errores y 0 omitidas dentro de esa seleccion**; una prueba ML fue deseleccionada expresamente. AST de 98 archivos Python, pyflakes, Bash -n y parser PowerShell aprobados. XML del scheduler validado contra XSD oficial con registro simulado; ninguna tarea real creada.

## Matriz de aceptacion

| # | Criterio | Estado honesto y evidencia faltante |
|---|---|---|
| 1 | Exclusiones coherentes | Implementado; registro/config/cola/seleccion/IDE/tests actualizados y tests negativos. Referencias historicas preservadas |
| 2 | YOLO26n pendiente | Implementado: activo; conclusion en auditoria; falta autorizacion si desea retirarlo |
| 3 | Metodologia y trazabilidad | Guardias preservadas/reforzadas. Falta ejecutar cadena ML y auditar datos reales |
| 4 | Optimizaciones justificadas | Documentadas; ninguna velocidad/seguridad sostenida se atribuye a hardware desconocido |
| 5 | Paralelo en tres PCs | Propietarios exclusivos implementados y probados con tres DB aisladas. Ejecucion real y equilibrio de carga pendientes |
| 6 | Duplicados/dependencias/overwrite | Tests aislados de propiedad, locks, archivos faltantes, conflictos, integridad y recuperacion. Falta evidencia operativa de tres instalaciones |
| 7 | Ruta dataset por equipo | Implementado segun identidad vinculada. Falta mapa humano, existencia y contenido real de tres rutas |
| 8 | Exportacion 21:00 antes de22 cuando viable | Scheduler/proceso/tarea/progreso implementados; tests de fecha e inicio tardio. Falta registro real Windows, disco/tamanos y logs antes22 |
| 9 | ZIP minimo suficiente | Snapshots confirmados y dependencias reales; tests de contenido. No exporta epoca incompleta; no promete resume exacto |
| 10 | Exportar/Importar por equipo | Rutas exactas y creacion implementadas. Permisos/espacio reales pendientes |
| 11 | Continuar tras exportar | Watcher independiente, sin detener worker. Falta observar continuidad de entrenamiento real |
| 12 | Dependencias externas explicitas | Rutas/jobs exactos en blocked_jobs.json; seleccion exige candidatos completos. Falta flujo fisico entre PCs |
| 13 | Importacion valida | SHA, compatibilidad, propietario, conflictos, idempotencia y journal probados con fixtures; procedencia declarada, sin firma |
| 14 | Recuperacion | COMPLETE/PRUNED, folds y expansion preservados; importacion tras corte probada. Epoch/optimizer/TPE RNG exactos no soportados automaticamente |
| 15 | Bash completo | Entregado con backend y wrapper; sintaxis comprobada. Ejecucion integral personal pendiente |
| 16 | logs_qa raiz | Implementado; todas las salidas/caches/temp QA dirigidas alli. No se crearon falsos logs de las PCs |
| 17 | Diagnostico detallado | PASS/FAIL/WARN/SKIP/BLOCKED, comandos/stdout/stderr/duracion/codigos/JSON/Markdown/XML. QA real pendiente |
| 18 | Sin pruebas remotas | Respetado: ninguna herramienta de escritorio remoto ni entrenamiento en sus equipos |
| 19 | Cambios documentados | CHANGE_MANIFEST.json con hashes; inventario original y documentos historicos conservados |
| 20 | Sin datos/resultados inventados | Sin metricas cientificas ni supuestos de hardware; solo fixtures explicitamente sinteticas |

Los estados implementados no se presentan como validacion de las tres maquinas. No se acredita aun funcionamiento integral ni utilizacion maxima sostenible.

## Pendientes que requieren evidencia o decision

1. Dataset real: 324/328 y label listado sin imagen; classes.txt, conteos, CRC/SHA, integridad y particiones reales.
2. Mapa explicito de identidades, seed e IDs de clase. No se eligen por usted.
3. Hardware, versiones y entorno de referencia de las tres. Si difieren campos criticos, decidir compatibilidad cientifica antes de relajar guardias.
4. Limites OEM/temperatura y reservas de RAM/disco verificados; sensor CPU ausente debe ser reconocido expresamente tras verificar protecciones.
5. Bash/Python/driver/CUDA y QA integral local, incluyendo modelos y todos los tests existentes.
6. Registro efectivo de tarea, continuidad tras exportar, copy/export/import reales y tiempo total antes22. Mantener eventos y ZIP/sidecar.
7. Seleccion, freeze, test y benchmark reales solo en su momento; decisiones heredadas del anexo siguen sin confirmar.
8. YOLO26n: se recomienda conservarlo hasta medir; ninguna exclusion adicional aplicada.
9. Cambios de microbatch, precision, workers con impacto RNG, determinismo, presupuesto o redistribucion requieren evaluacion; no se aplicaron como supuestos optimos.

## Evidencia que debe devolver

Comprima logs_qa de cada equipo, conservando subdirectorios y nombres. Incluya fleet_mapping.json, environment local/referencia y los logs de sesion (events/export_status/task_registration/watcher/worker_exit), last_export_progress.json y blocked_jobs.json si existen. Para diagnosticar sincronizacion bastan inicialmente manifiestos/checksums/logs y tamanos; no envie otra vez el dataset ni checkpoints enormes salvo necesidad concreta. No incluya tokens, contrasenas ni variables de entorno completas.

El informe puede revisarse con esos archivos sin acceso remoto. Las velocidades, seguridad sostenida y balance solo podran evaluarse con mediciones reales, sin inferir fechas de finalizacion.
