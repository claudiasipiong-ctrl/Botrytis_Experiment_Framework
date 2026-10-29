# Registro de ejecuciones y recuperacion

Bootstrap stdlib en cada entrada Python antes de imports opcionales/argparse/config: runs/v4/_bootstrap/UTC_PC_fase_tags_UUID.
execution.log,eventosJSONL,entorno/packageversions,config inicial redactada,metadata y estadofinal.
Invoke-Botrytis.ps1 agrega bootstrap PowerShell incluso si Python no inicia, falta archivo o hay error de sintaxis.
Prepare-Environment.ps1 y Invoke-QA.ps1 conservan tambien logs de arranque; todas las invocaciones personales
deben usar los lanzadores recomendados. Un fallo del propio disco puede impedir toda escritura: informar error,
no tratar ausencia de log como ejecucion correcta. Un corte electrico deja RUNNING que requiere revision.

RunContext agrega configuracion efectiva redactada,hashcodigo/dataset/split/modelo,entornoGPU/CUDA,metricas,
checkpoints,figuras,tiempos y motivo de salida. stdout/stderr y traceback completos se conservan con redaccion.
Cada intento tieneUUID yPC explicita; hostname es evidencia, nunca mapeo automatico de PC-01/02/03.
Las rutas estables por etapa conservan previous_attempts antes de reemplazar metadata; execution/events son append.
Se utilizan locksOS locales, escriturasJSON atomicas y flush; el identificador sobrevive el intercambio manual.
Subprocesos capturados registran salida/codigos; subprocesos heredados usan pumps de stdout/stderr.
Los que escriben directamente en archivo conservan ese log y su bootstrap propio; el padre registra comando/codigo.
Los .jsonl tambien se incluyen en snapshots. No se reescriben evidencias historicas.

SIGINT/KeyboardInterrupt => INTERRUPTED; errores/OOM => FAILED con traceback, reintentosOOM declarados y guardados.
Checkpoints de clasificador por epoca son atomicos y llevan recibo, estados de optimizer/scaler/RNG; detector usa
checkpoints nativos y recibos .resume.json. Solo archivos propios posteriores y verificados se pueden reanudar.
No hay garantia de reproducibilidad bit a bit entre librerias/GPUs diferentes; comprobarla personalmente.
Secretos se redactan por claves y argumentos sensibles, asignaciones password/token y URLs con credenciales.
No guardar secretos en nombres de archivo ni textos de metrica. No se buscaron credenciales ni se incluyeron caches.

DEV_CHECK_PASS/FAIL son resultados de desarrollo. QA_MANUAL_PENDIENTE son gates personales, inicialmente cerrados.
EXPERIMENTO_PENDIENTE se mantiene hasta ejecucion real con datos,pesos y aprobaciones. Ninguna suite aprueba QA por si sola.
