# PC-03: robustez ViT-B/16; correccion minima de reanudacion RNG

El registro del 2026-11-05 10:51:03 de PC-03 demuestra que el fold 0 de cfg_16fcf576, seed 73, esta completo y se reutilizo. El fold 1 fallo antes de continuar sus epocas: `TypeError: RNG state must be a torch.ByteTensor` al ejecutar `torch.cuda.set_rng_state_all(saved['cuda_rng'])`.

La fuente nativa b706 carga el checkpoint con `map_location=device`. Esto tambien transporta los tensores del estado RNG a CUDA. El generador CUDA exige que esos mismos bytes se entreguen en CPU. El codigo ya hacia `.cpu()` para RNG de torch CPU y del loader, pero no para los estados CUDA.

El ayudante externo aplica un hook solo durante la reanudacion de un clasificador con checkpoint propio verificado. Valida que haya un vector uint8 por GPU y devuelve esos mismos bytes en CPU contigua. No convierte dtypes, no genera semillas, no elimina RNG, no modifica checkpoints ni las fuentes. Delega al setter nativo y restaura siempre la funcion original, incluso ante excepciones. Registra CLASSIFIER_RNG_RESTORE_CPU con hashes de los estados, dispositivos de origen y reseeded=false. Los estados invalidos siguen deteniendo la ejecucion.

El dispatcher admite `--only-model vit_b16 --expected-pc PC-03`. El alcance y rtdetr_r50 diferido se registran en el resumen. Esta opcion procesa solo repeat/aggregate de ViT; nunca inicia baseline, HPO ni TEST. No cambia la cola de RT-DETR ni la de otras PCs. Conserva las revisiones y los limites de intentos. El trabajo mostrado tiene attempt=2; la siguiente reclamacion normal sera attempt=3, sin resetear ni ampliar limites.

La reentrada conserva los recibos de folds completos. Los checkpoints parciales siguen pasando las verificaciones originales de integridad, identidad de fold, HP, semilla, codigo, dataset y split. Un fallo critico se informa; no se marca trabajo como completado ni se reinicia un fold firmado.

Las pruebas usan CPU y dobles de transporte CUDA. Se reproduce el error original y se ejecuta el bloque real de reanudacion del clasificador b706 para demostrar que continua desde epoch+1, conservando historial/modelo/optimizador/scaler. Tambien se comprueba identidad de secuencias aleatorias, rechazo de RNG invalido, restauracion del hook ante errores, archivos inalterados, alcance ViT exclusivo y rechazo de modelo de otra PC. No se ejecutaron entrenamientos GPU ni Windows reales en la nube. La comprobacion del disco y del checkpoint real ocurre localmente.
