# Diferencias pequeñas de RAM utilizable entre estaciones Windows

El preflight comparaba `ram_total_mb` por igualdad exacta. Las estaciones
PC-02 y PC-03 informaron 261604 y 261587 MiB, respectivamente. Estos valores
son la RAM total utilizable que informa el sistema operativo, no un inventario
de los módulos físicos. Una diferencia de 17 MiB bloqueaba la comparación
de las tres estaciones pese a coincidir los campos de software y las huellas.

`compare_environments` ahora permite diferencias de RAM que cumplan **ambos**
límites: como máximo 64 MiB y como máximo el 0.1 % del menor valor. Son límites
de comparabilidad operacional; no son límites térmicos ni reservas de RAM para
entrenamiento. Una capacidad diferente, un dato ausente en una estación o un
valor inválido siguen produciendo una discrepancia. Los otros campos mantienen
su comparación exacta, incluidos VRAM, versiones de software y hashes.

Los informes conservan los valores originales: no se igualan, redondean ni
reescriben sus mediciones. La política de supervisión debe seguir comprobando
la RAM disponible real durante las pruebas y el entrenamiento.

La actualización cambia la huella del código. Aplicar la misma revisión en
las tres estaciones antes de entrenar y generar de nuevo los informes locales
con `00_check_environment.py`. Conservar los informes anteriores en una copia
de respaldo. Para actualizar los informes sin compararlos con las instantáneas
de código anteriores, usar `--no-compare` una sola vez; PC-01 también escribe
la nueva referencia con `--write-reference`. Después trasladar los tres
`environment_pc*.json` y la referencia de PC-01, y repetir la comparación
normal con todas las estaciones actualizadas. No modificar los informes a mano.

Los datos, las etiquetas, los pesos, las particiones y los folds existentes
se conservan. Esta corrección no aprueba los gates personales ni inicia
experimentos. RT-DETR-R50 continúa bloqueado.
