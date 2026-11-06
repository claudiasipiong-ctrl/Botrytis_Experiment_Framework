
## Correccion de comparacion de historiales

El verificador HPO compara las metricas numericas con la tolerancia original y los metadatos textuales por igualdad exacta. Evita aplicar np.allclose a strings sin modificar los historiales originales ni omitir columnas de metricas. Las discrepancias y valores numericos no finitos siguen bloqueando. La recuperacion admite el SHA del ejecutor original fecc8b45 ademas del actual, manteniendo identidad de job/intento, comprobacion de proceso muerto y limite de intentos; no recupera procesos vivos ni ejecutores desconocidos.

## Recuperacion del intento3 observado en PC3

`--reanudar-intento3-pc03` admite una sola transicion para el owner observado (job r50vd_01/hpo:rtdetr_r50, attempt3, SHA6ea86a71..., hostnameI21313, PID21072, create_time1793890437.2571075). Requiere proceso muerto comprobado y RUNNING/attempt3 coincidente. Antes de la transicion conserva owner, snapshot consistente de cola y decision documentada. No rebaja el contador: el siguiente claim es intento4. No cambia presupuesto15 ni epocas60 ni HP/splits. Otros propietarios e intentos, o procesos vivos, siguen bloqueados. La correccion no permite recuperaciones ilimitadas.
