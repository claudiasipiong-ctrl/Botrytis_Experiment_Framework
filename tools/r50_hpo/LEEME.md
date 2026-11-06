
## Correccion de comparacion de historiales

El verificador HPO compara las metricas numericas con la tolerancia original y los metadatos textuales por igualdad exacta. Evita aplicar np.allclose a strings sin modificar los historiales originales ni omitir columnas de metricas. Las discrepancias y valores numericos no finitos siguen bloqueando. La recuperacion admite el SHA del ejecutor original fecc8b45 ademas del actual, manteniendo identidad de job/intento, comprobacion de proceso muerto y limite de intentos; no recupera procesos vivos ni ejecutores desconocidos.
