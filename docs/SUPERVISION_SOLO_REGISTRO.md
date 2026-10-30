# Supervisión sin paradas por umbrales

El operador puede elegir explícitamente `--observe-only` para registrar
temperatura, utilización, memoria y espacio sin detener trabajos por los umbrales
de temperatura ni las reservas configuradas de RAM/disco. El modo predeterminado
sigue aplicando límites verificados; las políticas anteriores conservan su comportamiento.

```powershell
& $py tools/configure_safety.py --config artifacts/v4/local.yaml --gpu-index 0 --observe-only --replace-existing
```

El comando obtiene el UUID y modelo reales de la GPU local, conserva un respaldo
de la política y configuración anteriores y conecta `operations.safety_policy_file`
con el archivo creado. Mantiene las rutas y decisiones del dataset. No modifica
BIOS, potencia, frecuencias, voltajes, controles del controlador o firmware.
Cada PC debe crear su propia política; no copiarla entre estaciones con UUID distintos.

La política registra `mode: observe`, la elección explícita del operador y su
fecha. No inventa un límite térmico ni declara verificadas las protecciones OEM:
`cpu_thermal_protection_verified` permanece falso si no se proporcionó verificación.
La telemetría declara `resource_limits_enforced: false`. Su estado PASS significa
que se obtuvo la lectura y coincidió el dispositivo, no una certificación térmica.
Las comprobaciones de identidad, integridad, exclusión de GPU y errores reales
de ejecución siguen vigentes. El modo no promete continuidad ante OOM, disco
agotado, fallos del sistema o intervención del usuario/hardware.

## Primera prueba local de entrenamiento

```powershell
& $py tools/qa_training_smoke.py --config artifacts/v4/local.yaml --output $salida
```

Usar una carpeta de salida nueva y corta. El comando conserva el mismo lock
local de los workers, exige CUDA y realiza una época de ResNet50 y una de YOLOv8n
con imágenes sintéticas de 64 píxeles, batch 2 y workers 0. No descarga pesos,
no lee las fotos del experimento y no abre su test. Esta ejecución directa no
añade un límite de tiempo ni paradas por recursos en modo observe; termina
cuando concluye la prueba o aparece un error real/interrupción.

Registra telemetría antes y después de cada modelo y escribe `training_check.json`
con COMPLETE o FAILED. La prueba verifica forward/backward y publicación del
checkpoint del detector; no certifica todas las arquitecturas, AMP de todos
los modelos, recuperación de cortes ni rendimiento sostenido. El modo de
supervisión no modifica la inicialización ni los hiperparámetros científicos.
No crea aprobaciones personales ni lanza automáticamente el experimento oficial.
