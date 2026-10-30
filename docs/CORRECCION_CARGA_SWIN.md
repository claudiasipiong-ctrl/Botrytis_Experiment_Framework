# Compatibilidad de los pesos locales Swin-Tiny con timm

La prueba manual en PC-02 confirmó la carga de YOLO26l, pero la carga estricta
de Swin-Tiny rechazó claves `relative_position_index` y `attn_mask`. El archivo
local había aprobado tamaño, SHA256 y procedencia; no era una descarga incompleta.
Estas claves representan buffers de atención derivados que la versión instalada
de timm ya no persiste en el estado del modelo.

La construcción de Swin usa ahora `timm.models.swin_transformer.checkpoint_filter_fn`
antes de `load_state_dict(..., strict=True)`, como el cargador de checkpoints de
la propia biblioteca. Se sigue construyendo la implementación exacta del catálogo
con su cabeza ImageNet de 1000 clases, cargando el checkpoint local verificado
y reemplazando después la cabeza por la binaria. Se conserva la normalización
del modelo. No se usa `strict=False`, descarga implícita, sustitución de arquitectura
ni inicialización desde cero como alternativa a una carga fallida.

Las regresiones usan estados sintéticos de un Swin pequeño sin entrenamiento
ni pesos oficiales. Cubren buffers persistentes antiguos, estado moderno,
igualdad de todos los parámetros del backbone, guardado/carga binaria posterior
y rechazo de parámetros faltantes, inesperados o con dimensiones incompatibles
en atención. La carga del checkpoint oficial real debe repetirla el operador
de PC-02; los tests sintéticos no certifican ese resultado.

`tools/check_model_loading.py --config artifacts/v4/local.yaml --pc-id PC-02`
permite repetir la comprobación de los pesos disponibles asignados a cada PC
sin pegar código Python. Solo construye los modelos en CPU y escribe el informe;
no realiza forward, entrenamiento, evaluación ni aprobación de gates.
Las arquitecturas bloqueadas por el catálogo se informan por separado y no se
sustituyen; una salida 0 se refiere a los pesos disponibles comprobados.

Aplicar la misma revisión del código en las tres estaciones y renovar los informes
de entorno, conservando respaldo de los anteriores. Los datasets, splits, folds
y archivos de pesos existentes se mantienen. La política térmica continúa
pendiente de la confirmación del área de informática.
