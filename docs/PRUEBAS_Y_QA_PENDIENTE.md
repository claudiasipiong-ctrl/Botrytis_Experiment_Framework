# Pruebas ejecutadas y QA personal pendiente

## Alcance de esta entrega

El agente valida código en Linux/CPU con fixtures sintéticos. No se accedió a las tres estaciones, no se descargaron pesos ni datasets, no se entrenó, ejecutó HPO real, hizo benchmark GPU ni inferencia sobre datos reales, ni se abrió test científico. No se crearon aprobaciones personales operacionales; los registros de autorización en tests existen solo en carpetas temporales con datos explícitamente sintéticos.

La ejecución completa observada terminó con 311 PASS, 0 FAIL y 9 SKIP (320 casos). No interpretar este documento como READY_FOR_EXPERIMENT. A09, inventario real, pesos, entorno Windows/CUDA y QA personal continúan pendientes aunque las pruebas CPU aprueben.

## Entorno y comprobaciones realizadas

- Python 3.12.14, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, Ultralytics 8.4.170, timm 1.0.30, Optuna 5.0.0; dependencias base/dev fijadas. Instalación repetida, compatibilidad de paquetes, imports ML, autograd CPU y NMS de torchvision verificados.
- Resolución de dependencias para Windows x86_64 / CPython 3.12 con `requirements-ml.txt`, base y dev. Metadatos de la pareja 2.14.1+cu126 / 0.29.1+cu126 resueltos sin descargar o instalar esos wheels en esta sesión. No prueba ejecución de CUDA, driver ni Windows.
- AST de los 131 archivos Python de src/tools/tests/dev_tests aprobado; análisis de imports/variables de los módulos nuevos mediante pyflakes y `git diff --check`.
- PowerShell 7.5.4 de fuente oficial, SHA-256 verificado `1fd7983fe56ca9e6233f126925edb24bf6b6b33e356b69996d925c4db94e2fef`: parser sobre scripts existentes y 37 bloques documentados. Cero ejecución de sus comandos científicos. PowerShell 5.1/Windows 10 nativos requieren QA personal.
- Los tests no llaman `Study.optimize`. Los fixtures Optuna crean/cargan almacenamiento, insertan trials sintéticos o ejercitan contratos sin optimización/entrenamiento. Las cinco pruebas de entrenamiento permanecen deshabilitadas con `BOTRYTIS_ALLOW_TRAINING_TESTS=0`.
- Pruebas nuevas de rondas, aprobación, bloques, identidad, mirrors, cohortes, transferencia, concurrencia CPU, checkpoints y fold cache. Las funciones de cómputo de simulaciones están sustituidas por fixtures, no modelos entrenándose.
- Componentes ResNet/ViT/Swin construidos sin pesos preentrenados: roundtrip estricto de checkpoints propios, logits, normalización, transforms y Grad-CAM sintéticos. No certifica carga de los pesos oficiales reales.

## Ejecución final reproducible

Desde el checkout y el entorno CPU preparado:

```bash
source /workspace/botrytis-onboarding/activate.sh
BOTRYTIS_ALLOW_TRAINING_TESTS=0 python -m pytest tests dev_tests \
  --basetemp=/workspace/botrytis-audit/full-final \
  -o cache_dir=/workspace/botrytis-audit/cache \
  --junitxml=/workspace/botrytis-audit/full-final.xml --durations=15
```

`--basetemp` debe ser exclusivo y prescindible: pytest elimina su contenido al iniciar. Los logs, fixtures y checkpoints sintéticos pesados quedan fuera del repositorio. El resumen pequeño y versionable del resultado final se guarda en `docs/final/validation.json`.

Resultado de la suite completa: **311 aprobadas, 0 fallidas, 9 omitidas, 2 advertencias**, 320 casos, **1123,65 s (18 min 43 s)**, salida 0.

Las dos advertencias de Matplotlib provienen de pruebas que construyen más de 20 figuras antes de devolverlas al consumidor; se mantienen explícitas, no se suprimieron. El guard de cierre tras exportación (incluido fallo de disco) pasó su regresión.

La revisión posterior añadió dos regresiones: escritura del log de un job con ID de ronda y comparación de objetivos AP/mAP separados. Resultado de la ejecución enfocada posterior sobre control, rondas, intercambio, orquestación, almacenamiento, componentes ML y freeze: **154 aprobadas, 0 fallidas y 2 omitidas por Windows**, 156 casos, **69,14 s**, salida 0. Incluye las dos regresiones nuevas. No se suma este recuento a la suite completa porque se solapan casos. El generador SVG se validó además mediante AST, estructura/enlaces HTML y render estático.

Documentos HTML: estructura y enlaces locales comprobados, JavaScript con sintaxis válida y SVG renderizado mediante CairoSVG para inspección. No se ejecutó una prueba completa de navegador: Chromium no está instalado en el contenedor. PowerShell se verificó por parser, no mediante ejecución de los comandos operacionales.

Como referencia de diagnóstico anterior a las correcciones, el onboarding produjo 223 PASS, 25 FAIL y 9 SKIP. Esa cifra corresponde al código original y **no** acredita la versión modificada. Los fallos originales se contrastaron y se repararon o actualizaron fixtures incompatibles, conservando las aserciones de integridad.

## Omisiones que deben mantenerse explícitas

- Entrenamiento opt-in: cinco pruebas, no habilitadas por el agente.
- Dos casos exclusivos de Windows en el generador de tareas.
- Launcher gráfico externo situado junto al proyecto: no forma parte del repositorio recibido.
- Prueba con imágenes/checkpoints reales: recursos no suministrados.

Una prueba omitida no equivale a PASS. Los recuentos reales y motivos concretos se extraen del JUnit final.

## Checklist personal en las tres estaciones

| Área | Acción del operador | Evidencia que debe conservar |
| --- | --- | --- |
| Código | Clonar, fijar SHA completo aprobado; ninguna actualización durante trabajos activos; comparar hashes/commit entre PCs | HEAD, git status, code_hash y registro de versión |
| Identidad | Confirmar PC-01=i21315, PC-02=i21314, PC-03=i21313, hostname y UUID físico | Binding local, inventario hardware; no copiar binding de otra PC |
| Dependencias | Instalar mismo build, pip check, imports, CUDA/AMP y compatibilidad del driver | environment.json por PC y referencia revisada |
| Seguridad | Verificar sensores, protección térmica y límites oficiales del fabricante | Política de seguridad con fuente, telemetría y observabilidad |
| Datos | Conciliar 324/328/329; revisar clases, huérfanos, duplicados, anotaciones ambiguas y pairing | Inventario activo, decisiones de exclusión/revisión y SHA de cada fuente |
| Split/folds | Conservar split válido 70/20/10 seed67; folds dentro del train; igualdad entre PCs | Manifiestos sellados; revisión dHash si corresponde; no re-sortear |
| Derivados | Frutos solo clase0, crops GT, herencia de split, sellos de contenido | Manifiestos, conteos, archivos y hashes locales |
| Pesos | Adquisición personal/licencia, SHA/tamaño/procedencia; cero fallback a caché de otro modelo | Inventario verificado por arquitectura; no afirmar R50 listo |
| Baselines | Revisar pérdidas/métricas/folds; una GPU local; publicaciones y PENDING_REVIEW | Logs/curvas/recibos/tiempos y aprobación explícita |
| HPO | 15/20 trials totales, tres folds, TPE/MedianPruner; bloques y revisiones; no mezclar DBs | SQLite local, tablas de trials y fingerprints |
| Robustez | Top3 × 3 seeds × 3 folds; revisar dispersión | 27 evaluaciones de fold por arquitectura, agregación y decisión |
| Memoria/IO | Medir workers 0/2/4, persistencia, RSS del árbol y VRAM; no adoptar cambios numéricos sin QA | Mediciones por PC/modelo/configuración, no estimaciones inventadas |
| Batch/OOM | Verificar microbatch/acumulación/LR reales y efecto de BatchNorm; fallback desactivado hasta evidencia | Contrato efectivo y pruebas GPU/AMP |
| Recovery | Corte de proceso/energía controlado, checkpoint propio con recibo, RNG/optimizer/scaler/loader | Evidencia de antes/después; no prometer equivalencia bit a bit sin medir |
| Sync | ZIP/checksum, import idempotente, journal recuperable, conflicto preservado | Bundle, receipt, ledger y pruebas físicas de traslado |
| Transferencia | Detener RUNNING; source fence, recibo, confirmación, activación; conservar revisiones | Ownership local y cadena completa; no restaurar copias antiguas |
| Validation externo | Elegir ronda antes de abrir 20 %, autorizar una evaluación/calibración; no explorar rondas mirando ese conjunto | Ledger de autorización y consumo por arquitectura |
| RT-DETR-R50 | Decidir backend/equivalencia/checkpoint sin sustituir arquitectura; validar optimizador y grafo | Decisión metodológica explícita y QA real antes de cerrar comparación |
| Cohorte/pipeline | Siete candidatos íntegros, selección predeclarada, crops predichos | Cohort manifest, fingerprints de origen, equivalencia y evidencias completas |
| Freeze/test | Revisar freeze, SHA y autorización personal posterior; test una vez | FREEZE y TEST_RELEASE personales, manifiesto íntegro, test lock |
| Benchmark/figuras | PC-01 para latencia, datos medidos y figuras con procedencia/recursos liberados | Datos reales, comandos/fecha/operador, tablas y fuentes |

## Límites y decisiones externas

- RT-DETR-R50 sigue bloqueado de forma explícita; las otras arquitecturas independientes pueden avanzar. La selección oficial de cuatro detectores y el freeze completo no se certifican sin resolverlo.
- Un proceso vivo conserva locks del SO. No hay consenso global entre SQLite independientes: el protocolo exige operadores que respeten transferencias y no restauren discos/ownership antiguos.
- Checkpoints de época y recuperación GPU/AMP/DataLoader Windows requieren pruebas reales; preservar artefactos no demuestra equivalencia numérica tras un corte.
- La creación del PR requiere acceso a `api.github.com`; el proxy inicial lo denegó y se guardó ese dominio en el borrador de red. Publicar una rama Git y crear/verificar un PR son operaciones distintas; su estado real se informa al finalizar.
