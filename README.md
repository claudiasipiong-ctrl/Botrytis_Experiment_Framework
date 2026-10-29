# Botrytis Experiment Framework · semantica v4

Implementacion preparada para QA presencial. QA_MANUAL_PENDIENTE / EXPERIMENTO_PENDIENTE.
Cero pesos iniciales; no se ejecutaron entrenamientos ni test final durante esta correccion.

- [Guia de ejecucion](docs/GUIA_EJECUCION_3PCs_ACTUALIZADA.md) y [flujo](docs/FLUJO_COMPLETO_BOTRYTIS_HORIZONTAL.md).
- [Metodologia](docs/METODOLOGIA.md), [pesos y bloqueo RT-DETR](docs/PESOS.md), [figuras](docs/FIGURAS.md).
- HTML de entrega en `../entregables_botrytis/` (conservar como carpeta hermana).
- 0 arandano_a = unico fruto fisico/target fruit; 1 arandano_sospechoso = matching positivo; 2 signo_sospechoso_botrytis = auxiliar.
- Matching bipartito global max IoU con desempate ordinal exacto; IoU tecnico inicial 0.50 configurable. Conflictos bloquean etiquetas y negativos.
- Split por imagen 70/20/10 seed67; 3folds dentro de train; robustez seeds17/42/73. Test exige freeze integro y autorizacion personal posterior.
- Siete detectores y tres clasificadores. HPO15/20 trials, AP positiva clasificador, mAP50:95 detector.
- Namespace v4 evita reutilizar derivados y evidencias obsoletas. No se regeneraron splits reales.
- `dev_tests` son fixtures sinteticos y verificaciones de desarrollo. `tools/qa.py`, Invoke-QA y qa.sh son para el usuario; no fueron ejecutados aqui.
- Todos los comandos Python tienen bootstrap previo a imports opcionales; el launcher PS cubre tambien errores de Python/sintaxis/archivo ausente.
- El listado local de nombres encontro 324 imagenes y 324 labels, referencia329; no sustituye la auditoria personal.

Las evidencias previas conservadas en reports/, docs/historical_pre_v4/ y docs/HISTORICAL_ARTIFACTS.json son OBSOLETAS/NO VALIDADAS. Sus antiguos PASS no acreditan el nuevo protocolo.
