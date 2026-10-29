# Comandos PowerShell para las tres estaciones

Estado: código para QA personal. Ningún comando de entrenamiento, adquisición de pesos, aprobación o test fue ejecutado por el agente. Los ejemplos se ejecutan **personalmente**, después de revisar sus precondiciones. El proyecto trabaja con siete arquitecturas; RT-DETR-R50 permanece bloqueado. PowerShell nativo de Windows 10; rutas locales fuera de OneDrive/Drive/NAS. No usar SQLite compartida.

## 1. Clonar y fijar un commit aprobado, en cada PC

El SHA debe ser el commit de código revisado del PR, no el nombre mutable de una rama. No actualizar código con procesos activos.

```powershell
$ErrorActionPreference = 'Stop'
$Base = Read-Host 'Carpeta LOCAL de trabajo, fuera de carpetas sincronizadas'
$ApprovedCommit = Read-Host 'SHA completo de 40 caracteres del commit aprobado'
if ($ApprovedCommit -notmatch '^[0-9a-fA-F]{40}$') { throw 'SHA completo obligatorio' }
$RepoUrl = 'https://github.com/claudiasipiong-ctrl/Botrytis_Experiment_Framework.git'
$Branch = 'fix/botrytis-supervised-seven-models'
$root = Join-Path $Base 'Botrytis_Experiment_Framework'
if (Test-Path -LiteralPath $root) { throw 'Carpeta existente: preservar cambios y verificarla; no sobrescribir' }
git clone --branch $Branch -- $RepoUrl $root
if ($LASTEXITCODE -ne 0) { throw 'git clone fallo' }
Set-Location -LiteralPath $root
git checkout --detach $ApprovedCommit
if ($LASTEXITCODE -ne 0) { throw 'Commit no disponible; no sustituirlo por otro' }
if ((git rev-parse HEAD).Trim() -ne $ApprovedCommit.ToLower()) { throw 'Commit distinto' }
if (git status --porcelain) { throw 'Checkout con cambios' }
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 no disponible' }
$py = Join-Path $root '.venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$Utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $Utf8
$OutputEncoding = $Utf8
```

Instalación candidata reproducible: la pareja 2.14.1/0.29.1 fue ejecutada en CPU; se verificó resolución de paquetes para Windows y metadatos de sus builds cu126. Esto no acredita driver, CUDA ni ejecución Windows. Antes de instalar cu126, comprobar personalmente compatibilidad del driver con la matriz oficial de PyTorch/NVIDIA. Usar exactamente el mismo build en las tres PCs; si es necesario otro build, acordarlo y verificarlo en todas, sin modificarlo silenciosamente.

```powershell
& $py -m pip install 'torch==2.14.1' 'torchvision==0.29.1' --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'Instalacion PyTorch detenida; no cambiar versiones automaticamente' }
& $py -m pip install -r requirements.txt -r requirements-dev.txt -c requirements-ml.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependencias incompatibles' }
& $py -m pip check
if ($LASTEXITCODE -ne 0) { throw 'pip check fallo' }
```

## 2. Identidad explícita en cada PC

Ejecutar **solo** el bloque de la estación física correspondiente. El hostname se conserva y verifica; nunca determina la identidad por sí solo.

### PC-01

```powershell
$pc = 'PC-01'
$machine = 'i21315'
$models = @('YOLOv8n', 'YOLOv8l', 'ResNet50')
$first = 'YOLOv8n'
$firstRound = 'yolov8n-r01'
```

### PC-02

```powershell
$pc = 'PC-02'
$machine = 'i21314'
$models = @('YOLO26l', 'Swin-Tiny')
$first = 'YOLO26l'
$firstRound = 'yolo26l-r01'
```

### PC-03

```powershell
$pc = 'PC-03'
$machine = 'i21313'
$models = @('RT-DETR-R50', 'ViT-B/16')
$first = 'ViT-B/16'
$firstRound = 'vit_b16-r01'
```

En cada PC, después de su bloque:

```powershell
$env:BOTRYTIS_PC_ID = $pc
$operator = Read-Host 'Nombre del operador presente'
$dataset = Read-Host 'Raiz LOCAL activa del dataset con images, labels y classes.txt'
& $py tools/prepare_machine.py --pc-id $pc --machine-id $machine --dataset-dir $dataset
if ($LASTEXITCODE -ne 0) { throw 'Preparacion local detenida; conservar configuracion anterior' }
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/00_check_environment.py' -Arguments @('--config','configs/local.yaml','--pc-id',$pc,'--require-gpu','--require-ml','--gpu-selftest')
if ($LASTEXITCODE -ne 0) { throw 'Entorno/CUDA no validados' }
```

Solo PC-01 escribe una referencia después de revisarla personalmente. Las otras PCs reciben la base firmada y comparan su entorno. Esta operación no aprueba ningún gate.

```powershell
if ($pc -eq 'PC-01') {
    & $py src/00_check_environment.py --config configs/local.yaml --pc-id $pc --require-gpu --require-ml --gpu-selftest --write-reference
    if ($LASTEXITCODE -ne 0) { throw 'Referencia no verificada' }
}
```

## 3. Inventario, etiquetas, split y derivados

No existen imágenes reales en este repositorio. La referencia 329 no resuelve las menciones históricas de 324/328. Ejecutar inventario, revisar huérfanos, duplicados, anomalías y exclusiones. Si el conteo difiere, **detenerse**: una decisión documental personal debe fijar el inventario activo y `dataset.n_images_expected` en la configuración acordada, sin inventar muestras, activar `_off` ni desactivar la comprobación. Actualizar las tres configuraciones de manera consistente. La clase 0 es fruto físico; 1 sospecha positiva por pairing; 2 auxiliar. IoU 0,50 no es porcentaje enfermo.

```powershell
& $py src/19_audit_labels.py --dataset $dataset --output reports/v4/label_audit --min-iou 0.50
if ($LASTEXITCODE -ne 0) { throw 'Auditoria de labels pendiente' }
& $py src/01_validate_dataset.py --config configs/local.yaml --pc-id $pc
if ($LASTEXITCODE -ne 0) { throw 'Inventario/anotaciones no validos; no continuar' }
```

Solo PC-01 crea el split, una vez. Si ya es válido, el script lo verifica y conserva sus bytes. No usar `--force` para sortear otro split. Las otras PCs importan la base después de validar localmente el mismo dataset.

```powershell
if ($pc -eq 'PC-01') {
    & $py src/02_create_outer_split.py --config configs/local.yaml --pc-id $pc
    if ($LASTEXITCODE -ne 0) { throw 'Split detenido' }
    & $py src/03_create_inner_folds.py --config configs/local.yaml --pc-id $pc
    if ($LASTEXITCODE -ne 0) { throw 'Folds detenidos' }
    & $py src/orchestration/sync.py pack-base --config configs/local.yaml --out base_v4.zip
    if ($LASTEXITCODE -ne 0) { throw 'Exportacion base detenida' }
}
```

Copiar manualmente `base_v4.zip` y su `.zip.sha256` a PC-02/03. El ZIP base no contiene imágenes ni vinculaciones personales. No importar sobre una SQLite compartida. En cada receptora, tras su validación local:

```powershell
if ($pc -ne 'PC-01') {
    $baseZip = Read-Host 'Ruta LOCAL de base_v4.zip con su checksum'
    & $py src/orchestration/sync.py verify --config configs/local.yaml --zip $baseZip
    if ($LASTEXITCODE -ne 0) { throw 'ZIP base no verificado' }
    & $py src/orchestration/sync.py unpack --config configs/local.yaml --zip $baseZip
    if ($LASTEXITCODE -ne 0) { throw 'Hashes/configuracion distintos; no usar --force' }
}
& $py src/04_build_detection_dataset.py --config configs/local.yaml --pc-id $pc
if ($LASTEXITCODE -ne 0) { throw 'Dataset detector detenido' }
& $py src/05_build_classification_crops.py --config configs/local.yaml --pc-id $pc
if ($LASTEXITCODE -ne 0) { throw 'Crops detenidos' }
```

## 4. Pesos y QA: decisión y ejecución personales

Inventariar no descarga. Antes de adquirir un archivo, revisar su procedencia/licencia en el catálogo y sus hashes. RT-DETR-R50 sigue bloqueado: el R50-vd oficial y el grafo Ultralytics actual no son intercambiables. No sustituir por RT-DETR-L ni entrenar scratch dentro del experimento oficial preentrenado.

```powershell
& $py tools/manage_weights.py inventory
if ($LASTEXITCODE -ne 0) { throw 'Inventario de pesos detenido' }
```

Adquisición opcional de **un modelo**, únicamente por decisión personal. El siguiente bloque pide consentimiento adicional; no ejecutarlo para RT-DETR-R50 bloqueado. El catálogo aporta checksums/tamaño cuando están disponibles; un SHA independiente puede especificarse si procede. No existe descarga automática al construir modelos.

```powershell
& $py tools/manage_weights.py --help
if ((Read-Host 'Tras revisar licencia/procedencia, escriba ADQUIRIR para adquirir el modelo elegido') -ceq 'ADQUIRIR') {
    & $py tools/manage_weights.py acquire --architecture $first --allow-network --accept-license
    if ($LASTEXITCODE -ne 0) { throw 'Adquisicion no verificada; conservar .partial' }
}
```

Alternativa para un archivo ya transferido localmente, solo después de aceptar su licencia y disponer de SHA fiable. El archivo se verifica antes de publicarse en el almacenamiento del catálogo; no se sobrescriben pesos previos distintos.

```powershell
$weightFile = Read-Host 'Ruta LOCAL existente del archivo transferido'
$weightSha = Read-Host 'SHA256 completo obtenido de fuente fiable/recibo verificado'
& $py tools/manage_weights.py verify --architecture $first --file $weightFile --sha256 $weightSha --accept-license
if ($LASTEXITCODE -ne 0) { throw 'Pesos no verificados' }
& $py tools/qa.py --config configs/local.yaml --machine-id $machine
if ($LASTEXITCODE -ne 0) { throw 'QA tecnico pendiente; revisar logs_qa' }
```

El informe técnico no aprueba gates. Registrar únicamente comprobaciones realmente completadas, una por una. `READY_FOR_EXPERIMENT` exige los ocho gates previos. No ejecutar este ejemplo hasta contar con evidencia personal:

```powershell
$gate = Read-Host 'Gate realmente completado (consultar tools/manual_gates.py --help)'
$evidence = Read-Host 'Ruta o identificador de la evidencia personal'
& $py tools/manual_gates.py record --config configs/local.yaml --pc-id $pc --gate $gate --operator $operator --evidence $evidence --confirm-personal-execution
if ($LASTEXITCODE -ne 0) { throw 'Gate no registrado' }
```

## 5. Crear y ejecutar una sola arquitectura

Los bloques siguientes sirven con los valores `$first` y `$firstRound` específicos de cada PC de la sección 2. Así se reproduce el caso YOLOv8n / YOLO26l / ViT-B/16 simultáneos, cada uno desde su disco. No ejecutarlos hasta completar QA y verificar pesos del modelo.

```powershell
$roundConfig = (& $py tools/experiment_control.py --config configs/local.yaml new-round --round-id $firstRound --architecture $first --operator $operator --reason 'Primera ronda predeclarada' | ConvertFrom-Json)
if ($LASTEXITCODE -ne 0) { throw 'Ronda no creada; no reutilizar un ID existente' }
& $py tools/experiment_control.py --config $roundConfig status
if ($LASTEXITCODE -ne 0) { throw 'Estado no disponible' }
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/orchestration/worker.py' -Arguments @('--config',$roundConfig,'--pc-id',$pc,'--init-plan','--architectures',$first,'--kinds','baseline','--once')
if ($LASTEXITCODE -ne 0) { throw 'Baseline detenido; revisar logs y estado' }
& $py tools/experiment_control.py --config $roundConfig status
```

`--init-plan --architectures` crea solo los modelos solicitados y propios. Una ronda está ligada a una arquitectura: otros modelos requieren sus propias rondas. Es posible dejar todos los demás pendientes. Al terminar baseline: `PENDING_REVIEW`. Un fallo de publicación deja `PUBLICATION_PENDING`, nunca aprobación implícita.

Para otro modelo independiente de la misma PC, repetir `new-round` con ID nuevo, arquitectura activa propia y su motivo. Se puede ejecutar mientras el primero espera revisión, **secuencialmente** en esa GPU. Cada worker termina antes de lanzar otro worker local; el lock de proceso bloquea solapamientos. No se puede mover un modelo solo cambiando `pc_registry.first_batch`: usar la transferencia de la sección 9.

## 6. Revisar, aprobar y continuar por fases

Copiar el ID completo del job mostrado por `status` (incluye experimento/ronda). La evidencia debe describir resultados reales de esa etapa. Estos ejemplos **no** aprueban QA físico, freeze ni test.

```powershell
$jobId = Read-Host 'ID completo del job PENDING_REVIEW'
$evidence = Read-Host 'Evidencia revisada y decision'
& $py tools/experiment_control.py --config $roundConfig approve --job-id $jobId --operator $operator --evidence $evidence
if ($LASTEXITCODE -ne 0) { throw 'Etapa no aprobada' }
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/orchestration/worker.py' -Arguments @('--config',$roundConfig,'--pc-id',$pc,'--architectures',$first,'--kinds','hpo','--once')
if ($LASTEXITCODE -ne 0) { throw 'HPO detenido' }
```

El presupuesto total sigue siendo 15 trials/detector o 20/clasificador y tres folds. Para bloques, **predeclarar** `experiment.trial_block_size` con `--set` del worker; es una opción de ejecución, no modifica el presupuesto ni el espacio. Por ejemplo, sustituir el lanzamiento anterior por:

```powershell
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/orchestration/worker.py' -Arguments @('--config',$roundConfig,'--pc-id',$pc,'--set','experiment.trial_block_size=5','--architectures',$first,'--kinds','hpo','--once')
if ($LASTEXITCODE -ne 0) { throw 'Bloque HPO detenido' }
```

Un bloque parcial queda en revisión; `approve` vuelve a `PENDING`, no `COMPLETE`. Repetir el mismo comando tras revisar cada bloque hasta completar 15/20. Un cambio del espacio HPO o configuración científica exige nueva ronda, no `--n-trials` ni `--allow-budget-override` oficial. El estudio es secuencial y local.

Después del HPO completo, aprobar su job y ejecutar robustez. Se conserva top-3 × semillas 17/42/73 × tres folds; no distribuir el mismo estudio entre PCs sin la transferencia completa de ownership.

```powershell
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/orchestration/worker.py' -Arguments @('--config',$roundConfig,'--pc-id',$pc,'--architectures',$first,'--kinds','repeat','aggregate')
if ($LASTEXITCODE -ne 0) { throw 'Robustez detenida' }
& $py tools/experiment_control.py --config $roundConfig status
```

Al terminar `aggregate`, revisar y aprobar ese job. Solo entonces puede ejecutarse `outer_train`, con el 70 % completo y épocas fijadas por folds internos, sin validar externamente cada época.

```powershell
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/orchestration/worker.py' -Arguments @('--config',$roundConfig,'--pc-id',$pc,'--architectures',$first,'--kinds','outer_train','--once')
if ($LASTEXITCODE -ne 0) { throw 'Candidato detenido' }
```

## 7. Nueva ronda, repeat, retry, resume y publicación

Nueva ronda: cambia configuración con motivo explícito, sin tocar rondas previas. El siguiente ejemplo es una **decisión ilustrativa de configuración**, no una mejora medida ni recomendación de LR:

```powershell
$newConfig = (& $py tools/experiment_control.py --config $roundConfig new-round --round-id ($firstRound + '-revision2') --architecture $first --operator $operator --reason 'Problema diagnosticado y cambio predeclarado; ver evidencia' --change 'detector.baseline.lr0=0.002' | ConvertFrom-Json)
if ($LASTEXITCODE -ne 0) { throw 'Nueva ronda detenida' }
```

Para clasificadores usar una clave de `classifier`, no el LR de detector del ejemplo. `repeat` crea otro ID conservando configuración; no acepta cambios. Repetir un baseline completo requiere nueva ronda, nunca `--force`.

```powershell
$repeatConfig = (& $py tools/experiment_control.py --config $roundConfig repeat --round-id ($firstRound + '-repeat') --architecture $first --operator $operator --reason 'Repeticion explicita preservando ambas ejecuciones' | ConvertFrom-Json)
if ($LASTEXITCODE -ne 0) { throw 'Repeat detenido' }
$jobId = Read-Host 'ID de job FAILED, RUNNING interrumpido o PUBLICATION_PENDING'
$evidence = Read-Host 'Diagnostico del fallo y evidencia de procesos detenidos'
```

Elegir **solo una** acción según el estado. `retry` para FAILED (máximo tres intentos automáticos); `resume` para RUNNING interrumpido exige locks locales libres; `publish` reintenta publicar sin recalcular. Nunca usar `resume` con procesos vivos. Los folds completos y compatibles se conservan y verifican antes de reutilizarse. Los checkpoints de época requieren su recibo y configuración exacta; recuperación numérica de GPU/AMP/DataLoader de Windows pendiente de QA, no prometer equivalencia bit a bit.

```powershell
& $py tools/experiment_control.py --config $roundConfig retry --job-id $jobId --operator $operator --evidence $evidence
```

```powershell
& $py tools/experiment_control.py --config $roundConfig resume --job-id $jobId --operator $operator --evidence $evidence
```

```powershell
& $py tools/experiment_control.py --config $roundConfig publish --job-id $jobId --operator $operator --evidence $evidence
```

Después de retry/resume, invocar el worker de la misma ronda, etapa y modelo. Para preparar una reanudación de checkpoint propio (no ejecuta sin consentimiento adicional):

```powershell
$checkpoint = Read-Host 'Ruta local real del checkpoint de epoca con .resume.json'
& $py tools/resume_local.py --config $roundConfig --architecture $first --checkpoint $checkpoint --entry src/06_run_baseline.py
if ($LASTEXITCODE -ne 0) { throw 'Checkpoint incompatible; no inventar uno' }
```

## 8. Comparar rondas sin reentrenar

```powershell
$compareOut = Read-Host 'Carpeta local NUEVA para comparacion'
& $py tools/experiment_control.py compare --configs $roundConfig $newConfig --output $compareOut
if ($LASTEXITCODE -ne 0) { throw 'Comparacion detenida' }
```

`comparison.json`, `.csv` y `.png` conservan diferencias, métricas existentes, tiempos y VRAM. Si no hay datos, se indica pendiente; no se inventan. Revisar `methodologically_equivalent` y hashes de protocolo/espacio; una comparación visual no equivale a comparabilidad científica.

## 9. Transferencia manual de ownership (estudio/ronda completo)

Requiere misma versión Git, misma ronda/configuración científica y mismos hashes de datos/split/folds en ambas copias locales. En la receptora, usar `mirror-round` con el config y manifiesto originales de la ronda y su configuración base local validada. Esto conserva identidad científica y hashes declarados de pesos, con rutas/binding locales. No se necesitan pesos preentrenados para importar solo resultados; sí se verifican antes de entrenar allí. Verificar coincidencia con la oferta. No ejecutar la ronda receptora antes de activar. Detener procesos; nunca transferir RUNNING. Solo se transfiere la arquitectura completa de esa ronda, no trials sueltos entre dos dueños.


Copiar también los metadatos originales `config.yaml` y `manifest.json` de la ronda, sin editarlos. En la receptora, antes de recibir:

```powershell
$sourceConfig = Read-Host 'Config original de la ronda recibida'
$sourceManifest = Read-Host 'Manifest original de esa ronda'
$roundConfig = (& $py tools/experiment_control.py --config configs/local.yaml mirror-round --source-config $sourceConfig --source-manifest $sourceManifest --operator $operator --reason 'Espejo local para importacion/transferencia autorizada' | ConvertFrom-Json)
if ($LASTEXITCODE -ne 0) { throw 'Metadatos, base o codigo incompatibles' }
```

Origen, con snapshot ya publicable y motivo documentado:

```powershell
$bundle = Read-Host 'Ruta LOCAL NUEVA para transferencia.zip'
$target = Read-Host 'PC receptora (PC-01, PC-02 o PC-03)'
& $py tools/experiment_control.py --config $roundConfig transfer-offer --architecture $first --pc-id $pc --target $target --bundle $bundle --operator $operator --evidence 'Reasignacion pendiente autorizada presencialmente'
if ($LASTEXITCODE -ne 0) { throw 'Oferta detenida' }
```

El origen queda bloqueado **antes de entregar la oferta**. Copiar manualmente ZIP + `.zip.sha256` + `.offer.json`. La receptora no obtiene autoridad al importar solamente el ZIP. Usar su `$roundConfig` local correspondiente, `$first` del modelo transferido y `$pc` receptor:

```powershell
$offer = [IO.Path]::ChangeExtension($bundle, '.offer.json')
$receipt = Read-Host 'Ruta local nueva para recibo.json'
& $py tools/experiment_control.py --config $roundConfig transfer-receive --architecture $first --pc-id $pc --bundle $bundle --offer $offer --receipt $receipt --operator $operator
if ($LASTEXITCODE -ne 0) { throw 'Recepcion detenida; no entrenar' }
```

Copiar recibo al origen. Confirmar allí; copiar confirmación a la receptora. Mantener los archivos como evidencia y no restaurar snapshots antiguos del archivo ownership.

```powershell
$confirmation = Read-Host 'Ruta local nueva para confirmacion.json'
& $py tools/experiment_control.py --config $roundConfig transfer-confirm --architecture $first --receipt $receipt --confirmation $confirmation --operator $operator
if ($LASTEXITCODE -ne 0) { throw 'Confirmacion detenida' }
```

```powershell
& $py tools/experiment_control.py --config $roundConfig transfer-activate --architecture $first --receipt $receipt --confirmation $confirmation --operator $operator
if ($LASTEXITCODE -ne 0) { throw 'Activacion detenida' }
& $py src/orchestration/worker.py --config $roundConfig --pc-id $pc --init-plan --architectures $first --status
if ($LASTEXITCODE -ne 0) { throw 'Plan receptor detenido' }
```

Los estados extranjeros del ZIP son evidencia, nunca actualizan una cola local automáticamente. La oferta de transferencia, a diferencia de un ZIP ordinario, lleva una instantánea autorizada de jobs, trials y revisiones de esa arquitectura: la receptora conserva esas etapas y no repite un baseline ya aprobado. Una cola local con trabajo propio diferente bloquea la adopción. No marcar jobs como COMPLETE por copiar una SQLite. Ante corte entre pasos, conservar el origen bloqueado y revisar oferta/recibo/confirmación; no borrar ownership ni cambiar YAML para forzar avance. No existe garantía transaccional global ante restauración de discos o copias antiguas no controladas.

## 10. Exportar/importar resultados sin transferir ownership

```powershell
$exportZip = Read-Host 'Ruta local NUEVA del ZIP de resultados'
& $py src/orchestration/sync.py pack-results --config $roundConfig --pc-id $pc --out $exportZip --with-checkpoints
if ($LASTEXITCODE -ne 0) { throw 'Exportacion detenida; revisar PUBLICATION_PENDING' }
```

Copiar manualmente ZIP y checksum. En la receptora, usar configuración local de **esa misma ronda**, con su binding local; si aún no existe, usar `mirror-round` con los metadatos originales después de validar la misma base.

```powershell
$importZip = Read-Host 'Ruta LOCAL del ZIP recibido'
& $py src/orchestration/sync.py verify --config $roundConfig --zip $importZip
if ($LASTEXITCODE -ne 0) { throw 'ZIP no verificado' }
& $py src/orchestration/sync.py unpack --config $roundConfig --zip $importZip
if ($LASTEXITCODE -ne 0) { throw 'Conflicto/hash distinto: conservar archivos; no usar --force' }
```

`recover-import` completa un journal interrumpido; verificar argumentos con `--help`. Nunca resolver conflictos eliminando archivos locales diferentes. La exportación usa snapshots publicados entre jobs, no pesos mutando en entrenamiento. Los checkpoints de época solo se publican con un recibo SHA válido. GitHub transporta exclusivamente código/documentación/configuración versionable.

## 11. Validación externa, cohorte, selección, integración, freeze y test

Iterar únicamente en folds internos del 70 %. Antes de abrir el 20 %, elegir personalmente una ronda por arquitectura. Predeclarar una única autorización externa por arquitectura en el registro del experimento. Ni un cambio de ronda ni una nueva configuración permiten reutilizarla para explorar.

```powershell
& $py tools/experiment_control.py --config $roundConfig authorize-external --operator $operator --evidence 'Ronda elegida con folds internos; politica de calibracion/seleccion predeclarada'
if ($LASTEXITCODE -ne 0) { throw 'Validacion externa no autorizada' }
& $py src/orchestration/worker.py --config $roundConfig --pc-id $pc --architectures $first --kinds outer_eval thresholds
if ($LASTEXITCODE -ne 0) { throw 'Evaluacion/calibracion detenida' }
```

Una operación externa STARTED que falló sin resultado publicado puede recuperarse mediante `retry-external --stage outer_validation` o `threshold_tuning`, con operador/evidencia. Si ya existe resultado, se bloquea una segunda optimización; conservarlo y diagnosticar, no borrar el ledger.

PC-01 reúne por ZIP los resultados de las siete rondas elegidas. Sus configuraciones locales y artefactos deben estar disponibles. RT-DETR bloqueado impide completar la cohorte oficial: no omitirlo ni usar `--allow-partial`.

```powershell
$candidateConfigs = @(Read-Host 'Config local YOLOv8n'; Read-Host 'Config local YOLOv8l'; Read-Host 'Config local YOLO26l'; Read-Host 'Config local RT-DETR-R50'; Read-Host 'Config local ResNet50'; Read-Host 'Config local ViT-B/16'; Read-Host 'Config local Swin-Tiny')
$cohortConfig = (& $py tools/experiment_control.py assemble --configs $candidateConfigs --round-id cohorte-final-01 --operator $operator --reason 'Siete candidatos elegidos antes de consultar test' | ConvertFrom-Json)
if ($LASTEXITCODE -ne 0) { throw 'Cohorte incompleta o incompatible' }
& $py src/12_select_detector.py --config $cohortConfig --pc-id PC-01
if ($LASTEXITCODE -ne 0) { throw 'Seleccion detector detenida' }
& $py src/13_select_classifier.py --config $cohortConfig --pc-id PC-01
if ($LASTEXITCODE -ne 0) { throw 'Seleccion clasificador detenida' }
& $py src/14_validate_integrated_pipeline.py --config $cohortConfig --pc-id PC-01
if ($LASTEXITCODE -ne 0) { throw 'Pipeline detenido' }
```

Si `assemble` señala protocolos no equivalentes, revisar la comparación; solo una decisión explícita documentada puede aceptar `--accept-non-equivalent`, que queda registrada y no convierte los protocolos en equivalentes. El pipeline emplea bounding boxes predichas y crops predichos, no reutiliza crops GT como predicciones.

El freeze exige decisiones metodológicas personalmente confirmadas en la configuración de cohorte y gates pendientes. Estos comandos son para la fase final del usuario, no fueron ejecutados:

```powershell
& $py src/15_freeze_experiment.py --config $cohortConfig --pc-id PC-01
if ($LASTEXITCODE -ne 0) { throw 'Freeze no valido; revisar decisiones y evidencias' }
& $py tools/manual_gates.py record --config $cohortConfig --pc-id PC-01 --gate FREEZE --operator $operator --evidence 'Freeze revisado personalmente y checklist firmado' --confirm-personal-execution
if ($LASTEXITCODE -ne 0) { throw 'Gate FREEZE no registrado' }
& $py tools/manual_gates.py release-test --config $cohortConfig --pc-id PC-01 --operator $operator --evidence 'Autorizacion posterior al freeze, firmada personalmente' --confirm-personal-execution
if ($LASTEXITCODE -ne 0) { throw 'TEST_RELEASE no autorizado' }
& $py src/16_evaluate_test.py --config $cohortConfig --pc-id PC-01
if ($LASTEXITCODE -ne 0) { throw 'Test final detenido; no reiniciar ni borrar sus controles' }
& $py src/17_benchmark_latency.py --config $cohortConfig --pc-id PC-01
if ($LASTEXITCODE -ne 0) { throw 'Benchmark detenido' }
& $py src/18_export_master_results.py --config $cohortConfig --pc-id PC-01
if ($LASTEXITCODE -ne 0) { throw 'Exportacion maestra detenida' }
```

## 12. Actualización controlada de código

En cada estación, después de detener y verificar todos sus procesos, exportar snapshots y respaldar artefactos locales. No ejecutar `git pull` automáticamente ni resetear cambios del usuario. La nueva versión invalida identidades incompatibles; crear nueva ronda y repetir QA de compatibilidad antes de reanudar.

```powershell
if (git status --porcelain) { throw 'Preservar cambios locales antes de actualizar' }
$NewApprovedCommit = Read-Host 'Nuevo SHA completo revisado y aprobado'
if ($NewApprovedCommit -notmatch '^[0-9a-fA-F]{40}$') { throw 'SHA invalido' }
git fetch origin $Branch
if ($LASTEXITCODE -ne 0) { throw 'Fetch fallo' }
git checkout --detach $NewApprovedCommit
if ($LASTEXITCODE -ne 0) { throw 'Checkout detenido' }
if ((git rev-parse HEAD).Trim() -ne $NewApprovedCommit.ToLower()) { throw 'Version distinta' }
& $py -m pip install -r requirements.txt -r requirements-dev.txt -c requirements-ml.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependencias no verificadas' }
& $py -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Entorno incompatible' }
```
