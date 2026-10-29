# Guia de ejecucion sincronizada v4

QA_MANUAL_PENDIENTE · EXPERIMENTO_PENDIENTE · cero pesos iniciales.
HTML: [GUIA](../../entregables_botrytis/GUIA_EJECUCION_3PCs_ACTUALIZADA.html) · [FLUJO](../../entregables_botrytis/FLUJO_COMPLETO_BOTRYTIS_HORIZONTAL.html).

## 01 Inventario e identidad local

Cada PC · GT / preparacion

Entrada: ZIP completo, proyecto previo, carpeta activa; mapa acordado

Salida: configs/local.yaml y vinculacion explicita; originales intactos

Verificacion personal: Comparar MANIFEST_ENTREGA; leer classes.txt. El hostname solo se registra.

El inventario actual solo lista nombres: 324 imagenes/324 labels; referencia 329. No mover _off a images sin decision documentada.

```powershell
$ErrorActionPreference = 'Stop'
$base = 'C:\Users\Erick Obradovich\OneDrive\Documentos\ULima\2026-2\SEMINARIO DE INVESTIGACIÓN II\Código'
# En otra PC, indique SU carpeta local donde copio personalmente el paquete.
if (-not (Test-Path -LiteralPath $base -PathType Container)) { $base = Read-Host 'Ruta local de la carpeta Codigo' }
$root = Join-Path $base 'Botrytis_Experiment_Framework'
if (-not (Test-Path -LiteralPath (Join-Path $root 'configs\default.yaml'))) { throw 'Proyecto local ausente; extraiga el ZIP en una carpeta vacia y revise antes de integrar.' }
Set-Location -LiteralPath $root
$pc = Read-Host 'Identidad fisica: PC-01, PC-02 o PC-03'
$map = @{ 'PC-01'='i21315'; 'PC-02'='i21314'; 'PC-03'='i21313' }
if (-not $map.ContainsKey($pc)) { throw 'PC no valida' }
$machine = $map[$pc]
$env:BOTRYTIS_PC_ID = $pc
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) { $py = Read-Host 'Ruta completa de Python local real (no alias WindowsApps)' }
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) { throw 'Python local no disponible; no se instala automaticamente' }

$dataset = Read-Host 'Raiz ACTIVA del dataset local con images y labels'
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'tools/prepare_machine.py' -Arguments @('--pc-id',$pc,'--machine-id',$machine,'--dataset-dir',$dataset)
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
```

## 02 Entorno, CUDA y seguridad personales

Cada PC · Preparacion

Entrada: Python/venv locales y versiones acordadas

Salida: environment.json + evidencia CUDA local

Verificacion personal: Ejecutar en persona en las tres PCs. PC-01 registra referencia con --write-reference solo despues de comprobarla.

Historico no verificado ahora: la guia del 28/10 reporto Python 3.12.10, Torch 2.15.0+cu132, Torchvision 0.29.2+cu132, RTX 6000 Ada y CUDA 13.2 en i21315. No acredita el estado actual. No se cambian drivers/politicas.

```powershell
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/00_check_environment.py' -Arguments @('--config','configs/local.yaml','--require-gpu','--require-ml','--gpu-selftest')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
```

## 03 Gestion de pesos: inicialmente cero

Cada PC · Detector + clasificador

Entrada: Catalogo oficial fijado; conexion/licencia cuando el usuario decida

Salida: weights/ + .provenance.json con SHA256 y tamano

Verificacion personal: NO_DISPONIBLE inicial. Nunca se carga un cache como sustituto. RT-DETR-R50 pretrained bloqueado.

ResNet50 tiene prefijo oficial 11ad3fa6; el hash completo y tamano se registran localmente al verificar. ViT usa 21k→1k: declarar esta diferencia respecto a 1k. Sin Internet: transferencia manual de archivo y procedencia o esperar. Scratch requiere decision y experimento separado.

```powershell
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'tools/manage_weights.py' -Arguments @('inventory')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
# SOLO si decidio personalmente adquirirlos y reviso las licencias de cada fuente.
$catalog = Get-Content -LiteralPath 'configs\weights_registry.json' -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($m in $catalog.models) {
    if (-not $m.compatible_pretrained) { Write-Host ('BLOQUEADO: ' + $m.name + ' - ' + $m.block_reason); continue }
    $go = Read-Host ('Adquirir ' + $m.name + ' desde ' + $m.url + '? Escriba ADQUIRIR para aceptar su licencia')
    if ($go -ne 'ADQUIRIR') { Write-Host 'PENDIENTE_USUARIO'; continue }
    $wa = @('acquire','--architecture',[string]$m.slug,'--allow-network','--accept-license')
    if ($m.official_sha256) { $wa += @('--sha256',[string]$m.official_sha256) }
    if ($m.size_bytes) { $wa += @('--expected-size',[string]$m.size_bytes) }
    & .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'tools/manage_weights.py' -Arguments $wa
    if ($LASTEXITCODE -ne 0) { throw 'Adquisicion/verificacion detenida; se preserva .partial.' }
}

```

## 04 QA manual del usuario, gate cerrado

Cada PC · GATE personal

Entrada: Checklist; equipo fisico; datos/pesos listos segun prueba

Salida: Logs tecnicos; checklist firmado por la persona

Verificacion personal: El codigo de salida del runner no aprueba gates. Registrar cada resultado, comando, fecha, operador y evidencia.

Equivalentes disponibles: scripts/Invoke-QA.ps1 y scripts/qa.sh. NO fueron invocados en este encargo. El smoke de entrenamiento requiere decision personal adicional y pesos; nunca es un resultado cientifico.

```powershell
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'tools/qa.py' -Arguments @('--config','configs/local.yaml','--machine-id',$machine)
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
```

## 05 Clases y matching GT; resolver anomalias

Cada PC · GT auxiliar

Entrada: 0=arandano_a; 1=arandano_sospechoso; 2=signo_sospechoso_botrytis

Salida: dataset_inventory, issues, pairing audit/candidates/anomalies; hash

Verificacion personal: Solo 0 es fruto fisico; matching bipartito global max IoU, umbral tecnico 0.50 configurable. Conflicto bloquea/quarantena explicita; ausencia de 2 no cambia positivos. Revisar distribucion IoU antes de split.

Contar images real por stem, labels vacios, 3 clases, coordenadas, duplicados, positivos sin par y decisiones. La cobertura de clase 2 es descriptiva. Reconciliar 324 vs 329 sin borrar labels/imagenes.

```powershell
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/19_audit_labels.py' -Arguments @('--dataset',$dataset,'--output','reports/v4/label_audit','--min-iou','0.50')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/01_validate_dataset.py' -Arguments @('--config','configs/local.yaml')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
```

## 06 Split externo y tres folds internos

PC-01 crea una vez; resto importa manualmente · GT

Entrada: Auditoria resuelta; imagenes originales; seed 67

Salida: split 70/20/10; inner folds; test_lock; hashes

Verificacion personal: Estratos por imagen con positivos emparejados; cuotas globales por mayor resto. Folds solo del 70%. Todos los crops heredan imagen.

No re-sortear un split valido. Un split v3 o seed anterior es incompatible; --force SOLO tras cambio documentado conserva copia superseded. Este encargo no creo/regenero ningun split real.

```powershell
if ($pc -ne 'PC-01') { throw 'Solo PC-01 crea el split; importe manualmente la base comun.' }
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/02_create_outer_split.py' -Arguments @('--config','configs/local.yaml')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/03_create_inner_folds.py' -Arguments @('--config','configs/local.yaml')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
```

## 07 Detector una clase y crops exclusivamente clase 0

Cada PC · Detector / clasificador

Entrada: Split/folds firmados; bbox clase 0; etiquetas por matching

Salida: data/derived_v4/detection y classification; manifests + sellos

Verificacion personal: Una bbox 0→un crop; etiqueta positiva iff par 1↔0 valido, negativa solo imagen auditada. Comprobar hashes, filas y fuga.

La preparacion reproduce transformaciones predeclaradas. El test derivado sigue sellado; no se usa en HPO, tuning ni seleccion.

```powershell
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/04_build_detection_dataset.py' -Arguments @('--config','configs/local.yaml')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/05_build_classification_crops.py' -Arguments @('--config','configs/local.yaml')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
```

## 08 Baselines, HPO, robustez y candidatos

Cada PC · Detector + clasificador

Entrada: READY_FOR_EXPERIMENT aprobado personalmente; datos/modelos verificados

Salida: 15 trials/detector, 20/clasificador; top3 × seeds17/42/73 × folds3; candidatos entrenados70%

Verificacion personal: Baselines con configuracion efectiva; HPO solo validacion interna. El 20% externo se evalua despues del entrenamiento de epocas fijadas.

Un trabajo GPU por PC; asignacion fija documental. PC-03 mostrara BLOQUEADO en RT-DETR-R50 hasta resolver el backend/checkpoint. No omitirlo silenciosamente ni reducir presupuestos. El worker nunca se conecta a otras PCs.

```powershell
$assign = @{
    'PC-01'=@('YOLOv8n','YOLOv8m','YOLO26n','YOLO26s')
    'PC-02'=@('YOLOv8l','YOLO26l')
    'PC-03'=@('RT-DETR-R50','ResNet50','ViT-B/16','Swin-Tiny')
}
$wa = @('--config','configs/local.yaml','--pc-id',$pc,'--init-plan','--architectures') + $assign[$pc]
$wa += @('--kinds','baseline','hpo','repeat','aggregate','outer_train','outer_eval')
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/orchestration/worker.py' -Arguments $wa
if ($LASTEXITCODE -ne 0) { throw 'Worker local detenido; revisar evidencia y bloqueo, no reintentar a ciegas.' }

```

## 09 Intercambio manual e importacion segura

Cada PC · Artefactos

Entrada: Snapshots inmutables, propietario, hashes de codigo/datos/split

Salida: ZIP + sidecar SHA256, import_ledger/journal, evidencias por PC

Verificacion personal: CRC, SHA, rutas relativas, propietario y conflictos. Nunca importar sobre resultados locales distintos; recover-import recupera transaccion.

Copie el dataset original por separado personalmente si hace falta. Los bundles base/resultados no habilitan el test ni transfieren aprobaciones personales. No usar carpetas compartidas remotas en ejecucion.

```powershell
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/orchestration/sync.py' -Arguments @('pack-base','--config','configs/local.yaml','--out','base_v4.zip')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
# Tras cada tanda terminada en cada PC (la base solo la exporta PC-01):
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/orchestration/sync.py' -Arguments @('pack-results','--config','configs/local.yaml','--out','resultados_locales_v4.zip','--with-checkpoints')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
# Copie personalmente ZIP + .sha256 con USB/medio autorizado a la carpeta local de destino.
$bundle = Read-Host 'Ruta LOCAL del ZIP recibido'
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/orchestration/sync.py' -Arguments @('verify','--config','configs/local.yaml','--zip',$bundle)
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/orchestration/sync.py' -Arguments @('unpack','--config','configs/local.yaml','--zip',$bundle)
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
```

## 10 Seleccion externa de detector y clasificador

PC-01 coordinadora · Seleccion

Entrada: Metricas del 20% externo de los diez candidatos; sin test

Salida: selected_detector.json, selected_classifier.json, policy hash

Verificacion personal: mAP50:95 principal detector; AP positiva clasificador. Evidencia N real, desempates predeclarados. No mezclar HPO con validacion externa.

No continuar si faltan candidatos o el bloqueo RT-DETR sigue abierto. La politica queda fijada antes de examinar el test.

```powershell
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/12_select_detector.py' -Arguments @('--config','configs/local.yaml')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/13_select_classifier.py' -Arguments @('--config','configs/local.yaml')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
```

## 11 Umbrales con el 20% externo

PC-01 · Detector + clasificador

Entrada: Modelos seleccionados; predicciones externas

Salida: inference_thresholds; umbral prob, conf, NMS

Verificacion personal: Separar IoU tecnico GT, IoU evaluacion y NMS. YOLO26 puede usar head end-to-end; registrar rama real y pertinencia de NMS.



```powershell
$sd = Get-Content -LiteralPath 'artifacts\v4\selected_detector.json' -Raw -Encoding UTF8 | ConvertFrom-Json
$sc = Get-Content -LiteralPath 'artifacts\v4\selected_classifier.json' -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($selection in @($sd,$sc)) {
    $name = [string]$selection.architecture
    if (-not $name) { throw 'Seleccion sin architecture; revise el JSON y su procedencia.' }
    & .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/11_tune_inference_thresholds.py' -Arguments @('--config','configs/local.yaml','--architecture',$name)
    if ($LASTEXITCODE -ne 0) { throw 'Threshold tuning detenido' }
}

```

## 12 Validar pipeline integrado

PC-01 · Integracion

Entrada: Detector→crops predichos→clasificador; GTauditado

Salida: pipeline_validation; tablas detecciones/misses; matriz con no_detectado

Verificacion personal: Matching detecciones↔GT 1:1 por confianza/IoU declarado. E_det=FN/N_GT; E_cls=EC/N_match; E_global=(FN+EC)/N_GT. FP/duplicados aparte.

N_GT=0: error no definido; N_match=0: E_cls no definido. No hay loss ni entrenamiento conjunto del hibrido.

```powershell
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/14_validate_integrated_pipeline.py' -Arguments @('--config','configs/local.yaml')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
```

## 13 Freeze: sellar decisiones y checkpoints

PC-01 · GATE personal

Entrada: Seleccion y umbrales revisados; metadatos y todos los candidatos

Salida: freeze_manifest.json y SHA; test aun bloqueado

Verificacion personal: Confirmar decisiones por escrito y verificar hashes. Freeze no abre automaticamente el test.



```powershell
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/15_freeze_experiment.py' -Arguments @('--config','configs/local.yaml','--set','decisions.confirmed_by_user=true')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
```

## 14 Desbloqueo exclusivamente personal del test

PC-01 · GATE test

Entrada: Gate FREEZE registrado y freeze integro

Salida: manual_test_release.json ligado al SHA256 del freeze

Verificacion personal: Solo la persona autoriza; todos los gates se entregan NO EJECUTADO. Cambiar freeze invalida release.



```powershell
$operator = Read-Host 'Su nombre (ejecucion presencial)'
$evidence = Read-Host 'Ruta/identificador de evidencia del freeze revisado'
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'tools/manual_gates.py' -Arguments @('release-test','--config','configs/local.yaml','--pc-id','PC-01','--operator',$operator,'--evidence',$evidence,'--confirm-personal-execution')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
```

## 15 Test final una vez tras freeze y release

PC-01 · Test

Entrada: 10% externo sellado; freeze y autorizacion personal

Salida: test_results.json + hashes + errores cualitativos

Verificacion personal: Sin HPO/seleccion/tuning sobre test. --reproduce compara; --retry-after-crash exige evidencia de corte.



```powershell
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/16_evaluate_test.py' -Arguments @('--config','configs/local.yaml')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
```

## 16 Benchmark comun y eficiencia

PC-01 · Medicion

Entrada: Modelos congelados; PC-01 libre

Salida: latencia por etapas; VRAM; protocolo de benchmark

Verificacion personal: Mismo hardware, batch1, resolucion, precision, warmup y sincronizacion CUDA; tiempos de PCs distintas no se mezclan.



```powershell
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/17_benchmark_latency.py' -Arguments @('--config','configs/local.yaml')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
```

## 17 Figuras, fuentes y exportaciones

PC-01 · Informe

Entrada: CSV/JSON historiales reales y descriptores trazables

Salida: PNG300dpi + SVG/PDF; figure_manifest.csv/json y pendientes

Verificacion personal: Dos figuras principales de dos paneles. Si faltan losses/autenticidad, PENDIENTE. No sumar losses YOLO ni fabricar curvas del hibrido.

Algunas fuentes (historial seleccionado, estratos, activaciones, ejemplos) requieren contrato explicito en docs/FIGURAS.md. No se inventan columnas ni condiciones visuales.

```powershell
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'src/18_export_master_results.py' -Arguments @('--config','configs/local.yaml')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'tools/prepare_figure_sources.py' -Arguments @('--config','configs/local.yaml')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
& .\scripts\Invoke-Botrytis.ps1 -PythonPath $py -Entry 'tools/generate_all_figures.py' -Arguments @('--source-index','reports/v4/figure_sources.json')
if ($LASTEXITCODE -ne 0) { throw 'Comando detenido: revise runs/v4 y la evidencia antes de continuar.' }
```

