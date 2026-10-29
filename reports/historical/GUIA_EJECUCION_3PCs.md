# Guía de ejecución en las 3 PCs (paso a paso, con VS Code)

Esta guía asume Windows 10/11, una GPU NVIDIA por PC y VS Code. Todo se hace desde VS Code: terminal integrada (`Ctrl+ñ` o `Ctrl+\``) y tareas (`Ctrl+Shift+P` → **Tasks: Run Task**; las tareas están en `.vscode/tasks.json` y piden el ID de PC, la arquitectura o el ZIP con un menú).

> **Regla de oro:** el código y `configs/default.yaml` deben ser **idénticos** en las 3 PCs (el `code_hash` se compara). Si cambia una sola línea en una PC, hay que copiar el cambio a las otras dos **antes** de seguir.

## 0. Cómo funciona el reparto (léelo una vez)

- Cada PC tiene su **propia cola local** (`artifacts/queue/queue.sqlite`) y sus propios estudios Optuna (un `.db` por arquitectura). No hay servidor central.
- Por eso el reparto es **estático**: cada PC se lanza solo con sus arquitecturas (`--architectures`). Así nunca dos PCs entrenan la misma arquitectura.

| PC | Arquitecturas (slug) | Rol extra |
|---|---|---|
| **PC-01** | `yolov8n yolov8m yolo26n yolo26s` | Prepara los datos, **congela**, evalúa **test**, hace el **benchmark** de latencia y consolida |
| **PC-02** | `yolov8l yolo26l` | Solo entrena |
| **PC-03** | `rtdetr_r50 resnet50 vit_b16 swin_tiny` | Solo entrena |

- Cada arquitectura recorre sola toda la cadena: baseline (06) → HPO Optuna 15/20 trials (07) → repetición top-3 × 3 seeds (08) → entrenamiento outer (09) → evaluación en validation (10) → umbrales (11).
- El **test** (paso 16) y el **benchmark** (17) solo se hacen en PC-01, una vez, al final.

## 1. Preparación de CADA PC (PC-01, PC-02, PC-03)

### 1.1 Software
1. **Driver NVIDIA** actualizado. Verifica en PowerShell: `nvidia-smi` (debe listar la GPU y mostrar CUDA ≥ 12.6).
2. **Python 3.12 (64-bit)** desde python.org; marca *Add python.exe to PATH*. Verifica: `python --version`.
3. **VS Code** + extensiones: *Python*, *Pylance*, *YAML*, *Rainbow CSV* (VS Code las sugiere al abrir la carpeta).
4. Permite scripts (PowerShell como usuario normal, una sola vez):
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```

### 1.2 Copiar el proyecto (ruta CORTA y sin OneDrive)
1. Copia `Botrytis_Experiment_Framework_FINAL.zip` a la PC y descomprímelo en una ruta corta, **fuera de OneDrive**, p. ej. `C:\botrytis\`. Debe quedar `C:\botrytis\Botrytis_Experiment_Framework\`.
2. Comprueba la integridad (el `.sha256` viaja junto al ZIP):
   ```powershell
   Get-FileHash C:\botrytis\Botrytis_Experiment_Framework_FINAL.zip -Algorithm SHA256
   ```
   El hash debe coincidir con el del `.zip.sha256`.
3. En VS Code: **File → Open Folder…** → `C:\botrytis\Botrytis_Experiment_Framework`.

### 1.3 Entorno virtual e instalación (terminal de VS Code, en la raíz del proyecto)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 1) PyTorch con CUDA 12.6 (primero, desde su índice)
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
# 2) resto de dependencias (versiones fijadas)
pip install -r requirements.txt
```
Luego `Ctrl+Shift+P` → **Python: Select Interpreter** → elige `.venv\Scripts\python.exe`.

> Las **mismas versiones** en las 3 PCs son obligatorias (se comparan en el paso 00). No actualices nada "por tu cuenta".

### 1.4 Identidad de la PC
Una vez por PC (cambia el número). Cierra y reabre VS Code después:
```powershell
setx BOTRYTIS_PC_ID PC-01     # en PC-02: PC-02   |  en PC-03: PC-03
```

### 1.5 Ajustes de Windows para entrenar horas sin cortes
- Configuración → Sistema → Energía: **suspender = Nunca**; plan *Alto rendimiento*.
- Desactiva reinicios automáticos de Windows Update durante la ejecución.
- Antivirus: excluye la carpeta `C:\botrytis` (acelera y evita bloqueos de archivos).
- Cierra navegadores/juegos: la GPU debe estar libre para el entrenamiento.

### 1.6 Verificar el entorno
Tarea **A0 · entorno de ESTA PC** (elige tu PC) o manual:
```powershell
python src\00_check_environment.py --pc-id PC-01 --require-gpu --require-ml --gpu-selftest
```
Debe terminar sin `FAIL`. En PC-02/03, las diferencias contra la referencia aparecerán solo después del paso 3.

---

## 2. PC-01 · Preparar los datos (Fase A, solo PC-01)

### 2.1 Dataset
Copia las 328 imágenes y sus etiquetas YOLO (`.txt`, mismo nombre) a:
```
C:\botrytis\Botrytis_Experiment_Framework\data\raw\images\   (328 imágenes)
C:\botrytis\Botrytis_Experiment_Framework\data\raw\labels\   (328 .txt)
```
(Si están en otra carpeta: `setx BOTRYTIS_IMAGES_DIR "D:\...\images"` y `setx BOTRYTIS_LABELS_DIR "D:\...\labels"`, y reabre VS Code.)

### 2.2 Rellenar los 3 valores bloqueantes en `configs/default.yaml`
Abre `configs/default.yaml` en VS Code y busca `TODO_BLOQUEANTE` / `TODO_PREDECLARE`:

| Clave | Qué poner |
|---|---|
| `classes.fruit_class_id` | Número de la clase **fruto** en tus `.txt` YOLO (la línea, empezando en 0, donde aparece "fruto" en `classes.txt`) |
| `classes.sign_class_id` | Número de la clase **signo_visual_compatible** |
| `outer_split.seed` | Un entero que **decidas ahora** (p. ej. `20260101`), **antes de ver cualquier resultado**; anótalo con su fecha |

Ejemplo: si `classes.txt` dice `fruto` en la línea 1 y `signo_visual_compatible` en la 2 → `fruit_class_id: 0` y `sign_class_id: 1`.

### 2.3 Ejecutar los pasos 01–05 (Tareas A1 a A5, o terminal)
```powershell
python src\01_validate_dataset.py          --pc-id PC-01
python src\02_create_outer_split.py        --pc-id PC-01
python src\03_create_inner_folds.py        --pc-id PC-01
python src\04_build_detection_dataset.py   --pc-id PC-01
python src\05_build_classification_crops.py --pc-id PC-01
```
Qué mirar:
- **01**: `reports/` con el informe del dataset; debe decir 328 imágenes y **0 errores**. Si hay WARNINGS, léelos.
- **02**: `artifacts/split_manifest.csv` (70/20/10). **No se re-sortea jamás**.
- **03**: 3 folds internos dentro del 70 % de train.
- **04–05**: datasets de detección (clase única `fruit`) y crops de clasificación con la regla `intersección/área_fruto > 0.75`.

### 2.4 Referencia de entorno y ZIP base
```powershell
python src\00_check_environment.py --pc-id PC-01 --require-gpu --require-ml --write-reference
python src\orchestration\sync.py pack-base --pc-id PC-01 --out ..\sync_001_base.zip
Get-FileHash ..\sync_001_base.zip -Algorithm SHA256
```
(Tareas **A0b** y **S1**.) `sync_001_base.zip` contiene dataset crudo, splits, folds, manifiestos y la referencia de entorno. **Guarda el hash.**

---

## 3. PC-02 y PC-03 · Recibir la base

1. Copia `sync_001_base.zip` (por Drive/USB) **una carpeta por encima** del proyecto (`C:\botrytis\sync_001_base.zip`) y comprueba el hash (`Get-FileHash`): debe ser idéntico al de PC-01.
2. Desempaqueta (Tarea **S2**, o terminal):
   ```powershell
   python src\orchestration\sync.py verify  --pc-id PC-02 --zip ..\sync_001_base.zip
   python src\orchestration\sync.py unpack  --pc-id PC-02 --zip ..\sync_001_base.zip --force
   ```
   (en PC-03 cambia `PC-02` por `PC-03`).
3. **Regenera** los datasets derivados en la PC (producen los mismos hashes porque todo es determinista):
   ```powershell
   python src\01_validate_dataset.py          --pc-id PC-02
   python src\04_build_detection_dataset.py   --pc-id PC-02
   python src\05_build_classification_crops.py --pc-id PC-02
   ```
   **No ejecutes 02 ni 03**: ya vienen en el ZIP (solo el 02 con `--force` los reharía; no lo hagas).
4. Verifica el entorno contra la referencia:
   ```powershell
   python src\00_check_environment.py --pc-id PC-02 --require-gpu --require-ml --gpu-selftest
   ```
   Debe pasar. Si marca diferencias **críticas** (versión de torch, hash del dataset/split, `code_hash`), **no sigas**: arregla la PC (ver §9).
5. Comprueba que el manifiesto de clasificación coincide con PC-01:
   ```powershell
   Get-FileHash artifacts\classification_manifest.csv -Algorithm SHA256
   ```
   Debe dar el mismo valor en las 3 PCs (también puedes comparar `artifacts\classification_manifest.sha256`).

---

## 4. Ejecutar el entrenamiento (las 3 PCs, en paralelo)

Antes de lanzar, prueba **unos minutos** que nada revienta (en cada PC, una sola vez):
```powershell
python src\orchestration\worker.py --pc-id PC-01 --init-plan --dry-run --architectures yolov8n yolov8m yolo26n yolo26s
```
Muestra la cola y el siguiente job sin ejecutar nada.

### 4.1 Lanzar el worker (Tarea **B · WORKER**, o terminal)
```powershell
# PC-01
python src\orchestration\worker.py --pc-id PC-01 --init-plan --wait --architectures yolov8n yolov8m yolo26n yolo26s
# PC-02
python src\orchestration\worker.py --pc-id PC-02 --init-plan --wait --architectures yolov8l yolo26l
# PC-03
python src\orchestration\worker.py --pc-id PC-03 --init-plan --wait --architectures rtdetr_r50 resnet50 vit_b16 swin_tiny
```
- Déjalo corriendo en **una terminal de VS Code dedicada**. No cierres VS Code (o lanza el worker desde PowerShell normal fuera de VS Code).
- Ejecuta **1 job de GPU a la vez por PC**. Es lo correcto: no lo cambies.
- Para **pararlo limpio**: `Ctrl+C` una vez. El job en curso queda como `RUNNING` y se recupera después (§8).

### 4.2 Monitorizar (en otra terminal de VS Code)
```powershell
python src\orchestration\worker.py --pc-id PC-01 --status      # Tarea "B · worker: estado de la cola"
nvidia-smi                                                      # GPU: uso y memoria
Get-Content artifacts\queue\logs\*.log -Tail 30 -Wait           # log en vivo
```
Estados de job: `PENDING → RUNNING → COMPLETE` (o `FAILED`, o `PRUNED` para trials podados por Optuna, que es normal).
Cada trial deja su carpeta: `runs\<arquitectura>\<trial>\` con `execution.log`, `metrics.json`, `config.yaml`, `environment.json`, `checkpoint_metadata.json`, `figures\`.

### 4.3 Tiempos orientativos
Con 15/20 trials × 3 folds × hasta 100 épocas (con early stopping y pruning) cuenta **de horas a varios días por arquitectura** según la GPU. Los YOLO grandes y RT-DETR son los más lentos. Las PCs trabajan en paralelo; el cuello de botella es la PC con más carga (PC-01 y PC-03 con 4 arquitecturas cada una). Si una PC termina antes y otra va atrasada, **no muevas arquitecturas a mano** sin leer §8.

### 4.4 Qué hacer si algo falla
| Síntoma | Acción |
|---|---|
| Job `FAILED` | Lee `artifacts\queue\logs\<job>.log` y el `execution.log` del trial; corrige la causa; `worker.py --pc-id PC-0N --retry-failed` y relanza el worker |
| Corte de luz / reinicio / cerraste el worker con un job `RUNNING` | `worker.py --pc-id PC-0N --force-recover --dry-run` (revisa) y luego sin `--dry-run`; relanza el worker. Optuna y los checkpoints retoman donde quedaron |
| `CUDA out of memory` | El código ya reduce el micro-batch conservando el batch efectivo. Cierra otros programas; reintenta con `--retry-failed`. No cambies batch/imgsz a mano: si persiste, anótalo y decídelo antes de seguir |
| Entorno no coincide | `python src\00_check_environment.py --pc-id PC-0N` y corrige versiones |

---

## 5. Consolidar resultados en PC-01

Cuando **PC-02 y PC-03 terminen** todos sus jobs (`--status` sin PENDING/RUNNING; y PC-01 también los suyos):

1. **En PC-02 y PC-03** (Tarea **S3**):
   ```powershell
   python src\orchestration\sync.py pack-results --pc-id PC-02 --with-checkpoints --out ..\sync_results_PC-02.zip
   Get-FileHash ..\sync_results_PC-02.zip -Algorithm SHA256
   ```
   (`--with-checkpoints` es imprescindible: PC-01 necesita los pesos finales para el pipeline y el test.)
2. Lleva los ZIP a PC-01, verifica hashes, y en **PC-01** (Tarea **S5**):
   ```powershell
   python src\orchestration\sync.py verify --pc-id PC-01 --zip ..\sync_results_PC-02.zip
   python src\orchestration\sync.py unpack --pc-id PC-01 --zip ..\sync_results_PC-02.zip
   python src\orchestration\sync.py unpack --pc-id PC-01 --zip ..\sync_results_PC-03.zip
   ```
   No sobrescribe la cola de PC-01 (la cola nunca viaja en los ZIP). Si avisa de un **conflicto**, NO uses `--force` sin entender cuál de los dos es el correcto.
3. Comprueba que están las 10 arquitecturas: `dir artifacts\best_configs` (10 ficheros) y `dir runs` (10 carpetas).

---

## 6. PC-01 · Selección, congelación, test (en orden, sin saltarse ninguno)

```powershell
python src\12_select_detector.py             --pc-id PC-01   # elige el mejor de los 7 detectores (en validation)
python src\13_select_classifier.py           --pc-id PC-01   # elige el mejor de los 3 clasificadores
python src\14_validate_integrated_pipeline.py --pc-id PC-01 --retune-classifier-threshold
```
Revisa `reports\` y las figuras antes de continuar: detector y clasificador elegidos, empates, pipeline en validation con crops **predichos**.

**Antes de congelar**: abre `configs/default.yaml`, lee la lista `decisions.adopted_from_annex` (decisiones que adoptó el ANEXO de la metodología y tú debes confirmar) y, si estás de acuerdo con todas, cambia:
```yaml
decisions:
  confirmed_by_user: true
```
Si **no** estás de acuerdo con alguna, cámbiala ahora (y documenta el cambio). Después de editar el YAML, **copia el cambio a PC-02 y PC-03** si pensaras volver a ejecutar algo allí.

```powershell
python src\15_freeze_experiment.py           --pc-id PC-01   # CONGELA: irreversible, escribe freeze_manifest.json
python src\16_evaluate_test.py               --pc-id PC-01   # ÚNICA evaluación sobre test
```
> **Después de 15 no edites NADA de `src\`, `configs\` ni los datos**: el `code_hash` cambia y 16/17 se bloquean a propósito. El test se evalúa **una sola vez**; si se interrumpe sin resultados, existe `--retry-after-crash` (léelo en el log antes de usarlo).

---

## 7. PC-01 · Benchmark y exportación final

1. Cierra todo lo que use GPU (navegador, juegos, otros Python, **detén el worker**). Comprueba `nvidia-smi`: sin procesos `python.exe`.
2. Conecta la PC a la corriente y con plan Alto rendimiento. No uses la PC durante la medición.
```powershell
python src\17_benchmark_latency.py          --pc-id PC-01   # 10 warm-up + 100 mediciones, solo en PC-01
python src\18_export_master_results.py      --pc-id PC-01   # tablas maestras + auditoría científica de 8 puntos
```
Resultados:
- `reports\tables\` tablas maestras, `reports\figures\` figuras obligatorias.
- `reports\audit_report.md` + `.json`: **todos los puntos deben estar en PASS**. Si hay `FAIL`, el script termina con error y te dice cuál.

Guarda todo el proyecto (`runs\`, `artifacts\`, `reports\`) como respaldo final y comprímelo.

---

## 8. Recuperación y reparto dinámico (avanzado)

- **Reintentar fallidos:** `worker.py --pc-id PC-0N --retry-failed`.
- **Huérfanos tras un corte:** `--force-recover` (solo jobs `RUNNING` de *esa* PC; confirma antes con el Administrador de tareas que no hay `python.exe` entrenando).
- **Una PC se retrasa mucho:** puedes pasar una arquitectura *completa y sin empezar* a otra PC: en la PC receptora, lanza el worker con `--architectures <slug>` (la arquitectura no debe haber empezado en ninguna otra PC). Si ya tenía un `.db` de Optuna y `runs\<arq>\` en la PC original, copia `artifacts\optuna\botrytis_<slug>.db` y `runs\<slug>\` a la receptora **y déjala solo allí**. Nunca dos PCs con la misma arquitectura.

## 9. Solución de problemas comunes

| Error / síntoma | Causa y solución |
|---|---|
| `TODO_BLOQUEANTE ... classes.fruit_class_id` | Falta rellenar el YAML (§2.2) |
| `No existe la referencia de entorno` | En PC-01 falta `00 --write-reference` o no desempaquetaste la base en PC-02/03 |
| `00_check_environment` marca torch/python distintos | Reinstala exactamente `requirements.txt` + torch del §1.3 |
| Discrepancia de `code_hash` | Hay un archivo distinto entre PCs. Vuelve a descomprimir el ZIP del proyecto (o copia el archivo modificado) |
| `dataset_hash` / `split_hash` distintos | La PC no usó el ZIP base. Repite §3 |
| `Activate.ps1 ... ejecución de scripts deshabilitada` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `nvidia-smi` no existe / CUDA no disponible | Reinstala el driver NVIDIA; `python -c "import torch;print(torch.cuda.is_available())"` debe imprimir `True` |
| `17` dice "GPU ocupada" | Hay otro `python.exe` usando la GPU: ciérralo (no uses `--allow-busy-gpu` en el benchmark oficial) |
| `16`/`17` dicen que el freeze no coincide | Se modificó código/config/datos después del 15. Revierte el cambio (no re-congeles sin documentarlo) |
| Rutas con tildes/espacios dan errores | Usa una ruta corta sin tildes (`C:\botrytis`) |
| OneDrive bloquea archivos | No trabajes dentro de OneDrive/Documentos |

## 10. Lista de "sí / no"

**SÍ**: mismo código y YAML en las 3 PCs · predeclarar la seed del split · un solo worker por PC · revisar `--status` a diario · respaldar `runs\` y `artifacts\` · verificar hashes de cada ZIP.
**NO**: re-ejecutar 02/03 en PC-02/03 · tocar el test antes del paso 16 · editar código tras congelar · cambiar batch size/imgsz/presupuestos · arquitectura repetida en dos PCs · mover `queue.sqlite` entre PCs · usar `--force`/`--skip-env-check`/`--allow-*` en la corrida oficial.

## 11. Resumen de una línea por fase

1. Cada PC: Python 3.12 + venv + torch cu126 + `requirements.txt` + `setx BOTRYTIS_PC_ID`.
2. PC-01: datos → YAML (clases + seed) → 01–05 → `00 --write-reference` → `sync pack-base`.
3. PC-02/03: `sync unpack` base → 01, 04, 05 → `00`.
4. Las 3: `worker --init-plan --wait --architectures …`.
5. PC-02/03: `sync pack-results --with-checkpoints` → PC-01: `sync unpack` ×2.
6. PC-01: 12 → 13 → 14 → `confirmed_by_user: true` → 15 → 16 → 17 (GPU libre) → 18.
