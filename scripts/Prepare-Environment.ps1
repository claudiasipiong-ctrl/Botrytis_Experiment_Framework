param(
    [Parameter(Mandatory=$true)][string]$PythonExecutable,
    [Parameter(Mandatory=$true)][string]$TorchVersion,
    [Parameter(Mandatory=$true)][string]$TorchvisionVersion,
    [Parameter(Mandatory=$true)][string]$TorchIndexUrl
)
$traceRoot = Join-Path (Split-Path -Parent $PSScriptRoot) 'runs\v4\_powershell'
$traceId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss.fffffffZ') + '_prepare_' + [Guid]::NewGuid().ToString('N')
$traceDir = Join-Path $traceRoot $traceId
New-Item -ItemType Directory -Path $traceDir -Force | Out-Null
$traceCode = 1
@{event='bootstrap_start'; utc=[DateTime]::UtcNow.ToString('o'); phase='Prepare-Environment'; qa_status='QA_MANUAL_PENDIENTE'} | ConvertTo-Json -Compress | Set-Content -LiteralPath (Join-Path $traceDir 'events.jsonl') -Encoding UTF8
Start-Transcript -LiteralPath (Join-Path $traceDir 'execution.log') -Append | Out-Null
try {
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if ($TorchIndexUrl -notmatch '^https://download\.pytorch\.org/whl/cu[0-9]+/?$') {
    throw 'Indique un indice CUDA oficial verificado para el driver real.'
}
if ($TorchVersion -notmatch '^[0-9]+\.[0-9]+\.[0-9]+$' -or $TorchvisionVersion -notmatch '^[0-9]+\.[0-9]+\.[0-9]+$') {
    throw 'Las versiones deben ser exactas, numericas y la pareja debe comprobarse en PyTorch.'
}
& $PythonExecutable -c 'import os,sys; assert os.name=="nt" and sys.version_info>=(3,10); print(sys.executable,sys.version)'
if ($LASTEXITCODE -ne 0) { throw 'Python Windows >=3.10 no disponible.' }
if (Test-Path -LiteralPath '.venv') { throw 'Ya existe .venv; no se modifica silenciosamente.' }
& $PythonExecutable -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear .venv.' }
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $python -m pip install "torch==$TorchVersion" "torchvision==$TorchvisionVersion" --index-url $TorchIndexUrl
if ($LASTEXITCODE -ne 0) { throw 'Instalacion PyTorch fallida.' }
& $python -m pip install -r requirements.txt -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependencias fallidas.' }
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependencias incompatibles.' }
Write-Output 'Entorno creado. Todavia requiere QA; no se comprobo CUDA ni se inicio entrenamiento.'

$traceCode = 0
} catch {
    $_ | Out-String | Write-Host
    throw
} finally {
    @{event='bootstrap_end'; utc=[DateTime]::UtcNow.ToString('o'); returncode=$traceCode; qa_status='QA_MANUAL_PENDIENTE'} | ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $traceDir 'events.jsonl') -Encoding UTF8
    Stop-Transcript | Out-Null
}
