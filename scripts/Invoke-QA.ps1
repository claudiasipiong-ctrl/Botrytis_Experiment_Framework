param(
    [string]$BashPath,
    [string]$PythonPath,
    [string]$Config = 'configs/local.yaml',
    [ValidateSet('i21315','i21314','i21313')][string]$MachineId,
    [switch]$AllowCudaSelftest,
    [switch]$AllowTrainingSmoke,
    [switch]$FullSuite,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$QAArguments
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$logRoot = Join-Path $projectRoot 'logs_qa'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fffffff'
$bootDir = Join-Path $logRoot ("bootstrap_" + $env:COMPUTERNAME + "_" + $stamp)
New-Item -ItemType Directory -Path $bootDir | Out-Null
if ($PythonPath) { $env:BOTRYTIS_PYTHON = $PythonPath }
$candidates = @()
if ($BashPath) {
    $candidates += $BashPath
} else {
    foreach ($prefix in @($env:ProgramFiles, ${env:ProgramFiles(x86)}, $env:LOCALAPPDATA)) {
        if ($prefix) {
            $candidates += (Join-Path $prefix 'Git\bin\bash.exe')
            $candidates += (Join-Path $prefix 'Programs\Git\bin\bash.exe')
        }
    }
    $found = Get-Command bash.exe -ErrorAction SilentlyContinue
    if ($found) { $candidates += $found.Source }
}
$selected = $null
foreach ($candidate in ($candidates | Select-Object -Unique)) {
    if (-not (Test-Path -LiteralPath $candidate)) { continue }
    try {
        $probe = & $candidate --noprofile --norc -c 'printf "BOTRYTIS_BASH_%s\n" "$BASH_VERSION"' 2>&1
        if ($LASTEXITCODE -eq 0 -and ($probe -match 'BOTRYTIS_BASH_[0-9]')) {
            $selected = $candidate
            break
        }
    } catch { $_ | Out-String | Add-Content -LiteralPath (Join-Path $bootDir 'probe_errors.log') }
}
if (-not $selected) {
    @{
        status='BLOCKED'; reason='No se encontro Bash funcional. Instale o indique uno; no se instala WSL ni Git automaticamente.'
        candidates=$candidates; time=(Get-Date).ToString('o')
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $bootDir 'result.json') -Encoding UTF8
    Write-Error 'Bash no disponible. Evidencia en logs_qa. Indique -BashPath C:\ruta\bash.exe.'
    exit 3
}
@{status='PASS'; bash=$selected; time=(Get-Date).ToString('o')} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $bootDir 'result.json') -Encoding UTF8
$forward = @('--config', $Config)
if ($MachineId) { $forward += @('--machine-id', $MachineId) }
if ($AllowCudaSelftest) { $forward += '--allow-cuda-selftest' }
if ($AllowTrainingSmoke) { $forward += '--allow-training-smoke' }
if ($FullSuite) { $forward += '--full-suite' }
if ($QAArguments) { $forward += $QAArguments }
& $selected --noprofile --norc scripts/qa.sh @forward 2>&1 |
    Tee-Object -FilePath (Join-Path $bootDir 'console.log')
$qaExit = $LASTEXITCODE
exit $qaExit
