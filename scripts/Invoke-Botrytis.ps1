[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$PythonPath,
    [Parameter(Mandatory=$true)][string]$Entry,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss.fffffffZ')
$runId = $stamp + '_' + $env:BOTRYTIS_PC_ID + '_' + [Guid]::NewGuid().ToString('N')
$boot = Join-Path $root ('runs\v4\_powershell\' + $runId)
New-Item -ItemType Directory -Path $boot -Force | Out-Null
$events = Join-Path $boot 'events.jsonl'
$log = Join-Path $boot 'execution.log'
$result = 1
function Protect-Text([string]$Text) {
    $Text = [regex]::Replace($Text,'(?i)(password|passwd|token|secret|api[_-]?key)(\s*[=:]\s*|\s+)[^\s,;]+','$1=[REDACTED]')
    return [regex]::Replace($Text,'(?i)(Bearer\s+)[A-Za-z0-9_.+/=-]+','$1[REDACTED]')
}
function Add-Event([string]$Kind, [hashtable]$Data) {
    $row = @{ utc=[DateTime]::UtcNow.ToString('o'); event=$Kind; run_id=$runId; data=$Data }
    $safe = Protect-Text ($row | ConvertTo-Json -Compress -Depth 8)
    [IO.File]::AppendAllText($events, $safe + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))
}
try {
    Add-Event 'bootstrap_start' @{entry=$Entry; pc=$env:BOTRYTIS_PC_ID; hostname=$env:COMPUTERNAME}
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw 'PythonPath no existe: indique un Python local real.' }
    $target = [IO.Path]::GetFullPath((Join-Path $root $Entry))
    $prefix = [IO.Path]::GetFullPath($root).TrimEnd('\') + '\'
    if (-not $target.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'Entry debe estar dentro del proyecto local.' }
    if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw ('Entry ausente: ' + $target) }
    Push-Location -LiteralPath $root
    try {
        & $PythonPath -u $target @Arguments 2>&1 | ForEach-Object {
            $line = Protect-Text ($_ | Out-String)
            [IO.File]::AppendAllText($log,$line,(New-Object Text.UTF8Encoding($false)))
            Write-Host $line.TrimEnd()
        }
        $result = $LASTEXITCODE
    } finally { Pop-Location }
} catch {
    $detail = Protect-Text ($_ | Out-String)
    [IO.File]::AppendAllText($log,$detail,(New-Object Text.UTF8Encoding($false)))
    Add-Event 'bootstrap_error' @{reason=$detail}
    Write-Host $detail -ForegroundColor Red
    $result = 1
} finally {
    Add-Event 'bootstrap_end' @{returncode=$result; status=$(if($result -eq 0){'COMMAND_COMPLETE'}else{'COMMAND_FAILED'})}
    @{run_id=$runId; returncode=$result; qa_status='QA_MANUAL_PENDIENTE'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $boot 'final_status.json') -Encoding UTF8
}
exit $result
