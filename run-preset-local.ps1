param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('c1', 'c1-vision', 'c4')]
    [string]$Preset
)
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '..\Strata\.venv\Scripts\python.exe'
$taskConfig = Join-Path $PSScriptRoot "strata-$Preset-local.json"
$taskSettings = Get-Content -LiteralPath $taskConfig -Raw | ConvertFrom-Json
$taskListener = Get-NetTCPConnection -LocalPort $taskSettings.port -State Listen -ErrorAction SilentlyContinue
$taskEngine = Get-CimInstance Win32_Process -Filter "Name = 'strata.exe'" |
    Where-Object { $_.ExecutablePath -eq $taskSettings.exe }
if ($taskListener -or $taskEngine) {
    Write-Host 'Strata is already running, or port 8080 is occupied. Stop the current server before starting another preset.'
    exit 1
}
Write-Host "Starting Strata preset: $Preset. Press Ctrl+C to stop."
& $taskPython (Join-Path $PSScriptRoot 'serve\server.py') --engine strata --config $taskConfig --host 127.0.0.1 --port 8080
exit $LASTEXITCODE
