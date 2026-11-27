# Download and locally validate the V3 E1 Kaggle aggregate smoke output.
param(
    [string]$KaggleExe = "kaggle",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$root = Split-Path -Parent $PSScriptRoot
$kernelMeta = Get-Content (Join-Path $root "kaggle\kernel-v3-e1\kernel-metadata.json") -Raw |
    ConvertFrom-Json
$kernelSlug = $kernelMeta.id

$status = (& $KaggleExe kernels status $kernelSlug) -join " "
Write-Output "Kernel status: $status"
if ($status -notmatch "complete") {
    Write-Output "Kernel is not complete yet; nothing downloaded."
    exit 0
}

$downloadDir = Join-Path $root "temp\kaggle_output_v3_e1"
New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
& $KaggleExe kernels output $kernelSlug -p $downloadDir
if ($LASTEXITCODE -ne 0) {
    throw "V3 E1 kernel output download failed"
}

$artifact = Join-Path $downloadDir "v3_e1_smoke.json"
if (-not (Test-Path $artifact)) {
    throw "Expected V3 E1 aggregate output missing: $artifact"
}

$code = @'
import json
import sys

sys.path.insert(0, "src")
from consumer_complaint_intelligence.v3_kaggle import (
    load_v3_e1_config,
    validate_v3_e1_result,
)

config = load_v3_e1_config("config/v3_e1_execution.json")
with open("temp/kaggle_output_v3_e1/v3_e1_smoke.json", encoding="utf-8") as handle:
    result = json.load(handle)
summary = validate_v3_e1_result(result, config)
print(json.dumps(summary, indent=2, sort_keys=True))
'@

Push-Location $root
try {
    $code | & $PythonExe -
    if ($LASTEXITCODE -ne 0) {
        throw "Local validation of V3 E1 aggregate output failed"
    }
} finally {
    Pop-Location
}
Write-Output "Local V3 E1 aggregate validation passed."
