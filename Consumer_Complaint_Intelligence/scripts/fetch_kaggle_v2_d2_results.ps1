# Download, place, and locally revalidate the Kaggle V2-D2 outputs.
param(
    [string]$KaggleExe = "kaggle",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
# The Kaggle CLI writes the kernel log with the locale codec (cp1252 here)
# and dies on non-ASCII output. UTF-8 mode covers stdio and file writes.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$root = Split-Path -Parent $PSScriptRoot
$configView = (& $KaggleExe config view) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw "Kaggle CLI authentication failed"
}
$usernameMatch = [regex]::Match($configView, "username:\s*(\S+)")
$username = $usernameMatch.Groups[1].Value
if (-not $usernameMatch.Success -or $username -eq "None") {
    throw "Kaggle CLI did not resolve a username; token may be invalid."
}
# Read the kernel slug from the metadata so the two cannot drift apart.
$kernelMeta = Get-Content (Join-Path $root "kaggle\kernel-d2\kernel-metadata.json") -Raw |
    ConvertFrom-Json
$kernelSlug = $kernelMeta.id

$status = (& $KaggleExe kernels status $kernelSlug) -join " "
Write-Output "Kernel status: $status"
if ($status -notmatch "complete") {
    Write-Output "Kernel is not complete yet; nothing downloaded."
    exit 0
}

$downloadDir = Join-Path $root "temp\kaggle_output_d2"
New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
& $KaggleExe kernels output $kernelSlug -p $downloadDir
if ($LASTEXITCODE -ne 0) {
    throw "Kernel output download failed"
}

$artifact = Join-Path $downloadDir "v2_transformer_challenge.json"
$manifest = Join-Path $downloadDir "v2_transformer_results.json"
foreach ($file in @($artifact, $manifest)) {
    if (-not (Test-Path $file)) {
        throw "Expected output missing from kernel: $file"
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $root "temp\v2") | Out-Null
Copy-Item -Path $artifact -Destination (Join-Path $root "temp\v2") -Force
Copy-Item -Path $manifest -Destination (Join-Path $root "config") -Force
Write-Output ("Placed temp\v2\v2_transformer_challenge.json and " +
    "config\v2_transformer_results.json")

$code = @'
import json
import sys

sys.path.insert(0, "src")
from consumer_complaint_intelligence.v2_transformer import validate_d2_manifest

manifest = validate_d2_manifest(
    "config/v2_transformer_results.json",
    "temp/v2/v2_transformer_challenge.json",
)
with open("temp/v2/v2_transformer_challenge.json", encoding="utf-8") as handle:
    artifact = json.load(handle)
reported = artifact.get("reported") or {}
outer = reported.get("outer") or {}
metrics = outer.get("metrics") or {}
decision = artifact.get("decision") or {}
summary = {
    "status": manifest.get("status"),
    "complete": manifest.get("complete"),
    "outcome": decision.get("outcome"),
    "reported_seed": reported.get("seed"),
    "reported_outer_effective_overrides": outer.get("effective_overrides"),
    "reported_outer_critical_f1": metrics.get("critical_f1"),
    "reported_outer_critical_precision": metrics.get("critical_precision"),
    "reported_outer_critical_recall": metrics.get("critical_recall"),
    "reported_outer_macro_f1": metrics.get("macro_f1"),
    "critical_f1_vs_fallback": reported.get("critical_f1_vs_fallback"),
    "critical_f1_vs_incumbent": reported.get("critical_f1_vs_incumbent"),
    "seed_spread": artifact.get("seed_spread"),
}
for key, value in decision.items():
    if isinstance(value, bool):
        summary[f"decision_{key}"] = value
print(json.dumps(summary, indent=2, sort_keys=True))
'@
Push-Location $root
try {
    $code | & $PythonExe -
    if ($LASTEXITCODE -ne 0) {
        throw "Local revalidation of the V2 D2 manifest failed"
    }
} finally {
    Pop-Location
}
Write-Output "Local revalidation passed."
