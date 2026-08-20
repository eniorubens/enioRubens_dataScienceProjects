# Download, place, and locally revalidate the Kaggle V2 package outputs.
param(
    [string]$KaggleExe = "kaggle",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
# PYTHONIOENCODING only covers stdio; the CLI also writes the kernel log
# to a file with the locale codec, which is cp1252 here. Force UTF-8 mode.
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
$kernelMeta = Get-Content (Join-Path $root "kaggle\kernel-p\kernel-metadata.json") -Raw |
    ConvertFrom-Json
$kernelSlug = $kernelMeta.id

$status = (& $KaggleExe kernels status $kernelSlug) -join " "
Write-Output "Kernel status: $status"
if ($status -notmatch "complete") {
    Write-Output "Kernel is not complete yet; nothing downloaded."
    exit 0
}

$downloadDir = Join-Path $root "temp\kaggle_output_p"
New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
& $KaggleExe kernels output $kernelSlug -p $downloadDir
if ($LASTEXITCODE -ne 0) {
    throw "Kernel output download failed"
}

$artifact = Join-Path $downloadDir "v2_package.json"
$manifest = Join-Path $downloadDir "v2_results.json"
foreach ($file in @($artifact, $manifest)) {
    if (-not (Test-Path $file)) {
        throw "Expected output missing from kernel: $file"
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $root "temp\v2") | Out-Null
Copy-Item -Path $artifact -Destination (Join-Path $root "temp\v2") -Force
Copy-Item -Path $manifest -Destination (Join-Path $root "config") -Force
Write-Output "Placed temp\v2\v2_package.json and config\v2_results.json"

# The joblib bundle exists only when the reproduction gate passed.
$bundle = Join-Path $downloadDir "consumer_complaint_detector_v2.joblib"
if (Test-Path $bundle) {
    New-Item -ItemType Directory -Force -Path (Join-Path $root "artifacts\v2") |
        Out-Null
    Copy-Item -Path $bundle -Destination (Join-Path $root "artifacts\v2") -Force
    Write-Output "Placed artifacts\v2\consumer_complaint_detector_v2.joblib"
} else {
    Write-Output ("No joblib bundle in the kernel output. Expected only when " +
        "the run ended in REPRODUCTION_MISMATCH.")
}

$code = @'
import json
import sys

sys.path.insert(0, "src")
from consumer_complaint_intelligence.v2_package import validate_v2_manifest

manifest = validate_v2_manifest(
    "config/v2_results.json",
    "temp/v2/v2_package.json",
)
with open("temp/v2/v2_package.json", encoding="utf-8") as handle:
    artifact = json.load(handle)
gate = artifact.get("reproduction_gate") or {}
outer = (artifact.get("outer") or {}).get("metrics") or {}
calibration = artifact.get("calibration") or {}
summary = {
    "status": manifest.get("status"),
    "outcome": artifact.get("outcome"),
    "complete": artifact.get("complete"),
    "frozen": artifact.get("frozen"),
    "gate_passed": gate.get("passed"),
    "gate_check_count": gate.get("check_count"),
    "failed_checks": gate.get("failed_checks"),
    "calibrated_threshold": calibration.get("threshold"),
    "outer_effective_overrides": (artifact.get("outer") or {}).get(
        "effective_overrides"
    ),
    "outer_critical_f1": outer.get("critical_f1"),
    "outer_critical_precision": outer.get("critical_precision"),
    "outer_critical_recall": outer.get("critical_recall"),
    "outer_macro_f1": outer.get("macro_f1"),
    "hard_negative": artifact.get("hard_negative"),
    "bundle": artifact.get("bundle"),
}
if gate.get("divergences"):
    summary["divergences"] = gate["divergences"]
print(json.dumps(summary, indent=2, sort_keys=True))
'@
Push-Location $root
try {
    $code | & $PythonExe -
    if ($LASTEXITCODE -ne 0) {
        throw "Local revalidation of the V2 package manifest failed"
    }
} finally {
    Pop-Location
}
Write-Output "Local revalidation passed."
