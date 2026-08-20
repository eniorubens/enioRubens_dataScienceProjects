# Download, place, and locally revalidate the Kaggle V2-D1 outputs.
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
$kernelMeta = Get-Content (Join-Path $root "kaggle\kernel-metadata.json") -Raw |
    ConvertFrom-Json
$kernelSlug = $kernelMeta.id

$status = (& $KaggleExe kernels status $kernelSlug) -join " "
Write-Output "Kernel status: $status"
if ($status -notmatch "complete") {
    Write-Output "Kernel is not complete yet; nothing downloaded."
    exit 0
}

$downloadDir = Join-Path $root "temp\kaggle_output"
New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
& $KaggleExe kernels output $kernelSlug -p $downloadDir
if ($LASTEXITCODE -ne 0) {
    throw "Kernel output download failed"
}

$artifact = Join-Path $downloadDir "v2_classical_benchmark.json"
$manifest = Join-Path $downloadDir "v2_classical_results.json"
foreach ($file in @($artifact, $manifest)) {
    if (-not (Test-Path $file)) {
        throw "Expected output missing from kernel: $file"
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $root "temp\v2") | Out-Null
Copy-Item -Path $artifact -Destination (Join-Path $root "temp\v2") -Force
Copy-Item -Path $manifest -Destination (Join-Path $root "config") -Force
Write-Output "Placed temp\v2\v2_classical_benchmark.json and config\v2_classical_results.json"

$code = @'
import json
import sys

sys.path.insert(0, "src")
from consumer_complaint_intelligence.v2_benchmark import validate_v2_manifest

manifest = validate_v2_manifest(
    "config/v2_classical_results.json",
    "temp/v2/v2_classical_benchmark.json",
)
with open("temp/v2/v2_classical_benchmark.json", encoding="utf-8") as handle:
    artifact = json.load(handle)
evidence = artifact.get("hard_negative") or {}
baseline = (artifact.get("fallback_baseline") or {}).get(
    "outer_evaluation"
) or {}
deltas = [
    (candidate.get("outer") or {}).get("critical_f1_vs_fallback")
    for candidate in artifact.get("candidates") or ()
]
deltas = [value for value in deltas if value is not None]
print(
    json.dumps(
        {
            "status": manifest["status"],
            "complete": manifest["complete"],
            "selected": manifest["selected"],
            "candidate_count": manifest["candidate_count"],
            "degenerate_null": artifact.get("degenerate_null"),
            "selection_blocked_reason": artifact.get(
                "selection_blocked_reason"
            ),
            "fallback_outer_critical_f1": baseline.get("critical_f1"),
            "best_critical_f1_vs_fallback": max(deltas) if deltas else None,
            "funnel": {
                "margin_eligible_count": evidence.get("margin_eligible_count"),
                "effective_eligible_count": evidence.get(
                    "effective_eligible_count"
                ),
                "fallback_beating_eligible_count": evidence.get(
                    "fallback_beating_eligible_count"
                ),
            },
        },
        indent=2,
        sort_keys=True,
    )
)
'@
Push-Location $root
try {
    $code | & $PythonExe -
    if ($LASTEXITCODE -ne 0) {
        throw "Local revalidation of the V2 manifest failed"
    }
} finally {
    Pop-Location
}
Write-Output "Local revalidation passed."
