# Build and publish the V3 E1 Kaggle smoke scaffold after review approval.
param(
    [string]$KaggleExe = "kaggle",
    [string]$PythonExe = "python",
    [int]$DatasetReadyTimeoutSeconds = 600
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$root = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $root "config\v3_e1_execution.json"
$datasetDir = Join-Path $root "kaggle\dataset-v3-e1"
$kernelDir = Join-Path $root "kaggle\kernel-v3-e1"
$tokenPath = Join-Path $env:USERPROFILE ".kaggle\access_token"

$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if ($config.approved_status -ne "APPROVED_FOR_MANUAL_KAGGLE_RUN") {
    throw "V3 E1 approved_status drifted."
}
if ($config.kaggle.machine_shape -ne "NvidiaTeslaT4") {
    throw "V3 E1 refuses non-T4 Kaggle machine_shape"
}
if ($config.kaggle.internet_enabled -ne $true) {
    throw "V3 E1 requires internet enabled for Hugging Face downloads"
}
if ($config.accelerator.required_compute_capability -ne "sm_75") {
    throw "V3 E1 requires T4 compute capability sm_75"
}
$acceptedNames = @($config.accelerator.accepted_device_names)
$expectedNames = @("Tesla T4", "NVIDIA Tesla T4", "NVIDIA T4")
if (($acceptedNames -join "|") -ne ($expectedNames -join "|")) {
    throw "V3 E1 accepted device names drifted."
}
if ($config.status -eq "DRAFT" -and
    $config.approved_for_kaggle_execution -eq $false) {
    throw ("V3 E1 is still DRAFT. Set approved_for_kaggle_execution=true " +
        "only after user review before publishing or running Kaggle.")
}
if ($config.status -ne "APPROVED_FOR_MANUAL_KAGGLE_RUN" -or
    $config.approved_for_kaggle_execution -ne $true) {
    throw "V3 E1 status and approval flag disagree."
}

if (-not (Test-Path $tokenPath)) {
    throw "Kaggle access token file is missing at %USERPROFILE%\.kaggle\access_token"
}
$token = (Get-Content -LiteralPath $tokenPath -Raw).Trim()
if (-not $token.StartsWith("KGAT_")) {
    throw "Kaggle access token must start with KGAT_"
}
Remove-Variable -Name token -Force

if (-not (Get-Command $KaggleExe -ErrorAction SilentlyContinue)) {
    throw "Kaggle CLI not found. Activate the project environment first."
}

$configView = (& $KaggleExe config view) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw "Kaggle CLI authentication failed"
}
$usernameMatch = [regex]::Match($configView, "username:\s*(\S+)")
$username = $usernameMatch.Groups[1].Value
if (-not $usernameMatch.Success -or $username -eq "None") {
    throw "Kaggle CLI did not resolve a username; token may be invalid."
}
if ($username -ne "eniorubens") {
    throw "V3 E1 metadata is pinned to eniorubens; got Kaggle user $username"
}
Write-Output "Kaggle user: $username"

function Publish-Dataset {
    param([string]$Folder, [string]$Message)
    Write-Output "Publishing dataset from $Folder"
    $createOutput = (& $KaggleExe datasets create -p $Folder) -join "`n"
    Write-Output $createOutput
    if ($LASTEXITCODE -ne 0 -or $createOutput -match "already in use|error") {
        Write-Output "Create failed or dataset exists; publishing a version."
        & $KaggleExe datasets version -p $Folder -m $Message
        if ($LASTEXITCODE -ne 0) {
            throw "Dataset publish failed for $Folder"
        }
    }
}

function Wait-DatasetReady {
    param([string]$Slug)
    $deadline = (Get-Date).AddSeconds($DatasetReadyTimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $status = (& $KaggleExe datasets status $Slug) -join " "
        if ($LASTEXITCODE -eq 0 -and $status -match "ready") {
            Write-Output "Dataset ${Slug}: ready"
            return
        }
        Write-Output "Dataset ${Slug}: waiting (last status: $status)"
        Start-Sleep -Seconds 15
    }
    throw "Dataset $Slug was not ready within $DatasetReadyTimeoutSeconds s"
}

Push-Location $root
try {
    & $PythonExe scripts\build_kaggle_v3_e1_bundle.py
    if ($LASTEXITCODE -ne 0) {
        throw "V3 E1 bundle build failed"
    }
} finally {
    Pop-Location
}

$bundleZip = Join-Path $datasetDir "cci-v3-e1-bundle.zip"
if (-not (Test-Path $bundleZip)) {
    throw "V3 E1 bundle zip missing after build"
}
Publish-Dataset -Folder $datasetDir -Message "V3 E1 smoke bundle refresh"
Wait-DatasetReady -Slug "eniorubens/cci-v3-e1-bundle"

Write-Output "Pushing V3 E1 kernel with Kaggle T4 machine_shape."
& $KaggleExe kernels push -p $kernelDir
if ($LASTEXITCODE -ne 0) {
    throw "V3 E1 kernel push failed"
}

$kernelMeta = Get-Content (Join-Path $kernelDir "kernel-metadata.json") -Raw |
    ConvertFrom-Json
$kernelSlug = $kernelMeta.id
Write-Output "Kernel queued: $kernelSlug"
& $KaggleExe kernels status $kernelSlug
Write-Output ("Monitor with: kaggle kernels status $kernelSlug ; " +
    "fetch results with scripts\fetch_kaggle_v3_e1_results.ps1")
