# Publish the V2 bundle/cache datasets, then push the D2 GPU Kaggle kernel.
# Requires a Kaggle access token at %USERPROFILE%\.kaggle\access_token
# (or any auth method accepted by the Kaggle CLI 2.x).
param(
    [string]$KaggleExe = "kaggle",
    [int]$DatasetReadyTimeoutSeconds = 600,
    [switch]$SkipCacheUpload
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$kaggleDir = Join-Path $root "kaggle"
$kernelDir = Join-Path $kaggleDir "kernel-d2"

if (-not (Get-Command $KaggleExe -ErrorAction SilentlyContinue)) {
    throw "Kaggle CLI not found. Activate the project environment first."
}

$configView = (& $KaggleExe config view) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw ("Kaggle CLI authentication failed. Save an access token at " +
        "$env:USERPROFILE\.kaggle\access_token first.")
}
$usernameMatch = [regex]::Match($configView, "username:\s*(\S+)")
$username = $usernameMatch.Groups[1].Value
if (-not $usernameMatch.Success -or $username -eq "None") {
    throw "Kaggle CLI did not resolve a username; token may be invalid."
}
Write-Output "Kaggle user: $username"

$metaFiles = @(
    (Join-Path $kernelDir "kernel-metadata.json"),
    (Join-Path $kaggleDir "dataset-bundle\dataset-metadata.json"),
    (Join-Path $kaggleDir "dataset-cache\dataset-metadata.json")
)
foreach ($file in $metaFiles) {
    $content = Get-Content $file -Raw
    if ($content.Contains("INSERT_KAGGLE_USERNAME")) {
        $content = $content.Replace("INSERT_KAGGLE_USERNAME", $username)
        [System.IO.File]::WriteAllText($file, $content)
        Write-Output "Patched username into $file"
    }
}

function Publish-Dataset {
    param([string]$Folder, [string]$Message)
    Write-Output "Publishing dataset from $Folder"
    $createOutput = (& $KaggleExe datasets create -p $Folder) -join "`n"
    Write-Output $createOutput
    # The CLI can exit 0 while still refusing creation (e.g. title in use),
    # so detect refusal in the output text as well as the exit code.
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

$bundleZip = Join-Path $kaggleDir "dataset-bundle\cci-v2-bundle.zip"
if (-not (Test-Path $bundleZip)) {
    throw "Bundle zip missing. Run scripts\build_kaggle_bundle.py first."
}
Publish-Dataset -Folder (Join-Path $kaggleDir "dataset-bundle") `
    -Message "V2-D2 bundle refresh"
Wait-DatasetReady -Slug "$username/cci-v2-bundle"

if (-not $SkipCacheUpload) {
    $cacheSource = Join-Path $root "temp\s3\scientific.parquet"
    $cacheStage = Join-Path $kaggleDir "dataset-cache\scientific.parquet"
    if (-not (Test-Path $cacheSource)) {
        throw "Scientific cache missing at $cacheSource"
    }
    Copy-Item -Path $cacheSource -Destination $cacheStage -Force
    try {
        Publish-Dataset -Folder (Join-Path $kaggleDir "dataset-cache") `
            -Message "Scientific cache refresh"
        Wait-DatasetReady -Slug "$username/cci-scientific-cache"
    } finally {
        Remove-Item -Path $cacheStage -Force -Confirm:$false
    }
}

Write-Output "Pushing kernel (this queues a full Save & Run All execution)."
& $KaggleExe kernels push -p $kernelDir
if ($LASTEXITCODE -ne 0) {
    throw "Kernel push failed"
}

# Read the kernel slug from the metadata so the two cannot drift apart.
$kernelMeta = Get-Content (Join-Path $kernelDir "kernel-metadata.json") -Raw |
    ConvertFrom-Json
$kernelSlug = $kernelMeta.id
Write-Output "Kernel queued: $kernelSlug"
& $KaggleExe kernels status $kernelSlug
Write-Output ("Monitor with: kaggle kernels status $kernelSlug ; " +
    "fetch results with scripts\fetch_kaggle_v2_d2_results.ps1")
