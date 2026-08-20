param(
    [string]$PythonExe = "python",
    [ValidateSet("smoke", "full")]
    [string]$Mode = "full",
    [double]$MinimumAvailableGB = 0.75,
    [int]$MaximumLowMemorySamples = 3,
    [int]$SampleSeconds = 5
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$monitorPath = Join-Path $root "temp\v2\v2_resource_monitor.json"
$monitorDirectory = Split-Path -Parent $monitorPath
New-Item -ItemType Directory -Force -Path $monitorDirectory | Out-Null

$code = @"
import json; from consumer_complaint_intelligence.v2_benchmark import run_v2_benchmark; result = run_v2_benchmark("$Mode"); print(json.dumps(dict(status=result["status"], selected=result["selected"], candidate_count=len(result["candidates"]), eligible_count=result["hard_negative"]["margin_eligible_count"])))
"@.Trim()

$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $PythonExe
$escapedCode = $code.Replace('"', '\"')
$startInfo.Arguments = "-c `"$escapedCode`""
$startInfo.WorkingDirectory = $root
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$startInfo.EnvironmentVariables["PYTHONPATH"] = "src"

$process = New-Object System.Diagnostics.Process
$process.StartInfo = $startInfo
$startedAt = [DateTime]::UtcNow
$peakRssBytes = 0L
$minimumAvailableBytes = [long]::MaxValue
$samples = 0
$lowMemorySamples = 0
$status = "RUNNING"
$stdout = ""
$stderr = ""

try {
    if (-not $process.Start()) {
        throw "Unable to start V2 benchmark process."
    }
    while (-not $process.HasExited) {
        $process.Refresh()
        $peakRssBytes = [Math]::Max($peakRssBytes, $process.WorkingSet64)
        $os = Get-CimInstance Win32_OperatingSystem
        $availableBytes = [long]($os.FreePhysicalMemory * 1KB)
        $minimumAvailableBytes = [Math]::Min(
            $minimumAvailableBytes,
            $availableBytes
        )
        $samples += 1
        if ($availableBytes -lt ($MinimumAvailableGB * 1GB)) {
            $lowMemorySamples += 1
        } else {
            $lowMemorySamples = 0
        }
        if ($lowMemorySamples -ge $MaximumLowMemorySamples) {
            $status = "ABORTED_LOW_MEMORY"
            $process.Kill()
            break
        }
        Start-Sleep -Seconds $SampleSeconds
    }
    $process.WaitForExit()
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    if ($status -eq "RUNNING") {
        $status = if ($process.ExitCode -eq 0) { "COMPLETE" } else { "ERROR" }
    }
} finally {
    $endedAt = [DateTime]::UtcNow
    $payload = [ordered]@{
        status = $status
        started_at_utc = $startedAt.ToString("o")
        ended_at_utc = $endedAt.ToString("o")
        elapsed_seconds = ($endedAt - $startedAt).TotalSeconds
        exit_code = if ($process.HasExited) { $process.ExitCode } else { $null }
        peak_process_rss_gb = $peakRssBytes / 1GB
        min_system_available_gb = $minimumAvailableBytes / 1GB
        samples = $samples
        minimum_available_gb_guard = $MinimumAvailableGB
        stdout = $stdout
        stderr = $stderr
    }
    $payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $monitorPath
    $process.Dispose()
}

if ($status -ne "COMPLETE") {
    throw "V2 benchmark ended with status $status. See $monitorPath"
}

Write-Output $stdout
