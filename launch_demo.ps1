<#
.SYNOPSIS
    launch_demo.ps1 — Windows PowerShell Quickstart Launcher for ReAgent.

.DESCRIPTION
    Launches ReAgent Mission Control or fast CLI demonstrations with zero setup.
    Guarantees 100% offline execution, zero silent network calls, and zero secret leakage.

.EXAMPLE
    .\launch_demo.ps1
    Launches Mission Control at http://127.0.0.1:8080/

.EXAMPLE
    .\launch_demo.ps1 -Cli
    Runs the 4-second terminal judge demonstration.

.EXAMPLE
    .\launch_demo.ps1 -Check
    Runs core release-gate verification checks.
#>

[CmdletBinding()]
param(
    [switch]$Cli,
    [switch]$Check,
    [switch]$Open,
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"

# Verify python exists
$pythonPath = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $pythonPath) {
    Write-Host "[ERROR] Python was not found in your PATH." -ForegroundColor Red
    Write-Host "[INFO] Please install Python 3.10+ from python.org or the Microsoft Store." -ForegroundColor Yellow
    exit 1
}

$repoRoot = $PSScriptRoot
Set-Location $repoRoot

$arguments = @("launch_demo.py")
if ($Cli) { $arguments += "--cli" }
if ($Check) { $arguments += "--check" }
if ($Open) { $arguments += "--open" }
if ($Port -ne 8080) { $arguments += @("--port", $Port) }

& python @arguments
exit $LASTEXITCODE
