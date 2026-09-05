# Launch scripts for the reviewer-experiment cell scheduler (Windows PowerShell).
#
#   .\run_scheduler.ps1 status        show pending / completed cells
#   .\run_scheduler.ps1 dryrun        print the plan without running
#   .\run_scheduler.ps1 combined      recommended: one command for everything
#   .\run_scheduler.ps1 cpu           CPU-class cells only (troubleshooting)
#   .\run_scheduler.ps1 gpu           CUDA-heavy cells only (troubleshooting)
#   .\run_scheduler.ps1 aggregate     rebuild summaries from existing cells
#   .\run_scheduler.ps1 bench         benchmark worker counts
#   .\run_scheduler.ps1 gate          reproducibility gate

param(
    [Parameter(Position = 0)]
    [ValidateSet("status", "dryrun", "combined", "cpu", "gpu", "aggregate", "bench", "gate")]
    [string]$Action = "status",

    [int]$CpuWorkers = 3,
    [int]$GpuWorkers = 2
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Py)) {
    throw "Interpreter not found at $Py"
}

$AllAnalyses = "selection,resampling,gbm,hsj_budget"
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

switch ($Action) {
    "status" {
        & $Py Code\run_cells.py --list-pending --only $AllAnalyses
    }
    "dryrun" {
        & $Py Code\run_cells.py --dry-run --only $AllAnalyses `
            --cpu-workers $CpuWorkers --gpu-workers $GpuWorkers
    }
    "combined" {
        $log = Join-Path $LogDir "scheduler_combined.log"
        Write-Host "Running combined scheduler -> $log"
        & $Py Code\run_cells.py --resume --only $AllAnalyses `
            --cpu-workers $CpuWorkers --gpu-workers $GpuWorkers 2>&1 |
            Tee-Object -FilePath $log
    }
    "cpu" {
        $log = Join-Path $LogDir "scheduler_cpu.log"
        & $Py Code\run_cells.py --resume --only "selection,resampling,hsj_budget" `
            --cpu-workers $CpuWorkers --gpu-workers 0 2>&1 | Tee-Object -FilePath $log
    }
    "gpu" {
        $log = Join-Path $LogDir "scheduler_gpu.log"
        & $Py Code\run_cells.py --resume --only "surrogate" `
            --cpu-workers 1 --gpu-workers $GpuWorkers 2>&1 | Tee-Object -FilePath $log
    }
    "aggregate" {
        & $Py Code\run_cells.py --aggregate-only --only $AllAnalyses
    }
    "bench" {
        & $Py Code\bench_scheduler.py
    }
    "gate" {
        & $Py Code\bench_scheduler.py --gate
    }
}
