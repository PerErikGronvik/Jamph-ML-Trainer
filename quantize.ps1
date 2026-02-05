#!/usr/bin/env pwsh
# Quick start script for Jamph ML Trainer
# Usage: .\quantize.ps1 <model-id> [method]

param(
    [Parameter(Mandatory=$true)]
    [string]$ModelId,
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("Q4_K_M", "Q5_K_M", "Q8_0")]
    [string]$Method = "Q4_K_M",
    
    [switch]$SkipUpload,
    [switch]$NoMLflow,
    [switch]$UseDocker
)

# Check if mlflow.env exists
if (-not (Test-Path "mlflow.env")) {
    Write-Host "❌ mlflow.env not found!" -ForegroundColor Red
    Write-Host "Copy example.mlflow.env to mlflow.env and add your credentials" -ForegroundColor Yellow
    exit 1
}

Write-Host "🚀 Starting Jamph ML Trainer" -ForegroundColor Cyan
Write-Host "Model: $ModelId" -ForegroundColor White
Write-Host "Method: $Method" -ForegroundColor White

if ($UseDocker) {
    # Docker execution
    Write-Host "📦 Using Docker..." -ForegroundColor Cyan
    
    $dockerArgs = @(
        "-f", "docker-compose.quantize.yml",
        "run", "--rm",
        "quantizer",
        "process",
        "--model-id", $ModelId,
        "--method", $Method
    )
    
    if ($SkipUpload) {
        $dockerArgs += "--skip-upload"
    }
    
    if ($NoMLflow) {
        $dockerArgs += "--no-mlflow"
    }
    
    docker-compose @dockerArgs
    
} else {
    # Local UV execution
    Write-Host "🐍 Using local UV/Python..." -ForegroundColor Cyan
    
    # Check for UV
    if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
        Write-Host "❌ UV not found! Install with:" -ForegroundColor Red
        Write-Host "  curl -LsSf https://astral.sh/uv/install.sh | sh" -ForegroundColor Yellow
        exit 1
    }
    
    # Load environment variables from mlflow.env
    Get-Content mlflow.env | ForEach-Object {
        if ($_ -match '^([^#][^=]+)=(.+)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim().Trim('"').Trim("'")
            Set-Item -Path "env:$name" -Value $value
        }
    }
    
    # Build UV command
    $uvArgs = @(
        "run",
        "jamph-ml-trainer",
        "process",
        "--model-id", $ModelId,
        "--method", $Method
    )
    
    if ($SkipUpload) {
        $uvArgs += "--skip-upload"
    }
    
    if ($NoMLflow) {
        $uvArgs += "--no-mlflow"
    }
    
    uv @uvArgs
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Success! Check model training/Models/ for output" -ForegroundColor Green
} else {
    Write-Host "`n❌ Failed with exit code $LASTEXITCODE" -ForegroundColor Red
    Write-Host "Check logs/ directory for crash reports" -ForegroundColor Yellow
}
