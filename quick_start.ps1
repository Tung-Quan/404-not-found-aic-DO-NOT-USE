# =====================================================
# 🔍 Simplified AI Search System - Quick Start (PowerShell)
# =====================================================

param(
    [switch]$Status,
    [switch]$Manual,
    [switch]$Start,
    [switch]$Setup,
    [string]$Build
)

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "🔍 Simplified AI Search System - Quick Start" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found! Please install Python 3.8+" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if virtual environment exists
if (Test-Path ".venv") {
    Write-Host "🐍 Activating virtual environment..." -ForegroundColor Yellow
    & ".venv\Scripts\Activate.ps1"
} else {
    Write-Host "⚠️ No virtual environment found, using system Python" -ForegroundColor Yellow
}

# Handle different commands
if ($Status) {
    Write-Host "📊 Checking system status..." -ForegroundColor Blue
    python quick_start.py --status
}
elseif ($Manual) {
    Write-Host "📋 Showing manual commands..." -ForegroundColor Blue
    python quick_start.py --manual
}
elseif ($Start) {
    Write-Host "🚀 Starting web interface..." -ForegroundColor Green
    python quick_start.py --start
}
elseif ($Setup) {
    Write-Host "🔧 Setup only..." -ForegroundColor Blue
    python quick_start.py --setup-only
}
elseif ($Build) {
    if (-not (Test-Path $Build)) {
        Write-Host "❌ Directory not found: $Build" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "🔨 Building index from $Build..." -ForegroundColor Blue
    python quick_start.py --build-index $Build
}
else {
    # Default: Run complete workflow
    Write-Host "🚀 Running complete workflow..." -ForegroundColor Green
    python quick_start.py
}

# Show available commands
if (-not $Start) {
    Write-Host ""
    Write-Host "💡 Available commands:" -ForegroundColor Cyan
    Write-Host "   .\quick_start.ps1           - Complete workflow" -ForegroundColor White
    Write-Host "   .\quick_start.ps1 -Status   - Check system status" -ForegroundColor White
    Write-Host "   .\quick_start.ps1 -Manual   - Show manual commands" -ForegroundColor White
    Write-Host "   .\quick_start.ps1 -Start    - Start web interface only" -ForegroundColor White
    Write-Host "   .\quick_start.ps1 -Setup    - Setup dependencies only" -ForegroundColor White
    Write-Host "   .\quick_start.ps1 -Build 'frames' - Build index from directory" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to exit"
}
