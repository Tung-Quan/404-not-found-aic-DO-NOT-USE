# 🎯 Frame Search System - PowerShell Launcher
# ==============================================

Write-Host ""
Write-Host "🎯 FRAME SEARCH SYSTEM - PowerShell Launcher" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python detected: $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "❌ Python not found! Please install Python 3.8+" -ForegroundColor Red
    Write-Host "📦 Download from: https://python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Check if launcher exists
if (-not (Test-Path "launcher.py")) {
    Write-Host "❌ launcher.py not found!" -ForegroundColor Red
    Write-Host "📁 Make sure you're in the correct directory" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "🚀 Starting Frame Search Launcher..." -ForegroundColor Green
Write-Host "📍 Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Run the launcher
try {
    python launcher.py
} catch {
    Write-Host "❌ Error running launcher: $_" -ForegroundColor Red
} finally {
    Write-Host ""
    Write-Host "👋 Frame Search System stopped" -ForegroundColor Cyan
    Read-Host "Press Enter to exit"
}
