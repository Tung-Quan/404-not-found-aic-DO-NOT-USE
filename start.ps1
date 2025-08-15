# Enhanced Video Search System - Smart Auto-Install
# Modern PowerShell launcher for Windows

param(
    [switch]$NoWait,
    [switch]$Verbose
)

# Set console properties
$Host.UI.RawUI.WindowTitle = "Enhanced Video Search System - Smart Auto-Install"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Colors for better output
$ErrorColor = "Red"
$InfoColor = "Cyan"
$SuccessColor = "Green"
$WarningColor = "Yellow"

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Banner {
    Write-ColorOutput "================================================================" $InfoColor
    Write-ColorOutput "    Enhanced Video Search System - Smart Auto-Install" $InfoColor
    Write-ColorOutput "================================================================" $InfoColor
    Write-Host ""
}

function Test-PythonInstallation {
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "[INFO] Python found: $pythonVersion" $SuccessColor
            return $true
        }
    }
    catch {
        # Continue to error handling
    }
    
    Write-ColorOutput "[ERROR] Python not found. Please install Python 3.8+" $ErrorColor
    Write-ColorOutput "Download from: https://www.python.org/downloads/" $WarningColor
    return $false
}

function Test-ProjectDirectory {
    if (-not (Test-Path "start.py")) {
        Write-ColorOutput "[ERROR] start.py not found. Please run from project root directory." $ErrorColor
        return $false
    }
    return $true
}

function Invoke-VirtualEnvironment {
    $venvPaths = @(".venv\Scripts\Activate.ps1", "venv\Scripts\Activate.ps1")
    
    foreach ($venvPath in $venvPaths) {
        if (Test-Path $venvPath) {
            Write-ColorOutput "[INFO] Activating virtual environment: $venvPath" $InfoColor
            try {
                & $venvPath
                Write-ColorOutput "[SUCCESS] Virtual environment activated" $SuccessColor
                return $true
            }
            catch {
                Write-ColorOutput "[WARNING] Failed to activate virtual environment: $_" $WarningColor
            }
        }
    }
    
    Write-ColorOutput "[INFO] No virtual environment found, using system Python" $InfoColor
    return $false
}

function Start-SmartLauncher {
    # Set environment variables
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
    
    Write-ColorOutput "[INFO] Starting Smart Auto-Install Launcher..." $InfoColor
    Write-Host ""
    
    try {
        python start.py
        $exitCode = $LASTEXITCODE
        
        if ($exitCode -eq 0) {
            Write-ColorOutput "[SUCCESS] System exited normally." $SuccessColor
        }
        else {
            Write-ColorOutput "[ERROR] System exited with code: $exitCode" $ErrorColor
            Write-ColorOutput "Check the error messages above for details." $WarningColor
        }
        
        return $exitCode
    }
    catch {
        Write-ColorOutput "[ERROR] Failed to start the system: $_" $ErrorColor
        return 1
    }
}

# Main execution
Write-Banner

# Pre-flight checks
if (-not (Test-PythonInstallation)) {
    if (-not $NoWait) { Read-Host "Press Enter to exit" }
    exit 1
}

if (-not (Test-ProjectDirectory)) {
    if (-not $NoWait) { Read-Host "Press Enter to exit" }
    exit 1
}

# Setup environment
Invoke-VirtualEnvironment

# Launch the system
$exitCode = Start-SmartLauncher

# Wait for user input unless suppressed
if (-not $NoWait) {
    Write-Host ""
    Read-Host "Press Enter to exit"
}

exit $exitCode
