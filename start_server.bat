@echo off
title AI Video Search API Server
echo ======================================================
echo        AI Video Search API Server - NOTICE
echo ======================================================
echo.
echo ⚠️  NOTICE: This system now has TWO server versions!
echo.
echo 🚀 RECOMMENDED: start_server_simple.bat
echo    - Memory optimized, fast startup, stable
echo    - Port 8001, low resource usage
echo.
echo 🔥 ADVANCED: start_server_advanced.bat  
echo    - Full features, higher memory usage
echo    - Port 8000, may have memory issues
echo.
echo.
set /p choice="Which version do you want? (1=Simple, 2=Advanced, 3=Exit): "

if "%choice%"=="1" (
    echo Starting Simple Server...
    call start_server_simple.bat
) else if "%choice%"=="2" (
    echo Starting Advanced Server...
    call start_server_advanced.bat
) else (
    echo Goodbye!
    pause
    exit /b 0
)

echo.
echo Server selection complete.
pause
echo ================================================
echo Server stopped.
echo ================================================
pause
