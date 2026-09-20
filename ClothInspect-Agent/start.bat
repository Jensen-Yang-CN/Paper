@echo off
rem ============================================================
rem  ClothInspect-Agent V1.0 - Launcher
rem ------------------------------------------------------------
rem  Double-click this file to start the system.
rem  It just runs start.ps1 with Windows PowerShell.
rem
rem  IMPORTANT: keep this file PURE ASCII.
rem  cmd.exe parses .bat files using the console code page, so any
rem  non-ASCII byte here can shift the parser and break the script.
rem  All Chinese messages live in start.ps1 instead.
rem ============================================================

chcp 65001 >nul 2>&1
setlocal

where powershell >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Windows PowerShell not found.
  echo         Please run start.ps1 manually instead.
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*

if errorlevel 1 (
  echo.
  echo [ERROR] Startup failed. Please read the messages above.
  pause
  exit /b 1
)

endlocal
