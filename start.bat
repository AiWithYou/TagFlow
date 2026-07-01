@echo off
setlocal
REM Always run from this script's directory so double-click launch works reliably.
cd /d "%~dp0"

set "VENV_PYTHON=venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    call :find_python || goto :error
    %PYTHON_CMD% -m venv venv || goto :error
)

"%VENV_PYTHON%" -m pip install --upgrade pip || goto :error
"%VENV_PYTHON%" -m pip install -r requirements.txt || goto :error
"%VENV_PYTHON%" TagFlow.py
set "EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %EXIT_CODE%

:find_python
where python >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    exit /b 0
)
where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
    exit /b 0
)
echo Python 3 が見つかりません。Python をインストールしてから再実行してください。
exit /b 1

:error
echo.
echo 起動準備に失敗しました。
pause
exit /b 1
