@echo off
REM このファイルをダブルクリックで実行してください
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install -r requirements.txt
python TagFlow.py
pause
