# PowerShell 用スタートアップスクリプト
if (!(Test-Path -Path "venv")) {
    python -m venv venv
}
. .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python TagFlow.py
Pause
