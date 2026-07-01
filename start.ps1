# PowerShell 用スタートアップスクリプト
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

function Invoke-SystemPython {
    param([string[]]$PythonArgs)

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        & $python.Source @PythonArgs
        return $LASTEXITCODE
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        & $py.Source -3 @PythonArgs
        return $LASTEXITCODE
    }

    throw "Python 3 が見つかりません。Python をインストールしてから再実行してください。"
}

try {
    if (!(Test-Path -LiteralPath $venvPython)) {
        $exitCode = Invoke-SystemPython -PythonArgs @("-m", "venv", "venv")
        if ($exitCode -ne 0) {
            throw "仮想環境の作成に失敗しました。"
        }
    }

    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip の更新に失敗しました。" }

    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "依存パッケージのインストールに失敗しました。" }

    & $venvPython TagFlow.py
    $exitCode = $LASTEXITCODE
    Pause
    exit $exitCode
}
catch {
    Write-Host ""
    Write-Error $_
    Pause
    exit 1
}
