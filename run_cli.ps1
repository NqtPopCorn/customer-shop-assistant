# PowerShell launcher for Customer Shop Assistant CLI
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
Write-Host "Đang khởi động Customer Shop Assistant CLI..." -ForegroundColor Cyan
uv run python cli.py
