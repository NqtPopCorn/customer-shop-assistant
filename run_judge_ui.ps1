[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "  CUSTOMER SHOP ASSISTANT - TIER-2 JUDGE SHEET WEB UI" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Mo trinh duyet tai: http://127.0.0.1:8000/judge" -ForegroundColor Green
Write-Host "Nhan Ctrl+C de dung server." -ForegroundColor Yellow
Write-Host ""

uv run uvicorn customer_shop_assistant.main:app --reload --host 127.0.0.1 --port 8000
