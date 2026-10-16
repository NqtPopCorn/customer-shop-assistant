[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "  CUSTOMER SHOP ASSISTANT — BENCHMARK EVALUATION (M4)" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[1] Chạy Đánh giá nhanh (Dry-Run 10 câu - Không tốn GPU)"
Write-Host "[2] Chạy Đánh giá đầy đủ (Bộ 60 câu Golden Set)"
Write-Host "[3] Chạy Thí nghiệm E1 (So sánh 3 chế độ Clarification)"
Write-Host "[4] Thoát"
Write-Host ""

$opt = Read-Host "Chọn tùy chọn [1-4]"

switch ($opt) {
    "1" {
        Write-Host "Đang chạy Dry-Run..." -ForegroundColor Yellow
        uv run python -m customer_shop_assistant.eval.runner --dry-run --limit 10
    }
    "2" {
        Write-Host "Đang chạy kiểm thử 60 câu..." -ForegroundColor Yellow
        uv run python -m customer_shop_assistant.eval.runner
    }
    "3" {
        Write-Host "Đang chạy Thí nghiệm E1..." -ForegroundColor Yellow
        uv run python -m customer_shop_assistant.eval.experiments --experiment e1
    }
    default {
        Write-Host "Đã hủy."
    }
}
