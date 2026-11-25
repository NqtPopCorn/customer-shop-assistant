[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "  CUSTOMER SHOP ASSISTANT — RESEARCH EVALUATION SUITE" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[1] Tang 1 — Policy & KG Benchmark + Ablation (Nhanh, khong ton GPU)"
Write-Host "[2] Full Research E1 — Tang 1 + Tang 2 Multi-turn E2E (S1-S10 + 12 Test)"
Write-Host "[3] Dev Calibration — Hieu chinh & dong bang nguong frozen_thresholds.json"
Write-Host "[4] Tang 3 — Functional Regression (eval/questions.json v2)"
Write-Host "[5] Dry-Run nhanh (Mock LLM — kiem tra pipeline khong ton GPU)"
Write-Host "[6] Thoat"
Write-Host ""

$opt = Read-Host "Chon tuy chon [1-6]"

switch ($opt) {
    "1" {
        Write-Host "Dang chay Tang 1 Policy & KG Benchmark + Ablation..." -ForegroundColor Yellow
        uv run python -m customer_shop_assistant.eval.experiments --experiment e1 --tier1-only --ablation
    }
    "2" {
        Write-Host "Dang chay Full Research E1 (Tang 1 + Tang 2 E2E)..." -ForegroundColor Yellow
        uv run python -m customer_shop_assistant.eval.experiments --experiment e1 --ablation --e2e-limit 12
    }
    "3" {
        Write-Host "Dang hieu chinh nguong tren tap Dev (28 ca)..." -ForegroundColor Yellow
        uv run python -m customer_shop_assistant.eval.experiments --experiment e1 --tier1-only --calibrate --ablation
    }
    "4" {
        Write-Host "Dang chay Functional Regression (eval/questions.json v2)..." -ForegroundColor Yellow
        uv run python -m customer_shop_assistant.eval.runner
    }
    "5" {
        Write-Host "Dang chay Dry-Run..." -ForegroundColor Yellow
        uv run python -m customer_shop_assistant.eval.runner --dry-run --limit 10
    }
    default {
        Write-Host "Da huy."
    }
}
