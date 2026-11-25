@echo off
chcp 65001 > nul
echo ====================================================================
echo   CUSTOMER SHOP ASSISTANT — RESEARCH EVALUATION SUITE
echo ====================================================================
echo.
echo [1] Tang 1 — Policy ^& KG Benchmark + Ablation (Nhanh, khong ton GPU)
echo [2] Full Research E1 — Tang 1 + Tang 2 Multi-turn E2E (S1-S10 + 12 Test)
echo [3] Dev Calibration — Hieu chinh ^& dong bang nguong frozen_thresholds.json
echo [4] Tang 3 — Functional Regression (eval/questions.json v2)
echo [5] Dry-Run nhanh (Mock LLM — kiem tra pipeline khong ton GPU)
echo [6] Thoat
echo.
set /p opt="Chon tuy chon [1-6]: "

if "%opt%"=="1" (
    echo Dang chay Tang 1 Policy ^& KG Benchmark + Ablation...
    uv run python -m customer_shop_assistant.eval.experiments --experiment e1 --tier1-only --ablation
) else if "%opt%"=="2" (
    echo Dang chay Full Research E1...
    uv run python -m customer_shop_assistant.eval.experiments --experiment e1 --ablation --e2e-limit 12
) else if "%opt%"=="3" (
    echo Dang hieu chinh nguong tren tap Dev...
    uv run python -m customer_shop_assistant.eval.experiments --experiment e1 --tier1-only --calibrate --ablation
) else if "%opt%"=="4" (
    echo Dang chay Functional Regression...
    uv run python -m customer_shop_assistant.eval.runner
) else if "%opt%"=="5" (
    echo Dang chay Dry-Run...
    uv run python -m customer_shop_assistant.eval.runner --dry-run --limit 10
) else (
    echo Da huy.
)

pause
