@echo off
chcp 65001 > nul
echo ====================================================================
echo   CUSTOMER SHOP ASSISTANT — BENCHMARK EVALUATION (M4)
echo ====================================================================
echo.
echo [1] Chạy Đánh giá nhanh (Dry-Run 10 câu - Không tốn GPU)
echo [2] Chạy Đánh giá đầy đủ (Bộ 60 câu Golden Set)
echo [3] Chạy Thí nghiệm E1 (So sánh 3 chế độ Clarification)
echo [4] Thoát
echo.
set /p opt="Chọn tùy chọn [1-4]: "

if "%opt%"=="1" (
    echo Đang chạy Dry-Run...
    uv run python -m customer_shop_assistant.eval.runner --dry-run --limit 10
) else if "%opt%"=="2" (
    echo Đang chạy kiểm thử 60 câu...
    uv run python -m customer_shop_assistant.eval.runner
) else if "%opt%"=="3" (
    echo Đang chạy Thí nghiệm E1...
    uv run python -m customer_shop_assistant.eval.experiments --experiment e1
) else (
    echo Đã hủy.
)

pause
