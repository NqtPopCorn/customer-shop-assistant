@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
echo Dang khoi dong Customer Shop Assistant CLI...
uv run python cli.py
