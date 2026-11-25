"""FastAPI routes and UI endpoint for reviewing the Tier-2 human judge CSV."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.schemas.judge import (
    JudgeConfigResponse,
    JudgeScoreUpdate,
    JudgeSheetResponse,
    JudgeSummaryResponse,
    JudgeUpdateResponse,
)
from customer_shop_assistant.services.judge_sheet_service import (
    RowConflictError,
    compute_judge_summary,
    load_judge_rows,
    update_judge_row,
)

ui_router = APIRouter(tags=["Judge UI"])
router = APIRouter(prefix="/api/v1/judge", tags=["Judge"])

_UI_HTML_PATH = Path(__file__).resolve().parent.parent / "ui" / "judge_sheet.html"


def get_judge_sheet_path(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Path:
    """Resolve the path to eval/e1_tier2_judge_sheet.csv."""
    return settings.EVAL_DIR / settings.JUDGE_SHEET_FILENAME


@ui_router.get("/judge", response_class=HTMLResponse)
async def judge_ui_page() -> HTMLResponse:
    """Serve the offline single-page Human Judge UI."""
    if not _UI_HTML_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Judge UI HTML asset not found.",
        )
    return HTMLResponse(content=_UI_HTML_PATH.read_text(encoding="utf-8"))


@router.get("/config", response_model=JudgeConfigResponse)
async def get_judge_config(
    settings: Annotated[Settings, Depends(get_settings)],
) -> JudgeConfigResponse:
    """Return optional Google Apps Script bridge defaults from settings."""
    return JudgeConfigResponse(
        default_bridge_url=settings.JUDGE_BRIDGE_URL,
        default_bridge_token=settings.JUDGE_BRIDGE_TOKEN,
    )


@router.get("/rows", response_model=JudgeSheetResponse)
async def get_judge_rows(
    csv_path: Annotated[Path, Depends(get_judge_sheet_path)],
) -> JudgeSheetResponse:
    """Return all rows and progress statistics from the Tier-2 judge CSV."""
    try:
        rows = load_judge_rows(csv_path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

    return JudgeSheetResponse(
        csv_path=str(csv_path),
        total=len(rows),
        scored=sum(1 for r in rows if r.status == "scored"),
        partial=sum(1 for r in rows if r.status == "partial"),
        unscored=sum(1 for r in rows if r.status == "unscored"),
        rows=rows,
    )


@router.patch("/rows/{row_index}", response_model=JudgeUpdateResponse)
async def patch_judge_row(
    row_index: int,
    payload: JudgeScoreUpdate,
    csv_path: Annotated[Path, Depends(get_judge_sheet_path)],
) -> JudgeUpdateResponse | JSONResponse:
    """Update human rubric scores and notes for a single row in the CSV."""
    try:
        updated = update_judge_row(csv_path, row_index, payload)
        all_rows = load_judge_rows(csv_path)
    except RowConflictError as exc:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "detail": "ROW_CONFLICT",
                "conflict": True,
                "server_row": exc.server_row.model_dump(),
            },
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except IndexError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

    return JudgeUpdateResponse(
        row=updated,
        total=len(all_rows),
        scored=sum(1 for r in all_rows if r.status == "scored"),
        partial=sum(1 for r in all_rows if r.status == "partial"),
        unscored=sum(1 for r in all_rows if r.status == "unscored"),
    )


@router.get("/summary", response_model=JudgeSummaryResponse)
async def get_judge_summary(
    csv_path: Annotated[Path, Depends(get_judge_sheet_path)],
    suite: str = "all",
) -> JudgeSummaryResponse:
    """Aggregate human evaluation scores across ClarificationMode strategies."""
    try:
        rows = load_judge_rows(csv_path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

    return compute_judge_summary(rows, suite=suite)


@router.get("/export")
async def export_judge_csv(
    csv_path: Annotated[Path, Depends(get_judge_sheet_path)],
) -> FileResponse:
    """Download the current Tier-2 judge CSV file."""
    if not csv_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Judge sheet CSV not found: {csv_path}",
        )
    return FileResponse(
        path=csv_path,
        media_type="text/csv; charset=utf-8",
        filename="e1_tier2_judge_sheet.csv",
    )
