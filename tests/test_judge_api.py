"""Tests for Tier-2 human judge sheet service, FastAPI endpoints, and UI."""

import csv
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from customer_shop_assistant.api.judge import get_judge_sheet_path
from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.main import app
from customer_shop_assistant.services.judge_sheet_service import load_judge_rows

SAMPLE_FIELDNAMES: list[str] = [
    "source",
    "suite",
    "scenario_id",
    "task_type",
    "mode",
    "turns_to_completion",
    "question",
    "final_reply",
    "clarification_asked",
    "asked_slots",
    "decision_source",
    "reason_codes",
    "tools_used",
    "applied_rule_ids",
    "products_summary",
    "outfits_summary",
    "grounding_ok",
    "recommendation_relevance_1_5",
    "unnecessary_question_0_1",
    "missed_clarification_0_1",
    "task_completion_0_1",
    "explanation_faithfulness_1_5",
    "judge_notes",
]


def _write_sample_csv(path: Path) -> None:
    rows = [
        {
            "source": "tier2_e2e_llm",
            "suite": "contrastive",
            "scenario_id": "S1",
            "task_type": "b_over_asks_c_proceeds",
            "mode": "IMMEDIATE_ANSWER",
            "turns_to_completion": "1",
            "question": "Phối áo sơ mi trắng với quần gì đẹp?",
            "final_reply": "**Bộ 1**: Áo [486] + Quần [764]",
            "clarification_asked": "False",
            "asked_slots": "",
            "decision_source": "mode_a_immediate",
            "reason_codes": "MODE_A_STRICT_ZERO",
            "tools_used": "suggest_outfit",
            "applied_rule_ids": "PW_SMI_QAU",
            "products_summary": "[486] Áo sơ mi (840000d)",
            "outfits_summary": "[486] Áo + [764] Quần",
            "grounding_ok": "True",
            "recommendation_relevance_1_5": "",
            "unnecessary_question_0_1": "",
            "missed_clarification_0_1": "",
            "task_completion_0_1": "",
            "explanation_faithfulness_1_5": "",
            "judge_notes": "",
        },
        {
            "source": "tier2_e2e_llm",
            "suite": "stratified",
            "scenario_id": "E1_TEST_01",
            "task_type": "outfit_pairing",
            "mode": "CONDITIONAL_CLARIFICATION",
            "turns_to_completion": "2",
            "question": "Gợi ý đồ đi tiệc tối màu tối?",
            "final_reply": "Bạn thích váy hay blazer?",
            "clarification_asked": "True",
            "asked_slots": "category",
            "decision_source": "mode_c_counterfactual",
            "reason_codes": "HIGH_AMBIGUITY",
            "tools_used": "suggest_outfit",
            "applied_rule_ids": "PW_BLZ_VAY",
            "products_summary": "[102] Blazer Đen (990000d)",
            "outfits_summary": "[102] Blazer + [205] Quần Tây",
            "grounding_ok": "True",
            "recommendation_relevance_1_5": "5",
            "unnecessary_question_0_1": "0",
            "missed_clarification_0_1": "0",
            "task_completion_0_1": "1",
            "explanation_faithfulness_1_5": "5",
            "judge_notes": "Hỏi đúng trọng tâm",
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def judge_client(tmp_path: Path) -> Iterator[tuple[TestClient, Path]]:
    csv_path = tmp_path / "judge_sheet.csv"
    _write_sample_csv(csv_path)
    app.dependency_overrides[get_judge_sheet_path] = lambda: csv_path
    try:
        yield TestClient(app), csv_path
    finally:
        app.dependency_overrides.pop(get_judge_sheet_path, None)


def test_judge_ui_html_loads(judge_client: tuple[TestClient, Path]) -> None:
    client, _ = judge_client
    response = client.get("/judge")
    assert response.status_code == 200
    assert "Human Judge" in response.text
    assert "/api/v1/judge/rows" in response.text


def test_get_and_patch_judge_rows_persists_atomically(
    judge_client: tuple[TestClient, Path],
) -> None:
    client, csv_path = judge_client

    res = client.get("/api/v1/judge/rows")
    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 2
    assert body["scored"] == 1
    assert body["partial"] == 0
    assert body["unscored"] == 1
    assert body["rows"][0]["status"] == "unscored"
    assert body["rows"][1]["status"] == "scored"

    # Validation error for out-of-range relevance
    bad = client.patch(
        "/api/v1/judge/rows/0",
        json={"recommendation_relevance_1_5": 6},
    )
    assert bad.status_code == 422

    # Partial update
    partial = client.patch(
        "/api/v1/judge/rows/0",
        json={
            "recommendation_relevance_1_5": 4,
            "judge_notes": "Gợi ý tốt nhưng thiếu lọc giới tính",
        },
    )
    assert partial.status_code == 200
    assert partial.json()["row"]["status"] == "partial"
    assert partial.json()["partial"] == 1

    # Complete update
    full = client.patch(
        "/api/v1/judge/rows/0",
        json={
            "recommendation_relevance_1_5": 4,
            "unnecessary_question_0_1": 0,
            "missed_clarification_0_1": 1,
            "task_completion_0_1": 1,
            "explanation_faithfulness_1_5": 5,
            "judge_notes": "Gợi ý tốt nhưng nên hỏi giới tính",
        },
    )
    assert full.status_code == 200
    assert full.json()["row"]["status"] == "scored"
    assert full.json()["scored"] == 2
    assert full.json()["unscored"] == 0

    # Verify UTF-8-SIG and untouched read-only columns in CSV
    raw_bytes = csv_path.read_bytes()
    assert raw_bytes.startswith(b"\xef\xbb\xbf")
    reloaded = load_judge_rows(csv_path)
    assert reloaded[0].question == "Phối áo sơ mi trắng với quần gì đẹp?"
    assert reloaded[0].judge_notes == "Gợi ý tốt nhưng nên hỏi giới tính"
    assert reloaded[0].missed_clarification_0_1 == 1

    # Clearing scores back to None restores unscored
    cleared = client.patch("/api/v1/judge/rows/0", json={"judge_notes": ""})
    assert cleared.status_code == 200
    assert cleared.json()["row"]["status"] == "unscored"


def test_patch_out_of_bounds_returns_404(
    judge_client: tuple[TestClient, Path],
) -> None:
    client, _ = judge_client
    response = client.patch(
        "/api/v1/judge/rows/999",
        json={"task_completion_0_1": 1},
    )
    assert response.status_code == 404


def test_summary_and_export_endpoints(
    judge_client: tuple[TestClient, Path],
) -> None:
    client, _ = judge_client
    summary_all = client.get("/api/v1/judge/summary")
    assert summary_all.status_code == 200
    payload = summary_all.json()
    assert payload["total_rows"] == 2
    assert payload["scored_rows"] == 1
    assert "| `CONDITIONAL_CLARIFICATION` |" in payload["markdown_table"]

    summary_strat = client.get("/api/v1/judge/summary?suite=stratified")
    assert summary_strat.status_code == 200
    strat_body = summary_strat.json()
    assert strat_body["total_rows"] == 1
    assert len(strat_body["by_mode"]) == 1
    assert strat_body["by_mode"][0]["avg_relevance"] == 5.0
    assert strat_body["by_mode"][0]["task_completion_rate"] == 100.0

    export_res = client.get("/api/v1/judge/export")
    assert export_res.status_code == 200
    assert "text/csv" in export_res.headers["content-type"]
    assert "Phối áo sơ mi trắng" in export_res.text


def test_real_judge_csv_loads_read_only() -> None:
    real_csv = Path("eval/e1_tier2_judge_sheet.csv")
    if not real_csv.exists():
        pytest.skip("eval/e1_tier2_judge_sheet.csv not present")
    rows = load_judge_rows(real_csv)
    assert len(rows) == 72
    assert {r.mode for r in rows} == {
        "IMMEDIATE_ANSWER",
        "ALWAYS_CLARIFY",
        "CONDITIONAL_CLARIFICATION",
    }


def test_judge_config_endpoint(tmp_path: Path) -> None:
    test_settings = Settings(
        EVAL_DIR=tmp_path,
        JUDGE_BRIDGE_URL="https://script.google.com/macros/s/demo/exec",
        JUDGE_BRIDGE_TOKEN="secret-123",
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    try:
        client = TestClient(app)
        res = client.get("/api/v1/judge/config")
        assert res.status_code == 200
        body = res.json()
        assert (
            body["default_bridge_url"] == "https://script.google.com/macros/s/demo/exec"
        )
        assert body["default_bridge_token"] == "secret-123"
    finally:
        app.dependency_overrides.pop(get_settings, None)


def test_optimistic_locking_conflict_and_force_overwrite(
    judge_client: tuple[TestClient, Path],
) -> None:
    client, _ = judge_client

    first = client.patch(
        "/api/v1/judge/rows/0",
        json={
            "recommendation_relevance_1_5": 5,
            "judge_notes": "Alice note",
            "judge_name": "Alice",
            "expected_updated_at": "",
        },
    )
    assert first.status_code == 200
    first_row = first.json()["row"]
    assert first_row["judge_name"] == "Alice"
    assert first_row["updated_at"]

    conflict = client.patch(
        "/api/v1/judge/rows/0",
        json={
            "recommendation_relevance_1_5": 2,
            "judge_notes": "Bob stale note",
            "judge_name": "Bob",
            "expected_updated_at": "1999-01-01T00:00:00Z",
        },
    )
    assert conflict.status_code == 409
    conflict_body = conflict.json()
    assert conflict_body["conflict"] is True
    assert conflict_body["server_row"]["judge_name"] == "Alice"
    assert conflict_body["server_row"]["recommendation_relevance_1_5"] == 5

    forced = client.patch(
        "/api/v1/judge/rows/0",
        json={
            "recommendation_relevance_1_5": 3,
            "judge_notes": "Bob forced note",
            "judge_name": "Bob",
            "expected_updated_at": "1999-01-01T00:00:00Z",
            "force": True,
        },
    )
    assert forced.status_code == 200
    forced_row = forced.json()["row"]
    assert forced_row["judge_name"] == "Bob"
    assert forced_row["recommendation_relevance_1_5"] == 3
    assert forced_row["updated_at"] != first_row["updated_at"]


def test_apps_script_bridge_file_and_ui_markers(
    judge_client: tuple[TestClient, Path],
) -> None:
    bridge_path = Path("scripts/google_apps_script_judge_bridge.gs")
    assert bridge_path.exists()
    bridge_text = bridge_path.read_text(encoding="utf-8")
    for marker in (
        "function doGet",
        "function doPost",
        "LockService",
        "ROW_CONFLICT",
        "judge_name",
        "updated_at",
    ):
        assert marker in bridge_text

    client, _ = judge_client
    res = client.get("/judge")
    assert res.status_code == 200
    for marker in (
        "Google Sheet",
        "text/plain",
        "judge_name",
        "conflictModal",
        "unsavedModal",
        "btnCompareSaveAll",
    ):
        assert marker in res.text
