/**
 * Google Apps Script Bridge for Tier-2 Human Judge Sheet
 * ------------------------------------------------------
 * Setup:
 * 1. Import `eval/e1_tier2_judge_sheet.csv` into Google Sheets.
 * 2. Open Extensions -> Apps Script and paste this file.
 * 3. Either set Script Property `JUDGE_TOKEN` or edit `DEFAULT_TOKEN` below.
 * 4. Deploy -> New deployment -> Web app:
 *    - Execute as: Me
 *    - Who has access: Anyone
 * 5. Share the Web app URL + passcode token with your team.
 */

const DEFAULT_TOKEN = "change-me-team-secret";
const TARGET_SHEET_NAME = ""; // Leave blank to use the first sheet in the workbook.

const SCORE_COLUMNS = [
  "recommendation_relevance_1_5",
  "unnecessary_question_0_1",
  "missed_clarification_0_1",
  "task_completion_0_1",
  "explanation_faithfulness_1_5",
];

function _getActiveToken() {
  const propToken = PropertiesService.getScriptProperties().getProperty("JUDGE_TOKEN");
  return (propToken && propToken.trim()) ? propToken.trim() : DEFAULT_TOKEN;
}

function _isAuthorized(token) {
  const expected = _getActiveToken();
  if (!expected) return true;
  return String(token || "").trim() === expected;
}

function _getSheet() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  if (TARGET_SHEET_NAME) {
    const named = ss.getSheetByName(TARGET_SHEET_NAME);
    if (!named) {
      throw new Error("Sheet not found: " + TARGET_SHEET_NAME);
    }
    return named;
  }
  return ss.getSheets()[0];
}

function _ensureCollabColumns(sheet) {
  const lastCol = Math.max(1, sheet.getLastColumn());
  const headers = sheet.getRange(1, 1, 1, lastCol).getValues()[0].map((h) => String(h || "").trim());
  const required = ["judge_name", "updated_at"];
  let nextCol = lastCol;
  for (let i = 0; i < required.length; i++) {
    if (headers.indexOf(required[i]) === -1) {
      nextCol += 1;
      sheet.getRange(1, nextCol).setValue(required[i]);
      headers.push(required[i]);
    }
  }
  return headers;
}

function _parseOptionalInt(raw) {
  if (raw === null || raw === undefined || raw === "") return null;
  const num = Number(raw);
  if (!Number.isFinite(num)) return null;
  return Math.trunc(num);
}

function _parseBool(raw) {
  if (typeof raw === "boolean") return raw;
  return String(raw || "").trim().toLowerCase() === "true";
}

function _computeStatus(scores) {
  let filled = 0;
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] !== null) filled += 1;
  }
  if (filled === 0) return "unscored";
  if (filled === scores.length) return "scored";
  return "partial";
}

function _parseRow(rowIndex, headers, values) {
  const raw = {};
  for (let i = 0; i < headers.length; i++) {
    if (!headers[i]) continue;
    const cell = values[i];
    if (cell instanceof Date) {
      raw[headers[i]] = cell.toISOString();
    } else {
      raw[headers[i]] = cell === null || cell === undefined ? "" : String(cell);
    }
  }

  const relevance = _parseOptionalInt(raw.recommendation_relevance_1_5);
  const unnecessary = _parseOptionalInt(raw.unnecessary_question_0_1);
  const missed = _parseOptionalInt(raw.missed_clarification_0_1);
  const completion = _parseOptionalInt(raw.task_completion_0_1);
  const faithfulness = _parseOptionalInt(raw.explanation_faithfulness_1_5);
  const turns = _parseOptionalInt(raw.turns_to_completion) || 0;

  return {
    row_index: rowIndex,
    source: raw.source || "",
    suite: raw.suite || "",
    scenario_id: raw.scenario_id || "",
    task_type: raw.task_type || "",
    mode: raw.mode || "",
    turns_to_completion: turns,
    question: raw.question || "",
    final_reply: raw.final_reply || "",
    clarification_asked: _parseBool(raw.clarification_asked),
    asked_slots: raw.asked_slots || "",
    decision_source: raw.decision_source || "",
    reason_codes: raw.reason_codes || "",
    tools_used: raw.tools_used || "",
    applied_rule_ids: raw.applied_rule_ids || "",
    products_summary: raw.products_summary || "",
    outfits_summary: raw.outfits_summary || "",
    grounding_ok: _parseBool(raw.grounding_ok),
    recommendation_relevance_1_5: relevance,
    unnecessary_question_0_1: unnecessary,
    missed_clarification_0_1: missed,
    task_completion_0_1: completion,
    explanation_faithfulness_1_5: faithfulness,
    judge_notes: raw.judge_notes || "",
    judge_name: raw.judge_name || "",
    updated_at: raw.updated_at || "",
    status: _computeStatus([relevance, unnecessary, missed, completion, faithfulness]),
  };
}

function _computeProgressCounts(rows) {
  let scored = 0;
  let partial = 0;
  let unscored = 0;
  for (let i = 0; i < rows.length; i++) {
    if (rows[i].status === "scored") scored += 1;
    else if (rows[i].status === "partial") partial += 1;
    else unscored += 1;
  }
  return { total: rows.length, scored, partial, unscored };
}

function _jsonResponse(obj) {
  return ContentService
    .createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}

function _validateScore(value, minVal, maxVal, fieldName) {
  if (value === null || value === undefined || value === "") return "";
  const num = Number(value);
  if (!Number.isInteger(num) || num < minVal || num > maxVal) {
    throw new Error("Invalid " + fieldName + ": expected integer in [" + minVal + ", " + maxVal + "]");
  }
  return num;
}

function doGet(e) {
  try {
    const params = (e && e.parameter) ? e.parameter : {};
    if (!_isAuthorized(params.token)) {
      return _jsonResponse({ ok: false, error: "INVALID_TOKEN" });
    }

    const sheet = _getSheet();
    const action = String(params.action || "rows").trim();
    const sheetName = sheet.getName();

    if (action === "ping") {
      const rowCount = Math.max(0, sheet.getLastRow() - 1);
      return _jsonResponse({ ok: true, sheet_name: sheetName, row_count: rowCount });
    }

    const headers = _ensureCollabColumns(sheet);
    const lastRow = sheet.getLastRow();
    const rows = [];
    if (lastRow > 1) {
      const dataValues = sheet.getRange(2, 1, lastRow - 1, headers.length).getValues();
      for (let idx = 0; idx < dataValues.length; idx++) {
        rows.push(_parseRow(idx, headers, dataValues[idx]));
      }
    }

    const counts = _computeProgressCounts(rows);
    return _jsonResponse({
      ok: true,
      sheet_name: sheetName,
      csv_path: "Google Sheet: " + sheetName,
      total: counts.total,
      scored: counts.scored,
      partial: counts.partial,
      unscored: counts.unscored,
      rows: rows,
    });
  } catch (err) {
    return _jsonResponse({ ok: false, error: String(err && err.message ? err.message : err) });
  }
}

function doPost(e) {
  const lock = LockService.getScriptLock();
  try {
    const rawBody = (e && e.postData && e.postData.contents) ? e.postData.contents : "{}";
    const payload = JSON.parse(rawBody);

    if (!_isAuthorized(payload.token)) {
      return _jsonResponse({ ok: false, error: "INVALID_TOKEN" });
    }

    const action = String(payload.action || "update_row");
    if (action !== "update_row") {
      return _jsonResponse({ ok: false, error: "UNSUPPORTED_ACTION: " + action });
    }

    lock.waitLock(15000);

    const sheet = _getSheet();
    const headers = _ensureCollabColumns(sheet);
    const totalRows = Math.max(0, sheet.getLastRow() - 1);
    const rowIndex = Number(payload.row_index);

    if (!Number.isInteger(rowIndex) || rowIndex < 0 || rowIndex >= totalRows) {
      return _jsonResponse({
        ok: false,
        error: "ROW_OUT_OF_RANGE: " + rowIndex + " (total=" + totalRows + ")",
      });
    }

    const sheetRowNum = rowIndex + 2;
    const rowValues = sheet.getRange(sheetRowNum, 1, 1, headers.length).getValues()[0];
    const currentRow = _parseRow(rowIndex, headers, rowValues);

    const expectedUpdatedAt = String(payload.expected_updated_at || "");
    if (!payload.force && currentRow.updated_at && currentRow.updated_at !== expectedUpdatedAt) {
      return _jsonResponse({
        ok: false,
        conflict: true,
        error: "ROW_CONFLICT",
        server_row: currentRow,
      });
    }

    const updates = {
      recommendation_relevance_1_5: _validateScore(
        payload.recommendation_relevance_1_5, 1, 5, "recommendation_relevance_1_5"
      ),
      unnecessary_question_0_1: _validateScore(
        payload.unnecessary_question_0_1, 0, 1, "unnecessary_question_0_1"
      ),
      missed_clarification_0_1: _validateScore(
        payload.missed_clarification_0_1, 0, 1, "missed_clarification_0_1"
      ),
      task_completion_0_1: _validateScore(
        payload.task_completion_0_1, 0, 1, "task_completion_0_1"
      ),
      explanation_faithfulness_1_5: _validateScore(
        payload.explanation_faithfulness_1_5, 1, 5, "explanation_faithfulness_1_5"
      ),
      judge_notes: payload.judge_notes === null || payload.judge_notes === undefined
        ? ""
        : String(payload.judge_notes),
      judge_name: payload.judge_name === null || payload.judge_name === undefined
        ? currentRow.judge_name
        : String(payload.judge_name).trim(),
      updated_at: new Date().toISOString(),
    };

    const nextValues = rowValues.slice();
    const keys = Object.keys(updates);
    for (let k = 0; k < keys.length; k++) {
      const colName = keys[k];
      const colIdx = headers.indexOf(colName);
      if (colIdx !== -1) {
        nextValues[colIdx] = updates[colName];
        sheet.getRange(sheetRowNum, colIdx + 1).setValue(updates[colName]);
      }
    }

    const updatedRow = _parseRow(rowIndex, headers, nextValues);

    // Recompute summary counters across all rows
    const allValues = sheet.getRange(2, 1, totalRows, headers.length).getValues();
    const allRows = [];
    for (let idx = 0; idx < allValues.length; idx++) {
      allRows.push(_parseRow(idx, headers, allValues[idx]));
    }
    const counts = _computeProgressCounts(allRows);

    return _jsonResponse({
      ok: true,
      row: updatedRow,
      total: counts.total,
      scored: counts.scored,
      partial: counts.partial,
      unscored: counts.unscored,
    });
  } catch (err) {
    return _jsonResponse({ ok: false, error: String(err && err.message ? err.message : err) });
  } finally {
    try {
      lock.releaseLock();
    } catch (_ignore) {
      // Ignore if lock was never acquired.
    }
  }
}

