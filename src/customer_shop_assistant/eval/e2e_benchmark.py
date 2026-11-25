"""Tier 2 Multi-turn End-to-End LLM Benchmark & Service Preflight Checker."""

import json
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.eval.dataset import E1Scenario
from customer_shop_assistant.eval.metrics import (
    compute_slot_accuracy,
    evaluate_turn_structure_and_grounding,
    extract_token_usage,
)
from customer_shop_assistant.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ClarificationMode,
    UserProfile,
)
from customer_shop_assistant.services.outfit_kg_repository import (
    Neo4jOutfitKGRepository,
)


class ServiceHealthStatus(BaseModel):
    """Health status of external services used by E2E evaluation."""

    main_be_url: str
    main_be_ok: bool
    main_be_detail: str = ""
    kg_backend: str = "inmemory"
    neo4j_ok: bool | None = None
    neo4j_detail: str = ""


class ContrastiveModeExpectation(BaseModel):
    """Expected behavior of a single mode on a contrastive S1-S10 scenario."""

    turn1_action: Literal["ASK", "PROCEED", "GRAY_ZONE"]
    expected_decision_sources: list[str] = Field(default_factory=list)
    expected_reason_codes: list[str] = Field(default_factory=list)
    expected_asked_slots: list[str] = Field(default_factory=list)
    forbidden_asked_slots: list[str] = Field(default_factory=list)
    max_questions: int = 3
    expected_tools: list[str] = Field(default_factory=list)
    turn2_action: Literal["ASK", "PROCEED"] | None = None
    hard_constraints: dict[str, Any] = Field(default_factory=dict)


class ContrastiveScenario(BaseModel):
    """Multi-turn contrastive scenario (S1-S10) comparing Modes A, B, C."""

    id: str
    name: str
    group: str
    expected_intent: str
    turn1_prompt: str
    user_profile: UserProfile | None = None
    turn2_prompt_by_mode: dict[str, str] = Field(default_factory=dict)
    requires_main_be: bool = False
    expected_by_mode: dict[str, ContrastiveModeExpectation]


class E2ETurnRecord(BaseModel):
    """Telemetry captured for a single live/mock turn."""

    turn_number: int
    user_message: str
    reply: str
    intent: str | None = None
    clarification_asked: bool = False
    question_count: int = 0
    asked_slots: list[str] = Field(default_factory=list)
    decision_source: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    products_count: int = 0
    outfit_count: int = 0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class E2EScenarioRunResult(BaseModel):
    """Complete multi-turn E2E evaluation result for (scenario, mode)."""

    suite: Literal["contrastive", "stratified"]
    scenario_id: str
    task_type: str
    mode: str
    provider: str
    model_name: str
    question: str
    turns_to_completion: int
    clarification_asked_turn1: bool
    asked_slots_turn1: list[str] = Field(default_factory=list)
    expected_asked_slots: list[str] = Field(default_factory=list)
    slot_precision: float | None = None
    slot_recall: float | None = None
    slot_exact_match: bool | None = None
    decision_source_turn1: str | None = None
    reason_codes_turn1: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    applied_rule_ids: list[str] = Field(default_factory=list)
    products_summary: str = ""
    outfits_summary: str = ""
    final_reply: str = ""
    intent_match: bool = True
    action_match: bool = True
    decision_source_match: bool = True
    tool_match: bool = True
    mode_compliant: bool = True
    form_integrity_ok: bool = True
    grounding_ok: bool = True
    task_completed: bool = True
    passed: bool = True
    failure_reasons: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    error: str | None = None
    turns: list[E2ETurnRecord] = Field(default_factory=list)


class E2EAggregateMetrics(BaseModel):
    """Aggregated Tier-2 E2E metrics per ClarificationMode."""

    mode: str
    total: int
    error_count: int = 0
    pass_rate: float
    avg_turns: float
    clarification_rate: float
    mode_compliance_rate: float
    grounding_rate: float
    task_completion_rate: float
    avg_input_tokens: float
    avg_output_tokens: float
    avg_total_tokens: float
    avg_latency_ms: float


def resolve_active_model_name(settings: Settings) -> str:
    """Return configured active model identifier from settings."""
    if settings.LLM_PROVIDER == "ollama":
        return settings.OLLAMA_MODEL
    return settings.LMSTUDIO_MODEL


async def check_eval_services(
    settings: Settings | None = None,
    *,
    backend: Literal["inmemory", "neo4j"] = "inmemory",
) -> ServiceHealthStatus:
    """Preflight check Main BE and optional Neo4j connectivity before E2E runs."""
    cfg = settings or get_settings()
    base = cfg.MAIN_BE_BASE_URL.rstrip("/")
    if not base.endswith("/api"):
        base = f"{base}/api"
    probe_url = f"{base}/products"

    main_be_ok = False
    main_be_detail = ""
    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            resp = await client.get(probe_url, params={"limit": 1})
            main_be_ok = resp.status_code < 500
            main_be_detail = f"HTTP {resp.status_code}"
    except Exception as exc:
        main_be_detail = f"{type(exc).__name__}: {exc}"

    neo4j_ok: bool | None = None
    neo4j_detail = ""
    if backend == "neo4j":
        repo: Neo4jOutfitKGRepository | None = None
        try:
            repo = Neo4jOutfitKGRepository(
                uri=cfg.NEO4J_URI,
                user=cfg.NEO4J_USER,
                password=cfg.NEO4J_PASSWORD,
                database=cfg.NEO4J_DATABASE,
            )
            repo.verify_connectivity()
            neo4j_ok = True
            neo4j_detail = "connected"
        except Exception as exc:
            neo4j_ok = False
            neo4j_detail = f"{type(exc).__name__}: {exc}"
        finally:
            if repo is not None:
                repo.close()

    return ServiceHealthStatus(
        main_be_url=probe_url,
        main_be_ok=main_be_ok,
        main_be_detail=main_be_detail,
        kg_backend=backend,
        neo4j_ok=neo4j_ok,
        neo4j_detail=neo4j_detail,
    )


def load_contrastive_scenarios(
    filepath: Path | str = Path("eval/e1_contrastive_s1_s10.json"),
) -> list[ContrastiveScenario]:
    """Load validated S1-S10 contrastive multi-turn scenarios."""
    target = Path(filepath)
    if not target.exists():
        raise FileNotFoundError(f"Contrastive dataset not found at: {target}")
    raw = json.loads(target.read_text(encoding="utf-8"))
    return [
        ContrastiveScenario.model_validate(item) for item in raw.get("scenarios", [])
    ]


def select_stratified_scenarios(
    scenarios: list[E1Scenario],
    *,
    limit: int = 12,
    run_all: bool = False,
) -> list[E1Scenario]:
    """Select a balanced stratified subset across task types from scenarios."""
    if run_all or limit >= len(scenarios):
        return list(scenarios)
    if limit <= 0:
        return []

    buckets: dict[str, list[E1Scenario]] = {}
    for sc in scenarios:
        buckets.setdefault(sc.task_type, []).append(sc)

    priority_order = (
        "open",
        "anchor",
        "color_implicit",
        "color_ambiguous",
        "color_constraint",
        "search",
        "size",
        "body_fit",
        "non_shopping",
    )
    picks: list[E1Scenario] = []
    while len(picks) < limit:
        progressed = False
        for key in priority_order:
            pool = buckets.get(key, [])
            if pool and len(picks) < limit:
                picks.append(pool.pop(0))
                progressed = True
        if not progressed:
            for pool in buckets.values():
                if pool and len(picks) < limit:
                    picks.append(pool.pop(0))
                    progressed = True
        if not progressed:
            break
    return picks


def _extract_meta_dict(res: ChatResponse) -> dict[str, Any]:
    if isinstance(res.debug_meta, BaseModel):
        return res.debug_meta.model_dump()
    if isinstance(res.debug_meta, dict):
        return res.debug_meta
    return {}


def _build_turn2_answer_from_e1(
    scenario: E1Scenario, asked_slots: list[str]
) -> str | None:
    if scenario.user_answers.get("_decline") == "DECLINED":
        return "Nam nữ gì cũng được, bạn cứ gợi ý giúp mình"
    parts = [
        scenario.user_answers[slot]
        for slot in asked_slots
        if slot in scenario.user_answers
    ]
    if parts:
        return " ".join(parts)
    if scenario.user_answers:
        vals = [v for k, v in scenario.user_answers.items() if k != "_decline"]
        return " ".join(vals) if vals else None
    return None


def _summarize_products(res: ChatResponse) -> str:
    if not res.products:
        return ""
    return " | ".join(f"[{p.id}] {p.name} ({int(p.price)}d)" for p in res.products[:3])


def _summarize_outfits(res: ChatResponse) -> tuple[str, list[str]]:
    if not res.outfits:
        return "", []
    rules: list[str] = []
    for o in res.outfits:
        for rid in o.applied_rule_ids:
            if rid not in rules:
                rules.append(rid)
    summary = " | ".join(
        f"[{o.top.product_id}] {o.top.name} ({o.top.color}) + "
        f"[{o.bottom.product_id}] {o.bottom.name} ({o.bottom.color})"
        for o in res.outfits[:2]
    )
    return summary, rules


def _record_turn(
    turn_number: int, user_message: str, res: ChatResponse, latency_ms: float
) -> E2ETurnRecord:
    meta = _extract_meta_dict(res)
    tok = extract_token_usage(res.debug_meta, question=user_message, reply=res.reply)
    q_cnt = len(res.clarification_form.questions) if res.clarification_form else 0
    return E2ETurnRecord(
        turn_number=turn_number,
        user_message=user_message,
        reply=res.reply,
        intent=res.intent,
        clarification_asked=res.clarification_asked,
        question_count=q_cnt,
        asked_slots=list(meta.get("asked_slots") or []),
        decision_source=meta.get("decision_source"),
        reason_codes=list(meta.get("reason_codes") or []),
        tools_used=list(res.tools_used),
        products_count=len(res.products),
        outfit_count=len(res.outfits),
        latency_ms=round(latency_ms, 2),
        input_tokens=tok.input_tokens,
        output_tokens=tok.output_tokens,
        total_tokens=tok.total_tokens,
    )


def _evaluate_e2e_run(
    *,
    suite: Literal["contrastive", "stratified"],
    scenario_id: str,
    task_type: str,
    mode: ClarificationMode,
    provider: str,
    model_name: str,
    question: str,
    expected_intent: str,
    expected_turn1_action: str | None,
    expected_decision_sources: list[str],
    expected_asked_slots: list[str],
    forbidden_asked_slots: list[str],
    expected_tools: list[str],
    expected_turn2_action: str | None,
    hard_constraints: dict[str, Any],
    turns_records: list[E2ETurnRecord],
    responses: list[ChatResponse],
    error: str | None,
) -> E2EScenarioRunResult:
    reasons: list[str] = []
    if error:
        reasons.append(error)

    if not turns_records or not responses:
        return E2EScenarioRunResult(
            suite=suite,
            scenario_id=scenario_id,
            task_type=task_type,
            mode=mode.value,
            provider=provider,
            model_name=model_name,
            question=question,
            turns_to_completion=0,
            clarification_asked_turn1=False,
            expected_asked_slots=expected_asked_slots,
            intent_match=False,
            action_match=False,
            decision_source_match=False,
            tool_match=False,
            mode_compliant=False,
            form_integrity_ok=False,
            grounding_ok=False,
            task_completed=False,
            passed=False,
            failure_reasons=reasons or ["No turns executed."],
            error=error,
        )

    t1 = turns_records[0]
    r1 = responses[0]
    last_turn = turns_records[-1]
    last_resp = responses[-1]

    intent_match = (t1.intent or "").upper() == expected_intent.upper()
    if not intent_match:
        reasons.append(f"Intent mismatch: got {t1.intent}, expected {expected_intent}")

    action_match = True
    if expected_turn1_action == "ASK" and not t1.clarification_asked:
        action_match = False
        reasons.append("Turn 1 expected ASK but proceeded.")
    elif expected_turn1_action == "PROCEED" and t1.clarification_asked:
        action_match = False
        reasons.append("Turn 1 expected PROCEED but asked clarification.")

    if expected_turn2_action and len(turns_records) >= 2:
        t2 = turns_records[1]
        if expected_turn2_action == "ASK" and not t2.clarification_asked:
            action_match = False
            reasons.append("Turn 2 expected ASK but proceeded.")
        elif expected_turn2_action == "PROCEED" and t2.clarification_asked:
            action_match = False
            reasons.append("Turn 2 expected PROCEED but asked clarification.")

    decision_source_match = True
    if expected_decision_sources and (
        t1.decision_source not in expected_decision_sources
    ):
        decision_source_match = False
        reasons.append(
            f"Decision source mismatch: got '{t1.decision_source}', "
            f"expected one of {expected_decision_sources}"
        )

    if forbidden_asked_slots and any(
        s in t1.asked_slots for s in forbidden_asked_slots
    ):
        action_match = False
        reasons.append(
            f"Asked forbidden slots {forbidden_asked_slots}: got {t1.asked_slots}"
        )

    slot_acc = compute_slot_accuracy(expected_asked_slots, t1.asked_slots)

    all_tools: list[str] = []
    for tr in turns_records:
        for tool in tr.tools_used:
            if tool not in all_tools:
                all_tools.append(tool)

    tool_match = True
    if expected_tools and not any(t in all_tools for t in expected_tools):
        # Allow if the turn deliberately stayed in ASK (e.g. S6B Mode B asks 2 turns)
        if not last_turn.clarification_asked:
            tool_match = False
            reasons.append(f"Expected tools {expected_tools} not used; got {all_tools}")

    # Structural & grounding validation across turns
    mode_compliant = True
    form_integrity_ok = True
    grounding_ok = True
    hc_gender = hard_constraints.get("target_gender")
    hc_max_price = hard_constraints.get("max_price")

    for idx, (tr, resp) in enumerate(zip(turns_records, responses, strict=True)):
        meta = _extract_meta_dict(resp)
        check = evaluate_turn_structure_and_grounding(
            mode=mode,
            reply=resp.reply,
            intent=resp.intent,
            clarification_asked=resp.clarification_asked,
            clarification_form=resp.clarification_form,
            clarification_turns=int(meta.get("clarification_turns", idx) or 0),
            tools_used=resp.tools_used,
            products=resp.products,
            outfits=resp.outfits,
            target_gender=hc_gender,
            max_price=hc_max_price,
            user_question=tr.user_message,
        )
        if not check.mode_compliant:
            mode_compliant = False
        if not check.form_integrity_ok:
            form_integrity_ok = False
        if not check.grounding_ok:
            grounding_ok = False
        reasons.extend(check.violations)

    # Policy doc citation check if specified
    req_docs = hard_constraints.get("expected_policy_docs") or []
    if req_docs:
        low_final = last_resp.reply.lower()
        for doc in req_docs:
            if doc.lower() not in low_final:
                grounding_ok = False
                reasons.append(f"Missing policy citation '{doc}' in reply.")

    task_completed = (
        not last_turn.clarification_asked if expected_turn2_action != "ASK" else True
    )
    if not task_completed:
        reasons.append("Conversation did not reach completion on final turn.")

    outfit_sum, applied_rules = _summarize_outfits(last_resp)
    if not applied_rules:
        applied_rules = list(_extract_meta_dict(r1).get("evidence_rule_ids") or [])

    passed = (
        error is None
        and intent_match
        and action_match
        and decision_source_match
        and tool_match
        and mode_compliant
        and form_integrity_ok
        and grounding_ok
        and task_completed
    )

    return E2EScenarioRunResult(
        suite=suite,
        scenario_id=scenario_id,
        task_type=task_type,
        mode=mode.value,
        provider=provider,
        model_name=model_name,
        question=question,
        turns_to_completion=len(turns_records),
        clarification_asked_turn1=t1.clarification_asked,
        asked_slots_turn1=t1.asked_slots,
        expected_asked_slots=expected_asked_slots,
        slot_precision=slot_acc.precision,
        slot_recall=slot_acc.recall,
        slot_exact_match=slot_acc.exact_match,
        decision_source_turn1=t1.decision_source,
        reason_codes_turn1=t1.reason_codes,
        tools_used=all_tools,
        applied_rule_ids=applied_rules,
        products_summary=_summarize_products(last_resp),
        outfits_summary=outfit_sum,
        final_reply=last_resp.reply,
        intent_match=intent_match,
        action_match=action_match,
        decision_source_match=decision_source_match,
        tool_match=tool_match,
        mode_compliant=mode_compliant,
        form_integrity_ok=form_integrity_ok,
        grounding_ok=grounding_ok,
        task_completed=task_completed,
        passed=passed,
        failure_reasons=list(dict.fromkeys(reasons)),
        latency_ms=round(sum(t.latency_ms for t in turns_records), 2),
        input_tokens=sum(t.input_tokens for t in turns_records),
        output_tokens=sum(t.output_tokens for t in turns_records),
        total_tokens=sum(t.total_tokens for t in turns_records),
        error=error,
        turns=turns_records,
    )


async def run_e2e_contrastive_benchmark(
    contrastive_scenarios: list[ContrastiveScenario],
    *,
    backend: Literal["inmemory", "neo4j"] = "inmemory",
    settings: Settings | None = None,
    agent: Any = None,
    timeout_s: float = 120.0,
) -> list[E2EScenarioRunResult]:
    """Run S1-S10 multi-turn contrastive scenarios across Modes A, B, C."""
    from customer_shop_assistant.services.agent_service import ShopAssistantAgent

    cfg = (settings or get_settings()).model_copy(update={"OUTFIT_KG_BACKEND": backend})
    provider = cfg.LLM_PROVIDER
    model_name = resolve_active_model_name(cfg)
    active_agent = agent or ShopAssistantAgent(settings=cfg)

    modes = (
        ClarificationMode.IMMEDIATE_ANSWER,
        ClarificationMode.ALWAYS_CLARIFY,
        ClarificationMode.CONDITIONAL_CLARIFICATION,
    )
    results: list[E2EScenarioRunResult] = []

    for sc in contrastive_scenarios:
        for mode in modes:
            exp = sc.expected_by_mode.get(mode.value)
            if exp is None:
                continue
            sid = f"e2e-con-{sc.id}-{mode.value}-{int(time.time() * 1000)}"
            turns_records: list[E2ETurnRecord] = []
            responses: list[ChatResponse] = []
            err: str | None = None

            try:
                req1 = ChatRequest(
                    message=sc.turn1_prompt,
                    clarification_mode=mode,
                    user_profile=sc.user_profile,
                    session_id=sid,
                )
                t0 = time.perf_counter()
                res1 = (
                    await active_agent.chat(req1)
                    if timeout_s <= 0
                    else await _with_timeout(active_agent.chat(req1), timeout_s)
                )
                lat1 = (time.perf_counter() - t0) * 1000.0
                responses.append(res1)
                tr1 = _record_turn(1, sc.turn1_prompt, res1, lat1)
                turns_records.append(tr1)

                turn2_msg = sc.turn2_prompt_by_mode.get(mode.value)
                if res1.clarification_asked and turn2_msg:
                    req2 = ChatRequest(
                        message=turn2_msg,
                        clarification_mode=mode,
                        user_profile=sc.user_profile,
                        session_id=sid,
                    )
                    t1_start = time.perf_counter()
                    res2 = (
                        await active_agent.chat(req2)
                        if timeout_s <= 0
                        else await _with_timeout(active_agent.chat(req2), timeout_s)
                    )
                    lat2 = (time.perf_counter() - t1_start) * 1000.0
                    responses.append(res2)
                    turns_records.append(_record_turn(2, turn2_msg, res2, lat2))
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"

            results.append(
                _evaluate_e2e_run(
                    suite="contrastive",
                    scenario_id=sc.id,
                    task_type=sc.group,
                    mode=mode,
                    provider=provider,
                    model_name=model_name,
                    question=sc.turn1_prompt,
                    expected_intent=sc.expected_intent,
                    expected_turn1_action=exp.turn1_action,
                    expected_decision_sources=exp.expected_decision_sources,
                    expected_asked_slots=exp.expected_asked_slots,
                    forbidden_asked_slots=exp.forbidden_asked_slots,
                    expected_tools=exp.expected_tools,
                    expected_turn2_action=exp.turn2_action,
                    hard_constraints=exp.hard_constraints,
                    turns_records=turns_records,
                    responses=responses,
                    error=err,
                )
            )
    return results


async def run_e2e_stratified_benchmark(
    scenarios: list[E1Scenario],
    *,
    limit: int = 12,
    run_all: bool = False,
    backend: Literal["inmemory", "neo4j"] = "inmemory",
    settings: Settings | None = None,
    agent: Any = None,
    timeout_s: float = 120.0,
    output_jsonl_path: Path | None = None,
) -> list[E2EScenarioRunResult]:
    """Run multi-turn E2E evaluation on a stratified subset (or full Test split)."""
    from customer_shop_assistant.services.agent_service import ShopAssistantAgent

    picks = select_stratified_scenarios(scenarios, limit=limit, run_all=run_all)
    if not picks:
        return []

    cfg = (settings or get_settings()).model_copy(update={"OUTFIT_KG_BACKEND": backend})
    provider = cfg.LLM_PROVIDER
    model_name = resolve_active_model_name(cfg)
    active_agent = agent or ShopAssistantAgent(settings=cfg)

    modes = (
        ClarificationMode.IMMEDIATE_ANSWER,
        ClarificationMode.ALWAYS_CLARIFY,
        ClarificationMode.CONDITIONAL_CLARIFICATION,
    )
    results: list[E2EScenarioRunResult] = []

    for sc in picks:
        for mode in modes:
            sid = f"e2e-str-{sc.id}-{mode.value}-{int(time.time() * 1000)}"
            turns_records: list[E2ETurnRecord] = []
            responses: list[ChatResponse] = []
            err: str | None = None

            try:
                req1 = ChatRequest(
                    message=sc.question,
                    clarification_mode=mode,
                    user_profile=sc.user_profile,
                    session_id=sid,
                )
                t0 = time.perf_counter()
                res1 = await _with_timeout(active_agent.chat(req1), timeout_s)
                lat1 = (time.perf_counter() - t0) * 1000.0
                responses.append(res1)
                tr1 = _record_turn(1, sc.question, res1, lat1)
                turns_records.append(tr1)

                last_res = res1
                last_tr = tr1
                for turn_num in range(2, 4):
                    if not last_res.clarification_asked:
                        break
                    ans = _build_turn2_answer_from_e1(sc, last_tr.asked_slots)
                    if not ans:
                        break
                    req_next = ChatRequest(
                        message=ans,
                        clarification_mode=mode,
                        user_profile=sc.user_profile,
                        session_id=sid,
                    )
                    t_next_start = time.perf_counter()
                    res_next = await _with_timeout(
                        active_agent.chat(req_next), timeout_s
                    )
                    lat_next = (time.perf_counter() - t_next_start) * 1000.0
                    responses.append(res_next)
                    tr_next = _record_turn(turn_num, ans, res_next, lat_next)
                    turns_records.append(tr_next)
                    last_res = res_next
                    last_tr = tr_next
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"

            exp_action: str | None = None
            exp_slots: list[str] = []
            if mode == ClarificationMode.IMMEDIATE_ANSWER:
                exp_action = "PROCEED"
            elif mode == ClarificationMode.CONDITIONAL_CLARIFICATION:
                if sc.gold_label == "must_clarify":
                    exp_action = "ASK"
                    exp_slots = list(sc.expected_clarify_slots)
                elif sc.gold_label == "proceed":
                    exp_action = "PROCEED"

            expected_tools: list[str] = []
            if sc.expected_intent == "OUTFIT_ADVICE" and not sc.hard_constraints.get(
                "out_of_scope"
            ):
                expected_tools = ["suggest_outfit"]

            results.append(
                _evaluate_e2e_run(
                    suite="stratified",
                    scenario_id=sc.id,
                    task_type=sc.task_type,
                    mode=mode,
                    provider=provider,
                    model_name=model_name,
                    question=sc.question,
                    expected_intent=sc.expected_intent,
                    expected_turn1_action=exp_action,
                    expected_decision_sources=[],
                    expected_asked_slots=exp_slots,
                    forbidden_asked_slots=[],
                    expected_tools=expected_tools,
                    expected_turn2_action="PROCEED" if exp_action == "ASK" else None,
                    hard_constraints=sc.hard_constraints,
                    turns_records=turns_records,
                    responses=responses,
                    error=err,
                )
            )

    if output_jsonl_path is not None:
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl_path.open("w", encoding="utf-8") as f:
            for item in results:
                f.write(item.model_dump_json() + "\n")

    return results


async def _with_timeout(coro: Any, timeout_s: float) -> Any:
    import asyncio

    if timeout_s <= 0:
        return await coro
    return await asyncio.wait_for(coro, timeout=timeout_s)


def aggregate_e2e_results(
    results: list[E2EScenarioRunResult],
) -> dict[str, E2EAggregateMetrics]:
    """Aggregate Tier-2 E2E metrics grouped by ClarificationMode."""
    grouped: dict[str, list[E2EScenarioRunResult]] = {}
    for r in results:
        grouped.setdefault(r.mode, []).append(r)

    summary: dict[str, E2EAggregateMetrics] = {}
    for mode_name, items in grouped.items():
        tot = len(items)
        valid = [i for i in items if i.turns_to_completion > 0]
        v_cnt = len(valid)
        summary[mode_name] = E2EAggregateMetrics(
            mode=mode_name,
            total=tot,
            error_count=sum(1 for i in items if i.error is not None),
            pass_rate=round(sum(1 for i in items if i.passed) / tot * 100.0, 1),
            avg_turns=(
                round(sum(i.turns_to_completion for i in valid) / v_cnt, 2)
                if v_cnt > 0
                else 0.0
            ),
            clarification_rate=round(
                sum(1 for i in items if i.clarification_asked_turn1) / tot * 100.0, 1
            ),
            mode_compliance_rate=round(
                sum(1 for i in items if i.mode_compliant) / tot * 100.0, 1
            ),
            grounding_rate=round(
                sum(1 for i in items if i.grounding_ok) / tot * 100.0, 1
            ),
            task_completion_rate=round(
                sum(1 for i in items if i.task_completed) / tot * 100.0, 1
            ),
            avg_input_tokens=(
                round(sum(i.input_tokens for i in valid) / v_cnt, 1)
                if v_cnt > 0
                else 0.0
            ),
            avg_output_tokens=(
                round(sum(i.output_tokens for i in valid) / v_cnt, 1)
                if v_cnt > 0
                else 0.0
            ),
            avg_total_tokens=(
                round(sum(i.total_tokens for i in valid) / v_cnt, 1)
                if v_cnt > 0
                else 0.0
            ),
            avg_latency_ms=(
                round(sum(i.latency_ms for i in valid) / v_cnt, 1) if v_cnt > 0 else 0.0
            ),
        )
    return summary
