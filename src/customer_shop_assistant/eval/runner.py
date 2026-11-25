"""Automated benchmark runner for the Functional Golden Evaluation Suite."""

import argparse
import asyncio
import datetime
import sys
import time
from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.eval.dataset import EvaluationCase, load_golden_set
from customer_shop_assistant.eval.e2e_benchmark import (
    check_eval_services,
    resolve_active_model_name,
)
from customer_shop_assistant.eval.metrics import (
    EvaluationMetricResult,
    evaluate_response,
)
from customer_shop_assistant.schemas.chat import ChatRequest, ClarificationMode
from customer_shop_assistant.services.agent_service import ShopAssistantAgent

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


class FakeDryRunChatModel(GenericFakeChatModel):
    """Deterministic fake chat model for fast zero-cost dry-runs."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def create_dry_run_agent(case: EvaluationCase) -> ShopAssistantAgent:
    """Construct an agent pre-loaded with mock output matching test case."""
    gt = case.ground_truth
    kw = ", ".join(gt.required_keywords) if gt.required_keywords else "sản phẩm"

    if gt.is_out_of_scope:
        content = (
            "Hiện tại mình chưa hỗ trợ tác vụ cá nhân hoặc tra cứu ngoài phạm vi. "
            "Bạn vui lòng liên hệ nhân viên shop để được hỗ trợ nhé."
        )
    elif gt.should_clarify:
        content = (
            f"Bạn có thể chia sẻ thêm thông tin về {kw} để shop tư vấn "
            "chuẩn xác nhất nhé."
        )
    elif gt.expected_policy_doc:
        content = (
            f"Theo quy định tại [{gt.expected_policy_doc}: Điều khoản], "
            f"shop áp dụng chi tiết: {kw}."
        )
    elif gt.is_out_of_stock:
        content = (
            f"Mẫu này hiện tại đã hết hàng, "
            f"bạn có thể tham khảo mẫu gợi ý thay thế: {kw}."
        )
    else:
        content = (
            f"Dựa trên yêu cầu của bạn, shop gợi ý {kw} rất phù hợp, giá cả ưu đãi."
        )

    fake_model = FakeDryRunChatModel(messages=iter([AIMessage(content=content)]))
    return ShopAssistantAgent(chat_model=fake_model)


class EvalRunner:
    """Benchmark runner for automated customer shop assistant evaluation."""

    def __init__(
        self,
        settings: Settings | None = None,
        dry_run: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.dry_run = dry_run
        self.provider = "mock" if dry_run else self.settings.LLM_PROVIDER
        self.model_name = (
            "dry-run-fake" if dry_run else resolve_active_model_name(self.settings)
        )

    async def run_case(
        self,
        case: EvaluationCase,
        agent: ShopAssistantAgent | None = None,
    ) -> EvaluationMetricResult:
        """Run a single test case through the assistant and score it."""
        if agent is None:
            agent = (
                create_dry_run_agent(case)
                if self.dry_run
                else ShopAssistantAgent(settings=self.settings)
            )

        req = ChatRequest(
            message=case.question,
            clarification_mode=case.clarification_mode,
            user_profile=case.user_profile,
        )

        start = time.perf_counter()
        res = await agent.chat(req)
        latency_ms = (time.perf_counter() - start) * 1000.0

        tools_used = list(res.tools_used)
        if (
            self.dry_run
            and case.ground_truth.expected_tool
            and not res.clarification_asked
            and not tools_used
        ):
            tools_used = [case.ground_truth.expected_tool]

        intent = res.intent or "UNKNOWN"

        return evaluate_response(
            case=case,
            reply=res.reply,
            intent=intent,
            tools_used=tools_used,
            clarification_asked=res.clarification_asked,
            latency_ms=latency_ms,
            outfits=res.outfits,
            products=res.products,
            clarification_form=res.clarification_form,
            debug_meta=res.debug_meta,
            provider=self.provider,
            model_name=self.model_name,
            allow_token_estimate=self.dry_run,
        )

    async def run_suite(
        self,
        cases: list[EvaluationCase],
        output_file: Path | None = None,
    ) -> list[EvaluationMetricResult]:
        """Execute a list of evaluation cases sequentially."""
        print(f"\n{CYAN}{BOLD}{'=' * 72}{RESET}")
        title = "   CUSTOMER SHOP ASSISTANT — FUNCTIONAL REGRESSION RUNNER"
        print(f"{CYAN}{BOLD}{title}{RESET}")
        print(f"{CYAN}{BOLD}{'=' * 72}{RESET}")
        prov = (
            "MOCK (Dry Run)" if self.dry_run else f"{self.provider} / {self.model_name}"
        )
        print(f"{DIM}  • Provider : {RESET}{YELLOW}{prov}{RESET}")
        print(f"{DIM}  • Total    : {RESET}{BOLD}{len(cases)}{RESET} cases")

        if not self.dry_run:
            health = await check_eval_services(
                self.settings, backend=self.settings.OUTFIT_KG_BACKEND
            )
            be_col = GREEN if health.main_be_ok else RED
            be_txt = (
                f"{'ONLINE' if health.main_be_ok else 'OFFLINE'} "
                f"({health.main_be_detail})"
            )
            print(f"{DIM}  • Main BE  : {RESET}{be_col}{be_txt}{RESET}")
            if not health.main_be_ok and any(c.requires_main_be for c in cases):
                print(
                    f"{YELLOW}  ! Warning: Main BE is offline; cases requiring live "
                    f"catalog/vouchers may fail.{RESET}"
                )
        print()

        results: list[EvaluationMetricResult] = []

        for idx, case in enumerate(cases, start=1):
            sys.stdout.write(
                f"[{idx}/{len(cases)}] {case.id:<18} ({case.category:<20}) ... "
            )
            sys.stdout.flush()

            try:
                result = await self.run_case(case)
                results.append(result)

                if result.passed:
                    print(f"{GREEN}PASS{RESET} {DIM}({result.latency_ms:.0f}ms){RESET}")
                else:
                    reason_summary = "; ".join(result.failure_reasons[:2])
                    print(f"{RED}FAIL{RESET} {DIM}({reason_summary}){RESET}")
            except Exception as exc:
                print(f"{RED}ERROR: {exc}{RESET}")

        if output_file is None:
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            output_file = log_dir / f"eval_{now_str}.jsonl"
        else:
            output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(r.model_dump_json() + "\n")

        self.print_summary_table(results, output_file)
        return results

    def print_summary_table(
        self,
        results: list[EvaluationMetricResult],
        log_path: Path,
    ) -> None:
        """Display aggregated metrics table in terminal."""
        if not results:
            print(f"{YELLOW}No results to display.{RESET}")
            return

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        pass_rate = (passed / total) * 100.0 if total > 0 else 0.0
        avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0.0
        total_tokens = sum(r.total_tokens for r in results)

        print(f"\n{CYAN}{BOLD}{'=' * 72}{RESET}")
        print(f"{CYAN}{BOLD}                    EVALUATION SUMMARY REPORT{RESET}")
        print(f"{CYAN}{BOLD}{'=' * 72}{RESET}")
        col = GREEN if pass_rate >= 85.0 else (YELLOW if pass_rate >= 70.0 else RED)
        stat = f"{col}{BOLD}{pass_rate:.1f}%{RESET} ({passed}/{total} passed)"
        print(f"  • Overall Pass Rate : {stat}")
        print(f"  • Average Latency   : {BOLD}{avg_latency:.1f} ms{RESET}")
        print(f"  • Total Tokens      : {BOLD}{total_tokens:,}{RESET}")
        print(f"  • Telemetry Output  : {DIM}{log_path}{RESET}")
        outfit_cases = [r for r in results if r.outfit_returned is not None]
        if outfit_cases:
            ret_rate = (
                sum(1 for r in outfit_cases if r.outfit_returned)
                / len(outfit_cases)
                * 100.0
            )
            viol_rates = [
                r.outfit_rule_violation_rate
                for r in outfit_cases
                if r.outfit_rule_violation_rate is not None
            ]
            avg_viol = (
                (sum(viol_rates) / len(viol_rates)) * 100.0 if viol_rates else 0.0
            )
            print(f"  • Outfit Returned   : {BOLD}{ret_rate:.1f}%{RESET}")
            print(f"  • Rule Violations   : {BOLD}{avg_viol:.1f}%{RESET}")
        print(f"{CYAN}{'-' * 72}{RESET}")

        categories: dict[str, list[EvaluationMetricResult]] = {}
        for r in results:
            categories.setdefault(r.category, []).append(r)

        header = (
            f"  {'Category':<20} | {'Total':<5} | {'Pass':<5} "
            f"| {'Pass Rate':<10} | {'Avg Latency'}"
        )
        print(header)
        print(f"  {'-' * 20}-+-{'-' * 5}-+-{'-' * 5}-+-{'-' * 10}-+-{'-' * 12}")

        for cat, items in sorted(categories.items()):
            c_tot = len(items)
            c_pass = sum(1 for i in items if i.passed)
            c_rate = (c_pass / c_tot) * 100.0 if c_tot > 0 else 0.0
            c_lat = sum(i.latency_ms for i in items) / c_tot if c_tot > 0 else 0.0
            c_col = GREEN if c_rate >= 85.0 else (YELLOW if c_rate >= 70.0 else RED)
            rate_str = f"{c_col}{c_rate:>8.1f}%{RESET}"
            row_str = (
                f"  {cat:<18} | {c_tot:<5} | {c_pass:<5} "
                f"| {rate_str} | {c_lat:>7.1f} ms"
            )
            print(row_str)

        print(f"{CYAN}{'=' * 72}{RESET}\n")


async def async_main() -> None:
    """Async entrypoint for eval runner CLI."""
    parser = argparse.ArgumentParser(description="Evaluate Customer Shop Assistant")
    parser.add_argument(
        "--dataset",
        type=str,
        default="eval/questions.json",
        help="Path to questions dataset",
    )
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of cases")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Override clarification mode (immediate, conditional, always)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run with mock model (no LLM calls)"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Custom JSONL output path"
    )

    args = parser.parse_args()

    cases = load_golden_set(
        filepath=args.dataset,
        category=args.category,
        limit=args.limit,
    )

    if args.mode:
        m_lower = args.mode.lower()
        if "imm" in m_lower:
            mode_override = ClarificationMode.IMMEDIATE_ANSWER
        elif "alw" in m_lower:
            mode_override = ClarificationMode.ALWAYS_CLARIFY
        else:
            mode_override = ClarificationMode.CONDITIONAL_CLARIFICATION
        for c in cases:
            c.clarification_mode = mode_override

    runner = EvalRunner(dry_run=args.dry_run)
    out_path = Path(args.output) if args.output else None
    await runner.run_suite(cases, output_file=out_path)


def main() -> None:
    """CLI script entrypoint."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
