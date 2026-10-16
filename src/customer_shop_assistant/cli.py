import sys

import anyio

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ClarificationMode,
    UserProfile,
)
from customer_shop_assistant.schemas.shop import GenderTarget
from customer_shop_assistant.services.agent_service import ShopAssistantAgent

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ANSI Color codes for Windows & cross-platform terminal
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_banner(provider: str, model: str, mode: ClarificationMode) -> None:
    """Print welcoming CLI banner with current status."""
    print(f"\n{CYAN}{BOLD}{'=' * 66}{RESET}")
    print(f"{CYAN}{BOLD}   FASHION SHOP ASSISTANT — INTERACTIVE TERMINAL (CLI){RESET}")
    print(f"{CYAN}{BOLD}{'=' * 66}{RESET}")
    print(f"{DIM}  • LLM Provider   :{RESET} {YELLOW}{provider}{RESET}")
    print(f"{DIM}  • Active Model   :{RESET} {YELLOW}{model}{RESET}")
    print(f"{DIM}  • Active Mode    :{RESET} {GREEN}{mode.value}{RESET}")
    print(f"{CYAN}{'-' * 66}{RESET}")
    print(f"{DIM}  Lệnh tiện ích:{RESET}")
    print(f"  - {BOLD}/mode 1|2|3{RESET}       : Đổi chế độ làm rõ (1/2/3)")
    print(f"  - {BOLD}/profile [h] [w]{RESET} : Cập nhật số đo (vd: /profile 172 65)")
    print(f"  - {BOLD}/profile show|clear{RESET}: Xem hoặc xóa thông tin vóc dáng")
    print(f"  - {BOLD}/reset{RESET}             : Xóa lịch sử phiên hội thoại")
    print(f"  - {BOLD}/exit{RESET} hoặc {BOLD}quit{RESET}   : Thoát chương trình")
    print(f"{CYAN}{'=' * 66}{RESET}\n")


def parse_profile_command(
    parts: list[str], current: UserProfile | None
) -> UserProfile | None:
    """Parse profile modification commands from user input."""
    if len(parts) == 1 or parts[1].lower() in ("show", "info"):
        if not current:
            print(f"{YELLOW}[Hồ sơ]{RESET} Chưa có thông tin vóc dáng.")
        else:
            items = []
            if current.height_cm:
                items.append(f"Cao: {current.height_cm}cm")
            if current.weight_kg:
                items.append(f"Nặng: {current.weight_kg}kg")
            if current.gender:
                items.append(f"Giới tính: {current.gender.value}")
            if current.fit_preference:
                items.append(f"Gu: {current.fit_preference}")
            print(f"{GREEN}[Hồ sơ hiện tại]{RESET} {', '.join(items)}")
        return current

    if parts[1].lower() in ("clear", "reset"):
        print(f"{YELLOW}[Hồ sơ]{RESET} Đã xóa thông tin vóc dáng.")
        return None

    h: float | None = None
    w: float | None = None
    g: GenderTarget | None = None
    fit: str | None = None

    for p in parts[1:]:
        val_lower = p.lower()
        if val_lower in ("nam", "male"):
            g = GenderTarget.MALE
        elif val_lower in ("nu", "nữ", "female"):
            g = GenderTarget.FEMALE
        elif val_lower in ("om", "ôm", "slim", "tight"):
            fit = "TIGHT"
        elif val_lower in ("rong", "rộng", "loose", "oversize"):
            fit = "LOOSE"
        elif val_lower in ("vua", "vừa", "regular"):
            fit = "REGULAR"
        else:
            try:
                num = float(p.replace("cm", "").replace("kg", ""))
                if 100 <= num <= 230:
                    h = num
                elif 30 <= num < 180:
                    w = num
            except ValueError:
                pass

    new_profile = UserProfile(
        height_cm=h or (current.height_cm if current else None),
        weight_kg=w or (current.weight_kg if current else None),
        gender=g or (current.gender if current else None),
        fit_preference=fit or (current.fit_preference if current else None),
    )
    g_txt = new_profile.gender.value if new_profile.gender else "Chưa rõ"
    print(
        f"{GREEN}[Hồ sơ đã lưu]{RESET} Cao: {new_profile.height_cm}cm | "
        f"Nặng: {new_profile.weight_kg}kg | Phái: {g_txt}"
    )
    return new_profile


async def run_cli() -> None:
    """Run interactive terminal session for clothing assistant."""
    settings = get_settings()
    active_model = (
        settings.OLLAMA_MODEL
        if settings.LLM_PROVIDER == "ollama"
        else settings.LMSTUDIO_MODEL
    )
    mode = ClarificationMode.MINIMAL_CLARIFICATION
    profile: UserProfile | None = None
    history: list[ChatMessage] = []

    print_banner(settings.LLM_PROVIDER, active_model, mode)
    agent = ShopAssistantAgent(settings=settings)

    while True:
        try:
            prompt_prefix = f"{BOLD}👤 Bạn: {RESET}"
            user_input = input(prompt_prefix).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{CYAN}Tạm biệt bạn! Hẹn gặp lại. 👋{RESET}")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("exit", "quit", "q", "/exit", "/quit"):
            print(f"\n{CYAN}Tạm biệt bạn! Hẹn gặp lại. 👋{RESET}")
            break

        if cmd in ("/help", "help", "?"):
            print_banner(settings.LLM_PROVIDER, active_model, mode)
            continue

        if cmd in ("/reset", "reset", "/clear", "clear"):
            history.clear()
            print(f"{YELLOW}[Hệ thống]{RESET} Đã xóa lịch sử hội thoại.")
            continue

        if cmd.startswith("/mode"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1] in ("2", "immediate"):
                mode = ClarificationMode.IMMEDIATE_ANSWER
                print(f"{GREEN}[Chế độ]{RESET} IMMEDIATE_ANSWER (Trả lời ngay)")
            elif len(parts) > 1 and parts[1] in ("3", "always"):
                mode = ClarificationMode.ALWAYS_CLARIFY
                print(f"{GREEN}[Chế độ]{RESET} ALWAYS_CLARIFY (Luôn hỏi kỹ)")
            else:
                mode = ClarificationMode.MINIMAL_CLARIFICATION
                print(f"{GREEN}[Chế độ]{RESET} MINIMAL_CLARIFICATION (Tối thiểu)")
            continue

        if cmd.startswith("/profile"):
            parts = user_input.split()
            profile = parse_profile_command(parts, profile)
            continue

        # Execute conversation turn with live token streaming
        req = ChatRequest(
            message=user_input,
            clarification_mode=mode,
            user_profile=profile,
            history=history,
        )

        header_printed = False
        full_reply = ""
        tools_used_list: list[str] = []
        tool_in_progress = False

        async for chunk in agent.chat_stream(req):
            c_type = chunk.get("type")

            if c_type == "intent":
                intent_name = chunk.get("intent", "")
                tag = f" {DIM}[{intent_name}]{RESET}" if intent_name else ""
                print(
                    f"\n{MAGENTA}{BOLD}🤖 Assistant{RESET}{tag}:\n",
                    end="",
                    flush=True,
                )
                header_printed = True

            elif c_type == "clarification":
                if not header_printed:
                    intent_name = chunk.get("intent", "")
                    tag = f" {DIM}[{intent_name}]{RESET}" if intent_name else ""
                    print(
                        f"\n{MAGENTA}{BOLD}🤖 Assistant{RESET}{tag}:\n",
                        end="",
                        flush=True,
                    )
                    header_printed = True
                content = chunk.get("content", "")
                for char in content:
                    print(char, end="", flush=True)
                    await anyio.sleep(0.008)
                full_reply = content

            elif c_type == "tool_start":
                tool = chunk.get("tool", "")
                if tool not in tools_used_list:
                    tools_used_list.append(tool)
                print(
                    f"  {DIM}🔧 [Đang tra cứu dữ liệu: {tool}...]{RESET}\n",
                    flush=True,
                )
                tool_in_progress = True

            elif c_type == "token":
                if not header_printed:
                    print(
                        f"\n{MAGENTA}{BOLD}🤖 Assistant{RESET}:\n",
                        end="",
                        flush=True,
                    )
                    header_printed = True
                if tool_in_progress:
                    tool_in_progress = False
                token = chunk.get("content", "")
                print(token, end="", flush=True)
                full_reply += token

            elif c_type == "done":
                full_reply = chunk.get("reply", full_reply)
                tools_used_list = chunk.get("tools_used", tools_used_list)

            elif c_type == "error":
                print(f"\n{YELLOW}{chunk.get('content', '')}{RESET}")
                full_reply = chunk.get("content", "")

        print("\n")

        # Update turn history
        history.append(ChatMessage(role="user", content=user_input))
        history.append(ChatMessage(role="assistant", content=full_reply))


def main() -> None:
    """CLI script entrypoint supporting interactive chat, eval, and experiments."""
    args = sys.argv[1:]
    if args and args[0] in ("eval", "benchmark"):
        # Shift arguments and invoke eval runner
        sys.argv = [sys.argv[0]] + args[1:]
        from customer_shop_assistant.eval.runner import async_main

        try:
            anyio.run(async_main)
        except KeyboardInterrupt:
            sys.exit(0)
    elif args and args[0] in ("experiment", "exp"):
        # Shift arguments and invoke experiment runner
        sys.argv = [sys.argv[0]] + args[1:]
        from customer_shop_assistant.eval.experiments import async_main as exp_main

        try:
            anyio.run(exp_main)
        except KeyboardInterrupt:
            sys.exit(0)
    else:
        try:
            anyio.run(run_cli)
        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == "__main__":
    main()
