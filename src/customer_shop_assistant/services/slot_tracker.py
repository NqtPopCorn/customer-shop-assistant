"""Deterministic multi-turn slot tracker and turn intent classifier (P0-P3)."""

import re
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from customer_shop_assistant.schemas.chat import UserProfile
from customer_shop_assistant.schemas.shop import GenderTarget

SlotSource = Literal["explicit", "profile", "inferred"]


class TurnIntent(StrEnum):
    """High-level conversational intent for clarification planning and routing."""

    PRODUCT_SEARCH = "PRODUCT_SEARCH"
    OUTFIT_ADVICE = "OUTFIT_ADVICE"
    BODY_FIT_ADVICE = "BODY_FIT_ADVICE"
    SIZE_ADVICE = "SIZE_ADVICE"
    NON_SHOPPING = "NON_SHOPPING"


class SlotValue(BaseModel):
    """Typed slot payload tracking both extracted value and provenance."""

    value: Any = Field(..., description="Normalized slot value")
    source: SlotSource = Field(..., description="Origin: explicit, profile, inferred")


class BudgetValue(BaseModel):
    """Structured budget range in VND."""

    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)
    raw_text: str | None = Field(default=None)


class MeasurementsValue(BaseModel):
    """Customer body measurements for sizing and body-fit advice."""

    height_cm: float | None = Field(default=None, ge=50, le=250)
    weight_kg: float | None = Field(default=None, ge=20, le=200)
    waist_cm: float | None = Field(default=None, ge=30, le=150)


class ConversationSlots(BaseModel):
    """Multi-turn accumulated fashion slots organized by P0-P3 priority."""

    # P0 - Purpose & Core Context
    target_gender: SlotValue | None = None
    category: SlotValue | None = None
    occasion: SlotValue | None = None
    shopping_intent: SlotValue | None = None

    # P1 - Shopping Constraints
    budget: SlotValue | None = None
    fit: SlotValue | None = None
    size: SlotValue | None = None

    # P2 - Aesthetic Preferences
    style: SlotValue | None = None
    color: SlotValue | None = None
    material: SlotValue | None = None

    # P3 - Advanced Personalization
    measurements: SlotValue | None = None
    body_concern: SlotValue | None = None

    def has(self, name: str) -> bool:
        """Return True if the requested slot currently has a non-empty value."""
        if name == "category_subtype":
            cat_val = self.value_of("category")
            return cat_val is not None and cat_val not in ("tops", "pants")
        slot = getattr(self, name, None)
        if not isinstance(slot, SlotValue) or slot.value is None:
            return False
        if name == "measurements" and isinstance(slot.value, MeasurementsValue):
            return any(
                [
                    slot.value.height_cm is not None,
                    slot.value.weight_kg is not None,
                    slot.value.waist_cm is not None,
                ]
            )
        return True

    def value_of(self, name: str) -> Any:
        """Return underlying value for slot name, or None when unset."""
        slot = getattr(self, name, None)
        if isinstance(slot, SlotValue):
            return slot.value
        return None

    def to_debug_dict(self, intent: TurnIntent) -> dict[str, Any]:
        """Serialize slots for telemetry/debug including legacy summary keys."""
        gender_val = self.value_of("target_gender")
        known_gender = gender_val.lower() if isinstance(gender_val, str) else None
        return {
            "turn_intent": intent.value,
            "is_shopping_query": intent != TurnIntent.NON_SHOPPING,
            "is_size_inquiry": intent == TurnIntent.SIZE_ADVICE,
            "known_occasion": self.value_of("occasion"),
            "known_garment": self.value_of("category"),
            "known_gender": known_gender,
            "known_budget": self.has("budget"),
            "known_measurements": self.has("measurements"),
            "missing_priority_slots": get_missing_priority_slots(intent, self),
            "slots": self.model_dump(exclude_none=True),
        }


_OUT_OF_SCOPE_CUES = (
    "đơn hàng #",
    "đơn hàng vừa đặt",
    "tình trạng đơn",
    "hủy đơn",
    "huỷ đơn",
    "hủy giùm",
    "huỷ giùm",
    "khiếu nại",
    "chuyển tiền hoàn",
    "số tài khoản",
    "quên mật khẩu",
    "đổi mật khẩu",
    "viết code",
    "thời tiết hôm nay",
    "dự báo thời tiết",
)

_POLICY_AND_FAQ_CUES = (
    "đổi trả",
    "đổi size",
    "hoàn tiền",
    "phí ship",
    "tiền ship",
    "vận chuyển",
    "giao hàng",
    "freeship",
    "miễn phí ship",
    "thanh toán",
    "vietqr",
    "cod",
    "chuyển khoản",
    "bảo quản",
    "giặt ủi",
    "giặt máy",
    "giặt tay",
    "chính sách",
    "quy định",
    "địa chỉ shop",
    "cửa hàng ở đâu",
    "giờ mở cửa",
)

_VOUCHER_CUES = (
    "voucher",
    "mã giảm",
    "coupon",
    "khuyến mãi",
    "ưu đãi",
    "kiểm tra mã",
    "áp dụng mã",
    "áp được mã",
)

_DETAIL_BY_ID_PATTERNS = (
    r"\bchi tiết(?:\s+sản phẩm)?\b.*?(?:\bid\b|mã|#)\s*\d+",
    r"\bxem\s+(?:sản phẩm|mẫu)\s+(?:id\b|mã|#)\s*\d+",
    r"\bsản phẩm\s*(?:id|#)\s*\d+\b",
)

_SHOPPING_OR_GARMENT_CUES = (
    "áo",
    "quần",
    "đầm",
    "váy",
    "chân váy",
    "blazer",
    "hoodie",
    "bomber",
    "jacket",
    "polo",
    "sơ mi",
    "jeans",
    "kaki",
    "short",
    "thắt lưng",
    "túi",
    "mũ",
    "nón",
    "trang phục",
    "đồ",
    "outfit",
    "set",
    "phối",
    "mix",
    "tư vấn",
    "gợi ý",
    "tìm",
    "mua",
    "size",
    "mặc",
    "nam",
    "nữ",
    "unisex",
    "đi làm",
    "công sở",
    "đi chơi",
    "dạo phố",
    "dự tiệc",
    "đám cưới",
    "du lịch",
    "đi biển",
    "ngân sách",
    "giá",
    "đẹp trai",
    "đẹp gái",
    "xinh",
)

_GREETING_OR_THANKS_CUES = (
    "cảm ơn",
    "cám ơn",
    "thanks",
    "thank you",
    "xin chào",
    "chào shop",
    "chào bạn",
    "hello",
    "hi shop",
    "alo",
    "ok shop",
    "tạm biệt",
    "bye",
)

_INVENTORY_STOCK_CUES = (
    "còn hàng",
    "hết hàng",
    "tồn kho",
    "còn size",
    "hết size",
    "còn những size",
    "còn mẫu",
    "còn màu",
    "tạm hết",
)

_SIZE_ADVICE_CUES = (
    "bảng size",
    "tư vấn size",
    "chọn size",
    "lấy size",
    "mặc size",
    "size gì",
    "size nào",
    "vừa không",
    "vừa ko",
    "mặc vừa",
    "mặc chật",
    "mặc rộng",
    "số đo",
    "chiều cao",
    "cân nặng",
)

_BODY_FIT_CUES = (
    "đỡ gầy",
    "hơi gầy",
    "người gầy",
    "dáng gầy",
    "che bụng",
    "có bụng",
    "béo bụng",
    "đầy đặn",
    "mũm mĩm",
    "hack chiều cao",
    "nấm lùn",
    "trông cao hơn",
    "cao hơn",
    "thấp bé",
    "vóc dáng",
    "dáng người",
    "che khuyết điểm",
    "nên mặc áo form gì",
    "nên mặc form gì",
)

_OUTFIT_ADVICE_CUES = (
    "phối đồ",
    "phối với",
    "mix đồ",
    "mix & match",
    "mix match",
    "set đồ",
    "outfit",
    "mặc với gì",
    "kết hợp với",
    "nguyên set",
    "cả bộ",
)

_FEMALE_INFERRED_CATEGORIES = {"dress", "skirt"}
_SPECIFIC_CATEGORIES = {
    "shirt",
    "polo",
    "tshirt",
    "jacket",
    "jeans",
    "kaki",
    "short",
    "dress",
    "skirt",
    "accessory",
}


def classify_turn_intent(message: str) -> TurnIntent:
    """Classify a single customer turn into shopping vs non-shopping intent."""
    lower = message.strip().lower()
    if not lower:
        return TurnIntent.NON_SHOPPING

    if any(cue in lower for cue in _OUT_OF_SCOPE_CUES):
        return TurnIntent.NON_SHOPPING

    if any(cue in lower for cue in _POLICY_AND_FAQ_CUES):
        return TurnIntent.NON_SHOPPING

    if any(cue in lower for cue in _VOUCHER_CUES):
        return TurnIntent.NON_SHOPPING

    if any(re.search(pat, lower) for pat in _DETAIL_BY_ID_PATTERNS):
        return TurnIntent.NON_SHOPPING

    has_shopping_cue = any(cue in lower for cue in _SHOPPING_OR_GARMENT_CUES)
    if not has_shopping_cue and any(cue in lower for cue in _GREETING_OR_THANKS_CUES):
        return TurnIntent.NON_SHOPPING

    # Stock questions containing 'size' stay product search
    is_stock_query = any(cue in lower for cue in _INVENTORY_STOCK_CUES)
    if not is_stock_query and (
        any(cue in lower for cue in _SIZE_ADVICE_CUES)
        or re.search(r"\bsize\b", lower) is not None
    ):
        # Allow body-fit advice when user describes physique + asks for garment form
        if any(cue in lower for cue in _BODY_FIT_CUES) and not any(
            k in lower for k in ("size gì", "size nào", "bảng size", "tư vấn size")
        ):
            return TurnIntent.BODY_FIT_ADVICE
        return TurnIntent.SIZE_ADVICE

    if any(cue in lower for cue in _BODY_FIT_CUES):
        return TurnIntent.BODY_FIT_ADVICE

    if any(cue in lower for cue in _OUTFIT_ADVICE_CUES):
        return TurnIntent.OUTFIT_ADVICE

    return TurnIntent.PRODUCT_SEARCH


def _parse_price_token(num_str: str, unit: str) -> float | None:
    cleaned = num_str.replace(",", ".").strip()
    if cleaned.count(".") >= 1 and unit in ("", "đ", "d", "vnd", "vnđ"):
        # Handle Vietnamese thousands separator like 300.000
        parts = cleaned.split(".")
        if all(len(p) == 3 for p in parts[1:]):
            cleaned = "".join(parts)
    try:
        val = float(cleaned)
    except ValueError:
        return None

    u = unit.lower()
    if u in ("k", "nghìn", "ngàn"):
        val *= 1000
    elif u in ("tr", "triệu"):
        val *= 1_000_000
    elif val < 1000 and u == "":
        return None
    return val


def _extract_budget(lower: str) -> BudgetValue | None:
    # Handle explicit form option shortcuts or numeric ranges
    range_match = re.search(
        r"(?:từ\s*)?(\d+(?:[.,]\d+)?)\s*(k|nghìn|ngàn|triệu|tr|đ|vnd)?\s*"
        r"(?:-|–|đến|tới)\s*(\d+(?:[.,]\d+)?)\s*(k|nghìn|ngàn|triệu|tr|đ|vnd)?",
        lower,
    )
    if range_match:
        low_num, low_u, high_num, high_u = range_match.groups()
        eff_low_u = low_u or high_u or ""
        eff_high_u = high_u or low_u or ""
        min_p = _parse_price_token(low_num, eff_low_u)
        max_p = _parse_price_token(high_num, eff_high_u)
        if min_p is not None and max_p is not None:
            return BudgetValue(
                min_price=min_p,
                max_price=max_p,
                raw_text=range_match.group(0).strip(),
            )

    under_match = re.search(
        r"(?:dưới|tối đa|không quá|<=|<)\s*"
        r"(\d+(?:[.,]\d+)?)\s*(k|nghìn|ngàn|triệu|tr|đ|vnd)\b",
        lower,
    )
    if under_match:
        max_p = _parse_price_token(under_match.group(1), under_match.group(2))
        if max_p is not None:
            return BudgetValue(max_price=max_p, raw_text=under_match.group(0).strip())

    over_match = re.search(
        r"(?:trên|từ|tối thiểu|>=|>)\s*"
        r"(\d+(?:[.,]\d+)?)\s*(k|nghìn|ngàn|triệu|tr|đ|vnd)\b",
        lower,
    )
    if over_match:
        min_p = _parse_price_token(over_match.group(1), over_match.group(2))
        if min_p is not None:
            return BudgetValue(min_price=min_p, raw_text=over_match.group(0).strip())

    approx_match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(k|nghìn|ngàn|triệu|tr|đ|vnd)\b", lower
    )
    if approx_match:
        val = _parse_price_token(approx_match.group(1), approx_match.group(2))
        if val is not None:
            return BudgetValue(
                max_price=val,
                raw_text=approx_match.group(0).strip(),
            )

    return None


def _extract_gender(lower: str) -> str | None:
    # Check UNISEX first because phrases like 'Nam nữ đều mặc được' contain both
    if any(
        cue in lower
        for cue in ("unisex", "cả nam và nữ", "nam nữ đều", "nam nữ", "couple")
    ):
        return GenderTarget.UNISEX.value

    female_phrases = (
        "đẹp gái",
        "xinh gái",
        "xinh đẹp",
        "cho nữ",
        "đồ nữ",
        "con gái",
        "bạn gái",
        "nữ giới",
        "nữ tính",
        "phái nữ",
        "trang phục nữ",
        "áo nữ",
        "quần nữ",
    )
    if any(p in lower for p in female_phrases) or re.search(r"\b(?:nữ|vợ)\b", lower):
        return GenderTarget.FEMALE.value

    male_phrases = (
        "đẹp trai",
        "cho nam",
        "đồ nam",
        "con trai",
        "bạn trai",
        "nam giới",
        "nam tính",
        "phái nam",
        "trang phục nam",
        "áo nam",
        "quần nam",
    )
    if any(p in lower for p in male_phrases) or re.search(
        r"\b(?:nam|chồng|boy|men)\b", lower
    ):
        return GenderTarget.MALE.value

    return None


def _extract_category(lower: str) -> str | None:
    # Match multi-garment clarification form answers BEFORE specific keywords
    if any(
        phrase in lower
        for phrase in (
            "áo (sơ mi",
            "áo nam (áo",
            "áo nữ (",
            "áo sơ mi / áo polo",
            "áo thun / áo polo",
            "áo thun / áo khoác",
            "áo sơ mi / áo thun",
        )
    ):
        return "tops"

    if any(
        phrase in lower
        for phrase in (
            "quần (quần âu",
            "quần nam (quần",
            "quần nữ (",
            "quần âu / quần kaki",
            "quần jeans / quần short",
            "quần âu / quần jeans",
        )
    ):
        return "pants"

    if any(
        phrase in lower
        for phrase in (
            "đầm / váy",
            "đầm công sở liền thân",
            "đầm dạo phố",
        )
    ):
        return "dress"

    if any(k in lower for k in ("chân váy", "skirt")):
        return "skirt"
    if any(k in lower for k in ("đầm", "váy", "dress")):
        return "dress"
    if "sơ mi" in lower:
        return "shirt"
    if "polo" in lower:
        return "polo"
    if any(k in lower for k in ("áo thun", "t-shirt", "tshirt", "áo phông")):
        return "tshirt"
    if any(k in lower for k in ("áo khoác", "blazer", "hoodie", "bomber", "jacket")):
        return "jacket"
    if any(k in lower for k in ("jeans", "jean", "bò")):
        return "jeans"
    if "kaki" in lower or "khaki" in lower:
        return "kaki"
    if "short" in lower or "quần đùi" in lower:
        return "short"
    if any(k in lower for k in ("quần âu", "quần tây")):
        return "pants"
    if any(k in lower for k in ("thắt lưng", "túi xách", "mũ", "nón", "phụ kiện")):
        return "accessory"
    if re.search(r"\bquần\b", lower):
        return "pants"
    if re.search(r"\báo\b", lower):
        return "tops"
    return None


def _extract_occasion(lower: str) -> str | None:
    if any(
        k in lower
        for k in ("tiệc", "dạ hội", "party", "cưới", "sự kiện", "gala", "sinh nhật")
    ):
        return "party"
    if any(
        k in lower
        for k in (
            "công sở",
            "đi làm",
            "văn phòng",
            "đi họp",
            "phỏng vấn",
            "gặp khách",
            "smart-casual",
        )
    ):
        return "office"
    if any(k in lower for k in ("thể thao", "tập gym", "chạy bộ", "vận động", "gym")):
        return "sport"
    if any(
        k in lower
        for k in (
            "đi chơi",
            "dạo phố",
            "cà phê",
            "cafe",
            "du lịch",
            "đi biển",
            "hẹn hò",
            "cuối tuần",
            "hằng ngày",
            "hàng ngày",
        )
    ):
        return "casual"
    return None


def _extract_fit(lower: str) -> str | None:
    if any(k in lower for k in ("slim", "ôm nhẹ", "ôm dáng", "chiết eo", "bó")):
        return "SLIM"
    if any(
        k in lower
        for k in ("oversize", "oversized", "relaxed", "loose", "rộng rãi", "dáng xòe")
    ):
        return "LOOSE"
    if any(
        k in lower
        for k in ("regular", "vừa vặn", "chỉn chu", "dáng suông", "đứng form")
    ):
        return "REGULAR"
    return None


def _extract_measurements(lower: str) -> MeasurementsValue | None:
    height_cm: float | None = None
    weight_kg: float | None = None
    waist_cm: float | None = None

    m_h = re.search(r"(?:cao\s*)?(?:(\d)m(\d{1,2})|(\d{3})\s*cm|cao\s*(\d{3}))", lower)
    if m_h:
        if m_h.group(1) and m_h.group(2):
            cm_part = int(m_h.group(2))
            cm_norm = cm_part * 10 if cm_part < 10 else cm_part
            val = float(int(m_h.group(1)) * 100 + cm_norm)
            if 100 <= val <= 230:
                height_cm = val
        elif m_h.group(3) or m_h.group(4):
            val = float(m_h.group(3) or m_h.group(4))
            if 100 <= val <= 230:
                height_cm = val

    m_w = re.search(r"(?:nặng\s*(\d{2,3})|\b(\d{2,3})\s*(?:kg|ký|kí|cân))", lower)
    if m_w:
        w_str = m_w.group(1) or m_w.group(2)
        if w_str:
            val = float(w_str)
            if 30 <= val <= 200:
                weight_kg = val

    m_waist = re.search(r"(?:eo|vòng 2|vòng bụng)\s*[:=]?\s*(\d{2,3})", lower)
    if m_waist:
        val = float(m_waist.group(1))
        if 40 <= val <= 150:
            waist_cm = val

    if height_cm is None and weight_kg is None and waist_cm is None:
        return None
    return MeasurementsValue(
        height_cm=height_cm, weight_kg=weight_kg, waist_cm=waist_cm
    )


def _extract_color(lower: str) -> str | None:
    colors = (
        ("trắng", "Trắng"),
        ("đen", "Đen"),
        ("xanh navy", "Xanh navy"),
        ("navy", "Xanh navy"),
        ("xanh dương", "Xanh dương"),
        ("xanh pastel", "Xanh pastel"),
        ("xanh rêu", "Xanh rêu"),
        ("xanh", "Xanh"),
        ("be", "Be"),
        ("nâu", "Nâu"),
        ("xám", "Xám"),
        ("ghi", "Xám ghi"),
        ("đỏ", "Đỏ"),
        ("hồng", "Hồng"),
        ("vàng", "Vàng"),
        ("tím", "Tím"),
        ("trung tính", "Trung tính"),
        ("màu tối", "Tối màu"),
        ("màu sáng", "Sáng màu"),
    )
    for cue, label in colors:
        if re.search(rf"\b{re.escape(cue)}\b", lower):
            return label
    return None


def _extract_style(lower: str) -> str | None:
    styles = (
        ("tối giản", "Minimalist"),
        ("minimalist", "Minimalist"),
        ("streetwear", "Streetwear"),
        ("cá tính", "Streetwear / Cá tính"),
        ("sang trọng", "Sang trọng"),
        ("thanh lịch", "Thanh lịch"),
        ("trẻ trung", "Trẻ trung"),
        ("năng động", "Năng động"),
        ("công sở lịch sự", "Lịch sự công sở"),
        ("lịch sự", "Lịch sự"),
        ("casual", "Casual"),
    )
    for cue, label in styles:
        if cue in lower:
            return label
    return None


def _extract_material(lower: str) -> str | None:
    materials = (
        "cotton",
        "linen",
        "lụa",
        "ren",
        "kaki",
        "denim",
        "oxford",
        "rayon",
        "nỉ",
        "tuytsi",
        "sợi tre",
    )
    for mat in materials:
        if mat in lower:
            return mat.capitalize()
    return None


def _extract_size_label(lower: str) -> str | None:
    match = re.search(r"\bsize\s*(xs|s|m|l|xl|xxl|2xl|3xl|\d{2})\b", lower)
    if match:
        return match.group(1).upper()
    return None


def _extract_body_concern(lower: str) -> str | None:
    if any(k in lower for k in ("gầy", "đỡ gầy")):
        return "slim"
    if any(k in lower for k in ("che bụng", "có bụng", "béo bụng", "đầy đặn")):
        return "belly"
    if any(k in lower for k in ("hack chiều cao", "nấm lùn", "trông cao hơn", "thấp")):
        return "petite"
    return None


def extract_conversation_slots(
    messages: Sequence[BaseMessage | str],
    user_profile: UserProfile | None = None,
) -> tuple[ConversationSlots, TurnIntent]:
    """Accumulate P0-P3 slots across conversation history and classify latest turn."""
    slots = ConversationSlots()

    # 1. Seed from user profile (lowest priority among non-inferred sources)
    if user_profile:
        if user_profile.gender:
            slots.target_gender = SlotValue(
                value=user_profile.gender.value, source="profile"
            )
        if user_profile.fit_preference:
            slots.fit = SlotValue(
                value=user_profile.fit_preference.upper(), source="profile"
            )
        if any(
            [
                user_profile.height_cm is not None,
                user_profile.weight_kg is not None,
                user_profile.waist_cm is not None,
            ]
        ):
            slots.measurements = SlotValue(
                value=MeasurementsValue(
                    height_cm=user_profile.height_cm,
                    weight_kg=user_profile.weight_kg,
                    waist_cm=user_profile.waist_cm,
                ),
                source="profile",
            )

    # 2. Extract chronologically across all user messages (newer overrides older)
    user_texts: list[str] = []
    for msg in messages:
        if isinstance(msg, str):
            user_texts.append(msg)
        elif isinstance(msg, HumanMessage):
            user_texts.append(str(msg.content or ""))

    for text in user_texts:
        lower = text.strip().lower()
        if not lower:
            continue

        gender = _extract_gender(lower)
        if gender:
            slots.target_gender = SlotValue(value=gender, source="explicit")

        category = _extract_category(lower)
        if category:
            current_cat = slots.value_of("category")
            # Avoid downgrading a specific garment to generic tops/pants in follow-ups
            if not (
                current_cat in _SPECIFIC_CATEGORIES and category in ("tops", "pants")
            ):
                slots.category = SlotValue(value=category, source="explicit")

        occasion = _extract_occasion(lower)
        if occasion:
            slots.occasion = SlotValue(value=occasion, source="explicit")

        if any(cue in lower for cue in _OUTFIT_ADVICE_CUES):
            slots.shopping_intent = SlotValue(value="OUTFIT", source="explicit")
        elif category and not slots.has("shopping_intent"):
            slots.shopping_intent = SlotValue(value="PRODUCT", source="explicit")

        budget = _extract_budget(lower)
        if budget:
            slots.budget = SlotValue(value=budget, source="explicit")

        fit = _extract_fit(lower)
        if fit:
            slots.fit = SlotValue(value=fit, source="explicit")

        size_lbl = _extract_size_label(lower)
        if size_lbl:
            slots.size = SlotValue(value=size_lbl, source="explicit")

        color = _extract_color(lower)
        if color:
            slots.color = SlotValue(value=color, source="explicit")

        style = _extract_style(lower)
        if style:
            slots.style = SlotValue(value=style, source="explicit")

        material = _extract_material(lower)
        if material:
            slots.material = SlotValue(value=material, source="explicit")

        meas = _extract_measurements(lower)
        if meas:
            prev_meas = slots.value_of("measurements")
            if isinstance(prev_meas, MeasurementsValue):
                merged = MeasurementsValue(
                    height_cm=meas.height_cm or prev_meas.height_cm,
                    weight_kg=meas.weight_kg or prev_meas.weight_kg,
                    waist_cm=meas.waist_cm or prev_meas.waist_cm,
                )
            else:
                merged = meas
            slots.measurements = SlotValue(value=merged, source="explicit")

        concern = _extract_body_concern(lower)
        if concern:
            slots.body_concern = SlotValue(value=concern, source="explicit")

    # 3. High-confidence inference ONLY for unmistakably female categories
    if (
        not slots.has("target_gender")
        and slots.value_of("category") in _FEMALE_INFERRED_CATEGORIES
    ):
        slots.target_gender = SlotValue(
            value=GenderTarget.FEMALE.value, source="inferred"
        )

    latest_text = user_texts[-1] if user_texts else ""
    intent = classify_turn_intent(latest_text)
    if intent == TurnIntent.OUTFIT_ADVICE:
        slots.shopping_intent = SlotValue(value="OUTFIT", source="explicit")
    elif not slots.has("shopping_intent") and intent != TurnIntent.NON_SHOPPING:
        slots.shopping_intent = SlotValue(value="PRODUCT", source="inferred")

    return slots, intent


def extract_session_profile_updates(slots: ConversationSlots) -> dict[str, Any]:
    """Build safe session-scoped profile updates from confirmed/explicit slots."""
    updates: dict[str, Any] = {}

    if slots.target_gender is not None:
        g_val = slots.target_gender.value
        if slots.target_gender.source == "explicit" or (
            slots.target_gender.source == "inferred"
            and g_val == GenderTarget.FEMALE.value
        ):
            updates["gender"] = GenderTarget(g_val)

    if slots.fit is not None and slots.fit.source == "explicit":
        updates["fit_preference"] = slots.fit.value

    if (
        slots.measurements is not None
        and slots.measurements.source == "explicit"
        and isinstance(slots.measurements.value, MeasurementsValue)
    ):
        m = slots.measurements.value
        if m.height_cm is not None:
            updates["height_cm"] = m.height_cm
        if m.weight_kg is not None:
            updates["weight_kg"] = m.weight_kg
        if m.waist_cm is not None:
            updates["waist_cm"] = m.waist_cm

    return updates


def get_priority_slots(intent: TurnIntent, slots: ConversationSlots) -> list[str]:
    """Return ordered slot queue by priority (P0 -> P1 -> P2) for a given intent."""
    if intent == TurnIntent.SIZE_ADVICE:
        return ["target_gender", "category", "measurements", "fit"]

    if intent == TurnIntent.BODY_FIT_ADVICE:
        return ["target_gender", "category", "measurements", "occasion", "budget"]

    if intent == TurnIntent.OUTFIT_ADVICE:
        return ["target_gender", "occasion", "category", "budget", "style", "color"]

    cat_val = slots.value_of("category")
    if cat_val is None:
        return ["target_gender", "occasion", "category", "budget", "style"]
    if cat_val in ("tops", "pants"):
        return [
            "target_gender",
            "category_subtype",
            "occasion",
            "budget",
            "fit",
            "color",
        ]
    return ["target_gender", "occasion", "budget", "fit", "color", "style"]


def get_missing_priority_slots(
    intent: TurnIntent, slots: ConversationSlots
) -> list[str]:
    """Return ordered list of missing slots relevant to the active turn intent."""
    if intent == TurnIntent.NON_SHOPPING:
        return []
    queue = get_priority_slots(intent, slots)
    return [slot_name for slot_name in queue if not slots.has(slot_name)]


def _format_slot_display(name: str, val: Any) -> str:
    if name == "budget" and isinstance(val, BudgetValue):
        if val.min_price is not None and val.max_price is not None:
            return f"{val.min_price:,.0f}đ - {val.max_price:,.0f}đ"
        if val.max_price is not None:
            return f"tối đa {val.max_price:,.0f}đ"
        if val.min_price is not None:
            return f"trên {val.min_price:,.0f}đ"
        return val.raw_text or "đã xác định"
    if name == "measurements" and isinstance(val, MeasurementsValue):
        parts: list[str] = []
        if val.height_cm:
            parts.append(f"cao {val.height_cm:.0f}cm")
        if val.weight_kg:
            parts.append(f"nặng {val.weight_kg:.0f}kg")
        if val.waist_cm:
            parts.append(f"eo {val.waist_cm:.0f}cm")
        return ", ".join(parts)
    return str(val)


def format_slots_for_prompt(
    slots: ConversationSlots,
    intent: TurnIntent,
    include_missing: bool = True,
) -> str:
    """Format concise P0-P3 known/missing slot summary for prompt injection."""
    meas_prio = "P0" if intent == TurnIntent.SIZE_ADVICE else "P3"
    slot_labels = {
        "target_gender": ("Giới tính/Đối tượng", "P0"),
        "category": ("Danh mục", "P0"),
        "category_subtype": ("Loại sản phẩm cụ thể", "P0"),
        "occasion": ("Dịp sử dụng", "P0"),
        "budget": ("Ngân sách", "P1"),
        "fit": ("Form dáng", "P1"),
        "size": ("Size", "P1"),
        "color": ("Màu sắc", "P2"),
        "style": ("Phong cách", "P2"),
        "material": ("Chất liệu", "P2"),
        "measurements": ("Số đo cơ thể", meas_prio),
        "body_concern": ("Đặc điểm vóc dáng", "P3"),
    }

    known_items: list[str] = []
    for key in (
        "target_gender",
        "category",
        "occasion",
        "budget",
        "fit",
        "size",
        "color",
        "style",
        "material",
        "measurements",
        "body_concern",
    ):
        if slots.has(key):
            label, prio = slot_labels[key]
            disp = _format_slot_display(key, slots.value_of(key))
            src = getattr(slots, key).source
            known_items.append(f"{label} [{prio}]={disp} ({src})")

    known_str = "; ".join(known_items) if known_items else "Chưa có"
    if not include_missing:
        return f"TÓM TẮT NGỮ CẢNH ({intent.value}):\n- Đã biết: {known_str}"

    known_str = "; ".join(known_items) if known_items else "Chưa có"
    if not include_missing:
        return (
            f"TÓM TẮT NGỮ CẢNH ({intent.value}):\n"
            f"- Đã biết: {known_str}"
        )

    missing = get_missing_priority_slots(intent, slots)
    missing_items = [
        f"{slot_labels[k][0]} [{slot_labels[k][1]}]"
        for k in missing
        if k in slot_labels
    ]
    missing_str = ", ".join(missing_items) if missing_items else "Đủ thông tin cốt lõi"
    return (
        f"TÓM TẮT NGỮ CẢNH ({intent.value}):\n"
        f"- Đã biết: {known_str}\n"
        f"- Còn thiếu (theo thứ tự ưu tiên): {missing_str}"
    )
