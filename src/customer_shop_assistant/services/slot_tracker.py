"""Deterministic multi-turn slot tracker, lifecycle manager, and intent classifier."""

import re
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from customer_shop_assistant.schemas.chat import UserProfile
from customer_shop_assistant.schemas.clarification import SlotConfidence, SlotStatus
from customer_shop_assistant.schemas.shop import GenderTarget

SlotSource = Literal["explicit", "history", "profile", "inferred", "system"]

SESSION_PERSISTENT_SLOTS: frozenset[str] = frozenset(
    {"target_gender", "measurements", "body_concern", "fit", "size"}
)

TASK_SCOPED_SLOTS: frozenset[str] = frozenset(
    {
        "category",
        "category_subtype",
        "top_category",
        "bottom_category",
        "product_id",
        "occasion",
        "budget",
        "color",
        "top_color",
        "bottom_color",
        "disliked_colors",
        "style",
        "material",
    }
)

STANDARD_SLOTS: tuple[str, ...] = (
    "target_gender",
    "category",
    "category_subtype",
    "top_category",
    "bottom_category",
    "product_id",
    "occasion",
    "budget",
    "fit",
    "size",
    "style",
    "color",
    "top_color",
    "bottom_color",
    "disliked_colors",
    "material",
    "measurements",
    "body_concern",
)


class TurnIntent(StrEnum):
    """High-level conversational intent for clarification planning and routing."""

    PRODUCT_SEARCH = "PRODUCT_SEARCH"
    OUTFIT_ADVICE = "OUTFIT_ADVICE"
    BODY_FIT_ADVICE = "BODY_FIT_ADVICE"
    SIZE_ADVICE = "SIZE_ADVICE"
    NON_SHOPPING = "NON_SHOPPING"


class SlotValue(BaseModel):
    """Typed slot payload tracking normalized value, provenance, and 8-state status."""

    value: Any = Field(default=None, description="Normalized slot value")
    source: SlotSource = Field(
        default="explicit",
        description="Origin: explicit, history, profile, inferred, system",
    )
    status: SlotStatus = Field(
        default="PROVIDED",
        description="Official 8-state clarification slot status",
    )
    confidence: SlotConfidence | None = Field(
        default=None, description="Confidence level when status == 'INFERRED'"
    )
    conflict_detail: str | None = Field(
        default=None, description="Description when status == 'CONFLICTING'"
    )

    def is_resolved(self) -> bool:
        """Return True if slot value is settled without needing clarification."""
        if self.value is None:
            return False
        if self.status in ("PROVIDED", "RESOLVED_FROM_CONTEXT"):
            return True
        if self.status == "INFERRED" and self.confidence == "high":
            return True
        return False


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


_BOTTOM_WAIST_CATEGORIES = frozenset({"jeans", "kaki", "short", "pants", "skirt"})


class ConversationSlots(BaseModel):
    """Multi-turn accumulated fashion slots with 8-state tracking and lifecycle."""

    # P0 - Purpose & Core Context
    target_gender: SlotValue | None = None
    category: SlotValue | None = None
    top_category: SlotValue | None = None
    bottom_category: SlotValue | None = None
    product_id: SlotValue | None = None
    occasion: SlotValue | None = None
    shopping_intent: SlotValue | None = None

    # P1 - Shopping Constraints
    budget: SlotValue | None = None
    fit: SlotValue | None = None
    size: SlotValue | None = None

    # P2 - Aesthetic Preferences
    style: SlotValue | None = None
    color: SlotValue | None = None
    top_color: SlotValue | None = None
    bottom_color: SlotValue | None = None
    disliked_colors: SlotValue | None = None
    material: SlotValue | None = None

    # P3 - Advanced Personalization
    measurements: SlotValue | None = None
    body_concern: SlotValue | None = None

    def _raw_slot(self, name: str) -> SlotValue | None:
        slot = getattr(self, name, None)
        return slot if isinstance(slot, SlotValue) else None

    def status_of(self, name: str, intent: TurnIntent | None = None) -> SlotStatus:
        """Return the official SlotStatus for a slot name under an optional intent."""
        if intent is not None and name in get_not_applicable_slots(intent):
            return "NOT_APPLICABLE"
        if name == "category_subtype":
            cat_slot = self._raw_slot("category")
            if cat_slot is None:
                return "UNKNOWN"
            if cat_slot.status in ("DECLINED", "CONFLICTING", "AMBIGUOUS", "UNKNOWN"):
                return cat_slot.status
            if cat_slot.value in ("tops", "pants"):
                return "UNKNOWN"
            return cat_slot.status
        slot = self._raw_slot(name)
        if slot is None:
            return "UNKNOWN"
        return slot.status

    def is_declined(self, name: str) -> bool:
        """Return True when the customer declined or indicated uncertainty for slot."""
        slot = self._raw_slot(name)
        return slot is not None and slot.status == "DECLINED"

    def is_conflicting(self, name: str) -> bool:
        """Return True when contradictory constraints exist for slot in same turn."""
        slot = self._raw_slot(name)
        return slot is not None and slot.status == "CONFLICTING"

    def get_conflicting_slots(self) -> list[str]:
        """Return all slot names currently in CONFLICTING state."""
        return [s for s in STANDARD_SLOTS if self.is_conflicting(s)]

    def get_declined_slots(self) -> list[str]:
        """Return all slot names currently in DECLINED state."""
        return [s for s in STANDARD_SLOTS if self.is_declined(s)]

    def has(self, name: str) -> bool:
        """Return True if the requested slot currently has a resolved value."""
        if name == "category_subtype":
            cat_slot = self._raw_slot("category")
            if cat_slot is None or not cat_slot.is_resolved():
                return False
            return cat_slot.value not in ("tops", "pants")
        slot = self._raw_slot(name)
        if slot is None or not slot.is_resolved():
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

    def has_complete_measurements(self, category: str | None = None) -> bool:
        """Return True if measurements satisfy SIZE_ADVICE minimum requirements."""
        if not self.has("measurements"):
            return False
        meas = self.value_of("measurements")
        if not isinstance(meas, MeasurementsValue):
            return False
        if meas.height_cm is not None and meas.weight_kg is not None:
            return True
        if category in _BOTTOM_WAIST_CATEGORIES and meas.waist_cm is not None:
            return True
        return False

    def value_of(self, name: str) -> Any:
        """Return underlying value if resolved (or present non-conflict), else None."""
        slot = self._raw_slot(name)
        if slot is None:
            return None
        if slot.status in ("DECLINED", "CONFLICTING", "UNKNOWN"):
            return None
        return slot.value

    def reset_task_scoped(self) -> None:
        """Clear all task-scoped slots while preserving session-persistent slots."""
        for field_name in TASK_SCOPED_SLOTS:
            if hasattr(self, field_name):
                setattr(self, field_name, None)
        self.shopping_intent = None

    def to_slot_states_dict(self, intent: TurnIntent) -> dict[str, dict[str, Any]]:
        """Serialize snapshot of all standard slot states for telemetry logging."""
        not_applicable = get_not_applicable_slots(intent)
        snapshot: dict[str, dict[str, Any]] = {}
        for name in STANDARD_SLOTS:
            if name in not_applicable:
                snapshot[name] = {
                    "value": None,
                    "state": "NOT_APPLICABLE",
                    "source": "system",
                }
                continue
            if name == "category_subtype":
                cat_slot = self._raw_slot("category")
                sub_val = (
                    cat_slot.value
                    if cat_slot
                    and cat_slot.is_resolved()
                    and cat_slot.value not in ("tops", "pants")
                    else None
                )
                snapshot[name] = {
                    "value": sub_val,
                    "state": self.status_of(name, intent),
                    "source": cat_slot.source if cat_slot else "system",
                }
                continue
            slot = self._raw_slot(name)
            if slot is None:
                snapshot[name] = {
                    "value": None,
                    "state": "UNKNOWN",
                    "source": "system",
                }
            else:
                val = slot.value
                if isinstance(val, BaseModel):
                    val = val.model_dump()
                snapshot[name] = {
                    "value": val,
                    "state": slot.status,
                    "source": slot.source,
                }
        return snapshot

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
            "missing_required_slots": get_missing_required_slots(intent, self),
            "missing_priority_slots": get_missing_priority_slots(intent, self),
            "conflicting_slots": self.get_conflicting_slots(),
            "declined_slots": self.get_declined_slots(),
            "slot_states": self.to_slot_states_dict(intent),
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
    "phối cùng",
    "phối giúp",
    "mix đồ",
    "mix & match",
    "mix match",
    "set đồ",
    "bộ đồ",
    "1 bộ",
    "một bộ",
    "nguyên bộ",
    "1 set",
    "một set",
    "cả set",
    "combo",
    "outfit",
    "mặc với gì",
    "mặc cùng",
    "mặc chung với",
    "kết hợp với",
    "nguyên set",
    "cả bộ",
    "bên dưới mặc gì",
    "ở dưới mặc gì",
    "bên trên mặc gì",
    "ở trên mặc gì",
    "đồ mặc bên dưới",
    "đồ bên dưới",
    "mặc bên dưới",
    "phần dưới",
    "đồ mặc bên trên",
    "đồ bên trên",
    "mặc bên trên",
    "phần trên",
    "phù hợp với áo",
    "hợp với áo",
    "mặc với áo",
    "đi với áo",
    "phù hợp với quần",
    "hợp với quần",
    "mặc với quần",
    "đi với quần",
    "phù hợp với váy",
    "hợp với váy",
    "mặc với váy",
    "đi với váy",
    "phù hợp với mẫu",
    "hợp với mẫu",
    "mặc với mẫu",
    "mặc thêm gì",
    "phối thêm gì",
)

_OUTFIT_ADVICE_PATTERNS = (
    r"\bmặc\s+(?:gì|đồ\s+gì|loại\s+gì)\s+(?:ở\s+|bên\s+)?(?:dưới|trên)\b",
    r"\b(?:ở\s+|bên\s+)?(?:dưới|trên)\s+(?:nên\s+)?mặc\s+gì\b",
    r"\b(?:phù\s+hợp|hợp)\s+với\s+(?:mấy\s+|các\s+|những\s+)?"
    r"(?:mẫu\s+)?(?:áo|quần|váy|chân\s+váy)\b",
    r"\bphối\s+(?:giúp|cho)\b",
    r"\bphối\b.+\bvới\b",
    r"\bphối\s+(?:áo|quần|váy|chân\s+váy|đầm)\b",
    r"\b(?:áo|sơ\s+mi|polo|blazer|hoodie|bomber)\b.+\b"
    r"(?:chọn|mặc|phối|dùng|lấy|đi)\s+"
    r"(?:loại\s+|kiểu\s+|mẫu\s+|dáng\s+)?(?:quần|váy|chân\s+váy)\s*(?:nào|gì)\b",
    r"\b(?:quần|váy|chân\s+váy)\b.+\b"
    r"(?:chọn|mặc|phối|dùng|lấy|đi)\s+"
    r"(?:loại\s+|kiểu\s+|mẫu\s+|dáng\s+)?áo\s*(?:nào|gì)\b",
)

_SINGLE_ITEM_SWITCH_CUES = (
    "tìm áo",
    "tìm quần",
    "tìm đầm",
    "tìm váy",
    "tìm cho tôi loại",
    "tìm mẫu",
    "xem mẫu",
    "chỉ tìm áo",
    "chỉ tìm quần",
)

_EXPLICIT_TOPIC_SWITCH_CUES = (
    "giờ tìm",
    "giờ phối",
    "giờ chọn",
    "giờ xem",
    "đổi sang",
    "chuyển sang",
    "thôi tìm",
    "thôi đổi",
    "thôi xem",
    "sang tìm",
    "bỏ qua cái cũ",
    "nhu cầu khác",
    "món khác",
    "tìm thêm bộ khác",
)

_CONTINUATION_CUES = (
    "phù hợp",
    "hợp với",
    "đi kèm",
    "cùng bộ",
    "mặc cùng",
    "mặc với",
    "mặc bên",
    "bên dưới",
    "bên trên",
    "phối với",
    "phối cùng",
    "tiếp tục",
    "như trên",
    "mình chọn:",
    "xác nhận",
)

_GENERAL_DECLINE_CUES = (
    "không biết",
    "ko biết",
    "k biết",
    "chưa biết",
    "không rõ",
    "ko rõ",
    "sao cũng được",
    "thế nào cũng được",
    "gì cũng được",
    "nào cũng được",
    "tùy shop",
    "tuỳ shop",
    "tùy bạn",
    "tuỳ bạn",
    "gợi ý đại",
    "chọn đại",
    "bỏ qua",
    "không quan trọng",
    "không nhớ",
    "chưa đo",
)

_SLOT_SPECIFIC_DECLINE_CUES: dict[str, tuple[str, ...]] = {
    "measurements": (
        "không nhớ chiều cao",
        "không nhớ cân nặng",
        "không rõ số đo",
        "chưa đo",
        "không nhớ số đo",
        "quên số đo",
        "không biết chiều cao",
        "không biết cân nặng",
        "không biết số đo",
    ),
    "target_gender": (
        "nam nữ gì cũng được",
        "nam hay nữ đều được",
        "giới tính nào cũng được",
    ),
    "budget": (
        "giá nào cũng được",
        "không giới hạn giá",
        "ngân sách nào cũng được",
        "bao nhiêu cũng được",
        "tầm giá nào cũng được",
    ),
    "occasion": (
        "dịp nào cũng được",
        "mặc lúc nào cũng được",
    ),
    "category": (
        "loại nào cũng được",
        "món nào cũng được",
    ),
}

_TARGET_REFERENCE_SPLIT_CUES = (
    "phù hợp với",
    "hợp với",
    "đi cùng với",
    "đi cùng",
    "đi với",
    "mặc cùng với",
    "mặc cùng",
    "mặc chung với",
    "mặc với",
    "phối cùng",
    "phối với",
    "kết hợp với",
)

_GENERIC_BOTTOM_CUES = (
    "đồ mặc bên dưới",
    "đồ bên dưới",
    "mặc bên dưới",
    "phần dưới",
    "đồ dưới",
)

_GENERIC_TOP_CUES = (
    "đồ mặc bên trên",
    "đồ bên trên",
    "mặc bên trên",
    "phần trên",
    "đồ trên",
)

_FORM_MULTI_GARMENT_PHRASES = (
    "áo (sơ mi",
    "áo nam (áo",
    "áo nữ (",
    "áo sơ mi / áo polo",
    "áo thun / áo polo",
    "áo thun / áo khoác",
    "áo sơ mi / áo thun",
    "quần (quần âu",
    "quần nam (quần",
    "quần nữ (",
    "quần âu / quần kaki",
    "quần jeans / quần short",
    "quần âu / quần jeans",
    "đầm / váy",
    "đầm công sở liền thân",
    "đầm dạo phố",
)

_TOP_COMPONENT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("áo sơ mi", "shirt"),
    ("sơ mi", "shirt"),
    ("áo polo", "polo"),
    ("polo", "polo"),
    ("áo thun", "tshirt"),
    ("áo phông", "tshirt"),
    ("t-shirt", "tshirt"),
    ("tshirt", "tshirt"),
    ("áo khoác", "jacket"),
    ("blazer", "jacket"),
    ("hoodie", "jacket"),
    ("bomber", "jacket"),
    ("jacket", "jacket"),
)

_BOTTOM_COMPONENT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("chân váy", "skirt"),
    ("skirt", "skirt"),
    ("quần jeans", "jeans"),
    ("quần jean", "jeans"),
    ("quần bò", "jeans"),
    ("jeans", "jeans"),
    ("jean", "jeans"),
    ("quần kaki", "kaki"),
    ("kaki", "kaki"),
    ("khaki", "kaki"),
    ("quần short", "short"),
    ("quần đùi", "short"),
    ("short", "short"),
    ("quần âu", "pants"),
    ("quần tây", "pants"),
)

_FEMALE_INFERRED_CATEGORIES = {"dress", "skirt"}
_SPECIFIC_TOP_CATEGORIES = {"shirt", "polo", "tshirt", "jacket"}
_SPECIFIC_PANT_CATEGORIES = {"jeans", "kaki", "short"}

_FEMALE_EXPLICIT_PHRASES = (
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
    "đầm nữ",
    "váy nữ",
)

_MALE_EXPLICIT_PHRASES = (
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

_UNISEX_PHRASES = (
    "unisex",
    "cả nam và nữ",
    "nam nữ đều",
    "nam nữ",
    "couple",
)

_COLOR_LIST: tuple[tuple[str, str], ...] = (
    ("xanh rêu khaki", "Xanh rêu khaki"),
    ("xanh lam nhạt", "Xanh lam nhạt"),
    ("xanh cổ vịt", "Xanh cổ vịt"),
    ("xanh ngọc bích", "Xanh ngọc bích"),
    ("xanh ngọc", "Xanh ngọc"),
    ("xanh pastel", "Xanh pastel"),
    ("xanh dương", "Xanh dương"),
    ("xanh navy", "Xanh navy"),
    ("xanh rêu", "Xanh rêu khaki"),
    ("xanh mint", "Xanh mint"),
    ("xanh lá", "Xanh lá"),
    ("trắng kem", "Trắng kem"),
    ("be sữa", "Be sữa"),
    ("be sáng", "Be sáng"),
    ("nâu camel", "Nâu camel"),
    ("nâu be", "Nâu be"),
    ("xám tiêu", "Xám tiêu"),
    ("xám đậm", "Xám đậm"),
    ("xám nhạt", "Xám nhạt"),
    ("hồng pastel", "Hồng pastel"),
    ("hồng đất", "Hồng đất"),
    ("đỏ đô", "Đỏ đô"),
    ("đỏ tươi", "Đỏ tươi"),
    ("cam đào", "Cam đào"),
    ("cam đất", "Cam đất"),
    ("vàng mù tạt", "Vàng mù tạt"),
    ("vàng kem", "Vàng kem"),
    ("vàng ánh kim", "Vàng ánh kim"),
    ("tím than", "Tím than"),
    ("tím lilac", "Tím lilac"),
    ("trắng", "Trắng"),
    ("đen", "Đen"),
    ("navy", "Xanh navy"),
    ("xanh", "Xanh"),
    ("be", "Be"),
    ("nâu", "Nâu"),
    ("xám", "Xám"),
    ("ghi", "Xám ghi"),
    ("đỏ", "Đỏ"),
    ("hồng", "Hồng"),
    ("vàng", "Vàng"),
    ("cam", "Cam"),
    ("tím", "Tím"),
    ("bạc", "Bạc"),
    ("trung tính", "Trung tính"),
    ("màu tối", "Tối màu"),
    ("màu sáng", "Sáng màu"),
)


def _has_outfit_advice_cue(lower: str) -> bool:
    return any(cue in lower for cue in _OUTFIT_ADVICE_CUES) or any(
        re.search(pat, lower) for pat in _OUTFIT_ADVICE_PATTERNS
    )


def classify_turn_intent(message: str) -> TurnIntent:
    """Classify a single customer turn into one of the 5 system intents."""
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
            k in lower
            for k in (
                "size gì",
                "size nào",
                "bảng size",
                "tư vấn size",
                "khung size",
            )
        ):
            return TurnIntent.BODY_FIT_ADVICE
        return TurnIntent.SIZE_ADVICE

    if any(cue in lower for cue in _BODY_FIT_CUES):
        return TurnIntent.BODY_FIT_ADVICE

    if _has_outfit_advice_cue(lower):
        return TurnIntent.OUTFIT_ADVICE

    top_cat, bot_cat, _, _ = _extract_outfit_components(lower)
    if top_cat is not None and bot_cat is not None:
        return TurnIntent.OUTFIT_ADVICE

    if not has_shopping_cue and _detect_decline_cues(lower)[0]:
        return TurnIntent.PRODUCT_SEARCH

    return TurnIntent.PRODUCT_SEARCH


def _parse_price_token(num_str: str, unit: str) -> float | None:
    cleaned = num_str.replace(",", ".").strip()
    if cleaned.count(".") >= 1 and unit in ("", "đ", "d", "vnd", "vnđ"):
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
    range_match = re.search(
        r"(?:từ\s*)?(\d+(?:[.,]\d+)?)\s*(?:(k|nghìn|ngàn|triệu|tr|đ|vnđ|vnd)\b)?\s*"
        r"(?:-|–|đến|tới)\s*(\d+(?:[.,]\d+)?)\s*"
        r"(?:(k|nghìn|ngàn|triệu|tr|đ|vnđ|vnd)\b)?",
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


def _detect_budget_conflict(lower: str) -> tuple[bool, str | None]:
    """Detect contradictory budget bounds within the same utterance."""
    under_match = re.search(
        r"(?:dưới|tối đa|không quá|<=|<)\s*"
        r"(\d+(?:[.,]\d+)?)\s*(k|nghìn|ngàn|triệu|tr|đ|vnd)\b",
        lower,
    )
    over_match = re.search(
        r"(?:trên|tối thiểu|>=|>)\s*"
        r"(\d+(?:[.,]\d+)?)\s*(k|nghìn|ngàn|triệu|tr|đ|vnd)\b",
        lower,
    )
    if under_match and over_match:
        max_p = _parse_price_token(under_match.group(1), under_match.group(2))
        min_p = _parse_price_token(over_match.group(1), over_match.group(2))
        if max_p is not None and min_p is not None and min_p > max_p:
            return (
                True,
                f"Ngân sách mâu thuẫn: tối đa {max_p:,.0f}đ nhưng yêu cầu trên "
                f"{min_p:,.0f}đ.",
            )

    extracted = _extract_budget(lower)
    if (
        extracted is not None
        and extracted.min_price is not None
        and extracted.max_price is not None
        and extracted.min_price > extracted.max_price
    ):
        return (
            True,
            f"Khoảng giá mâu thuẫn: từ {extracted.min_price:,.0f}đ đến "
            f"{extracted.max_price:,.0f}đ.",
        )

    return False, None


def _has_explicit_female_cue(lower: str) -> bool:
    return any(p in lower for p in _FEMALE_EXPLICIT_PHRASES) or bool(
        re.search(r"\b(?:nữ|vợ)\b", lower)
    )


def _has_explicit_male_cue(lower: str) -> bool:
    return any(p in lower for p in _MALE_EXPLICIT_PHRASES) or bool(
        re.search(r"\b(?:nam|chồng|boy|men)\b", lower)
    )


def _extract_gender(lower: str) -> str | None:
    if any(cue in lower for cue in _UNISEX_PHRASES):
        return GenderTarget.UNISEX.value

    if _has_explicit_female_cue(lower):
        return GenderTarget.FEMALE.value

    if _has_explicit_male_cue(lower):
        return GenderTarget.MALE.value

    return None


def _detect_gender_conflict(lower: str) -> tuple[bool, str | None]:
    """Detect contradictory gender constraints within the same utterance."""
    if any(cue in lower for cue in _UNISEX_PHRASES):
        return False, None

    has_male = _has_explicit_male_cue(lower)
    has_female = _has_explicit_female_cue(lower)
    if has_male and has_female:
        return True, "Yêu cầu chứa đồng thời cả đối tượng Nam và Nữ."

    if has_male:
        cat = _extract_category(lower)
        _, bot_cat, _, _ = _extract_outfit_components(lower)
        if cat in _FEMALE_INFERRED_CATEGORIES or bot_cat == "skirt":
            return (
                True,
                "Yêu cầu kết hợp trang phục đặc thù Nữ (đầm/chân váy) "
                "với đối tượng Nam.",
            )

    return False, None


def _extract_category_from_text(text: str) -> str | None:
    if any(
        phrase in text
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
        phrase in text
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
        phrase in text
        for phrase in (
            "đầm / váy",
            "đầm công sở liền thân",
            "đầm dạo phố",
        )
    ):
        return "dress"

    if any(k in text for k in ("chân váy", "skirt")):
        return "skirt"
    if any(k in text for k in ("đầm", "váy", "dress")):
        return "dress"
    if "sơ mi" in text:
        return "shirt"
    if "polo" in text:
        return "polo"
    if any(k in text for k in ("áo thun", "t-shirt", "tshirt", "áo phông")):
        return "tshirt"
    if any(k in text for k in ("áo khoác", "blazer", "hoodie", "bomber", "jacket")):
        return "jacket"
    if any(k in text for k in ("jeans", "jean", "bò")):
        return "jeans"
    if "kaki" in text or "khaki" in text:
        return "kaki"
    if "short" in text or "quần đùi" in text:
        return "short"
    if any(k in text for k in ("quần âu", "quần tây")):
        return "pants"
    if any(k in text for k in ("thắt lưng", "túi xách", "mũ", "nón", "phụ kiện")):
        return "accessory"
    if any(k in text for k in _GENERIC_BOTTOM_CUES):
        return "pants"
    if any(k in text for k in _GENERIC_TOP_CUES):
        return "tops"
    if re.search(r"\bquần\b", text):
        return "pants"
    if re.search(r"\báo\b", text):
        return "tops"
    return None


def _extract_category(lower: str) -> str | None:
    """Extract target category, prioritizing target clause before pairing cues."""
    split_positions = [
        pos for cue in _TARGET_REFERENCE_SPLIT_CUES if (pos := lower.find(cue)) > 0
    ]
    if split_positions:
        target_clause = lower[: min(split_positions)].strip()
        if target_clause:
            target_cat = _extract_category_from_text(target_clause)
            if target_cat is not None:
                return target_cat
    return _extract_category_from_text(lower)


def _extract_product_id(lower: str) -> int | None:
    match = re.search(
        r"(?:\[(?:id\s*:?\s*)?(\d+)\]|"
        r"(?:sản\s*phẩm\s*(?:id|mã|#)?|mã\s*(?:sản\s*phẩm|sp)|(?<!\w)id)"
        r"\s*[:#]?\s*(\d+))",
        lower,
    )
    if match:
        raw_num = match.group(1) or match.group(2)
        if raw_num:
            return int(raw_num)
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

    m_h_range = re.search(
        r"(\d)m(\d{1,2})\s*(?:-|–|đến|tới)\s*(\d)m(\d{1,2})",
        lower,
    )
    if m_h_range:
        c1 = int(m_h_range.group(2))
        c2 = int(m_h_range.group(4))
        h1 = float(int(m_h_range.group(1)) * 100 + (c1 * 10 if c1 < 10 else c1))
        h2 = float(int(m_h_range.group(3)) * 100 + (c2 * 10 if c2 < 10 else c2))
        if 100 <= h1 <= 230 and 100 <= h2 <= 230:
            height_cm = round((h1 + h2) / 2, 1)

    if height_cm is None:
        m_h = re.search(
            r"(?:cao\s*)?(?:(\d)m(\d{1,2})|(\d{3})\s*cm|cao\s*(\d{3}))", lower
        )
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

    m_w_range = re.search(
        r"\b(\d{2,3})\s*(?:-|–|đến|tới)\s*(\d{2,3})\s*(?:kg|ký|kí|cân)\b",
        lower,
    )
    if m_w_range:
        w1 = float(m_w_range.group(1))
        w2 = float(m_w_range.group(2))
        if 30 <= w1 <= 200 and 30 <= w2 <= 200:
            weight_kg = round((w1 + w2) / 2, 1)

    if weight_kg is None:
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
    for cue, label in _COLOR_LIST:
        if re.search(rf"\b{re.escape(cue)}\b", lower):
            return label
    return None


_NEGATED_COLOR_PHRASE = re.compile(
    r"(?:không\s+thích|tránh|đừng\s+(?:lấy|chọn|gợi\s+ý)|ghét|bỏ)\s+(?:màu\s+)?"
    r"([a-zA-ZÀ-ỹ\s,]+?)(?:[.;!?]|$|\bnhưng\b|\bmà\b|\bcho\b|\bvà\b)",
    re.IGNORECASE,
)

_AMBIGUOUS_COLOR_CUES = (
    "màu sáng hay tối",
    "màu sáng hay màu tối",
    "màu tối hay sáng",
    "màu tối hay màu sáng",
    "màu nào nổi bật",
    "màu nào tôn da",
    "màu nào hợp da",
    "chọn màu nào",
    "chọn màu gì",
    "tông màu nào",
    "phân vân màu",
    "phân vân về màu",
    "chưa biết chọn màu",
    "nên mặc màu gì",
    "mặc màu gì đẹp",
)


def _extract_disliked_colors(lower: str) -> list[str]:
    """Extract explicitly disliked colors from negative preference phrases."""
    disliked: list[str] = []
    for match in _NEGATED_COLOR_PHRASE.finditer(lower):
        segment = match.group(1).strip().lower()
        for cue, label in _COLOR_LIST:
            if re.search(rf"\b{re.escape(cue)}\b", segment) and label not in disliked:
                disliked.append(label)
                segment = re.sub(rf"\b{re.escape(cue)}\b", "", segment)
    return disliked


def _extract_outfit_components(
    lower: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Extract independent (top_cat, bottom_cat, top_col, bottom_col) from text."""
    if any(phrase in lower for phrase in _FORM_MULTI_GARMENT_PHRASES):
        return None, None, None, None

    top_cat: str | None = None
    top_idx: int | None = None
    for cue, code in _TOP_COMPONENT_PATTERNS:
        pos = lower.find(cue)
        if pos != -1 and (top_idx is None or pos < top_idx):
            top_cat = code
            top_idx = pos

    bot_cat: str | None = None
    bot_idx: int | None = None
    for cue, code in _BOTTOM_COMPONENT_PATTERNS:
        pos = lower.find(cue)
        if pos != -1 and (bot_idx is None or pos < bot_idx):
            bot_cat = code
            bot_idx = pos

    top_col: str | None = None
    bot_col: str | None = None
    if top_idx is not None and bot_idx is not None:
        split_at = max(top_idx, bot_idx)
        seg1 = lower[:split_at]
        seg2 = lower[split_at:]
        if top_idx < bot_idx:
            top_col = _extract_color(seg1)
            bot_col = _extract_color(seg2)
        else:
            bot_col = _extract_color(seg1)
            top_col = _extract_color(seg2)
    elif top_idx is not None:
        top_col = _extract_color(lower)
    elif bot_idx is not None:
        bot_col = _extract_color(lower)

    return top_cat, bot_cat, top_col, bot_col


def _extract_style(lower: str) -> str | None:
    styles = (
        ("smart-casual", "Smart Casual"),
        ("smart casual", "Smart Casual"),
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
    body_text = "\n".join(
        line for line in lower.splitlines() if "khung size" not in line
    )
    if any(k in body_text for k in ("gầy", "đỡ gầy")):
        return "slim"
    if any(k in body_text for k in ("che bụng", "có bụng", "béo bụng", "đầy đặn")):
        return "belly"
    if any(
        k in body_text
        for k in (
            "hack chiều cao",
            "hack dáng",
            "chiều cao khiêm tốn",
            "nấm lùn",
            "trông cao hơn",
            "thấp",
        )
    ):
        return "petite"
    return None


def _detect_decline_cues(lower: str) -> tuple[bool, list[str]]:
    """Detect general uncertainty/refusal and slot-specific decline phrases."""
    specific: list[str] = []
    for slot_name, cues in _SLOT_SPECIFIC_DECLINE_CUES.items():
        if any(cue in lower for cue in cues):
            specific.append(slot_name)

    has_general = bool(specific) or any(cue in lower for cue in _GENERAL_DECLINE_CUES)
    return has_general, specific


def detect_topic_switch(
    prev_slots: ConversationSlots,
    prev_intent: TurnIntent | None,
    current_text: str,
    current_intent: TurnIntent,
) -> bool:
    """Deterministically detect whether the customer switched to a new shopping task."""
    if prev_intent is None or current_intent == TurnIntent.NON_SHOPPING:
        return False

    lower = current_text.strip().lower()
    if not lower:
        return False

    if any(cue in lower for cue in _EXPLICIT_TOPIC_SWITCH_CUES):
        return True

    if any(cue in lower for cue in _CONTINUATION_CUES):
        return False

    prev_occ = prev_slots.value_of("occasion")
    curr_occ = _extract_occasion(lower)
    if prev_occ is not None and curr_occ is not None and curr_occ != prev_occ:
        return True

    if (
        prev_intent == TurnIntent.OUTFIT_ADVICE
        and current_intent == TurnIntent.PRODUCT_SEARCH
        and any(cue in lower for cue in _SINGLE_ITEM_SWITCH_CUES)
    ):
        return True

    return False


def _apply_turn_extractions(
    slots: ConversationSlots,
    lower: str,
    *,
    status: SlotStatus,
    source: SlotSource,
    check_conflicts: bool,
) -> None:
    """Extract and merge slot values from a single user message."""
    gender_conflict = False
    gender_detail: str | None = None
    if check_conflicts:
        gender_conflict, gender_detail = _detect_gender_conflict(lower)

    if gender_conflict:
        slots.target_gender = SlotValue(
            value=None,
            source=source,
            status="CONFLICTING",
            conflict_detail=gender_detail,
        )
    else:
        gender = _extract_gender(lower)
        if gender:
            slots.target_gender = SlotValue(value=gender, source=source, status=status)

    is_outfit_turn = _has_outfit_advice_cue(lower)

    category = _extract_category(lower)
    if category and not (is_outfit_turn and category in ("tops", "pants")):
        current_cat = slots.value_of("category")
        is_same_group_downgrade = (
            current_cat in _SPECIFIC_TOP_CATEGORIES and category == "tops"
        ) or (current_cat in _SPECIFIC_PANT_CATEGORIES and category == "pants")
        if not is_same_group_downgrade:
            slots.category = SlotValue(value=category, source=source, status=status)

    prod_id = _extract_product_id(lower)
    if prod_id is not None:
        slots.product_id = SlotValue(value=prod_id, source=source, status=status)

    occasion = _extract_occasion(lower)
    if occasion:
        slots.occasion = SlotValue(value=occasion, source=source, status=status)

    if is_outfit_turn:
        slots.shopping_intent = SlotValue(value="OUTFIT", source=source, status=status)
    elif category and not slots.has("shopping_intent"):
        slots.shopping_intent = SlotValue(value="PRODUCT", source=source, status=status)

    budget_conflict = False
    budget_detail: str | None = None
    if check_conflicts:
        budget_conflict, budget_detail = _detect_budget_conflict(lower)

    if budget_conflict:
        slots.budget = SlotValue(
            value=None,
            source=source,
            status="CONFLICTING",
            conflict_detail=budget_detail,
        )
    else:
        budget = _extract_budget(lower)
        if budget:
            slots.budget = SlotValue(value=budget, source=source, status=status)

    fit = _extract_fit(lower)
    if fit:
        slots.fit = SlotValue(value=fit, source=source, status=status)

    size_lbl = _extract_size_label(lower)
    if size_lbl:
        slots.size = SlotValue(value=size_lbl, source=source, status=status)

    disliked = _extract_disliked_colors(lower)
    if disliked:
        slots.disliked_colors = SlotValue(value=disliked, source=source, status=status)

    pos_lower = _NEGATED_COLOR_PHRASE.sub(" ", lower)
    color = _extract_color(pos_lower)
    top_cat, bot_cat, top_col, bot_col = _extract_outfit_components(pos_lower)

    explicit_cols = [c for c in (color, top_col, bot_col) if c is not None]
    conflicting_col = next((c for c in explicit_cols if c in disliked), None)
    is_ambiguous_phrase = is_outfit_turn and any(
        cue in pos_lower for cue in _AMBIGUOUS_COLOR_CUES
    )
    if check_conflicts and conflicting_col is not None:
        slots.color = SlotValue(
            value=None,
            source=source,
            status="CONFLICTING",
            conflict_detail=(
                f"Màu '{conflicting_col}' vừa được yêu cầu vừa không thích."
            ),
        )
    elif is_ambiguous_phrase:
        slots.color = SlotValue(value=None, source=source, status="AMBIGUOUS")
        top_col = None
        bot_col = None
    elif color and color not in disliked:
        was_conflicting = (
            slots.color is not None and slots.color.status == "CONFLICTING"
        )
        slots.color = SlotValue(value=color, source=source, status=status)
        if top_cat is None and bot_cat is None and status == "PROVIDED":
            if slots.top_color is not None and (
                was_conflicting or slots.top_color.value in disliked or top_col is None
            ):
                if slots.bottom_color is None:
                    slots.top_color = SlotValue(
                        value=color, source=source, status=status
                    )
            elif (
                slots.bottom_color is not None
                and slots.top_color is None
                and bot_col is None
            ):
                slots.bottom_color = SlotValue(
                    value=color, source=source, status=status
                )

    if top_cat:
        slots.top_category = SlotValue(value=top_cat, source=source, status=status)
    if bot_cat:
        slots.bottom_category = SlotValue(value=bot_cat, source=source, status=status)
    if top_col and top_col not in disliked:
        slots.top_color = SlotValue(value=top_col, source=source, status=status)
    if bot_col and bot_col not in disliked:
        slots.bottom_color = SlotValue(value=bot_col, source=source, status=status)

    style = _extract_style(lower)
    if style:
        slots.style = SlotValue(value=style, source=source, status=status)

    material = _extract_material(lower)
    if material:
        slots.material = SlotValue(value=material, source=source, status=status)

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
        slots.measurements = SlotValue(value=merged, source=source, status=status)

    concern = _extract_body_concern(lower)
    if concern:
        slots.body_concern = SlotValue(value=concern, source=source, status=status)


def extract_conversation_slots_with_lifecycle(
    messages: Sequence[BaseMessage | str],
    user_profile: UserProfile | None = None,
    *,
    declined_slots: set[str] | None = None,
    last_asked_slots: Sequence[str] | None = None,
) -> tuple[ConversationSlots, TurnIntent, bool]:
    """Accumulate slots with 8-state provenance, precedence, and topic-switch reset."""
    slots = ConversationSlots()

    # 1. Seed from user profile (lowest priority among non-inferred sources)
    if user_profile:
        if user_profile.gender:
            slots.target_gender = SlotValue(
                value=user_profile.gender.value,
                source="profile",
                status="RESOLVED_FROM_CONTEXT",
            )
        if user_profile.fit_preference:
            slots.fit = SlotValue(
                value=user_profile.fit_preference.upper(),
                source="profile",
                status="RESOLVED_FROM_CONTEXT",
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
                status="RESOLVED_FROM_CONTEXT",
            )

    user_texts: list[str] = []
    for msg in messages:
        if isinstance(msg, str):
            user_texts.append(msg)
        elif isinstance(msg, HumanMessage):
            user_texts.append(str(msg.content or ""))

    running_intent: TurnIntent | None = None
    latest_resolved_intent: TurnIntent | None = None
    topic_switched = False

    # 2. Extract chronologically across history and current turn
    for idx, text in enumerate(user_texts):
        lower = text.strip().lower()
        if not lower:
            continue

        is_latest = idx == len(user_texts) - 1
        is_form_submit = lower.startswith("mình chọn:")
        turn_intent = classify_turn_intent(text)
        if (
            turn_intent == TurnIntent.PRODUCT_SEARCH
            and slots.value_of("shopping_intent") == "OUTFIT"
            and not any(cue in lower for cue in _SINGLE_ITEM_SWITCH_CUES)
            and not any(cue in lower for cue in _EXPLICIT_TOPIC_SWITCH_CUES)
        ):
            turn_intent = TurnIntent.OUTFIT_ADVICE
        elif (
            running_intent == TurnIntent.SIZE_ADVICE
            and turn_intent in (TurnIntent.PRODUCT_SEARCH, TurnIntent.BODY_FIT_ADVICE)
            and (
                is_form_submit
                or (
                    turn_intent == TurnIntent.PRODUCT_SEARCH
                    and not any(cue in lower for cue in _SINGLE_ITEM_SWITCH_CUES)
                    and not any(cue in lower for cue in _EXPLICIT_TOPIC_SWITCH_CUES)
                    and not any(cue in lower for cue in _INVENTORY_STOCK_CUES)
                )
            )
        ):
            turn_intent = TurnIntent.SIZE_ADVICE
        elif (
            running_intent == TurnIntent.BODY_FIT_ADVICE
            and turn_intent == TurnIntent.PRODUCT_SEARCH
            and is_form_submit
        ):
            turn_intent = TurnIntent.BODY_FIT_ADVICE

        switched = detect_topic_switch(slots, running_intent, text, turn_intent)
        if switched:
            slots.reset_task_scoped()
            if is_latest:
                topic_switched = True

        if turn_intent != TurnIntent.NON_SHOPPING:
            running_intent = turn_intent
        if is_latest:
            latest_resolved_intent = turn_intent

        turn_status: SlotStatus = "PROVIDED" if is_latest else "RESOLVED_FROM_CONTEXT"
        turn_source: SlotSource = "explicit" if is_latest else "history"

        _apply_turn_extractions(
            slots,
            lower,
            status=turn_status,
            source=turn_source,
            check_conflicts=is_latest,
        )

    latest_text = user_texts[-1] if user_texts else ""
    latest_lower = latest_text.strip().lower()
    intent = (
        latest_resolved_intent
        if latest_resolved_intent is not None
        else classify_turn_intent(latest_text)
    )

    # 3. Apply persisted declined_slots (unless topic switched or provided now)
    if declined_slots and not topic_switched:
        for d_slot in declined_slots:
            if hasattr(slots, d_slot):
                curr = getattr(slots, d_slot)
                if curr is None or curr.status != "PROVIDED":
                    setattr(
                        slots,
                        d_slot,
                        SlotValue(value=None, source="history", status="DECLINED"),
                    )

    # 4. Detect newly declined slots in latest user turn
    has_decline, specific_declines = _detect_decline_cues(latest_lower)
    if has_decline and intent != TurnIntent.NON_SHOPPING:
        for d_slot in specific_declines:
            if hasattr(slots, d_slot):
                curr = getattr(slots, d_slot)
                if curr is None or not curr.is_resolved() or curr.status != "PROVIDED":
                    setattr(
                        slots,
                        d_slot,
                        SlotValue(value=None, source="explicit", status="DECLINED"),
                    )
        if last_asked_slots:
            for asked_slot in last_asked_slots:
                if hasattr(slots, asked_slot):
                    curr = getattr(slots, asked_slot)
                    if curr is None or curr.status != "PROVIDED":
                        setattr(
                            slots,
                            asked_slot,
                            SlotValue(value=None, source="explicit", status="DECLINED"),
                        )

    # 5. High-confidence inference ONLY for unmistakably female categories
    if (
        not slots.has("target_gender")
        and not slots.is_declined("target_gender")
        and not slots.is_conflicting("target_gender")
        and (
            slots.value_of("category") in _FEMALE_INFERRED_CATEGORIES
            or slots.value_of("bottom_category") == "skirt"
        )
    ):
        slots.target_gender = SlotValue(
            value=GenderTarget.FEMALE.value,
            source="inferred",
            status="INFERRED",
            confidence="high",
        )

    if intent == TurnIntent.OUTFIT_ADVICE:
        slots.shopping_intent = SlotValue(
            value="OUTFIT", source="explicit", status="PROVIDED"
        )
        if (
            not slots.has("top_category")
            and slots.value_of("category") in _SPECIFIC_TOP_CATEGORIES
        ):
            slots.top_category = SlotValue(
                value=slots.value_of("category"),
                source="inferred",
                status="INFERRED",
                confidence="high",
            )
        if not slots.has("bottom_category") and slots.value_of("category") in (
            *_SPECIFIC_PANT_CATEGORIES,
            "skirt",
        ):
            slots.bottom_category = SlotValue(
                value=slots.value_of("category"),
                source="inferred",
                status="INFERRED",
                confidence="high",
            )
    elif intent == TurnIntent.PRODUCT_SEARCH:
        slots.shopping_intent = SlotValue(
            value="PRODUCT", source="explicit", status="PROVIDED"
        )
    elif not slots.has("shopping_intent") and intent != TurnIntent.NON_SHOPPING:
        slots.shopping_intent = SlotValue(
            value="PRODUCT",
            source="inferred",
            status="INFERRED",
            confidence="high",
        )

    return slots, intent, topic_switched


def extract_conversation_slots(
    messages: Sequence[BaseMessage | str],
    user_profile: UserProfile | None = None,
    *,
    declined_slots: set[str] | None = None,
    last_asked_slots: Sequence[str] | None = None,
) -> tuple[ConversationSlots, TurnIntent]:
    """Accumulate slots across conversation history and classify latest turn."""
    slots, intent, _ = extract_conversation_slots_with_lifecycle(
        messages,
        user_profile,
        declined_slots=declined_slots,
        last_asked_slots=last_asked_slots,
    )
    return slots, intent


def extract_session_profile_updates(slots: ConversationSlots) -> dict[str, Any]:
    """Build safe session-scoped profile updates from confirmed/explicit slots."""
    updates: dict[str, Any] = {}

    if slots.target_gender is not None and slots.target_gender.is_resolved():
        g_val = slots.target_gender.value
        if slots.target_gender.source in ("explicit", "history") or (
            slots.target_gender.status == "INFERRED"
            and slots.target_gender.confidence == "high"
            and g_val == GenderTarget.FEMALE.value
        ):
            updates["gender"] = GenderTarget(g_val)

    if (
        slots.fit is not None
        and slots.fit.is_resolved()
        and slots.fit.source in ("explicit", "history")
    ):
        updates["fit_preference"] = slots.fit.value

    if (
        slots.measurements is not None
        and slots.measurements.is_resolved()
        and slots.measurements.source in ("explicit", "history")
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


def get_not_applicable_slots(intent: TurnIntent) -> set[str]:
    """Return slots strictly forbidden for the given intent (Spec Section 4.3)."""
    if intent == TurnIntent.PRODUCT_SEARCH:
        return {"measurements", "body_concern", "top_category", "bottom_category"}
    if intent == TurnIntent.OUTFIT_ADVICE:
        return {"measurements", "body_concern", "category", "category_subtype"}
    if intent == TurnIntent.SIZE_ADVICE:
        return {"body_concern", "material", "top_category", "bottom_category"}
    if intent == TurnIntent.BODY_FIT_ADVICE:
        return {"size", "top_category", "bottom_category"}
    return set(STANDARD_SLOTS)


def get_required_slots(intent: TurnIntent, slots: ConversationSlots) -> list[str]:
    """Return official Required Slots for the given intent (Spec Section 4.3)."""
    if intent == TurnIntent.NON_SHOPPING:
        return []
    if intent == TurnIntent.PRODUCT_SEARCH:
        if slots.has("product_id"):
            return []
        return ["target_gender", "category"]
    if intent == TurnIntent.OUTFIT_ADVICE:
        has_anchor = (
            slots.has("occasion")
            or slots.has("top_category")
            or slots.has("bottom_category")
            or slots.has("product_id")
            or slots.is_declined("occasion")
        )
        req = ["target_gender"]
        if not has_anchor:
            req.append("occasion")
        return req
    if intent == TurnIntent.SIZE_ADVICE:
        if slots.has("product_id"):
            return ["measurements"]
        return ["target_gender", "category", "measurements"]
    if intent == TurnIntent.BODY_FIT_ADVICE:
        return ["target_gender", "body_concern"]
    return []


def get_conditional_slots(intent: TurnIntent, slots: ConversationSlots) -> list[str]:
    """Return official Conditional Slots for the given intent (Spec Section 4.3)."""
    if intent == TurnIntent.NON_SHOPPING:
        return []
    if intent == TurnIntent.PRODUCT_SEARCH:
        cat_val = slots.value_of("category")
        cond: list[str] = []
        if cat_val in ("tops", "pants"):
            cond.append("category_subtype")
        cond.extend(["occasion", "budget", "fit", "material", "size"])
        return cond
    if intent == TurnIntent.OUTFIT_ADVICE:
        cond = []
        if (
            slots.has("top_category")
            or slots.has("bottom_category")
            or slots.has("product_id")
        ):
            cond.append("occasion")
        cond.extend(
            [
                "top_category",
                "bottom_category",
                "budget",
                "style",
                "color",
                "top_color",
                "bottom_color",
            ]
        )
        return cond
    if intent == TurnIntent.SIZE_ADVICE:
        return ["fit", "product_id"]
    if intent == TurnIntent.BODY_FIT_ADVICE:
        return ["category", "measurements", "occasion", "budget"]
    return []


def get_optional_slots(intent: TurnIntent) -> list[str]:
    """Return official Optional Slots for the given intent (Spec Section 4.3)."""
    if intent == TurnIntent.PRODUCT_SEARCH:
        return ["color", "style"]
    if intent == TurnIntent.OUTFIT_ADVICE:
        return ["fit", "material", "size"]
    if intent == TurnIntent.SIZE_ADVICE:
        return ["occasion", "style", "color", "budget"]
    if intent == TurnIntent.BODY_FIT_ADVICE:
        return ["color", "style", "material", "fit"]
    return []


def get_missing_required_slots(
    intent: TurnIntent, slots: ConversationSlots
) -> list[str]:
    """Return unresolved Required Slots and conflicts in priority order."""
    if intent == TurnIntent.NON_SHOPPING:
        return []

    missing: list[str] = list(slots.get_conflicting_slots())
    req_slots = get_required_slots(intent, slots)

    for slot_name in req_slots:
        if slot_name in missing:
            continue
        if slots.is_declined(slot_name):
            continue
        if intent == TurnIntent.SIZE_ADVICE and slot_name == "measurements":
            if not slots.has_complete_measurements(slots.value_of("category")):
                missing.append(slot_name)
            continue
        if intent == TurnIntent.BODY_FIT_ADVICE and slot_name == "body_concern":
            if not (slots.has("body_concern") or slots.has("measurements")):
                missing.append(slot_name)
            continue
        if not slots.has(slot_name):
            missing.append(slot_name)

    return missing


def get_priority_slots(intent: TurnIntent, slots: ConversationSlots) -> list[str]:
    """Return Required then Conditional slots (excluding Not Applicable & Optional)."""
    if intent == TurnIntent.NON_SHOPPING:
        return []
    not_applicable = get_not_applicable_slots(intent)
    ordered: list[str] = []
    candidates = (
        *get_required_slots(intent, slots),
        *get_conditional_slots(intent, slots),
    )
    for s in candidates:
        if s not in not_applicable and s not in ordered:
            ordered.append(s)
    return ordered


def get_missing_priority_slots(
    intent: TurnIntent, slots: ConversationSlots
) -> list[str]:
    """Return ordered missing Required/Conditional slots (never DECLINED/Optional)."""
    if intent == TurnIntent.NON_SHOPPING:
        return []
    missing_req = get_missing_required_slots(intent, slots)
    result: list[str] = list(missing_req)
    for slot_name in get_conditional_slots(intent, slots):
        if slot_name in result or slots.is_declined(slot_name):
            continue
        if not slots.has(slot_name):
            result.append(slot_name)
    return result


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
    if name == "disliked_colors" and isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val)


def format_slots_for_prompt(
    slots: ConversationSlots,
    intent: TurnIntent,
    include_missing: bool = True,
) -> str:
    """Format concise known/missing slot summary with 8-state provenance tags."""
    meas_prio = "P0" if intent == TurnIntent.SIZE_ADVICE else "P3"
    slot_labels = {
        "target_gender": ("Giới tính/Đối tượng", "P0"),
        "category": ("Danh mục", "P0"),
        "top_category": ("Loại áo phối", "P0"),
        "bottom_category": ("Loại quần/váy phối", "P0"),
        "category_subtype": ("Loại sản phẩm cụ thể", "P0"),
        "product_id": ("Mã sản phẩm", "P0"),
        "occasion": ("Dịp sử dụng", "P0"),
        "budget": ("Ngân sách", "P1"),
        "fit": ("Form dáng", "P1"),
        "size": ("Size", "P1"),
        "color": ("Màu sắc", "P2"),
        "top_color": ("Màu áo", "P2"),
        "bottom_color": ("Màu quần/váy", "P2"),
        "disliked_colors": ("Màu tránh", "P2"),
        "style": ("Phong cách", "P2"),
        "material": ("Chất liệu", "P2"),
        "measurements": ("Số đo cơ thể", meas_prio),
        "body_concern": ("Đặc điểm vóc dáng", "P3"),
    }

    base_keys: tuple[str, ...] = (
        "target_gender",
        "category",
        "product_id",
        "occasion",
        "budget",
        "fit",
        "size",
        "color",
        "disliked_colors",
        "style",
        "material",
        "measurements",
        "body_concern",
    )
    if intent == TurnIntent.OUTFIT_ADVICE:
        has_outfit_component = slots.has("top_category") or slots.has("bottom_category")
        ordered_keys: tuple[str, ...] = (
            "target_gender",
            *(("category",) if not has_outfit_component else ()),
            "top_category",
            "bottom_category",
            "product_id",
            "occasion",
            "budget",
            "fit",
            "size",
            "color",
            "top_color",
            "bottom_color",
            "disliked_colors",
            "style",
            "material",
        )
    else:
        ordered_keys = base_keys

    not_applicable = get_not_applicable_slots(intent)
    known_items: list[str] = []
    special_notes: list[str] = []

    for key in ordered_keys:
        if key in not_applicable:
            continue
        raw_slot = slots._raw_slot(key)
        if raw_slot is None:
            continue
        label, prio = slot_labels[key]
        if raw_slot.status == "CONFLICTING":
            special_notes.append(
                f"{label} [CONFLICTING]: {raw_slot.conflict_detail or 'mâu thuẫn'}"
            )
            continue
        if raw_slot.status == "DECLINED":
            special_notes.append(f"{label} [DECLINED]: khách bỏ qua/tùy shop")
            continue
        if slots.has(key):
            disp = _format_slot_display(key, slots.value_of(key))
            tag = (
                "suy luận độ tin cậy cao"
                if raw_slot.status == "INFERRED"
                else raw_slot.source
            )
            known_items.append(f"{label} [{prio}]={disp} ({tag})")

    known_str = "; ".join(known_items) if known_items else "Chưa có"
    lines = [
        f"TÓM TẮT NGỮ CẢNH ({intent.value}):",
        f"- Đã biết: {known_str}",
    ]
    if special_notes:
        lines.append(f"- Lưu ý trạng thái đặc biệt: {'; '.join(special_notes)}")

    if not include_missing:
        return "\n".join(lines)

    missing_req = get_missing_required_slots(intent, slots)
    missing_items = [
        f"{slot_labels[k][0]} [{slot_labels[k][1]}]"
        for k in missing_req
        if k in slot_labels
    ]
    missing_str = ", ".join(missing_items) if missing_items else "Đủ thông tin bắt buộc"
    lines.append(f"- Còn thiếu (bắt buộc): {missing_str}")
    return "\n".join(lines)
