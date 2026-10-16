import re
from typing import NamedTuple

from customer_shop_assistant.schemas.chat import ClarificationMode, UserProfile
from customer_shop_assistant.schemas.shop import GenderTarget


class ExtractedEntities(NamedTuple):
    """Entities extracted from user utterance using deterministic rules."""

    height_cm: float | None
    weight_kg: float | None
    waist_cm: float | None
    gender: GenderTarget | None
    fit_preference: str | None
    category: str | None
    voucher_code: str | None


class ClarificationDecision(NamedTuple):
    """Decision on whether clarification is required."""

    needs_clarification: bool
    clarification_question: str | None
    extracted: ExtractedEntities


class ClarificationEngine:
    """Manages clarification gates based on active ClarificationMode."""

    @staticmethod
    def extract_entities(text: str) -> ExtractedEntities:
        """Extract measurements, gender, fit preferences and keywords from query."""
        lower = text.lower()

        # Height extraction (e.g. 1m72, 172cm, cao 172, 1m70)
        height: float | None = None
        m_height1 = re.search(r"(\d)m(\d{1,2})", lower)
        if m_height1:
            meters = int(m_height1.group(1))
            cms = int(m_height1.group(2).ljust(2, "0"))
            height = float(meters * 100 + cms)
        else:
            m_height2 = re.search(r"(?:cao\s*)?(\d{3})\s*(?:cm)?\b", lower)
            if m_height2:
                val = float(m_height2.group(1))
                if 120 <= val <= 220:
                    height = val

        # Weight extraction (e.g. 65kg, nặng 65, 65 cân)
        weight: float | None = None
        m_weight = re.search(r"(?:nặng\s*)?(\d{2,3})\s*(?:kg|cân|kí)\b", lower)
        if m_weight:
            val = float(m_weight.group(1))
            if 30 <= val <= 180:
                weight = val
        else:
            m_weight_bare = re.search(r"nặng\s*(\d{2,3})\b", lower)
            if m_weight_bare:
                val = float(m_weight_bare.group(1))
                if 30 <= val <= 180:
                    weight = val

        # Waist extraction (e.g. eo 72, vòng 2 72, vòng bụng 78cm, vòng bụng mình 78)
        waist: float | None = None
        m_waist = re.search(
            r"(?:eo|vòng 2|vòng bụng|bụng)(?:\s+\w+){0,2}\s*[:=]?\s*(\d{2,3})",
            lower,
        )
        if m_waist:
            waist = float(m_waist.group(1))

        # Gender extraction
        gender: GenderTarget | None = None
        if re.search(r"\b(?:nam|con trai|nam giới)\b", lower):
            gender = GenderTarget.MALE
        elif re.search(r"\b(?:nữ|con gái|nữ giới|váy|đầm)\b", lower):
            gender = GenderTarget.FEMALE

        # Fit preference
        fit_pref: str | None = None
        if any(w in lower for w in ["ôm", "slim", "bó"]):
            fit_pref = "TIGHT"
        elif any(w in lower for w in ["rộng", "oversize", "thoải mái", "loose"]):
            fit_pref = "LOOSE"
        elif any(w in lower for w in ["vừa vặn", "regular", "chuẩn"]):
            fit_pref = "REGULAR"

        # Clothing category
        category: str | None = None
        if any(w in lower for w in ["áo", "t-shirt", "sơ mi", "polo", "khoác"]):
            category = "tops"
        elif any(w in lower for w in ["quần", "jean", "kaki", "short"]):
            category = "bottoms"
        elif any(w in lower for w in ["váy", "đầm", "dress"]):
            category = "dresses"

        # Voucher code
        voucher_code: str | None = None
        m_vouch = re.search(r"(?:mã|voucher|code)\s+([A-Z0-9_-]{4,15})\b", text)
        if m_vouch:
            voucher_code = m_vouch.group(1)

        return ExtractedEntities(
            height_cm=height,
            weight_kg=weight,
            waist_cm=waist,
            gender=gender,
            fit_preference=fit_pref,
            category=category,
            voucher_code=voucher_code,
        )

    def evaluate(
        self,
        query: str,
        intent: str,
        mode: ClarificationMode,
        profile: UserProfile | None = None,
    ) -> ClarificationDecision:
        """Evaluate if clarification is required under the specified mode."""
        extracted = self.extract_entities(query)
        lower = query.lower()

        h = extracted.height_cm or (profile.height_cm if profile else None)
        w = extracted.weight_kg or (profile.weight_kg if profile else None)
        g = extracted.gender or (profile.gender if profile else None)
        fit = extracted.fit_preference or (profile.fit_preference if profile else None)

        # Mode 1: IMMEDIATE_ANSWER never blocks with clarification questions
        if mode == ClarificationMode.IMMEDIATE_ANSWER:
            return ClarificationDecision(
                needs_clarification=False,
                clarification_question=None,
                extracted=extracted,
            )

        # Policy inquiry & general chat never need clarification
        # Policy inquiry, out of scope & general chat never need clarification
        if intent in ("POLICY_INQUIRY", "GENERAL_CHAT", "OUT_OF_SCOPE"):
            return ClarificationDecision(
                needs_clarification=False,
                clarification_question=None,
                extracted=extracted,
            )

        # Mode 2: MINIMAL_CLARIFICATION (Default)
        if mode == ClarificationMode.MINIMAL_CLARIFICATION:
            if intent == "SIZE_ADVICE":
                # Critical info missing: height, weight, AND waist
                if h is None and w is None and extracted.waist_cm is None:
                    cat_hint = (
                        f" cho {extracted.category}" if extracted.category else ""
                    )
                    return ClarificationDecision(
                        needs_clarification=True,
                        clarification_question=(
                            f"Để tư vấn size chuẩn xác nhất{cat_hint}, bạn vui lòng "
                            "cho mình biết chiều cao (cm) và cân nặng (kg) của bạn nhé!"
                        ),
                        extracted=extracted,
                    )
            elif intent == "PRODUCT_ADVICE":
                has_category = bool(extracted.category)
                has_occasion = any(
                    k in lower
                    for k in [
                        "đi làm",
                        "công sở",
                        "đi tiệc",
                        "dự tiệc",
                        "đi chơi",
                        "dạo phố",
                        "du lịch",
                        "đi biển",
                        "đám cưới",
                    ]
                )
                has_spec = any(
                    k in lower
                    for k in [
                        "gầy",
                        "bụng",
                        "nấm lùn",
                        "thoáng mát",
                        "giữ ấm",
                        "chống nhăn",
                        "phối với",
                        "mát",
                        "ấm",
                    ]
                )
                if not has_category and not has_occasion and not has_spec:
                    return ClarificationDecision(
                        needs_clarification=True,
                        clarification_question=(
                            "Bạn đang tìm trang phục cho dịp nào (đi làm, đi chơi, "
                            "hay dự tiệc) hoặc phong cách yêu thích của bạn là gì ạ?"
                        ),
                        extracted=extracted,
                    )

            return ClarificationDecision(
                needs_clarification=False,
                clarification_question=None,
                extracted=extracted,
            )

        # Mode 3: ALWAYS_CLARIFY
        if mode == ClarificationMode.ALWAYS_CLARIFY:
            if intent == "SIZE_ADVICE":
                if (h is None or w is None) and extracted.waist_cm is None:
                    return ClarificationDecision(
                        needs_clarification=True,
                        clarification_question=(
                            "Bạn hãy cung cấp chiều cao (cm) và cân nặng (kg) "
                            "để mình chọn size chuẩn xác theo số đo nhé!"
                        ),
                        extracted=extracted,
                    )
                if fit is None:
                    return ClarificationDecision(
                        needs_clarification=True,
                        clarification_question=(
                            "Bạn thích mặc form ôm dáng (Slim), vừa vặn (Regular) "
                            "hay rộng rãi thoải mái (Oversize)?"
                        ),
                        extracted=extracted,
                    )
            elif intent in ("PRODUCT_ADVICE", "CATALOG_SEARCH"):
                if g is None and not extracted.category:
                    return ClarificationDecision(
                        needs_clarification=True,
                        clarification_question=(
                            "Bạn muốn tìm sản phẩm cho Nam hay Nữ ạ?"
                        ),
                        extracted=extracted,
                    )
                if h is None or w is None:
                    return ClarificationDecision(
                        needs_clarification=True,
                        clarification_question=(
                            "Bạn hãy cung cấp chiều cao (cm) và cân nặng (kg) "
                            "để mình tư vấn chuẩn xác vóc dáng nhé!"
                        ),
                        extracted=extracted,
                    )
                if fit is None:
                    return ClarificationDecision(
                        needs_clarification=True,
                        clarification_question=(
                            "Bạn thích mặc form ôm dáng (Slim), vừa vặn (Regular) "
                            "hay rộng rãi thoải mái (Oversize)?"
                        ),
                        extracted=extracted,
                    )

        return ClarificationDecision(
            needs_clarification=False,
            clarification_question=None,
            extracted=extracted,
        )
