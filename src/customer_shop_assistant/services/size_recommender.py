import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.schemas.shop import GenderTarget, ProductFit
from customer_shop_assistant.services.shop_client import ShopClient

logger = logging.getLogger(__name__)


class SizeRecommendationResult(BaseModel):
    """Result returned by SizeRecommender."""

    primary_size: str = Field(..., description="Top recommended size")
    secondary_size: str | None = Field(
        default=None, description="Alternative size if close to boundary"
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        ..., description="Confidence score of recommendation"
    )
    reasoning: str = Field(..., description="Explanation of why this size was chosen")
    review_insight: str | None = Field(
        default=None, description="Actual customer review feedback on form/fit"
    )
    category_name: str = Field(..., description="Resolved clothing category")


class SizeRecommender:
    """Deterministic size recommendation engine using size_charts.json."""

    def __init__(
        self,
        settings: Settings | None = None,
        shop_client: ShopClient | None = None,
        data_path: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.shop_client = shop_client
        self.data_path = data_path or (self.settings.DATA_DIR / "size_charts.json")
        self._charts_cache: dict[str, Any] | None = None

    def _load_charts(self) -> dict[str, Any]:
        if self._charts_cache is None:
            if not self.data_path.exists():
                logger.error("Size chart file not found at %s", self.data_path)
                return {}
            with open(self.data_path, encoding="utf-8") as f:
                self._charts_cache = json.load(f)
        return self._charts_cache

    def _normalize_category(self, raw_cat: str) -> str:
        cat_lower = raw_cat.strip().lower()
        top_keywords = ["áo", "top", "t-shirt", "sơ mi", "polo", "khoác"]
        if any(w in cat_lower for w in top_keywords):
            return "tops"
        bottom_keywords = ["quần", "bottom", "jean", "kaki", "short"]
        if any(w in cat_lower for w in bottom_keywords):
            return "bottoms"
        if any(w in cat_lower for w in ["váy", "đầm", "dress"]):
            return "dresses"
        return "tops"

    async def recommend_size(
        self,
        height_cm: float | None = None,
        weight_kg: float | None = None,
        chest_cm: float | None = None,
        waist_cm: float | None = None,
        hip_cm: float | None = None,
        category: str = "tops",
        gender: GenderTarget = GenderTarget.UNISEX,
        fit_preference: str = "REGULAR",
        product_id: int | None = None,
    ) -> SizeRecommendationResult:
        """Calculate size recommendation based on body measurements."""
        charts_data = self._load_charts()
        charts = charts_data.get("charts", {})
        norm_cat = self._normalize_category(category)

        # 1. Select the relevant size entries table
        size_entries: list[dict[str, Any]] = []
        category_title = norm_cat

        if norm_cat == "tops":
            tops_charts = charts.get("tops", {}).get("genders", {})
            if gender == GenderTarget.MALE:
                size_entries = tops_charts.get("men", {}).get("sizes", [])
                category_title = "Áo Nam"
            elif gender == GenderTarget.FEMALE:
                size_entries = tops_charts.get("women", {}).get("sizes", [])
                category_title = "Áo Nữ"
            else:
                size_entries = tops_charts.get("unisex", {}).get("sizes", [])
                category_title = "Áo Unisex"

        elif norm_cat == "bottoms":
            bottoms_types = charts.get("bottoms", {}).get("types", {})
            if gender == GenderTarget.FEMALE:
                size_entries = bottoms_types.get("women", {}).get("sizes", [])
                category_title = "Quần Nữ"
            else:
                # Default to numeric jeans/kaki sizes for male
                size_entries = bottoms_types.get("men_numeric", {}).get("sizes", [])
                category_title = "Quần Nam"

        elif norm_cat == "dresses":
            size_entries = charts.get("dresses", {}).get("sizes", [])
            category_title = "Váy & Đầm"

        if not size_entries:
            return SizeRecommendationResult(
                primary_size="M",
                secondary_size="L",
                confidence="LOW",
                reasoning="Không tìm thấy bảng size, đề xuất size M phổ biến.",
                category_name=category_title,
            )

        # 2. Score each size based on measurement distance
        scored_sizes: list[tuple[str, float, dict[str, Any]]] = []

        for entry in size_entries:
            size_label = entry["size"]
            total_penalty = 0.0
            factors = 0

            # Height matching
            h_range = entry.get("height_range")
            if height_cm is not None and h_range:
                factors += 1
                if h_range[0] <= height_cm <= h_range[1]:
                    penalty = 0.0
                elif height_cm < h_range[0]:
                    penalty = (h_range[0] - height_cm) * 1.5
                else:
                    penalty = (height_cm - h_range[1]) * 1.5
                total_penalty += penalty * 0.4

            # Weight matching
            w_range = entry.get("weight_range")
            if weight_kg is not None and w_range:
                factors += 1
                if w_range[0] <= weight_kg <= w_range[1]:
                    penalty = 0.0
                elif weight_kg < w_range[0]:
                    penalty = (w_range[0] - weight_kg) * 2.0
                else:
                    penalty = (weight_kg - w_range[1]) * 2.0
                total_penalty += penalty * 0.6

            # Waist matching (for bottoms or dresses)
            waist_range = entry.get("waist_range")
            if waist_cm is not None and waist_range:
                factors += 1
                if waist_range[0] <= waist_cm <= waist_range[1]:
                    penalty = 0.0
                elif waist_cm < waist_range[0]:
                    penalty = (waist_range[0] - waist_cm) * 2.0
                else:
                    penalty = (waist_cm - waist_range[1]) * 2.0
                total_penalty += penalty * 0.5

            if factors > 0:
                scored_sizes.append((size_label, total_penalty, entry))

        # Sort sizes by lowest penalty (best match)
        scored_sizes.sort(key=lambda x: x[1])

        if not scored_sizes:
            return SizeRecommendationResult(
                primary_size="M",
                secondary_size="L",
                confidence="LOW",
                reasoning="Thiếu thông số chiều cao và cân nặng để tính toán.",
                category_name=category_title,
            )

        best_size, best_score, best_entry = scored_sizes[0]
        second_size: str | None = None

        if len(scored_sizes) > 1:
            second_candidate, second_score, _ = scored_sizes[1]
            if (second_score - best_score) <= 4.0:
                second_size = second_candidate

        # 3. Consider fit preference (TIGHT vs LOOSE)
        fit_pref = fit_preference.upper()
        size_order = [e["size"] for e in size_entries]

        if fit_pref in ("TIGHT", "SLIM") and second_size:
            if best_size in size_order and second_size in size_order:
                if size_order.index(second_size) < size_order.index(best_size):
                    best_size, second_size = second_size, best_size

        elif fit_pref in ("LOOSE", "OVERSIZE") and second_size:
            if best_size in size_order and second_size in size_order:
                if size_order.index(second_size) > size_order.index(best_size):
                    best_size, second_size = second_size, best_size

        confidence: Literal["HIGH", "MEDIUM", "LOW"] = (
            "HIGH"
            if best_score == 0.0 and second_size is None
            else "MEDIUM"
            if best_score <= 3.0
            else "LOW"
        )

        reasoning = (
            f"Dựa trên chiều cao {height_cm:.0f}cm và cân nặng "
            f"{weight_kg:.0f}kg, size {best_size} là phù hợp nhất theo "
            f"bảng quy chuẩn {category_title}."
            if height_cm and weight_kg
            else f"Size {best_size} phù hợp với thông số được cung cấp."
        )

        if second_size:
            reasoning += (
                f" Số đo của bạn nằm tiệm cận giữa size {best_size} và "
                f"{second_size}: chọn {best_size} nếu thích vừa vặn, hoặc "
                f"{second_size} nếu thích mặc rộng thoải mái."
            )

        # 4. Check review fit signal if product_id provided
        review_insight: str | None = None
        if product_id is not None and self.shop_client is not None:
            try:
                summary = await self.shop_client.get_product_reviews(product_id)
                if summary.total_reviews >= 10 and summary.dominant_fit:
                    review_insight = summary.fit_recommendation
                    if summary.dominant_fit == ProductFit.TIGHT and second_size:
                        if size_order.index(second_size) > size_order.index(best_size):
                            best_size, second_size = second_size, best_size
                            reasoning += (
                                f" (Lưu ý: Mẫu này có {summary.total_reviews} "
                                f"đánh giá, khách nhận xét form ôm nên đã ưu "
                                f"tiên gợi ý size {best_size})."
                            )
            except Exception as e:
                logger.warning(
                    "Could not fetch reviews for product %s: %s",
                    product_id,
                    e,
                )

        return SizeRecommendationResult(
            primary_size=best_size,
            secondary_size=second_size,
            confidence=confidence,
            reasoning=reasoning,
            review_insight=review_insight,
            category_name=category_title,
        )
