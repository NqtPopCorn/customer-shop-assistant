import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.schemas.shop import (
    MIN_REVIEWS_FOR_FIT_INSIGHT,
    GenderTarget,
)
from customer_shop_assistant.services.shop_client import ShopClient

logger = logging.getLogger(__name__)


class SizeRecommendationResult(BaseModel):
    """Result returned by SizeRecommender."""

    status: Literal["ok", "insufficient_data", "no_size_chart"] = Field(
        default="ok", description="Status of the recommendation"
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Missing measurement fields when data is insufficient",
    )
    primary_size: str | None = Field(default=None, description="Top recommended size")
    secondary_size: str | None = Field(
        default=None, description="Alternative size if close to boundary"
    )
    secondary_direction: Literal["larger", "smaller"] | None = Field(
        default=None,
        description="Whether secondary_size is larger or smaller than primary_size",
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        default="MEDIUM", description="Confidence score of recommendation"
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

    async def recommend_size(
        self,
        height_cm: float | None = None,
        weight_kg: float | None = None,
        chest_cm: float | None = None,
        waist_cm: float | None = None,
        hip_cm: float | None = None,
        category: Literal["tops", "bottoms", "dresses"] = "tops",
        gender: GenderTarget = GenderTarget.UNISEX,
        fit_preference: str = "REGULAR",
        product_id: int | None = None,
    ) -> SizeRecommendationResult:
        """Calculate size recommendation based on deterministic size chart bounds."""
        charts_data = self._load_charts()
        charts = charts_data.get("charts", {})

        # 1. Resolve size entries table by category and gender
        size_entries: list[dict[str, Any]] = []
        category_title = category

        if category == "tops":
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

        elif category == "bottoms":
            bottoms_types = charts.get("bottoms", {}).get("types", {})
            if gender == GenderTarget.FEMALE:
                size_entries = bottoms_types.get("women", {}).get("sizes", [])
                category_title = "Quần Nữ"
            else:
                size_entries = bottoms_types.get("men_numeric", {}).get("sizes", [])
                category_title = "Quần Nam"

        elif category == "dresses":
            size_entries = charts.get("dresses", {}).get("sizes", [])
            category_title = "Váy & Đầm"

        if not size_entries:
            return SizeRecommendationResult(
                status="no_size_chart",
                primary_size=None,
                missing_fields=[],
                reasoning=(
                    f"Không tìm thấy bảng quy chuẩn kích cỡ cho danh mục '{category}'."
                ),
                category_name=category_title,
            )

        # 2. Check for required measurements per category
        missing_fields: list[str] = []
        if category == "tops":
            if height_cm is None and weight_kg is None and chest_cm is None:
                missing_fields = ["height_cm", "weight_kg"]
        elif category == "bottoms":
            if waist_cm is None and weight_kg is None and height_cm is None:
                missing_fields = ["waist_cm", "weight_kg"]
        elif category == "dresses":
            if (
                height_cm is None
                and weight_kg is None
                and waist_cm is None
                and chest_cm is None
                and hip_cm is None
            ):
                missing_fields = ["height_cm", "weight_kg", "waist_cm"]

        if missing_fields:
            return SizeRecommendationResult(
                status="insufficient_data",
                primary_size=None,
                missing_fields=missing_fields,
                reasoning="Chưa đủ thông số đo cơ bản để tư vấn size chính xác.",
                category_name=category_title,
            )

        # 3. Deterministic matching: smallest size where all measurements fit
        size_order = [e["size"] for e in size_entries]

        def fits_in_size(entry: dict[str, Any]) -> bool:
            """Check if all provided measurements fit within the entry's upper range."""
            h_range = entry.get("height_range")
            if height_cm is not None and h_range and height_cm > h_range[1]:
                return False

            w_range = entry.get("weight_range")
            if weight_kg is not None and w_range and weight_kg > w_range[1]:
                return False

            c_range = entry.get("chest_range") or entry.get("bust_range")
            if chest_cm is not None and c_range and chest_cm > c_range[1]:
                return False

            waist_range = entry.get("waist_range")
            if waist_cm is not None and waist_range and waist_cm > waist_range[1]:
                return False

            hip_range = entry.get("hip_range")
            if hip_cm is not None and hip_range and hip_cm > hip_range[1]:
                return False

            return True

        chosen_idx: int = -1
        for idx, entry in enumerate(size_entries):
            if fits_in_size(entry):
                chosen_idx = idx
                break

        # If user exceeds all sizes, select the largest size available
        if chosen_idx == -1:
            chosen_idx = len(size_entries) - 1

        best_size = size_entries[chosen_idx]["size"]
        best_entry = size_entries[chosen_idx]
        second_size: str | None = None

        # 4. Boundary check for secondary candidate
        # If measurements are close to the boundary between sizes
        is_near_upper_boundary = False
        is_near_lower_boundary = False

        h_range = best_entry.get("height_range")
        if height_cm is not None and h_range:
            if abs(height_cm - h_range[1]) <= 2.0:
                is_near_upper_boundary = True
            elif abs(height_cm - h_range[0]) <= 2.0:
                is_near_lower_boundary = True

        w_range = best_entry.get("weight_range")
        if weight_kg is not None and w_range:
            if abs(weight_kg - w_range[1]) <= 1.5:
                is_near_upper_boundary = True
            elif abs(weight_kg - w_range[0]) <= 1.5:
                is_near_lower_boundary = True

        waist_range = best_entry.get("waist_range")
        if waist_cm is not None and waist_range:
            if abs(waist_cm - waist_range[1]) <= 1.5:
                is_near_upper_boundary = True
            elif abs(waist_cm - waist_range[0]) <= 1.5:
                is_near_lower_boundary = True

        if is_near_upper_boundary and chosen_idx + 1 < len(size_entries):
            second_size = size_entries[chosen_idx + 1]["size"]
        elif is_near_lower_boundary and chosen_idx - 1 >= 0:
            second_size = size_entries[chosen_idx - 1]["size"]

        # 5. Fit preference adjustment (LOOSE vs TIGHT)
        fit_pref = (fit_preference or "REGULAR").upper()

        def safe_index(size_label: str | None) -> int:
            if size_label and size_label in size_order:
                return size_order.index(size_label)
            return -1

        if fit_pref in ("LOOSE", "OVERSIZE") and second_size:
            if safe_index(second_size) > safe_index(best_size):
                best_size, second_size = second_size, best_size

        elif fit_pref in ("TIGHT", "SLIM") and second_size:
            if safe_index(second_size) < safe_index(best_size):
                best_size, second_size = second_size, best_size

        # 6. Check review fit signal if product_id provided
        review_insight: str | None = None
        if product_id is not None and self.shop_client is not None:
            try:
                summary = await self.shop_client.get_product_reviews(product_id)
                if (
                    summary.total_reviews >= MIN_REVIEWS_FOR_FIT_INSIGHT
                    and summary.fit_counts
                ):
                    tight_cnt = summary.fit_counts.get("TIGHT", 0)
                    true_cnt = summary.fit_counts.get("TRUE_TO_SIZE", 0)
                    loose_cnt = summary.fit_counts.get("LOOSE", 0)

                    if tight_cnt > true_cnt and tight_cnt > loose_cnt:
                        review_insight = (
                            f"Đa số khách hàng ({tight_cnt}/{summary.total_reviews}) "
                            "đánh giá form ôm, nên cân nhắc tăng 1 size "
                            "nếu thích thoải mái."
                        )
                        if second_size and safe_index(second_size) > safe_index(
                            best_size
                        ):
                            best_size, second_size = second_size, best_size
                    elif loose_cnt > true_cnt and loose_cnt > tight_cnt:
                        review_insight = (
                            f"Đa số khách hàng ({loose_cnt}/{summary.total_reviews}) "
                            "đánh giá form rộng, nên cân nhắc giảm 1 size "
                            "nếu thích vừa vặn."
                        )
                        if second_size and safe_index(second_size) < safe_index(
                            best_size
                        ):
                            best_size, second_size = second_size, best_size
            except Exception as e:
                logger.warning(
                    "Could not fetch reviews for product %s: %s",
                    product_id,
                    e,
                )

        confidence: Literal["HIGH", "MEDIUM", "LOW"] = (
            "HIGH" if second_size is None else "MEDIUM"
        )

        reasoning = (
            f"Dựa trên chiều cao {height_cm:.0f}cm và cân nặng {weight_kg:.0f}kg, "
            f"size {best_size} là phù hợp nhất theo bảng quy chuẩn {category_title}."
            if height_cm and weight_kg
            else (
                f"Size {best_size} phù hợp nhất với các thông số đo "
                f"được cung cấp theo bảng {category_title}."
            )
        )

        secondary_direction: Literal["larger", "smaller"] | None = None
        if second_size:
            is_smaller = safe_index(second_size) < safe_index(best_size)
            secondary_direction = "smaller" if is_smaller else "larger"
            fit_phrase = (
                "nếu muốn mặc ôm gọn hơn"
                if is_smaller
                else "nếu muốn mặc rộng thoải mái hơn"
            )
            reasoning += (
                f" Số đo của bạn nằm tiệm cận giữa size {best_size} và {second_size}: "
                f"ưu tiên {best_size} vừa vặn hoặc {second_size} {fit_phrase}."
            )

        return SizeRecommendationResult(
            status="ok",
            primary_size=best_size,
            secondary_size=second_size,
            secondary_direction=secondary_direction,
            confidence=confidence,
            reasoning=reasoning,
            review_insight=review_insight,
            category_name=category_title,
        )

    @staticmethod
    def _format_chart_table(title: str, sizes: list[dict[str, Any]]) -> str:
        """Format a single size table section into clean Markdown."""
        if not sizes:
            return ""
        has_chest = any("chest_range" in s or "bust_range" in s for s in sizes)
        has_waist = any("waist_range" in s for s in sizes)

        headers = ["Size", "Chiều cao", "Cân nặng"]
        if has_chest:
            headers.append("Vòng ngực")
        if has_waist:
            headers.append("Vòng eo")

        lines = [
            f"**{title}:**",
            f"| {' | '.join(headers)} |",
            f"| {' | '.join('---' for _ in headers)} |",
        ]
        for entry in sizes:
            h = entry.get("height_range")
            w = entry.get("weight_range")
            row = [
                str(entry.get("size", "")),
                f"{h[0]}–{h[1]}cm" if h else "—",
                f"{w[0]}–{w[1]}kg" if w else "—",
            ]
            if has_chest:
                c = entry.get("chest_range") or entry.get("bust_range")
                row.append(f"{c[0]}–{c[1]}cm" if c else "—")
            if has_waist:
                waist = entry.get("waist_range")
                row.append(f"{waist[0]}–{waist[1]}cm" if waist else "—")
            lines.append(f"| {' | '.join(row)} |")
        return "\n".join(lines)

    def format_size_chart(
        self,
        category: str | None = None,
        gender: GenderTarget | str | None = None,
    ) -> str:
        """Return standard size chart Markdown from data/size_charts.json."""
        charts_data = self._load_charts()
        charts = charts_data.get("charts", {})
        if not charts:
            return (
                "Hiện tại chưa tải được bảng size chuẩn. "
                "Bạn vui lòng cung cấp chiều cao và cân nặng để mình kiểm tra lại nhé."
            )

        norm_gender: GenderTarget | None = None
        if isinstance(gender, GenderTarget):
            norm_gender = gender
        elif isinstance(gender, str) and gender.strip():
            g_up = gender.strip().upper()
            if g_up in GenderTarget.__members__:
                norm_gender = GenderTarget[g_up]

        cat_clean = (category or "").strip().lower()
        sections: list[str] = []

        tops_genders = charts.get("tops", {}).get("genders", {})
        bottoms_types = charts.get("bottoms", {}).get("types", {})
        dresses_data = charts.get("dresses", {})

        def _append_tops() -> None:
            if norm_gender == GenderTarget.MALE:
                g_data = tops_genders.get("men", {})
                sections.append(
                    self._format_chart_table(
                        g_data.get("title", "Áo Nam"), g_data.get("sizes", [])
                    )
                )
            elif norm_gender == GenderTarget.FEMALE:
                g_data = tops_genders.get("women", {})
                sections.append(
                    self._format_chart_table(
                        g_data.get("title", "Áo Nữ"), g_data.get("sizes", [])
                    )
                )
            elif norm_gender == GenderTarget.UNISEX:
                g_data = tops_genders.get("unisex", {})
                sections.append(
                    self._format_chart_table(
                        g_data.get("title", "Áo Unisex"), g_data.get("sizes", [])
                    )
                )
            else:
                for key, fallback_title in (("men", "Áo Nam"), ("women", "Áo Nữ")):
                    g_data = tops_genders.get(key, {})
                    sections.append(
                        self._format_chart_table(
                            g_data.get("title", fallback_title), g_data.get("sizes", [])
                        )
                    )

        def _append_bottoms(is_short: bool = False) -> None:
            if norm_gender == GenderTarget.FEMALE:
                b_data = bottoms_types.get("women", {})
                sections.append(
                    self._format_chart_table(
                        b_data.get("title", "Quần Nữ & Chân Váy"),
                        b_data.get("sizes", []),
                    )
                )
            elif norm_gender == GenderTarget.MALE:
                key = "men_short" if is_short else "men_numeric"
                b_data = bottoms_types.get(key, {})
                sections.append(
                    self._format_chart_table(
                        b_data.get("title", "Quần Nam"), b_data.get("sizes", [])
                    )
                )
            else:
                men_key = "men_short" if is_short else "men_numeric"
                for key, fallback_title in (
                    (men_key, "Quần Nam"),
                    ("women", "Quần Nữ & Chân Váy"),
                ):
                    b_data = bottoms_types.get(key, {})
                    sections.append(
                        self._format_chart_table(
                            b_data.get("title", fallback_title),
                            b_data.get("sizes", []),
                        )
                    )

        if cat_clean in ("shirt", "polo", "tshirt", "jacket", "tops"):
            _append_tops()
        elif cat_clean == "short":
            _append_bottoms(is_short=True)
        elif cat_clean in ("jeans", "kaki", "pants", "bottoms"):
            _append_bottoms(is_short=False)
        elif cat_clean in ("dress", "dresses"):
            sections.append(
                self._format_chart_table(
                    dresses_data.get("name", "Váy liền & Đầm (Nữ)"),
                    dresses_data.get("sizes", []),
                )
            )
        elif cat_clean == "skirt":
            b_data = bottoms_types.get("women", {})
            sections.append(
                self._format_chart_table(
                    b_data.get("title", "Quần Nữ & Chân Váy"),
                    b_data.get("sizes", []),
                )
            )
        else:
            _append_tops()
            _append_bottoms(is_short=False)

        tables_text = "\n\n".join(s for s in sections if s)
        note = (
            "Lưu ý chọn size: Vì chưa có số đo cụ thể nên bạn có thể đối chiếu "
            "theo bảng quy chuẩn trên; nếu số đo nằm giữa 2 size, chọn size nhỏ hơn "
            "để mặc vừa vặn hoặc tăng 1 size nếu thích thoải mái."
        )
        return (
            "Dưới đây là bảng size chuẩn để bạn tự đối chiếu:\n\n"
            f"{tables_text}\n\n{note}"
        )
