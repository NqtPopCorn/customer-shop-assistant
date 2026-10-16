import httpx
import pytest

from customer_shop_assistant.config import Settings
from customer_shop_assistant.schemas.shop import GenderTarget
from customer_shop_assistant.services.shop_client import ShopClient
from customer_shop_assistant.services.size_recommender import SizeRecommender


@pytest.fixture
def recommender() -> SizeRecommender:
    return SizeRecommender()


@pytest.mark.anyio
async def test_standard_men_shirt(recommender: SizeRecommender) -> None:
    result = await recommender.recommend_size(
        height_cm=168,
        weight_kg=60,
        category="tops",
        gender=GenderTarget.MALE,
    )
    assert result.primary_size == "M"
    assert result.confidence in ("HIGH", "MEDIUM")
    assert "M" in result.reasoning


@pytest.mark.anyio
async def test_boundary_men_shirt(recommender: SizeRecommender) -> None:
    # 171cm, 65kg sits between M (165-172, 56-65) and L (170-178, 66-75)
    result = await recommender.recommend_size(
        height_cm=171,
        weight_kg=65.5,
        category="tops",
        gender=GenderTarget.MALE,
    )
    assert result.primary_size in ("M", "L")
    assert result.secondary_size in ("M", "L")
    assert result.secondary_size is not None


@pytest.mark.anyio
async def test_female_dress(recommender: SizeRecommender) -> None:
    result = await recommender.recommend_size(
        height_cm=155,
        weight_kg=44,
        waist_cm=64,
        category="dresses",
        gender=GenderTarget.FEMALE,
    )
    assert result.primary_size == "S"
    assert result.category_name == "Váy & Đầm"


@pytest.mark.anyio
async def test_men_numeric_jeans(recommender: SizeRecommender) -> None:
    result = await recommender.recommend_size(
        height_cm=168,
        weight_kg=61,
        waist_cm=78,
        category="bottoms",
        gender=GenderTarget.MALE,
    )
    assert result.primary_size == "30"


@pytest.mark.anyio
async def test_fit_preference_loose(recommender: SizeRecommender) -> None:
    # Boundary case with loose preference should prioritize larger size
    result = await recommender.recommend_size(
        height_cm=171,
        weight_kg=65.5,
        category="tops",
        gender=GenderTarget.MALE,
        fit_preference="LOOSE",
    )
    assert result.primary_size == "L"


@pytest.mark.anyio
async def test_review_fit_adjustment() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "items": [{"id": 1, "productId": 1, "rating": 5, "fit": "TIGHT"}],
                "stats": {
                    "total": 15,
                    "averageRating": 4.8,
                    "fitDistribution": {"TIGHT": 12, "TRUE_TO_SIZE": 3, "LOOSE": 0},
                },
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as mock_http:
        settings = Settings(MAIN_BE_BASE_URL="http://test/api")
        shop_client = ShopClient(settings=settings, client=mock_http)
        recommender = SizeRecommender(settings=settings, shop_client=shop_client)

        result = await recommender.recommend_size(
            height_cm=171,
            weight_kg=65.5,
            category="tops",
            gender=GenderTarget.MALE,
            product_id=1,
        )

        assert result.review_insight is not None
        assert "ôm" in result.review_insight
        # Upgraded to larger size due to TIGHT reviews
        assert result.primary_size == "L"
