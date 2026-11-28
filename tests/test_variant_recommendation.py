"""Unit tests for variant-level recommendation, image selection, and deep-linking."""

from customer_shop_assistant.schemas.chat import ChatRecommendedProduct, OutfitItem
from customer_shop_assistant.schemas.shop import (
    CategorySchema,
    GenderTarget,
    ProductImageSchema,
    ProductSchema,
    ProductVariantSchema,
)
from customer_shop_assistant.services.agent_service import ShopAssistantAgent
from customer_shop_assistant.services.outfit_kg_repository import (
    InMemoryOutfitKGRepository,
)
from customer_shop_assistant.services.outfit_retrieval import (
    OutfitRetrievalAndReranker,
)
from customer_shop_assistant.tools.catalog_tool import product_schema_to_recommended


def _sample_shirt() -> ProductSchema:
    return ProductSchema(
        id=436,
        name="Áo sơ mi Nam Linen",
        slug="ao-so-mi-nam-linen",
        targetGender=GenderTarget.MALE,
        category=CategorySchema(id=10, name="Áo sơ mi"),
        images=[ProductImageSchema(id=1, url="https://img/white-default.jpg")],
        variants=[
            ProductVariantSchema(
                id=1001,
                sku="HM-W-S",
                color="Trắng",
                size="S",
                imageUrl="https://img/white-s.jpg",
                listPrice=271000,
                sellingPrice=271000,
                finalPrice=244000,
                stock=10,
            ),
            ProductVariantSchema(
                id=1002,
                sku="HM-R-S",
                color="Đỏ đô",
                size="S",
                imageUrl="https://img/burgundy-s.jpg",
                listPrice=271000,
                sellingPrice=271000,
                finalPrice=244000,
                stock=0,
            ),
            ProductVariantSchema(
                id=1003,
                sku="HM-R-M",
                color="Đỏ đô",
                size="M",
                imageUrl="https://img/burgundy-m.jpg",
                listPrice=271000,
                sellingPrice=260000,
                finalPrice=235000,
                stock=5,
            ),
            ProductVariantSchema(
                id=1004,
                sku="HM-B-L",
                color="Xanh dương",
                size="L",
                imageUrl="https://img/blue-l.jpg",
                listPrice=271000,
                sellingPrice=271000,
                finalPrice=244000,
                stock=8,
            ),
        ],
    )


def test_product_schema_to_recommended_without_filters_keeps_default() -> None:
    shirt = _sample_shirt()
    rec = product_schema_to_recommended(shirt)
    assert rec.id == 436
    assert rec.image_url == "https://img/white-default.jpg"
    assert rec.variant_id is None
    assert rec.color is None
    assert rec.size is None
    assert rec.price == 235000


def test_product_schema_to_recommended_matches_color_preferring_in_stock() -> None:
    shirt = _sample_shirt()
    # Accent-insensitive query; skips out-of-stock S (1002) and picks M (1003)
    rec = product_schema_to_recommended(shirt, preferred_color="do do")
    assert rec.variant_id == 1003
    assert rec.color == "Đỏ đô"
    assert rec.size == "M"
    assert rec.image_url == "https://img/burgundy-m.jpg"
    assert rec.price == 235000
    assert rec.original_price == 271000
    assert rec.discount_percent == 13


def test_product_schema_to_recommended_matches_exact_color_and_size() -> None:
    shirt = _sample_shirt()
    rec = product_schema_to_recommended(
        shirt, preferred_color="Xanh dương", preferred_size="l"
    )
    assert rec.variant_id == 1004
    assert rec.color == "Xanh dương"
    assert rec.size == "L"
    assert rec.image_url == "https://img/blue-l.jpg"


def test_outfit_retrieval_populates_variant_id_and_image_url() -> None:
    shirt = _sample_shirt()
    trousers = ProductSchema(
        id=501,
        name="Quần Âu Nam",
        slug="quan-au-nam",
        targetGender=GenderTarget.MALE,
        category=CategorySchema(id=20, name="Quần âu"),
        images=[ProductImageSchema(id=2, url="https://img/pants-default.jpg")],
        variants=[
            ProductVariantSchema(
                id=2001,
                sku="PANTS-BLK-31",
                color="Đen",
                size="31",
                imageUrl="https://img/pants-black.jpg",
                listPrice=300000,
                sellingPrice=250000,
                finalPrice=250000,
                stock=4,
            )
        ],
    )
    reranker = OutfitRetrievalAndReranker(kg_repo=InMemoryOutfitKGRepository())
    result = reranker.retrieve_outfits(
        live_products=[shirt, trousers],
        product_id=436,
        color="Đỏ đô",
        limit=1,
    )
    assert result.outfits
    top = result.outfits[0].top
    assert top.product_id == 436
    assert top.color == "Đỏ đô"
    assert top.variant_id == 1003
    assert top.size == "M"
    assert top.image_url == "https://img/burgundy-m.jpg"


def test_enrich_recommended_from_outfit_item_overrides_default_image() -> None:
    cached = ChatRecommendedProduct(
        id=436,
        name="Áo sơ mi Nam Linen",
        slug="ao-so-mi-nam-linen",
        price=244000,
        original_price=271000,
        discount_percent=10,
        image_url="https://img/white-default.jpg",
        category="Áo sơ mi",
        brand="H&M",
    )
    item = OutfitItem(
        product_id=436,
        name="Áo sơ mi Nam Linen",
        slug="ao-so-mi-nam-linen",
        category="Áo sơ mi",
        color="Đỏ đô",
        variant_id=1003,
        size="M",
        in_stock=True,
        image_url="https://img/burgundy-m.jpg",
        price=235000,
    )
    enriched = ShopAssistantAgent._enrich_recommended_from_outfit_item(cached, item)
    assert enriched.image_url == "https://img/burgundy-m.jpg"
    assert enriched.variant_id == 1003
    assert enriched.color == "Đỏ đô"
    assert enriched.size == "M"
    assert enriched.price == 235000
    assert enriched.discount_percent == 13
