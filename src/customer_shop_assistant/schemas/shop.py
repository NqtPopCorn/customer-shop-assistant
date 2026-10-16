from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class GenderTarget(StrEnum):
    """Target gender for clothing products."""

    MALE = "MALE"
    FEMALE = "FEMALE"
    UNISEX = "UNISEX"


class ProductFit(StrEnum):
    """Customer fit feedback rating from product reviews."""

    TIGHT = "TIGHT"
    TRUE_TO_SIZE = "TRUE_TO_SIZE"
    LOOSE = "LOOSE"


class BrandSchema(BaseModel):
    """Brand information for a fashion product."""

    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(..., description="Brand unique identifier")
    name: str = Field(..., description="Brand name")
    slug: str = Field(..., description="Brand URL slug")
    logo: str | None = Field(default=None, description="Brand logo image URL")


class CategorySchema(BaseModel):
    """Category tree node for clothing taxonomy."""

    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(..., description="Category unique identifier")
    name: str = Field(..., description="Category name")
    parent_id: int | None = Field(
        default=None, alias="parentId", description="Parent category identifier"
    )
    children: list["CategorySchema"] | None = Field(
        default=None, description="Child subcategories"
    )


class ProductImageSchema(BaseModel):
    """Product gallery image."""

    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(..., description="Image unique identifier")
    url: str = Field(..., description="Image public URL")
    alt_text: str | None = Field(
        default=None, alias="altText", description="Accessibility alt text"
    )
    sort_order: int = Field(
        default=0, alias="sortOrder", description="Display ordering index"
    )


class ProductVariantSchema(BaseModel):
    """Specific stock-keeping unit (SKU) variant of a clothing product."""

    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(..., description="Variant unique identifier")
    sku: str = Field(..., description="Stock keeping unit code")
    barcode: str | None = Field(default=None, description="EAN/UPC barcode")
    size: str | None = Field(
        default=None, description="Size label (e.g. S, M, L, XL or 29, 30)"
    )
    color: str | None = Field(
        default=None, description="Color name (e.g. Trắng, Đen, Xanh navy)"
    )
    color_hex: str | None = Field(
        default=None, alias="colorHex", description="Hex color code"
    )
    image_url: str | None = Field(
        default=None, alias="imageUrl", description="Variant image URL"
    )
    list_price: float = Field(
        ..., alias="listPrice", ge=0, description="Original list price (VND)"
    )
    selling_price: float = Field(
        ..., alias="sellingPrice", ge=0, description="Base selling price (VND)"
    )
    final_price: float | None = Field(
        default=None,
        alias="finalPrice",
        description="Discounted final price if promotions apply",
    )
    stock: int = Field(default=0, ge=0, description="Available inventory count")
    weight: int | None = Field(
        default=None, description="Weight in grams for shipping calculation"
    )


class ReviewItemSchema(BaseModel):
    """Customer review item with sizing feedback."""

    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(..., description="Review identifier")
    product_id: int = Field(..., alias="productId", description="Associated product ID")
    rating: int = Field(..., ge=1, le=5, description="Star rating from 1 to 5")
    content: str | None = Field(default=None, description="Customer review text")
    fit: ProductFit | None = Field(
        default=None, description="Customer fit verdict (TIGHT, TRUE_TO_SIZE, LOOSE)"
    )
    user_height: int | None = Field(
        default=None, alias="userHeight", description="Customer height in cm"
    )
    user_weight: int | None = Field(
        default=None, alias="userWeight", description="Customer weight in kg"
    )


class ProductSchema(BaseModel):
    """Complete product details entity returned by catalog endpoints."""

    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(..., description="Product unique identifier")
    name: str = Field(..., description="Product commercial name")
    slug: str = Field(..., description="SEO friendly unique URL slug")
    description: str | None = Field(
        default=None, description="Product detailed description"
    )
    target_gender: GenderTarget = Field(
        default=GenderTarget.UNISEX,
        alias="targetGender",
        description="Target gender (MALE, FEMALE, UNISEX)",
    )
    brand_id: int | None = Field(
        default=None, alias="brandId", description="Brand identifier"
    )
    category_id: int | None = Field(
        default=None, alias="categoryId", description="Category identifier"
    )
    brand: BrandSchema | None = Field(
        default=None, description="Brand details if populated"
    )
    category: CategorySchema | None = Field(
        default=None, description="Category details if populated"
    )
    material: str | None = Field(
        default=None, description="Fabric material composition (e.g. 100% Cotton)"
    )
    care_instructions: str | None = Field(
        default=None,
        alias="careInstructions",
        description="Washing and care instructions",
    )
    season: str | None = Field(
        default=None, description="Season collection (e.g. Xuân Hè, Bốn mùa)"
    )
    provider: str | None = Field(
        default=None, description="Supplier or manufacturer name"
    )
    rating_average: float = Field(
        default=0.0,
        alias="ratingAverage",
        ge=0,
        le=5,
        description="Average customer rating",
    )
    review_count: int = Field(
        default=0,
        alias="reviewCount",
        ge=0,
        description="Total approved customer reviews",
    )
    variants: list[ProductVariantSchema] = Field(
        default_factory=list, description="List of color and size variants"
    )
    images: list[ProductImageSchema] = Field(
        default_factory=list, description="Product image gallery"
    )
    similarity: float | None = Field(
        default=None, description="Cosine similarity score for semantic search"
    )


class ProductFilterParams(BaseModel):
    """Query parameters for filtering products in catalog."""

    search: str | None = Field(
        default=None, description="Text keyword to match product name"
    )
    mode: str | None = Field(
        default="semantic", description="Search mode: 'semantic' or 'lexical'"
    )
    category_id: int | None = Field(default=None, description="Filter by category ID")
    category: str | None = Field(default=None, description="Filter by category name")
    brand_id: int | None = Field(default=None, description="Filter by brand ID")
    target_gender: GenderTarget | None = Field(
        default=None, description="Filter by target gender"
    )
    min_price: float | None = Field(
        default=None, ge=0, description="Minimum price filter"
    )
    max_price: float | None = Field(
        default=None, ge=0, description="Maximum price filter"
    )
    in_stock_only: bool = Field(
        default=True, description="When true, only include products with stock > 0"
    )
    page: int | None = Field(default=1, ge=1, description="Page number")
    limit: int | None = Field(default=20, ge=1, description="Page item limit")


class VoucherCheckResult(BaseModel):
    """Result of checking voucher validity against order amount."""

    model_config = ConfigDict(populate_by_name=True)

    is_valid: bool = Field(
        default=False,
        alias="isValid",
        description="Whether voucher is currently usable",
    )
    code: str = Field(..., description="Voucher promo code")
    name: str | None = Field(default=None, description="Voucher title")
    discount_amount: float = Field(
        default=0.0,
        alias="discountAmount",
        ge=0,
        description="Calculated discount amount in VND",
    )
    discount_type: str | None = Field(
        default=None, alias="discountType", description="PERCENT or FIXED"
    )
    min_order_amount: float | None = Field(
        default=None,
        alias="minOrderAmount",
        description="Minimum order amount required",
    )
    message: str | None = Field(default=None, description="Human explanation message")


class ProductReviewSummary(BaseModel):
    """Aggregated review metrics and fit recommendations for a product."""

    product_id: int
    total_reviews: int = 0
    average_rating: float = 0.0
    dominant_fit: ProductFit | None = None
    fit_counts: dict[str, int] = Field(default_factory=dict)
    fit_recommendation: str | None = None
    recent_reviews: list[ReviewItemSchema] = Field(default_factory=list)


class DiscountItemSchema(BaseModel):
    """Active discount promotion applied on products."""

    model_config = ConfigDict(populate_by_name=True)

    id: int
    name: str
    description: str | None = None
    priority: int = 0
    starts_at: str | None = Field(default=None, alias="startsAt")
    ends_at: str | None = Field(default=None, alias="endsAt")
    active: bool = True
