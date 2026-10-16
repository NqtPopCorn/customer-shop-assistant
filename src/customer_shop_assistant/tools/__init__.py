from customer_shop_assistant.tools.catalog_tool import (
    get_product_detail,
    search_products,
)
from customer_shop_assistant.tools.policy_tool import search_policy
from customer_shop_assistant.tools.size_tool import recommend_size
from customer_shop_assistant.tools.voucher_tool import check_voucher

ALL_TOOLS = [
    search_products,
    get_product_detail,
    recommend_size,
    check_voucher,
    search_policy,
]

__all__ = [
    "ALL_TOOLS",
    "search_products",
    "get_product_detail",
    "recommend_size",
    "check_voucher",
    "search_policy",
]
