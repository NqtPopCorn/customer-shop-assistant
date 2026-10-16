from langchain_core.tools import tool

from customer_shop_assistant.services.shop_client import ShopClient


@tool
async def check_voucher(code: str, order_value: float) -> str:
    """Kiểm tra tính hợp lệ và số tiền giảm giá của mã voucher / coupon."""
    client = ShopClient()
    res = await client.check_voucher(code.strip().upper(), order_value)

    if not res.is_valid:
        req_min = (
            f" (Yêu cầu đơn tối thiểu: {res.min_order_amount:,.0f}đ)"
            if res.min_order_amount
            else ""
        )
        msg = res.message or "Không áp dụng được"
        return f"Mã voucher '{res.code}' KHÔNG HỢP LỆ: {msg}{req_min}."

    min_str = (
        f"- Điều kiện đơn tối thiểu: {res.min_order_amount:,.0f}đ\n"
        if res.min_order_amount
        else ""
    )
    name_str = f" ({res.name})" if res.name else ""
    return (
        f"Mã voucher '{res.code}'{name_str} HỢP LỆ!\n"
        f"- Giảm giá: {res.discount_amount:,.0f}đ\n"
        f"{min_str}"
        f"- Trạng thái: {res.message or 'Áp dụng thành công'}"
    )
