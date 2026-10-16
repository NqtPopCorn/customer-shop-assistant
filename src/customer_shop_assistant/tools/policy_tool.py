from langchain_core.tools import tool

from customer_shop_assistant.services.policy_retriever import PolicyRetriever


@tool
def search_policy(query: str, k: int = 2) -> str:
    """Tra cứu các chính sách cửa hàng: đổi trả hàng, hoàn tiền, freeship,
    thanh toán VietQR/COD, và hướng dẫn bảo quản trang phục.
    """
    retriever = PolicyRetriever()
    results = retriever.retrieve(query.strip(), k=k)

    if not results:
        return "Không tìm thấy điều khoản chính sách nào phù hợp với câu hỏi."

    formatted_sections: list[str] = []
    for chunk in results:
        citation = retriever.format_citation(chunk)
        formatted_sections.append(
            f"Nguồn trích dẫn: {citation}\nNội dung chính sách:\n{chunk.content}"
        )

    return "\n\n---\n\n".join(formatted_sections)
