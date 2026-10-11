# Customer Shop Assistant

Hệ thống trợ lý ảo bán hàng (Customer Shop Assistant) xây dựng bằng **FastAPI**, **LangChain**, và **Ollama**, quản lý môi trường và gói phụ thuộc qua **uv**.

---

## Cấu trúc thư mục

```text
customer-shop-assistant/
├── pyproject.toml
├── .python-version
├── .env.example
├── .gitignore
├── README.md
└── src/
    └── customer_shop_assistant/
        ├── __init__.py
        ├── main.py
        ├── api/
        │   ├── __init__.py
        │   └── routes.py
        ├── core/
        │   ├── __init__.py
        │   └── config.py
        └── services/
            ├── __init__.py
            └── chat_service.py
```

---

## Yêu cầu môi trường

- **Python**: `>= 3.14`
- **uv**: Trình quản lý package & môi trường ảo
- **Ollama**: Đã cài đặt và đang chạy mô hình (mặc định: `qwen3.5:4b`)

---

## Hướng dẫn cài đặt & khởi chạy

### 1. Cài đặt môi trường ảo và thư viện

```powershell
uv sync
```

### 2. Thiết lập biến môi trường

Tạo file `.env` từ file mẫu `.env.example`:

```powershell
Copy-Item .env.example .env
```

Nội dung `.env` mẫu:

```dotenv
APP_NAME="Customer Shop Assistant"
APP_ENV="development"
HOST="0.0.0.0"
PORT=8000
DEBUG=True

OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="qwen3.5:4b"
```

### 3. Tải mô hình Ollama (nếu chưa có)

```powershell
ollama run qwen3.5:4b
```

### 4. Khởi chạy ứng dụng

Chạy bằng script đã cấu hình trong `pyproject.toml`:

```powershell
uv run customer-shop-assistant
```

Hoặc chạy trực tiếp với uvicorn:

```powershell
uv run uvicorn customer_shop_assistant.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API Documentation

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Các Endpoint chính

| Phương thức | Đường dẫn | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/` | Kiểm tra trạng thái root |
| `GET` | `/api/v1/health` | Healthcheck ứng dụng & cấu hình mô hình |
| `POST` | `/api/v1/chat` | Gửi tin nhắn và nhận phản hồi từ trợ lý |
| `POST` | `/api/v1/chat/stream` | Nhận phản hồi dạng streaming |

#### Ví dụ gọi `/api/v1/chat`:

```json
POST /api/v1/chat
Content-Type: application/json

{
  "message": "Xin chào! Bạn có thể tư vấn sản phẩm cho tôi không?",
  "history": []
}
```
