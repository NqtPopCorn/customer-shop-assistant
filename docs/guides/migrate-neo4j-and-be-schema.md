# Hướng dẫn Migrate sang Neo4j & Cập nhật DB Schema phía BE

Tài liệu này tổng hợp các thay đổi so với `origin/main` của repo `e-commerce-clothes` (nhánh `feat-update-retrieval`, gồm 2 commit `0768061` và `65424fe`) cùng các bước đồng bộ với `customer-shop-assistant`.

> **Quy ước đường dẫn (Aliases)**:
> - `<E_COMMERCE_DIR>`: thư mục repo `e-commerce-clothes` (ví dụ: `../e-commerce-clothes`)
> - `<ASSISTANT_DIR>`: thư mục repo `customer-shop-assistant` (thư mục gốc hiện tại `.`)

---

## 1. Tóm tắt thay đổi so với `origin/main`

### Phía BE (`<E_COMMERCE_DIR>`)
- **Docker Compose** (`docker-compose.yml`):
  - Thêm service `neo4j` (`neo4j:5-community`, container `fashion-shop-neo4j`, ports `7474:7474` và `7687:7687`, volume `neo4j_data`, auth mặc định `neo4j/fashion_kg_password`).
- **Prisma Schema & Migration** (`fashion-shop-be/prisma/schema.prisma` & `migrations/20261124120000_upgrade_product_embedding_bge_m3_and_hybrid_search/migration.sql`):
  - Bật PostgreSQL extensions: `vector`, `unaccent`.
  - Xóa cột `embedding` cũ (`vector(768)`) và tạo lại `embedding vector(1024)` cho model `bge-m3`.
  - Thêm các cột mới vào bảng `products`:
    - `embedding_model` (`TEXT`)
    - `pattern` (`TEXT`)
    - `fit` (`TEXT`)
    - `occasion` (`TEXT`)
    - `search_vector` (`tsvector`)
  - Tạo index:
    - `products_embedding_idx` (HNSW, `vector_cosine_ops`)
    - `products_search_vector_idx` (GIN trên `search_vector`)
- **Data Backfill & Embedding Scripts**:
  - Thêm `fashion-shop-be/prisma/backfill-product-attrs.ts` để điền `pattern`, `fit`, `occasion` cho dữ liệu sản phẩm hiện có.
  - Cập nhật `fashion-shop-be/prisma/seed-embeddings.ts`, `seed-300.ts`, `seed-1000.ts` để sinh embedding 1024 chiều (`bge-m3`) và cập nhật `search_vector`.
- **Hybrid Search API**:
  - Thêm `POST /api/products/search` (`product-search.service.ts`, `product-search.repository.ts`, `search-products.dto.ts`) kết hợp Dense (pgvector) + Full-Text Search (`tsvector`) với RRF.
- **Biến môi trường mới** (`fashion-shop-be/.env.example`):
  - `OLLAMA_BASE_URL`, `OLLAMA_EMBEDDING_MODEL="bge-m3"`, `OLLAMA_EMBEDDING_DIM=1024`, `SEMANTIC_MIN_SIMILARITY=0.35`.

### Phía AI Assistant (`<ASSISTANT_DIR>`)
- Thêm dependency `neo4j` và cấu hình `OUTFIT_KG_BACKEND`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`.
- Đổi `OLLAMA_EMBEDDING_MODEL` mặc định sang `bge-m3`.
- Thêm script `scripts/build_snapshot_manifest.py` và `scripts/seed_neo4j_kg.py` để validate & seed Category-level Outfit KG vào Neo4j.

---

## 2. Các bước thực hiện (Windows PowerShell)

### Bước 1: Chuẩn bị model embedding `bge-m3` trên Ollama
```powershell
ollama pull bge-m3
```

### Bước 2: Khởi động PostgreSQL và Neo4j bằng Docker
```powershell
cd <E_COMMERCE_DIR>
docker-compose up -d
```
Kiểm tra container đang chạy:
```powershell
docker-compose ps
```
> Neo4j Browser truy cập tại `http://localhost:7474` (User: `neo4j`, Password: `fashion_kg_password`).

### Bước 3: Cập nhật `.env` và migrate DB phía BE
1. Mở `<E_COMMERCE_DIR>/fashion-shop-be/.env` và bổ sung/cập nhật các biến sau (theo `.env.example`):
   ```env
   OLLAMA_BASE_URL="http://localhost:11434"
   OLLAMA_EMBEDDING_MODEL="bge-m3"
   OLLAMA_EMBEDDING_DIM=1024
   SEMANTIC_MIN_SIMILARITY=0.35
   ```
2. Chạy Prisma migration và generate client:
   ```powershell
   cd <E_COMMERCE_DIR>/fashion-shop-be
   npm run prisma:deploy
   npm run prisma:generate
   ```
   > **Lưu ý**: Migration này sẽ drop cột `embedding` cũ (`vector(768)`) và tạo mới `vector(1024)`, nên bắt buộc chạy lại bước tạo embedding bên dưới.

3. Backfill thuộc tính sản phẩm (`pattern`, `fit`, `occasion`):
   - **Nếu giữ dữ liệu DB hiện tại**:
     ```powershell
     npx tsx prisma/backfill-product-attrs.ts
     ```
   - **Hoặc nếu muốn seed mới 1000 sản phẩm**:
     ```powershell
     npm run prisma:seed:1000
     ```

4. Re-embed toàn bộ sản phẩm với `bge-m3` và cập nhật `search_vector`:
   ```powershell
   npm run embed:products
   ```

5. Khởi động lại BE:
   ```powershell
   npm run start:dev
   ```

### Bước 4: Cấu hình & Seed Outfit KG sang Neo4j phía AI Assistant
1. Mở `<ASSISTANT_DIR>/.env` và cập nhật:
   ```env
   OLLAMA_EMBEDDING_MODEL="bge-m3"

   # Outfit Knowledge Graph Settings
   OUTFIT_KG_ENABLED=True
   OUTFIT_KG_BACKEND="neo4j"
   COUNTERFACTUAL_POLICY_ENABLED=True
   SNAPSHOT_CATALOG_FILENAME="hm_seed_1000.json"

   # Neo4j Category-level KG Settings
   NEO4J_URI="bolt://localhost:7687"
   NEO4J_USER="neo4j"
   NEO4J_PASSWORD="fashion_kg_password"
   NEO4J_DATABASE="neo4j"
   ```

2. Đồng bộ thư viện (cài driver `neo4j`), kiểm tra snapshot và seed dữ liệu KG vào Neo4j:
   ```powershell
   cd <ASSISTANT_DIR>
   uv sync
   uv run python scripts/build_snapshot_manifest.py
   uv run python scripts/seed_neo4j_kg.py --reset
   ```
   > Có thể chạy `uv run python scripts/seed_neo4j_kg.py --validate-only` trước nếu chỉ muốn kiểm tra ontology mà chưa ghi vào Neo4j.

---

## 3. Kiểm tra sau khi Migrate

1. **Kiểm tra Neo4j KG**:
   - Mở `http://localhost:7474` và chạy Cypher:
     ```cypher
     MATCH (n) RETURN labels(n) AS label, count(n) AS total;
     ```
     (Phải thấy đủ các node `Category`, `Style`, `Occasion`, `Color`.)
   - Chạy test đối chiếu giữa In-Memory KG và Neo4j KG:
     ```powershell
     cd <ASSISTANT_DIR>
     $env:DEBUG="false"
     uv run pytest tests/test_neo4j_kg_parity.py
     ```

2. **Kiểm tra Hybrid Search phía BE**:
   ```powershell
   Invoke-RestMethod -Method Post -Uri "http://localhost:3001/api/products/search" `
     -ContentType "application/json" `
     -Body '{"query":"áo sơ mi nam đi làm","limit":5}'
   ```
   (Đảm bảo trả về `items` có các trường `pattern`, `fit`, `occasion`, `scores` và `meta.strategy` là `hybrid_rrf`.)
