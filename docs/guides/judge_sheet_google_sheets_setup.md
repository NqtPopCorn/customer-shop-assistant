# Hướng dẫn Public Judge Sheet qua Google Sheets + Apps Script Bridge

Tài liệu này hướng dẫn cách đưa file [eval/e1_tier2_judge_sheet.csv](file:///h:/Code/PET/customer-shop-assistant/eval/e1_tier2_judge_sheet.csv) lên Google Sheets để nhiều thành viên trong team cùng chấm trên một bảng chung (chia 72 ca, lưu tên người chấm `judge_name`, thời gian `updated_at`, và cảnh báo xung đột nếu 2 người sửa cùng một dòng).

---

## Phần 1: Đưa file CSV lên Google Sheets & Cài Apps Script Bridge

1. Mở [Google Sheets](https://sheets.google.com) và tạo một bảng tính mới.
2. Chọn **Tệp (File) → Nhập (Import) → Tải lên (Upload)** → chọn file `eval/e1_tier2_judge_sheet.csv`:
   - Vị trí nhập: **Thay thế trang tính hiện tại** (*Replace current sheet*).
   - Loại dấu phân cách: **Tự động phát hiện** (*Detect automatically*).
3. Mở **Tiện ích mở rộng (Extensions) → Apps Script**.
4. Xoá mã mẫu mặc định, sao chép toàn bộ nội dung file [scripts/google_apps_script_judge_bridge.gs](file:///h:/Code/PET/customer-shop-assistant/scripts/google_apps_script_judge_bridge.gs) dán vào.
5. Thiết lập mã bảo vệ chung (Passcode) cho team:
   - Cách 1 (khuyên dùng): Vào **Project Settings (⚙️) → Script Properties → Add script property**, đặt Property `JUDGE_TOKEN` với giá trị mật khẩu của team.
   - Cách 2: Sửa trực tiếp hằng số `DEFAULT_TOKEN` ở đầu script.
6. Lưu dự án (`Ctrl+S`), sau đó chọn **Triển khai (Deploy) → Triển khai mới (New deployment)**:
   - Chọn loại: **Ứng dụng web (Web app)**.
   - Thực thi với tư cách (**Execute as**): **Tôi (Me)**.
   - Ai có quyền truy cập (**Who has access**): **Bất kỳ ai (Anyone)**.
   - Bấm **Triển khai**, cấp quyền truy cập Google Sheets, rồi sao chép **URL ứng dụng web** (`https://script.google.com/macros/s/.../exec`).

> [!NOTE]
> Khi có người chấm ca đầu tiên, Apps Script sẽ tự động thêm 2 cột `judge_name` và `updated_at` vào cuối bảng tính để theo dõi người chấm và chống ghi đè.

---

## Phần 2: Hai cách team mở UI để cùng chấm

### Cách A — Không cần cài Python (Gửi thẳng 1 file HTML hoặc host tĩnh)

1. Gửi file [src/customer_shop_assistant/ui/judge_sheet.html](file:///h:/Code/PET/customer-shop-assistant/src/customer_shop_assistant/ui/judge_sheet.html) cho các thành viên (hoặc đẩy lên GitHub Pages / static host nội bộ).
2. Thành viên mở file `judge_sheet.html` trực tiếp bằng trình duyệt (Chrome / Edge).
3. Hộp thoại **Kết nối & Danh tính giám khảo** sẽ tự mở:
   - Chọn nguồn dữ liệu: **Google Sheets Bridge (Apps Script)**.
   - Nhập **Tên giám khảo** (ví dụ: `Minh`, `An`, `Trường`).
   - Dán **Apps Script Web App URL** và **Passcode (token)**.
   - Bấm **Lưu & Kết nối**.

> [!TIP]
> Nếu host file HTML qua HTTP/HTTPS (ví dụ GitHub Pages hoặc server local), bạn có thể bấm nút **Copy link cấu hình** trong hộp thoại Kết nối để gửi link có sẵn `?bridge=...&token=...` cho team.

### Cách B — Chạy local qua FastAPI (`run_judge_ui.ps1`)

1. Thêm 2 biến vào file `.env` (không commit `.env`):
   ```env
   JUDGE_BRIDGE_URL="https://script.google.com/macros/s/YOUR_DEPLOY_ID/exec"
   JUDGE_BRIDGE_TOKEN="your-team-passcode"
   ```
2. Khởi chạy server bằng [run_judge_ui.ps1](file:///h:/Code/PET/customer-shop-assistant/run_judge_ui.ps1):
   ```powershell
   .\run_judge_ui.ps1
   ```
3. Mở trình duyệt tại `http://127.0.0.1:8000/judge`. UI sẽ tự nhận cấu hình Bridge từ `/api/v1/judge/config`; bạn chỉ cần nhập **Tên giám khảo** là bắt đầu chấm ngay.
   - Bất kỳ lúc nào cũng có thể bấm nút **Kết nối** trên thanh công cụ để chuyển qua lại giữa **Local CSV** và **Google Sheet**.

---

## Phần 3: Quy trình chia việc & Chống xung đột khi chấm chung

1. **Chia việc nhanh cho team**:
   - Dùng bộ lọc **Trạng thái → `Chưa có người chấm`** để chỉ hiển thị các ca chưa ai nhận, hoặc phân chia theo Suite (`contrastive` / `stratified`) hoặc theo Scenario ID.
   - Sau khi chọn điểm và ghi nhận xét cho một ca (hoặc các cột đang so sánh), bấm nút **Lưu thay đổi (`Ctrl+S`)** để ghi lên Sheet/CSV và tự động gắn tên bạn vào cột `judge_name` (cách lưu chủ động này tránh xung đột do độ trễ mạng của Google Sheets). Nếu chuyển ca khi chưa lưu, giao diện sẽ hỏi lưu hoặc bỏ thay đổi.
   - Khi muốn lấy dữ liệu mới nhất từ các thành viên khác, bấm nút **🔄 Đồng bộ** trên thanh công cụ.
2. **Xử lý xung đột đồng thời (Optimistic Locking)**:
   - Nếu 2 người vô tình mở và sửa cùng một dòng, hệ thống sẽ kiểm tra `updated_at` và bật hộp thoại **Phát hiện xung đột dữ liệu**, cho phép bạn xem song song 2 bản điểm và chọn **Tải bản mới nhất trên Sheet** hoặc **Ghi đè bằng bản của tôi**.
3. **Tổng hợp & Xuất CSV về repo**:
   - Bấm **Tổng hợp điểm** để xem bảng thống kê theo mode và copy thẳng bảng Markdown vào báo cáo nghiên cứu.
   - Khi team chấm xong, bấm **Tải CSV** trên UI (hoặc tải từ Google Sheets dạng CSV UTF-8) để lưu đè lại vào `eval/e1_tier2_judge_sheet.csv`.

