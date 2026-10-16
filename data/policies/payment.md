# Hướng dẫn & Phương thức Thanh toán (Payment Policy)

> **Mã tài liệu:** POL-PAY-01  
> **Áp dụng:** Tất cả giao dịch đặt hàng tại hệ thống Fashion Shop  
> **Cập nhật lần cuối:** 16/10/2026

---

## 1. Các phương thức thanh toán được hỗ trợ

Shop hỗ trợ 3 phương thức thanh toán an toàn, tiện lợi và linh hoạt nhất hiện nay tại Việt Nam:

### A. Thanh toán tiền mặt khi nhận hàng (COD - Cash On Delivery)
- **Cách thức hoạt động**: Khách hàng kiểm tra hàng và thanh toán trực tiếp số tiền trên hóa đơn cho nhân viên giao hàng (shipper) khi nhận kiện hàng.
- **Phạm vi áp dụng**: 100% tất cả các tỉnh thành và địa chỉ trên toàn quốc.
- **Phí thu hộ (Phí COD)**: **Hoàn toàn miễn phí** (shop chịu phí thu hộ cho khách hàng).
- **Lưu ý**: Khách hàng vui lòng chuẩn bị số tiền mặt tương ứng để việc giao nhận diễn ra nhanh chóng.

### B. Thanh toán quét mã QR tự động (VietQR / Chuyển khoản nhanh 24/7)
- **Cách thức hoạt động**:
  - Tại trang xác nhận đơn hàng, hệ thống sẽ tự động tạo một mã **VietQR động** chứa chính xác số tiền cần thanh toán và **mã nội dung chuyển khoản độc nhất** (gắn liền với mã đơn hàng).
  - Khách hàng mở ứng dụng ngân hàng bất kỳ (Vietcombank, MBBank, Techcombank, VPBank, ACB...) hoặc ví điện tử (MoMo, ZaloPay, Viettel Money) và quét mã QR.
- **Ưu điểm**:
  - **Khớp lệnh tức thì**: Tiền vào tài khoản hệ thống sẽ tự động xác nhận `PAID` trong vòng 5 - 15 giây qua webhook tích hợp SePay/MBBank.
  - Không cần nhập tay số tài khoản hoặc số tiền, tránh tuyệt đối sai sót chuyển nhầm.
- **Thông tin ngân hàng thụ hưởng chính thức của shop**:
  - Ngân hàng: **MBBank (Ngân hàng Quân Đội)**
  - Tên chủ tài khoản: **FASHION SHOP OFFICIAL**
  - Số tài khoản: **0987654321**

### C. Chuyển khoản ngân hàng truyền thống (Bank Transfer thủ công)
- Dành cho khách hàng không quét được mã QR. Khách hàng chuyển khoản trực tiếp vào tài khoản ngân hàng của shop và ghi rõ nội dung chuyển khoản là: `MADONHANG SĐT`.
- Sau khi chuyển khoản, đơn hàng sẽ được bộ phận kế toán kiểm tra và xác nhận thủ công trong vòng 15 - 30 phút trong giờ hành chính.

---

## 2. Quy định an toàn và bảo mật giao dịch

1. **Tuyệt đối không thanh toán vào tài khoản cá nhân nhân viên**: Shop chỉ sử dụng duy nhất tài khoản doanh nghiệp chính thức mang tên **FASHION SHOP OFFICIAL**. Mọi yêu cầu chuyển tiền vào tài khoản cá nhân khác đều là giả mạo.
2. **Hóa đơn và chứng từ**: Mỗi đơn hàng thành công đều có biên lai giao dịch điện tử lưu lại trên tài khoản khách hàng và hóa đơn bán lẻ gửi kèm trong kiện hàng.
3. **Chính sách hoàn tiền (Refund)**:
   - Các trường hợp hoàn tiền do hủy đơn trước khi giao, trả hàng do lỗi kỹ thuật hoặc đổi hàng chênh lệch giá sẽ được chuyển khoản trực tiếp về số tài khoản của khách hàng trong vòng 24 - 48 giờ làm việc.
