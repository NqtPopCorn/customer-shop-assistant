# Business Requirements — Customer Shop Assistant

> **Phiên bản:** 1.0  
> **Ngày:** 16/10/2026  
> **Trạng thái:** Đã thống nhất theo PRD v1.0  
> **Ngôn ngữ sản phẩm:** Tiếng Việt  
> **Tài liệu nguồn:** PRD — Chatbot tư vấn khách hàng shop thời trang (Customer Shop Assistant)

---

## 1. Mục đích tài liệu

Tài liệu này xác định nhu cầu nghiệp vụ, mục tiêu, phạm vi, yêu cầu cấp nghiệp vụ, quy tắc vận hành và tiêu chí nghiệm thu cho Customer Shop Assistant. Đây là cơ sở để phân rã thành functional requirements, thiết kế kỹ thuật, kế hoạch triển khai và bộ kiểm thử.

Tài liệu không thay thế đặc tả API, thiết kế kiến trúc, schema dữ liệu hoặc kế hoạch thí nghiệm chi tiết.

## 2. Bối cảnh nghiệp vụ

Shop thời trang cần cung cấp một trợ lý hội thoại giúp khách hàng tìm và lựa chọn sản phẩm bằng ngôn ngữ tự nhiên. Khách hàng có thể cần tìm sản phẩm theo loại quần áo, ngân sách, thương hiệu, màu sắc, kích thước, chất liệu, mùa hoặc phong cách; kiểm tra giá, tồn kho, khuyến mãi; tham khảo size; và tìm hiểu chính sách mua sắm.

Customer Shop Assistant là dịch vụ tư vấn độc lập. Dịch vụ sử dụng dữ liệu từ REST API của `fashion-shop-be` làm nguồn dữ liệu nghiệp vụ có thẩm quyền, kết hợp tìm kiếm theo thuộc tính và tìm kiếm ngữ nghĩa. Nội dung chính sách shop được cung cấp qua kho tài liệu nội bộ có thể truy xuất và trích dẫn.

Dự án đồng thời là nền tảng nghiên cứu thực nghiệm về AI Agent, RAG và chiến lược hỏi lại làm rõ (clarification strategy).

## 3. Mục tiêu nghiệp vụ

- **BO-01 — Hỗ trợ tìm sản phẩm:** Giúp khách hàng mô tả nhu cầu bằng tiếng Việt tự nhiên và nhận được các sản phẩm phù hợp.
- **BO-02 — Cung cấp thông tin đáng tin cậy:** Đảm bảo giá, tồn kho, khuyến mãi, voucher và thông tin sản phẩm dựa trên dữ liệu được hệ thống shop cung cấp.
- **BO-03 — Hỗ trợ quyết định chọn sản phẩm, size:** Đưa ra gợi ý sản phẩm kèm các size, variant phù hợp có căn cứ, minh bạch về mức độ chắc chắn và có thể tham khảo phản hồi fit từ khách mua trước.
- **BO-04 — Giải đáp chính sách:** Trả lời câu hỏi về chính sách shop kèm nguồn dẫn và không tự bịa khi thiếu thông tin.
- **BO-05 — Tối ưu trải nghiệm hội thoại:** Giữ câu trả lời tiếng Việt tối giản, trực diện, lịch thiệp và hạn chế lượt hỏi không cần thiết.
- **BO-06 — Hỗ trợ nghiên cứu:** Cho phép chạy hội thoại tương tác và evaluation hàng loạt, đồng thời đo lường chất lượng, độ trễ và token usage.
- **BO-07 — Duy trì ranh giới hệ thống:** Giữ `fashion-shop-be` làm nguồn dữ liệu nghiệp vụ chuẩn; Assistant không thay thế backend thương mại hoặc truy cập cơ sở dữ liệu theo cách bỏ qua API nghiệp vụ.

## 4. Stakeholders và nhóm người dùng

| Nhóm | Nhu cầu / trách nhiệm |
|---|---|
| Khách hàng tiềm năng | Tìm sản phẩm, kiểm tra giá và tồn kho, nhận tư vấn size, tra cứu khuyến mãi và chính sách trước khi mua |
| Nhà phát triển | Dùng API và CLI để phát triển, debug, kiểm thử hội thoại và truy vấn công cụ |
| Nhà nghiên cứu | Chạy Golden Set, so sánh các chế độ clarification và kiến trúc/model theo chỉ số nhất quán |
| Nhóm phát triển Main BE | Cung cấp dữ liệu chuẩn và các endpoint nghiệp vụ cần thiết |
| Chủ shop / đội vận hành | Duy trì dữ liệu sản phẩm, khuyến mãi, voucher và chính sách shop chính xác |

## 5. Phạm vi nghiệp vụ

### 5.1 Trong phạm vi v1

1. Tư vấn cho các nhóm sản phẩm thời trang có trong catalog của shop.
2. Tìm kiếm sản phẩm bằng ngôn ngữ tự nhiên, kết hợp bộ lọc có cấu trúc và tìm kiếm ngữ nghĩa.
3. Đề xuất sản phẩm ngay hoặc hỏi làm rõ tùy chế độ clarification được cấu hình.
4. Hiển thị giá sau giảm và trạng thái còn hàng dựa trên dữ liệu Main BE.
5. Thông báo sản phẩm/size hết hàng khi khách hỏi đích danh và đề xuất 1–2 lựa chọn thay thế còn hàng nếu tìm được.
6. Tư vấn size dựa trên bảng size xác định trước và tín hiệu review fit khi đủ dữ liệu.
7. Tra cứu chiến dịch giảm giá, discount và điều kiện voucher qua API Main BE.
8. Trả lời FAQ/chính sách từ tài liệu nội bộ và trích dẫn tên chính sách/điều khoản liên quan.
9. Từ chối ngắn gọn các yêu cầu ngoài phạm vi đã xác định.
10. Cung cấp FastAPI REST API và CLI tương tác/debug/evaluation.
11. Ghi telemetry từng lượt chat để phục vụ debug và nghiên cứu.
12. Chạy Golden Set evaluation để so sánh chất lượng và các chế độ clarification.

### 5.2 Ngoài phạm vi v1

- Tra cứu dữ liệu cá nhân hoặc thông tin đơn hàng của khách hàng.
- Thực hiện giao dịch thay khách hàng: thêm giỏ hàng, đặt hàng, áp voucher khi thanh toán hoặc thanh toán.
- Hủy đơn, đổi địa chỉ, xử lý đổi trả theo từng đơn hoặc các thao tác yêu cầu xác thực người dùng.
- Tích hợp kênh bên ngoài như Zalo OA, Facebook Messenger hoặc Telegram.
- Xây dựng widget giao diện web hoàn chỉnh; v1 cung cấp API để tích hợp sau.
- Phân tích hình ảnh, thử đồ ảo hoặc các chức năng vision multimodal.

## 6. Yêu cầu nghiệp vụ (Business Requirements)

### BR-01 — Tìm kiếm và tư vấn sản phẩm

Hệ thống phải giúp khách hàng tìm sản phẩm từ mô tả tự nhiên, bao gồm cả trường hợp khách hàng không biết tên hoặc mã sản phẩm chính xác.

**Kết quả kỳ vọng:** Hệ thống có thể sử dụng các tiêu chí được nhận diện để truy xuất, xếp hạng và trình bày sản phẩm liên quan.

### BR-02 — Tìm kiếm kết hợp thuộc tính và ngữ nghĩa

Hệ thống phải hỗ trợ đồng thời:

- Tiêu chí có cấu trúc như danh mục, thương hiệu, khoảng giá, màu sắc, size, mùa và chất liệu khi dữ liệu nguồn hỗ trợ.
- Mô tả ngữ nghĩa như phong cách, mục đích sử dụng hoặc yêu cầu diễn đạt bằng ngôn ngữ tự nhiên.

**Quy tắc:** Điểm phù hợp ngữ nghĩa không được dùng để bù cho việc vi phạm điều kiện mà khách hàng nêu rõ là bắt buộc.

### BR-03 — Hành vi hỏi lại làm rõ có thể cấu hình

Hệ thống phải hỗ trợ ba chế độ clarification để có thể so sánh hành vi tư vấn:

- `IMMEDIATE_ANSWER`: tìm và đề xuất ngay theo thông tin hiện có; khi thiếu số đo, có thể đưa bảng size tham khảo để khách tự đối chiếu, không hỏi lại.
- `MINIMAL_CLARIFICATION`: mặc định; với yêu cầu sản phẩm thông thường, tìm ngay. Chỉ hỏi một câu làm rõ khi khách hỏi trực tiếp về chọn size nhưng thiếu thông tin cần thiết; câu hỏi có thể thu thập chiều cao, cân nặng và form dáng mong muốn.
- `ALWAYS_CLARIFY`: hỏi làm rõ nhu cầu trước khi đưa ra gợi ý sản phẩm/size.

Chế độ có thể được cấu hình qua thiết lập hệ thống hoặc trong request theo hợp đồng đã thống nhất.

### BR-04 — Minh bạch về tồn kho

- Với yêu cầu tìm kiếm/gợi ý chung, chỉ đề xuất sản phẩm hoặc biến thể còn hàng (`stock > 0`).
- Nếu khách hỏi đích danh sản phẩm hoặc size đã hết hàng, thông báo rõ tình trạng hết hàng.
- Khi có lựa chọn phù hợp, chủ động đề xuất 1–2 mẫu thay thế tương đương còn hàng.
- Không mô tả sản phẩm/biến thể là còn hàng nếu dữ liệu nguồn không xác nhận điều đó.

### BR-05 — Chính xác về giá và giảm giá

Hệ thống phải hiển thị giá sau giảm từ kết quả API Main BE. Mọi con số liên quan đến discount, campaign, voucher hoặc giá phải thanh toán phải lấy từ nguồn nghiệp vụ có thẩm quyền; Assistant không được tự tính hoặc suy diễn thay cho quy tắc của Main BE.

### BR-06 — Tư vấn chọn size có căn cứ

Hệ thống phải hỗ trợ gợi ý size dựa trên bảng size có cấu trúc theo danh mục và đối tượng. Kết quả phải nêu 1–2 size phù hợp khi có đủ dữ liệu và thể hiện mức độ chắc chắn phù hợp.

Khi sản phẩm có ít nhất 10 đánh giá và có tín hiệu fit đáng tin cậy, hệ thống nên sử dụng các nhãn `TIGHT`, `TRUE_TO_SIZE` hoặc `LOOSE` để bổ sung lưu ý thực tế. Tín hiệu review phải được trình bày như xu hướng tham khảo, không phải cam kết chắc chắn về độ vừa.

### BR-07 — Tra cứu khuyến mãi và voucher

Hệ thống phải hỗ trợ tra cứu:

- Campaign đang diễn ra và discount áp dụng cho sản phẩm.
- Mã voucher có tồn tại hay không.
- Trạng thái active và thời gian hiệu lực.
- Điều kiện giá trị đơn hàng tối thiểu.
- Giới hạn lượt dùng/ngân sách khi API nguồn cung cấp các thông tin này.

Assistant phải phản ánh kết quả kiểm tra từ Main BE và không khẳng định voucher hợp lệ nếu chưa có đủ dữ kiện xác nhận.

### BR-08 — Hỏi đáp chính sách có nguồn dẫn

Hệ thống phải hỗ trợ các chủ đề chính sách được cung cấp trong kho tài liệu nội bộ, gồm:

- Đổi trả hàng.
- Phí và thời gian vận chuyển.
- Phương thức thanh toán: COD, VietQR và chuyển khoản.
- Giặt ủi, bảo quản sản phẩm.

Câu trả lời phải nêu tên chính sách/điều khoản làm căn cứ. Nếu không tìm thấy căn cứ phù hợp, hệ thống phải nói rõ chưa có thông tin hoặc chưa có quy định cho trường hợp đó; không tự tạo chính sách.

### BR-09 — Từ chối yêu cầu ngoài phạm vi

Đối với yêu cầu tra cứu đơn hàng cá nhân, hủy đơn, thao tác thanh toán đơn cụ thể hoặc nghiệp vụ không được hỗ trợ, hệ thống phải từ chối ngắn gọn và hướng khách hàng liên hệ nhân viên shop khi phù hợp.

Thông điệp chuẩn có thể dùng:

> Hiện tại mình chưa hỗ trợ tra cứu đơn hàng/thao tác cá nhân. Bạn vui lòng liên hệ nhân viên shop để được hỗ trợ nhé.

### BR-10 — Phong cách giao tiếp

Câu trả lời phải bằng tiếng Việt, theo phong cách tối giản, trực diện, lịch thiệp và khách quan; xưng hô “mình - bạn”. Ưu tiên trả lời trực tiếp dữ kiện cần tìm, tránh xã giao kéo dài và văn phong quảng cáo không cần thiết. Tiền tệ hiển thị theo định dạng quen thuộc, ví dụ `250.000đ`.

### BR-11 — Hai hình thức sử dụng cho v1

Hệ thống phải cung cấp:

- REST API qua FastAPI để client gửi yêu cầu chat và nhận câu trả lời cùng metadata/sản phẩm gợi ý.
- CLI để hội thoại tương tác, chạy chế độ debug và chạy evaluation hàng loạt.

API hỗ trợ response JSON và/hoặc streaming SSE theo hợp đồng API được thống nhất khi thiết kế chi tiết.

### BR-12 — Ghi log và khả năng truy vết

Hệ thống phải ghi log từng lượt hội thoại ở mức đủ phục vụ debug và thực nghiệm, gồm session, nội dung yêu cầu, chế độ clarification, tool call, tài liệu truy xuất, latency, token usage và câu trả lời cuối.

Log phải giúp nhóm phát triển xác định công cụ nào được gọi, dữ liệu nào được truy xuất và thời gian xử lý. Dữ liệu nhạy cảm ngoài phạm vi không được chủ động thu thập; cần có chính sách lưu trữ và loại bỏ log phù hợp trước khi triển khai thực tế.

### BR-13 — Evaluation tái lập được

Hệ thống phải cung cấp bộ Golden Set và runner tự động để đánh giá chất lượng trả lời, độ trung thực dữ kiện, hành vi tool, xử lý edge case và tuân thủ chế độ clarification. Cùng một bộ dữ liệu và cấu hình phải có thể chạy lại để so sánh kết quả.

## 7. Quy tắc nghiệp vụ

1. Main BE là nguồn có thẩm quyền cho sản phẩm, biến thể, giá sau giảm, tồn kho, chiến dịch, discount, voucher và review được cung cấp qua API.
2. Assistant không được truy cập trực tiếp database theo cách bỏ qua API nghiệp vụ và không được để LLM tự viết SQL.
3. Kết quả tìm kiếm chung chỉ gồm sản phẩm/biến thể còn hàng.
4. Sản phẩm hết hàng chỉ được nhắc đến khi cần phản hồi yêu cầu đích danh; phải ghi rõ hết hàng và chỉ thay thế bằng lựa chọn còn hàng đã kiểm tra.
5. Giá, điều kiện voucher và thông tin khuyến mãi phải theo kết quả Main BE; không tự suy diễn.
6. Quy tắc tính size phải dựa trên bảng size được quản lý rõ ràng; review fit là tín hiệu tham khảo bổ sung.
7. Chỉ dùng review fit để đưa lưu ý theo xu hướng khi số lượng review đạt ngưỡng quy định (`reviewCount >= 10`).
8. Nội dung chính sách chỉ được trả lời khi có căn cứ trong tài liệu truy xuất; câu trả lời cần nêu nguồn chính sách/điều khoản.
9. Nếu hệ thống không lấy được dữ liệu nguồn do lỗi, phải phân biệt lỗi truy vấn với việc không có sản phẩm hoặc chính sách phù hợp.
10. Chế độ clarification đang có hiệu lực phải được tuân thủ và ghi nhận trong telemetry.
11. Mỗi lượt xử lý phải tuân thủ giới hạn tối đa 5 bước và timeout API 3 giây theo cấu hình v1; cách xử lý khi chạm giới hạn phải có kiểm soát.
12. Assistant chỉ hỗ trợ tư vấn và tra cứu trong v1; không thực hiện giao dịch thay khách hàng.

## 8. Luồng nghiệp vụ cấp cao

1. Khách hàng gửi câu hỏi hoặc yêu cầu tư vấn.
2. Hệ thống nhận diện nhóm nhu cầu: sản phẩm/size, khuyến mãi/voucher, chính sách/FAQ hoặc ngoài phạm vi.
3. Với yêu cầu liên quan, hệ thống áp dụng chế độ clarification được cấu hình.
4. Hệ thống gọi công cụ phù hợp để lấy dữ liệu từ Main BE hoặc truy xuất tài liệu chính sách cục bộ.
5. Dữ kiện được kiểm tra theo quy tắc nghiệp vụ, gồm tồn kho, giá, điều kiện voucher và nguồn trích dẫn.
6. Assistant tạo phản hồi tối giản dựa trên dữ liệu đã truy xuất.
7. Hệ thống trả nội dung, sản phẩm gợi ý và metadata phù hợp qua API hoặc CLI.
8. Hệ thống ghi telemetry cho lượt xử lý.
9. Khách hàng có thể tiếp tục hội thoại hoặc tinh chỉnh yêu cầu trong cùng session.

## 9. Yêu cầu dữ liệu và phụ thuộc nghiệp vụ

### 9.1 Dữ liệu từ Main BE

Assistant cần khả năng truy xuất các nhóm dữ liệu sau, qua API đã có hoặc API cần bổ sung:

- Danh mục, thương hiệu, sản phẩm và biến thể.
- Size, màu sắc, mã màu, giá niêm yết, giá bán/giá sau giảm và tồn kho.
- Campaign, discount và voucher cùng điều kiện áp dụng.
- Review sản phẩm có nhãn fit, chiều cao/cân nặng và thông tin liên quan khi có dữ liệu.
- Thuộc tính đối tượng `targetGender` để hỗ trợ phân loại bảng size phù hợp.
- Tìm kiếm semantic/hybrid có lọc cứng qua PostgreSQL/pgvector hoặc endpoint tương đương.

### 9.2 Dữ liệu do Assistant quản lý

- `data/size_charts.json`: bảng size theo danh mục và đối tượng.
- `data/policies/shipping.md`: chính sách vận chuyển.
- `data/policies/returns.md`: chính sách đổi trả.
- `data/policies/payment.md`: phương thức thanh toán.
- `data/policies/care_guide.md`: hướng dẫn giặt ủi và bảo quản.
- `eval/questions.json`: bộ câu hỏi Golden Set.

Nguồn dữ liệu nội bộ của Assistant không thay thế nguồn dữ liệu giao dịch của Main BE.

## 10. Tiêu chí thành công và nghiệm thu

### 10.1 Mục tiêu định lượng v1

| Chỉ số | Cách đo | Ngưỡng mục tiêu |
|---|---|---:|
| Tính trung thực dữ kiện (giá sau giảm, tồn kho, voucher) | Đối chiếu với kết quả API Main BE | >= 95% |
| Chọn đúng tool và tham số | So sánh chuỗi tool calls với ground truth | >= 85% |
| Độ chính xác RAG chính sách | Top-3 chunk chứa căn cứ đúng | >= 85% |
| Xử lý hàng hết tồn kho | Thông báo đúng và có sản phẩm thay thế còn hàng khi tìm được | >= 90% |
| Tuân thủ chế độ clarification | Hành vi khớp mode đang bật | >= 95% |
| Từ chối đúng ngoài phạm vi | Từ chối đúng, không bịa thông tin | >= 95% |
| Độ trễ phản hồi | Đo theo turn trên môi trường đánh giá mục tiêu | < 4.0 giây |

Các ngưỡng trên là mục tiêu nghiệm thu của PRD. Báo cáo đánh giá phải ghi rõ phần cứng, model, cấu hình, kích thước tập test và phương pháp đo; không được suy rộng kết quả ngoài môi trường đã kiểm thử.

### 10.2 Điều kiện nghiệm thu nghiệp vụ

MVP được nghiệm thu khi:

- Khách hàng có thể tìm và hỏi tư vấn sản phẩm qua API.
- Ba chế độ clarification hoạt động theo đúng định nghĩa.
- Kết quả tìm kiếm chung chỉ đưa ra sản phẩm/biến thể còn hàng.
- Yêu cầu về sản phẩm/size hết hàng được phản hồi trung thực và có đề xuất thay thế phù hợp khi khả dụng.
- Giá, tồn kho, khuyến mãi và voucher được dựa trên dữ liệu Main BE.
- Tư vấn size có căn cứ và không trình bày review fit như bảo đảm tuyệt đối.
- FAQ chính sách có trích dẫn; trường hợp không có căn cứ được xử lý trung thực.
- Yêu cầu ngoài phạm vi được từ chối theo quy tắc.
- CLI có chế độ chat, debug và eval; telemetry đủ để phân tích các lượt xử lý.
- Golden Set evaluation chạy được lặp lại và báo cáo được các chỉ số đã thống nhất.

## 11. Kế hoạch đánh giá nghiên cứu liên quan

Các thực nghiệm sau hỗ trợ mục tiêu học tập/nghiên cứu. Đây không phải các tính năng nghiệp vụ riêng của khách hàng, nhưng cần được hỗ trợ bởi CLI, telemetry và evaluation framework.

- **E1 — Clarification strategy:** So sánh `IMMEDIATE_ANSWER`, `MINIMAL_CLARIFICATION` và `ALWAYS_CLARIFY` về số turn, token, latency và tỷ lệ hoàn thành nhu cầu.
- **E2 — Kiến trúc Agent:** So sánh ReAct một vòng lặp với router phân loại intent và subgraphs chuyên biệt.
- **E3 — LLM local:** So sánh độ chính xác tool calling và chất lượng tiếng Việt giữa các model được cấu hình.
- **E4 — RAG policy:** So sánh kích thước chunk và giá trị Top-k.
- **E5 — Persona/prompting:** So sánh phong cách Fact-first với câu trả lời dài hơn về token và độ trễ, đồng thời kiểm tra chất lượng thông tin.

Golden Set dự kiến có quy mô 40–120 câu, bao gồm truy vấn clarification, khuyến mãi/voucher, sản phẩm hết hàng, tư vấn size/review fit và yêu cầu ngoài phạm vi.

## 12. Rủi ro và hướng xử lý

| Rủi ro | Tác động nghiệp vụ | Hướng xử lý |
|---|---|---|
| API Main BE thiếu bộ lọc hoặc dữ liệu | Kết quả tìm kiếm thiếu hoặc không chính xác | Chốt data contract và bổ sung endpoint tối thiểu |
| Dữ liệu tồn kho/giá thay đổi sau khi truy xuất | Tư vấn thông tin đã cũ | Xác minh thông tin quan trọng bằng API có thẩm quyền gần thời điểm trả lời |
| Dữ liệu size chart không đủ chính xác | Gợi ý size gây hiểu nhầm | Ghi rõ mức độ chắc chắn, cho phép tham khảo bảng size và review |
| Review fit ít hoặc thiên lệch | Khuyến nghị không đại diện | Chỉ dùng khi đạt ngưỡng tối thiểu và diễn đạt là tín hiệu tham khảo |
| LLM gọi sai tool hoặc diễn giải sai dữ liệu | Phản hồi sai | Kiểm tra điều kiện bằng logic xác định, đánh giá tool call và groundedness |
| RAG không tìm được chính sách đúng | Trả lời sai quy định shop | Trích dẫn nguồn và từ chối khi không có căn cứ |
| Model/API chậm hoặc lỗi | Trải nghiệm kém hoặc câu trả lời thiếu | Có timeout, bắt lỗi và phản hồi lỗi minh bạch |
| Telemetry chứa dữ liệu không cần thiết | Rủi ro riêng tư | Giới hạn dữ liệu log, kiểm soát quyền truy cập và chính sách lưu trữ |

## 13. Các điểm cần xác minh trong quá trình triển khai

Các điểm sau cần được xác minh với hệ thống thật; chúng không làm thay đổi mục tiêu nghiệp vụ đã thống nhất:

- API Main BE hiện trả giá sau giảm, tồn kho theo biến thể, voucher, campaign, discount và review ở định dạng nào.
- Phạm vi và semantics chính xác của `inStock=true` trong endpoint hybrid search.
- Quy tắc tính size theo từng danh mục/đối tượng và mức độ đầy đủ của dữ liệu `targetGender`.
- Cách tính `reviewCount` và cách tổng hợp các nhãn fit.
- Cơ chế và thời điểm cập nhật chỉ mục pgvector khi sản phẩm thay đổi.
- Cơ chế stream SSE và response JSON, bao gồm metadata và lỗi.
- Cách cấu hình clarification mode theo hệ thống và theo request, cùng quy tắc ưu tiên khi hai nguồn khác nhau.
- Định nghĩa chính xác của “hoàn thành nhu cầu”, “độ hài lòng dữ kiện” và ground truth trong thực nghiệm E1.

## 14. Liên kết với tài liệu liên quan

- **PRD:** Chatbot tư vấn khách hàng shop thời trang (Customer Shop Assistant), v1.0.
- **Macro Plan:** `customer-shop-assistant-macro-plan.md`.
- **API Contract:** sẽ được đặc tả riêng giữa `customer-shop-assistant` và `fashion-shop-be`.
- **Evaluation Plan:** sẽ mô tả chi tiết Golden Set, ground truth, cách chấm điểm và quy trình chạy các thực nghiệm E1–E5.

---

**Kết luận:** Customer Shop Assistant v1 tập trung vào tư vấn và truy xuất thông tin thời trang một cách có căn cứ; kiểm soát rõ hành vi hỏi lại; duy trì dữ liệu nghiệp vụ trong Main BE; và cung cấp công cụ đánh giá lặp lại để nghiên cứu chất lượng Agent/RAG.
