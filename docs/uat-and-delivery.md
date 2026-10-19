# UAT và bàn giao

Kịch bản người vận hành và planner cùng chạy. Kết quả ghi pass/fail, không chỉ demo.

| ID | Tình huống | Kỳ vọng |
| --- | --- | --- |
| UAT-01 | 3 sản phẩm tốt, đúng mã WO, đúng người | accept, counted = 3 |
| UAT-02 | Có một sản phẩm xước | reject, defect scratch |
| UAT-03 | Đếm lệch số lượng lệnh | review, không nhập kho |
| UAT-04 | OCR ra mã khác work order | review |
| UAT-05 | Ảnh người chưa enroll | verify = false |
| UAT-06 | Hỏi chatbot trường hợp reject | câu trả lời có trích SOP |

## Đào tạo tại chỗ

1. Cách đặt hàng trong vùng camera.
2. Nghĩa của accept, reject, review.
3. Ai được enroll, ai được xem log.
4. Phiếu ERP nhận field nào.

## Feedback

Mỗi lần planner sửa quyết định của máy, ghi `inspection id`, quyết định người, lý do. Đưa các ca đó vào bộ nhãn vòng sau. Đó là nguồn dữ liệu thật, ưu tiên hơn ảnh synthetic.
