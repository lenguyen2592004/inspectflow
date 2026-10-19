# Kiến trúc InspectFlow

Một trạm camera cuối line. Mỗi frame trả về một phiếu kiểm tra mà phân hệ chất lượng ERP có thể nhận.

```text
camera / video
    |  sample frame
    v
detect parts (OpenCV contour -> YOLO/RT-DETR)
    |  boxes + count
    v
classify crop (edge baseline -> DefectCNN -> MobileNet)
    |
read work order (template OCR -> PaddleOCR/CRNN)
    |
verify operator (HOG embedding -> ArcFace), chỉ với người đã enroll
    v
decision: accept | reject | review
    v
REST /api/v1/inspections/frame
    v
ERP quality inspection + inspection log
```

Chatbot SOP đứng cạnh, không nằm trên đường quyết định lô. Nó trả lời từ `docs/corpus` và luôn kèm trích dẫn.

## Vì sao baseline chạy được trước model lớn

Dữ liệu nhà máy thường chưa có ở tuần đầu. Contour, HOG và template OCR cho một đường chạy đo được độ trễ và hợp đồng API. Khi có nhãn, thay từng khối bằng model học sâu mà giữ nguyên schema box, nhãn lỗi và phiếu ERP.

## Edge và server

Server là FastAPI trong Docker. Edge là file ONNX của DefectCNN, cùng một kích thước crop 64. Đổi runtime sang ONNX Runtime trên máy trạm khi đo latency.

## Việc không làm trong bản này

Không nhận diện người ngoài danh sách enroll. Không lưu ảnh khuôn mặt. Không tự huấn luyện YOLO vì chưa có bộ nhãn thật.
