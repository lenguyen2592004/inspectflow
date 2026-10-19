# InspectFlow

Cổng kiểm tra chất lượng bằng camera tại một trạm dây chuyền. Hệ thống đếm sản phẩm, phân loại lỗi xước, đọc mã lệnh sản xuất, xác thực người vận hành đã đăng ký, rồi đẩy phiếu kiểm tra sang ERP.

Bản này là đường chạy tái lập được: dữ liệu synthetic, baseline OpenCV, CNN PyTorch nhỏ, API, Docker, nhật ký thí nghiệm, SOP và kịch bản UAT. Model lớn hơn gắn vào cùng schema khi có ảnh line thật.

## Khớp với công việc

| Việc cần làm | Chỗ trong project |
| --- | --- |
| Xử lý ảnh và video | `StationPipeline.inspect_frame`, `inspect_frames` lấy trung vị số lượng |
| Face recognition | `src/inspectflow/cv/face.py`, enroll embedding, API `/operators` |
| Object detection và đếm | `src/inspectflow/cv/detection.py` |
| Image classification | `DefectCNN` và script huấn luyện |
| Chuẩn bị dữ liệu, tăng cường | `data/synthetic.py`, `data/augment.py`, quy ước nhãn trong `docs/experiment-protocol.md` |
| Huấn luyện, đánh giá, tối ưu | `scripts/train_defect_classifier.py`, precision/recall/F1, latency |
| Tích hợp ứng dụng, server, edge | FastAPI, `Dockerfile`, `scripts/export_onnx.py` |
| OCR | `src/inspectflow/cv/ocr.py` cho font cố định của máy in line |
| LLM / RAG / chatbot | `src/inspectflow/rag/assistant.py` trên SOP trong `docs/corpus` |
| Git, REST, Docker | repo này, `/api/v1`, `docker-compose.yml` |
| Ghi thí nghiệm để làm lại | `experiments/runs.jsonl`, `experiments/inspections.jsonl` |
| ERP, UAT, đào tạo, bàn giao | `docs/uat-and-delivery.md`, phiếu `ErpInspection` |

## Luồng nghiệp vụ

1. ERP có work order với số lượng cần qua trạm `LINE-A-QC-01`.
2. Camera chụp pallet hoặc đoạn băng chuyền.
3. Pipeline đếm, tìm xước, đọc mã `WO-...`, nhận người vận hành nếu trạm bật bước này.
4. Quyết định:
   - `reject` khi có lỗi xước
   - `review` khi lệch số lượng, lệch mã lệnh, hoặc người chưa xác thực
   - `accept` khi các tín hiệu đã cấp đều khớp
5. API ghi phiếu và trả document cho phân hệ chất lượng.
6. Ca planner sửa quyết định được ghi lại để gắn nhãn vòng sau.

Chatbot chỉ trả lời quy trình từ SOP và kèm file nguồn. Quyết định lô không đi qua chatbot.

## Chạy

Trên máy này `python` chưa có trên PATH. Dùng `py -3`.

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
python scripts/train_defect_classifier.py
python scripts/evaluate.py
python scripts/export_onnx.py
uvicorn inspectflow.api.main:app --app-dir src --reload
```

Docker:

```powershell
docker compose up --build
```

Sau khi API chạy, `GET /health` và `POST /api/v1/chat` với câu trong SOP. Ảnh đưa vào `POST /api/v1/inspections/frame`.

## Kết quả held-out

Tập đo không trùng seed huấn luyện (train `100–179`, đo từ `1000`). Nhiễu camera gồm blur 3×3, hệ số sáng 0.85–1.15, lệch ±12 và hạt σ = 3. Số liệu nằm ở `results/benchmark.json`. Chạy lại bằng `python scripts/evaluate.py`.

| Chỉ số | Ảnh sạch (80 frame) | Có nhiễu camera (80 frame) |
| --- | --- | --- |
| Detection F1 @ IoU 0.5 | 1.00 | 1.00 |
| Đếm đúng số lượng | 1.00 | 1.00 |
| OCR đúng nguyên mã lệnh | 1.00 | 0.638 |
| Macro F1 lỗi xước | 1.00 | 0.888 |
| Đúng quyết định accept / reject / review | 1.00 | 0.75 |
| Latency trung vị | 10.6 ms | 9.9 ms |

Đếm trên video (trung vị 5 frame, 30 đoạn, một frame bị cụt): 30/30 đúng.

Xác thực người vận hành, 12 danh tính đã enroll, 132 cặp impostor, ngưỡng cosine 0.90: TAR 1.00 tại FAR 0.00 (genuine thấp nhất 0.993, impostor cao nhất 0.831).

Các con số này là trên ảnh synthetic cùng renderer với pipeline, chưa phải footage nhà máy. OCR là khối tụt khi có nhiễu, và quyết định lô tụt theo OCR.

## Nhận diện khuôn mặt

Dùng cho người vận hành đã đồng ý, tại trạm. Lưu vector đặc trưng, không lưu ảnh. Baseline là HOG và cosine similarity để đo ngưỡng trên tập enroll nhỏ. Bước nghiên cứu tiếp theo là ArcFace hoặc FaceNet, giữ nguyên API enroll/verify.

## Việc tiếp khi có ảnh line

OCR đang kéo quyết định lô xuống 0.75 khi có nhiễu. Ảnh máy in thật là phép đo kế tiếp. Detector contour giữ F1 1.00 trên hình chữ nhật synthetic; đổi sang YOLO khi box thật không còn là hình học đơn giản.

Chi tiết kiến trúc, nhật ký và UAT nằm trong `docs/`. Mục LaTeX để dán vào CV nằm ở `docs/cv.tex`.
