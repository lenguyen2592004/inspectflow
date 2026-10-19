# Ghi thí nghiệm

Mỗi lần huấn luyện ghi một dòng JSON vào `experiments/runs.jsonl`.

Bắt buộc có:

- `run_id`
- `task`
- `data_version`: hash phần dữ liệu đã dùng
- `model_name`, `model_version`
- `params`: seed, epoch, learning rate, kích thước ảnh
- `metrics`: accuracy, precision_macro, recall_macro, f1_macro
- `notes`: dữ liệu synthetic hay dữ liệu line

Suy luận từng frame ghi ở `experiments/inspections.jsonl` cùng `model_versions` để đối chiếu phiếu ERP.

Khi đổi bộ nhãn, tăng `data_version` bằng cách chạy lại script. Không ghi đè file weights cũ nếu version trong `configs/pipeline.yaml` chưa tăng.

## Nhãn

- Detection: một file YOLO cho mỗi ảnh, class `part`.
- Classification: thư mục `ok/` và `scratch/`, crop đúng vật thể.
- OCR: ảnh nhãn và chuỗi work order.
- Face: ảnh người đã đồng ý, lưu embedding sau enroll, không commit ảnh.

## Chỉ số dùng khi báo cáo

- Đếm: accuracy số lượng trên video mẫu, sai số tuyệt đối trung bình.
- Detection: mAP khi có nhãn box.
- Classification: macro F1 và ma trận nhầm.
- OCR: exact match trên mã work order.
- Face: TAR tại FAR đã chốt trong UAT.
- Hiệu năng: `latency_ms` trên đúng máy sẽ deploy.
