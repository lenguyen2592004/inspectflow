import cv2
from fastapi.testclient import TestClient

from inspectflow.api.main import app
from inspectflow.data.synthetic import make_frame


def test_health_and_sop_chat() -> None:
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    chat = client.post("/api/v1/chat", json={"question": "Khi nao lo bi reject vi xuoc?"})
    assert chat.status_code == 200
    assert chat.json()["citations"]


def test_frame_inspection_posts_erp_document() -> None:
    frame = make_frame(ok_count=2, scratch_count=1, seed=9)
    ok, encoded = cv2.imencode(".png", frame.image)
    assert ok
    client = TestClient(app)
    response = client.post(
        "/api/v1/inspections/frame",
        files={"image": ("frame.png", encoded.tobytes(), "image/png")},
        data={"expected_quantity": "3", "expected_work_order_id": "WO-1001"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["erp_document"]["decision"] == "reject"
    assert body["inspection"]["counted_quantity"] == 3
