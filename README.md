# InspectFlow

Camera quality gate for one production-line station. It counts parts, classifies scratches, reads the work-order code, checks an enrolled operator, and posts a quality-inspection document to an ERP-style API.

The runnable path is reproducible: synthetic frames, an OpenCV detector, a small PyTorch CNN, a REST API, Docker, experiment logs, SOPs, and a UAT checklist. Swap in a larger model later without changing the inspection schema.

## Layout

| Piece | Where |
| --- | --- |
| Image and video | `StationPipeline.inspect_frame`; `inspect_frames` takes the median count |
| Face verification | `src/inspectflow/cv/face.py`, enrollment embeddings, `/api/v1/operators` |
| Object detection and counting | `src/inspectflow/cv/detection.py` |
| Image classification | `DefectCNN` and `scripts/train_defect_classifier.py` |
| Data, labels, augmentation | `data/synthetic.py`, `data/augment.py`, `docs/experiment-protocol.md` |
| Train, evaluate, export | precision / recall / F1, latency, `scripts/evaluate.py`, `scripts/export_onnx.py` |
| Server and edge | FastAPI, `Dockerfile`, ONNX export |
| OCR | `src/inspectflow/cv/ocr.py` for a fixed line-printer font |
| SOP chatbot | `src/inspectflow/rag/assistant.py` over `docs/corpus` |
| Experiment log | `experiments/runs.jsonl`, `experiments/inspections.jsonl` |
| ERP handoff and UAT | `ErpInspection`, `docs/uat-and-delivery.md` |

## Station flow

1. ERP sends a work order and the quantity due at station `LINE-A-QC-01`.
2. The camera captures a pallet or a short belt clip.
3. The pipeline counts parts, looks for scratches, reads a `WO-...` code, and can require an enrolled operator.
4. Decision:
   - `reject` when any part is scratched
   - `review` when the count, the work-order code, or the operator check disagrees
   - `accept` when every supplied signal matches
5. The API stores the inspection and returns the ERP document.
6. A planner override is the labeling source for the next training round.

The chatbot answers only from station SOPs and returns a source citation. It does not decide the lot.

## Run

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

On macOS or Linux, create the environment with `python3 -m venv .venv` and activate `.venv/bin/activate`.

Docker:

```powershell
docker compose up --build
```

After the API is up, `GET /health` and `POST /api/v1/chat` with a question from the SOP. Send a frame to `POST /api/v1/inspections/frame`.

## Held-out results

Evaluation seeds start at `1000`. Training used seeds `100–179`. Camera noise is a 3×3 blur, gain 0.85–1.15, bias ±12, and grain σ = 3. Figures are in `results/benchmark.json`. Regenerate them with `python scripts/evaluate.py`.

| Metric | Clean (80 frames) | Camera noise (80 frames) |
| --- | --- | --- |
| Detection F1 @ IoU 0.5 | 1.00 | 1.00 |
| Exact count | 1.00 | 1.00 |
| Work-order OCR exact match | 1.00 | 0.638 |
| Scratch macro-F1 | 1.00 | 0.888 |
| Lot decision accuracy | 1.00 | 0.75 |
| Median latency | 10.6 ms | 9.9 ms |

Video counting uses the median of five frames. On 30 clips with one partial frame, 30/30 counts were exact.

Operator verification on 12 enrolled identities and 132 impostor pairs, cosine threshold 0.90: TAR 1.00 at FAR 0.00 (lowest genuine score 0.993, highest impostor score 0.831).

These figures are on synthetic frames from the same renderer as the pipeline, not on factory footage. Under noise, OCR is the module that drops, and the lot decision drops with it.

## Operator verification

Enrollment is for people who agreed to be checked at the station. The store keeps an embedding, not the photo. The baseline is an aligned-crop descriptor with cosine similarity. ArcFace or FaceNet can replace the embedder without changing the enroll and verify API.

## Next measurement

Real printer images are the next OCR test. The contour detector scores 1.00 F1 on synthetic rectangles; replace it with YOLO when real boxes are no longer simple geometry.

Architecture, the experiment protocol, and UAT cases are in `docs/`.
