FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY configs configs
COPY docs/corpus docs/corpus
COPY src src
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "inspectflow.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
