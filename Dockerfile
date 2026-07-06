# thin churn scorer API. expects the prod model exported to api/model/ first:
#   python -m src.export_model   &&   docker build -t churn-api .   &&   docker run -p 8000:8000 churn-api
FROM python:3.12-slim

WORKDIR /app

# install deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code + the exported model (no MLflow registry / sqlite needed at serve time)
COPY src/ src/
COPY api/ api/

ENV MODEL_PATH=/app/api/model
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
