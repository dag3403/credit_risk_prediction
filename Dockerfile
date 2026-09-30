FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CREDIT_RISK_MODEL_BUNDLE=/app/artifacts/credit_risk_model_bundle.joblib

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1

COPY deployment/requirements.txt /app/deployment/requirements.txt
RUN pip install --no-cache-dir -r /app/deployment/requirements.txt

COPY app /app/app
COPY artifacts/credit_risk_model_bundle.joblib /app/artifacts/credit_risk_model_bundle.joblib

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
