FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

RUN pip install --no-cache-dir \
    "fastapi>=0.115,<1.0" \
    "uvicorn>=0.34,<1.0" \
    "tensorflow==2.21.0"

COPY src ./src
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]