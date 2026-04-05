FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build Next.js dashboard if frontend exists
RUN if [ -d "frontend" ]; then \
    cd frontend && \
    npm install && \
    npm run build && \
    mkdir -p /app/static && \
    cp -r .next/static /app/static/_next/static && \
    cp -r public/* /app/static/ 2>/dev/null || true; \
    fi

ENV API_BASE_URL="https://api-inference.huggingface.co/v1"
ENV MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE="true"

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
