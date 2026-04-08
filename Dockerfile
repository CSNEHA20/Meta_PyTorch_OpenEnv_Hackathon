FROM python:3.11-slim

WORKDIR /app

# Install system dependencies in a separate layer for Docker cache efficiency
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before application code for cache efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code last
COPY . .

# Build Next.js dashboard — next.config.js sets output:'export' + distDir:'dist'
# so `npm run build` emits static files directly to frontend/dist,
# which is the path mounted by server/app.py StaticFiles("/dashboard").
RUN if [ -d "frontend" ]; then \
    cd frontend && \
    npm install && \
    npm run build; \
    fi

ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
# HF_TOKEN has no default — must be provided at runtime
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE="true"

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "4"]
