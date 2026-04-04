FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV API_BASE_URL="http://localhost:8000"
ENV MODEL_NAME="ambulance-rl-model"
ENV PYTHONUNBUFFERED=1

# Expose port for HuggingFace Spaces
EXPOSE 7860

# Default command
CMD ["python", "inference.py"]
