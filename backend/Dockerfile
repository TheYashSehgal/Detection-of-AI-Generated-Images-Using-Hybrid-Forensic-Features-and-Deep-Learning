FROM python:3.11-slim

# Set up a new user named "user" with user ID 1000
# Hugging Face Spaces require a non-root user
RUN useradd -m -u 1000 user

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY . .

# Create cache and upload directories, and give ownership to the user
RUN mkdir -p /app/.cache/huggingface /app/checkpoints /app/uploads /app/samples && \
    chown -R user:user /app

# Switch to the non-root user
USER user

# Set environment variables for the container
ENV HF_HOME=/app/.cache/huggingface
ENV DEVICE=cpu
ENV CORS_ORIGINS="*"

# Hugging Face Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
