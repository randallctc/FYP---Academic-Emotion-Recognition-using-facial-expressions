FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run.py .
COPY startup.sh .
COPY app/ app/
COPY templates/ templates/

RUN chmod +x startup.sh

# Pre-download ResNet50 weights during build so Keras doesn't fetch them
# at runtime (where eventlet breaks DNS resolution).
RUN python3 -c "from tensorflow.keras.applications import ResNet50; ResNet50(weights='imagenet', include_top=False); print('ResNet50 weights cached.')"

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 5000

CMD ["./startup.sh"]
