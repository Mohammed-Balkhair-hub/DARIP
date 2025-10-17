FROM python:3.11-slim

# ffmpeg for audio stitching + probes
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

ENV PYTHONUNBUFFERED=1

# Default: run-once mode (overridden by compose command if needed)
CMD ["bash", "entrypoint.sh"]