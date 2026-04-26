# Use Python 3.11 explicitly — no ambiguity
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install CPU-only torch first, then everything else
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Run the bot
CMD ["python", "bot.py"]
