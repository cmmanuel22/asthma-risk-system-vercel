# Use lightweight Python image
FROM python:3.10-slim

# Disable interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies needed for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy everything into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port used by Flask
EXPOSE 7860

# Run Flask app
CMD ["python", "api/index.py"]
