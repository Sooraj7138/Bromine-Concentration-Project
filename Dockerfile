# Use a lightweight Python base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire application code, including models and data
COPY app/ app/

# Expose Flask port
EXPOSE 5000

# Start the Flask app
CMD ["python", "app/main.py"]