FROM python:3.11-slim

# Install system dependencies (none needed for this microservice)

# Set working directory
WORKDIR /app

# Copy only requirements first for better cache utilisation
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY dispatcher.py ./dispatcher.py

# Expose port 8000 for local testing (Render will override via $PORT)
EXPOSE 8000

# Start the FastAPI app with uvicorn. Render sets the PORT environment
# variable dynamically, so we use it with a default fallback to 8000.
CMD ["bash", "-c", "exec uvicorn dispatcher:app --host 0.0.0.0 --port ${PORT:-8000}"]