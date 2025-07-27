FROM python:3.11-slim
WORKDIR /app

# Copy requirements and install dependencies first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (dispatcher and frontend)
COPY . .

# Expose default port (Render overrides PORT env var)
EXPOSE 8000

# Ensure output is unbuffered
ENV PYTHONUNBUFFERED=1

# Launch the FastAPI app via uvicorn; use PORT env var if provided
CMD ["uvicorn", "dispatcher:app", "--host", "0.0.0.0", "--port", "8000"]
