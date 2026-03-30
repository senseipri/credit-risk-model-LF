FROM python:3.10-slim

WORKDIR /app

# Copy backend code
COPY backend/ /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# IMPORTANT: main.py is directly in /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]