FROM python:3.10-slim 

WORKDIR /app 

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/* 

COPY . .

RUN pip install --upgrade pip 
RUN pip install --no-cache-dir -r requirements_docker.txt 

# ---- Set environment variables (optional) ----
RUN mkdir -p /app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

ENV TRANSFORMERS_NO_SAFE_TENSORS_CONVERSION=1

# ---- Expose the port FastAPI will run on ----
EXPOSE 8000

# ---- Run the FastAPI app ----
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]