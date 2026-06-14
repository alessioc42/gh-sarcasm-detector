FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sarcasm_detector/ sarcasm_detector/
COPY system_prompt.txt models.txt ./

ENV PYTHONUNBUFFERED=1 \
    SQLITE_DB=/data/db/sarcasm.db \
    RAW_DATA_DIR=/data/raw_data \
    SYSTEM_PROMPT_PATH=/app/system_prompt.txt \
    MODELS_PATH=/app/models.txt \
    OLLAMA_ENDPOINT=http://host.docker.internal:11434

RUN mkdir -p /data/db /data/raw_data /data/config

ENTRYPOINT ["python", "-m", "sarcasm_detector"]
CMD ["run"]
