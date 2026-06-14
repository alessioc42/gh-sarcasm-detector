# gh-sarcasm-detector

Batch-evaluate LLM sarcasm-detection confidence on text and audio clips (English and German) via Ollama. State is stored in SQLite so runs survive crashes and restarts.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Ollama](https://ollama.com/) running locally with models listed in `models.txt` pulled (`ollama pull llama3.2`, etc.)
- Clip archives in `./raw_data/` (not included in git)

## Quick start (Docker)

Build locally:

```bash
docker build -t gh-sarcasm-detector .
```

Or pull from GitHub Container Registry after CI publishes:

```bash
docker pull ghcr.io/alessioc42/gh-sarcasm-detector:latest
```

Create host directories:

```bash
mkdir -p raw_data data config
# copy your *.zip / *.tar.gz archives into raw_data/
# optionally override models.txt or system_prompt.txt in config/
```

### 1. Import data

One-time (or when archives change):

```bash
docker run --rm \
  -v "$PWD/raw_data:/data/raw_data" \
  -v "$PWD/data:/data/db" \
  -v "$PWD/config:/data/config" \
  -e OLLAMA_ENDPOINT=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  gh-sarcasm-detector import
```

### 2. Compress audio (recommended)

Import stores raw audio blobs (large). Compress normalizes to **16 kHz mono FLAC** — the same format multimodal encoders (Qwen2-Audio / Whisper) use internally before mel-spectrogram extraction. FLAC is lossless at that stage, so embedding quality is preserved while the DB shrinks dramatically (especially for high-bitrate WAV clips):

```bash
docker run --rm \
  -v "$PWD/data:/data/db" \
  gh-sarcasm-detector compress
```

### 3. Run evaluation jobs

Processes all pending jobs; safe to stop and restart:

```bash
docker run --rm \
  -v "$PWD/raw_data:/data/raw_data" \
  -v "$PWD/data:/data/db" \
  -v "$PWD/config:/data/config" \
  -e OLLAMA_ENDPOINT=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  gh-sarcasm-detector run
```

### 4. Check status

```bash
docker run --rm \
  -v "$PWD/data:/data/db" \
  gh-sarcasm-detector status
```

Inspect raw LLM outputs:

```bash
sqlite3 data/sarcasm.db "SELECT status, COUNT(*) FROM jobs GROUP BY status;"
sqlite3 data/sarcasm.db "SELECT raw_response_body FROM job_outputs LIMIT 1;"
```

## Configuration

| Variable | Default (Docker) | Description |
|----------|------------------|-------------|
| `OLLAMA_ENDPOINT` | `http://host.docker.internal:11434` | Ollama base URL |
| `OLLAMA_API_TOKEN` | (empty) | Bearer token if required |
| `SQLITE_DB` | `/data/db/sarcasm.db` | SQLite database path |
| `RAW_DATA_DIR` | `/data/raw_data` | Directory with clip archives |
| `SYSTEM_PROMPT_PATH` | `/app/system_prompt.txt` | System prompt file |
| `MODELS_PATH` | `/app/models.txt` | One Ollama model per line |

Mount `./config/` to override `models.txt` or `system_prompt.txt` without rebuilding the image.

Audio jobs are skipped automatically for models that do not report `audio` capability via Ollama's `/api/show` endpoint.

### Audio storage and LLM embeddings

Multimodal models (e.g. Qwen2-Audio) resample all input to **16 kHz mono** and compute mel-spectrograms — they never see raw MP3/WAV containers. The `compress` command pre-normalizes to that rate and stores **FLAC** (lossless compression of the PCM the encoder would receive). Lossy codecs like MP3 are avoided for storage because they can alter prosody and tone, which matter for sarcasm detection.

## Development (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
# requires ffmpeg on PATH

python -m sarcasm_detector import
python -m sarcasm_detector compress
python -m sarcasm_detector run
python -m sarcasm_detector status

pytest   # runs tests with ≥90% coverage requirement
```

## CI

On push to `main`, GitHub Actions builds the Docker image and publishes it to `ghcr.io/<owner>/gh-sarcasm-detector`.

## License

AGPL-3.0 — see [LICENSE](LICENSE).
