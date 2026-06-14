# gh-sarcasm-detector

Batch-evaluate LLM sarcasm-detection confidence on text and audio clips (English and German) via Ollama. State is stored in SQLite so runs survive crashes and restarts.

Designed to run **bare-metal on a Mac mini** — models are pulled from Ollama automatically, evaluated, then deleted to save disk space.

## Prerequisites (Mac mini)

- macOS with [Homebrew](https://brew.sh/)
- [Ollama](https://ollama.com/) — `brew install ollama` and `brew services start ollama`
- ffmpeg — `brew install ffmpeg`
- Python 3.11+
- Clip archives in `./raw_data/` (not included in git)

## Quick start

```bash
git clone https://github.com/alessioc42/gh-sarcasm-detector.git
cd gh-sarcasm-detector

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

mkdir -p raw_data data
# copy your *.zip / *.tar.gz archives into raw_data/
# edit models.txt — one Ollama model per line
```

### 1. Import data

One-time (or when archives change):

```bash
python -m sarcasm_detector import
```

### 2. Compress audio (recommended)

Import stores raw audio blobs (large). Compress normalizes to **16 kHz mono FLAC** — the same format multimodal encoders (Qwen2-Audio / Whisper) use internally before mel-spectrogram extraction. FLAC is lossless at that stage, so embedding quality is preserved while the DB shrinks dramatically:

```bash
python -m sarcasm_detector compress
```

### 3. Run evaluation

For each model in `models.txt`, the runner will:

1. **Pull** the model from Ollama (or use a prefetched download)
2. **Run** all pending evaluation jobs for that model
3. **Delete** the model from Ollama to free disk/RAM

While model *N* is being evaluated, model *N+1* is downloaded in the background to overlap slow downloads with inference.

```bash
python -m sarcasm_detector run
```

Or use the convenience script:

```bash
chmod +x scripts/run_eval.sh
./scripts/run_eval.sh
```

### 4. Check status

```bash
python -m sarcasm_detector status
sqlite3 sarcasm.db "SELECT status, COUNT(*) FROM jobs GROUP BY status;"
sqlite3 sarcasm.db "SELECT raw_response_body FROM job_outputs LIMIT 1;"
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_ENDPOINT` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_API_TOKEN` | (empty) | Bearer token if required |
| `SQLITE_DB` | `sarcasm.db` | SQLite database path |
| `RAW_DATA_DIR` | `raw_data` | Directory with clip archives |
| `SYSTEM_PROMPT_PATH` | `system_prompt.txt` | System prompt file |
| `MODELS_PATH` | `models.txt` | One Ollama model per line |

Audio jobs are skipped automatically for models that do not report `audio` capability via Ollama's `/api/show` endpoint.

### Audio storage and LLM embeddings

Multimodal models (e.g. Qwen2-Audio) resample all input to **16 kHz mono** and compute mel-spectrograms — they never see raw MP3/WAV containers. The `compress` command pre-normalizes to that rate and stores **FLAC** (lossless compression of the PCM the encoder would receive).

## Development

```bash
pip install -r requirements-dev.txt
pytest   # runs tests with ≥90% coverage requirement
```

## CI

On push/PR to `main`, GitHub Actions runs unit tests (no Ollama required).

## License

AGPL-3.0 — see [LICENSE](LICENSE).
