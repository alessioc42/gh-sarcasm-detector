# gh-sarcasm-detector

Batch-evaluate LLM sarcasm-detection confidence on text and audio clips (English and German) via Ollama. State is stored in SQLite so runs survive crashes and restarts.

Designed to run **bare-metal on a Mac mini** — models are pulled from Ollama automatically, evaluated, and **kept on disk** when space allows. Models are deleted only when free space is insufficient for the next pull.

## Prerequisites (Mac mini)

- macOS with [Homebrew](https://brew.sh/)
- [Ollama](https://ollama.com/) — `brew install ollama` and `brew services start ollama`
- ffmpeg — `brew install ffmpeg`
- Python 3.11+
- Clip archives in `./raw_data/` (not included in git)
- System prompts as `.txt` files in `./prompts/` (see `prompts/default.txt`)

## Quick start

```bash
git clone https://github.com/alessioc42/gh-sarcasm-detector.git
cd gh-sarcasm-detector

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

mkdir -p raw_data data prompts
# copy your *.zip / *.tar.gz archives into raw_data/
# edit models.txt — one Ollama model per line
# add or edit prompt files in prompts/*.txt
```

### Fresh database setup

The schema is not migrated from older databases. To start clean:

```bash
rm -f sarcasm.db sarcasm.db-wal sarcasm.db-shm
python -m sarcasm_detector import
python -m sarcasm_detector compress   # recommended
python -m sarcasm_detector run
python -m sarcasm_detector parse
```

### 1. Import data

One-time (or when archives change):

```bash
python -m sarcasm_detector import
```

Import registers all models from `models.txt` and all prompts from `prompts/*.txt`, then creates evaluation jobs for every `(clip × model × prompt × modality × language)` combination.

### 2. Compress audio (recommended)

Import stores raw audio blobs (large). Compress normalizes to **16 kHz mono FLAC** — the same format multimodal encoders (Qwen2-Audio / Whisper) use internally before mel-spectrogram extraction. FLAC is lossless at that stage, so embedding quality is preserved while the DB shrinks dramatically:

```bash
python -m sarcasm_detector compress
```

### 3. Run evaluation

For each model in `models.txt`, the runner will:

1. **Reuse** the model if it is already installed in Ollama, otherwise **pull** it (with background prefetch of the next model)
2. **Run** all pending evaluation jobs for that model (each job uses its prompt from `prompts/`)
3. **Retain** the model on disk after evaluation; evict models only when disk space is needed for an upcoming pull

**Eviction order** when space is low:

1. Orphans — installed models not listed in `models.txt`
2. Completed evaluation models — in `models.txt`, finished this run, with no pending jobs (oldest first)

Never evicted: the model currently evaluating, models with pending jobs, models scheduled for prefetch, and models not yet pulled from the current queue.

Transient inference failures (e.g. Ollama HTTP 500) are **retried** up to `MAX_JOB_ATTEMPTS` (default 3). Model pull failures leave jobs pending for the next run.

While model *N* is being evaluated, model *N+1* is downloaded in the background to overlap slow downloads with inference.

```bash
python -m sarcasm_detector run
```

Or use the convenience script:

```bash
chmod +x scripts/run_eval.sh
./scripts/run_eval.sh
```

### Reload models or prompts without re-importing

After editing `models.txt` or adding files under `prompts/`, sync and create missing evaluation jobs without touching clip data:

```bash
python -m sarcasm_detector sync-models
```

This is safe to run at any time: existing clips, assets, and completed jobs are preserved. Re-running `import` is also safe (existing clips are skipped), but `sync-models` is faster when only the model or prompt list changed.

`status` applies the same sync automatically so `Jobs to run` reflects the current `models.txt` and `prompts/`.

### 4. Check status

```bash
python -m sarcasm_detector status
sqlite3 sarcasm.db "SELECT status, COUNT(*) FROM jobs GROUP BY status;"
sqlite3 sarcasm.db "SELECT failure_kind, COUNT(*) FROM jobs WHERE failure_kind IS NOT NULL GROUP BY failure_kind;"
```

### 5. Parse results

After evaluation completes, extract structured verdicts from **completed** jobs only:

```bash
python -m sarcasm_detector parse
python -m sarcasm_detector status
```

#### Job execution status (`jobs.status`)

| Status | Meaning |
|--------|---------|
| `pending` | Awaiting execution (includes retries) |
| `running` | Currently executing |
| `completed` | Inference succeeded; eligible for parsing |
| `failed` | Terminal failure (retries exhausted or non-retryable error) |
| `skipped` | Permanently not runnable (e.g. model lacks audio capability) |

Terminal non-completed jobs may also have `failure_kind`:

| `failure_kind` | Meaning |
|----------------|---------|
| `missing_asset` | Required clip asset missing before inference |
| `inference_error` | Ollama error after all retry attempts |
| `unsupported_modality` | Audio job skipped (model has no audio capability) |

#### Parse verdicts (`job_verdicts`, completed jobs only)

| Verdict | Meaning |
|---------|---------|
| `SARCASTIC` | Parsed JSON with `"sarcastic": true` |
| `NOT_SARCASTIC` | Parsed JSON with `"sarcastic": false` |
| `LLM_ERR` | Completed job but response had no valid schema JSON |

`confidence` is stored as an integer 0–10 when the LLM returned one; otherwise `NULL`.

```bash
sqlite3 sarcasm.db "SELECT verdict, COUNT(*) FROM job_verdicts GROUP BY verdict;"
```

### 6. Analysis notebook

Visualize parsed results with descriptive charts (no performance interpretation in the notebook text):

```bash
pip install -r requirements-notebook.txt
python -m sarcasm_detector parse   # if not already done
jupyter notebook notebooks/eval_analysis.ipynb
```

Set `SQLITE_DB` if your database is not `./sarcasm.db`.

For accuracy analysis, filter on `job_status == 'completed'` rather than verdict type.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_ENDPOINT` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_API_TOKEN` | (empty) | Bearer token if required |
| `SQLITE_DB` | `sarcasm.db` | SQLite database path |
| `RAW_DATA_DIR` | `raw_data` | Directory with clip archives |
| `PROMPTS_DIR` | `prompts` | Directory of `.txt` system prompt files |
| `MAX_JOB_ATTEMPTS` | `3` | Total execution attempts per job (including retries) |
| `MODELS_PATH` | `models.txt` | One Ollama model per line |
| `OLLAMA_MODELS_DIR` | `$OLLAMA_MODELS` or `~/.ollama/models` | Directory used for disk-space checks |
| `MIN_FREE_DISK_BYTES` | `2000000000` (2 GiB) | Minimum free space to keep after any pull |
| `MODEL_PULL_RESERVE_BYTES` | `8000000000` (8 GiB) | Estimated pull size when target model size is unknown |
| `LOG_LEVEL` | `INFO` | Runner log verbosity (`DEBUG`, `INFO`, `WARNING`, …) |

Each `.txt` file in `PROMPTS_DIR` creates a separate job matrix (same clip/model/modality/language evaluated with every prompt).

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
