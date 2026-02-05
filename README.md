# Jamph ML Trainer

Model quantization toolkit with Ollama.com deployment and RAG-friendly metadata.

## Features

- **One-command workflow**: Download → Quantize → Upload → Cleanup
- **Multi-Method Quantization**: Process one model with multiple quantization methods efficiently
- **Automatic Cleanup**: Deletes source and quantized files after successful upload
- **RAG-Friendly Metadata**: Structured JSON for easy API/RAG integration
- **Jamph Namespace**: All models prefixed with `jamph-` (configurable via MODEL_PREFIX)
- **Cross-platform**: Docker + UV package manager
- **CPU/GPU Auto-detect**: Works on CPU-only systems, uses GPU if available
- **Multiple Quantization Methods**: Q4_K_M (default), Q5_K_M, Q8_0
- **Ollama.com Distribution**: Direct upload to your Ollama.com namespace

## Quick Start

### Prerequisites

1. **Ollama.com credentials** - Copy `example.ollama.env` to `ollama.env` and add your credentials
2. **Ollama.com account** - Get token from https://ollama.com/settings/keys
3. **Ollama CLI installed** - Download from https://ollama.com/download

### Docker Usage (Recommended)

```bash
# 1. Build the quantization container
docker-compose -f docker-compose.quantize.yml build

# 2. Run complete pipeline (download + quantize + upload)
docker-compose -f docker-compose.quantize.yml run quantizer process --model-id qwen/Qwen2.5-Coder-1.5B

# 3. Check output in model training/Models/
```

### UV/Python Usage (Local Development)

```bash
# 1. Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Run CLI
uv run jamph-ml-trainer process --model-id qwen/Qwen2.5-Coder-1.5B
```

## Security Best Practices

### ✅ Credential Protection

**Current safeguards:**
- `.gitignore` excludes `ollama.env` (real credentials)
- `.dockerignore` prevents env files from being copied into image
- `example.ollama.env` is tracked (template only - no real credentials)
- Docker uses `env_file` at runtime (credentials not baked into image layers)

**Your responsibilities:**
1. **NEVER commit `ollama.env`** with real credentials
2. **Store credentials securely** (password managers, Azure Key Vault, etc.)
3. **Rotate tokens regularly** (Ollama, HuggingFace)
4. **Use `.env.local`** for personal overrides (also gitignored)

### 🔒 Production Deployment

For production environments, use proper secret management:

**Kubernetes:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: jamph-ml-secrets
type: Opaque
stringData:
  OLLAMA_TOKEN: <base64-encoded-token>
```

**Docker Swarm:**
```bash
echo "your-token" | docker secret create ollama_token -
docker service create --secret ollama_token jamph-quantizer
```

**Azure/NAIS:**
Use Azure Key Vault or NAIS secret injection instead of env files.

### 🛡️ Container Security

- Running containers expose env vars to `docker inspect` - don't run on shared machines with untrusted users
- Use `--read-only` filesystem where possible
- Limit resource usage with `deploy.resources.limits` in docker-compose

## CLI Commands

### `process` - Complete Pipeline

Download, quantize, and upload in one command:

```bash
# Full pipeline with default Q4_K_M quantization
uv run jamph-ml-trainer process --model-id qwen/Qwen2.5-Coder-1.5B

# Multiple quantization methods (efficient: download once, quantize 3 times)
uv run jamph-ml-trainer process --model-id qwen/Qwen2.5-Coder-1.5B --methods Q4_K_M,Q5_K_M,Q8_0

# Skip upload (quantize only)
uv run jamph-ml-trainer process --model-id qwen/Qwen2.5-Coder-1.5B --skip-upload

# Skip download if model exists
uv run jamph-ml-trainer process --model-id qwen/Qwen2.5-Coder-1.5B --skip-download

# Keep files after upload (by default, files are deleted after successful upload)
uv run jamph-ml-trainer process --model-id qwen/Qwen2.5-Coder-1.5B --keep-files
```

### `download` - Download Only

```bash
uv run jamph-ml-trainer download qwen/Qwen2.5-Coder-1.5B
```

### `quantize` - Quantize Existing Model

```bash
uv run jamph-ml-trainer quantize ./model\ training/Models/jamph-qwen2.5-coder-1.5b --method Q4_K_M
```

### `upload` - Upload to Ollama.com

```bash
uv run jamph-ml-trainer upload \
  ./model\ training/Models/jamph-qwen2.5-coder-1.5b-q4_k_m/jamph-qwen2.5-coder-1.5b-q4_k_m.gguf \
  --source-model qwen/Qwen2.5-Coder-1.5B
```

## Quantization Methods

| Method | Description | Size Reduction | RAM Required | Speed |
|--------|-------------|----------------|--------------|-------|
| **Q4_K_M** | 4-bit, medium quality | ~75% | 6-8GB | Fast |
| Q5_K_M | 5-bit, higher quality | ~70% | 8-10GB | Moderate |
| Q8_0 | 8-bit, excellent quality | ~50% | 12-16GB | Slower |

## Model Naming Convention

A team prefix (default: `jamph-`, configurable via `MODEL_PREFIX` env var) is added during quantization to avoid namespace conflicts:

- **Input**: `qwen/Qwen2.5-Coder-1.5B`
- **Downloaded as**: `qwen2.5-coder-1.5b/` (directory, normalized, no prefix)
- **Quantized file**: `jamph-qwen2.5-coder-1.5b-q4_k_m.gguf` (prefix added)
- **Quantized dir**: `jamph-qwen2.5-coder-1.5b-q4_k_m/` (contains .gguf + metadata)
- **Ollama.com**: `pererikgronvik/jamph-qwen2.5-coder-1.5b-q4_k_m` (your username namespace)
- **RAG metadata**: `/models/.metadata/qwen2.5-coder-1.5b.json` (API-friendly format)

**Multiple quantizations coexist without collisions:**
- `jamph-qwen2.5-coder-1.5b-q4_k_m.gguf` (4-bit, smaller, faster)
- `jamph-qwen2.5-coder-1.5b-q5_k_m.gguf` (5-bit, better quality)
- `jamph-qwen2.5-coder-1.5b-q8_0.gguf` (8-bit, highest quality)

**Team Customization:**
Other teams can use their own namespace by setting `MODEL_PREFIX=myteam` in `ollama.env`:
- Download: `qwen2.5-coder-1.5b/` (same)
- Quantized: `myteam-qwen2.5-coder-1.5b-q4_k_m.gguf`
- Ollama: `username/myteam-qwen2.5-coder-1.5b-q4_k_m`

## File Cleanup

By default, the pipeline automatically cleans up files after successful upload:

1. **Source model files** deleted after all quantizations complete
2. **Quantized GGUF files** deleted after successful upload to Ollama.com
3. **RAG metadata** persisted in `/models/.metadata/` for API consumption

This saves disk space and ensures models only exist in Ollama.com (single source of truth).

To keep files locally, use `--keep-files` flag.

## RAG-Friendly Metadata

Each model gets a JSON metadata file optimized for RAG/API consumption:

**Location**: `/models/.metadata/{model-name}.json`

**Example** (`/models/.metadata/qwen2.5-coder-1.5b.json`):

```json
{
  "model": {
    "name": "qwen2.5-coder-1.5b",
    "full_name": "jamph-qwen2.5-coder-1.5b",
    "source": {
      "huggingface": "qwen/Qwen2.5-Coder-1.5B",
      "type": "transformer"
    }
  },
  "quantizations": [
    {
      "method": "Q4_K_M",
      "size_mb": 987.45,
      "uploaded_at": "2024-01-15T10:30:00",
      "ollama_url": "https://ollama.com/pererikgronvik/jamph-qwen2.5-coder-1.5b-q4_k_m",
      "ollama_command": "ollama run pererikgronvik/jamph-qwen2.5-coder-1.5b-q4_k_m"
    },
    {
      "method": "Q5_K_M",
      "size_mb": 1234.56,
      "uploaded_at": "2024-01-15T10:35:00",
      "ollama_url": "https://ollama.com/pererikgronvik/jamph-qwen2.5-coder-1.5b-q5_k_m",
      "ollama_command": "ollama run pererikgronvik/jamph-qwen2.5-coder-1.5b-q5_k_m"
    }
  ],
  "metadata": {
    "created_by": "pererik",
    "team": "ResearchOps",
    "created_at": "2024-01-15T10:30:00",
    "prefix": "jamph"
  },
  "usage": {
    "description": "Quantized versions of qwen/Qwen2.5-Coder-1.5B for efficient inference",
    "recommended": "Q4_K_M",
    "notes": "Q4_K_M for best speed/quality balance, Q5_K_M for better quality, Q8_0 for highest quality"
  }
}
```

**Benefits for RAG/API:**
- Structured, parseable format
- Direct Ollama URLs and commands
- Size information for filtering
- Timestamp for freshness checks
- Team/creator attribution

## Environment Variables

Required in `ollama.env`:

```bash
# Ollama.com (for model distribution)
OLLAMA_USERNAME=pererikgronvik
OLLAMA_TOKEN=oll_your_token_here

# HuggingFace (optional, for downloading private models)
HF_TOKEN=hf_your_token_here

# Model naming prefix
MODEL_PREFIX=jamph

# Optional: Developer metadata
GITHUB_HANDLE=YourName
TEAM=YourTeam
```

## Docker Details

### Multi-Stage Build

1. **llama-builder**: Builds llama.cpp quantization tools from source
2. **python-builder**: Creates UV Python environment with dependencies
3. **runtime**: Slim image with llama.cpp + Python environment

### Volume Mounts

- `/models` - Persistent model storage
- `/logs` - Crash reports and logs
- `/cache` - HuggingFace cache
- `/training-data` - Fine-tuning datasets (read-only)

### Resource Limits

Default docker-compose settings:
- **CPUs**: 4-8 cores
- **Memory**: 8-16GB
- Adjust in `docker-compose.quantize.yml` based on your system

## Project Structure

```
Jamph-ML-Trainer/
├── src/jamph_ml_trainer/
│   ├── __init__.py
│   ├── cli.py                  # Main CLI entry point
│   ├── ollama_upload.py        # Ollama.com upload
│   └── utils.py                # Shared utilities
├── model training/
│   ├── Models/                 # Downloaded and quantized models
│   └── .metadata/              # RAG-friendly JSON metadata
├── logs/                       # Crash reports
├── pyproject.toml              # UV package configuration
├── Dockerfile.quantize         # Multi-stage build
├── docker-compose.quantize.yml # Docker orchestration
└── ollama.env                  # Credentials (gitignored)
```

## Legacy Scripts

The following standalone scripts are replaced by the unified CLI:

- `download_model.py` → `jamph-ml-trainer download`
- `quantize_model.py` → `jamph-ml-trainer quantize`
- *(new)* → `jamph-ml-trainer upload`

Legacy scripts remain for reference but the CLI is the recommended approach.

## Troubleshooting

### llama.cpp not found

The Docker image builds llama.cpp automatically. For local development:

```bash
# Clone llama.cpp alongside this repo
cd ..
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build
```

### HuggingFace download fails

Verify token (if downloading private models):

```bash
huggingface-cli whoami
```

Token needs `read` permissions.

### Ollama upload fails

**Check Ollama CLI is installed:**

```bash
ollama --version
```

**Verify credentials:**

```bash
ollama login
# Paste your OLLAMA_TOKEN when prompted
```

**Test local model creation:**

```bash
cd "model training/Models/jamph-model-name"
ollama create test-model -f Modelfile
ollama run test-model  # Should work locally
```

## Development

### Running Tests

```bash
uv run pytest
```

### Linting

```bash
uv run ruff check src/
uv run ruff format src/
```

### Type Checking

```bash
uv run mypy src/
```

## License

Apache 2.0 - See LICENSE file

## Contributing

See CONTRIBUTING file for guidelines.
