# Jamph ML Trainer

Model quantization and fine-tuning toolkit with MLflow tracking and Ollama.com deployment.

## Features

- **One-command workflow**: Download → Quantize → Upload
- **MLflow Integration**: Automatic experiment tracking and model registry
- **Jamph Namespace**: All models prefixed with `jamph-` to avoid conflicts
- **Cross-platform**: Docker + UV package manager
- **Multiple Quantization Methods**: Q4_K_M (default), Q5_K_M, Q8_0
- **Ollama.com Distribution**: Direct upload to your Ollama.com namespace

## Quick Start

### Prerequisites

1. **MLflow credentials** - Copy `example.mlflow.env` to `mlflow.env` and add your credentials
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
- `.gitignore` excludes `mlflow.env` (real credentials)
- `.dockerignore` prevents env files from being copied into image
- `example.mlflow.env` is tracked (template only - no real credentials)
- Docker uses `env_file` at runtime (credentials not baked into image layers)

**Your responsibilities:**
1. **NEVER commit `mlflow.env`** with real credentials
2. **Store credentials securely** (password managers, Azure Key Vault, etc.)
3. **Rotate tokens regularly** (Ollama, MLflow, HuggingFace)
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
  MLFLOW_TRACKING_PASSWORD: <base64-encoded-pwd>
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

# Custom quantization method
uv run jamph-ml-trainer process --model-id qwen/Qwen2.5-Coder-1.5B --method Q5_K_M

# Skip upload (quantize only)
uv run jamph-ml-trainer process --model-id qwen/Qwen2.5-Coder-1.5B --skip-upload

# Skip download if model exists
uv run jamph-ml-trainer process --model-id qwen/Qwen2.5-Coder-1.5B --skip-download

# Disable MLflow tracking
uv run jamph-ml-trainer process --model-id qwen/Qwen2.5-Coder-1.5B --no-mlflow
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
- **MLflow model name**: `jamph-qwen2.5-coder-1.5b-q4_k_m.gguf` (matches filename exactly)
- **MLflow run name**: `jamph-qwen2.5-coder-1.5b-q4_k_m.gguf-<git_username>-<date>`
- **MLflow experiment**: `jamph-quantization` (prefix-quantization)
- **Ollama.com**: `pererikgronvik/jamph-qwen2.5-coder-1.5b-q4_k_m` (your username namespace)

**Multiple quantizations coexist without collisions:**
- `jamph-qwen2.5-coder-1.5b-q4_k_m.gguf` (4-bit, smaller, faster)
- `jamph-qwen2.5-coder-1.5b-q5_k_m.gguf` (5-bit, better quality)
- `jamph-qwen2.5-coder-1.5b-q8_0.gguf` (8-bit, highest quality)

**Team Customization:**
Other teams can use their own namespace by setting `MODEL_PREFIX=myteam` in `mlflow.env`:
- Download: `qwen2.5-coder-1.5b/` (same)
- Quantized: `myteam-qwen2.5-coder-1.5b-q4_k_m.gguf`
- Experiment: `myteam-quantization`

## MLflow Tracking

### Model Registry Only

MLflow is used **only for Model Registry** - not for experiment tracking. Quantization is a build step, not an experiment.

**What gets registered:**
- **Model Artifact**: The GGUF file itself
- **Model Name**: Filename (e.g., `jamph-qwen2.5-coder-1.5b-q4_k_m.gguf`)
- **Tags**: 
  - `created_at`: ISO timestamp
  - `filename`: GGUF filename
  - `source_model`: Original HuggingFace model
  - `quantization_method`: Q4_K_M, Q5_K_M, or Q8_0
  - `creator`: Git username
  - `team`: Team name

**What MLflow is for:**
- Model evaluation experiments
- Performance comparison
- Testing different models
- A/B testing results

**Not for:**
- Quantization metrics (stored in MODEL_LOG.md instead)
- Build/compilation tracking

### Model Documentation

Each quantized model includes comprehensive documentation:

**MODEL_LOG.md** - Human-readable documentation with:
- Original model name and size
- Quantization method and parameters
- Creator (git username) and team
- Creation timestamp
- Processing time and compression ratio
- Usage examples

**quantization_metadata.json** - Machine-readable metadata for automation

These files are automatically:
- Saved in the model directory
- Logged to MLflow as artifacts
- Uploaded to HuggingFace/Ollama.com

**Why filename = model name?**
This ensures predictability when using models in other software. The MLflow model name matches the actual GGUF file exactly, so experiments can reliably reference models by their filesystem name.

### Viewing Results

Visit https://mlflow.vishvadukan.no to see:
- **Model Registry**: All quantized models with versions and metadata
- Use MLflow for model evaluation experiments (performance testing, comparison)

## Environment Variables

Required in `mlflow.env`:

```bash
# MLflow server
MLFLOW_TRACKING_URI=https://mlflow.vishvadukan.no
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password

# Ollama.com (for model distribution)
OLLAMA_USERNAME=pererikgronvik
OLLAMA_TOKEN=your_ollama_token

# Model naming prefix
MODEL_PREFIX=jamph

# Optional: Developer metadata
GITHUB_HANDLE=YourName
TEAM=YourTeam
ROLE=ML Engineer
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
│   ├── mlflow_tracking.py      # MLflow integration
│   ├── ollama_upload.py        # HuggingFace upload
│   └── utils.py                # Shared utilities
├── model training/
│   └── Models/                 # Downloaded and quantized models
├── training data for finetuning/
│   └── v1/                     # JSONL training data
├── logs/                       # Crash reports
├── pyproject.toml              # UV package configuration
├── Dockerfile.quantize         # Multi-stage build
├── docker-compose.quantize.yml # Docker orchestration
└── mlflow.env                  # Credentials (gitignored)
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

### MLflow connection fails

Check `mlflow.env` credentials and verify server access:

```bash
curl -u username:password https://mlflow.vishvadukan.no/api/2.0/mlflow/experiments/list
```

### HuggingFace upload fails

Verify token permissions:

```bash
huggingface-cli whoami
```

Token needs `write` permissions.

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
