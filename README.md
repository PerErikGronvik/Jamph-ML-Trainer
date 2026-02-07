# Jamph ML Trainer

Download HuggingFace models, quantize to GGUF format, and upload to HuggingFace Hub.

## Quantize a Model (5-Second Guide)

```bash
# 1. Setup credentials (one-time)
cp example.jamph.env jamph.env
# Edit jamph.env: Add your HF_USERNAME and HF_TOKEN from https://huggingface.co/settings/tokens

# 2. Run quantization
docker-compose build quantizer
docker-compose run quantizer process qwen/Qwen2.5-0.5B

# 3. Done! Model uploaded to: https://huggingface.co/{your-username}/jamph-qwen2.5-0.5b-q4_k_m
```

## Usage

### Basic Quantization
```bash
# Single method (Q4_K_M - default, 75% size reduction)
docker-compose run quantizer process qwen/Qwen2.5-0.5B

# Multiple methods (creates separate folders: jamph-model-q4_k_m/, jamph-model-q5_k_m/, jamph-model-q8_0/)
docker-compose run quantizer process qwen/Qwen2.5-0.5B --methods Q4_K_M,Q5_K_M,Q8_0

# Skip upload (local quantization only)
docker-compose run quantizer process qwen/Qwen2.5-0.5B --skip-upload

# Async upload (default, bypasses MerkleDB errors)
docker-compose run quantizer process qwen/Qwen2.5-0.5B --async-upload
```

### Quantization Methods
| Method | Size | Quality | Use Case |
|--------|------|---------|----------|
| **Q4_K_M** | 75% smaller | Good | Default, best balance |
| **Q5_K_M** | 70% smaller | Better | Higher quality needs |
| **Q8_0** | 50% smaller | Best | Maximum quality |

## Environment Variables

Edit `jamph.env` (copy from `example.jamph.env`):

```bash
# Required
HF_USERNAME=your-username      # HuggingFace username
HF_TOKEN=hf_xxxxx              # Get from https://huggingface.co/settings/tokens

# Optional
MODEL_PREFIX=jamph             # Prefix for model names (default: jamph)
ORGANIZATION=YourOrg           # Organization name in metadata
TEAM=YourTeam                  # Team name in metadata
GITHUB_HANDLE=YourName         # Your name in logs
```

## Output Structure

For `--methods Q4_K_M,Q5_K_M` on `qwen/Qwen2.5-0.5B`:

```
model training/Models/
├── jamph-qwen2.5-0.5b-q4_k_m/
│   ├── jamph-qwen2.5-0.5b-q4_k_m.gguf  # Quantized model
│   ├── MODEL_LOG.md                     # Quantization details
│   └── quantization_metadata.json       # API metadata
└── jamph-qwen2.5-0.5b-q5_k_m/
    ├── jamph-qwen2.5-0.5b-q5_k_m.gguf
    ├── MODEL_LOG.md
    └── quantization_metadata.json
```

Each folder uploaded to: `https://huggingface.co/{your-username}/{folder-name}`

## Local Development (No Docker)

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and run
uv sync
uv run jamph-ml-trainer process qwen/Qwen2.5-0.5B
```

## Troubleshooting

**Build fails**: Ensure Docker has 8GB+ RAM allocated (Docker Desktop → Settings → Resources)

**Upload fails**: Check `HF_TOKEN` has write permissions → https://huggingface.co/settings/tokens

**MerkleDB error**: Use `--async-upload` (default behavior)

---

**License**: Apache 2.0 | **Contributing**: See [CONTRIBUTING](CONTRIBUTING)
