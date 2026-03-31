# Jamph ML Trainer

Download HuggingFace models, quantize them to GGUF format using llama.cpp, and optionally upload to HuggingFace Hub.

**Local Docker-based quantization** - No cloud deployment required.

> **First time?** See [GETTING_STARTED.md](GETTING_STARTED.md) for a step-by-step checklist.

## Quick Start (Fresh Clone)

### Prerequisites
- **Docker Desktop** installed and running ([Download](https://www.docker.com/products/docker-desktop))
- **HuggingFace account** for model download/upload ([Sign up](https://huggingface.co/join))

### 1. Clone and Setup

```powershell
# Clone the repository
git clone <your-repo-url>
cd Jamph-ML-Trainer

# Create configuration file
Copy-Item example.jamph.env jamph.env

# Edit jamph.env with your credentials:
# - HF_USERNAME: Your HuggingFace username
# - HF_TOKEN: Get from https://huggingface.co/settings/tokens (needs write permission)
notepad jamph.env
```

### 2. Build Docker Image

```powershell
docker-compose build quantizer
```

This builds a Docker image with:
- llama.cpp compiled from source
- Python environment with dependencies
- All quantization tools ready

**Build time:** ~5-10 minutes (one-time only)

### 3. Quantize a Model

```powershell
# Download + Quantize + Upload to HuggingFace (default: Q4_K_M)
docker-compose run --rm quantizer process Qwen/Qwen2.5-Coder-7B

# Keep files locally (don't delete after upload)
docker-compose run --rm quantizer process Qwen/Qwen2.5-Coder-7B --keep-files

# Skip upload entirely (local quantization only)
docker-compose run --rm quantizer process Qwen/Qwen2.5-Coder-7B --skip-upload

# Multiple quantization methods
docker-compose run --rm quantizer process Qwen/Qwen2.5-Coder-7B --methods Q4_K_M,Q5_K_M,Q8_0
```

**Output:** `model training/Models/jamph-qwen2.5-coder-7b-q4_k_m/`

## Usage Examples

### Basic Commands

```powershell
# 1. Download only (no quantization)
docker-compose run --rm quantizer download Qwen/Qwen2.5-Coder-7B

# 2. Quantize existing downloaded model
docker-compose run --rm quantizer quantize /models/qwen2.5-coder-7b --methods Q4_K_M

# 3. Complete pipeline (download → quantize → upload)
docker-compose run --rm quantizer process Qwen/Qwen2.5-Coder-7B
```

### Quantization Methods

| Method | Size Reduction | Quality | Use Case |
|--------|---------------|---------|----------|
| **Q4_K_M** | ~75% smaller | Good | Default, best speed/quality balance |
| **Q5_K_M** | ~70% smaller | Better | Higher quality, slightly larger |
| **Q8_0** | ~50% smaller | Best | Maximum quality, least compression |

### Options

```powershell
# Keep source files after quantization
--keep-files

# Skip HuggingFace upload
--skip-upload

# Use specific quantization methods
--methods Q4_K_M,Q5_K_M,Q8_0

# Skip download if model already exists locally
--skip-download
```

## Configuration

Edit `jamph.env`:

```bash
# REQUIRED
HF_USERNAME=your-username
HF_TOKEN=hf_xxxxx

# OPTIONAL
MODEL_PREFIX=jamph                # Prefix for model names
GITHUB_HANDLE=YourName           # Your name in logs
TEAM=YourTeam                    # Team name
ORGANIZATION=YourOrg             # Organization
```

**Get HF Token:**
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Select "Write" permission (for uploads)
4. Copy token to `jamph.env`

## Output Structure

```
model training/Models/
├── .metadata/                                    # RAG-friendly metadata
│   └── qwen2.5-coder-7b.json
└── jamph-qwen2.5-coder-7b-q4_k_m/               # Quantized model folder
    ├── jamph-qwen2.5-coder-7b-q4_k_m.gguf       # Quantized model file
    ├── MODEL_LOG.md                              # Quantization details
    ├── quantization_metadata.json                # Metadata
    └── README.md                                 # HuggingFace model card
```

**Note:** By default, GGUF files are deleted after upload to save disk space. Use `--keep-files` to retain them locally.

## Troubleshooting

### Docker not running
```
Error: cannot connect to Docker daemon
```
**Solution:** Start Docker Desktop and wait for it to fully initialize

### No configuration file
```
no configuration file provided: not found
```
**Solution:** Create `jamph.env` from the example:
```powershell
Copy-Item example.jamph.env jamph.env
```

### Build fails
**Solution:** Ensure Docker has sufficient resources:
- **Memory:** 8GB minimum (16GB recommended)
- **Disk:** 20GB free space
- Check: Docker Desktop → Settings → Resources

### Upload fails
```
HTTP 401 Unauthorized
```
**Solution:** 
1. Verify `HF_TOKEN` in `jamph.env` is correct
2. Ensure token has **write** permission
3. Create new token at https://huggingface.co/settings/tokens

### Model not found
```
Repository not found
```
**Solution:** Use correct HuggingFace model ID format:
- ✅ `Qwen/Qwen2.5-Coder-7B`
- ❌ `qwen/qwen2.5-coder:7b`

## Advanced Usage

### PowerShell Helper Script

```powershell
.\quantize.ps1 -ModelId "Qwen/Qwen2.5-Coder-7B" -Method Q4_K_M -SkipUpload
```

### Individual CLI Commands

```powershell
# Download
docker-compose run --rm quantizer download Qwen/Qwen2.5-Coder-7B

# Quantize
docker-compose run --rm quantizer quantize /models/qwen2.5-coder-7b

# Upload
docker-compose run --rm quantizer upload /models/path-to.gguf Qwen/Qwen2.5-Coder-7B

# Full pipeline
docker-compose run --rm quantizer process Qwen/Qwen2.5-Coder-7B
```

---

**License:** Apache 2.0  
**Created for:** NAV Educational Initiative
