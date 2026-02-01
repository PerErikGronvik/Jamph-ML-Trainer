# Model Training & Deployment

This directory contains all model-related files, training scripts, and Docker deployment configuration.

## Directory Structure

```
model training/
├── Models/                      # All trained and quantized models
│   ├── qwen2.5-coder-1.5b-instruct/           # Original model (unquantized)
│   ├── qwen2.5-coder-1.5b-instruct_q4_k_m/    # Quantized (Q4_K_M)
│   ├── qwen2.5-coder-7b-instruct/             # Original model (unquantized)
│   └── qwen2.5-coder-7b-instruct_q4_k_m/      # Quantized (Q4_K_M)
├── Dockerfile                   # Docker image for Ollama with models
├── .dockerignore                # Exclude files from Docker build
└── run_models_config.json       # Configuration for which models to load
```

## Model Configuration

The `run_models_config.json` file controls which models are loaded into Ollama:

```json
{
  "models": [
    {
      "name": "qwen2.5-coder-1.5b-instruct",
      "path": "qwen2.5-coder-1.5b-instruct_q4_k_m",
      "modelfile": "Modelfile",
      "enabled": true,
      "description": "Qwen 2.5 Coder 1.5B - Q4_K_M quantized"
    }
  ],
  "settings": {
    "default_model": "qwen2.5-coder-1.5b-instruct",
    "auto_load_enabled_models": true
  }
}
```

### Model Properties
- **name**: Name used in Ollama (`ollama run <name>`)
- **path**: Directory name in `Models/` (relative path)
- **modelfile**: Modelfile to use (usually "Modelfile")
- **enabled**: Set to `true` to load this model on startup
- **description**: Human-readable description

## Model Documentation

Each model directory contains a `MODEL_LOG.md` file with an **append-only** log of all operations performed on that model:

- Download entries (from `download_model.py`) - Initial creation
- Fine-tuning entries (from `finetune_model.py`)
- Quantization entries (from `quantize_model.py`)
- Configuration details
- Size reduction metrics
- System requirements

This single file replaces the previous multi-file documentation approach (README.md, QUANTIZATION.md, FINETUNING.md, etc.).

The MODEL_LOG.md is created when the model is first downloaded and then each subsequent operation appends to it, creating a complete audit trail.

## Docker Deployment

### Build and Run

From the repository root:

```bash
# Build the Docker image
docker-compose build

# Start Ollama with configured models
docker-compose up -d

# Check logs
docker-compose logs -f ollama

# Test the model
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5-coder-1.5b-instruct",
  "prompt": "Write a hello world in Python"
}'
```

### Adding New Models

1. Place quantized model in `Models/` directory
2. Ensure it has a `Modelfile` and `MODEL_LOG.md`
3. Add entry to `run_models_config.json`:
   ```json
   {
     "name": "my-new-model",
     "path": "my-new-model_q4_k_m",
     "modelfile": "Modelfile",
     "enabled": true,
     "description": "Description of model"
   }
   ```
4. Rebuild and restart: `docker-compose up -d --build`

### Resource Management

The Docker container uses conservative settings (configured in `docker-compose.yml`):
- `OLLAMA_NUM_PARALLEL=1` - Process 1 request at a time
- `OLLAMA_MAX_LOADED_MODELS=1` - Keep 1 model in RAM

Adjust based on target system RAM:
- **8GB RAM**: Use 1.5B model, settings above
- **16GB+ RAM**: Can use 7B model, increase parallel to 2
- **32GB+ RAM**: Can run both models, parallel 2-4

## Training Scripts

Located in parent directory (`ai_models_training/`):

### 1. Download Models
```bash
python download_model.py <huggingface_model_id> [revision]
```
- Downloads models from HuggingFace Hub
- Creates initial MODEL_LOG.md with download information
- Captures model metadata (author, version, SHA, last modified)
- Creates download_metadata.json for full traceability
- Output: `model training/Models/<model_name>/`

**Examples:**
```bash
python download_model.py Qwen/Qwen2.5-Coder-1.5B-Instruct
python download_model.py meta-llama/Llama-2-7b-hf main
```

### 2. Fine-Tuning
```bash
python finetune_model.py <model_path>
```
- Applies LoRA fine-tuning to base models
- Uses training data from `training data/v1/`
- Appends fine-tuning log to MODEL_LOG.md
- Output: `model training/Models/<model_name>_ft_v1/`

### 3. Quantization
```bash
python quantize_model.py <model_path>
```
- Converts HuggingFace models to GGUF Q4_K_M format
- Copies all source documentation for traceability
- Generates Modelfile for Ollama
- Appends quantization log to MODEL_LOG.md
- Output: `model training/Models/<model_name>_q4_k_m/`

## Workflow

```
1. Download model:  python download_model.py Qwen/Qwen2.5-Coder-1.5B-Instruct
   → Creates model training/Models/qwen2.5-coder-1.5b-instruct/
   → Initial MODEL_LOG.md with download info
   
2. (Optional) Fine-tune: python finetune_model.py <model_path>
   → Creates model training/Models/<model>_ft_v1/
   → Appends to MODEL_LOG.md
   
3. Quantize: python quantize_model.py <model_path>
   → Creates model training/Models/<model>_q4_k_m/
   → Inherits and appends to MODEL_LOG.md
   
4. Add to run_models_config.json
   
5. Deploy: docker-compose up -d --build
```

Each step appends to the MODEL_LOG.md, creating a complete history of the model's lifecycle.
