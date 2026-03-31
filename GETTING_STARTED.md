# Getting Started Checklist

Follow these steps to set up and run the quantization pipeline from a fresh clone.

## ✅ Prerequisites

- [ ] Docker Desktop installed and **running**
- [ ] HuggingFace account created at https://huggingface.co

## ✅ Setup Steps

1. **Clone repository**
   ```powershell
   git clone <repository-url>
   cd Jamph-ML-Trainer
   ```

2. **Create configuration file**
   ```powershell
   Copy-Item example.jamph.env jamph.env
   ```

3. **Edit jamph.env with your credentials**
   - Open `jamph.env` in your editor
   - Set `HF_USERNAME` to your HuggingFace username
   - Set `HF_TOKEN` to your token from https://huggingface.co/settings/tokens
     - Token needs **write** permission for uploads

4. **Build Docker image** (one-time, ~5-10 minutes)
   ```powershell
   docker-compose build quantizer
   ```

5. **Run quantization**
   ```powershell
   # Example: Quantize Qwen 7B model
   docker-compose run --rm quantizer process Qwen/Qwen2.5-Coder-7B
   ```

## ✅ Verification

After successful run, you should see:
- `model training/Models/jamph-{model-name}-q4_k_m/` directory created
- Message: `✓✓✓ Pipeline complete for jamph-{model-name}!`
- Model uploaded to: `https://huggingface.co/{your-username}/jamph-{model-name}-q4_k_m`

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `cannot connect to Docker daemon` | Start Docker Desktop |
| `no configuration file provided` | Run: `Copy-Item example.jamph.env jamph.env` |
| `HTTP 401 Unauthorized` | Check `HF_TOKEN` has write permission |
| `Repository not found` | Verify HuggingFace model ID format (e.g., `Qwen/Qwen2.5-Coder-7B`) |

## 📝 Common Commands

```powershell
# Keep files locally (don't auto-cleanup)
docker-compose run --rm quantizer process Qwen/Qwen2.5-Coder-7B --keep-files

# Skip upload (local only)
docker-compose run --rm quantizer process Qwen/Qwen2.5-Coder-7B --skip-upload

# Multiple quantization methods
docker-compose run --rm quantizer process Qwen/Qwen2.5-Coder-7B --methods Q4_K_M,Q5_K_M,Q8_0
```

See [README.md](README.md) for complete documentation.
