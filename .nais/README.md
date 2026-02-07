# NAIS Deployment Configuration

## Prerequisites

NAV platform team must create these resources before deployment:

### 1. Azure Key Vault Secret

Create secret `jamph-ml-secrets` with these keys:

```bash
kubectl create secret generic jamph-ml-secrets \
  --namespace=team-researchops \
  --from-literal=HF_TOKEN='hf_xxxxx' \
  --from-literal=HF_USERNAME='navikt'
```

### 2. Persistent Volume Claims

Create PVCs for model storage and logs:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: jamph-ml-models
  namespace: team-researchops
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: jamph-ml-logs
  namespace: team-researchops
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

## Usage

### Deploy
Deployment happens automatically via GitHub Actions when pushing to `main` branch.

### Run Quantization Job

```bash
# Execute in the running pod
kubectl exec -it deployment/jamph-ml-trainer -n team-researchops -- \
  jamph-ml-trainer process --model-id qwen/Qwen2.5-Coder-1.5B --method Q4_K_M

# Or scale up and shell in
kubectl scale deployment jamph-ml-trainer --replicas=1 -n team-researchops
kubectl exec -it deployment/jamph-ml-trainer -n team-researchops -- bash
```

### Check Logs

```bash
kubectl logs -f deployment/jamph-ml-trainer -n team-researchops
```

### View Models

```bash
kubectl exec -it deployment/jamph-ml-trainer -n team-researchops -- ls -lh /models
```

## Output

- Quantized models uploaded to Ollama.com under `navikt/` namespace
- Models available at: `https://ollama.com/navikt/nav-{model-name}-{quantization}`
- RAG-friendly metadata saved to `/models/.metadata/`
