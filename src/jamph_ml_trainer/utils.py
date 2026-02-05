"""Utility functions for jamph-ml-trainer."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def normalize_model_name(model_id: str) -> str:
    """
    Normalize HuggingFace model ID (WITHOUT jamph- prefix).
    
    Examples:
        qwen/Qwen2.5-Coder-1.5B -> qwen2.5-coder-1.5b
        meta-llama/Llama-3.2-1B -> llama-3.2-1b
    """
    # Extract model name from org/model format
    if "/" in model_id:
        model_name = model_id.split("/")[-1]
    else:
        model_name = model_id
    
    # Normalize: lowercase, remove special chars except hyphens/dots
    normalized = model_name.lower().replace("_", "-")
    
    return normalized


def get_jamph_name(model_id: str, include_prefix: bool = True) -> str:
    """
    Convert HuggingFace model ID with optional team prefix.
    
    Args:
        model_id: HuggingFace model ID
        include_prefix: Whether to add team prefix (True for quantized/upload)
    
    Examples:
        get_jamph_name("qwen/Qwen2.5-Coder-1.5B", False) -> qwen2.5-coder-1.5b
        get_jamph_name("qwen/Qwen2.5-Coder-1.5B", True) -> jamph-qwen2.5-coder-1.5b
    """
    import os
    prefix = os.getenv("MODEL_PREFIX", "jamph")
    normalized = normalize_model_name(model_id)
    
    if include_prefix and not normalized.startswith(f"{prefix}-"):
        return f"{prefix}-{normalized}"
    
    return normalized


def append_to_model_log(log_path: Path, entry: str) -> None:
    """Append entry to MODEL_LOG.md with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n## [{timestamp}]\n{entry}\n")


def save_json_metadata(path: Path, data: Dict[str, Any]) -> None:
    """Save metadata dictionary as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_models_dir() -> Path:
    """Get the models directory path."""
    if os.path.exists("/models"):
        return Path("/models")
    return Path(__file__).parent.parent.parent / "model training" / "Models"


def create_rag_metadata(
    model_name: str,
    source_model: str,
    quantizations: list[Dict[str, Any]],
    created_by: str,
    team: str,
    ollama_username: str
) -> Dict[str, Any]:
    """
    Create RAG-friendly metadata for model quantizations.
    
    Args:
        model_name: Normalized model name (e.g., qwen2.5-coder-1.5b)
        source_model: HuggingFace model ID (e.g., qwen/Qwen2.5-Coder-1.5B)
        quantizations: List of quantization info dicts with method, size_mb, uploaded_at
        created_by: GitHub username or developer identifier
        team: Team name
        ollama_username: Ollama.com username for URL construction
    
    Returns:
        RAG-friendly metadata dictionary
    """
    prefix = os.getenv("MODEL_PREFIX", "jamph")
    
    return {
        "model": {
            "name": model_name,
            "full_name": f"{prefix}-{model_name}",
            "source": {
                "huggingface": source_model,
                "type": "transformer"
            }
        },
        "quantizations": [
            {
                "method": q["method"],
                "size_mb": round(q["size_mb"], 2),
                "uploaded_at": q["uploaded_at"],
                "ollama_url": f"https://ollama.com/{ollama_username}/{prefix}-{model_name}-{q['method'].lower()}",
                "ollama_command": f"ollama run {ollama_username}/{prefix}-{model_name}-{q['method'].lower()}"
            }
            for q in quantizations
        ],
        "metadata": {
            "created_by": created_by,
            "team": team,
            "created_at": datetime.now().isoformat(),
            "prefix": prefix
        },
        "usage": {
            "description": f"Quantized versions of {source_model} for efficient inference",
            "recommended": quantizations[0]["method"] if quantizations else "Q4_K_M",
            "notes": "Q4_K_M for best speed/quality balance, Q5_K_M for better quality, Q8_0 for highest quality"
        }
    }


def cleanup_source_model(model_dir: Path) -> None:
    """Delete downloaded source model files after quantization."""
    import shutil
    if model_dir.exists() and model_dir.is_dir():
        shutil.rmtree(model_dir)


def cleanup_quantized_file(gguf_file: Path) -> None:
    """Delete quantized GGUF file after successful upload."""
    if gguf_file.exists() and gguf_file.is_file():
        gguf_file.unlink()
