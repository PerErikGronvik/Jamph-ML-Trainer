"""Unified CLI for model download, quantization, and deployment."""

import os
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .mlflow_tracking import MLflowTracker
from .ollama_upload import OllamaUploader
from .utils import get_jamph_name, get_models_dir

app = typer.Typer(help="Jamph ML Trainer - Model quantization and fine-tuning toolkit")
console = Console()


def download_model_impl(model_id: str, revision: str = "main") -> Path:
    """Download model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download
    
    # Download without jamph- prefix
    normalized_name = get_jamph_name(model_id, include_prefix=False)
    models_dir = get_models_dir()
    output_dir = models_dir / normalized_name
    
    console.print(f"[cyan]Downloading:[/cyan] {model_id}")
    console.print(f"[cyan]Saving as:[/cyan] {normalized_name}")
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        progress.add_task("Downloading model files...", total=None)
        
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
    
    console.print(f"[green]✓[/green] Downloaded to: {output_dir}")
    return output_dir


def quantize_model_impl(
    model_path: Path,
    method: str = "Q4_K_M",
    mlflow_tracker: Optional[MLflowTracker] = None
) -> Path:
    """Quantize model using llama.cpp."""
    import subprocess
    import shutil
    
    import os
    prefix = os.getenv("MODEL_PREFIX", "jamph")
    model_name = model_path.name
    # Add team prefix at quantization stage
    prefixed_model_name = f"{prefix}-{model_name}" if not model_name.startswith(f"{prefix}-") else model_name
    quantized_name = f"{prefixed_model_name}-{method.lower()}"
    quantized_dir = model_path.parent / quantized_name
    quantized_dir.mkdir(exist_ok=True)
    
    # Find llama.cpp tools (Docker: /llama.cpp, Local: relative path)
    if Path("/llama.cpp").exists():
        # Docker environment
        llama_cpp_dir = Path("/llama.cpp")
        convert_script = llama_cpp_dir / "scripts" / "convert_hf_to_gguf.py"
        quantize_tool = llama_cpp_dir / "bin" / "llama-quantize"
    else:
        # Local environment
        llama_cpp_dir = Path(__file__).parent.parent.parent.parent / "llama.cpp"
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
        quantize_tool = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    
    if not convert_script.exists():
        raise FileNotFoundError(f"llama.cpp conversion script not found: {convert_script}")
    if not quantize_tool.exists():
        raise FileNotFoundError(f"llama-quantize tool not found: {quantize_tool}")
    
    # Step 1: Convert to FP16 GGUF
    fp16_file = quantized_dir / f"{model_name}-fp16.gguf"
    console.print(f"[cyan]Converting to FP16 GGUF...[/cyan]")
    
    start_time = time.time()
    
    import sys
    subprocess.run(
        [sys.executable, str(convert_script), str(model_path), "--outfile", str(fp16_file)],
        check=True
    )
    
    # Step 2: Quantize FP16 to target method
    quantized_file = quantized_dir / f"{quantized_name}.gguf"
    console.print(f"[cyan]Quantizing to {method}...[/cyan]")
    
    subprocess.run(
        [str(quantize_tool), str(fp16_file), str(quantized_file), method],
        check=True
    )
    
    # Delete intermediate FP16 file
    fp16_file.unlink()
    console.print("[green]✓[/green] Deleted intermediate FP16 file")
    
    processing_time = time.time() - start_time
    
    # Create MODEL_LOG.md with complete documentation
    from datetime import datetime
    from .utils import save_json_metadata
    
    git_username = os.getenv("GITHUB_HANDLE", os.getenv("USER", "unknown"))
    team = os.getenv("TEAM", "Unknown")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    model_log = quantized_dir / "MODEL_LOG.md"
    model_log.write_text(f"""# Model Documentation

## Overview

- **Quantized Model**: {quantized_name}.gguf
- **Original Model**: {model_path.name}
- **Quantization Method**: {method}
- **Created**: {timestamp}
- **Created By**: {git_username}
- **Team**: {team}

## Original Model

- **Source**: {model_path.name}
- **Size**: {sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024**3):.2f} GB

## Quantization Details

- **Method**: {method}
- **Quantized Size**: {quantized_file.stat().st_size / (1024**3):.2f} GB
- **Processing Time**: {processing_time:.1f} seconds
- **Compression**: {(1 - (quantized_file.stat().st_size / sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()))) * 100:.1f}%

## Usage

```bash
# With Ollama
ollama run {quantized_name}

# Direct GGUF loading
llama-cli -m {quantized_name}.gguf
```

## System Information

- **Quantization System**: {os.getenv('MODEL_PREFIX', 'jamph')}-quantization pipeline
- **Docker**: Multi-stage build with llama.cpp
- **UV Package Manager**: Python dependency management
""", encoding="utf-8")
    
    # Save quantization metadata as JSON
    metadata = {{
        "model_name": quantized_name,
        "original_model": model_path.name,
        "quantization_method": method,
        "created_at": timestamp,
        "created_by": git_username,
        "team": team,
        "original_size_mb": sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024**2),
        "quantized_size_mb": quantized_file.stat().st_size / (1024**2),
        "processing_time_seconds": processing_time
    }}
    save_json_metadata(quantized_dir / "quantization_metadata.json", metadata)
    
    console.print("[green]✓[/green] Created MODEL_LOG.md and metadata files")
    
    # Log metrics to MLflow if tracker provided
    if mlflow_tracker:
        git_username = os.getenv("GITHUB_HANDLE", os.getenv("USER", "unknown"))
        team = os.getenv("TEAM", "Unknown")
        
        mlflow_tracker.register_model(
            model_path=quantized_file,
            source_model=model_path.name,
            quantization_method=method,
            git_username=git_username,
            team=team
        )
    
    console.print(f"[green]✓[/green] Quantized model: {quantized_file}")
    return quantized_file


@app.command()
def download(
    model_id: str = typer.Argument(..., help="HuggingFace model ID (e.g., qwen/Qwen2.5-Coder-1.5B)"),
    revision: str = typer.Option("main", help="Model revision/branch")
):
    """Download a model from HuggingFace Hub."""
    try:
        output_dir = download_model_impl(model_id, revision)
        console.print(f"[bold green]✓ Download complete![/bold green] Saved to: {output_dir}")
    except Exception as e:
        console.print(f"[bold red]✗ Download failed:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def quantize(
    model_path: str = typer.Argument(..., help="Path to model directory"),
    method: str = typer.Option("Q4_K_M", help="Quantization method (Q4_K_M, Q5_K_M, Q8_0)"),
    no_mlflow: bool = typer.Option(False, help="Disable MLflow tracking")
):
    """Quantize a model using llama.cpp."""
    try:
        model_dir = Path(model_path)
        if not model_dir.exists():
            console.print(f"[bold red]✗ Model not found:[/bold red] {model_dir}")
            raise typer.Exit(1)
        
        mlflow_tracker = None if no_mlflow else MLflowTracker()
        
        quantized_file = quantize_model_impl(model_dir, method, mlflow_tracker)
        
        console.print(f"[cyan]Quantized model:[/cyan] {quantized_file.name}")
    except Exception as e:
        console.print(f"[bold red]✗ Quantization failed:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def upload(
    model_path: str = typer.Argument(..., help="Path to quantized model GGUF file"),
    source_model: str = typer.Option(..., help="Original HuggingFace model ID")
):
    """Upload quantized model to HuggingFace/Ollama."""
    try:
        gguf_file = Path(model_path)
        if not gguf_file.exists():
            console.print(f"[bold red]✗ Model file not found:[/bold red] {gguf_file}")
            raise typer.Exit(1)
        
        model_name = get_jamph_name(source_model)
        uploader = OllamaUploader()
        
        # Extract quantization method from filename
        method = "Q4_K_M"  # default
        if "q5" in gguf_file.name.lower():
            method = "Q5_K_M"
        elif "q8" in gguf_file.name.lower():
            method = "Q8_0"
        
        repo_name = uploader.upload_model(
            model_name=model_name,
            model_dir=gguf_file.parent,
            gguf_file=gguf_file,
            source_model=source_model,
            quantization_method=method
        )
        
        console.print(f"[bold green]✓ Upload complete![/bold green] View at: https://huggingface.co/{repo_name}")
    except Exception as e:
        console.print(f"[bold red]✗ Upload failed:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def process(
    model_id: str = typer.Argument(..., help="HuggingFace model ID"),
    method: str = typer.Option("Q4_K_M", help="Quantization method"),
    skip_download: bool = typer.Option(False, help="Skip download if model exists"),
    skip_upload: bool = typer.Option(False, help="Skip upload to HuggingFace"),
    no_mlflow: bool = typer.Option(False, help="Disable MLflow tracking")
):
    """Complete pipeline: download → quantize → upload."""
    try:
        # Download uses normalized name (no jamph- prefix)
        normalized_name = get_jamph_name(model_id, include_prefix=False)
        console.print(f"[bold cyan]Starting pipeline for {normalized_name}[/bold cyan]")
        
        # Step 1: Download
        models_dir = get_models_dir()
        model_dir = models_dir / normalized_name
        
        if skip_download and model_dir.exists():
            console.print(f"[yellow]⊙[/yellow] Skipping download, using existing: {model_dir}")
        else:
            console.print("\n[bold]Step 1/3: Download[/bold]")
            model_dir = download_model_impl(model_id)
        
        # Step 2: Quantize
        console.print("\n[bold]Step 2/3: Quantize[/bold]")
        mlflow_tracker = None if no_mlflow else MLflowTracker()
        
        quantized_file = quantize_model_impl(model_dir, method, mlflow_tracker)
        
        console.print(f"[cyan]Model registered as:[/cyan] {quantized_file.name}")
        
        # Step 3: Upload
        if not skip_upload:
            console.print("\n[bold]Step 3/3: Upload[/bold]")
            uploader = OllamaUploader()
            
            prefix = os.getenv("MODEL_PREFIX", "jamph")
            prefixed_quantized_name = f"{prefix}-{normalized_name}-{method.lower()}"
            
            uploader.upload_model(
                model_name=prefixed_quantized_name,
                model_dir=quantized_file.parent,
                gguf_file=quantized_file,
                source_model=model_id,
                quantization_method=method
            )
        else:
            console.print("\n[yellow]⊙[/yellow] Skipping upload")
        
        prefix = os.getenv("MODEL_PREFIX", "jamph")
        prefixed_quantized_name = f"{prefix}-{normalized_name}-{method.lower()}"
        console.print(f"\n[bold green]✓✓✓ Pipeline complete for {prefixed_quantized_name}![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Pipeline failed:[/bold red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
