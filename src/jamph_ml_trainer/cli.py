"""Unified CLI for model download, quantization, and deployment."""

import os
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .env_utils import get_git_username, get_hf_username, get_model_prefix, get_organization, get_team_name
from .huggingface_upload import HuggingFaceUploader
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
    
    console.print(f"[CLI-001] [cyan]Downloading:[/cyan] {model_id}")
    console.print(f"[CLI-002] [cyan]Saving as:[/cyan] {normalized_name}")
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        progress.add_task("Downloading model files...", total=None)
        
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
    
    console.print(f"[CLI-003] [green]✓[/green] Downloaded to: {output_dir}")
    return output_dir


def quantize_model_impl(
    model_path: Path,
    method: str = "Q4_K_M"
) -> Path:
    """Quantize model using llama.cpp."""
    import subprocess
    import shutil
    
    prefix = get_model_prefix()
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
    console.print(f"[CLI-004] [cyan]Converting to FP16 GGUF...[/cyan]")
    
    start_time = time.time()
    
    import sys
    subprocess.run(
        [sys.executable, str(convert_script), str(model_path), "--outfile", str(fp16_file)],
        check=True
    )
    
    # Step 2: Quantize FP16 to target method
    quantized_file = quantized_dir / f"{quantized_name}.gguf"
    console.print(f"[CLI-005] [cyan]Quantizing to {method}...[/cyan]")
    
    subprocess.run(
        [str(quantize_tool), str(fp16_file), str(quantized_file), method],
        check=True
    )
    
    # Delete intermediate FP16 file
    fp16_file.unlink()
    console.print("[CLI-006] [green]✓[/green] Deleted intermediate FP16 file")
    
    processing_time = time.time() - start_time
    
    # Create MODEL_LOG.md with complete documentation
    from datetime import datetime
    from .utils import save_json_metadata
    

    git_username = get_git_username()
    team = get_team_name()
    organization = get_organization()
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

- **Quantization System**: {get_model_prefix()}-quantization pipeline
- **Docker**: Multi-stage build with llama.cpp
- **UV Package Manager**: Python dependency management
""", encoding="utf-8")
    
    # Save quantization metadata as JSON
    metadata = {
        "model_name": quantized_name,
        "original_model": model_path.name,
        "quantization_method": method,
        "hyperparameters": {
            "quantization_type": method,
            "bits": "4" if "Q4" in method else "5" if "Q5" in method else "8",
            "k_quant": "K_M" if "K_M" in method else "K_S" if "K_S" in method else "K_L" if "K_L" in method else "none"
        },
        "license": "apache-2.0",
        "created_at": timestamp,
        "created_by": git_username,
        "team": team,
        "organization": organization,
        "original_size_mb": sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024**2),
        "quantized_size_mb": quantized_file.stat().st_size / (1024**2),
        "processing_time_seconds": processing_time
    }
    save_json_metadata(quantized_dir / "quantization_metadata.json", metadata)
    
    console.print("[CLI-007] [green]✓[/green] Created MODEL_LOG.md and metadata files")
    
    console.print(f"[CLI-008] [green]✓[/green] Quantized model: {quantized_file}")
    return quantized_file


@app.command()
def download(
    model_id: str = typer.Argument(..., help="HuggingFace model ID (e.g., qwen/Qwen2.5-Coder-1.5B)"),
    revision: str = typer.Option("main", help="Model revision/branch")
):
    """Download a model from HuggingFace Hub."""
    try:
        output_dir = download_model_impl(model_id, revision)
        console.print(f"[CLI-009] [bold green]✓ Download complete![/bold green] Saved to: {output_dir}")
    except Exception as e:
        console.print(f"[CLI-010] [bold red]✗ Download failed:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def quantize(
    model_path: str = typer.Argument(..., help="Path to model directory"),
    method: str = typer.Option("Q4_K_M", help="Quantization method (Q4_K_M, Q5_K_M, Q8_0)")
):
    """Quantize a model using llama.cpp."""
    try:
        model_dir = Path(model_path)
        if not model_dir.exists():
            console.print(f"[CLI-011] [bold red]✗ Model not found:[/bold red] {model_dir}")
            raise typer.Exit(1)
        
        quantized_file = quantize_model_impl(model_dir, method)
        
        console.print(f"[CLI-012] [cyan]Quantized model:[/cyan] {quantized_file.name}")
    except Exception as e:
        console.print(f"[CLI-013] [bold red]✗ Quantization failed:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def upload(
    model_path: str = typer.Argument(..., help="Path to quantized model GGUF file"),
    source_model: str = typer.Option(..., help="Original HuggingFace model ID"),
    license: Optional[str] = typer.Option(None, help="License identifier (inherits from source if not set)"),
    language: Optional[str] = typer.Option(None, help="Comma-separated language codes (inherits from source if not set)"),
    tags: Optional[str] = typer.Option(None, help="Comma-separated custom tags"),
    pipeline_tag: Optional[str] = typer.Option(None, help="Pipeline tag (inherits from source if not set)"),
    author: Optional[str] = typer.Option(None, help="Model card author"),
    summary: Optional[str] = typer.Option(None, help="Brief model summary"),
    team: Optional[str] = typer.Option(None, help="Team name")
):
    """Upload quantized model to HuggingFace/Ollama."""
    try:
        gguf_file = Path(model_path)
        if not gguf_file.exists():
            console.print(f"[CLI-014] [bold red]✗ Model file not found:[/bold red] {gguf_file}")
            raise typer.Exit(1)
        
        model_name = get_jamph_name(source_model)
        uploader = HuggingFaceUploader()
        
        # Extract quantization method from filename
        method = "Q4_K_M"  # default
        if "q5" in gguf_file.name.lower():
            method = "Q5_K_M"
        elif "q8" in gguf_file.name.lower():
            method = "Q8_0"
        
        # Prepare metadata dict
        metadata = {}
        if license:
            metadata['license'] = license
        if language:
            metadata['language'] = [lang.strip() for lang in language.split(',')]
        if tags:
            metadata['tags'] = [tag.strip() for tag in tags.split(',')]
        if pipeline_tag:
            metadata['pipeline_tag'] = pipeline_tag
        if author:
            metadata['author'] = author
        if summary:
            metadata['summary'] = summary
        if team:
            metadata['team'] = team
        
        repo_name = uploader.upload_model(
            model_name=model_name,
            model_dir=gguf_file.parent,
            gguf_file=gguf_file,
            source_model=source_model,
            quantization_method=method,
            metadata=metadata
        )
        
        console.print(f"[CLI-015] [bold green]✓ Upload complete![/bold green] View at: https://huggingface.co/{repo_name}")
    except Exception as e:
        console.print(f"[CLI-016] [bold red]✗ Upload failed:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def process(
    model_id: str = typer.Argument(..., help="HuggingFace model ID"),
    methods: str = typer.Option("Q4_K_M", help="Comma-separated quantization methods (Q4_K_M,Q5_K_M,Q8_0)"),
    skip_download: bool = typer.Option(False, help="Skip download if model exists"),
    skip_upload: bool = typer.Option(False, help="Skip upload to Ollama"),
    license: Optional[str] = typer.Option(None, help="License identifier (inherits from source if not set)"),
    language: Optional[str] = typer.Option(None, help="Comma-separated language codes (inherits from source if not set)"),
    tags: Optional[str] = typer.Option(None, help="Comma-separated custom tags"),
    pipeline_tag: Optional[str] = typer.Option(None, help="Pipeline tag (inherits from source if not set)"),
    author: Optional[str] = typer.Option(None, help="Model card author"),
    summary: Optional[str] = typer.Option(None, help="Brief model summary"),
    team: Optional[str] = typer.Option(None, help="Team name"),
    keep_files: bool = typer.Option(False, help="Keep source and quantized files after upload"),
    async_upload: bool = typer.Option(True, "--async-upload/--sync-upload", help="Upload asynchronously (default, helps avoid MerkleDB errors)")
):
    """Complete pipeline: download → quantize (multiple methods) → upload → cleanup."""
    try:
        from datetime import datetime
        from .utils import create_rag_metadata, cleanup_source_model, cleanup_quantized_file
        
        # Parse quantization methods
        method_list = [m.strip() for m in methods.split(",")]
        
        # Download uses normalized name (no prefix)
        normalized_name = get_jamph_name(model_id, include_prefix=False)
        console.print(f"[CLI-017] [bold cyan]Starting pipeline for {normalized_name}[/bold cyan]")
        console.print(f"[CLI-018] [cyan]Quantization methods:[/cyan] {', '.join(method_list)}")
        
        # Step 1: Download
        models_dir = get_models_dir()
        model_dir = models_dir / normalized_name
        
        if skip_download and model_dir.exists():
            console.print(f"[CLI-019] [yellow]⊚[/yellow] Skipping download, using existing: {model_dir}")
        else:
            console.print(f"[CLI-020] \n[bold]Step 1/{len(method_list) + 2}: Download[/bold]")
            model_dir = download_model_impl(model_id)
        
        # Step 2: Quantize (multiple methods)
        quantized_files = []
        quantizations_metadata = []
        
        for idx, method in enumerate(method_list, start=1):
            console.print(f"[CLI-021] \n[bold]Step {idx + 1}/{len(method_list) + 2}: Quantize ({method})[/bold]")
            quantized_file = quantize_model_impl(model_dir, method)
            quantized_files.append(quantized_file)
            
            # Store metadata for RAG
            quantizations_metadata.append({
                "method": method,
                "size_mb": quantized_file.stat().st_size / (1024**2),
                "uploaded_at": datetime.now().isoformat()
            })
            
            console.print(f"[CLI-022] [green]✓[/green] Quantized: {quantized_file.name}")
        
        # Step 3: Upload
        if not skip_upload:
            console.print(f"[CLI-023] \n[bold]Step {len(method_list) + 2}/{len(method_list) + 2}: Upload to HuggingFace[/bold]")
            uploader = HuggingFaceUploader()
            
            prefix = get_model_prefix()
            hf_username = get_hf_username()
            
            for quantized_file, method in zip(quantized_files, method_list):
                prefixed_quantized_name = f"{prefix}-{normalized_name}-{method.lower()}"
                
                # Prepare metadata dict
                metadata = {}
                if license:
                    metadata['license'] = license
                if language:
                    metadata['language'] = [lang.strip() for lang in language.split(',')]
                if tags:
                    metadata['tags'] = [tag.strip() for tag in tags.split(',')]
                if pipeline_tag:
                    metadata['pipeline_tag'] = pipeline_tag
                if author:
                    metadata['author'] = author
                if summary:
                    metadata['summary'] = summary
                if team:
                    metadata['team'] = team
                
                console.print(f"[CLI-024] [cyan]Uploading {method}...[/cyan]")
                uploader.upload_model(
                    model_name=prefixed_quantized_name,
                    model_dir=quantized_file.parent,
                    gguf_file=quantized_file,
                    source_model=model_id,
                    quantization_method=method,
                    metadata=metadata,
                    run_as_future=async_upload
                )
                console.print(f"[CLI-025] [green]✓[/green] Uploaded: {hf_username}/{prefixed_quantized_name}")
                
                # Cleanup quantized file after successful upload
                if not keep_files:
                    cleanup_quantized_file(quantized_file)
                    console.print(f"[CLI-026] [dim]Cleaned up: {quantized_file.name}[/dim]")
            
            # Create RAG-friendly metadata
            metadata_dir = models_dir / ".metadata"
            metadata_dir.mkdir(exist_ok=True)
            
            git_username = get_git_username()
            team = get_team_name()
            
            rag_metadata = create_rag_metadata(
                model_name=normalized_name,
                source_model=model_id,
                quantizations=quantizations_metadata,
                created_by=git_username,
                team=team,
                hf_username=hf_username
            )
            
            from .utils import save_json_metadata
            metadata_file = metadata_dir / f"{normalized_name}.json"
            save_json_metadata(metadata_file, rag_metadata)
            console.print(f"[CLI-027] [green]✓[/green] RAG metadata saved: {metadata_file}")
        else:
            console.print(f"[CLI-028] \n[yellow]⊚[/yellow] Skipping upload")
        
        # Cleanup source model files after all processing
        if not keep_files:
            console.print(f"[CLI-029] \n[dim]Cleaning up source model files...[/dim]")
            cleanup_source_model(model_dir)
            console.print(f"[CLI-030] [dim]Cleaned up: {model_dir}[/dim]")
        
        prefix = get_model_prefix()
        console.print(f"[CLI-031] \n[bold green]✓✓✓ Pipeline complete for {prefix}-{normalized_name}![/bold green]")
        console.print(f"[CLI-032] [green]Processed {len(method_list)} quantization(s): {', '.join(method_list)}[/green]")
        
    except Exception as e:
        console.print(f"[CLI-033] \n[bold red]✗ Pipeline failed:[/bold red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
