"""Upload quantized models to Ollama.com registry."""

import os
import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


class OllamaUploader:
    """Handles uploading quantized models to Ollama.com."""

    def __init__(self, username: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize uploader with Ollama credentials.
        
        Args:
            username: Ollama.com username (defaults to OLLAMA_USERNAME env var)
            token: Ollama.com API token (defaults to OLLAMA_TOKEN env var)
        """
        self.username = username or os.getenv("OLLAMA_USERNAME")
        self.token = token or os.getenv("OLLAMA_TOKEN")
        
        if not self.username:
            raise ValueError("Ollama username not found. Set OLLAMA_USERNAME environment variable.")
        if not self.token:
            raise ValueError("Ollama token not found. Set OLLAMA_TOKEN environment variable.")
        
        console.print(f"[green]✓[/green] Ollama uploader initialized for user: {self.username}")

    def upload_model(
        self,
        model_name: str,
        model_dir: Path,
        gguf_file: Path,
        source_model: str,
        quantization_method: str
    ) -> str:
        """
        Upload quantized model to Ollama.com under user namespace.
        
        Args:
            model_name: Model name (without username prefix)
            model_dir: Directory containing model files
            gguf_file: Path to quantized GGUF file
            source_model: Original HuggingFace model ID
            quantization_method: Quantization method used
            
        Returns:
            Full Ollama model name (username/model-name)
        """
        full_model_name = f"{self.username}/{model_name}"
        
        try:
            # Generate Modelfile
            console.print(f"[cyan]Creating Modelfile for:[/cyan] {full_model_name}")
            modelfile_content = self._generate_modelfile(
                gguf_filename=gguf_file.name,
                source_model=source_model,
                quantization_method=quantization_method
            )
            
            modelfile_path = model_dir / "Modelfile"
            modelfile_path.write_text(modelfile_content, encoding="utf-8")
            console.print("[green]✓[/green] Created Modelfile")
            
            # Login to Ollama (if not already logged in)
            console.print("[cyan]Authenticating with Ollama.com...[/cyan]")
            try:
                subprocess.run(
                    ["ollama", "login"],
                    input=self.token.encode(),
                    check=True,
                    capture_output=True,
                    cwd=str(model_dir)
                )
                console.print("[green]✓[/green] Authenticated")
            except subprocess.CalledProcessError:
                console.print("[yellow]⚠[/yellow] Authentication may have failed, continuing...")
            
            # Create model from Modelfile
            console.print(f"[cyan]Creating Ollama model:[/cyan] {full_model_name}")
            subprocess.run(
                ["ollama", "create", full_model_name, "-f", "Modelfile"],
                check=True,
                cwd=str(model_dir)
            )
            console.print("[green]✓[/green] Model created locally")
            
            # Push to Ollama.com
            console.print(f"[cyan]Pushing to Ollama.com...[/cyan]")
            subprocess.run(
                ["ollama", "push", full_model_name],
                check=True,
                cwd=str(model_dir)
            )
            console.print("[green]✓[/green] Model pushed to Ollama.com")
            
            console.print(f"[bold green]✓ Upload complete![/bold green]")
            console.print(f"[cyan]View at:[/cyan] https://ollama.com/{self.username}/{model_name}")
            console.print(f"[cyan]Pull with:[/cyan] ollama pull {full_model_name}")
            
            return full_model_name

        except FileNotFoundError:
            console.print("[bold red]✗ Ollama CLI not found![/bold red]")
            console.print("Install Ollama: https://ollama.com/download")
            raise
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]✗ Upload failed:[/bold red] {e}")
            raise

    def _generate_modelfile(
        self,
        gguf_filename: str,
        source_model: str,
        quantization_method: str
    ) -> str:
        """Generate Modelfile for Ollama."""
        return f"""# Modelfile for {gguf_filename}
# Source: {source_model}
# Quantization: {quantization_method}

FROM ./{gguf_filename}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

# System prompt (customize as needed)
SYSTEM "You are a helpful AI assistant."
"""
