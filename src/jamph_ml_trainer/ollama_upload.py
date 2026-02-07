"""Upload quantized models to Ollama.com registry."""

import os
import subprocess
import time
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
        
        console.print(f"[green]✓[/green] Ollama uploader initialized for user: {self.username}")
        if not self.token:
            console.print(f"[yellow]⚠[/yellow] No OLLAMA_TOKEN found - will require browser authentication")

    def _start_ollama_service(self) -> subprocess.Popen:
        """Start Ollama service in background."""
        console.print("[cyan]Starting Ollama service...[/cyan]")
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait for service to be ready (check for a few seconds)
        max_attempts = 15
        for attempt in range(max_attempts):
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    check=False,
                    capture_output=True,
                    timeout=2
                )
                if result.returncode == 0:
                    console.print("[green]✓[/green] Ollama service ready")
                    return process
            except (subprocess.TimeoutExpired, Exception):
                pass
            
            time.sleep(1)
        
        console.print("[yellow]⚠[/yellow] Ollama service may not be fully ready, continuing...")
        return process

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
        ollama_process = None
        
        try:
            # Start Ollama service in background
            ollama_process = self._start_ollama_service()
            
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
            
            # Check if already authenticated (credentials exist)
            ollama_dir = Path.home() / ".ollama"
            if (ollama_dir / "id_ed25519").exists():
                console.print("[dim]Using existing Ollama credentials...[/dim]")
            else:
                console.print("[yellow]⚠ No Ollama credentials found![/yellow]")
                console.print("[yellow]You need to authenticate once outside Docker:[/yellow]")
                console.print("[dim]1. Install Ollama on your host: https://ollama.com[/dim]")
                console.print("[dim]2. Run: ollama push <any-model-name>[/dim]")
                console.print("[dim]3. Complete browser authentication[/dim]")
                console.print("[dim]4. Mount ~/.ollama in docker-compose.yml[/dim]")
                console.print()
                console.print("[yellow]Attempting browser authentication (requires manual action)...[/yellow]")
            
            # Attempt push - will use existing credentials or trigger browser auth
            result = subprocess.run(
                ["ollama", "push", full_model_name],
                cwd=str(model_dir),
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                if "need to be signed in" in result.stderr.lower():
                    console.print(f"[bold red]✗ Authentication required![/bold red]")
                    console.print()
                    console.print("[yellow]To enable automatic uploads:[/yellow]")
                    console.print("1. On your host machine, run: [cyan]ollama pull llama3.2[/cyan]")
                    console.print("   (this will trigger browser authentication)")
                    console.print("2. Complete the browser login")
                    console.print("3. Add to docker-compose.quantize.yml volumes:")
                    console.print("   [cyan]- ${HOME}/.ollama:/root/.ollama:ro[/cyan]")
                    console.print("4. Rebuild and rerun")
                    console.print()
                    raise RuntimeError("Ollama authentication required - see instructions above")
                else:
                    console.print(f"[bold red]✗ Push failed:[/bold red] {result.stderr}")
                    raise RuntimeError(f"Failed to push model: {result.stderr}")
            
            console.print("[green]✓[/green] Model pushed to Ollama.com")
            
            # Verify the model is accessible
            console.print(f"[cyan]Verifying model availability...[/cyan]")
            time.sleep(2)  # Brief wait for propagation
            
            verify_result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )
            
            if full_model_name in verify_result.stdout:
                console.print(f"[green]✓[/green] Model verified locally")
            else:
                console.print(f"[yellow]⚠[/yellow] Model not found in local list")
            
            console.print(f"[bold green]✓ Upload complete![/bold green]")
            console.print(f"[cyan]View at:[/cyan] https://ollama.com/{self.username}/{model_name}")
            console.print(f"[cyan]Pull with:[/cyan] ollama pull {full_model_name}")
            console.print(f"[dim]Note: It may take a few minutes for the model to appear on ollama.com[/dim]")
            
            return full_model_name

        except FileNotFoundError:
            console.print("[bold red]✗ Ollama CLI not found![/bold red]")
            console.print("Install Ollama: https://ollama.com/download")
            raise
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]✗ Upload failed:[/bold red] {e}")
            raise
        finally:
            # Clean up Ollama service
            if ollama_process:
                console.print("[dim]Stopping Ollama service...[/dim]")
                ollama_process.terminate()
                try:
                    ollama_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    ollama_process.kill()

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
