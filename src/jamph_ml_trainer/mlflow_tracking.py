"""MLflow Model Registry integration for versioned model artifacts."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

import mlflow
from rich.console import Console

console = Console()


class MLflowTracker:
    """Manages MLflow Model Registry for quantized models."""

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI (defaults to env var)
        """
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI",
            f"file://{Path.cwd() / 'mlruns'}"
        )
        
        mlflow.set_tracking_uri(self.tracking_uri)
        
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")
        if username and password:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        
        console.print(f"[cyan]MLflow Model Registry:[/cyan] {self.tracking_uri}")

    def register_model(
        self,
        model_path: Path,
        source_model: str,
        quantization_method: str,
        git_username: Optional[str] = None,
        team: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register quantized model in MLflow Model Registry.
        Model name matches GGUF filename for predictability.
        
        Args:
            model_path: Path to quantized GGUF file
            source_model: Original HuggingFace model ID
            quantization_method: Q4_K_M, Q5_K_M, or Q8_0
            git_username: Creator username
            team: Team name
            tags: Optional additional tags
        """
        model_name = model_path.name
        
        if tags is None:
            tags = {}
        
        tags["created_at"] = datetime.now().isoformat()
        tags["filename"] = model_name
        tags["source_model"] = source_model
        tags["quantization_method"] = quantization_method
        
        if git_username:
            tags["creator"] = git_username
        if team:
            tags["team"] = team
        
        try:
            # Register directly without experiment run
            mlflow.register_model(
                model_uri=f"file://{model_path.absolute()}",
                name=model_name,
                tags=tags
            )
            
            console.print(f"[green]✓[/green] Registered model: [bold]{model_name}[/bold]")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Model registration failed: {e}")
