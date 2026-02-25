"""Upload quantized models to HuggingFace Hub."""

import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi
from rich.console import Console

from .env_utils import get_hf_token, get_hf_username

console = Console()


class HuggingFaceUploader:
    """Handles uploading quantized models to HuggingFace Hub."""

    def __init__(self, username: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize uploader with HuggingFace credentials.
        
        Args:
            username: HuggingFace username (defaults to HF_USERNAME env var)
            token: HuggingFace API token (defaults to HF_TOKEN env var)
        """
        self.username = username or get_hf_username()
        self.token = token or get_hf_token()
        
        if not self.username:
            raise ValueError("HuggingFace username not found. Set HF_USERNAME environment variable.")
        
        if not self.token:
            raise ValueError("HuggingFace token not found. Set HF_TOKEN environment variable.")
        
        # Initialize HuggingFace API
        self.api = HfApi(token=self.token)
        
        console.print(f"[HUF-001] [green]✓[/green] HuggingFace uploader initialized for user: {self.username}")
    
    def _check_disk_space(self, required_mb: int = 500) -> bool:
        """Check if sufficient disk space is available.
        
        Args:
            required_mb: Minimum required space in MB
            
        Returns:
            True if sufficient space available
        """
        try:
            stat = shutil.disk_usage("/")
            available_mb = stat.free / (1024 * 1024)
            
            if available_mb < required_mb:
                console.print(f"[HUF-004] [red]✗[/red] Insufficient disk space: {available_mb:.0f}MB available, {required_mb}MB required")
                return False
            
            console.print(f"[HUF-005] [green]✓[/green] Disk space: {available_mb:.0f}MB available")
            return True
        except Exception as e:
            console.print(f"[HUF-006] [yellow]⚠[/yellow] Could not check disk space: {e}")
            return True  # Continue anyway

    def upload_model(
        self,
        model_name: str,
        model_dir: Path,
        gguf_file: Path,
        source_model: str,
        quantization_method: str,
        metadata: Optional[dict] = None,
        run_as_future: bool = False
    ) -> str:
        """
        Upload quantized model to HuggingFace Hub.
        
        Args:
            model_name: Model name (without username prefix)
            model_dir: Directory containing model files
            gguf_file: Path to quantized GGUF file
            source_model: Original HuggingFace model ID
            quantization_method: Quantization method used
            metadata: Optional metadata dict with keys: license, language, tags, pipeline_tag, author, summary, team
            run_as_future: If True, upload runs asynchronously (may help avoid MerkleDB errors)
            
        Returns:
            Full HuggingFace repo name (username/model-name)
        """
        repo_id = f"{self.username}/{model_name}"
        
        # Pre-upload checks
        console.print("[HUF-007] [cyan]Running pre-upload checks...[/cyan]")
        
        # Check disk space (estimate: 3x file size for safety)
        required_mb = int((gguf_file.stat().st_size / (1024 * 1024)) * 3)
        if not self._check_disk_space(required_mb):
            raise RuntimeError(f"Insufficient disk space. Need {required_mb}MB free.")
        
        try:
            # Create repository if it doesn't exist
            console.print(f"[HUF-008] [cyan]Creating/checking repository:[/cyan] {repo_id}")
            self.api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True
            )
            console.print("[HUF-009] [green]✓[/green] Repository ready")
            
            # Create README.md
            console.print(f"[HUF-010] [cyan]Creating README...[/cyan]")
            readme_content = self._generate_readme(
                model_name=model_name,
                source_model=source_model,
                quantization_method=quantization_method,
                gguf_filename=gguf_file.name,
                metadata=metadata or {}
            )
            
            readme_path = model_dir / "README.md"
            readme_path.write_text(readme_content, encoding="utf-8")
            console.print("[HUF-011] [green]✓[/green] Created README.md")
            
            # Upload entire folder (simple approach)
            console.print(f"[HUF-012] [cyan]Uploading model folder...[/cyan]")
            self.api.upload_folder(
                folder_path=str(model_dir),
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {quantization_method} quantized model",
            )
            console.print(f"[HUF-013] [green]✓[/green] Upload complete")
            
            console.print(f"[HUF-014] [bold green]✓ Upload complete![/bold green]")
            console.print(f"[HUF-015] [cyan]View at:[/cyan] https://huggingface.co/{repo_id}")
            console.print(f"[HUF-016] [cyan]Download with:[/cyan] huggingface-cli download {repo_id}")
            
            return repo_id

        except Exception as e:
            error_msg = str(e)
            console.print(f"[HUF-017] [bold red]✗ Upload failed:[/bold red] {error_msg}")
            console.print("[HUF-018] [yellow]Tip: Run the same command again - uploads resume from where they stopped.[/yellow]")
            
            raise

    def _fetch_source_metadata(self, source_model: str) -> dict:
        """Fetch metadata from source model on HuggingFace."""
        try:
            model_info = self.api.model_info(source_model)
            metadata = {}
            
            # Extract metadata from model card
            if hasattr(model_info, 'cardData') and model_info.cardData:
                card_data = model_info.cardData
                if card_data.get('license'):
                    metadata['license'] = card_data['license']
                if card_data.get('language'):
                    metadata['language'] = card_data['language'] if isinstance(card_data['language'], list) else [card_data['language']]
                if card_data.get('pipeline_tag'):
                    metadata['pipeline_tag'] = card_data['pipeline_tag']
                # Get relevant tags (filter for useful ones like 'code', 'coder', etc.)
                if card_data.get('tags'):
                    useful_tags = [tag for tag in card_data['tags'] if tag in ['code', 'coder', 'coding', 'python', 'javascript', 'sql', 'chat', 'instruct', 'conversational']]
                    if useful_tags:
                        metadata['inherited_tags'] = useful_tags
            
            return metadata
        except Exception as e:
            console.print(f"[HUF-029] [yellow]⚠[/yellow] Could not fetch source metadata: {e}")
            return {}
    
    def _generate_readme(
        self,
        model_name: str,
        source_model: str,
        quantization_method: str,
        gguf_filename: str,
        metadata: dict
    ) -> str:
        """Generate README.md for HuggingFace model card."""
        # Fetch source model metadata
        source_metadata = self._fetch_source_metadata(source_model)
        
        # Merge metadata: CLI flags override source metadata
        final_license = metadata.get('license') or source_metadata.get('license', '')
        final_language = metadata.get('language') or source_metadata.get('language', [])
        final_pipeline_tag = metadata.get('pipeline_tag') or source_metadata.get('pipeline_tag', '')
        
        # Combine tags: inherited + quantization + custom
        final_tags = ['quantized', 'gguf', quantization_method.lower(), 'llama-cpp']
        if source_metadata.get('inherited_tags'):
            final_tags.extend(source_metadata['inherited_tags'])
        if metadata.get('tags'):
            final_tags.extend(metadata['tags'])
        
        # Build YAML frontmatter
        yaml_parts = []
        if final_license:
            yaml_parts.append(f"license: {final_license}")
        yaml_parts.append(f"base_model: {source_model}")
        if final_language:
            yaml_parts.append("language:")
            for lang in final_language:
                yaml_parts.append(f"- {lang}")
        if final_pipeline_tag:
            yaml_parts.append(f"pipeline_tag: {final_pipeline_tag}")
        yaml_parts.append("tags:")
        for tag in final_tags:
            yaml_parts.append(f"- {tag}")
        
        yaml_section = "---\n" + "\n".join(yaml_parts) + "\n---"
        
        # Build minimal README content
        summary = metadata.get('summary', f"Quantized version of [{source_model}](https://huggingface.co/{source_model}) using {quantization_method}.")
        author = metadata.get('author', '')
        team = metadata.get('team', '')
        
        # Read default description if available
        default_desc = ""
        # Try multiple locations: workspace root (mounted volume) then package location
        desc_paths = [
            Path("/workspace/default_description.md"),  # Mounted workspace
            Path(__file__).parent.parent.parent / "default_description.md"  # Project root
        ]
        
        for default_desc_path in desc_paths:
            if default_desc_path.exists():
                try:
                    default_desc = default_desc_path.read_text(encoding="utf-8").strip()
                    break
                except Exception as e:
                    console.print(f"[HUF-030] [yellow]⚠[/yellow] Could not read {default_desc_path}: {e}")
        
        content_parts = [
            yaml_section,
            "",
            f"# {model_name}",
            "",
            summary,
            "",
            "### Model Description",
            ""
        ]
        
        # Add default description in Model Description section
        if default_desc:
            content_parts.append(default_desc)
            content_parts.append("")
        
        content_parts.extend([
            "## Details",
            "",
            f"- **Base Model**: [{source_model}](https://huggingface.co/{source_model})",
            f"- **Quantization**: {quantization_method}",
            f"- **Format**: GGUF (`{gguf_filename}`)"
        ])
        
        if author or team:
            content_parts.append("")
            if author:
                content_parts.append(f"- **Author**: {author}")
            if team:
                content_parts.append(f"- **Team**: {team}")
        
        content_parts.extend([
            "",
            "## Usage",
            "",
            "```bash",
            f"# Download",
            f"huggingface-cli download {self.username}/{model_name} {gguf_filename} --local-dir .",
            "",
            "# Run with llama.cpp",
            f"./llama-cli -m {gguf_filename} -p \"Your prompt\"",
            "```",
            ""
        ])
        
        return "\n".join(content_parts)
