#!/usr/bin/env python3
"""Model Download Script - HuggingFace Hub"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load developer configuration from root
DEVELOPER_ENV_PATH = Path(__file__).parent.parent / "developer.env"
if DEVELOPER_ENV_PATH.exists():
    load_dotenv(DEVELOPER_ENV_PATH)

# ============================================================================
# CONFIGURATION
# ============================================================================

DOWNLOAD_CONFIG = {
    "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",  
    "revision": "main",  
}

METADATA_CONFIG = {
    "downloaded_by": os.getenv("GITHUB_HANDLE", "Unknown Developer"),
    "team": os.getenv("TEAM", "Unknown Team"),
    "role": os.getenv("ROLE", "Unknown Role"),
}

SCRIPT_DIR = Path(__file__).parent
OUTPUT_CONFIG = {
    "models_dir": SCRIPT_DIR / "model training" / "Models",
    "crash_reports_dir": SCRIPT_DIR / "logs",
}

# ============================================================================
# UTILITIES
# ============================================================================

def print_header(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")

def print_info(text):
    print(f"[INFO] {text}")

def print_success(text):
    print(f"[SUCCESS] {text}")

def print_error(text):
    print(f"[ERROR] {text}")

def print_warning(text):
    print(f"[WARNING] {text}")

def write_crash_report(error, error_context=""):
    """Write detailed crash report to file"""
    os.makedirs(OUTPUT_CONFIG["crash_reports_dir"], exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_name = f"crash_report_download_model_{timestamp}.log"
    report_path = OUTPUT_CONFIG["crash_reports_dir"] / report_name
    
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL DOWNLOAD CRASH REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Script: download_model.py\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"Platform: {sys.platform}\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Model ID: {DOWNLOAD_CONFIG.get('model_id', 'N/A')}\n")
        f.write(f"  User: {METADATA_CONFIG['downloaded_by']}\n")
        f.write(f"  Team: {METADATA_CONFIG['team']}\n\n")
        
        if error_context:
            f.write(f"Context: {error_context}\n\n")
        
        f.write("Error:\n")
        f.write(f"{type(error).__name__}: {str(error)}\n\n")
        
        f.write("Traceback:\n")
        f.write(traceback.format_exc())
        
        f.write("\n" + "=" * 80 + "\n")
    
    print_error(f"Crash report saved: {report_path}")
    return report_path

# ============================================================================
# MODEL DOWNLOAD
# ============================================================================

def get_model_metadata(model_id, revision="main"):
    """Get metadata from HuggingFace Hub"""
    from huggingface_hub import HfApi, model_info
    
    print_info("Fetching model metadata from HuggingFace Hub...")
    
    try:
        api = HfApi()
        info = model_info(model_id, revision=revision)
        
        metadata = {
            "model_id": model_id,
            "revision": revision,
            "author": info.author if hasattr(info, 'author') else info.id.split('/')[0],
            "model_name": info.modelId,
            "last_modified": info.lastModified.isoformat() if hasattr(info, 'lastModified') else None,
            "sha": info.sha,
            "tags": info.tags if hasattr(info, 'tags') else [],
            "pipeline_tag": info.pipeline_tag if hasattr(info, 'pipeline_tag') else None,
            "library_name": info.library_name if hasattr(info, 'library_name') else None,
            "downloads": info.downloads if hasattr(info, 'downloads') else 0,
            "likes": info.likes if hasattr(info, 'likes') else 0,
            "card_data": info.card_data.to_dict() if hasattr(info, 'card_data') and info.card_data else {},
        }
        
        print_success(f"Model author: {metadata['author']}")
        print_success(f"Model: {metadata['model_name']}")
        print_success(f"Revision: {metadata['revision']} (SHA: {metadata['sha'][:8]}...)")
        
        return metadata
    
    except Exception as e:
        print_warning(f"Could not fetch full metadata: {e}")
        return {
            "model_id": model_id,
            "revision": revision,
            "author": model_id.split('/')[0] if '/' in model_id else "unknown",
            "model_name": model_id,
            "error": str(e)
        }

def download_model(model_id, revision="main", max_retries=3):
    """Download model from HuggingFace Hub with retry logic"""
    from huggingface_hub import snapshot_download
    import time
    
    print_header(f"Downloading Model: {model_id}")
    
    # Get metadata first
    metadata = get_model_metadata(model_id, revision)
    
    # Determine output directory
    model_name = model_id.split('/')[-1].lower()
    output_dir = OUTPUT_CONFIG["models_dir"] / model_name
    
    print_info(f"Output directory: {output_dir}")
    
    # Download model with retry logic
    for attempt in range(1, max_retries + 1):
        try:
            print_info(f"Downloading from HuggingFace Hub... (Attempt {attempt}/{max_retries})")
            print_info(f"URL: https://huggingface.co/{model_id}")
            
            downloaded_path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,
                resume_download=True,  # Resume if interrupted
                max_workers=4,  # Parallel downloads
            )
            
            print_success(f"Downloaded to: {output_dir}")
            
            return output_dir, metadata
        
        except KeyboardInterrupt:
            print_warning("\nInterrupted by user")
            raise
        
        except Exception as e:
            if attempt < max_retries:
                wait_time = attempt * 5  # Exponential backoff: 5s, 10s, 15s
                print_warning(f"Download interrupted: {e}")
                print_info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print_error(f"Download failed after {max_retries} attempts: {e}")
                raise

# ============================================================================
# DOCUMENTATION
# ============================================================================

def create_download_metadata(output_dir, model_id, revision, metadata):
    """Create download_metadata.json"""
    
    download_metadata = {
        "download_info": {
            "model_id": model_id,
            "huggingface_url": f"https://huggingface.co/{model_id}",
            "revision": revision,
            "sha": metadata.get('sha'),
            "downloaded_date": datetime.now().isoformat(),
            "downloaded_by": METADATA_CONFIG['downloaded_by'],
            "team": METADATA_CONFIG['team'],
            "role": METADATA_CONFIG['role'],
        },
        "model_info": {
            "author": metadata.get('author'),
            "model_name": metadata.get('model_name'),
            "last_modified": metadata.get('last_modified'),
            "tags": metadata.get('tags', []),
            "pipeline_tag": metadata.get('pipeline_tag'),
            "library_name": metadata.get('library_name'),
            "downloads": metadata.get('downloads', 0),
            "likes": metadata.get('likes', 0),
        },
        "card_data": metadata.get('card_data', {}),
    }
    
    metadata_path = output_dir / "download_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(download_metadata, f, indent=2)
    
    print_success(f"Created: download_metadata.json")
    return download_metadata

def create_model_log(output_dir, model_id, revision, metadata, download_metadata):
    """Create initial MODEL_LOG.md"""
    
    print_header("Creating MODEL_LOG.md")
    
    author = metadata.get('author', 'Unknown')
    model_name = metadata.get('model_name', model_id)
    sha = metadata.get('sha', 'unknown')
    last_modified = metadata.get('last_modified', 'Unknown')
    tags = ', '.join(metadata.get('tags', [])) if metadata.get('tags') else 'None'
    library = metadata.get('library_name', 'N/A')
    
    # Create relative path from script directory
    relative_path = output_dir.relative_to(SCRIPT_DIR.parent)
    
    log_content = f"""# Model Activity Log

This file tracks all operations performed on this model.

================================================================================
## Model Download
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Downloaded by**: {METADATA_CONFIG['downloaded_by']} ({METADATA_CONFIG['team']}, {METADATA_CONFIG['role']})  

### Source Information
- **Model ID**: `{model_id}`
- **HuggingFace URL**: https://huggingface.co/{model_id}
- **Author/Organization**: {author}
- **Revision**: {revision}
- **Commit SHA**: {sha[:8]}...
- **Last Modified**: {last_modified}

### Model Details
- **Library**: {library}
- **Tags**: {tags}
- **Pipeline**: {metadata.get('pipeline_tag', 'N/A')}
- **Downloads**: {metadata.get('downloads', 0):,}
- **Likes**: {metadata.get('likes', 0):,}

### Download Configuration
- **Local Path**: `{relative_path}`
- **Downloaded Files**: All model files from HuggingFace repository
- **Metadata**: `download_metadata.json`

### License
See LICENSE file in this directory for licensing information.

### Next Steps
1. Review model documentation (README.md)
2. (Optional) Fine-tune model for specific use case
3. Quantize model for deployment

================================================================================

"""
    
    log_path = output_dir / "MODEL_LOG.md"
    with open(log_path, "w") as f:
        f.write(log_content)
    
    print_success(f"Created: MODEL_LOG.md")

# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        print_header("Model Download Script")
        print_info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get model ID from config or command line argument
        if len(sys.argv) > 1:
            # Command line argument takes precedence
            model_id = sys.argv[1]
            revision = sys.argv[2] if len(sys.argv) > 2 else "main"
        else:
            # Use configured values
            model_id = DOWNLOAD_CONFIG.get("model_id")
            revision = DOWNLOAD_CONFIG.get("revision", "main")
            
            if not model_id:
                print_error("No model ID specified in DOWNLOAD_CONFIG['model_id']")
                print_error("Usage: python download_model.py <model_id> [revision]")
                print_error("   OR: Set 'model_id' in DOWNLOAD_CONFIG")
                print_error("\nExamples:")
                print_error("  python download_model.py Qwen/Qwen2.5-Coder-1.5B-Instruct")
                print_error("  python download_model.py meta-llama/Llama-2-7b-hf main")
                sys.exit(1)
        
        print_info(f"Model ID: {model_id}")
        print_info(f"Revision: {revision}")
        
        # Check if huggingface_hub is installed
        try:
            import huggingface_hub
            print_success(f"HuggingFace Hub library: v{huggingface_hub.__version__}")
        except ImportError:
            print_error("huggingface_hub not installed")
            print_info("Install with: pip install huggingface_hub")
            sys.exit(1)
        
        # Create output directory
        os.makedirs(OUTPUT_CONFIG["models_dir"], exist_ok=True)
        
        # Download model
        output_dir, metadata = download_model(model_id, revision)
        
        # Create documentation
        download_metadata = create_download_metadata(output_dir, model_id, revision, metadata)
        create_model_log(output_dir, model_id, revision, metadata, download_metadata)
        
        print_header("Complete")
        print_success(f"Model downloaded to: {output_dir}")
        print_success("Files created:")
        print_info(f"  - MODEL_LOG.md (activity log)")
        print_info(f"  - download_metadata.json (download metadata)")
        print_info(f"  - All model files from HuggingFace")
        
    except KeyboardInterrupt:
        print_warning("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"Failed: {e}")
        write_crash_report(e, f"Model: {sys.argv[1] if len(sys.argv) > 1 else DOWNLOAD_CONFIG.get('model_id', 'unknown')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
