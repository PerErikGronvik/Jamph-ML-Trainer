#!/usr/bin/env python3
"""Check if HuggingFace repo exists and list files"""
from huggingface_hub import HfApi

api = HfApi()
repo_id = "pererikgronvik/jamph-qwen2.5-0.5b-q4_k_m"

try:
    files = api.list_repo_files(repo_id, repo_type="model")
    print(f"✓ Repo exists: {repo_id}")
    print(f"\nFiles in repo ({len(files)}):")
    for f in files:
        print(f"  - {f}")
except Exception as e:
    print(f"✗ Error: {e}")
