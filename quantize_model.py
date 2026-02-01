#!/usr/bin/env python3
"""Model Quantization Script - CPU Optimized"""

import os
import sys
import json
import subprocess
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

QUANTIZATION_CONFIG = {
    "method": "Q4_K_M",
    "source_model_path": "model training/Models/qwen2.5-coder-1.5b-instruct",
}

METADATA_CONFIG = {
    "quantized_by": os.getenv("GITHUB_HANDLE", "Unknown Developer"),
    "team": os.getenv("TEAM", "Unknown Team"),
    "role": os.getenv("ROLE", "Unknown Role"),
}

SCRIPT_DIR = Path(__file__).parent
OUTPUT_CONFIG = {
    "models_dir": SCRIPT_DIR / "model training" / "Models",
    "crash_reports_dir": SCRIPT_DIR / "logs",
}

SOURCE_SYSTEM_CONFIG = {
    "gpu": "NVIDIA RTX 4070 Mobile",
    "vram": "8GB",
    "cuda_version": "12.0",
    "ram": "32GB",
}

TARGET_SYSTEM_CONFIG = {
    "device": "CPU Only",
    "recommended_ram": "16GB minimum for Q4_K_M",
    "recommended_threads": "8+ CPU threads",
}

QUANTIZATION_METHODS = {
    "Q4_K_M": {"description": "4-bit, medium quality", "size_reduction": "~75%", "quality": "Excellent", "ram_required": "~6-8GB", "speed": "Fast"},
    "Q5_K_M": {"description": "5-bit, higher quality", "size_reduction": "~70%", "quality": "Better", "ram_required": "~8-10GB", "speed": "Moderate"},
    "Q8_0": {"description": "8-bit, excellent quality", "size_reduction": "~50%", "quality": "Near-original", "ram_required": "~12-16GB", "speed": "Slower"},
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
    report_name = f"crash_report_quantize_model_{timestamp}.log"
    report_path = OUTPUT_CONFIG["crash_reports_dir"] / report_name
    
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("QUANTIZATION SCRIPT CRASH REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Script: quantize_model.py\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"Platform: {sys.platform}\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Method: {QUANTIZATION_CONFIG['method']}\n")
        f.write(f"  User: {METADATA_CONFIG['quantized_by']}\n")
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

def check_system():
    print_header("System Check")
    print_info(f"Source: {SOURCE_SYSTEM_CONFIG['gpu']}, {SOURCE_SYSTEM_CONFIG['ram']}")
    print_info(f"Target: {TARGET_SYSTEM_CONFIG['device']}")
    os.makedirs(OUTPUT_CONFIG["models_dir"], exist_ok=True)
    print_success("Output directories ready")

def copy_source_documentation(source_model_path, output_dir):
    """Copy all documentation and config files from source model for full traceability"""
    import shutil
    source_path = Path(source_model_path)
    output_path = Path(output_dir)
    
    files_copied = []
    was_finetuned = False
    
    # Copy LICENSE
    source_license = source_path / "LICENSE"
    if source_license.exists():
        dest_license = output_path / "LICENSE"
        shutil.copy2(source_license, dest_license)
        files_copied.append("LICENSE")
        print_success("Copied LICENSE")
    else:
        # Log missing license
        os.makedirs(OUTPUT_CONFIG["crash_reports_dir"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = OUTPUT_CONFIG["crash_reports_dir"] / f"missing_license_{timestamp}.log"
        
        with open(log_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MISSING LICENSE WARNING\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source Model: {source_model_path}\n")
            f.write(f"Output Directory: {output_dir}\n")
            f.write(f"\nWARNING: No LICENSE file found in source model.\n")
            f.write(f"This may indicate licensing compliance issues.\n")
            f.write(f"Please verify the licensing terms before distribution.\n")
        
        print_warning(f"No LICENSE found - logged to: {log_path}")
    
    # Copy README.md from source if exists
    source_readme = source_path / "README.md"
    if source_readme.exists():
        dest_readme = output_path / "SOURCE_README.md"
        shutil.copy2(source_readme, dest_readme)
        files_copied.append("SOURCE_README.md")
        print_success("Copied README.md → SOURCE_README.md")
    
    # Copy MODEL_LOG.md if exists (inherits history from source model)
    source_log = source_path / "MODEL_LOG.md"
    if source_log.exists():
        dest_log = output_path / "MODEL_LOG.md"
        shutil.copy2(source_log, dest_log)
        files_copied.append("MODEL_LOG.md (inherited)")
        print_success("Copied MODEL_LOG.md (inheriting model history)")
    
    # Copy FINETUNING.md if exists (for finetuned → quantized workflow)
    source_finetuning = source_path / "FINETUNING.md"
    if source_finetuning.exists():
        dest_finetuning = output_path / "FINETUNING.md"
        shutil.copy2(source_finetuning, dest_finetuning)
        files_copied.append("FINETUNING.md")
        print_success("Copied FINETUNING.md (finetuned model detected)")
        was_finetuned = True
    
    # Copy config files for full traceability
    config_files = ["config.json", "tokenizer_config.json", "generation_config.json"]
    for config_file in config_files:
        source_config = source_path / config_file
        if source_config.exists():
            dest_config = output_path / f"SOURCE_{config_file}"
            shutil.copy2(source_config, dest_config)
            files_copied.append(f"SOURCE_{config_file}")
    
    if files_copied:
        print_success(f"Copied {len(files_copied)} file(s) for traceability: {', '.join(files_copied)}")
    
    return was_finetuned

def calculate_size_reduction(original_path, quantized_path):
    def get_dir_size(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size / (1024**3)
    
    original_size_gb = get_dir_size(original_path)
    quantized_size_gb = get_dir_size(quantized_path)
    reduction_pct = ((original_size_gb - quantized_size_gb) / original_size_gb) * 100 if original_size_gb > 0 else 0
    return original_size_gb, quantized_size_gb, reduction_pct

# ============================================================================
# CONVERSION
# ============================================================================

def convert_to_gguf(model_path):
    print_header(f"Converting Model: {model_path}")
    
    model_name = Path(model_path).name
    quant_method = QUANTIZATION_CONFIG["method"]
    model_suffix = f"_{quant_method.lower()}"
    output_name = f"{model_name}{model_suffix}"
    output_dir = os.path.join(OUTPUT_CONFIG["models_dir"], output_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy source documentation (LICENSE, README.md, FINETUNING.md if exists)
    was_finetuned = copy_source_documentation(model_path, output_dir)
    
    print_info(f"Method: {quant_method}")
    print_info(f"Output: {output_dir}")
    
    fp16_path = os.path.join(output_dir, f"{model_name}-f16.gguf")
    
    print_info("\nStep 1/2: Converting to GGUF FP16...")
    try:
        llama_cpp_dir = SCRIPT_DIR.parent / "llama.cpp"
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
        venv_python = SCRIPT_DIR.parent / ".venv" / "bin" / "python"
        
        if not convert_script.exists():
            print_error(f"Conversion script not found: {convert_script}")
            return output_dir, model_name, "tool_not_found"
        
        convert_cmd = [str(venv_python), str(convert_script), str(Path(model_path).resolve()), "--outfile", fp16_path, "--outtype", "f16"]
        print_info(f"Command: {' '.join(convert_cmd)}")
        
        result = subprocess.run(convert_cmd, capture_output=True, text=True, cwd=str(llama_cpp_dir))
        
        if result.returncode != 0:
            print_error("GGUF conversion failed")
            print(result.stderr)
            
            from transformers import AutoTokenizer, AutoModelForCausalLM
            print_info("Attempting alternative conversion...")
            
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="float16", device_map="cpu")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            temp_dir = os.path.join(output_dir, "temp_conversion")
            os.makedirs(temp_dir, exist_ok=True)
            model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            
            print_warning(f"Manual conversion required: {temp_dir}")
            return output_dir, model_name, model_suffix, "manual_conversion_needed"
        
        print_success("GGUF FP16 complete")
        
    except FileNotFoundError:
        print_error("llama.cpp not found")
        return output_dir, model_name, model_suffix, "tool_not_found"
    except Exception as e:
        print_error(f"Conversion failed: {e}")
        return output_dir, model_name, model_suffix, "conversion_failed"
    
    print_info(f"\nStep 2/2: Quantizing to {quant_method}...")
    quantized_path = os.path.join(output_dir, f"{model_name}-{quant_method}.gguf")
    
    try:
        llama_cpp_dir = SCRIPT_DIR.parent / "llama.cpp"
        quantize_tool = llama_cpp_dir / "build" / "bin" / "llama-quantize"
        
        if not quantize_tool.exists():
            print_error(f"Quantization tool not found: {quantize_tool}")
            return output_dir, model_name, model_suffix, "tool_not_found"
        
        quantize_cmd = [str(quantize_tool), fp16_path, quantized_path, quant_method]
        print_info(f"Command: {' '.join(quantize_cmd)}")
        
        result = subprocess.run(quantize_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print_error("Quantization failed")
            print(result.stderr)
            return output_dir, model_name, model_suffix, "quantization_failed"
        
        print_success(f"Quantization complete")
        
        # Delete FP16 intermediate file after successful quantization
        if os.path.exists(fp16_path):
            os.remove(fp16_path)
            print_success(f"Removed intermediate FP16 file")
        
        config_path = os.path.join(output_dir, "quantization_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "quantization_method": "gguf",
                "quant_type": quant_method,
                "original_model": model_path,
                "quantized_date": datetime.now().isoformat(),
                "changed_by": METADATA_CONFIG['quantized_by'],
                "team": METADATA_CONFIG['team'],
                "role": METADATA_CONFIG['role'],
                "source_system": SOURCE_SYSTEM_CONFIG,
                "target_system": TARGET_SYSTEM_CONFIG,
                "method_details": QUANTIZATION_METHODS.get(quant_method, {}),
            }, f, indent=2)
        
        print_success(f"Saved: {quantized_path}")
        return output_dir, model_name, model_suffix, was_finetuned, "success"
        
    except FileNotFoundError:
        print_error("llama-quantize not found")
        return output_dir, model_name, model_suffix, was_finetuned, "tool_not_found"
    except Exception as e:
        print_error(f"Quantization failed: {e}")
        return output_dir, model_name, model_suffix, was_finetuned, "error"

# ============================================================================
# DOCUMENTATION
# ============================================================================

def generate_documentation(model_name, model_suffix, quantized_path, original_path, was_finetuned=False, status="success"):
    print_header("Generating Documentation")
    
    quant_method = QUANTIZATION_CONFIG["method"]
    method_info = QUANTIZATION_METHODS.get(quant_method, {})
    output_dir = Path(quantized_path)
    
    # Create relative paths from workspace root
    relative_path = Path(quantized_path).relative_to(SCRIPT_DIR.parent)
    
    # Calculate sizes
    try:
        original_size_gb, quantized_size_gb, reduction_pct = calculate_size_reduction(original_path, quantized_path)
    except:
        original_size_gb = 0
        quantized_size_gb = 0
        reduction_pct = 0
    
    # Generate Modelfile for Ollama
    gguf_files = list(output_dir.glob("*.gguf"))
    gguf_filename = gguf_files[0].name if gguf_files else f"{model_name}-{quant_method}.gguf"
    
    modelfile_content = f"""# Ollama Modelfile for {model_name}
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

FROM ./{gguf_filename}

# Temperature: Lower = more focused, Higher = more creative
PARAMETER temperature 0.7

# Top P: Nucleus sampling threshold
PARAMETER top_p 0.9

# Top K: Limits token selection to top K tokens
PARAMETER top_k 40

# Repeat Penalty: Reduces repetition
PARAMETER repeat_penalty 1.1

# System prompt
SYSTEM \"\"\"You are a helpful AI assistant.\"\"\"
"""
    
    modelfile_path = output_dir / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    print_success(f"Created: Modelfile")
    
    # Append to MODEL_LOG.md
    log_entry = f"""
{'='*80}
## Quantization Log Entry
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Model**: {model_name}{model_suffix}  
**Quantized by**: {METADATA_CONFIG['quantized_by']} ({METADATA_CONFIG['team']}, {METADATA_CONFIG['role']})  

### Quantization Configuration
- **Method**: GGUF {quant_method}
- **Description**: {method_info.get('description', 'N/A')}
- **Quality**: {method_info.get('quality', 'N/A')}
- **Speed**: {method_info.get('speed', 'N/A')}
- **Pipeline**: {"Original → Fine-tuned → Quantized" if was_finetuned else "Original → Quantized"}

### Model Sizes
- **Original**: {original_size_gb:.2f} GB
- **Quantized**: {quantized_size_gb:.2f} GB
- **Reduction**: {reduction_pct:.1f}% (Expected: {method_info.get('size_reduction', 'N/A')})

### System Requirements
- **Target**: CPU-only (no GPU required)
- **RAM**: {method_info.get('ram_required', 'N/A')}
- **CPU**: {TARGET_SYSTEM_CONFIG['recommended_threads']}

### Source System Configuration
- **GPU**: {SOURCE_SYSTEM_CONFIG['gpu']}
- **VRAM**: {SOURCE_SYSTEM_CONFIG['vram']}
- **RAM**: {SOURCE_SYSTEM_CONFIG['ram']}
- **CUDA**: {SOURCE_SYSTEM_CONFIG['cuda_version']}

### Conversion Process
1. Converted HuggingFace model to GGUF FP16
2. Quantized FP16 to {quant_method}
3. Removed intermediate FP16 file
4. Generated Ollama Modelfile

### Output
- **Location**: `{relative_path}`
- **Config**: `{relative_path}/quantization_config.json`
- **Status**: {"SUCCESS" if status == "success" else f"WARNING - {status}"}

### Usage with Ollama
```bash
ollama create {model_name} -f Modelfile
ollama run {model_name}
```

{'='*80}
"""
    
    log_path = output_dir / "MODEL_LOG.md"
    
    # If MODEL_LOG.md doesn't exist, create header first
    if not log_path.exists():
        header = """# Model Activity Log

This file tracks all operations performed on this model.

"""
        with open(log_path, "w") as f:
            f.write(header)
        print_info("Created new MODEL_LOG.md")
    
    # Append quantization entry to MODEL_LOG.md
    with open(log_path, "a") as f:
        f.write(log_entry)
    
    print_success(f"Appended to: {log_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        print_header(f"Quantization Script - {QUANTIZATION_CONFIG['method']}")
        print_info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get model path from config or command line argument
        if len(sys.argv) > 1:
            # Command line argument takes precedence
            model_path = sys.argv[1]
        else:
            # Use configured path
            configured_path = QUANTIZATION_CONFIG.get("source_model_path")
            if not configured_path:
                print_error("No model path specified in QUANTIZATION_CONFIG['source_model_path']")
                print_error("Usage: python quantize_model.py <model_path>")
                print_error("   OR: Set 'source_model_path' in QUANTIZATION_CONFIG")
                sys.exit(1)
            
            # Resolve relative path from script directory
            if not os.path.isabs(configured_path):
                model_path = str(SCRIPT_DIR / configured_path)
            else:
                model_path = configured_path
        
        if not os.path.exists(model_path):
            print_error(f"Model not found: {model_path}")
            sys.exit(1)
        
        print_info(f"Source model: {model_path}")
        
        check_system()
        
        quantized_path, model_name, model_suffix, was_finetuned, status = convert_to_gguf(model_path)
        generate_documentation(model_name, model_suffix, quantized_path, model_path, was_finetuned, status)
        
        if status == "success":
            print_header("Complete")
            print_success(f"Model: {quantized_path}")
        else:
            print_warning(f"Status: {status}")
            
    except KeyboardInterrupt:
        print_warning("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"Failed: {e}")
        write_crash_report(e, f"Model: {sys.argv[1] if len(sys.argv) > 1 else 'unknown'}")
        sys.exit(1)

if __name__ == "__main__":
    main()
