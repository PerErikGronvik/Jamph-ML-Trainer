#!/usr/bin/env python3
"""
Compatibility Test Script
Tests dependencies and common runtime errors for the project
Logs output to timestamped log files

.venv/bin/python "ai_models_training/test_compability.py"
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path


def create_log_folder():
    """Create log folder if it doesn't exist"""
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_log_filename():
    """Generate log filename with current date and time"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"compability_test_{timestamp}.log"


class Logger:
    """Logger class to write to both console and file"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.file_handle = open(log_file, 'w')
    
    def log(self, message):
        """Write message to both console and log file"""
        print(message)
        self.file_handle.write(message + '\n')
        self.file_handle.flush()
    
    def close(self):
        """Close log file"""
        self.file_handle.close()


def test_cuda(logger):
    """Test CUDA availability"""
    logger.log("\nCUDA TEST")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        logger.log(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            logger.log(f"CUDA Version: {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.log(f"Device {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1024**3:.2f} GB)")
        
        logger.log("CUDA test passed")
        return True
        
    except Exception as e:
        logger.log(f"CUDA test failed: {str(e)}")
        return False


def test_pytorch(logger):
    """Test PyTorch functionality"""
    logger.log("\nPYTORCH TEST")
    
    try:
        import torch
        logger.log(f"PyTorch Version: {torch.__version__}")
        
        # Test basic tensor operations
        x = torch.rand(5, 3)
        y = torch.rand(5, 3)
        z = x + y
        logger.log(f"CPU tensor operations: OK")
        
        # Test CUDA tensor if available
        if torch.cuda.is_available():
            x_cuda = torch.rand(5, 3).cuda()
            y_cuda = torch.rand(5, 3).cuda()
            z_cuda = x_cuda + y_cuda
            logger.log(f"CUDA tensor operations: OK")
        
        # Test autograd
        x = torch.ones(2, 2, requires_grad=True)
        y = x + 2
        z = y * y * 3
        out = z.mean()
        out.backward()
        logger.log(f"Autograd: OK")
        
        logger.log("PyTorch test passed")
        return True
        
    except Exception as e:
        logger.log(f"PyTorch test failed: {str(e)}")
        return False


def test_cmake(logger):
    """Test CMake availability"""
    logger.log("\nCMAKE TEST")
    
    try:
        result = subprocess.run(['cmake', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            logger.log(f"CMake: {version}")
            logger.log("CMake test passed")
            return True
        else:
            logger.log("CMake not found")
            return False
    except Exception as e:
        logger.log(f"CMake test failed: {str(e)}")
        return False


def test_docker(logger):
    """Test Docker availability"""
    logger.log("\nDOCKER TEST")
    
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.log(f"Docker: {version}")
            
            # Test Docker daemon
            daemon_test = subprocess.run(['docker', 'info'], capture_output=True, text=True)
            if daemon_test.returncode == 0:
                logger.log("Docker daemon: Running")
            else:
                logger.log("Docker daemon: Not running or no permissions")
                return False
            
            logger.log("Docker test passed")
            return True
        else:
            logger.log("Docker not found")
            return False
    except Exception as e:
        logger.log(f"Docker test failed: {str(e)}")
        return False


def test_docker_compose(logger):
    """Test Docker Compose availability"""
    logger.log("\nDOCKER COMPOSE TEST")
    
    try:
        # Try docker compose (v2)
        result = subprocess.run(['docker', 'compose', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.log(f"Docker Compose: {version}")
            logger.log("Docker Compose test passed")
            return True
        
        # Try docker-compose (v1)
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.log(f"Docker Compose (legacy): {version}")
            logger.log("Docker Compose test passed")
            return True
        
        logger.log("Docker Compose not found")
        return False
    except Exception as e:
        logger.log(f"Docker Compose test failed: {str(e)}")
        return False


def test_llama_cpp(logger):
    """Test llama.cpp installation"""
    logger.log("\nLLAMA.CPP TEST")
    
    try:
        # Check for llama.cpp directory (relative to script location)
        script_dir = Path(__file__).parent
        llama_cpp_path = script_dir.parent / "llama.cpp"
        if not llama_cpp_path.exists():
            logger.log("llama.cpp directory not found")
            return False
        
        logger.log(f"llama.cpp directory: llama.cpp/")
        
        # Check for built executables
        quantize_tool = llama_cpp_path / "build" / "bin" / "llama-quantize"
        if quantize_tool.exists():
            logger.log(f"llama-quantize found: llama.cpp/build/bin/llama-quantize")
        else:
            logger.log("llama-quantize not found (build required)")
            return False
        
        # Check for conversion script
        convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
        if convert_script.exists():
            logger.log(f"convert_hf_to_gguf.py found")
        else:
            logger.log("convert_hf_to_gguf.py not found")
            return False
        
        logger.log("llama.cpp test passed")
        return True
        
    except Exception as e:
        logger.log(f"llama.cpp test failed: {str(e)}")
        return False


def test_python_packages(logger):
    """Test required Python packages"""
    logger.log("\nPYTHON PACKAGES TEST")
    
    packages = {
        'transformers': 'Transformers',
        'accelerate': 'Accelerate',
        'sentencepiece': 'SentencePiece',
        'google.protobuf': 'Protobuf',
        'huggingface_hub': 'HuggingFace Hub',
        'peft': 'PEFT (LoRA)',
        'datasets': 'Datasets',
        'dotenv': 'Python-dotenv'
    }
    
    all_passed = True
    
    for module_name, display_name in packages.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            logger.log(f"{display_name}: {version}")
        except ImportError:
            logger.log(f"{display_name}: NOT INSTALLED")
            all_passed = False
    
    if all_passed:
        logger.log("Python packages test passed")
    else:
        logger.log("Python packages test failed")
    
    return all_passed


def test_tensorflow(logger):
    """Test TensorFlow functionality"""
    logger.log("\nTENSORFLOW TEST")
    
    try:
        import tensorflow as tf
        logger.log(f"TensorFlow Version: {tf.__version__}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        logger.log(f"GPUs Available: {len(gpus)}")
        
        # Test basic operations
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        logger.log(f"CPU tensor operations: OK")
        
        # Test with GPU if available
        if gpus:
            with tf.device('/GPU:0'):
                x = tf.random.normal([1000, 1000])
                y = tf.random.normal([1000, 1000])
                z = tf.matmul(x, y)
                logger.log(f"GPU computation: OK")
        
        logger.log("TensorFlow test passed")
        return True
        
    except Exception as e:
        logger.log(f"TensorFlow test failed: {str(e)}")
        return False


# ============================================================================
# COMPATIBILITY TESTS
# ============================================================================

def test_llama_cpp_conversion(logger):
    """Test if llama.cpp conversion script can import required modules"""
    logger.log("\nLLAMA.CPP CONVERSION SCRIPT TEST")
    
    try:
        script_dir = Path(__file__).parent
        llama_cpp_dir = script_dir.parent / "llama.cpp"
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
        venv_python = script_dir.parent / ".venv" / "bin" / "python"
        
        if not convert_script.exists():
            logger.log(f"Conversion script not found: {convert_script}")
            return False
        
        # Test if the script can import transformers
        test_cmd = [str(venv_python), "-c", "from transformers import AutoConfig; print('OK')"]
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "OK" in result.stdout:
            logger.log("Conversion script dependencies: OK")
            logger.log(f"Script location: {convert_script}")
            return True
        else:
            logger.log("Conversion script cannot import transformers")
            logger.log(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.log(f"Conversion script test failed: {str(e)}")
        return False


def test_quantize_script_paths(logger):
    """Test if quantize script can find required tools"""
    logger.log("\nQUANTIZE SCRIPT PATHS TEST")
    
    try:
        script_dir = Path(__file__).parent
        llama_cpp_dir = script_dir.parent / "llama.cpp"
        
        # Check conversion script
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
        if convert_script.exists():
            logger.log(f"Conversion script: FOUND")
        else:
            logger.log(f"Conversion script: MISSING")
            return False
        
        # Check quantize tool
        quantize_tool = llama_cpp_dir / "build" / "bin" / "llama-quantize"
        if quantize_tool.exists():
            logger.log(f"Quantize tool: FOUND")
        else:
            logger.log(f"Quantize tool: MISSING")
            return False
        
        # Check venv python
        venv_python = script_dir.parent / ".venv" / "bin" / "python"
        if venv_python.exists():
            logger.log(f"VEnv Python: FOUND")
        else:
            logger.log(f"VEnv Python: MISSING")
            return False
        
        logger.log("All script paths valid")
        return True
        
    except Exception as e:
        logger.log(f"Path test failed: {str(e)}")
        return False


def test_training_data_structure(logger):
    """Test if training data structure is correct"""
    logger.log("\nTRAINING DATA STRUCTURE TEST")
    
    try:
        script_dir = Path(__file__).parent
        training_data_dir = script_dir / "training data"
        
        if not training_data_dir.exists():
            logger.log("Training data directory: MISSING")
            return False
        
        logger.log(f"Training data directory: FOUND")
        
        # Check for v1 folder
        v1_dir = training_data_dir / "v1"
        if not v1_dir.exists():
            logger.log("v1 folder: MISSING")
            return False
        
        logger.log("v1 folder: FOUND")
        
        # Check for jsonl files
        jsonl_files = list(v1_dir.glob("*.jsonl"))
        if not jsonl_files:
            logger.log("JSONL files: NONE FOUND")
            return False
        
        logger.log(f"JSONL files: {len(jsonl_files)} found")
        
        # Validate JSONL format
        import json
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r') as f:
                    lines = f.readlines()
                    if not lines:
                        logger.log(f"{jsonl_file.name}: EMPTY")
                        return False
                    
                    # Test first line
                    first_line = json.loads(lines[0])
                    required_keys = ['instruction', 'output']
                    if not all(key in first_line for key in required_keys):
                        logger.log(f"{jsonl_file.name}: Missing required keys")
                        return False
                    
                    logger.log(f"{jsonl_file.name}: Valid ({len(lines)} examples)")
            except Exception as e:
                logger.log(f"{jsonl_file.name}: Invalid format - {e}")
                return False
        
        logger.log("Training data structure: VALID")
        return True
        
    except Exception as e:
        logger.log(f"Training data test failed: {str(e)}")
        return False


def test_output_directories(logger):
    """Test if output directories can be created"""
    logger.log("\nOUTPUT DIRECTORIES TEST")
    
    try:
        script_dir = Path(__file__).parent
        
        # Test directories
        test_dirs = {
            "Models": script_dir / "Models",
            "model documentetion": script_dir.parent / "model documentetion",
            "model crash reports": script_dir / "model crash reports",
            "compatibility logs": script_dir / "compatibility logs",
        }
        
        all_ok = True
        for name, path in test_dirs.items():
            if path.exists():
                logger.log(f"{name}: EXISTS")
            else:
                # Try to create
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.log(f"{name}: CREATED")
                except Exception as e:
                    logger.log(f"{name}: CANNOT CREATE - {e}")
                    all_ok = False
        
        return all_ok
        
    except Exception as e:
        logger.log(f"Output directories test failed: {str(e)}")
        return False


def main():
    """Main function to run all tests"""
    # Create log folder and file
    log_dir = create_log_folder()
    log_filename = get_log_filename()
    log_path = os.path.join(log_dir, log_filename)
    
    logger = Logger(log_path)
    
    logger.log("=" * 80)
    logger.log("COMPATIBILITY TEST SUITE")
    logger.log("=" * 80)
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Python: {sys.version.split()[0]}")
    logger.log(f"Platform: {sys.platform}")
    
    # SECTION 1: DEPENDENCIES
    logger.log("\n" + "=" * 80)
    logger.log("SECTION 1: DEPENDENCIES")
    logger.log("=" * 80)
    
    dependency_results = {
        'CUDA': test_cuda(logger),
        'PyTorch': test_pytorch(logger),
        'CMake': test_cmake(logger),
        'Docker': test_docker(logger),
        'Docker Compose': test_docker_compose(logger),
        'llama.cpp': test_llama_cpp(logger),
        'Python Packages': test_python_packages(logger),
        'TensorFlow': test_tensorflow(logger)
    }
    
    # SECTION 2: COMPATIBILITY TESTS
    logger.log("\n" + "=" * 80)
    logger.log("SECTION 2: COMPATIBILITY TESTS")
    logger.log("=" * 80)
    
    compatibility_results = {
        'llama.cpp Conversion': test_llama_cpp_conversion(logger),
        'Quantize Script Paths': test_quantize_script_paths(logger),
        'Training Data Structure': test_training_data_structure(logger),
        'Output Directories': test_output_directories(logger)
    }
    
    # Combined results
    all_results = {**dependency_results, **compatibility_results}
    
    # Summary
    logger.log("\n" + "=" * 80)
    logger.log("TEST SUMMARY")
    logger.log("=" * 80)
    
    logger.log("\nDependencies:")
    for test_name, passed in dependency_results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.log(f"  {test_name}: {status}")
    
    logger.log("\nCompatibility:")
    for test_name, passed in compatibility_results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.log(f"  {test_name}: {status}")
    
    total_passed = sum(all_results.values())
    total_tests = len(all_results)
    logger.log(f"\nTotal: {total_passed}/{total_tests} passed")
    
    if all(all_results.values()):
        logger.log("\n✅ ALL TESTS PASSED - Project ready to use")
    else:
        logger.log("\n⚠️  SOME TESTS FAILED - Check issues above")
    
    logger.log(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 80)
    
    logger.close()
    
    print(f"\nLog file saved to: {log_path}")
    
    # Exit with appropriate code
    sys.exit(0 if all(all_results.values()) else 1)


if __name__ == "__main__":
    main()
