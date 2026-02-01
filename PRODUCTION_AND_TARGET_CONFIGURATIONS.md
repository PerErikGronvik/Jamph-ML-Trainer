# System Information

**Document Purpose**: This file provides comprehensive system specifications for AI/ML development work on this machine and it's target machine. It should be referenced when making decisions about model configurations, batch sizes, and computational requirements.

---

## System Overview

**Hostname**: pererikmsi  
**User**: pererikgronvik

---

## Operating System

- **Distribution**: Ubuntu 24.04.3 LTS (Noble)
- **Kernel**: Linux 6.14.0-37-generic
- **Architecture**: x86_64
- **Kernel Type**: SMP PREEMPT_DYNAMIC

---

## CPU Specifications

- **Model**: Intel Core Ultra 9 185H
- **Architecture**: x86_64
- **Total CPUs**: 22 logical processors
- **Physical Cores**: 16
- **Threads per Core**: 2 (Hyper-Threading enabled)
- **Sockets**: 1
- **CPU Frequency**:
  - **Max**: 5100 MHz (5.1 GHz)
  - **Min**: 400 MHz

---

## GPU Configuration

### ⚡ NVIDIA RTX GPU (Dedicated for AI/CUDA Tasks)

**This GPU is 100% available for AI workloads - NOT used by the OS display**

- **Model**: NVIDIA GeForce RTX 4070 Max-Q / Mobile (AD106M)
- **VRAM**: 8188 MiB (8 GB)
- **NVIDIA Driver**: 580.95.05
- **CUDA Driver Version**: 13.0 (maximum supported)
- **CUDA Toolkit Version**: 12.0.140 (installed)
- **Bus ID**: 00000000:01:00.0



### Integrated GPU (Handles OS Display)

- **Model**: Intel Arc Graphics (Meteor Lake-P)
- **Bus ID**: 0000:00:02.0
- **Purpose**: Dedicated to OS display rendering

**System Configuration**: The laptop is configured with GPU offloading - the integrated Intel Arc GPU handles all OS display tasks, leaving the RTX 4070 completely available for AI computations (CUDA, TensorFlow, PyTorch, etc.).

---

## Memory

- **Total RAM**: 32 GiB
- **Swap**: 8 GiB

---

## Storage
- **Type**: NVMe SSD

---

## Python Environment

- **Python Version**: 3.12.3
- **Virtual Environment**: `/home/pererikgronvik/Github/jamph-sql-ki-assistent/.venv`

---

## CUDA Toolkit Configuration

**Status**: Fully Installed and Configured

- **CUDA Toolkit Version**: 12.0.140
- **CUDA Driver Support**: Up to version 13.0
- **Compiler**: nvcc available
- **Ready for**: PyTorch, TensorFlow, and other CUDA-enabled frameworks

---

### GPU Utilization
- **Full RTX 4070 Access**: No display overhead on the RTX GPU
- **CUDA 13.0 Compatible**: Use latest PyTorch/TensorFlow versions supporting CUDA 13
- **Power Profile**: 45W TDP (Mobile GPU)

### Compute Capabilities
- **CPU-Intensive Tasks**: 22 threads available (16 cores with HT)
- **Parallel Processing**: Excellent for data preprocessing and multi-threaded operations
- **GPU Acceleration**: Full 8GB VRAM dedicated to inference/training

## System Architecture considerations

1. **Optimal GPU Allocation**: RTX GPU is not burdened with display rendering
2. **Efficient Resource Usage**: Intel Arc handles UI, RTX handles compute
3. **Maximum VRAM Availability**: Full 8 GB available for model weights and activations
4. **Thermal Management**: Display tasks don't compete for GPU thermal headroom
5. **Performance Isolation**: Display lag won't impact AI workload performance
6. **Powesupply**: The laptop is battery and powersupply powered. During high full load it drains the battery and lasts for 6 hours. Tasks that require more than 6 hours should be planned in steps or with breaks or throttled.

---

## AI/ML Development Setup

### Installation Status

- **CUDA Toolkit**: v12.0.140
- **NVIDIA Drivers**: v580.95.05
- **PyTorch with CUDA**: v2.9.1+cu128
- **TensorFlow with CUDA**: v2.20.0

### Installed Frameworks

#### PyTorch (v2.9.1+cu128)
- **CUDA Version**: 12.8
- **Compute Capability**: 8.9
- **Packages**: torch, torchvision, torchaudio

#### TensorFlow (v2.20.0)
- **CUDA**: Built-in CUDA support
- **Backend**: Keras 3.13.0

### Optimal Batch Size Recommendations

**Training Configuration**:
- **Small Models** (< 1GB): Batch size 32-64
- **Medium Models** (1-3GB): Batch size 16-32
- **Large Models** (3-6GB): Batch size 4-16
- **Very Large Models** (> 6GB): Batch size 1-4 or use gradient accumulation

### Performance Optimization Tips

1. **Mixed Precision Training**: Use FP16/BF16 to reduce memory usage and increase speed
2. **Gradient Checkpointing**: Trade compute for memory on large models
3. **Model Parallelism**: Consider for models > 7GB
4. **Data Loading**: Utilize all 22 CPU threads for data preprocessing
5. **Batch Processing**: Process inference in batches to maximize GPU utilization

### Target system
- **NO GPU**
- **CPU**