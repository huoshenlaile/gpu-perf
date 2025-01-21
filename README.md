# PyTorch GEMM TFLOPS and Memory Bandwidth Benchmark

This Python script uses PyTorch to measure the performance of General Matrix Multiply (GEMM) operations in TFLOPS and memory bandwidth in GB/s on a given hardware device. It supports CPU, CUDA-enabled GPUs, and Apple MPS.

## Features

- Benchmarks GEMM (matrix multiplication) performance in TFLOPS.
- Estimates memory bandwidth in GB/s.
- Supports multiple data types: FP64, FP32, FP16, BFLOAT16, INT32, and INT8.
- Works with CPU, CUDA-enabled GPUs, and Apple MPS.
- Includes optional TensorFloat-32 (TF32) support for GPUs.

## Requirements

- Python 3.9 or later
- [PyTorch](https://pytorch.org/get-started/locally/) installed
- A CUDA-enabled GPU or Apple MPS device (optional)

## Usage

### Run the Script

Simple usage:

```bash
python benchmark.py
```

Example output:

```
Using device: CUDA (cuda)

--- fp32 Benchmark ---
Matrix Multiplication Performance: 53.19 TFLOPS
Memory Bandwidth: 916.60 GB/s

--- fp16 Benchmark ---
Matrix Multiplication Performance: 176.39 TFLOPS
Memory Bandwidth: 916.18 GB/s
```


### Advanced Command-Line Arguments

The script accepts the following arguments:

| Argument             | Default Value | Description                                                   |
|----------------------|---------------|---------------------------------------------------------------|
| `--device`           | `auto`        | Device to benchmark: `auto`, `cpu`, `cuda`, or `mps`.         |
| `--matmul-size`      | `4096`        | Size of matrices for GEMM benchmark.                         |
| `--memory-size`      | `8192`        | Size of tensors for memory bandwidth benchmark.              |
| `--iterations-matmul`| `10`          | Number of iterations for GEMM benchmark.                     |
| `--iterations-memory`| `1000`        | Number of iterations for memory bandwidth benchmark.         |
| `--tf32`             | `False`       | Enable TensorFloat-32 (TF32) for supported CUDA GPUs.        |
| `--types`            | `fp32,fp16`  | Data type for GEMM benchmark.                                 |

Possibly supported (depends on the specific device) data types: `fp64`, `fp32`, `fp16`, `bf16`, `int32`, `int8`. Specify multiple types separated by commas.

```bash
python benchmark.py --device cuda --matmul-size 4096 --memory-size 8192 --iterations-matmul 10 --iterations-memory 1000 --tf32 --types fp32,fp16
```
