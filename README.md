# PyTorch GEMM TFLOPS and Memory Bandwidth Benchmark

This Python script uses PyTorch to measure the performance of General Matrix Multiply (GEMM) operations in TFLOPS and memory bandwidth in GB/s on a given hardware device. It supports CPU, CUDA-enabled GPUs, Apple MPS, Huawei Ascend, and Moore Threads GPU. This script **may not** reflect the peak performance of the hardware, but it can be useful to reflect the actual performance in PyTorch machine learning applications.

## Features

- Benchmarks GEMM (matrix multiplication) performance in TFLOPS.
- Estimates memory bandwidth in GB/s.
- Supports multiple data types: FP64, FP32, FP16, BFLOAT16, INT32, INT16, and INT8. (depends on the specific device and PyTorch version)
- Works with CPU, CUDA-enabled GPUs, and Apple MPS.
- Includes optional TensorFloat-32 (TF32) support for GPUs.

## Requirements

- Python 3.10 or later
- [PyTorch](https://pytorch.org/get-started/locally/) installed
- Nvidia GPU (optional)
- Apple MPS (optional)
- Huawei Ascend (optional)
- Moore Threads GPU (optional)

## Usage

### Run the Script

Simple usage:

```bash
python benchmark.py
```

Example output:

```
Using backend: cuda (NVIDIA GeForce RTX 5090)

--- fp64 Benchmark ---
Matrix Multiplication Performance:            1.74 TFLOPS
Memory Bandwidth:                             1424.08 GiB/s (1529.09 GB/s)
Roofline Ridge Point (Arithmetic Intensity):  1.25 FLOPS/Byte

--- fp32 Benchmark ---
Matrix Multiplication Performance:            66.40 TFLOPS
Memory Bandwidth:                             1421.18 GiB/s (1525.98 GB/s)
Roofline Ridge Point (Arithmetic Intensity):  47.84 FLOPS/Byte

--- fp16 Benchmark ---
Matrix Multiplication Performance:            237.74 TFLOPS
Memory Bandwidth:                             1421.02 GiB/s (1525.81 GB/s)
Roofline Ridge Point (Arithmetic Intensity):  171.31 FLOPS/Byte

--- bf16 Benchmark ---
Matrix Multiplication Performance:            240.31 TFLOPS
Memory Bandwidth:                             1421.16 GiB/s (1525.96 GB/s)
Roofline Ridge Point (Arithmetic Intensity):  173.15 FLOPS/Byte
```


### Advanced Command-Line Arguments

The script accepts the following arguments:

| Argument             | Default Value | Description                                                                 |
|----------------------|---------------|-----------------------------------------------------------------------------|
| `--device`           | `auto`        | Device to benchmark: `auto`, `cpu`, `cuda`, or `mps`.                       |
| `--types`            | `fp32,fp16`   | Comma-separated list of data types to benchmark (e.g., `fp64,fp32,fp16`).   |
| `--sec`              | `1.5`         | Time in seconds to run the benchmark.                                       |
| `--m`                | `16384`       | Matrix size M for GEMM benchmark.                                           |
| `--n`                | `16384`       | Matrix size N for GEMM benchmark.                                           |
| `--k`                | `16384`       | Matrix size K for GEMM benchmark.                                           |
| `--mem`              | `1073741824`  | Tensor size for memory benchmark.                                           |
| `--tf32`             | `False`       | Enable TensorFloat-32 (TF32) for supported CUDA GPUs.                       |
| `--all-zeros`        | `False`       | Use all-zero tensors for benchmarking instead of random tensors.            |


Possibly supported (depends on the specific device and PyTorch version) data types: `fp64`, `fp32`, `fp16`, `bf16`, `int32`, `int16`, `int8`. Specify multiple types separated by commas.

To test one specific CUDA device, use `--device cuda:x`, where `x` is the device index.

For certain devices, you need to increase the `--m`, `--n`, and `--k` to better reflect the peak performance. 
For certain devices, you may also need to increase the `--mem` to better reflect the peak memory bandwidth.

For certain devices, using all-zero tensors might show a better performance because of power throttling. The transistor switches less frequently when the tensors are all zeros, which can lead to a higher performance.

```bash
python benchmark.py --device cuda:0 --types fp64,fp32,fp16,bf16
```

