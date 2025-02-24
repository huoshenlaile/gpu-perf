# PyTorch GEMM TFLOPS and Memory Bandwidth Benchmark

This Python script uses PyTorch to measure the performance of General Matrix Multiply (GEMM) operations in TFLOPS and memory bandwidth in GB/s on a given hardware device. It supports CPU, CUDA-enabled GPUs, and Apple MPS. This script will **not** likely reflect the peak performance of the hardware, but it can be useful to reflect the actual performance in PyTorch machine learning applications.

## Features

- Benchmarks GEMM (matrix multiplication) performance in TFLOPS.
- Estimates memory bandwidth in GB/s.
- Supports multiple data types: FP64, FP32, FP16, BFLOAT16, INT32, INT16, and INT8. (depends on the specific device and PyTorch version)
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
Using backend: cuda (NVIDIA GeForce RTX 5090)

--- fp64 Benchmark ---
Matrix Multiplication Performance: 1.72 TFLOPS
Memory Bandwidth: 1520.62 GB/s
Roofline Ridge Point (Arithmetic Intensity): 1.16 FLOPS/Byte

--- fp32 Benchmark ---
Matrix Multiplication Performance: 63.06 TFLOPS
Memory Bandwidth: 1527.49 GB/s
Roofline Ridge Point (Arithmetic Intensity): 42.27 FLOPS/Byte

--- fp16 Benchmark ---
Matrix Multiplication Performance: 237.53 TFLOPS
Memory Bandwidth: 1529.51 GB/s
Roofline Ridge Point (Arithmetic Intensity): 159.03 FLOPS/Byte

--- bf16 Benchmark ---
Matrix Multiplication Performance: 239.05 TFLOPS
Memory Bandwidth: 1529.61 GB/s
Roofline Ridge Point (Arithmetic Intensity): 160.03 FLOPS/Byte
```


### Advanced Command-Line Arguments

The script accepts the following arguments:

| Argument             | Default Value | Description                                                  |
|----------------------|---------------|--------------------------------------------------------------|
| `--device`           | `auto`        | Device to benchmark: `auto`, `cpu`, `cuda`, or `mps`.        |
| `--types`            | `fp32,fp16`   | Data type for GEMM benchmark.                                |
| `--matmul-size`      | `4096`        | Size of matrices for GEMM benchmark.                         |
| `--memory-size`      | `8192`        | Size of tensors for memory bandwidth benchmark.              |
| `--iterations-matmul`| `100`          | Number of iterations for GEMM benchmark.                     |
| `--iterations-memory`| `1000`        | Number of iterations for memory bandwidth benchmark.         |
| `--tf32`             | `False`       | Enable TensorFloat-32 (TF32) for supported CUDA GPUs.        |


Possibly supported (depends on the specific device and PyTorch version) data types: `fp64`, `fp32`, `fp16`, `bf16`, `int32`, `int16`, `int8`. Specify multiple types separated by commas.

To test one specific CUDA device, using `--device cuda:x`, where `x` is the device index.

```bash
python benchmark.py --device cuda:0 --types fp64,fp32,fp16,bf16
```
Notice for certain devices, you need to increase the `--matmul-size` to see the peak performance. For example, `--matmul-size 8192` is recommended for NVIDIA RTX 5090.
