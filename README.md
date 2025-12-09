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

## Sample data

I collected some typical GPU performance data using this script. The results may vary depending on the specific hardware and PyTorch version.

The following table summarizes the performance of various GPUs and devices, including their throughput, memory bandwidth, and roofline ridge point (arithmetic intensity).

| Chip                        | fp64 (tflops) | fp32 (tflops) | fp16 (tflops) | bf16 (tflops) | bandwidth (GB/s) | bandwidth (GiB/s) | fp64 (ops/byte) | fp32 (ops/byte) | fp16 (ops/byte) | bf16 (ops/byte) |
| --------------------------- | ------------- | ------------- | ------------- | ------------- | ---------------- | ----------------- | --------------- | --------------- | --------------- | --------------- |
| RTX 3080 Ti                 | 0.54          | 24.51         | 75.06         | 75.92         | 790.35           | 736.07            | 0.68            | 31.01           | 94.97           | 96.06           |
| RTX 3090                    | 0.56          | 23.09         | 68.68         | 69.63         | 839.95           | 782.26            | 0.67            | 27.49           | 81.77           | 82.90           |
| RTX 4070 Laptop             | 0.31          | 10.06         | 40.06         | 39.54         | 227.43           | 211.81            | 1.36            | 44.23           | 176.14          | 173.86          |
| RTX 4090                    | 1.25          | 57.60         | 176.66        | 176.71        | 920.27           | 857.07            | 1.36            | 62.59           | 191.97          | 192.02          |
| RTX 5070                    | 0.49          | 23.22         | 69.40         | 69.37         | 574.36           | 534.91            | 0.85            | 40.43           | 120.83          | 120.78          |
| RTX 5090                    | 1.74          | 66.40         | 237.74        | 240.31        | 1529.09          | 1424.08           | 1.14            | 43.42           | 155.48          | 157.16          |
| RTX A6000                   | 0.56          | 22.70         | 114.86        | 117.39        | 716.17           | 666.99            | 0.78            | 31.70           | 160.38          | 163.91          |
| RTX Pro 6000 Server Edition | 1.54          | 79.26         | 378.77        | 391.11        | 1462.07          | 1361.66           | 1.05            | 54.21           | 259.06          | 267.50          |
| GB10 (DGX Spark)            | 0.41          | 19.27         | 63.09         | 63.41         | 244.95           | 228.13            | 1.67            | 78.67           | 257.56          | 258.87          |
| V100 SXM2 16GB              | 7.23          | 14.07         | 93.70         | 0.00          | 824.05           | 767.46            | 8.77            | 17.07           | 113.71          | 0.00            |
| A100 SXM                    | 15.11         | 16.50         | 260.44        | 266.37        | 1380.51          | 1285.70           | 10.95           | 11.95           | 188.65          | 192.95          |
| H20                         | 0.24          | 24.80         | 142.50        | 142.55        | 3350.56          | 3120.45           | 0.07            | 7.40            | 42.53           | 42.55           |
| H100 PCIe                   | 31.02         | 34.03         | 348.27        | 385.59        | 1703.31          | 1586.33           | 18.21           | 19.98           | 204.47          | 226.38          |
| H100 SXM                    | 49.47         | 52.95         | 656.26        | 685.49        | 3055.17          | 2845.35           | 16.19           | 17.33           | 214.80          | 224.37          |
| B200                        | 36.16         | 67.23         | 1400.88       | 1479.68       | 6542.81          | 6093.47           | 5.53            | 10.28           | 214.11          | 226.15          |
| Apple M1 8 cores            | 0.00          | 1.57          | 2.26          | 1.10          | 61.00            | 56.81             | 0.00            | 25.74           | 37.05           | 18.03           |
| Apple M2 Max 30 cores       | 0.00          | 9.11          | 9.83          | 4.72          | 377.47           | 351.55            | 0.00            | 24.13           | 26.04           | 12.50           |
| Apple M3 10 cores           | 0.00          | 2.93          | 3.23          | 3.23          | 87.40            | 81.40             | 0.00            | 33.52           | 36.96           | 36.96           |
| Huawei Ascend 910B2         | 0.00          | 72.84         | 230.28        | 257.62        | 1274.44          | 1186.91           | 0.00            | 57.15           | 180.69          | 202.14          |
| Moore Threads MTTS4000      | 0.00          | 13.79         | 72.71         | 76.04         | 694.49           | 646.79            | 0.00            | 19.86           | 104.70          | 109.49          |
