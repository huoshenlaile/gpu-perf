import torch
try:
    import torch_npu
except ImportError:
    pass

import time
import argparse

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    elif hasattr(torch, 'npu'):
        return torch.device('npu')
    else:
        return torch.device('cpu')

def get_device_name(device):
    if device.type == 'cuda':
        return f'({torch.cuda.get_device_name(device)})'
    elif device.type == 'npu':
        return f'({torch.npu.get_device_name(device)})'
    else:
        return ''

def cleanup_device(device):
    """
    Cleans up the device by emptying the cache.
    """
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'npu':
        torch.npu.empty_cache()
    else:
        pass  # No cleanup needed for CPU

def synchronize_device(device):
    """
    Synchronizes the device to ensure all operations are complete.
    """
    if device.type == 'cuda':
        torch.cuda.synchronize(device=device)
    elif device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'npu':
        torch.npu.synchronize(device=device)
    else:
        pass  # No synchronization needed for CPU

def benchmark_matmul(device, dtype, sec=1.0, M = 16384, N = 16384, K = 16384, all_zeros=False):
    """
    Benchmarks matrix multiplication performance in TFLOPS.
    """
    
    cleanup_device(device)
    if all_zeros:
        A = torch.zeros((M, K), dtype=dtype, device=device)
        B = torch.zeros((K, N), dtype=dtype, device=device)
    else:
        A = torch.randn((M, K), dtype=dtype, device=device)
        B = torch.randn((K, N), dtype=dtype, device=device)
    C = torch.empty((M, N), dtype=dtype, device=device)
    
    iterations = 0
    elapsed_time = 0
    
    warmup_time = 1.0
    
    # Warmup device, and estimate iterations
    start_time = time.time()
    while elapsed_time < warmup_time:
        synchronize_device(device)
        torch.matmul(A, B, out=C)
        synchronize_device(device)
        iterations += 1
        elapsed_time = time.time() - start_time
    
    required_iterations = int(sec / elapsed_time * iterations) + 1
    # print(f"Warmup completed: {iterations} iterations in {elapsed_time:.2f} seconds")
    # print(f"Estimated iterations for {sec} seconds: {required_iterations}")

    synchronize_device(device)
    start_time = time.time()
    for _i in range(required_iterations):
        torch.matmul(A, B, out=C)
    synchronize_device(device)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    flops_per_iteration = 2 * (M * K * N)
    total_flops = flops_per_iteration * required_iterations
    tflops = total_flops / (elapsed_time * 1e12)
    cleanup_device(device)
    return tflops

def benchmark_memory_bandwidth(device, dtype, sec=1.0, sz=1e9, all_zeros=False):
    """
    Estimates memory bandwidth in GB/s by measuring the time taken to copy large tensors.
    """
    # Calculate the number of elements required to reach the target size
    cleanup_device(device)
    num_elements = int(sz / torch.tensor([], dtype=dtype).element_size())
    if all_zeros:
        A = torch.zeros(num_elements, dtype=dtype, device=device)
    else:
        A = torch.randn(num_elements, dtype=dtype, device=device)
    B = torch.empty_like(A)
    
    iterations = 0
    elapsed_time = 0
    
    warmup_time = 1.0
    
    # Warmup device, and estimate iterations
    start_time = time.time()
    while elapsed_time < warmup_time:
        synchronize_device(device)
        B.copy_(A, non_blocking=True)
        synchronize_device(device)
        iterations += 1
        elapsed_time = time.time() - start_time

    required_iterations = int(sec / elapsed_time * iterations) + 1
    # print(f"Warmup completed: {iterations} iterations in {elapsed_time:.2f} seconds")
    # print(f"Estimated iterations for {sec} seconds: {required_iterations}")
    
    synchronize_device(device)
    start_time = time.time()
    for _ in range(required_iterations):
        B.copy_(A, non_blocking=True)
    synchronize_device(device)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    bytes_per_copy = A.numel() * A.element_size() * 2  # Read and write
    total_bytes = bytes_per_copy * required_iterations
    bandwidth_gib_s = total_bytes / (elapsed_time * 1073741824)
    bandwidth_gb_s = total_bytes / (elapsed_time * 1e9)  # Convert to GB/s
    cleanup_device(device)
    return bandwidth_gib_s, bandwidth_gb_s


def parse_types(value:str):
    # Split the input string by ',' and return as a list
    return value.split(',')

def main():
    
    torch.backends.cuda.matmul.allow_tf32 = False
    # Define data types to test
    supported_data_types = {
        'fp64': torch.float64,
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'int32': torch.int32,
        'int16': torch.int16,
        'int8': torch.int8,
    }
    
    # parse arguments
    parser = argparse.ArgumentParser(description='Benchmarking script for matrix multiplication and memory bandwidth')
    parser.add_argument('--device', type=str, default='auto', help='Device to benchmark (auto, cpu, cuda, mps), default: auto')
    parser.add_argument('--types', type=parse_types, default=['fp32','fp16'], help='Comma-separated list of data types to benchmark (e.g., \'fp64,fp32,fp16,bf16\'), default: fp32,fp16')
    parser.add_argument('--sec', type=float, default=1.5, help='Time in seconds to run the benchmark, default: 1.5')
    parser.add_argument('--m', type=int, default=16384, help='Matrix size for compute benchmark, default: 16384')
    parser.add_argument('--n', type=int, default=16384, help='Matrix size for compute benchmark, default: 16384')
    parser.add_argument('--k', type=int, default=16384, help='Matrix size for compute benchmark, default: 16384')
    parser.add_argument('--mem', type=float, default=1073741824, help='Tensor size for memory benchmark, default: 1073741824 (1 GiB)')
    parser.add_argument('--tf32', action='store_true', help='Enable TensorFloat-32 (TF32) on supported hardware, default: False')
    parser.add_argument('--all-zeros', action='store_true', help='Use all-zero tensors for benchmarking, default: False')

    args = parser.parse_args()
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device.lower())
    

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        print("TensorFloat-32 (TF32) enabled")

    data_types = {name: supported_data_types[name] for name in args.types}

    device_name = get_device_name(device)
    print(f"Using backend: {device} {device_name}\n")

    for name, dtype in data_types.items():
        print(f"--- {name} Benchmark ---")
        try:
            # Compute Benchmark
            tflops = benchmark_matmul(device, dtype, sec=args.sec, M=args.m, N=args.n, K=args.k, all_zeros=args.all_zeros)
            print(f"Matrix Multiplication Performance:            {tflops:.2f} TFLOPS")
        except RuntimeError as e:
            tflops = -1
            print(f"Matrix Multiplication failed on {device}: {e}")
        
        try:
            # Memory Bandwidth Benchmark
            bandwidth_gib_s, bandwidth_gb_s = benchmark_memory_bandwidth(device, dtype, sec=args.sec, sz=args.mem, all_zeros=args.all_zeros)
            print(f"Memory Bandwidth:                             {bandwidth_gib_s:.2f} GiB/s ({bandwidth_gb_s:.2f} GB/s)")
        except RuntimeError as e:
            bandwidth_gib_s = -1
            print(f"Memory Bandwidth test failed on {device}: {e}\n")

        if (tflops != -1) and (bandwidth_gib_s != -1):
            ridge_point = tflops * 1024 / bandwidth_gib_s
            print(f"Roofline Ridge Point (Arithmetic Intensity):  {ridge_point:.2f} FLOPS/Byte\n")

if __name__ == "__main__":
    main()
