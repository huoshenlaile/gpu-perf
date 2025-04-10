import torch
import time
import argparse

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_device_name(device):
    if device.type == 'cuda':
        return torch.cuda.get_device_name(device)
    elif device.type == 'mps':
        return 'MPS'
    else:
        return 'CPU'

def benchmark_matmul(device, dtype, sec=1.0, M = 16384, N = 16384, K = 16384):
    """
    Benchmarks matrix multiplication performance in TFLOPS.
    """
    
    
    A = torch.randn((M, K), dtype=dtype, device=device)
    B = torch.randn((K, N), dtype=dtype, device=device)
    C = torch.empty((M, N), dtype=dtype, device=device)
    
    iterations = 0.5
    elpased_time = 0
    
    while elpased_time < sec:
        
        iterations = int(iterations * 2)
            
        # Synchronize
        if device.type == 'cuda':
            torch.cuda.synchronize(device=device)
        elif device.type == 'mps':
            torch.mps.synchronize()
        else:
            pass
        
        start_time = time.time()
        
        for _ in range(iterations):
            torch.matmul(A, B, out=C)

        # Synchronize
        if device.type == 'cuda':
            torch.cuda.synchronize(device=device)
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.synchronize()
        else:
            pass
        
        end_time = time.time()
        elpased_time = end_time - start_time

        
    flops_per_iteration = 2 * (M * K * N)
    total_flops = flops_per_iteration * iterations
    tflops = total_flops / (elpased_time * 1e12)
    return tflops

def benchmark_memory_bandwidth(device, dtype, sec=1.0, sz = 1e9):
    """
    Estimates memory bandwidth in GB/s by measuring the time taken to copy large tensors.
    """
    # Initialize large tensors
    # calculate the number of elements reqired to reach the target size
    sz = int(sz)
    A = torch.randn(sz, dtype=dtype, device=device)
    B = torch.empty_like(A)
    
    iterations = 32
    elpased_time = 0
    
    while elpased_time < sec:
        
        iterations = int(iterations * 2)
    
        # Synchronize 
        if device.type == 'cuda':
            torch.cuda.synchronize(device=device)
        elif device.type == 'mps':
            torch.mps.synchronize()
        else:
            pass
        
        start_time = time.time()
        
        for _ in range(iterations):
            B.copy_(A)
        
        # Synchronize
        if device.type == 'cuda':
            torch.cuda.synchronize(device=device)
        elif device.type == 'mps':
            torch.mps.synchronize()
        else:
            pass
        
        end_time = time.time()
        elpased_time = end_time - start_time
    
    # Calculate total bytes moved: 2 * N * N elements * element size (read and write)
    bytes_per_copy = A.numel() * A.element_size() * 2
    total_bytes = bytes_per_copy * iterations
    bandwidth_gb_s = total_bytes / (elpased_time * 1e9)
    return bandwidth_gb_s


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
    parser.add_argument('--sec', type=float, default=0.5, help='Time in seconds to run the benchmark, default: 0.5')
    parser.add_argument('--m', type=int, default=16384, help='Matrix size for compute benchmark, default: 16384')
    parser.add_argument('--n', type=int, default=16384, help='Matrix size for compute benchmark, default: 16384')
    parser.add_argument('--k', type=int, default=16384, help='Matrix size for compute benchmark, default: 16384')
    parser.add_argument('--mem', type=float, default=2e9, help='Tensor size for memory benchmark, default: 1e9')
    parser.add_argument('--tf32', action='store_true', help='Enable TensorFloat-32 (TF32) on supported hardware, default: False')
    
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
    print(f"Using backend: {device} ({device_name})\n")

    for name, dtype in data_types.items():
        print(f"--- {name} Benchmark ---")
        try:
            # Compute Benchmark
            tflops = benchmark_matmul(device, dtype, sec=args.sec, M=args.m, N=args.m, K=args.k)
            print(f"Matrix Multiplication Performance: {tflops:.2f} TFLOPS")
        except RuntimeError as e:
            tflops = -1
            print(f"Matrix Multiplication failed on {device}: {e}")
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()
        else:
            pass
        
        try:
            # Memory Bandwidth Benchmark
            bandwidth = benchmark_memory_bandwidth(device, dtype, sec=args.sec, sz=args.mem)
            print(f"Memory Bandwidth: {bandwidth:.2f} GB/s")
        except RuntimeError as e:
            bandwidth = -1
            print(f"Memory Bandwidth test failed on {device}: {e}\n")

        if (tflops != -1) and (bandwidth != -1):
            ridge_point = tflops * 1024 / bandwidth
            print(f"Roofline Ridge Point (Arithmetic Intensity): {ridge_point:.2f} FLOPS/Byte\n")

if __name__ == "__main__":
    main()
