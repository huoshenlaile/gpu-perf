import torch
import time
import argparse

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda'), 'CUDA'
    elif hasattr(torch, 'has_mps') and torch.has_mps:
        return torch.device('mps'), 'Apple MPS'
    else:
        return torch.device('cpu'), 'CPU'

def benchmark_matmul(device, dtype, M=4096, N=4096, K=4096, iterations=10):
    """
    Benchmarks matrix multiplication performance in TFLOPS.
    """
    # Initialize random matrices
    A = torch.randn((M, K), dtype=dtype, device=device)
    B = torch.randn((K, N), dtype=dtype, device=device)
    
    # Warm-up
    for _ in range(5):
        C = torch.matmul(A, B)
    
    # Synchronize before starting timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        # Move to CPU to ensure MPS operations are complete
        C.cpu()
    else:
        pass  # CPU operations are synchronous
    
    start_time = time.time()
    
    for _ in range(iterations):
        C = torch.matmul(A, B)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        # Move to CPU to ensure MPS operations are complete
        C.cpu()
    else:
        pass  # CPU operations are synchronous
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Number of floating point operations: 2 * N^3 per matmul
    flops_per_iteration = 2 * (N ** 3)
    total_flops = flops_per_iteration * iterations
    tflops = total_flops / (total_time * 1e12)
    return tflops

def benchmark_memory_bandwidth(device, dtype, N=4096, iterations=100):
    """
    Estimates memory bandwidth in GB/s by measuring the time taken to copy large tensors.
    """
    # Initialize large tensors
    A = torch.randn((N, N), dtype=dtype, device=device)
    B = torch.empty_like(A)
    
    # Warm-up
    for _ in range(10):
        B.copy_(A)
    
    # Synchronize before starting timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        # Move to CPU to ensure MPS operations are complete
        B.cpu()
    else:
        pass  # CPU operations are synchronous
    
    start_time = time.time()
    
    for _ in range(iterations):
        B.copy_(A)
    
    # Synchronize after operations
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        # Move to CPU to ensure MPS operations are complete
        B.cpu()
    else:
        pass  # CPU operations are synchronous
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate total bytes moved: 2 * N * N elements * element size (read and write)
    bytes_per_copy = A.numel() * A.element_size() * 2
    total_bytes = bytes_per_copy * iterations
    bandwidth_gb_s = total_bytes / (total_time * 1e9)
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
        'int8': torch.int8,
    }
    
    # parse arguments
    parser = argparse.ArgumentParser(description='Benchmarking script for matrix multiplication and memory bandwidth')
    parser.add_argument('--device', type=str, default='auto', help='Device to benchmark (cpu, cuda, mps)', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--matmul-size', type=int, default=4096, help='Matrix size for compute benchmark')
    parser.add_argument('--memory-size', type=int, default=8192, help='Tensor size for memory benchmark')
    parser.add_argument('--iterations-matmul', type=int, default=10, help='Number of iterations for compute benchmark')
    parser.add_argument('--iterations-memory', type=int, default=1000, help='Number of iterations for memory benchmark')
    parser.add_argument('--tf32', action='store_true', help='Enable TensorFloat-32 (TF32) on supported hardware')
    parser.add_argument('--types', type=parse_types, default=['fp32', 'fp16'], help="Comma-separated list of data types to benchmark (e.g., 'fp64,fp32,fp16')")


    args = parser.parse_args()
    if args.device == 'auto':
        device, device_name = get_device()
    elif args.device == 'cuda':
        device = torch.device('cuda')
        device_name = 'CUDA'
    elif args.device == 'mps':
        device = torch.device('mps')
        device_name = 'Apple MPS'
    elif args.device == 'cpu':
        device = torch.device('cpu')
        device_name = 'CPU'
    else:
        raise ValueError(f"Unknown device: {args.device}")
    
    N_matmul = args.matmul_size
    N_memory = args.memory_size
    iterations_matmul = args.iterations_matmul
    iterations_memory = args.iterations_memory

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        print("TensorFloat-32 (TF32) enabled")

    data_types = {name: supported_data_types[name] for name in args.types}

    print(f"Using device: {device_name} ({device})\n")
    for name, dtype in data_types.items():
        print(f"--- {name} Benchmark ---")
        try:
            # Compute Benchmark
            tflops = benchmark_matmul(device, dtype, M=N_matmul, N=N_matmul, K=N_matmul, iterations=iterations_matmul)
            print(f"Matrix Multiplication Performance: {tflops:.2f} TFLOPS")
        except RuntimeError as e:
            print(f"Matrix Multiplication failed on {device_name}: {e}")
        
        try:
            # Memory Bandwidth Benchmark
            bandwidth = benchmark_memory_bandwidth(device, dtype, N=N_memory, iterations=iterations_memory)
            print(f"Memory Bandwidth: {bandwidth:.2f} GB/s\n")
        except RuntimeError as e:
            print(f"Memory Bandwidth test failed on {device_name}: {e}\n")

if __name__ == "__main__":
    main()
