import torch
import time
import math
import pandas as pd

from systems.annotated_attention import vanilla_attention

def benchmark_config(d_model, seq_len, batch_size = 8, use_pytorch_compile = False):
    torch.cuda.empty_cache()

    # Create Q, K, V
    q = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)

    # Vanilla/torch.compile for attention function
    attn_fn = torch.compile(vanilla_attention) if use_pytorch_compile else vanilla_attention

    # Dummy Gradient for backward pass
    grad_out = torch.randn(batch_size, seq_len, d_model, device='cuda')

    # Warmup Stage
    for _ in range(10):
        out = attn_fn(q, k, v)
        out.backward(grad_out)
    torch.cuda.synchronize()

    # Measure Peak Memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    out = attn_fn(q, k, v)
    mem_used_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    out = None
    torch.cuda.empty_cache()

    # 100 Forward passes
    torch.cuda.synchronize()
    start_fwd = time.perf_counter()
    for _ in range(100):
        out = attn_fn(q, k, v)
    torch.cuda.synchronize()
    fwd_time_ms = ((time.perf_counter() - start_fwd) / 100) * 1000

    # 100 Backward passes
    torch.cuda.synchronize()
    start_bwd = time.perf_counter()
    for _ in range(100):
        out = attn_fn(q, k, v)
        out.backward(grad_out)
    torch.cuda.synchronize()

    # Backward Pass duration
    total_time_ms = ((time.perf_counter() - start_bwd) / 100) * 1000
    bwd_time_ms = total_time_ms - fwd_time_ms

    return {
        'Compiled': 'Yes' if use_pytorch_compile else 'No',
        'd_model' : d_model,
        'seq_len' : seq_len,
        'Memory_in_MB' : round(mem_used_mb, 2),
        'Fwd_Time_in_msec' : round(fwd_time_ms, 2),
        'Bwd_Time_in_msec' : round(bwd_time_ms, 2),
        'Status' : 'Success'
    }

def run_benchmarks(batch_size, d_models, seq_lens, use_pytorch_compile = False):
    results = []
    for use_compile in [False, True]:
        print(f"Running Benchmarks (JIT Compiled: {use_compile})")
        for d_model in d_models:
            for seq_len in seq_lens:
                try: 
                    res = benchmark_config(d_model, seq_len, batch_size, use_compile)
                    results.append(res)
                # For OOM Crashes
                except Exception as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        results.append({
                            'Compiled': 'Yes' if use_pytorch_compile else 'No',
                            'd_model' : d_model,
                            'seq_len' : seq_len,
                            'Memory_in_MB' : None,
                            'Fwd_Time_in_msec' : None,
                            'Bwd_Time_in_msec' : None,
                            'Status' : 'OOM'
                        })

                    else: 
                        raise e
                
    return results

def display_results(results):
    clean_results = [{k: v for k, v in r.items() if k != 'Status'} for r in results]
    df = pd.DataFrame(clean_results)
    print(df)

def benchmark_vanilla_attention():
    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]

    print("Starting PyTorch Vanilla Attention Benchmark")
    print("=" * 60)

    results = run_benchmarks(batch_size, d_models, seq_lens)
    display_results(results)

if __name__ == "__main__":
    benchmark_vanilla_attention()
