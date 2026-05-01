import torch
import triton #type: ignore
import pandas as pd
import math
from datetime import datetime
import os

from basics.basics.model import scaled_dot_product_attention
from systems.triton_flashattn2 import TritonFlashAttention2

def run_benchmark():
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_models = [16, 32, 64, 128]
    dtypes = [torch.float32, torch.bfloat16]
    
    results = []
    
    for dtype in dtypes:
        dtype_str = "FP32" if dtype == torch.float32 else "BF16"
        for d in d_models:
            for s in seq_lens:
                print(f"Benchmarking: {dtype_str} | d={d} | seq={s}")
                
                q = torch.randn(1, s, d, device='cuda', dtype=dtype, requires_grad=True)
                k = torch.randn(1, s, d, device='cuda', dtype=dtype, requires_grad=True)
                v = torch.randn(1, s, d, device='cuda', dtype=dtype, requires_grad=True)
                grad_out = torch.randn(1, s, d, device='cuda', dtype=dtype)
                
                seq_idx = torch.arange(s, device='cuda')
                causal_mask = seq_idx[:, None] >= seq_idx[None, :]
                causal_mask = causal_mask.unsqueeze(0) 

                try:
                    torch.cuda.empty_cache()
                    fwd_fn = lambda: scaled_dot_product_attention(q, k, v, mask=causal_mask)
                    vanilla_fwd = triton.testing.do_bench(fwd_fn)
                    
                    out_v = fwd_fn()
                    
                    bwd_fn = lambda: out_v.backward(grad_out, retain_graph=True)
                    vanilla_bwd = triton.testing.do_bench(bwd_fn)
                    
                    def e2e_v():
                        o = scaled_dot_product_attention(q, k, v, mask=causal_mask)
                        o.backward(grad_out)
                    vanilla_e2e = triton.testing.do_bench(e2e_v)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        vanilla_fwd, vanilla_bwd, vanilla_e2e = "OOM", "OOM", "OOM"
                        torch.cuda.empty_cache()
                    else:
                        raise e
                    
                try:
                    torch.cuda.empty_cache()
                    fwd_fn_t = lambda: TritonFlashAttention2.apply(q, k, v, True)
                    triton_fwd = triton.testing.do_bench(fwd_fn_t)
                    
                    out_t = fwd_fn_t()
                    
                    bwd_fn_t = lambda: out_t.backward(grad_out, retain_graph=True)
                    triton_bwd = triton.testing.do_bench(bwd_fn_t)
                    
                    def e2e_t():
                        o = TritonFlashAttention2.apply(q, k, v, True)
                        o.backward(grad_out)
                    triton_e2e = triton.testing.do_bench(e2e_t)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        triton_fwd, triton_bwd, triton_e2e = "OOM", "OOM", "OOM"
                        torch.cuda.empty_cache()
                    else:
                        raise e
                        
                results.append({
                    "DType": dtype_str,
                    "d_model": d,
                    "seq_len": s,
                    "Vanilla-Fwd-in-msec": round(vanilla_fwd, 2) if isinstance(vanilla_fwd, float) else vanilla_fwd,
                    "Triton-Fwd-in-msec": round(triton_fwd, 2) if isinstance(triton_fwd, float) else triton_fwd,
                    "Vanilla-Bwd-in-msec": round(vanilla_bwd, 2) if isinstance(vanilla_bwd, float) else vanilla_bwd,
                    "Triton-Bwd-in-msec": round(triton_bwd, 2) if isinstance(triton_bwd, float) else triton_bwd,
                    "Vanilla-E2E-in-msec": round(vanilla_e2e, 2) if isinstance(vanilla_e2e, float) else vanilla_e2e,
                    "Triton-E2E-in-msec": round(triton_e2e, 2) if isinstance(triton_e2e, float) else triton_e2e,
                })

    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("| Flash Attention 2 Benchmark |")
    print("="*80)
    print(df)
    benchmark_dir = 'benchmark-dir'
    os.makedirs(benchmark_dir, exist_ok=True)
    df.to_csv(os.path.join(benchmark_dir, f"flash_attention_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)

if __name__ == "__main__":
    run_benchmark()