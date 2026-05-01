import argparse
import torch
import numpy as np
from basics.basics.model import BasicsTransformerLM
import torch.cuda.nvtx as nvtx
import timeit
from datetime import datetime
import os

import contextlib

def get_arguments(): 
    parser = argparse.ArgumentParser(description='Arguments for Performance Evaluator Script')
    parser.add_argument('--d_model', type = int, default = 768)
    parser.add_argument('--d_ff', type = int, default = 3072)
    parser.add_argument('--num_heads', type = int, default = 12)
    parser.add_argument('--num_layers', type = int, default = 12)
    parser.add_argument('--vocab_size', type = int, default = 10000)
    parser.add_argument('--batch_size', type = int, default = 4)
    parser.add_argument('--context_length', type = int, default = 128)
    parser.add_argument('--w', type = int, default = 5)
    parser.add_argument('--measure_steps', type = int, default = 10)
    parser.add_argument('--backward', action="store_true")
    parser.add_argument("--optimizer", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--compile", action="store_true")

    return parser.parse_args()

def main():
    args = get_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    if args.profile: 
        try: 
            import basics.basics.model
            from systems.annotated_attention import annotated_scaled_dot_product_attention

            basics.basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
            print("Using annotated attention for profiling.")
        except ImportError:
            print(f"Could not import annotated attention: {ImportError}")

    # Model init
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10000
    ).to(device)

    # Random data
    x = torch.randint(
        0, 
        args.vocab_size, 
        (args.batch_size, args.context_length)
        ).to(device)
    
    # Backward pass loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    dummy = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Get range for nvtx
    def get_nvtx_range(name): 
        if args.profile and device.type == "cuda":
            return torch.cuda.nvtx.range(name)
        return contextlib.nullcontext()

    # For autocast cases
    def get_autocast_context():
        if args.autocast and device.type == "cuda":
            target_dtype = torch.bfloat16 if args.bf16 else torch.float16
            return torch.autocast(device_type=device.type, dtype=target_dtype)
        return contextlib.nullcontext()

    # A single pass
    def one_step(phase_name = 'measurement'): 
        if args.optimizer: 
            optimizer.zero_grad()

        # forward
        with get_nvtx_range(f"{phase_name}-forward-pass"):
            with get_autocast_context():    
                logits = model(x)

        if args.backward or args.optimizer:
            with get_nvtx_range(f"{phase_name}-loss-or-backward-pass"):
                with get_autocast_context():
                    loss = loss_fn(logits.view(-1, args.vocab_size), dummy.view(-1))
                loss.backward()

        if args.optimizer:
            with get_nvtx_range(f"{phase_name}-optimizer-step"):
                optimizer.step()

    # Warmup Phase
    with get_nvtx_range("warmup-phase"):
        for _ in range(args.w):
            one_step(phase_name="warmup")
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Memory Profiling
    if args.profile_memory and device.type == "cuda":
        torch.cuda.memory._record_memory_history(max_entries=100000)
        
    # Measuring Performance
    lap_times = []

    with get_nvtx_range("measurement-phase"):
        for _ in range(args.measure_steps):
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start_time = timeit.default_timer()
            one_step(phase_name='measurement')

            if device.type == 'cuda':
                torch.cuda.synchronize()
        
            # one lap is end - start
            lap_times.append(timeit.default_timer() - start_time)

    # End Memory Profiling
    if args.profile_memory and device.type == "cuda":
        os.makedirs("memory-profiles", exist_ok=True)
        context = args.context_length
        mode = 'train' if args.optimizer else 'inference'
        precision = 'fp16' if args.autocast else 'fp32'
        filename = f'memorydump-{mode}-{precision}-{context}-{datetime.now().strftime("%Y%m%d_%H%M%S")}.pickle'
        torch.cuda.memory._dump_snapshot(os.path.join("memory-profiles", filename))
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f'Memory Snapshot saved to {os.path.join("memory-profiles", filename)}')

    mean_time = np.mean(lap_times)
    std_time = np.std(lap_times)

    # mode = "Forward + backward" if args.backward else "Only Forward"
    if args.optimizer: 
        mode = "forward+backward+optimizer step"
    elif args.backward: 
        mode = "forward+backward"
    else: 
        mode = "only-forward"

    # Autocast Check
    if args.autocast:
        precision_str = "bf16" if args.bf16 else "fp16"
        mode += f" | (Autocast: {precision_str})"
    else:
        mode += " | (FP32)"

    print(f"Mode of Running: {mode}")
    print(f"Mean time: {mean_time:.4f} seconds")
    print(f"Std time: {std_time:.4f} seconds")


if __name__ == "__main__":
    main()

