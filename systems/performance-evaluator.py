import argparse
import torch
import numpy as np
from basics.basics.model import BasicsTransformerLM
import timeit

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

    return parser.parse_args()

def main():
    args = get_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
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

    def one_step(): 
        # forward
        logits = model(x)

        if args.backward:
            loss = loss_fn(logits.view(-1, args.vocab_size), dummy.view(-1))
            loss.backward()

    # Run one step
    for _ in range(args.w):
        one_step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
    # Measuring Performance
    lap_times = []

    for _ in range(args.measure_steps):
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = timeit.default_timer()
        one_step()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # one lap is end - start
        lap_times.append(timeit.default_timer() - start_time)

    mean_time = np.mean(lap_times)
    std_time = np.std(lap_times)

    mode = "Forward + backward" if args.backward else "Only Forward"
    print(f"Mode of Running: {mode}")
    print(f"Mean time: {mean_time} seconds")
    print(f"Std time: {std_time} seconds")


if __name__ == "__main__":
    main()

