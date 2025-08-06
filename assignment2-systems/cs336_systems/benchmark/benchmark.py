import timeit
import torch
import pandas as pd
from statistics import mean, stdev
from cs336_basics.model import BasicsTransformerLM
import argparse

# Define the model sizes (from the table)
model_configs = [
    {"size": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    {
        "size": "medium",
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    {"size": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    {"size": "xl", "d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    {"size": "2.7B", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
]

# Constants
vocab_size = 10_000
context_length = 256
batch_size = 8
rope_theta = 10000.0
warmup_steps = 5
timing_steps = 10
device = "cuda" if torch.cuda.is_available() else "cpu"


def benchmark(model, x, y, mode):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    def step_forward():
        with torch.no_grad():
            _ = model(x)

    def step_forward_backward():
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

    step_fn = step_forward if mode == "forward" else step_forward_backward

    # Warm-up
    for _ in range(warmup_steps):
        step_fn()

    # Timed steps
    times = []
    for _ in range(timing_steps):
        start = timeit.default_timer()
        step_fn()
        torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end - start)

    return mean(times), stdev(times)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Transformer models.")
    parser.add_argument("--d_model", type=int, help="Model dimension")
    parser.add_argument("--d_ff", type=int, help="Feedforward dimension")
    parser.add_argument("--num_layers", type=int, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, help="Number of attention heads")
    parser.add_argument("--all", action="store_true", help="Run all predefined configurations")
    parser.add_argument("--context_length", type=int, help="Sequence context length")
    parser.add_argument("--warmup_steps", type=int, help="Number of warmup steps")
    return parser.parse_args()


def main():
    args = parse_args()
    global context_length
    if args.context_length:
        context_length = args.context_length
    global warmup_steps
    if args.warmup_steps:
        warmup_steps = args.warmup_steps
    results = []

    configs_to_run = []

    if args.all:
        configs_to_run = model_configs
    elif args.d_model and args.d_ff and args.num_layers and args.num_heads:
        configs_to_run = [{
            "size": "custom",
            "d_model": args.d_model,
            "d_ff": args.d_ff,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
        }]
    else:
        raise ValueError("Must specify either --all or all custom model hyperparameters.")
    
    print("\nRunning the following configurations:")
    for cfg in configs_to_run:
        print(f"  - {cfg}")
    print()

    for config in configs_to_run:
        for mode in ["forward", "forward_backward"]:
            print(f"Running {config['size']} model [{mode}]...")

            model = BasicsTransformerLM(
                vocab_size=vocab_size,
                context_length=context_length,
                d_model=config["d_model"],
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
                d_ff=config["d_ff"],
                rope_theta=rope_theta,
            ).to(device)

            # Create random input data  
            x = torch.randint(
                0, vocab_size, (batch_size, context_length), device=device
            )
            y = torch.randint(
                0, vocab_size, (batch_size, context_length), device=device
            )

            avg, std = benchmark(model, x, y, mode)
            
            print(f"  - {config['size']} [{mode}]: Avg Time = {avg:.6f}s, Std Dev = {std:.6f}s")
            del model, x, y 
            torch.cuda.empty_cache()
            

            results.append(
                {
                    "Size": config["size"],
                    "Mode": mode,
                    "d_model": config["d_model"],
                    "d_ff": config["d_ff"],
                    "num_layers": config["num_layers"],
                    "num_heads": config["num_heads"],
                    "Context Length": context_length,
                    "Avg Time (s)": round(avg, 6),
                    "Std Dev (s)": round(std, 6),
                    "Warmup Steps": warmup_steps,
                }
            )


    # Output results
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))

    # Save to file
    with open("benchmark_results.md", "w") as f:
        f.write(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
