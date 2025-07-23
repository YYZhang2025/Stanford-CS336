import timeit
import torch
import pandas as pd
from statistics import mean, stdev
from cs336_basics.model import BasicsTransformerLM

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
vocab_size = 10000
context_length = 32
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


def main():
    results = []

    for config in model_configs:
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

            x = torch.randint(
                0, vocab_size, (batch_size, context_length), device=device
            )
            y = torch.randint(
                0, vocab_size, (batch_size, context_length), device=device
            )

            avg, std = benchmark(model, x, y, mode)

            results.append(
                {
                    "Size": config["size"],
                    "Mode": mode,
                    "Avg Time (s)": round(avg, 6),
                    "Std Dev (s)": round(std, 6),
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
