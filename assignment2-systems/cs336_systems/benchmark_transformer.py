import argparse
import timeit
import torch
from cs336_basics.model import BasicsTransformerLM


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Transformer forward/backward passes."
    )
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--context_length", type=int, default=32)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--timing_steps", type=int, default=50)
    parser.add_argument(
        "--mode", choices=["forward", "forward_backward"], default="forward"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    # Model initialization
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)
    model.train()

    # Random batch generation
    x = torch.randint(
        0, args.vocab_size, (args.batch_size, args.context_length), device=args.device
    )
    y = torch.randint(
        0, args.vocab_size, (args.batch_size, args.context_length), device=args.device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    def step_forward():
        with torch.no_grad():
            _ = model(x)

    def step_forward_backward():
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.view(-1, args.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

    # Warm-up
    for _ in range(args.warmup_steps):
        if args.mode == "forward":
            step_forward()
        else:
            step_forward_backward()

    # Timing
    times = []
    for _ in range(args.timing_steps):
        start = timeit.default_timer()
        if args.mode == "forward":
            step_forward()
        else:
            step_forward_backward()

        torch.cuda.synchronize()  # Ensure all operations are complete before measuring time
        end = timeit.default_timer()
        times.append((end - start))

    avg_time = sum(times) / len(times)
    print(f"Average time per step ({args.mode}): {avg_time:.6f} seconds")


if __name__ == "__main__":
    main()
