#!/bin/bash
# Run benchmark for all model sizes and save output to benchmark_results.txt

PYTHON_SCRIPT="cs336_systems/benchmark_transformer.py"
LOGFILE="benchmark_results.txt"

# Remove old log file
rm -f "$LOGFILE"

# Run the benchmark and save output
python "$PYTHON_SCRIPT" | tee "$LOGFILE"

echo "Benchmark results saved to $LOGFILE"
