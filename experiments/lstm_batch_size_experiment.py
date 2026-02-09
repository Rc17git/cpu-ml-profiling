# experiments/lstm_batch_size_experiment.py

import csv
import torch

from cpu_ml_profiling.models.lstm import get_lstm_model
from cpu_ml_profiling.profiling.benchmark import benchmark_model

# CPU thread control
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


BATCH_SIZES = [1, 8, 16, 32]
SEQ_LEN = 100
INPUT_DIM = 300
OUTPUT_FILE = "results/lstm_batch_size_results.csv"


def run_experiment():
    model = get_lstm_model()
    results = []

    for batch_size in BATCH_SIZES:
        print(f"Running LSTM batch size: {batch_size}")

        dummy_input = torch.randn(batch_size, SEQ_LEN, INPUT_DIM)

        metrics = benchmark_model(
            model=model,
            input_tensor=dummy_input,
            num_runs=20,
            warmup_runs=5
        )

        metrics["batch_size"] = batch_size
        results.append(metrics)

    return results


def save_to_csv(results):
    fieldnames = results[0].keys()

    with open(OUTPUT_FILE, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    results = run_experiment()
    save_to_csv(results)
    print(f"Results saved to {OUTPUT_FILE}")
