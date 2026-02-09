# profiling/benchmark.py

import time
import psutil
import torch
import numpy as np
from typing import Dict
from cpu_ml_profiling.models.resnet import get_resnet18

def benchmark_model(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 50,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """
    Benchmark CPU inference performance of a PyTorch model.
    Measures latency, throughput, CPU utilization, and memory usage.
    """

    process = psutil.Process()
    model.eval()

    # Warm-up runs (avoid cold-start effects)
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)

    latencies = []

    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            latencies.append(end - start)

    avg_latency = np.mean(latencies)
    batch_size = input_tensor.size(0)

    latency_ms_per_sample = (avg_latency / batch_size) * 1000
    throughput = batch_size / avg_latency

    cpu_usage = psutil.cpu_percent(interval=None)
    memory_usage_mb = process.memory_info().rss / (1024 ** 2)

    return {
        "latency_ms_per_sample": latency_ms_per_sample,
        "throughput_samples_per_sec": throughput,
        "cpu_usage_percent": cpu_usage,
        "memory_usage_mb": memory_usage_mb
    }


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    model = get_resnet18()
    dummy_input = torch.randn(1, 3, 224, 224)

    metrics = benchmark_model(model, dummy_input)
    print(metrics)
