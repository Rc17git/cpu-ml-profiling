# experiments/telemetry_experiment.py

import time
import csv
import torch

from cpu_ml_profiling.telemetry.logger import TelemetryLogger
from cpu_ml_profiling.models.resnet import get_resnet18
from cpu_ml_profiling.models.lstm import get_lstm_model

# CPU thread control (critical for stable telemetry)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

OUTPUT_FILE = "results/telemetry_resnet_vs_lstm.csv"


def run_inference_with_telemetry(model, input_tensor, logger):
    model.eval()

    # Phase 1: Model initialization
    logger.run_for_duration(phase="model_init", duration_sec=0.5)

    # Phase 2: Warm-up inference
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_tensor)
            logger.log(phase="warmup")

    # Phase 3: Steady-state inference
    with torch.no_grad():
        for _ in range(30):
            _ = model(input_tensor)
            logger.log(phase="steady_inference")
            time.sleep(0.05)  # allow telemetry to capture behavior

    # Phase 4: Idle
    logger.run_for_duration(phase="idle", duration_sec=0.5)


def main():
    logger = TelemetryLogger(interval_sec=0.1)

    # ---- CNN workload ----
    resnet = get_resnet18()
    cnn_input = torch.randn(8, 3, 224, 224)

    print("Running ResNet telemetry...")
    run_inference_with_telemetry(resnet, cnn_input, logger)

    # ---- LSTM workload ----
    lstm = get_lstm_model()
    lstm_input = torch.randn(8, 100, 300)

    print("Running LSTM telemetry...")
    run_inference_with_telemetry(lstm, lstm_input, logger)

    # Save telemetry
    records = logger.get_records()
    fieldnames = records[0].keys()

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Telemetry saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
