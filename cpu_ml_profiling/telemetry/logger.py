import time
import psutil
from typing import List, Dict


class TelemetryLogger:
    """
    Logs CPU and memory telemetry at fixed intervals with phase labels.
    """

    def __init__(self, interval_sec: float = 0.1):
        self.interval_sec = interval_sec
        self.process = psutil.Process()
        self.records: List[Dict] = []

    def log(self, phase: str):
        timestamp = time.time()
        cpu = psutil.cpu_percent(interval=None)
        memory_mb = self.process.memory_info().rss / (1024 ** 2)

        self.records.append({
            "timestamp": timestamp,
            "phase": phase,
            "cpu_percent": cpu,
            "memory_mb": memory_mb
        })

    def run_for_duration(self, phase: str, duration_sec: float):
        start = time.time()
        while time.time() - start < duration_sec:
            self.log(phase)
            time.sleep(self.interval_sec)

    def get_records(self) -> List[Dict]:
        return self.records
