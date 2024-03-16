# GPUMonitor class to monitor GPU utilization in the background

import subprocess
import shutil
import os


def gpu_util():
    format = "csv,nounits,noheader"
    queries = ["utilization.gpu", "memory.used", "memory.free", "utilization.memory"]
    gpu_query = ",".join(queries)
    gpu_ids = 0

    result = subprocess.run(
        [
            shutil.which("nvidia-smi"),
            f"--query-gpu={gpu_query}",
            f"--format={format}",
            f"--id={gpu_ids}",
        ],
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    stats = [
        [float(x) for x in s.split(", ")]
        for s in result.stdout.strip().split(os.linesep)
    ][0]
    utilization = {}
    for k, q in enumerate(queries):
        utilization[q] = stats[k]
    return utilization


from threading import Thread
from time import sleep, time
import matplotlib.pyplot as plt


class GPUMonitor:
    def __init__(self, timeout=1):
        self.timeout = timeout

        self._thread = Thread(target=self._monitor, daemon=True)
        self.utilization = None
        self.done = False
        self.start_time = self.stop_time = None

    def start(self):
        self._thread.start()
        self.start_time = time()
        return self

    def _monitor(self):
        while True:
            if self.done:
                break
            utilization = gpu_util()
            if self.utilization is None:
                self.utilization = {k: [] for k in utilization}
            for k, v in utilization.items():
                self.utilization[k].append(v)
            sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.stop_time = time()
        self.done = True

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()

    def plot_all(self, figsize=(10, 10)):
        f, axs = plt.subplots(2, 2, figsize=figsize)
        for i in range(2):
            for j in range(2):
                ax = axs[i][j]
                k = list(self.utilization.keys())[i * 2 + j]
                data = self.utilization[k]
                ax.plot(range(len(data)), data)
                ax.title.set_text(k)
        plt.suptitle("GPU monitoring")

    def plot(self, k):
        assert k in self.utilization, f"Unknown metric {k}"
        data = self.utilization[k]
        plt.plot(range(len(data)), data)
        plt.title("GPU monitoring: " + k)