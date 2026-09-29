"""
================================================================================
EXPERIMENT 5 — PROCESS-LEVEL & MODEL PARAMETER RESOURCE MONITORING
================================================================================
Monitors process-level CPU, wall-clock, memory (RSS), and model parameter memory:
  - High-resolution timing (perf_counter, process_time)
  - Background psutil polling for peak and average RAM RSS
  - Model parameter memory calculation (serialized state & tree node footprint)
================================================================================
"""

import os
import time
import pickle
import threading
import psutil
import numpy as np


class ResourceMonitor:
    """
    Measures CPU and memory utilization during code execution.
    Combines high-resolution timer measurements with psutil process metrics.
    """

    def __init__(self, sample_interval=0.005):
        self.interval = sample_interval
        self.process = psutil.Process()
        self._cpu_samples = []
        self._memory_samples = []
        self._running = False
        self._thread = None

        self._start_wall = 0.0
        self._start_cpu = 0.0
        self._start_proc_cpu = None
        self._start_ram = 0.0

    def _sample_loop(self):
        while self._running:
            try:
                cpu_pct = self.process.cpu_percent(interval=None)
                if cpu_pct > 0.0:
                    self._cpu_samples.append(cpu_pct)
                mem = self.process.memory_info().rss / (1024 ** 2)
                self._memory_samples.append(mem)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(self.interval)

    def start(self):
        """Begin monitoring."""
        self._cpu_samples = []
        self._memory_samples = []
        self._running = True

        self.process.cpu_percent(interval=None)
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()
        self._start_proc_cpu = self.process.cpu_times()
        try:
            self._start_ram = self.process.memory_info().rss / (1024 ** 2)
            self._memory_samples.append(self._start_ram)
        except Exception:
            self._start_ram = 0.0

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring and return detailed resource metrics."""
        end_wall = time.perf_counter()
        end_cpu = time.process_time()
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=0.5)

        try:
            end_proc_cpu = self.process.cpu_times()
            end_ram = self.process.memory_info().rss / (1024 ** 2)
            self._memory_samples.append(end_ram)
            final_cpu = self.process.cpu_percent(interval=None)
            if final_cpu > 0.0:
                self._cpu_samples.append(final_cpu)
        except Exception:
            end_proc_cpu = self._start_proc_cpu
            end_ram = self._start_ram

        wall_time = max(end_wall - self._start_wall, 1e-6)
        process_cpu_time = max(end_cpu - self._start_cpu, 0.0)

        if self._start_proc_cpu is not None and end_proc_cpu is not None:
            cpu_user = max(end_proc_cpu.user - self._start_proc_cpu.user, 0.0)
            cpu_system = max(end_proc_cpu.system - self._start_proc_cpu.system, 0.0)
            psutil_cpu_total = cpu_user + cpu_system
        else:
            cpu_user = process_cpu_time * 0.8
            cpu_system = process_cpu_time * 0.2
            psutil_cpu_total = process_cpu_time

        total_cpu = max(process_cpu_time, psutil_cpu_total)

        cpu_samples = self._cpu_samples if self._cpu_samples else [0.0]
        mem_samples = self._memory_samples if self._memory_samples else [end_ram]

        avg_cpu_pct = float(np.mean(cpu_samples))
        peak_cpu_pct = float(np.max(cpu_samples))
        avg_ram_mb = float(np.mean(mem_samples))
        peak_ram_mb = float(np.max(mem_samples))

        return {
            'wall_clock_time': float(wall_time),
            'total_cpu_time': float(total_cpu),
            'cpu_user_time': float(cpu_user),
            'cpu_system_time': float(cpu_system),
            'avg_cpu_percent': avg_cpu_pct,
            'peak_cpu_percent': peak_cpu_pct,
            'avg_ram_mb': avg_ram_mb,
            'peak_ram_mb': peak_ram_mb,
        }


def measure_execution(fn, *args, **kwargs):
    """
    Convenience wrapper to run a callable and record execution metrics.
    Returns (result, metrics_dict).
    """
    monitor = ResourceMonitor()
    monitor.start()
    result = fn(*args, **kwargs)
    metrics = monitor.stop()
    return result, metrics


def compute_model_memory(model):
    """
    Measures the memory footprint of an estimator or ensemble in KB:
      1. Serialized byte size via pickle
      2. Tree node count if tree-based
    """
    try:
        serialized = pickle.dumps(model)
        size_bytes = len(serialized)
        size_kb = size_bytes / 1024.0
    except Exception:
        size_kb = 0.0

    total_nodes = 0
    if hasattr(model, 'estimators_'):
        for est in model.estimators_:
            if hasattr(est, 'tree_'):
                total_nodes += est.tree_.node_count
            elif isinstance(est, (list, np.ndarray)):
                for sub in est:
                    if hasattr(sub, 'tree_'):
                        total_nodes += sub.tree_.node_count

    return {
        'model_size_kb': float(size_kb),
        'model_size_mb': float(size_kb / 1024.0),
        'tree_node_count': total_nodes
    }
