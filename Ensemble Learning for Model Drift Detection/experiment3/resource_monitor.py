"""
================================================================================
EXPERIMENT 3 — PROCESS-LEVEL RESOURCE MONITORING
================================================================================
Monitors process-level CPU and memory resources using psutil.
Provides accurate wall-clock, CPU user/system/total time, CPU utilization %,
and RAM usage (average and peak).
================================================================================
"""

import os
import time
import threading
import psutil
import numpy as np


class ResourceMonitor:
    """
    Measures CPU and memory utilization during code execution.
    Combines high-resolution timer measurements (time.perf_counter, time.process_time)
    with psutil process metrics and background sampling.
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
        
        # Prime psutil cpu_percent
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
            
        # Use whichever CPU measurement has highest resolution
        total_cpu_time = max(process_cpu_time, psutil_cpu_total)
        
        # Calculate CPU utilization %
        if len(self._cpu_samples) > 0:
            avg_cpu = float(np.mean(self._cpu_samples))
            peak_cpu = float(np.max(self._cpu_samples))
        else:
            # When wall_time is very short (< 0.016s timer tick on Windows),
            # process_cpu_time may jump by 1 tick (15.6ms) due to timer resolution.
            # Avoid artifactual high percentages (e.g. 700% on 2ms single-threaded execution).
            n_cpus = os.cpu_count() or 1
            if wall_time < 0.016:
                bounded_cpu = min(total_cpu_time, wall_time)
                avg_cpu = min((bounded_cpu / wall_time) * 100.0, 100.0)
            else:
                avg_cpu = min((total_cpu_time / wall_time) * 100.0, 100.0 * n_cpus)
            peak_cpu = avg_cpu
            
        # Memory metrics
        if len(self._memory_samples) > 0:
            avg_ram = float(np.mean(self._memory_samples))
            peak_ram = float(np.max(self._memory_samples))
        else:
            avg_ram = (self._start_ram + end_ram) / 2.0
            peak_ram = max(self._start_ram, end_ram)
            
        return {
            'wall_clock_time': float(wall_time),
            'total_cpu_time': float(total_cpu_time),
            'cpu_user_time': float(cpu_user),
            'cpu_system_time': float(cpu_system),
            'avg_cpu_percent': float(avg_cpu),
            'peak_cpu_percent': float(peak_cpu),
            'avg_ram_mb': float(avg_ram),
            'peak_ram_mb': float(peak_ram),
            'ram_before_mb': float(self._start_ram),
            'ram_during_mb': float(avg_ram),
            'ram_after_mb': float(end_ram),
        }


def measure_execution(fn, *args, **kwargs):
    """
    Convenience wrapper to run a function and return (result, resource_metrics).
    """
    monitor = ResourceMonitor()
    monitor.start()
    result = fn(*args, **kwargs)
    metrics = monitor.stop()
    return result, metrics
