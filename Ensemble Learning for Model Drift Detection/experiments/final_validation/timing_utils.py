"""
Unified Timing Instrumentation Module for Final RAPT-E Validation
Provides high-precision time.perf_counter() measurements across all evaluated methods:
  - Frozen
  - Event-Driven
  - Full Retraining
  - Original RAPT
  - RAPT-E (Fixed RAPT-v2)

Ensures exact, fair measurement of:
  - prediction_time
  - adaptation_time (checkpoint_lookup, weight_update, policy_update, model_fit, fallback_fit)
  - total_window_time
  - total_runtime
"""

import time
import pandas as pd
import numpy as np

class TimingTracker:
    def __init__(self, method_name, dataset_label, seed):
        self.method_name = method_name
        self.dataset_label = dataset_label
        self.seed = seed
        
        self.stream_start_wall = 0.0
        self.stream_end_wall = 0.0
        
        self.prediction_time = 0.0
        self.adaptation_time = 0.0
        self.checkpoint_lookup_time = 0.0
        self.weight_update_time = 0.0
        self.model_fit_time = 0.0
        self.fallback_fit_time = 0.0
        
        self.fit_trace_records = []

    def start_stream(self):
        self.stream_start_wall = time.perf_counter()

    def end_stream(self):
        self.stream_end_wall = time.perf_counter()

    def get_total_runtime(self):
        return self.stream_end_wall - self.stream_start_wall

    def log_fit_event(self, model_name, regime_id, num_samples, fit_duration, fit_type="full"):
        self.fit_trace_records.append({
            'timestamp': time.time(),
            'dataset': self.dataset_label,
            'seed': self.seed,
            'method': self.method_name,
            'model': model_name,
            'regime': regime_id,
            'number_of_samples': num_samples,
            'fit_duration': fit_duration,
            'fit_type': fit_type
        })
