import sys, os
sys.path.insert(0, "experiments/exp9a")
from three_dataset_load import LOADERS, build_windows
import three_dataset_run as R
import pandas as pd

data = LOADERS["UGR'16"]()
stream, sd = build_windows(data, max_samples=None)
pw, sm = R.run_seed(stream, sd, 42)
print("=== ORIGINAL runner, UGR'16 seed 42 ===")
for _, r in sm.iterrows():
    print(f"  {r['method']:16s} per_window_F1={r['macro_f1']:.4f} "
          f"adapt_cpu={r['adaptation_cpu_sec']:.3f} retr={r['retrain_events']} "
          f"reuse={r['reuse_events']} parity={r.get('parity_refits')}")
old = pd.read_csv("results/experiment_9a_three/raw/summary_ugr16_full.csv")
old = old.query("seed==42")[["method", "macro_f1", "adaptation_cpu_sec",
                             "retrain_events", "reuse_events"]]
print("\n=== committed ===")
print(old.to_string(index=False))
