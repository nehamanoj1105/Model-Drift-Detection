"""Micro-benchmark of refit time over the (trees x buffer) grid, 30 repeats."""

from __future__ import annotations

import time
import numpy as np

from models import create_ensemble


def run(cfg, refit_cfg, seed=42):
    import pandas as pd
    rng = np.random.RandomState(seed)
    n_feat = 20
    X_pool = rng.normal(size=(2000, n_feat))
    y_pool = rng.randint(0, 3, size=2000)

    rows = []
    for trees in refit_cfg["trees"]:
        for buf in refit_cfg["buffers"]:
            Xb, yb = X_pool[:buf], y_pool[:buf]
            times = []
            for r in range(refit_cfg["repeats"]):
                e = create_ensemble(seed + r, cfg)
                e.n_estimators = trees
                from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
                e.rf = RandomForestClassifier(n_estimators=trees,
                                              max_depth=cfg["ensemble"]["max_depth"],
                                              random_state=seed + r, n_jobs=1)
                e.et = ExtraTreesClassifier(n_estimators=trees,
                                            max_depth=cfg["ensemble"]["max_depth"],
                                            random_state=seed + r, n_jobs=1)
                t = time.process_time()
                e.fit(Xb, yb)
                times.append(time.process_time() - t)
            rows.append({"trees": trees, "buffer": buf,
                         "mean_ms": float(np.mean(times) * 1000),
                         "std_ms": float(np.std(times, ddof=1) * 1000),
                         "n_repeats": refit_cfg["repeats"]})
    return pd.DataFrame(rows)
