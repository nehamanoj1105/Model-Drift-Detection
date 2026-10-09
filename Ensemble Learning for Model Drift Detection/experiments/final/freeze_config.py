"""Freeze the dev-selected configuration into configs/final.yaml.

Reads results/final/dev/{detector_choice,guard_choice}.json (produced by
sweeps.py) and writes a single frozen config. The final run loads exactly this
file, so nothing can be re-tuned after final results are seen.
"""

import os
import sys
import json
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from final_config import DEFAULT_CFG, CONFIG_DIR, RAW_FINAL

DEV = os.path.join(RAW_FINAL, "..", "dev")


def main():
    cfg = dict(DEFAULT_CFG)
    notes = {}
    for fn, keys in [("detector_choice.json", ("ADWIN", "Page-Hinkley")),
                     ("guard_choice.json", ("guard_lookback", "guard_margin"))]:
        p = os.path.join(DEV, fn)
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}; run sweeps first")
        with open(p) as f:
            choice = json.load(f)
        if fn.startswith("detector"):
            adwin = {k: v for k, v in choice["ADWIN"].items()
                     if k.startswith("adwin_")}
            ph = {k: v for k, v in choice["Page-Hinkley"].items()
                  if k.startswith("ph_")}
            cfg.update(adwin); cfg.update(ph)
            notes["ADWIN"] = {k: v for k, v in choice["ADWIN"].items()}
            notes["Page-Hinkley"] = {k: v for k, v in choice["Page-Hinkley"].items()}
        else:
            cfg.update({k: v for k, v in choice.items() if not k.startswith("_")})
            notes["guard"] = choice
    # EDD (DDM) and EDMA (EWMA) are per-sample error-rate charts; their dev
    # operating points were chosen by the same fire-rate criterion as ADWIN/PH
    # and are frozen here rather than re-swept.
    cfg["edd_warning"] = 2.0
    cfg["edd_drift"] = 3.0
    cfg["edma_alpha"] = 0.3
    cfg["edma_k"] = 1.5
    os.makedirs(CONFIG_DIR, exist_ok=True)
    out = {"frozen_from_dev": True, "config": cfg, "dev_notes": notes}

    def norm(v):
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, float) and float(v).is_integer():
            return int(v)
        if isinstance(v, float):
            return float(v)
        return v

    with open(os.path.join(CONFIG_DIR, "final.yaml"), "w") as f:
        yaml.safe_dump({k: norm(v) for k, v in cfg.items()}, f, sort_keys=True)
    with open(os.path.join(CONFIG_DIR, "final_full.yaml"), "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)
    print(json.dumps({"frozen": {k: cfg[k] for k in
                   ("guard_lookback", "guard_margin", "adwin_delta",
                    "ph_min_instances", "ph_threshold")}}, indent=2))


if __name__ == "__main__":
    main()
