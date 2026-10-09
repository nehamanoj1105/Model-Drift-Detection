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
            adwin = {k: v for k, v in choice["ADWIN"].items() if not k.startswith("_")}
            ph = {k: v for k, v in choice["Page-Hinkley"].items() if not k.startswith("_")}
            cfg.update(adwin); cfg.update(ph)
            notes["ADWIN"] = {k: v for k, v in choice["ADWIN"].items()}
            notes["Page-Hinkley"] = {k: v for k, v in choice["Page-Hinkley"].items()}
        else:
            cfg.update({k: v for k, v in choice.items() if not k.startswith("_")})
            notes["guard"] = choice
    os.makedirs(CONFIG_DIR, exist_ok=True)
    out = {"frozen_from_dev": True, "config": cfg, "dev_notes": notes}
    with open(os.path.join(CONFIG_DIR, "final.yaml"), "w") as f:
        yaml.safe_dump({k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                        for k, v in cfg.items()}, f, sort_keys=True)
    with open(os.path.join(CONFIG_DIR, "final_full.yaml"), "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)
    print(json.dumps({"frozen": {k: cfg[k] for k in
                   ("guard_lookback", "guard_margin", "adwin_delta",
                    "ph_min_instances", "ph_threshold")}}, indent=2))


if __name__ == "__main__":
    main()
