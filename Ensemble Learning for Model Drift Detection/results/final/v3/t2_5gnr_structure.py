"""T2: 5G NR stream structure and cadence-matched cheap refresh.

Answers, from the loader (experiments/exp9b/load_and_prepare_stream_9b.py) and
the processed stream itself:
  * what one row is (one 500-packet telemetry window, not one packet);
  * the number of rows and the prefix size;
  * every window-count setting expressed in both rows and packets;
  * the cheap-refresh result with a cadence matched to Full Retraining's retrain
    count, with refit counts.

Outputs:
  T2_stream_structure.csv       one row per stream: rows, packets/row, prefix
  T2_window_settings.csv        window-count settings in rows and packets
  T2_cheap_cadence.csv          RAPT-Cheap vs cadence-matched variant on 5G NR
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402

PACKETS_PER_ROW = 500  # load_and_prepare_stream_9b.py WINDOW_SIZE


def structure():
    rows = []
    for ds in lib.DATASETS:
        stream, sd = lib.S.get_stream(ds)
        slug = lib.S.SLUG[ds]
        if slug == "5g_nr":
            ppr, kind = PACKETS_PER_ROW, "window (500 packets/row)"
            n_rows = len(stream)
            n_packets = n_rows * ppr
            prefix_rows = sd["initial_train_windows"]
            eval_rows = sd["total_windows"] - prefix_rows
        else:
            ppr, kind = 1, "sample (1 packet/row)"
            n_rows = len(stream)
            n_packets = n_rows
            prefix_rows = sd["initial_train_windows"] * sd["window_size"]
            eval_rows = n_rows - prefix_rows
        rows.append({
            "dataset": slug, "row_kind": kind,
            "n_rows": int(n_rows), "packets_per_row": int(ppr),
            "n_packets": int(n_packets),
            "window_size_rows": int(sd["window_size"]),
            "total_windows": int(sd["total_windows"]),
            "prefix_rows": int(prefix_rows),
            "prefix_packets": int(prefix_rows * ppr),
            "eval_windows": int(sd["total_windows"] - sd["initial_train_windows"]),
            "eval_rows": int(eval_rows),
            "eval_packets": int(eval_rows * ppr),
        })
    return pd.DataFrame(rows)


def window_settings():
    stream, sd = lib.S.get_stream("5G NR")
    ppr = PACKETS_PER_ROW
    rows = []
    for k in [1, 5, 10, 50, 80, 100, 400, 499]:
        rows.append({"setting": f"{k} window(s) on 5G NR",
                     "rows": k, "packets": k * ppr})
    rows.append({"setting": "full stream", "rows": len(stream),
                 "packets": len(stream) * ppr})
    rows.append({"setting": "prefix (initial train)",
                 "rows": sd["initial_train_windows"],
                 "packets": sd["initial_train_windows"] * ppr})
    return pd.DataFrame(rows)


def cheap_cadence():
    """Full Retraining's retrain count on 5G NR fixes the matched cadence."""
    stream, sd = lib.S.get_stream("5G NR")
    out = []
    for seed in lib.SEEDS:
        pw, sm, pooled = lib.run_stream(
            stream, sd, seed,
            ["Frozen", "Full Retraining", "RAPT-Cheap-Original", "RAPT-Cheap"])
        fr = sm[sm.method == "Full Retraining"].iloc[0]
        rc = sm[sm.method == "RAPT-Cheap-Original"].iloc[0]
        rcd = sm[sm.method == "RAPT-Cheap"].iloc[0]
        # cadence matched to FR retrains over the eval windows
        eval_w = sd["total_windows"] - sd["initial_train_windows"]
        matched_every = max(1, int(round(eval_w / max(1, fr["retrain_events"]))))
        # run the matched-cadence variant
        lib.PERIODIC_MATCHED = matched_every
        pw2, sm2, pooled2 = lib.run_stream(
            stream, sd, seed, ["Periodic-Cheap-Matched", "RAPT-Cheap-WinEq"])
        pm = sm2[sm2.method == "Periodic-Cheap-Matched"].iloc[0]
        we = sm2[sm2.method == "RAPT-Cheap-WinEq"].iloc[0]
        pf = pooled.set_index("method")["pooled_macro_f1"]
        pf2 = pooled2.set_index("method")["pooled_macro_f1"]
        out.append({
            "seed": seed,
            "fr_retrains": int(fr["retrain_events"]),
            "fr_pooled_f1": float(pf["Full Retraining"]),
            "fr_adapt_cpu": float(fr["adaptation_cpu_sec"]),
            "rapt_cheap_periodic_every": 5,
            "rapt_cheap_refresh_events": int(rc["refresh_events"]),
            "rapt_cheap_refits": int(rc["n_refits"]),
            "rapt_cheap_pooled_f1": float(pf["RAPT-Cheap-Original"]),
            "rapt_cheap_adapt_cpu": float(rc["adaptation_cpu_sec"]),
            "matched_every_rows": matched_every,
            "matched_every_packets": matched_every * PACKETS_PER_ROW,
            "matched_refresh_events": int(pm["retrain_events"]),
            "matched_pooled_f1": float(pf2["Periodic-Cheap-Matched"]),
            "matched_adapt_cpu": float(pm["adaptation_cpu_sec"]),
            "wineq_refresh_events": int(we["refresh_events"]),
            "wineq_pooled_f1": float(pf2["RAPT-Cheap-WinEq"]),
            "wineq_adapt_cpu": float(we["adaptation_cpu_sec"]),
        })
    return pd.DataFrame(out)


def main():
    lib.ensure_dirs()
    st = structure()
    ws = window_settings()
    cc = cheap_cadence()
    lib.write_csv(st, "T2_stream_structure.csv")
    lib.write_csv(ws, "T2_window_settings.csv")
    lib.write_csv(cc, "T2_cheap_cadence.csv")
    print(st.to_string(index=False))
    print()
    print(ws.to_string(index=False))
    print()
    print(cc.round(4).to_string(index=False))
    lib.gate("T2", "PASS", "5G NR structure + cadence-matched cheap refresh")


if __name__ == "__main__":
    main()
