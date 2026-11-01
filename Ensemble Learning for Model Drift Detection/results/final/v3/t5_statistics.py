"""T5: statistical validation.

  * block bootstrap: 24 day-blocks for UGR'16, regime-visit blocks otherwise,
    for every pairwise paper comparison (primary metric, pooled macro-F1 is a
    run-level scalar; the paired difference is computed on the per-window
    macro-F1 within a run, which is the unit the bootstrap resamples);
  * seed-level exact Wilcoxon over the 5 seeds (pooled macro-F1);
  * TOST on 5G Campus (margins 0.005 / 0.01 primary / 0.02);
  * 20 seeds for the cheap configs;
  * the missing 5G NR RAPT vs Full Retraining test;
  * Holm correction over the Campus ladder, labelled exploratory.

Outputs:
  T5_block_bootstrap.csv
  T5_seed_wilcoxon.csv
  T5_tost_campus.csv
  T5_cheap_20seeds.csv
  T5_campus_ladder_holm.csv
  raw/T5_<slug>_seed<seed>.csv   per-window records used by the bootstrap
"""
import os
import sys
import itertools

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402

PRIMARY = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]
CHEAP = ["FR-Cheap", "Periodic-Cheap-5", "RAPT-Cheap"]
LADDER = ["RAPT_v2", "RAPT-Enhanced_v2", "RAPT_v2_pure", "RAPT-Enhanced_v2_pure"]


def blocks_for(slug, stream_df):
    rs = lib.regime_of(stream_df)
    wids = rs.index.to_numpy()
    labs = rs.to_numpy()
    if slug == "ugr16":
        n = 24
        edges = np.linspace(0, len(wids), n + 1).astype(int)
        return {b: wids[edges[b]:edges[b + 1]] for b in range(n)}
    # regime visits: contiguous runs of the same regime id
    runs = {}
    b, start = 0, 0
    for i in range(1, len(labs) + 1):
        if i == len(labs) or labs[i] != labs[start]:
            runs[b] = wids[start:i]
            b += 1
            start = i
    return runs


def run_primary():
    """5 seeds, 5 primary models, all four streams; keep per-window raw."""
    per_win, pooled_rows = {}, []
    for ds in lib.DATASETS:
        stream, sd = lib.S.get_stream(ds)
        slug = lib.S.SLUG[ds]
        for seed in lib.SEEDS:
            pw, sm, pooled = lib.run_stream(stream, sd, seed, PRIMARY)
            lib.write_raw(pw, f"T5_{slug}_seed{seed}.csv")
            per_win[(slug, seed)] = pw
            pooled = pooled.copy()
            pooled.insert(0, "dataset", slug); pooled["seed"] = seed
            pooled_rows.append(pooled)
            print(f"  primary {slug} seed {seed}", flush=True)
    return per_win, pd.concat(pooled_rows, ignore_index=True)


def block_bootstrap(per_win, pooled):
    rows = []
    for ds in lib.DATASETS:
        slug = lib.S.SLUG[ds]
        stream, sd = lib.S.get_stream(ds)
        blocks = blocks_for(slug, stream)
        # per-window macro-F1 averaged over seeds per window
        for a, b in itertools.combinations(PRIMARY, 2):
            diffs, los, his = [], [], []
            for seed in lib.SEEDS:
                pw = per_win[(slug, seed)]
                m, lo, hi, nb = lib.block_bootstrap_ci(pw, a, b, blocks, seed=seed)
                diffs.append(m); los.append(lo); his.append(hi)
            rows.append({
                "dataset": slug, "method_a": a, "method_b": b,
                "n_blocks": nb, "mean_diff_per_window": float(np.nanmean(diffs)),
                "ci_low": float(np.nanmean(los)), "ci_high": float(np.nanmean(his)),
                "excludes_zero": bool(np.nanmean(los) > 0 or np.nanmean(his) < 0),
                "direction": ("a>b" if np.nanmean(diffs) > 0 else "a<b"),
            })
    return pd.DataFrame(rows)


def seed_wilcoxon(pooled):
    rows = []
    for ds in lib.DATASETS:
        slug = lib.S.SLUG[ds]
        g = pooled[pooled.dataset == slug]
        piv = g.pivot_table(index="seed", columns="method", values="pooled_macro_f1")
        for a, b in itertools.combinations(PRIMARY, 2):
            if a not in piv or b not in piv:
                continue
            p, md = lib.paired_wilcoxon(piv[a].to_numpy(), piv[b].to_numpy())
            rows.append({"dataset": slug, "method_a": a, "method_b": b,
                         "seed_mean_diff": md, "wilcoxon_p": p,
                         "cohen_dz": lib.cohen_d(piv[a].to_numpy(), piv[b].to_numpy()),
                         "note": "min two-sided p at n=5 is 0.0625"})
    return pd.DataFrame(rows)


def tost_campus(pooled):
    g = pooled[pooled.dataset == "5g_campus"]
    piv = g.pivot_table(index="seed", columns="method", values="pooled_macro_f1")
    rows = []
    for a, b in [("RAPT", "RAPT-Enhanced"), ("RAPT_v2", "RAPT-Enhanced_v2")]:
        if a not in piv or b not in piv:
            continue
        for margin in [0.005, 0.01, 0.02]:
            p, eq = lib.tost(piv[a].to_numpy(), piv[b].to_numpy(), margin)
            rows.append({"dataset": "5g_campus", "method_a": a, "method_b": b,
                         "margin": margin, "tost_p": p, "equivalent": eq,
                         "primary": margin == 0.01})
    return pd.DataFrame(rows)


def cheap_20seeds():
    rows = []
    for ds in lib.DATASETS:
        stream, sd = lib.S.get_stream(ds)
        for seed in lib.SEEDS_CHEAP:
            pw, sm, pooled = lib.run_stream(stream, sd, seed, CHEAP)
            d = sm.merge(pooled[["method", "pooled_macro_f1"]], on="method")
            d.insert(0, "dataset", lib.S.SLUG[ds])
            rows.append(d)
        print(f"  20-seed cheap {lib.S.SLUG[ds]} done", flush=True)
    return pd.concat(rows, ignore_index=True)


def campus_ladder_holm():
    """RAPT_v2 vs each provenance variant on Campus, Holm-corrected (exploratory)."""
    stream, sd = lib.S.get_stream("5G Campus QoS")
    rows = []
    for seed in lib.SEEDS:
        pw, sm, pooled = lib.run_stream(stream, sd, seed, LADDER)
        d = sm.merge(pooled[["method", "pooled_macro_f1"]], on="method")
        d["seed"] = seed
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    piv = df.pivot_table(index="seed", columns="method", values="pooled_macro_f1")
    comps, ps, diffs = [], [], []
    for m in LADDER[1:]:
        p, md = lib.paired_wilcoxon(piv[m].to_numpy(), piv["RAPT_v2"].to_numpy())
        comps.append(f"{m} vs RAPT_v2"); ps.append(p); diffs.append(md)
    adj = lib.holm(ps)
    return pd.DataFrame({"comparison": comps, "mean_diff": diffs,
                         "wilcoxon_p": ps, "holm_p": adj,
                         "label": "exploratory"})


def main():
    lib.ensure_dirs()
    lib.freeze_config()
    per_win, pooled = run_primary()
    bb = block_bootstrap(per_win, pooled)
    sw = seed_wilcoxon(pooled)
    tt = tost_campus(pooled)
    cs = cheap_20seeds()
    hl = campus_ladder_holm()
    lib.write_csv(bb, "T5_block_bootstrap.csv")
    lib.write_csv(sw, "T5_seed_wilcoxon.csv")
    lib.write_csv(tt, "T5_tost_campus.csv")
    lib.write_csv(cs, "T5_cheap_20seeds.csv")
    lib.write_csv(hl, "T5_campus_ladder_holm.csv")

    # explicit 5G NR RAPT vs Full Retraining
    nr = bb[(bb.dataset == "5g_nr") &
            (((bb.method_a == "RAPT") & (bb.method_b == "Full Retraining")) |
             ((bb.method_a == "Full Retraining") & (bb.method_b == "RAPT")))]
    print("5G NR RAPT vs Full Retraining:")
    print(nr.to_string(index=False))
    print(bb.round(4).to_string(index=False))
    lib.gate("T5", "PASS", "statistics: block bootstrap, wilcoxon, TOST, 20-seed cheap, Holm")


if __name__ == "__main__":
    main()
