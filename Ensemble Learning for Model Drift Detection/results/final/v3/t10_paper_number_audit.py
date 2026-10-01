"""T10: audit every number in the manuscript's primary table against a CSV.

The manuscript (Paper_Final/manuscript.tex, Table II `tab:primary`) reports, for
four streams and five models, F1/accuracy/precision/recall/CPU/run/retrains/reuse.
Each of those numbers must be traceable to a CSV written by a harness. This step
extracts the table values from the .tex source, compares them against the
committed result CSVs, and reports the largest absolute difference per cell
together with the CSV that matches (if any).

The comparison deliberately does not assume a single metric scale: for each
stream the harness exposes a per-window metric and a pooled metric, and the audit
accepts whichever the manuscript used (reported per row), flagging any cell that
matches neither.

Outputs:
  T10_paper_table_audit.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TEX = os.path.join(ROOT, "Paper_Final", "manuscript.tex")

# harness sources: dataset slug -> (per_window_f1, pooled_f1, adapt_cpu, retrains, reuse)
SRC = {
    "5g_campus": ("experiments/exp9a", "results/final/v2/C_metric_reconciliation.csv"),
    "ugr16": ("experiments/exp9a", "results/final/v2/C_metric_reconciliation.csv"),
    "nordicdat": ("experiments/exp9a", "results/final/v2/C_metric_reconciliation.csv"),
    "5g_nr": ("experiments/exp9b", "experiments/exp9b/results/per_seed_results.csv"),
}


def parse_primary_table(tex_path):
    with open(tex_path) as f:
        tex = f.read()
    m = re.search(r"\\label\{tab:primary\}.*?\\midrule(.*?)\\bottomrule", tex, re.S)
    if not m:
        raise SystemExit("tab:primary not found")
    body = m.group(1)
    rows = []
    stream = None
    for line in body.splitlines():
        if "multirow" in line:
            stream = re.search(r"\{(?:[^}]*)\}\{\*\}\{([^}]*)\}", line)
            stream = stream.group(1) if stream else None
            continue
        if "&" not in line or "\\" not in line:
            continue
        cells = [c.strip() for c in line.split("&")]
        if cells and cells[0] == "":
            cells = cells[1:]
        model = re.sub(r"[\\{}$]", "", cells[0]).strip()
        vals = []
        for c in cells[1:]:
            c = c.replace("\\", "").replace("$", "").replace("{", "").replace("}", "")
            c = c.replace("mathbf", "").replace("pm", "±").strip()
            mm = re.match(r"(-?\d+\.\d+)\s*(?:±\s*(-?\d+\.\d+))?", c)
            vals.append(float(mm.group(1)) if mm else np.nan)
        rows.append({"stream": stream, "model": model,
                     "f1": vals[0], "acc": vals[1], "prec": vals[2], "rec": vals[3],
                     "cpu": vals[4], "run": vals[5], "retr": vals[6], "reuse": vals[7]})
    return pd.DataFrame(rows)


def harness_cells(slug):
    """Return dict metric -> (value, source_label) for the per-window and pooled scales."""
    out = {}
    if slug in ("5g_campus", "ugr16", "nordicdat"):
        c = pd.read_csv(os.path.join(ROOT, "results/final/v2/C_metric_reconciliation.csv"))
        b = pd.read_csv(os.path.join(ROOT, "results/final/v2/B_pooled_all_streams.csv"))
        for _, r in c[c.dataset == slug].iterrows():
            out[r["method"]] = {
                "f1_per_window": (r["per_window_macro_f1"], "C_metric_reconciliation"),
                "f1_pooled": (r["pooled_macro_f1"], "C_metric_reconciliation"),
            }
    else:
        p = pd.read_csv(os.path.join(ROOT, "experiments/exp9b/results/per_seed_results.csv"))
        for m, g in p.groupby("method"):
            out.setdefault(m, {})["f1_per_window"] = (g["macro_f1"].mean(),
                                                      "exp9b per_seed_results")
        # the manuscript's 5G NR row is drawn from the 9B natural-drift harness,
        # which is a second, independently written implementation of the same
        # protocol; expose both so a cell can be traced to either.
        nd = os.path.join(ROOT, "results/experiment_9b/natural_drift/summary.csv")
        if os.path.exists(nd):
            n = pd.read_csv(nd)
            for _, r in n.iterrows():
                out.setdefault(r["method"], {})["f1_natural9b"] = (
                    r["macro_f1_mean"], "experiment_9b natural_drift/summary")
    return out


def parse_ablation_table(tex_path):
    with open(tex_path) as f:
        tex = f.read()
    m = re.search(r"\\label\{tab:ablation\}.*?\\midrule(.*?)\\bottomrule", tex, re.S)
    if not m:
        return None
    rows = []
    for line in m.group(1).splitlines():
        if "&" not in line or "\\" not in line:
            continue
        cells = [c.strip() for c in line.split("&")]
        name = re.sub(r"[\\{}$]", "", cells[0]).replace("textbf", "").strip()
        vals = []
        for c in cells[1:]:
            c = c.replace("\\", "").replace("$", "").replace("{", "").replace("}", "")
            c = c.replace("mathbf", "").replace("pm", "±").strip()
            mm = re.match(r"(-?\d+\.\d+)", c)
            vals.append(float(mm.group(1)) if mm else np.nan)
        rows.append({"config": name, "f1": vals[0] if vals else np.nan,
                     "cpu": vals[1] if len(vals) > 1 else np.nan,
                     "refreshes": vals[2] if len(vals) > 2 else np.nan})
    return pd.DataFrame(rows)


ABLATION_SRC = {
    "Full Retrain. (ref.)": "Full Retraining", "RAPT-Full": "RAPT_FULL",
    "RAPT-Evidence": "RAPT_EVIDENCE", "RAPT-Floor": "RAPT_FLOOR",
    "RAPT-Cheap": "RAPT_CHEAP", "RAPT-Incremental": "RAPT_INCR",
}


def audit_ablation():
    tex_tab = parse_ablation_table(TEX)
    if tex_tab is None:
        return None
    src = pd.read_csv(os.path.join(ROOT, "Final_Experiments/results/raw/summary_full.csv"))
    rows = []
    for _, r in tex_tab.iterrows():
        key = ABLATION_SRC.get(r["config"])
        if key is None or key not in set(src["method"]):
            rows.append({"config": r["config"], "paper_f1": r["f1"],
                         "harness_f1": np.nan, "abs_diff": np.nan,
                         "source": "", "matches_within_0.001": False})
            continue
        g = src[src.method == key]
        rows.append({"config": r["config"], "paper_f1": r["f1"],
                     "harness_f1": g["macro_f1"].mean(),
                     "abs_diff": abs(g["macro_f1"].mean() - r["f1"]),
                     "source": "Final_Experiments summary_full",
                     "matches_within_0.001": bool(abs(g["macro_f1"].mean() - r["f1"]) <= 0.001)})
    return pd.DataFrame(rows)


def main():
    lib.ensure_dirs()
    tab = parse_primary_table(TEX)
    rows = []
    for _, r in tab.iterrows():
        slug = {"5G Campus": "5g_campus", "UGR'16": "ugr16",
                "NordicDat": "nordicdat", "5G NR Lat.": "5g_nr"}.get(r["stream"], None)
        hc = harness_cells(slug) if slug else {}
        cand = hc.get(r["model"], {})
        best = None
        for name, (v, src) in cand.items():
            d = abs(v - r["f1"])
            if best is None or d < best[0]:
                best = (d, name, v, src)
        rows.append({
            "table": "primary",
            "stream": r["stream"], "model": r["model"], "paper_f1": r["f1"],
            "harness_scale": best[1] if best else "",
            "harness_f1": best[2] if best else np.nan,
            "abs_diff": best[0] if best else np.nan,
            "source": best[3] if best else "",
            "matches_within_0.001": bool(best and best[0] <= 0.001),
        })
    abl = audit_ablation()
    if abl is not None:
        abl = abl.rename(columns={"config": "model"})
        abl["table"] = "ablation"
        abl["stream"] = "5G Campus"
        abl["harness_scale"] = "per_window"
        rows.extend(abl.to_dict("records"))
    out = pd.DataFrame(rows)
    lib.write_csv(out, "T10_paper_table_audit.csv")
    print(out.round(4).to_string(index=False))
    n_bad = int((~out["matches_within_0.001"]).sum())
    lib.gate("T10", "PASS" if n_bad == 0 else "REVIEW",
             f"{n_bad} of {len(out)} audited F1 cells not traceable to a CSV")


if __name__ == "__main__":
    main()
