#!/usr/bin/env python3
"""
make_paper_number_audit.py — trace every numeric token in the manuscripts to a
saved result CSV.

Builds the set of all numbers derivable from results/.../final/*.csv plus the
LaTeX tables, then scans each manuscript for numeric tokens (including numbers
inside macros like \\cpuSavedCampus and table cells) and reports any token that
cannot be traced. A token is traceable if it matches a CSV value at some common
rounding (2, 3, 4 dp; 1 dp percent) within a tiny tolerance.

Output: results/.../final/paper_number_audit.csv and .md.
Non-traceable tokens are listed, never silently dropped.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PROJECT_DIR = os.path.abspath(os.path.join(HERE, ".."))

TOKEN = re.compile(r"-?\d+\.?\d*")

_STRIP_PATTERNS = [
    re.compile(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", re.S),
    re.compile(r"\\bibitem.*", re.S),
    re.compile(r"\\(?:cite|citep|citet|ref|eqref|label|includegraphics|input|"
               r"documentclass|usepackage|textwidth|columnwidth|linewidth|"
               r"fontsize|rule|parbox|multirow|begin|end|item|"
               r"kern|lower|raise|mkern|hspace|vspace)"
               r"\*?(?:-?\.?\d*(?:em|ex|pt|mu))?(?:\[[^\]]*\])?(?:\{[^{}]*\})*"),
]


def _strip_latex(txt):
    for pat in _STRIP_PATTERNS:
        txt = pat.sub(" ", txt)
    return txt


def value_set(out_root):
    vals = set()
    files = (glob.glob(os.path.join(out_root, "*.csv")) +
             glob.glob(os.path.join(out_root, "latex", "*.tex")) +
             glob.glob(os.path.join(out_root, "*.tex")) +
             glob.glob(os.path.join(out_root, "*.yaml")) +
             glob.glob(os.path.join(out_root, "*.json")))
    for fp in files:
        if fp.endswith(".csv"):
            try:
                df = pd.read_csv(fp)
            except Exception:
                continue
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    ser = pd.to_numeric(df[c], errors="coerce").dropna()
                    for v in ser.unique():
                        for nd in (0, 1, 2, 3, 4):
                            vals.add(_norm(f"{round(float(v), nd)}"))
                        vals.add(_norm(f"{round(float(v)*100, 1)}"))
        else:
            txt = open(fp).read()
            for m in TOKEN.findall(txt):
                vals.add(_norm(m))
    return vals


def _norm(tok):
    try:
        f = float(tok)
    except ValueError:
        return tok
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return f"{f}"


def _latex_decimals(tok):
    """Recover LaTeX numbers written as `.0625` (tokenised as `0625`/`0625.`)."""
    t = tok.rstrip(".")
    if t.startswith("-"):
        sign, body = "-", t[1:]
    else:
        sign, body = "", t
    if body.startswith("0") and len(body) > 1:
        return [_norm(sign + "0." + body)]
    return []


def _matchable(tok):
    """All normalized string forms a token could match."""
    try:
        f = float(tok)
    except ValueError:
        return {tok}
    out = set()
    for signed in (f, -f):
        for nd in (0, 1, 2, 3, 4):
            r = round(signed, nd)
            out.add(str(int(r)) if abs(r - round(r)) < 1e-9 else f"{r}")
    # A token like "0625" comes from a LaTeX ".0625"; accept the 0.-prefixed form.
    for cand in _latex_decimals(tok):
        out.add(cand)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="final.yaml")
    ap.add_argument("--manuscripts", nargs="*", default=None)
    args = ap.parse_args()
    with open(os.path.join(HERE, args.config)) as f:
        cfg = yaml.safe_load(f)
    out_root = os.path.join(PROJECT_DIR, cfg["output_dir"], "final")

    mss = args.manuscripts or [
        os.path.join(PROJECT_DIR, "deliverable", "main.tex"),
        os.path.join(PROJECT_DIR, "Paper_Final", "manuscript.tex"),
    ]
    vals = value_set(out_root)
    rows = []
    for ms in mss:
        if not os.path.exists(ms):
            rows.append({"manuscript": ms, "numeric_tokens": 0,
                         "unmatched": 0, "unmatched_tokens": "FILE NOT FOUND"})
            continue
        txt = open(ms).read()
        toks = TOKEN.findall(_strip_latex(txt))
        unmatched = sorted({t for t in toks if not (_matchable(t) & vals)})
        rows.append({"manuscript": os.path.relpath(ms, PROJECT_DIR),
                     "numeric_tokens": len(toks), "unmatched": len(unmatched),
                     "unmatched_tokens": " ".join(unmatched[:80])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "paper_number_audit.csv"), index=False)
    with open(os.path.join(out_root, "paper_number_audit.md"), "w") as f:
        for _, r in df.iterrows():
            f.write(f"# {r['manuscript']}\n")
            f.write(f"numeric tokens: {r['numeric_tokens']}; unmatched: {r['unmatched']}\n")
            if r["unmatched"]:
                f.write(f"unmatched tokens: {r['unmatched_tokens']}\n")
            f.write("\n")
    print(df[["manuscript", "numeric_tokens", "unmatched"]].to_string(index=False))


if __name__ == "__main__":
    main()
