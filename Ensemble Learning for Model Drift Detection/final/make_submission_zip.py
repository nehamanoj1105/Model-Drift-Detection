#!/usr/bin/env python3
"""
make_submission_zip.py — assemble the COMSNETS submission package from the
committed deliverable/ tree and the generated 9B artefacts.

The package is a flat, self-contained directory that compiles with
`pdflatex main && bibtex main && pdflatex main && pdflatex main`, and it carries
the 9B drift-severity evidence (figures + tables + report) alongside the paper
figures. Nothing is typed in: every file is copied from its canonical location.

Called as the final stage of final/run_all.py.
"""

from __future__ import annotations

import os
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DELIV = os.path.join(ROOT, "deliverable")
E9B = os.path.join(ROOT, "results", "experiment_9b")
DEST = os.path.join(ROOT, "RAPT_COMSNETS_submission")
ZIP = os.path.join(ROOT, "RAPT_COMSNETS_submission.zip")

PAPER_FILES = ["main.tex", "main.bbl", "references.bib", "main.pdf"]


def copy(src, dst):
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    if os.path.isdir(DEST):
        shutil.rmtree(DEST)
    os.makedirs(os.path.join(DEST, "figures"))
    os.makedirs(os.path.join(DEST, "experiment_9b", "figures"))
    os.makedirs(os.path.join(DEST, "experiment_9b", "tables"))

    for f in PAPER_FILES:
        copy(os.path.join(DELIV, f), os.path.join(DEST, f))
    for f in sorted(os.listdir(os.path.join(DELIV, "figures"))):
        copy(os.path.join(DELIV, "figures", f),
             os.path.join(DEST, "figures", f))

    for f in sorted(os.listdir(os.path.join(E9B, "figures"))):
        if f.endswith(".png"):
            copy(os.path.join(E9B, "figures", f),
                 os.path.join(DEST, "experiment_9b", "figures", f))
    for f in sorted(os.listdir(os.path.join(E9B, "tables"))):
        copy(os.path.join(E9B, "tables", f),
             os.path.join(DEST, "experiment_9b", "tables", f))
    copy(os.path.join(ROOT, "EXPERIMENT_9B_FINAL_REPORT.md"),
         os.path.join(DEST, "experiment_9b", "EXPERIMENT_9B_FINAL_REPORT.md"))

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _, files in os.walk(DEST):
            for fn in files:
                fp = os.path.join(base, fn)
                z.write(fp, os.path.relpath(fp, os.path.dirname(DEST)))
    print(f"submission package: {ZIP} "
          f"({os.path.getsize(ZIP) / 1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
