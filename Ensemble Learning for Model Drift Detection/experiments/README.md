# Reproducing the audited manuscript

All paths are relative to `Ensemble Learning for Model Drift Detection/`.
Python is `/usr/local/bin/python` with `numpy pandas matplotlib psutil scipy`.

## 1. Build the paper

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Result: 7 pages, 0 errors, 0 undefined references. The original, never modified,
is `paper/main_original.tex`.

## 2. Regenerate the figures

```bash
cd paper
python make_paper_figures.py     # writes figures/*.png and figures/*.pdf
```

Sources: `results/experiment_9a_three/raw/summary_*_full.csv` and
`Final_Experiments/results/`.

## 3. Re-run the audit experiments

```bash
cd audit
python exp_e1_rapt_equivalence.py   # RAPT copies are identical
python exp_e2_ugr_mechanism.py      # UGR'16 mechanism isolation
python exp_e3_ugr16.py              # recurrence / frozen-F1 statistics
python exp_e3b_conditional.py       # conditional vs covariate change
python exp_e4_tost.py               # TOST equivalence
python exp_e5_safe_variant.py       # absolute-floor variant
```

Console output for each is committed as `audit/e*_out.txt`.

## 4. Re-validate every number in the paper

```bash
cd audit
python validate_phase1_full.py      # 143 checks against raw artifacts
python verify_bib.py                # Crossref check of references.bib
```

## 5. Underlying experiments (unchanged)

```bash
python experiments/exp9a/three_dataset_run.py full      # Table II
python experiments/exp9a/three_dataset_figures.py full
python experiments/exp9b/run_exp9b_drift.py             # natural drift
python Final_Experiments/run_final.py                   # ablation ladder
```

The audit scripts import the 9A harness (`experiments/exp9a/`) and the
`Final_Experiments` modules directly and do not modify them.
