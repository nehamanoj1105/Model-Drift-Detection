# Reproducibility Guide for Exp2 Final

All random seeds are explicit. Pre-registered configuration files and hashes govern all parameter settings.

## Environment
- Python 3.10+
- Requirements: numpy, pandas, scikit-learn, river (optional for S4 Elec2 dataset loading)

## Execution Instructions

### 1. Complete End-to-End Study
To run the full experimental pipeline across all streams and execute controls:

```bash
python experiments/exp2_final/src/main.py
```

This command will:
1. Compute the dynamic G2 reference benchmark on S2 (9B dataset).
2. Generate/load streams S1, N1, N2, S2, S4 (and S3 if path provided).
3. Execute all 8 core methods + appendix methods.
4. Save raw result JSONs to `experiments/exp2_final/runs/final/study_results_raw.json`.
5. Run the complete control suite K1-K12 and write `experiments/exp2_final/audit/gate_results.json`.

### 2. Generate Final Paper Reports & Disclosures
To generate markdown tables and `LIMITATIONS_AUTO.md`:

```bash
python experiments/exp2_final/src/report.py
```

### 3. Verify Pre-Registered Config Hashes
To verify that no config files were modified post-experiment:

```python
from experiments.exp2_final.src.controls import check_k4_config_hashes
passed, msg = check_k4_config_hashes()
print(f"K4 Hash Check: {msg}")
```

### 4. Verify Leakage Prevention
To execute the K2 train/eval leak test:

```python
from experiments.exp2_final.src.controls import check_k2_train_eval_leak
passed, msg = check_k2_train_eval_leak()
print(f"K2 Leak Test: {msg}")
```
