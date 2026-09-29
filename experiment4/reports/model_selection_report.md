# Model Selection Report: Component Candidate Benchmark & Multi-Objective Evaluation

**Evaluation Split:** Exactly 16,000 Training / 4,000 Validation samples from Initial 20,000 Reference Telemetry (Seed 42)

## 1. Candidate Models Benchmark Results (10 Evaluated Architectures)

| model | family | val_f1 | val_accuracy | val_precision | val_recall | val_balanced_acc | fit_time_ms | pred_time_ms | cpu_time_ms | avg_pairwise_disagreement | avg_pairwise_error_corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM | Tree / Boosting | 0.6848 | 0.826 | 0.7214 | 0.6517 | 0.7745 | 88.5 | 12.5 | 93.8 | 0.033 | 0.8881 |
| LogisticRegression | Linear Discriminative | 0.6865 | 0.8285 | 0.7305 | 0.6474 | 0.7749 | 32.6 | 2.7 | 31.2 | 0.0356 | 0.8788 |
| GaussianNB | Generative Bayesian | 0.7065 | 0.8013 | 0.6178 | 0.825 | 0.8083 | 7.3 | 2.1 | 0.0 | 0.126 | 0.5875 |
| RandomForest | Tree / Bagging | 0.6874 | 0.8245 | 0.7109 | 0.6655 | 0.7775 | 609.3 | 9.5 | 593.8 | 0.034 | 0.8849 |
| ExtraTrees | Tree / Bagging | 0.6595 | 0.818 | 0.7209 | 0.6078 | 0.7558 | 138.4 | 10.0 | 140.6 | 0.0454 | 0.8481 |
| XGBoost | Tree / Boosting | 0.6788 | 0.8237 | 0.7198 | 0.6422 | 0.7701 | 170.8 | 5.1 | 265.6 | 0.0317 | 0.8932 |
| CatBoost | Tree / Boosting | 0.6867 | 0.8257 | 0.7174 | 0.6586 | 0.7763 | 343.9 | 1.8 | 718.8 | 0.0316 | 0.8929 |
| HistGradientBoosting | Tree / Boosting | 0.6773 | 0.8227 | 0.7175 | 0.6414 | 0.7691 | 2624.5 | 5.5 | 296.9 | 0.0318 | 0.893 |
| LinearSVC | Linear Discriminative | 0.6853 | 0.828 | 0.73 | 0.6457 | 0.7741 | 58.0 | 5.2 | 109.4 | 0.0361 | 0.8773 |
| SmallMLP | Neural Network | 0.6991 | 0.8297 | 0.7171 | 0.6819 | 0.786 | 973.3 | 1.7 | 953.1 | 0.0336 | 0.8853 |

## 2. Top 15 Candidate 3-Model Ensembles (Multi-Objective Ranking)

| ensemble_models | model1 | model2 | model3 | families | distinct_families | ensemble_val_f1 | ensemble_val_acc | trio_avg_disagreement | trio_avg_error_corr | total_fit_time_ms | composite_selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GaussianNB + ExtraTrees + LinearSVC | GaussianNB | ExtraTrees | LinearSVC | Generative Bayesian, Linear Discriminative, Tree / Bagging | 3 | 0.7082 | 0.8175 | 0.157 | 0.6564 | 203.7 | 0.6011 |
| LogisticRegression + GaussianNB + ExtraTrees | LogisticRegression | GaussianNB | ExtraTrees | Generative Bayesian, Linear Discriminative, Tree / Bagging | 3 | 0.708 | 0.8173 | 0.1562 | 0.658 | 178.3 | 0.6008 |
| LightGBM + GaussianNB + ExtraTrees | LightGBM | GaussianNB | ExtraTrees | Generative Bayesian, Tree / Bagging, Tree / Boosting | 3 | 0.7073 | 0.8165 | 0.151 | 0.6705 | 234.2 | 0.5964 |
| GaussianNB + ExtraTrees + XGBoost | GaussianNB | ExtraTrees | XGBoost | Generative Bayesian, Tree / Bagging, Tree / Boosting | 3 | 0.709 | 0.8177 | 0.1497 | 0.6741 | 316.5 | 0.5953 |
| LightGBM + GaussianNB + LinearSVC | LightGBM | GaussianNB | LinearSVC | Generative Bayesian, Linear Discriminative, Tree / Boosting | 3 | 0.7122 | 0.8207 | 0.1407 | 0.69 | 153.8 | 0.5946 |
| LightGBM + LogisticRegression + GaussianNB | LightGBM | LogisticRegression | GaussianNB | Generative Bayesian, Linear Discriminative, Tree / Boosting | 3 | 0.7116 | 0.8203 | 0.1403 | 0.691 | 128.4 | 0.5943 |
| LogisticRegression + GaussianNB + XGBoost | LogisticRegression | GaussianNB | XGBoost | Generative Bayesian, Linear Discriminative, Tree / Boosting | 3 | 0.7118 | 0.8203 | 0.1417 | 0.6884 | 210.7 | 0.5941 |
| GaussianNB + XGBoost + LinearSVC | GaussianNB | XGBoost | LinearSVC | Generative Bayesian, Linear Discriminative, Tree / Boosting | 3 | 0.7118 | 0.8203 | 0.1422 | 0.6874 | 236.1 | 0.594 |
| GaussianNB + ExtraTrees + CatBoost | GaussianNB | ExtraTrees | CatBoost | Generative Bayesian, Tree / Bagging, Tree / Boosting | 3 | 0.7083 | 0.8165 | 0.1467 | 0.6801 | 489.6 | 0.591 |
| GaussianNB + CatBoost + LinearSVC | GaussianNB | CatBoost | LinearSVC | Generative Bayesian, Linear Discriminative, Tree / Boosting | 3 | 0.7114 | 0.8195 | 0.138 | 0.6963 | 409.2 | 0.5892 |
| LogisticRegression + GaussianNB + CatBoost | LogisticRegression | GaussianNB | CatBoost | Generative Bayesian, Linear Discriminative, Tree / Boosting | 3 | 0.7111 | 0.8193 | 0.1375 | 0.6972 | 383.8 | 0.5891 |
| LogisticRegression + GaussianNB + LinearSVC | LogisticRegression | GaussianNB | LinearSVC | Generative Bayesian, Linear Discriminative | 2 | 0.7099 | 0.82 | 0.1313 | 0.7119 | 97.9 | 0.5885 |
| LightGBM + GaussianNB + XGBoost | LightGBM | GaussianNB | XGBoost | Generative Bayesian, Tree / Boosting | 2 | 0.711 | 0.8197 | 0.1323 | 0.7108 | 266.6 | 0.5871 |
| LogisticRegression + GaussianNB + RandomForest | LogisticRegression | GaussianNB | RandomForest | Generative Bayesian, Linear Discriminative, Tree / Bagging | 3 | 0.7105 | 0.8195 | 0.1385 | 0.695 | 649.2 | 0.5861 |
| GaussianNB + RandomForest + LinearSVC | GaussianNB | RandomForest | LinearSVC | Generative Bayesian, Linear Discriminative, Tree / Bagging | 3 | 0.7108 | 0.8197 | 0.1388 | 0.6946 | 674.6 | 0.5861 |

## 3. Recommended Winning Ensemble

**Selected Ensemble:** `GaussianNB + ExtraTrees + LinearSVC`

- **Distinct Learning Families:** 3
- **Validation Ensemble F1:** 0.7082
- **Validation Accuracy:** 0.8175
- **Pairwise Error Correlation:** 0.6564
- **Prediction Disagreement:** 0.157
- **Total Fit Time:** 203.7 ms
- **Composite Score:** 0.6011
