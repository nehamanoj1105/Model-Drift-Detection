# Table 2: Base Model & Ensemble Configuration

| Model | Estimators | Max Depth | Learning Rate / Split | Horizon Buffer |
|---|---|---|---|---|
| Random Forest (RF) | 50 | 7 | min_split=4 | 3,500 |
| Extra Trees (ET) | 50 | 7 | min_split=6 | 2,500 |
| Gradient Boosting (GB) | 50 | 4 | lr=0.08, sub=0.85 | 1,800 |
