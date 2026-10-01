"""E3b -- direct test of the UGR'16 'same regime, changed label semantics' claim.

The claim under test: regime identifiers recur on UGR'16, but the conditional
relationship P(Y|X) does not stay put. A frozen-model F1 gap is a poor proxy
because UGR'16 is imbalanced and macro-F1 is noisy on near-empty windows. So we
measure the conditional relationship itself, pairing each recurrence with the
first occurrence of the same regime:

  * dP(Y=1)    change in anomaly prevalence within the regime occurrence;
  * |dAUC|     change in the frozen model's label association on that occurrence
               (a proxy for how well the stored feature-to-label mapping still
               separates the classes);
  * dCentroid  movement of the standardised feature centroid, scaled by the
               between-regime centroid spread -- the covariate component.

If dCentroid is small relative to between-regime spread while dP(Y=1) and dAUC
move, the drift is conditional rather than purely covariate.
"""
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "exp9a"))

from three_dataset_load import LOADERS, build_windows  # noqa: E402
from models_9a import create_base_ensemble  # noqa: E402


def auc(binary, score):
    pos, neg = score[binary], score[~binary]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(stats.mannwhitneyu(pos, neg, alternative="greater").statistic
                 / (len(pos) * len(neg)))


def analyse(loader, positive):
    stream, sd = build_windows(loader())
    feat = sd["feature_columns"]
    X = stream[feat].values.astype(np.float64)
    y = stream["y"].values.astype(int)
    reg = stream["regime_id"].values
    wid = stream["window_id"].values
    n_init = sd["initial_train_windows"]

    mu = X[wid < n_init].mean(axis=0)
    s = np.where(X[wid < n_init].std(axis=0) < 1e-9, 1.0, X[wid < n_init].std(axis=0))
    Xs = (X - mu) / s

    m = create_base_ensemble(seed=42)
    m.fit(Xs[wid < n_init], y[wid < n_init])
    cols = list(m.classes_)

    occurrences = {}
    for w in range(n_init, sd["total_windows"]):
        r = reg[wid == w][0]
        occurrences.setdefault(r, []).append(np.where(wid == w)[0])

    cents = np.array([Xs[np.concatenate(v)].mean(axis=0) for v in occurrences.values()])
    spread = float(np.mean(np.linalg.norm(cents - cents.mean(axis=0), axis=1)))

    dP, dAUC, dCen = [], [], []
    for r, occs in occurrences.items():
        if len(occs) < 2:
            continue
        first = occs[0]
        for idx in occs[1:]:
            yv, yf = y[idx], y[first]
            dP.append(float(np.mean(yv == positive) - np.mean(yf == positive)))
            dCen.append(float(np.linalg.norm(Xs[idx].mean(axis=0) - Xs[first].mean(axis=0))))
            if len(np.unique(yv)) < 2 or len(np.unique(yf)) < 2:
                continue
            a_r = auc(yv == positive, m.predict_proba(Xs[idx])[:, cols.index(positive)])
            a_f = auc(yf == positive, m.predict_proba(Xs[first])[:, cols.index(positive)])
            if not (np.isnan(a_r) or np.isnan(a_f)):
                dAUC.append(a_r - a_f)
    return np.array(dP), np.array(dAUC), np.array(dCen), spread


print("E3b -- conditional vs covariate change on regime recurrence\n")
print(f"{'Stream':8s} {'n':>5s} {'|dP(Y=1)|':>10s} {'|dAUC|':>8s} {'dCentroid':>10s} "
      f"{'between-regime':>15s} {'covariate ratio':>16s}")
for name, loader, pos in [("Campus", LOADERS["5G Campus QoS"], 2),
                          ("UGR16", LOADERS["UGR'16"], 1),
                          ("Nordic", LOADERS["NordicDat"], 2)]:
    dP, dAUC, dCen, spread = analyse(loader, pos)
    n = min(len(dP), len(dAUC))
    print(f"{name:8s} {n:5d} {np.mean(np.abs(dP)):10.4f} {np.mean(np.abs(dAUC)):8.4f} "
          f"{dCen.mean():10.4f} {spread:15.4f} {dCen.mean()/spread:16.3f}")

print("\nReading: a low covariate ratio with a large |dP| and |dAUC| means the")
print("regime looks similar but its label relationship has changed.")

# full paired detail for the stream that matters
print("\nUGR'16 paired detail (first occurrence -> each recurrence):")
dP, dAUC, dCen, spread = analyse(LOADERS["UGR'16"], 1)
for i in range(len(dP)):
    da = f"{dAUC[i]:+.4f}" if i < len(dAUC) else "   n/a"
    print(f"  pair {i+1}: dP={dP[i]:+.4f}  dAUC={da}  dCentroid={dCen[i]:.4f}")

