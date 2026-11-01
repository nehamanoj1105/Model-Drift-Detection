"""Validate the Experiment 9B final report tables against the 9B result CSVs.

Read-only. Prints PASS/FAIL per claim plus a summary.
"""
import csv
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R9B = os.path.join(ROOT, "results", "experiment_9b")
FAILS = []


def check(label, got, want, tol=5e-5):
    ok = abs(got - want) <= tol
    if not ok:
        FAILS.append(label)
    print(f"{'PASS' if ok else 'FAIL'}  {label:60s} got={got:<12.6g} report={want}")


def check_timing(label, got, want, rel=0.15):
    """Timing columns are not deterministic across runs (AGENTS.md); compare loosely."""
    ok = abs(got - want) <= rel * max(abs(want), 1e-9)
    if not ok:
        FAILS.append(label)
    print(f"{'PASS' if ok else 'FAIL'}  {label:60s} got={got:<12.6g} report={want} (rel<= {rel})")


def load(path):
    return list(csv.DictReader(open(os.path.join(R9B, path))))


def agg(path, key_cols, val_cols):
    out = {}
    for r in load(path):
        k = tuple(r[c] for c in key_cols)
        out.setdefault(k, {c: [] for c in val_cols})
        for c in val_cols:
            out[k][c].append(float(r[c]))
    return {k: {c: sum(v) / len(v) for c, v in d.items()} for k, d in out.items()}


# -------------------------------------------------------------------------
print("=" * 84)
print("TABLE 9B-1 natural drift (5G NR), from natural_drift/summary.csv")
print("=" * 84)
NAT = {r["method"]: r for r in load("natural_drift/summary.csv")}
T1 = {"Frozen": (0.8961, 0.9075), "Event-Driven": (0.8903, 0.9055),
      "Full Retraining": (0.9027, 0.9145), "RAPT": (0.8894, 0.9025),
      "RAPT-Enhanced": (0.8915, 0.9055)}
for m, (f1, acc) in T1.items():
    check(f"9B1 {m} F1", round(float(NAT[m]["macro_f1_mean"]), 4), f1)
    check(f"9B1 {m} Acc", round(float(NAT[m]["accuracy_mean"]), 4), acc)

# -------------------------------------------------------------------------
print("\n" + "=" * 84)
print("TABLE 9B-2 covariate drift (from covariate_drift/aggregated.csv)")
print("=" * 84)
COV = agg("covariate_drift/aggregated.csv", ["method", "drift_level"],
          ["macro_f1_mean", "accuracy_mean", "adaptation_cpu_sec_mean", "retrain_events_mean"])
T2 = {
    ("Frozen", "0.1"): (0.5265, 0.4900, 0.0000, 0.00),
    ("Frozen", "1.0"): (0.4893, 0.3700, 0.0000, 0.00),
    ("Event-Driven", "0.1"): (0.6819, 0.6000, 0.2362, 2.00),
    ("Full Retraining", "0.2"): (0.7894, 0.7770, 0.2473, 3.00),
    ("Full Retraining", "1.0"): (0.8080, 0.7940, 0.2471, 3.00),
    ("RAPT", "0.1"): (0.6854, 0.6140, 0.3350, 3.00),
    ("RAPT", "0.2"): (0.7944, 0.7880, 0.3334, 3.00),
    ("RAPT", "1.0"): (0.8122, 0.8000, 0.3386, 3.00),
    ("RAPT-Enhanced", "1.0"): (0.8122, 0.8000, 0.3310, 3.00),
}
for key, (f1, acc, cpu, retr) in T2.items():
    r = COV[key]
    check(f"9B2 {key[0]} {key[1]} F1", round(r["macro_f1_mean"], 4), f1)
    check(f"9B2 {key[0]} {key[1]} Acc", round(r["accuracy_mean"], 4), acc)
    check_timing(f"9B2 {key[0]} {key[1]} CPU", round(r["adaptation_cpu_sec_mean"], 4), cpu)
    check(f"9B2 {key[0]} {key[1]} Retr", round(r["retrain_events_mean"], 4), retr)

# -------------------------------------------------------------------------
print("\n" + "=" * 84)
print("TABLE 9B-3 concept drift (from concept_drift/aggregated.csv)")
print("=" * 84)
CON = agg("concept_drift/aggregated.csv", ["method", "drift_level"],
          ["macro_f1_mean", "accuracy_mean", "adaptation_cpu_sec_mean", "retrain_events_mean"])
T3 = {
    ("Frozen", "0.1"): (0.8777, 0.9725, 0.0000, 0.00),
    ("Frozen", "1.0"): (0.7916, 0.8627, 0.0000, 0.00),
    ("Event-Driven", "0.1"): (0.9241, 0.9804, 0.1568, 1.00),
    ("Event-Driven", "0.5"): (0.8274, 0.9098, 0.2251, 1.80),
    ("Full Retraining", "0.1"): (0.8777, 0.9725, 0.0000, 0.00),
    ("RAPT", "0.1"): (0.8777, 0.9725, 0.0790, 0.00),
    ("RAPT", "1.0"): (0.7916, 0.8627, 0.0784, 0.00),
    ("RAPT-Enhanced", "1.0"): (0.7916, 0.8627, 0.0782, 0.00),
}
for key, (f1, acc, cpu, retr) in T3.items():
    r = CON[key]
    check(f"9B3 {key[0]} {key[1]} F1", round(r["macro_f1_mean"], 4), f1)
    check(f"9B3 {key[0]} {key[1]} Acc", round(r["accuracy_mean"], 4), acc)
    check_timing(f"9B3 {key[0]} {key[1]} CPU", round(r["adaptation_cpu_sec_mean"], 4), cpu)
    check(f"9B3 {key[0]} {key[1]} Retr", round(r["retrain_events_mean"], 4), retr)

# -------------------------------------------------------------------------
print("\n" + "=" * 84)
print("TABLE 9B-4 recovery (from */recovery_aggregated.csv)")
print("=" * 84)
for path, key, want in [
    ("concept_drift/recovery_aggregated.csv", ("RAPT", "1.0"), (1.0000, 0.1667, 0.4483, 40.00, -0.5517)),
    ("covariate_drift/recovery_aggregated.csv", ("RAPT", "1.0"), None),
]:
    rows = load(path)
    cols = [c for c in rows[0] if c not in ("method", "model", "drift_level", "severity")]
    r = [x for x in rows if (x.get("method") or x.get("model")) == key[0]
         and x.get("drift_level") == key[1]]
    if r and want:
        r = r[0]
        for c, w in zip(["Pre_drift_F1", "Minimum_F1", "Recovery_F1", "Recovery_windows", "Delta_F1"], want):
            if c in r:
                check(f"9B4 {path.split('/')[0]} {key} {c}", round(float(r[c]), 4), w)

# -------------------------------------------------------------------------
print("\n" + "=" * 84)
print("TABLE 9B-5 RAPT reuse (from tables/table_9b5_rapt_reuse.csv)")
print("=" * 84)
R5 = {(r["Scenario"], r["Severity"]): r for r in load("tables/table_9b5_rapt_reuse.csv")}
for key, want in [(("natural", "-"), (6.0, 600.0, 400.0, 0.3831)),
                  (("covariate", "10%"), (0.0, 0.0, 400.0, 0.335)),
                  (("concept", "10%"), (0.0, 0.0, 100.0, 0.079))]:
    r = R5[key]
    for c, w in zip(["Reuse_events", "Reused_trees", "Retrained_trees", "Adapt_CPU_s"], want):
        fn = check_timing if c == "Adapt_CPU_s" else check
        fn(f"9B5 {key[0]} {key[1]} {c}", round(float(r[c]), 4), w, **({} if c == "Adapt_CPU_s" else {"tol": 5e-4}))

# -------------------------------------------------------------------------
print("\n" + "=" * 84)
print("Recurring concept drift (from recurring_concept_drift/aggregated.csv)")
print("=" * 84)
REC = agg("recurring_concept_drift/aggregated.csv", ["method", "drift_level"],
          ["macro_f1_mean", "accuracy_mean", "reused_checkpoints_mean"])
for key, want in [(("Frozen", "0.1"), (0.5632, 0.6525, 0.0)),
                  (("RAPT", "0.1"), (0.5632, 0.6525, 1.0)),
                  (("RAPT", "1.0"), (0.4890, 0.5675, 1.0)),
                  (("Event-Driven", "1.0"), (0.5919, 0.6100, 0.0)),
                  (("Full Retraining", "1.0"), (0.4968, 0.5775, 0.0))]:
    r = REC[key]
    check(f"9BD {key[0]} {key[1]} F1", round(r["macro_f1_mean"], 4), want[0])
    check(f"9BD {key[0]} {key[1]} Acc", round(r["accuracy_mean"], 4), want[1])
    check(f"9BD {key[0]} {key[1]} reuse", round(r["reused_checkpoints_mean"], 4), want[2])

# -------------------------------------------------------------------------
print("\n" + "=" * 84)
print("Concept-drift diagnostics (px_invariance / model_based_py_change)")
print("=" * 84)
px = load("concept_drift/px_invariance.csv")
check("concept: KS of every affected feature is 0",
      float(max(float(r["ks_plain_vs_transformed"]) for r in px)), 0.0)
mb = {r["drift_level"]: r for r in load("concept_drift/model_based_py_change.csv")}
for lvl, want in [("0.1", (1.0, 0.6533, -0.3467)), ("0.5", (1.0, 0.4786, -0.5214)),
                  ("1.0", (1.0, 0.4483, -0.5517))]:
    r = mb[lvl]
    check(f"concept P(Y|X) {lvl} f1_pre", round(float(r["f1_pre_holdout"]), 4), want[0])
    check(f"concept P(Y|X) {lvl} f1_post", round(float(r["f1_post_drift"]), 4), want[1])
    check(f"concept P(Y|X) {lvl} delta", round(float(r["delta_f1"]), 4), want[2])

# -------------------------------------------------------------------------
print("\n" + "=" * 84)
print("Covariate vs concept: P(X) preserved only for concept")
print("=" * 84)
cov = load("covariate_drift/aggregated.csv")
check("covariate changes P(X): Frozen F1 falls with severity",
      float(1.0 if float(agg("covariate_drift/aggregated.csv", ["method", "drift_level"],
                             ["macro_f1_mean"])[("Frozen", "1.0")]["macro_f1_mean"])
            < float(agg("covariate_drift/aggregated.csv", ["method", "drift_level"],
                        ["macro_f1_mean"])[("Frozen", "0.1")]["macro_f1_mean"]) else 0.0), 1.0)
check("concept preserves P(X) but drops F1",
      float(1.0 if (float(mb["1.0"]["f1_post_drift"]) < float(mb["1.0"]["f1_pre_holdout"])) else 0.0), 1.0)

print("\n" + "=" * 84)
print(f"RESULT: {len(FAILS)} failing claim(s)")
for f in FAILS:
    print("   FAIL:", f)
print("=" * 84)
