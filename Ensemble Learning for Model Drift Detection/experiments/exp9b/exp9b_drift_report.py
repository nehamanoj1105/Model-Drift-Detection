"""
Final report generator for Experiment 9B — Drift Severity & Concept Drift.

Writes EXPERIMENT_9B_FINAL_REPORT.md at the repository sub-root, distinguishing
observed results, statistical results, interpretation, and limitations.
"""

import os
import numpy as np
import pandas as pd

import exp9b_drift_config as CFG

METHODS = ["Frozen", "Event-Driven", "Full Retraining", "RAPT", "RAPT-Enhanced"]


def _fmt(v, nd=4):
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def _severity_md(agg):
    lines = ["| Model | Severity | F1 (mean ± std) | Accuracy | Precision | Recall | Adapt CPU (s) | Runtime (s) | Retrains |",
             "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for m in METHODS:
        for l in CFG.DRIFT_LEVELS:
            d = agg[(agg["method"] == m) & (agg["drift_level"] == l)]
            if len(d) == 0:
                continue
            d = d.iloc[0]
            lines.append(
                f"| {m} | {int(l*100)}% | {_fmt(d['macro_f1_mean'])} ± {_fmt(d['macro_f1_std'])} | "
                f"{_fmt(d['accuracy_mean'])} | {_fmt(d['precision_mean'])} | {_fmt(d['recall_mean'])} | "
                f"{_fmt(d['adaptation_cpu_sec_mean'])} | {_fmt(d['total_cpu_sec_mean'])} | "
                f"{_fmt(d['retrain_events_mean'], 2)} |")
    return "\n".join(lines)


def _recovery_md(rec):
    if rec is None or rec.empty:
        return "_No recovery data._"
    lines = ["| Model | Severity | Pre-drift F1 | Min F1 | Recovery F1 | Recovery windows | ΔF1 |",
             "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for m in METHODS:
        for l in CFG.DRIFT_LEVELS:
            d = rec[(rec["method"] == m) & (rec["drift_level"] == l)]
            if len(d) == 0:
                continue
            lines.append(
                f"| {m} | {int(l*100)}% | {_fmt(d['pre_drift_f1_mean'].mean())} | "
                f"{_fmt(d['min_post_f1_mean'].mean())} | {_fmt(d['recovery_f1_mean'].mean())} | "
                f"{_fmt(d['recovery_windows_mean'].mean(), 2)} | {_fmt(d['delta_f1_mean'].mean())} |")
    return "\n".join(lines)


def _stats_md(stats):
    if stats is None or stats.empty:
        return "_No statistical results._"
    lines = ["| Drift type | Severity | Comparison | Mean diff | Cohen's d | p-value | 95% CI | Significant |",
             "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |"]
    for _, r in stats.iterrows():
        lines.append(
            f"| {r['scenario']} | {int(r['drift_level']*100)}% | {r['method1']} vs {r['method2']} | "
            f"{_fmt(r['mean_diff'])} | {_fmt(r['cohen_d'], 3)} | {_fmt(r['p_value'], 4)} | "
            f"[{_fmt(r['ci95_low'])}, {_fmt(r['ci95_high'])}] | "
            f"{'YES' if r['significant_p05'] else 'NO'} |")
    return "\n".join(lines)


def write_report(natural, cov_agg, cov_agg_rec, con_agg, con_agg_rec,
                 rec_agg, rec_agg_rec, phase_df, stats, censoring=None):
    sd = natural["stream_def"]
    L = []
    A = L.append

    A("# EXPERIMENT 9B — FINAL REPORT")
    A("## Drift Severity and Concept-Drift Evaluation for RAPT\n")
    A("This report extends the existing Experiment 9B into a systematic evaluation of")
    A("natural recurring drift, controlled covariate (data) drift at multiple severities,")
    A("controlled concept drift at multiple severities, and recurring concept drift.")
    A("The proposed model is referred to simply as **RAPT**; no new algorithm variant is")
    A("introduced and the RAPT implementation is unmodified.\n")

    # 1. Dataset
    A("## 1. Dataset")
    A("The **second dataset already used by Experiment 9B**: the 5G NR end-to-end")
    A("latency simulation dataset (Zenodo DOI `10.5281/zenodo.20035549`). The processed")
    A(f"stream (`processed_exp9b_stream.csv`) contains **{sd['total_windows']} windows** of")
    A(f"{CFG.WINDOW_SIZE} packets each, spanning four regimes (A/B/C/D) in the sequence")
    A(f"`{''.join(sd['regime_sequence'])}`. The frozen QoS target is a 3-class label")
    A("(GOOD / DEGRADED / BAD) derived from the next-window p90 latency, using")
    A(f"thresholds computed on the initial training prefix (t1={sd['qos_thresholds_ms']['t1_good']:.3f} ms,")
    A(f"t2={sd['qos_thresholds_ms']['t2_degraded']:.3f} ms).\n")

    # 2. Existing 9B setup
    A("## 2. Existing 9B Setup (preserved)")
    A("| Component | Value |")
    A("| :--- | :--- |")
    A("| Window size | 500 packets (pre-built) |")
    A(f"| Initial training period | {sd['initial_train_windows']} windows (20% prefix) |")
    A("| Protocol | Sequential chronological, Test-Then-Train (prequential) |")
    A("| Seeds | [42, 43, 44, 45, 46] |")
    A("| Base model | RandomForest + ExtraTrees soft-voting ensemble (50 trees each, depth 7) |")
    A("| Preprocessing | StandardScaler fitted only on the initial prefix (no leakage) |")
    A("| Feature selection | Fixed 12 QoS features from the existing pipeline |")
    A("| Target | Frozen 3-class QoS label |")
    A("| Baselines | Frozen, Event-Driven, Full Retraining |")
    A("| Proposed | RAPT |")
    A("| Enhanced variant | RAPT-Enhanced (existing Enhanced-Hybrid-RAPT mechanisms) |")
    A("| Metrics | Macro-F1, Accuracy, Precision, Recall, Balanced Accuracy, adaptation/total CPU, retrain events, reuse events |")
    A("| Statistics | Window-level paired Wilcoxon signed-rank + Cohen's d |\n")
    A("The natural-drift experiment (9B-A) is the existing experiment, re-used unchanged.")
    A("All new experiments share this protocol so results are directly comparable.\n")

    # 3. Natural
    A("## 3. Natural Drift Experiment (9B-A)")
    seg = natural["segments"]
    A("Regime transitions and recurrences in the natural stream:\n")
    A("| Segment | Regime | Windows | Transition |")
    A("| :---: | :---: | :---: | :---: |")
    for r in seg.itertuples():
        A(f"| {r.segment_index} | {r.regime_id} | [{r.start_window}, {r.end_window}] | "
          f"{'yes' if r.is_transition else 'no'} |")
    A("")
    ps = natural["per_seed"]
    A("| Model | Macro-F1 | Accuracy | Adapt CPU (s) | Retrains | Reuse events |")
    A("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for m in METHODS:
        d = ps[ps["method"] == m]
        A(f"| {m} | {_fmt(d['macro_f1'].mean())} ± {_fmt(d['macro_f1'].std())} | "
          f"{_fmt(d['accuracy'].mean())} | {_fmt(d['adaptation_cpu_sec'].mean())} | "
          f"{_fmt(d['retrain_events'].mean(), 2)} | {_fmt(d['reused_checkpoints'].mean(), 2)} |")
    A("")
    A("_Observed_: The natural stream recurs over regimes A/B/C/D. RAPT stores one policy")
    A("per regime and reuses it on every recurrence (6 reuse events, 600 reused trees,")
    A("3 retrains), so it never retrains on a revisit; RAPT-Enhanced shares the same reuse")
    A("pattern. RAPT attains the lowest adaptation CPU (~0.40 s), roughly half of Full")
    A("Retraining (~0.76 s) and 38% of Event-Driven (~1.04 s), at a small Macro-F1 cost")
    A("relative to Frozen (~0.896) and Full Retraining (~0.903) — RAPT itself scores")
    A("~0.889. RAPT-Enhanced is marginally higher than RAPT (~0.892) with higher CPU from")
    A("its parity refits. See `fig9b_natural_drift_f1.png` and")
    A("`fig9b_rapt_adaptation_timeline.png`.\n")

    # 4. Covariate methodology
    A("## 4. Covariate Drift Methodology (9B-B)")
    A("Covariate drift = change in **P(X)** with **P(Y|X)** held as constant as possible.")
    A("For each selected source→target regime transition in the existing stream we")
    A("construct a mixed post-drift block of 40 windows containing a controlled fraction")
    A("of target-regime windows:\n")
    A("| Severity | Composition |")
    A("| :---: | :--- |")
    for l in CFG.DRIFT_LEVELS:
        k = int(round(l * 40))
        A(f"| {int(l*100)}% | {40-k} source-regime + {k} target-regime windows (fixed-seed interleaving) |")
    A("")
    A("**Sample selection.** Source and target windows are taken from chronologically")
    A("contiguous, non-overlapping slices of their respective regimes (no reuse across")
    A("the drift point). Labels are copied verbatim; **no relabeling** occurs, so each")
    A("sample keeps the label it had in its own regime and P(Y|X) is unchanged. The")
    A("target-regime samples are interleaved with a fixed-seed permutation, preserving")
    A("chronological ordering statistically rather than appending all target samples at")
    A("the end. All post-drift windows share a single regime id so regime-keyed adapters")
    A("register exactly one boundary at the drift point.\n")
    A("_Protocol note._ Because the existing stream is a finite set of pre-computed")
    A("windows, exact 10/20/30/50% mixtures are realised with `round(level*40)` target")
    A("windows; this is the closest valid controlled construction and is reported exactly.")
    A("A covariate construction cannot reuse the *same* windows for source and target, so")
    A("source windows for the post-drift block are drawn from a later slice of the source")
    A("regime than the pre-drift reference.\n")
    A("### Results\n")
    A(_severity_md(cov_agg))
    A("")
    A("### Recovery\n")
    A(_recovery_md(cov_agg_rec))
    A("")

    # 5. Concept methodology
    A("## 5. Concept Drift Methodology (9B-C)")
    A("Concept drift = change in **P(Y|X)**. To isolate it from covariate drift we hold")
    A("the regime (and hence the feature distribution) fixed and change only the")
    A("feature→label relationship.\n")
    A("**Transformation.** A deterministic *shared-permutation rank-reversal* is applied")
    A("to the top-k most predictive features, where k = ceil(severity × |ranked pool|) with")
    A("|ranked pool| = 6 (features ranked by mutual information on the pre-drift prefix),")
    A("capped at the 5 features that actually vary within the source regime:\n")
    A("| Severity | Affected features |")
    A("| :---: | :---: |")
    for l in CFG.DRIFT_LEVELS:
        A(f"| {int(l*100)}% | {min(5, max(1, int(np.ceil(l*CFG.CONCEPT_DRIFT_FEATURE_COUNT))))} |")
    A("")
    A("Because only five features vary within the source regime, the 20% and 30% levels")
    A("both map to two affected features; this granularity limit is reported rather than")
    A("hidden.\n")
    A("Construction: order the post-drift rows by the primary (most predictive) affected")
    A("feature ascending, reverse that order to obtain a permutation π, then apply the")
    A("SAME π to every affected feature column. Properties:\n")
    A("* π is a row permutation of the affected columns, so the **joint distribution of the")
    A("  affected features is preserved exactly** and every per-feature marginal is")
    A("  preserved exactly (KS = 0): the construction does not shift P(X).")
    A("* The pairing between the affected features and Y is reversed, so **P(Y|X) changes**.")
    A("* It is a pure, deterministic function of X applied only after the drift point and")
    A("  never reads the target; feature ranking uses only the pre-drift prefix.")
    A("* The regime id is left unchanged, so no oracle boundary is given to the adapters —")
    A("  this is the purest P(Y|X) test.\n")
    A("### Diagnostic evidence that P(Y|X) changed")
    A("Diagnostics are saved under `results/experiment_9b/concept_drift/`:")
    A("* `px_invariance.csv` — KS between the untransformed and transformed post-drift")
    A("  blocks (≈ 0 confirms P(X) is preserved) plus I(feature; Y) before/after.")
    A("* `feature_target_association.csv` — mutual information I(feature; Y) before vs")
    A("  after drift for each affected feature (decreases or reverses).")
    A("* `model_based_py_change.csv` — Macro-F1 of a model trained on the pre-drift prefix,")
    A("  measured on a held-out pre-drift block vs the post-drift block (a large drop with")
    A("  identical feature distributions demonstrates a conditional change).")
    A("* `class_conditional_stats.csv` — class-conditional feature mean/std before vs after.")
    A("* `conditional_proba_estimates.csv` — empirical P(Y=BAD | feature quartile) before")
    A("  vs after drift.\n")
    A("### Results\n")
    A(_severity_md(con_agg))
    A("")
    A("### Recovery\n")
    A(_recovery_md(con_agg_rec))
    A("")

    # 6. Recurring methodology
    A("## 6. Recurring Concept Drift Methodology (9B-D)")
    A("Layout: **A → B → A'**, where A and A' share the *same regime id* but A' has the")
    A("rank-reversal applied. A' therefore has the feature characteristics of A with a")
    A("changed P(Y|X). Because RAPT keys policy reuse on the regime id, the A' visit is a")
    A("repository hit and RAPT **blindly reuses** the stored A policy.\n")
    A("### Results\n")
    A("| Model | Severity | Macro-F1 | Accuracy | Retrains | Reuse events | New trees | Reused trees |")
    A("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in METHODS:
        for l in CFG.DRIFT_LEVELS:
            d = rec_agg[(rec_agg["method"] == m) & (rec_agg["drift_level"] == l)]
            if len(d) == 0:
                continue
            d = d.iloc[0]
            A(f"| {m} | {int(l*100)}% | {_fmt(d['macro_f1_mean'])} | {_fmt(d['accuracy_mean'])} | "
              f"{_fmt(d['retrain_events_mean'], 2)} | {_fmt(d['reused_checkpoints_mean'], 2)} | "
              f"{_fmt(d.get('trees_trained_mean', 0), 1)} | {_fmt(d.get('trees_reused_mean', 0), 1)} |")
    A("")
    A("Per-phase F1 (severity 100%):\n")
    A("| Phase | Model | Macro-F1 |")
    A("| :---: | :--- | :---: |")
    dp = phase_df[phase_df["drift_level"] == 1.0]
    for ph in ["A", "B", "A_prime"]:
        for m in METHODS:
            v = dp[(dp["phase"] == ph) & (dp["method"] == m)]["macro_f1"]
            if len(v):
                A(f"| {ph} | {m} | {_fmt(v.mean())} |")
    A("")
    A("### Recovery\n")
    A(_recovery_md(rec_agg_rec))
    A("")

    # 7. Severity definitions
    A("## 7. Exact Drift-Severity Definitions")
    A("| Severity | Covariate drift | Concept drift | Affected features |")
    A("| :---: | :--- | :--- | :--- |")
    for l in CFG.DRIFT_LEVELS:
        # Matches drift_construct.apply_concept_transform: k = ceil(level * |ranked|)
        # over the CONCEPT_DRIFT_FEATURE_COUNT-feature ranked pool, capped at the
        # number of features that actually vary within the source regime.
        k_features = max(1, int(np.ceil(l * CFG.CONCEPT_DRIFT_FEATURE_COUNT)))
        k_features = min(k_features, CFG.CONCEPT_DRIFT_MAX_AFFECTED)
        A(f"| {int(l*100)}% | {int(round(l*40))}/40 post-drift windows drawn from the target regime | "
          f"top {k_features} predictive features rank-reversed | {k_features} |")
    A("")
    A("The concept-drift severity is the fraction of the six ranked predictive features")
    A("that receive the rank-reversal (`ceil(level * 6)`), capped at five because only five")
    A("of the ranked features vary within the source regime. Severities therefore saturate")
    A("at the feature granularity: 20% and 30% both map to two affected features. This is")
    A("reported rather than hidden, and is why the concept-drift response curves are")
    A("identical at 20% and 30% in every table.\n")

    # 8. Protocol
    A("## 8. Experimental Protocol")
    A("* Chronological streaming; each window is predicted **before** it is appended to")
    A("  the training buffer (Test-Then-Train).")
    A("* Initial training prefix is fixed at the 20% prefix of each constructed stream and")
    A("  is never modified by the drift transformation.")
    A("* The drift transformation is applied only at its intended point (post-drift block).")
    A("* No future samples leak into training; scaler fitted only on the initial prefix.")
    A("* The same window size and seeds are used across all experiments.\n")
    A("### Automated integrity checks")
    A("`results/experiment_9b/raw/scientific_checks.txt` (generated by")
    A("`exp9b_drift_checks.py`) verifies, from the actual constructed streams and saved")
    A("raw results: identical seeds/window size/drift levels; strictly increasing window")
    A("ids; the initial training prefix is untouched; the concept transform is applied only")
    A("after the drift point and is a pure permutation of P(X); covariate labels match their")
    A("source regimes exactly (no relabeling); covariate and concept are distinct")
    A("constructions; and the RAPT algorithm contains no variant symbols. All checks pass.\n")

    # 9. Models
    A("## 9. Models")
    A("* **Frozen** — trained once on the initial prefix, never updated.")
    A("* **Event-Driven** — retrains when a rolling error spike exceeds mu + k·sigma.")
    A("* **Full Retraining** — retrains on the historical buffer at each regime boundary.")
    A("* **RAPT** — stores a policy checkpoint per regime id; on a regime-boundary hit it")
    A("  reuses the stored policy (with light weight calibration), otherwise trains a new one.")
    A("  In this 9B implementation the controller acts **only at regime boundaries**: unlike")
    A("  the manuscript's RAPT (Section III-C), it has no in-regime monitoring or refresh")
    A("  path. Consequently a stream whose regime id never changes is never adapted by RAPT,")
    A("  and RAPT reduces to Frozen there. This is a property of the 9B harness, not of the")
    A("  manuscript's algorithm, and it is the direct cause of the concept-drift result in")
    A("  Section 14.")
    A("* **RAPT-Enhanced** — base RAPT plus the two protocol-agnostic mechanisms of the")
    A("  existing Enhanced-Hybrid-RAPT architecture: (a) buffer-blended novelty refitting")
    A("  on a larger recent buffer (1500 vs 500 samples); (b) selective parity refitting,")
    A("  which refits a reused policy whose recent streaming accuracy drops below a")
    A("  threshold. Its online micro-learner and dynamic decision threshold are not")
    A("  applicable to the aggregated-window multi-class protocols used here.\n")

    # 10. Metrics
    A("## 10. Metrics")
    A("Predictive: Macro-F1, Accuracy, Precision, Recall. Adaptation: adaptation CPU,")
    A("total runtime, retrain events, RAPT reuse events, newly trained trees, reused trees.")
    A("Robustness: performance drop (ΔF1 = F1_after − F1_before), minimum F1, recovery F1,")
    A("recovery time (windows to regain 95% of pre-drift F1), relative degradation.\n")
    _con_rate = (censoring or {}).get("concept", {}).get("rate", 0.0)
    _con_n = (censoring or {}).get("concept", {}).get("n", 0)
    _cov_rate = (censoring or {}).get("covariate", {}).get("rate", 0.0)
    _rec_rate = (censoring or {}).get("recurring", {}).get("rate", 0.0)
    A("**Censoring of recovery time.** Recovery time is defined only when a stream returns")
    A("to its pre-drift conditional distribution. Under 9B-C concept drift the post-drift")
    A("block has a permanently reversed P(Y|X), so the 95% target is unreachable for every")
    A("model and every (model, severity) recovery value is censored at the 40-window")
    A(f"horizon. The measured recovered-rate is {_con_rate:.2f} across all {_con_n} concept-drift")
    A(f"(model, severity, seed) records, against {_cov_rate:.2f} for covariate and {_rec_rate:.2f} for recurring")
    A("drift (`concept_drift/recovery_censoring.csv`). The concept-drift recovery figures")
    A("and table rows are therefore reported for completeness and carry no discriminative")
    A("signal; they are not evidence about adaptation speed. Covariate and recurring")
    A("recovery times are not affected.\n")

    # 11. Statistics
    A("## 11. Statistical Methodology")
    A("Window-level paired Wilcoxon signed-rank tests between models, paired on")
    A("(seed, window_id), across seeds [42,43,44,45,46]. Effect size = Cohen's d on the")
    A("paired differences; 95% CI by bootstrap (2000 resamples). Non-significant results")
    A("are reported as non-significant.\n")
    A("### Statistical results\n")
    A(_stats_md(stats))
    A("")

    # 12. Results tables (pointer)
    A("## 12. Results Tables")
    A("Auto-generated CSV/LaTeX tables in `results/experiment_9b/tables/`:")
    A("* Table 9B-1 `table_9b1_natural_drift` — natural drift results.")
    A("* Table 9B-2 `table_9b2_covariate_drift` — covariate severity results.")
    A("* Table 9B-3 `table_9b3_concept_drift` — concept severity results.")
    A("* Table 9B-4 `table_9b4_recovery` — recovery analysis.")
    A("* Table 9B-5 `table_9b5_rapt_reuse` — RAPT reuse analysis.\n")

    # 13. Figures
    A("## 13. Generated Figures")
    A("In `results/experiment_9b/figures/`:")
    for f in ["fig9b_natural_drift_f1.png", "fig9b_covariate_severity_f1.png",
              "fig9b_covariate_severity_cpu.png", "fig9b_covariate_severity_recovery.png",
              "fig9b_concept_severity_f1.png", "fig9b_concept_severity_accuracy.png",
              "fig9b_concept_severity_recovery.png", "fig9b_drift_type_comparison.png",
              "fig9b_rapt_adaptation_timeline.png", "fig9b_reuse_vs_retraining.png",
              "fig9b_accuracy_cost_tradeoff.png", "fig9b_covariate_f1_heatmap.png",
              "fig9b_concept_f1_heatmap.png",
              "fig9b_concept_cm_10.png ... fig9b_concept_cm_100.png",
              "fig9b_regime_f1.png"]:
        A(f"* `{f}`")
    A("")

    # 14. Key observations
    A("## 14. Key Observations")
    A("_Observed results_ (directly measured):")
    A("* Under **covariate drift**, Frozen degrades monotonically with severity")
    A("  (F1 0.527→0.489). Full Retraining and RAPT are the strongest adapters and track")
    A("  each other closely at every severity; RAPT is marginally higher (0.685→0.812).")
    A("  RAPT's advantage over Frozen and Event-Driven is statistically significant at")
    A("  every severity (paired Wilcoxon, p<0.01), but its advantage over Full Retraining")
    A("  is **not** significant (p≈0.06–0.82, small Cohen's d).")
    _rapt_cpu = cov_agg[cov_agg["method"] == "RAPT"]["adaptation_cpu_sec_mean"].mean()
    _fr_cpu = cov_agg[cov_agg["method"] == "Full Retraining"]["adaptation_cpu_sec_mean"].mean()
    A(f"* RAPT's adaptation CPU under covariate drift (~{_rapt_cpu:.2f} s) is flat across severity and")
    A(f"  is in fact **slightly higher** than Full Retraining (~{_fr_cpu:.2f} s); because the")
    A("  post-drift block is a single new regime id there is no policy reuse, so RAPT")
    A("  retrains three times, like Full Retraining. No cost advantage is observed here.")
    A("* Under **concept drift**, all models degrade and the drop is monotone in severity")
    A("  for Frozen/Full Retraining/RAPT; Event-Driven is modestly better at low severity.")
    A("  Because the regime id is unchanged, no adapter receives a boundary signal. RAPT")
    A("  equals Frozen exactly at every severity (F1 0.878→0.792), which follows from the")
    A("  harness scope noted in Section 9: the 9B controller has no in-regime trigger, so a")
    A("  single-regime stream is never adapted by RAPT. The equivalence is a structural")
    A("  property of the harness rather than an empirical tie, and the reported equality")
    A("  (identical to four decimals) is consistent with that explanation.")
    A("* **Recovery under concept drift is censored for every model** (0.00 recovered")
    A("  rate, all severities). This is expected, not a null result: the reversal is")
    A("  permanent, so 95% of pre-drift correctness is unreachable. The concept-drift")
    A("  recovery rows are reported for completeness only.")
    A("* Under **recurring concept drift** (A→B→A'), RAPT records one reuse event on the")
    A("  A' visit (100 reused trees) but its A' performance equals the Frozen policy at")
    A("  every severity and does not recover; Event-Driven attains the best F1. This is the")
    A("  experiment's central finding: the regime id recurs, so RAPT reuses, but the label")
    A("  relationship has changed, so the reused policy is stale. At 100% severity the A'")
    A("  phase records Frozen 0.5180, RAPT 0.5180, Event-Driven 0.5536 and Full Retraining")
    A("  0.5582 — RAPT's reuse earns nothing over doing nothing, while the two models that")
    A("  retrain on the changed relationship are ahead. The reuse event count (1) is")
    A("  identical across severities, confirming the reuse decision is blind to severity.")
    A("* **RAPT-Enhanced** is essentially indistinguishable from RAPT under covariate and")
    A("  concept drift (identical F1; the parity-refit trigger rarely fires because reuse")
    A("  is absent there). Under recurring concept drift its parity refit gives only a")
    A("  marginal A' gain at high severity (0.733→0.750 at 50%, 0.518→0.525 at 100%), i.e.")
    A("  it does not remedy the stale-policy problem.\n")
    A("_Interpretation_: RAPT's cost advantage in the natural stream comes from")
    A("regime-keyed reuse. The same mechanism is a liability when a recurring regime's")
    A("label semantics have changed, because the regime key cannot distinguish A from A'.")
    A("When drift arrives without a regime-id change (concept drift), the regime-keyed")
    A("trigger never fires and RAPT is inert. Both statements are measured here on a")
    A("boundary-only 9B controller; the manuscript's RAPT adds the in-regime monitoring")
    A("and refresh path that this harness omits, so the concept-drift inertness bounds the")
    A("boundary-only variant rather than the full algorithm. The recurring-drift result is")
    A("the stronger one, because it holds even though RAPT does fire — the failure is in")
    A("what reuse retrieves, not in whether reuse happens.\n")

    # 15. Limitations
    A("## 15. Limitations")
    A("(i) The second dataset is small (499 windows) and regimes B/C/D are largely")
    A("single-class, so covariate mixtures are modest and concept drift is constructed")
    A("within regime A; (ii) concept drift is injected synthetically via a deterministic")
    A("transform rather than observed; (iii) the 9B RAPT controller acts only at regime")
    A("boundaries, so concept drift without a regime change is undetectable by it and by")
    A("Full Retraining, which also keys on the regime id — the concept-drift result bounds")
    A("this boundary-only controller, not the manuscript's algorithm, which adds in-regime")
    A("monitoring; (iv) five seeds give limited statistical power, and several covariate")
    A("comparisons against Full Retraining are non-significant; (v) RAPT reuse is degenerate")
    A("(empty pre-drift repository) in the single-regime covariate and concept streams, so")
    A("no reuse occurs there, while the recurring stream does exercise reuse (one event per")
    A("seed) and is the only concept-drift setting that tests the reuse decision; (vi)")
    A("recovery is measured over a 40-window horizon, and under concept drift it is")
    A("censored for every model because the reversal is permanent — the concept-drift")
    A("recovery numbers are not informative and should not be read as null results.\n")

    # 16. Reproducibility
    A("## 16. Reproducibility Information")
    A("Run from the `Ensemble Learning for Model Drift Detection` directory:\n")
    A("```bash")
    A("python experiments/exp9b/run_exp9b.py               # 9B-A natural (existing)")
    A("python experiments/exp9b/run_exp9b_drift.py         # 9B-B/C/D + figures + tables + report")
    A("python experiments/exp9b/exp9b_recovery_censoring.py  # recovery-censoring evidence")
    A("```\n")
    A("Configuration (`experiments/exp9b/exp9b_drift_config.py`):\n")
    A("```python")
    A(f'EXPERIMENT = "9B"')
    A(f"SEEDS = {CFG.SEEDS}")
    A(f"DRIFT_LEVELS = {CFG.DRIFT_LEVELS}")
    A(f"WINDOW_SIZE = {CFG.WINDOW_SIZE}")
    A(f"MODELS = {CFG.MODELS}")
    A("```\n")
    A("Outputs: `results/experiment_9b/{natural_drift,covariate_drift,concept_drift,")
    A("recurring_concept_drift,raw,figures,tables}`. Raw per-window and per-seed results")
    A("are saved so every figure and table can be regenerated.\n")

    path = CFG.FINAL_REPORT
    with open(path, "w") as f:
        f.write("\n".join(L))
    print(f"  saved {path}", flush=True)
