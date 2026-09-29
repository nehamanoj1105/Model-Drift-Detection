# Paste the complete CrewAI Python code from the uploaded script here.
"""
CrewAI pipeline for Experiment 2: Probability-Guided Regime Transfer.
Clones/uses the Model-Drift-Detection repo, builds and validates the final
experiment, and writes paper-ready results to a dedicated output folder.
Run overnight: nohup python run_crew.py > crew_log.txt 2>&1 &
"""

import os
import subprocess
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import FileReadTool, FileWriterTool, DirectoryReadTool
from crewai.tools import BaseTool

# ----------------------------------------------------------------------------
# FILL THESE IN
# ----------------------------------------------------------------------------
LOCAL_CLONE_DIR = r"C:\Users\emhaenn\Downloads\Model-Drift-Detection"  # if already present locally, this is used as-is; if missing, it's cloned here
REPO_URL = "https://github.com/nehamanoj1105/Model-Drift-Detection.git"
INSECTS_PATH = ""  # path to an INSECTS incremental-reoccurring CSV, or blank to skip S3
LLM_MODEL = "ollama/qwen3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

local_llm = LLM(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
)

# ----------------------------------------------------------------------------
# TOOLS
# ----------------------------------------------------------------------------
read_tool = FileReadTool()
write_tool = FileWriterTool()


class RunShellTool(BaseTool):
    name: str = "run_shell"
    description: str = (
        "Runs a shell command as a BLOCKING subprocess and returns return code, "
        "stdout (tail) and stderr (tail). Use for git clone/pull, pip install, "
        "and `python <script> <args>`. This is the ONLY way to execute anything — "
        "write .py files with the file-writer tool, then run them with this tool. "
        "Use a generous timeout_sec (up to 21600 for the full experiment run)."
    )

    def _run(self, command: str, cwd: str = LOCAL_CLONE_DIR, timeout_sec: int = 3600) -> str:
        try:
            result = subprocess.run(
                command, shell=True, cwd=cwd,
                capture_output=True, text=True, timeout=timeout_sec
            )
            return (
                f"CWD: {cwd}\nCMD: {command}\nRETURN CODE: {result.returncode}\n"
                f"STDOUT (tail):\n{result.stdout[-8000:]}\n"
                f"STDERR (tail):\n{result.stderr[-4000:]}"
            )
        except subprocess.TimeoutExpired as e:
            return f"TIMEOUT after {timeout_sec}s. Partial stdout:\n{(e.stdout or '')[-8000:]}"


shell_tool = RunShellTool()
ALL_TOOLS = [read_tool, write_tool, shell_tool]

# ----------------------------------------------------------------------------
# SHARED CONTEXT
# ----------------------------------------------------------------------------
MISSION_CONTEXT = f"""
REPO: {REPO_URL}
LOCAL PATH: {LOCAL_CLONE_DIR}  — if this already exists, use it and `git pull`
inside it first; if it doesn't exist, `git clone {REPO_URL} "{LOCAL_CLONE_DIR}"`.

KNOWN REPO LAYOUT (from prior sessions — VERIFY, do not trust blindly, see
Task 1): the real project root for this experiment is NOT the repo's
top-level `experiments/` folder. It is:

    {LOCAL_CLONE_DIR}\\Ensemble Learning for Model Drift Detection\\

Inside THAT folder there should be an `experiments/` directory containing:
    experiments/exp2_corrected/   — a prior, BUGGY implementation. Read-only
                                     reference for what NOT to repeat. Do not
                                     build on top of it or copy its code.
    experiments/exp2_final/       — partial prior work-in-progress on this
                                     exact experiment (audit/phase0_findings.md,
                                     run_phase1_study.py, results/). You may
                                     read audit/phase0_findings.md for
                                     background, but do NOT trust or reuse any
                                     of its cached results or its estimator/
                                     labeling code without re-verifying it —
                                     it had a known leak (local-retrain scored
                                     ~0.70 macro F1 on pure noise).
    experiments/final_validation/ — the ORIGINAL reference implementation
                                     ("Exp1"): the base ensemble class
                                     (HeterogeneousBaseEnsemble),
                                     StreamingPreprocessor, the Event-Driven
                                     drift detector, RAPT-E, and
                                     final_summary.csv with reference F1
                                     numbers. Vendor this code unchanged.
    experiments/exp9b/data/processed_exp9b_stream.csv — the real S2 dataset
                                     (499 rows, 24 columns, 5G NR latency).
    experiments/exp9/              — legacy, read-only reference only.

DECOY / DO NOT USE unless Task 1 proves otherwise:
    {LOCAL_CLONE_DIR}\\experiments\\                         (top-level, separate from the one above)
    {LOCAL_CLONE_DIR}\\Ensemble Learning for Model Drift Drift Detection\\   (note the doubled "Drift" — likely a stray duplicate)
These two look like duplicates or decoys. Task 1 must confirm which
directory tree actually contains exp2_corrected, exp9b's CSV, and
final_validation/final_summary.csv, and use ONLY that tree from then on.
If the top-level `experiments/` folder turns out to be the real one instead,
say so explicitly in recon.md and everyone downstream uses that path instead.

WORKING ROOT for all new code and outputs (once confirmed):
    <verified project root>\\experiments\\exp2_build\\        — all new pipeline code and run logs go here, a FRESH folder, not exp2_final
FINAL PAPER DELIVERABLE ROOT (separate, as the user asked for):
    {LOCAL_CLONE_DIR}\\paper_package\\                        — the ONLY folder whose contents are meant for the paper. Nothing else should be treated as a deliverable.

RESEARCH IDEA: at each decision point in a data stream, several historical
model checkpoints (one per past regime) are candidate policies. Being
similar to the current data does not mean a checkpoint will actually help
if transferred. An online logistic regression predicts P(positive transfer)
from distributional-similarity and historical-performance features, and the
system transfers only when P clears a threshold, otherwise it falls back to
locally retraining. The paper needs honest evidence for or against this,
including negative results if that is what the data shows.

KNOWN PAST BUGS IN exp2_corrected AND exp2_final — DO NOT REPEAT THESE:
- outcomes recorded after the run instead of online (falsified rolling-origin claims)
- an "Oracle" baseline that never updated and stayed equal to the frozen model
- a "weighted" method that was actually an argmax, identical to a simpler baseline
- a probability-guided method that always abstained and was byte-identical to its fallback
- a console success message printed regardless of whether checks actually passed
- a validation threshold edited after seeing results
- a local-retraining baseline that scored ~0.70 macro F1 on PURE NOISE labels (a train/eval leak)
- an estimator AUROC of exactly 0.5 with a zero-width confidence interval (a dead, constant model)
- a hardcoded reference number used instead of running the real reference code
- an agent reporting results as "verified" without printing an actual value read from disk

GOLDEN RULE: a claim is only real if code just read it from a file on disk
and printed it. Never write "PASS", "verified", or "complete" in prose that
no script produced and that you did not just read back yourself.

INSECTS_PATH = "{INSECTS_PATH or '(not provided, skip S3)'}"
"""

# ----------------------------------------------------------------------------
# AGENTS
# ----------------------------------------------------------------------------

recon_engineer = Agent(
    role="Repo Recon Engineer",
    goal=(
        "Get the repo checked out locally, and definitively establish the "
        "real directory structure, resolving the top-level-vs-nested "
        "experiments/ ambiguity and the duplicate folder, before anyone "
        "writes a line of pipeline code."
    ),
    backstory=MISSION_CONTEXT,
    tools=ALL_TOOLS,
    llm=local_llm,
    allow_delegation=False,
    max_iter=30,
    verbose=True,
)

ml_engineer = Agent(
    role="Online Experiment Pipeline Engineer",
    goal=(
        "Build a leak-proof, cached, resumable pipeline implementing the "
        "exact protocol given, in experiments/exp2_build/, and run it end "
        "to end: smoke test first, then the full multi-seed run."
    ),
    backstory=MISSION_CONTEXT,
    tools=ALL_TOOLS,
    llm=local_llm,
    allow_delegation=False,
    max_iter=60,
    verbose=True,
)

qa_auditor = Agent(
    role="Independent Validation Auditor",
    goal=(
        "Trust nothing the engineer claims. Run every control below as "
        "executable code against files on disk, with explicit numeric "
        "thresholds. Fix genuine bugs yourself, up to 3 attempts per "
        "failing stream, logging each attempt. NEVER relax a threshold, "
        "change a label rule, or alter the estimator to make a result look "
        "better. If a stream still fails after 3 honest attempts, mark it "
        "INVALID and continue to the next stage automatically. Do not stop "
        "and wait for anyone."
    ),
    backstory=MISSION_CONTEXT,
    tools=ALL_TOOLS,
    llm=local_llm,
    allow_delegation=False,
    max_iter=60,
    verbose=True,
)

statistician = Agent(
    role="Experimental Statistician",
    goal=(
        "Compute every statistic and claim verdict from raw per-decision "
        "logs on disk, never from a summary another agent wrote in prose."
    ),
    backstory=MISSION_CONTEXT,
    tools=ALL_TOOLS,
    llm=local_llm,
    allow_delegation=False,
    max_iter=40,
    verbose=True,
)

report_writer = Agent(
    role="Paper Package Assembler",
    goal=(
        "Assemble everything the paper needs into paper_package/, using "
        "only files that exist on disk, with no invented numbers, no "
        "softened negative results, and every limitation disclosed."
    ),
    backstory=MISSION_CONTEXT,
    tools=ALL_TOOLS,
    llm=local_llm,
    allow_delegation=False,
    max_iter=20,
    verbose=True,
)

# ----------------------------------------------------------------------------
# TASKS
# ----------------------------------------------------------------------------

task1_recon = Task(
    description=f"""
1. If "{LOCAL_CLONE_DIR}" already exists and is a git repo, cd into it and
   `git pull`. Otherwise `git clone {REPO_URL} "{LOCAL_CLONE_DIR}"`.
2. Resolve the directory ambiguity. Check BOTH of these candidate roots for
   the presence of exp2_corrected/, exp9b/data/processed_exp9b_stream.csv,
   and final_validation/final_summary.csv:
     a) "{LOCAL_CLONE_DIR}\\Ensemble Learning for Model Drift Detection\\experiments\\"
     b) "{LOCAL_CLONE_DIR}\\experiments\\"
   Also list the contents of "{LOCAL_CLONE_DIR}\\Ensemble Learning for Model
   Drift Drift Detection\\" (note the doubled "Drift") to check whether it
   is a stray duplicate or contains something unique. Write your finding
   (which root is real, with proof: exact file listing and sizes) into
   recon.md at "{LOCAL_CLONE_DIR}\\recon.md". This decision governs every
   path every other agent uses from here on — get it right.
3. Inside the confirmed real root's experiments/final_validation/, identify
   the exact classes and functions for the base ensemble
   (HeterogeneousBaseEnsemble), StreamingPreprocessor, the Event-Driven
   drift detector, and RAPT-E, plus final_summary.csv's reference numbers.
   Copy these source files UNCHANGED into
   <real root>/experiments/exp2_build/vendor/exp1/, and record SHA-256 of
   each copied file.
4. Verify experiments/exp9b/data/processed_exp9b_stream.csv: 499 rows, 24
   columns, record its SHA-256.
5. Read experiments/exp2_final/audit/phase0_findings.md if present, purely
   for historical context on what was already tried and what failed. Note
   in recon.md that its cached results and estimator code are NOT to be
   reused without independent verification.
6. pip install river if needed; fetch river.datasets.Elec2() (45,312 rows),
   cache under experiments/exp2_build/data/elec2_cache/, record SHA-256.
7. If INSECTS_PATH is non-empty, verify it exists, record SHA-256 and row
   count. If empty or missing, say clearly in recon.md that S3 is skipped.
8. Create the output skeleton: <real root>/experiments/exp2_build/ (working
   folder) and "{LOCAL_CLONE_DIR}\\paper_package\\" (final deliverable
   folder, empty for now).
Write config_hashes.txt next to recon.md listing every hash.
""",
    expected_output=(
        "recon.md exists at the repo root, explicitly stating which "
        "directory tree is real (with proof), listing every vendored file "
        "and dataset with its SHA-256, confirming exp2_build/ and "
        "paper_package/ now exist. Report the exact resolved paths back to "
        "me by reading recon.md, not from memory."
    ),
    agent=recon_engineer,
)

task2_build = Task(
    description="""
Work only inside <real root>/experiments/exp2_build/ (the root confirmed in
recon.md). Write README.md (design decisions) and config/*.json (frozen
parameters) BEFORE writing any pipeline code, and hash them into
config_hashes.txt.

STREAMS (window = fixed block of consecutive instances; regimes are
fixed-width blocks decided in advance, NEVER derived from any method's
output):
- S1 (synthetic recurring control): window=20 instances, >=6000 windows, 60
  segments of 100 windows. 10 features, 2 classes. 4 base concepts C0..C3
  cycling C0,C1,C2,C3,C0,... Each concept has a covariate mean shift and a
  linear label boundary. Segment types: 60% exact recurrence, 20% partial
  recurrence (boundary = weighted mix of two concepts), 20% decoy
  (covariates match concept i, labels follow concept j, so nearest-by-
  similarity is deliberately wrong). 5% label noise. Tune ONLY covariate-
  shift scale and boundary separation, max 4 attempts, logged, using
  baseline/label statistics only, NEVER the probability-guided method's
  results, until: local retraining beats frozen by >=0.02, split-half label
  agreement >=0.75 and Cohen's kappa >=0.30, >=50% of decisions have a
  significant positive candidate, nearest-by-similarity candidate is
  significantly positive in 35-75% of decisions. If nothing qualifies after
  4 tries, keep the best attempt and flag S1 as failing its own positive-
  control bar.
- N1 (pure noise): same covariates as S1, labels are independent coin
  flips. Nothing here is learnable — a canary stream.
- N2 (fresh concepts): like S1 but every segment introduces a brand-new,
  never-repeated concept, so history is useless but local retraining works.
- S2 (real, 9B, from vendor/exp1): Exp1 protocol exactly (n_init=99,
  StreamingPreprocessor). Candidates = one checkpoint per closed regime
  segment in the file.
- S4 (real, Elec2, tier 2): window=50 instances, regime block=20 windows
  (1000 instances), warmup=20 windows, K=5 windows.
- S3 (real, INSECTS, tier 2, optional only if the file was found in Task 1).

MANDATORY TWO-REAL-DATASET VALIDATION: S2 (real 5G NR latency) AND S4
(real Elec2) must both be run and represented in the final results. S1/N1/N2
are controls and do not substitute for S2/S4.

DECISION POINTS: every K windows, K EQUALS the shadow horizon H, so
decision intervals never overlap. Choose K per stream so each interval has
>=300 labeled instances (>=200 for S4/S3), and each stream supports >=100
decisions per seed where the data allows.

CANDIDATES: when a regime block closes, fit one checkpoint on X_init plus
that block ("Policy B" — every model-fitting method gets X_init plus its
own data, so none is data-starved relative to another). Cache that
checkpoint's predictions on every later window ONCE — never refit to score
a later interval. All F1 numbers come from cached per-window confusion
counts, never from re-running a model.

LOCAL-RETRAIN baseline: at every decision, refit on X_init plus the most
recent labeled windows (same length as one regime block). This is both a
baseline row and the fallback whenever a transfer method abstains.

LEAK-PROOF LABELS: for each decision and each available candidate
(available only if its block closed strictly before this decision), take
interval I_t of the next K windows. Using cached confusion counts, run a
paired bootstrap (200 resamples over windows in I_t, same resample indices
for candidate and local) of dF1 = macroF1(candidate) - macroF1(local).
Positive if the 90% CI lower bound is above 0, negative if the upper bound
is below 0, else neutral. Also compute a fixed-margin sensitivity label at
eps in {0, 0.005, 0.01, 0.02}. Report split-half label agreement (odd vs
even windows within each interval) as raw agreement AND Cohen's kappa, for
every stream.

LEAK-PROOF FEATURES (only information available before I_t; write the list
to config/features.json and add a unit test asserting no feature is a
function of I_t's outcome): Wasserstein distance, MMD, cosine similarity
(recent pre-decision data vs candidate's training block, on <=200-point
subsamples), candidate's own held-out F1 from creation, age in blocks, pool
size, local model's realized F1 on the previous interval, each distance
feature minus the minimum across candidates at that decision.

ESTIMATOR: online logistic regression, running standardization, balanced
class weights, refit at each decision using only decisions strictly before
it. If history has <50 rows or one class, output the historical positive
rate and mark unfitted. Add Platt calibration once >=200 stored out-of-
sample scores with both classes exist. THRESHOLD tau chosen ONLINE at each
decision from grid {0.20..0.90 step 0.05, plus always-abstain}, maximizing
mean realized dF1 of what would have been accepted over strictly earlier
decisions (needs >=20 prior decisions, else abstain). Freeze feature list
and regularization in config, hash before the test region runs.

METHODS (main table, 8, identical decisions per stream): Frozen;
Event-Driven (vendored Exp1 detector, tune sensitivity ONLY on S1 baseline
if it never fires, log trigger counts per seed); Local-Retrain;
Similarity-Only (always transfers nearest candidate); Similarity-Weighted
(softmax over negative distance, temperature chosen on dev split = first
30% of post-warmup decisions, real soft blend of candidate probabilities,
not an argmax); Hist-Reliability (always transfers best historical F1);
Probability-Guided (transfer argmax-P candidate if P>=tau, else
Local-Retrain); Oracle (per decision, best of all candidates and local by
realized interval F1, must never score below any other method on any
single decision).
Appendix only: Random-Historical; Shadow-Best (transfers whichever
candidate had the best F1 over the PREVIOUS interval); RAPT-E and RAPT-E+
(vendored, S2 only, reproduction reference).

Write a smoke-test runner (2 seeds, S1 truncated to ~1500 windows) and RUN
it. It must complete without error and print real numbers (not "done") for
every stream and method before this task is finished.
""",
    expected_output=(
        "README.md, config/*.json with hashes, a working pipeline (streams, "
        "candidate caching, labeling, features, estimator, all methods), "
        "and a completed smoke-test run whose output you have read back and "
        "can quote real per-method F1 numbers from for S1, N1, N2, S2 and S4. The smoke test must exercise both real datasets S2 and S4."
    ),
    agent=ml_engineer,
    context=[task1_recon],
)

task3_controls_smoke = Task(
    description="""
Run every control below AS CODE against the smoke-test output. Write
gate_results_smoke.json (machine-written, not your prose) and
controls_table.md (generated by a script that reads gate_results_smoke.json)
into experiments/exp2_build/audit/.

K1 (threshold: [0.45, 0.55]) on N1, every model-fitting method (frozen,
event-driven, local, similarity-only, similarity-weighted, hist-
reliability, probability-guided) must land in macro F1 [0.45, 0.55]. If
local-retrain scores well above 0.5 on pure noise, that IS a train/eval
leak: find and fix it before anything else. Do not proceed past this check.
K2 train/eval disjointness: log every fit's training and evaluation index
ranges, assert with a unit test that training always precedes evaluation.
K3 leakage replay: randomize every label/outcome from each decision t
onward, recompute through the exact online code path, assert bit-identical
results to the original.
K4 canary (proves K3 works): copy the estimator, add ONE feature equal to
the real outcome of the interval being predicted. K3's randomization test
on this copy MUST fail, and AUROC on it must exceed 0.95 (threshold). If
the canary doesn't fail, the leakage test itself is broken: fix the test.
K5 (threshold: score std > 1e-6) dead-estimator check: only compute AUROC
if score std exceeds 1e-6; a zero-width CI is flagged, not a pass. CIs are
block bootstrap over regime segments, 1000 resamples.
K6 identical-prediction matrix: pairwise byte-identity check across all
methods' per-window predictions, per stream. Every identical pair needs a
specific written reason.
K7 Oracle dominance: per decision and per stream, Oracle's interval F1
must be >= every other method's. Explain any exception.
K8 seed sanity: std across seeds nonzero for at least one method per
stream; rerunning one seed twice gives bit-identical results.
K9 (threshold: 0.005 macro F1) real reproduction, S2 only: run the
VENDORED unmodified Exp1 code on the exact S2 file (n_init=99), write
reference/exp1_s2.json. Compare frozen/event-driven/RAPT-E to it within
0.005 macro F1. No hardcoded reference numbers anywhere. If it fails, mark
S2 baselines NOT REPRODUCED, do not widen the tolerance.
K10 null-stream sanity: on N1 and N2, estimator AUROC's 95% CI must
include 0.5, probability-guided's F1 minus local's must have a paired CI
including 0 or average under 0.005. On N2, local must beat frozen by
>=0.10.
K11 no NaN or inf anywhere in a reported metric.
K12 (tolerance: 1e-9) consistency: recompute every method's macro F1 from
raw per-window predictions, assert it matches the cached per-interval
aggregation.

For any FAILURE that is a genuine implementation bug, fix it yourself, up
to 3 attempts per failing stream, logging each attempt with before/after
numbers in changelog.md. Never fix a failure by loosening a threshold,
changing a label rule, changing tau's rule, or altering what counts as a
feature. If a stream still fails after 3 honest attempts, mark it INVALID
in gate_results_smoke.json and CONTINUE automatically to the next task. Do
not stop and wait for review from anyone.
""",
    expected_output=(
        "gate_results_smoke.json and controls_table.md exist, generated by "
        "code, listing K1 through K12 with pass/fail/invalid and the actual "
        "numbers behind each, plus changelog.md documenting any fixes and "
        "confirming the canary (K4) actually fails as designed."
    ),
    agent=qa_auditor,
    context=[task2_build],
)

task4_full_run = Task(
    description="""
Proceed automatically once K1, K2, K3, K4, K9, K11, K12 show pass or
documented INVALID in gate_results_smoke.json (K5-K8, K10 get re-checked on
full data). Do not wait for anyone's approval.

Time-box: at most 4 hours from now. Time one full stream-seed job per
stream, then pick seeds = min(10, however many fit in remaining time with
25% slack), never below 6. Run order: S1, N1, N2, S2, then S4, then S3 (if
its file was found). If time runs short, cut in this order and say so in
status.md: 1) drop S3, 2) drop appendix-only methods, 3) cut seeds to 6.
NEVER drop S2 or S4: both real datasets are mandatory validation targets.
Never drop controls, S1, N1, N2, S2, or S4, and never skip writing whatever
output exists.

Make the run resumable: write each stream-seed job's result to disk the
moment it finishes, support skipping already-finished jobs on restart.
Store raw per-decision, per-method logs (not just summaries) under
experiments/exp2_build/runs/final/. Update status.md after every stream
with elapsed time.
""",
    expected_output=(
        "experiments/exp2_build/runs/final/ contains raw per-decision, "
        "per-method logs for every stream and seed that completed, plus a "
        "summary.json generated by reading those logs back, plus status.md "
        "showing what ran, what was cut, and total elapsed time."
    ),
    agent=ml_engineer,
    context=[task3_controls_smoke],
)

task5_controls_full = Task(
    description="""
Re-run K1, K5, K6, K7, K8, K9, K10, K11, K12 (same thresholds as before)
against the FULL run's raw logs, not the smoke test. Explicitly verify that
both mandatory real datasets S2 and S4 have complete raw logs before accepting
the final validation stage. Write
gate_results_final.json and update controls_table.md with full-data
numbers. If a stream fails a control here that passed on the smoke test,
you get ONE fix attempt per stream, logged in changelog.md; if it still
fails, mark that stream INVALID for claims and state exactly which claims
that removes. Continue automatically to the next task regardless of
outcome, as long as the failure is documented.
""",
    expected_output=(
        "gate_results_final.json and an updated controls_table.md, both "
        "generated by code reading runs/final/, with any INVALID stream "
        "clearly justified."
    ),
    agent=qa_auditor,
    context=[task4_full_run],
)

task6_stats = Task(
    description="""
Before computing paper claims, enforce the TWO-REAL-DATASET GATE: S2 and S4
must both have complete raw per-decision/per-method logs and valid full-run
control results. If either is absent or invalid, mark the two-dataset
validation incomplete.

Using ONLY runs/final/ raw logs and gate_results_final.json, compute
everything below and write outputs into experiments/exp2_build/results/:

Per stream/method: mean, std, 95% CI of test-region (last 70% of
post-warmup decisions) macro F1, via segment bootstrap (resample regime
segments with replacement, pool confusion counts, recompute F1, 1000
resamples, same resample indices across methods so differences are paired).

Paired bootstrap p-values for: probability-guided vs local-retrain, vs
similarity-only, vs hist-reliability. Also Wilcoxon signed-rank across
seeds. Holm correction across all comparisons and streams.

Probability quality per eligible stream (>=100 decisions, >=10 segments,
split-half agreement >=0.75, kappa >=0.30): AUROC with CI, AUPRC lift over
base rate, Brier skill vs base rate, ECE (15 equal-mass bins), single-
feature AUROC baselines (similarity alone, historical F1 alone).

Transfer rate, abstention rate, negative-transfer rate among executed
transfers for probability-guided vs similarity-only.

Admission per stream (local beats frozen by >=0.02, local F1 >=0.70,
frozen F1 <=0.95), reported for every stream regardless.

CLAIMS, by code, as PASS / FAIL / NOT_TESTABLE:
C1: probability-guided beats similarity-only AND hist-reliability by
>=0.005 macro F1, Holm p<0.05, on >=2 admitted REAL streams, never
significantly worse than local-retrain anywhere.
C2: negative-transfer rate at least 50% lower than similarity-only, paired
CI excludes 0. NOT_TESTABLE if fewer than 10 transfers executed.
C3: on >=2 probability-eligible streams (S1 may count as one, at least one
must be real), AUROC CI lower bound >0.60 and above best single-feature
AUROC, Brier skill >0, ECE <=0.10.
C4: probability-guided never worse than frozen by more than 0.002 macro F1
on any stream.
If fewer than 2 real streams are admitted, mark C1 and C3 NOT_TESTABLE.
For this experiment, S2 and S4 are the required two real datasets; do not count
S1, N1, or N2 toward the two-real-dataset requirement.

Produce exactly 5 figures as PDFs (main F1-by-method-and-stream with CIs;
reliability diagrams; risk-coverage curve over the tau grid; dF1
distribution of executed transfers, probability-guided vs similarity-only;
feature-ablation bar chart), plus paper_tables.md and tables.tex
(booktabs), plus claims_evidence.md tying each claim to exact numbers and
file paths.
""",
    expected_output=(
        "results/figures/*.pdf (exactly 5), results/paper_tables.md, "
        "results/tables.tex, and results/claims_evidence.md, all generated "
        "by a script whose output you read back from disk, with C1-C4 "
        "verdicts and the numbers behind each."
    ),
    agent=statistician,
    context=[task5_controls_full],
)

task7_package = Task(
    description="""
Assemble the FINAL PAPER PACKAGE at "{repo}\\paper_package\\" using ONLY
files that exist on disk under experiments/exp2_build/audit/, results/, and
figures/. Do not add any number you have not just read from a file. Copy
(not move) the finished artifacts there:
- paper_package/RESULTS_REPORT.md — controls table verbatim, main results
  per stream (mean, std, CI, n seeds actually used), probability quality
  table, claims C1-C4 with one-line evidence pointers, any INVALID or
  skipped streams and why, elapsed time per stage, and a plain 10-line
  verdict in your own words on what the evidence actually supports. If
  nothing beats local-retrain, say so in the first line, do not soften it.
- paper_package/LIMITATIONS.md — S1 is a planted-mechanism control, not
  evidence about real-world data; regimes are fixed blocks decided in
  advance; K equals H so decisions don't overlap; every model-fitting
  method uses the same training-data policy; tau is chosen online; S2 has
  very few decisions; any streams skipped, cut, or marked invalid; any
  config changes made after the first run (from changelog.md).
- paper_package/figures/ — the 5 PDFs.
- paper_package/tables.tex and paper_package/paper_tables.md.
- paper_package/audit/ — controls_table.md, gate_results_final.json,
  changelog.md, config_hashes.txt, claims_evidence.md, reproduce.md (exact
  commands to regenerate everything from scratch).

Your final message must contain ONLY the absolute path and SHA-256 of
paper_package/RESULTS_REPORT.md. No "PASS" or "verified" language in your
own prose.
""".format(repo=LOCAL_CLONE_DIR),
    expected_output=(
        "paper_package/ exists at the repo root with everything listed "
        "above; your final message is just RESULTS_REPORT.md's path and "
        "SHA-256."
    ),
    agent=report_writer,
    context=[task6_stats],
)

# ----------------------------------------------------------------------------
# LOCAL OLLAMA SANITY CHECK
# ----------------------------------------------------------------------------
def check_local_ollama():
    import urllib.request
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) as r:
            payload = r.read().decode("utf-8")
        if r.status != 200 or "qwen3:8b" not in payload:
            raise RuntimeError("qwen3:8b is not available")
    except Exception as exc:
        raise RuntimeError(
            "Ollama is not reachable or qwen3:8b is missing. "
            "Run `ollama run qwen3:8b` first. Details: " + str(exc)
        ) from exc

check_local_ollama()

# ----------------------------------------------------------------------------
# CREW
# ----------------------------------------------------------------------------
crew = Crew(
    agents=[recon_engineer, ml_engineer, qa_auditor, statistician, report_writer],
    tasks=[
        task1_recon, task2_build, task3_controls_smoke,
        task4_full_run, task5_controls_full, task6_stats, task7_package,
    ],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    result = crew.kickoff()
    print(result)