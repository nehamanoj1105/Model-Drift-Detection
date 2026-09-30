# V. EXPERIMENTAL RESULTS

This section reports the measured behaviour of RAPT and the compared models.
All values are taken from generated result files; the authoritative source of
each table and figure is recorded in `results_claims_check.md` and in the header
comments of `tables/final_results_tables.tex`. Where a comparison is described as
*not significant*, the corresponding test did not establish a difference at
$\alpha = 0.05$; this is not a claim that the models are equivalent.

The evaluation covers four data streams. Three are the 9A telecom streams
(5G Campus QoS, UGR'16, NordicDat) evaluated with five models
(Frozen, Event-Driven, Full Retraining, RAPT, RAPT-Enhanced). The fourth is the
9B 5G latency QoS stream, used both for the natural-drift evaluation and for the
controlled covariate-, concept- and recurring-concept-drift experiments at
severities $\{0.10, 0.20, 0.30, 0.50, 1.00\}$. The RAPT efficiency ablation was
additionally run on the 5G Campus QoS stream. All experiments use the same
prequential Test-Then-Train protocol, the same window size per stream, the same
initial training prefix, and seeds $[42, 43, 44, 45, 46]$.

---

## A. Overall Predictive Performance

Table~\ref{tab:results_main_performance} reports accuracy, Macro-F1, precision and
recall for the five models on each dataset.

On **5G Campus QoS**, Full Retraining recorded the highest Macro-F1
($0.9836 \pm 0.0013$), followed by Event-Driven ($0.9636 \pm 0.0006$). RAPT and
RAPT-Enhanced both recorded $0.9381 \pm 0.0016$, a difference of
$-0.0455$ Macro-F1 relative to Full Retraining. On this stream RAPT does not
improve predictive performance over Full Retraining; the measured difference is
negative and the paired test reports a significant difference
(Section~\ref{sec:stats}).

On **UGR'16** the ordering differs. Frozen recorded the highest Macro-F1
($0.9690 \pm 0.0029$) and Full Retraining $0.9595 \pm 0.0012$. RAPT recorded
$0.8360 \pm 0.0262$, which is $0.1236$ below Full Retraining, whereas
RAPT-Enhanced recorded $0.9276 \pm 0.0123$, recovering $0.0917$ of that gap. This
is the clearest case in which base RAPT underperforms the baselines, and it
coincides with the dataset that has the strongest natural regime recurrence
(168 recurrence events).

On **NordicDat**, a stream with a weak frozen baseline, Full Retraining recorded
the highest Macro-F1 ($0.4227 \pm 0.0299$), while RAPT recorded
$0.3818 \pm 0.0184$ ($-0.0409$) but the highest accuracy of any model
($0.6172 \pm 0.0127$ versus $0.5694 \pm 0.0075$ for Full Retraining). The
precision/recall split indicates that RAPT traded recall for precision on this
stream; the paired Macro-F1 comparison against Full Retraining was not
significant (Section~\ref{sec:stats}).

On the **9B 5G latency QoS** stream the five models are closely grouped:
Full Retraining $0.9027 \pm 0.0059$, Frozen $0.8961 \pm 0.0099$, RAPT-Enhanced
$0.8915 \pm 0.0117$, Event-Driven $0.8903 \pm 0.0107$ and RAPT
$0.8894 \pm 0.0110$. The spread between the highest and lowest Macro-F1 is
$0.0133$.

Three observations follow from Table~\ref{tab:results_main_performance}. First,
the rank order of the models is not consistent across datasets: Full Retraining
has the highest Macro-F1 on three of four streams, but on NordicDat RAPT has the
highest accuracy while Full Retraining has the highest Macro-F1. Second, base
RAPT does not improve predictive performance relative to Full Retraining on any
of the four streams; the measured Macro-F1 differences are $-0.0455$ (5G Campus
QoS), $-0.1236$ (UGR'16), $-0.0409$ (NordicDat) and $-0.0133$ (9B 5G latency
QoS). Third, RAPT-Enhanced narrows the gap on the two streams where the gap is
largest (UGR'16: $0.8360 \rightarrow 0.9276$) and leaves it essentially unchanged
on 5G Campus QoS ($0.9381 \rightarrow 0.9381$).

Fig.~\ref{fig:final_f1_bars} summarises the Macro-F1 comparison and
Fig.~\ref{fig:three_dataset_accuracy} the accuracy comparison across the three 9A
streams. Fig.~\ref{fig:cross_dataset_f1} places the 9A UGR'16 and 9B 5G latency
QoS results on a common axis for the head-to-head cross-dataset comparison.
Fig.~\ref{fig:three_dataset_macro_f1} shows the same Macro-F1 comparison in the
per-dataset grouped layout, and Fig.~\ref{fig:three_dataset_precision} and
Fig.~\ref{fig:three_dataset_recall} show the precision and recall components
separately, which is the basis for the precision/recall trade-off noted above for
NordicDat.

---

## B. Adaptation Cost and Computational Efficiency

Table~\ref{tab:results_cost} reports adaptation CPU, total runtime, retraining
counts and RAPT reuse events. Two quantities are reported separately and must not
be conflated:

* **Adaptation CPU** is the process time spent inside adaptation routines only
  (detector evaluation, policy construction, refitting, refresh, reuse decisions).
* **Runtime** is the total streaming-evaluation time and additionally contains
  feature construction, scaling and per-sample prediction.

A reduction in adaptation CPU therefore does not necessarily produce a
proportionate reduction in runtime.

The measured reduction in adaptation CPU relative to Full Retraining is
consistent in direction across all four streams: $1.4008 \rightarrow 0.2292$ s on
5G Campus QoS ($-83.6\%$), $23.6318 \rightarrow 1.9711$ s on UGR'16 ($-91.7\%$),
$2.2034 \rightarrow 0.2594$ s on NordicDat ($-88.2\%$) and
$0.7636 \rightarrow 0.3955$ s on 9B 5G latency QoS ($-48.2\%$).

The corresponding reduction in **runtime** is smaller in every case: $-44.0\%$,
$-82.9\%$, $-40.6\%$ and $-14.5\%$ respectively. The gap between the two figures
is the fixed prediction and feature-construction cost that adaptation does not
affect. On UGR'16, where the retraining count is high ($144$ for Full
Retraining versus $11$ retrains and $133$ reuses for RAPT), the runtime reduction
closely tracks the adaptation reduction. On 9B 5G latency QoS, where Full
Retraining performs only $9$ retrains, the adaptation saving is a small fraction
of the total and the runtime saving is correspondingly small ($2.4689 \rightarrow
2.1112$ s).

The adaptation events differ substantially by model. Frozen performs no
adaptation. Event-Driven performs $1$--$11$ retrains depending on the stream.
Full Retraining performs a retrain on every regime transition ($14$, $144$, $18$
and $9$ retrains respectively). RAPT performs $2$--$11$ retrains and reuses a
stored policy $6$--$133$ times; on UGR'16 the $133$ reuse events against $11$
retrains are the source of its large adaptation saving on that stream.
Fig.~\ref{fig:three_dataset_streaming_f1} shows the per-window streaming Macro-F1
on the three 9A streams and Fig.~\ref{fig:pareto} the accuracy/cost Pareto view
for the main model set.

Fig.~\ref{fig:three_dataset_adaptation_cpu} shows the adaptation CPU comparison
across the three 9A streams and Fig.~\ref{fig:three_dataset_runtime} the
corresponding total-runtime comparison; the two figures differ in the magnitude
of the RAPT advantage for the reason given above. Fig.~\ref{fig:three_dataset_new_trees}
shows the number of newly trained trees, which is the quantity that drives the
retraining cost, and Fig.~\ref{fig:cross_dataset_reuse} shows the reuse counts in
the head-to-head cross-dataset layout.

---

## C. RAPT Policy Reuse and Adaptation Behaviour

Table~\ref{tab:results_cost} records reuse and retraining events, and
Fig.~\ref{fig:three_dataset_reuse} and Fig.~\ref{fig:three_dataset_retraining}
show the corresponding counts.

RAPT reuses a stored policy rather than retraining on most regime transitions:
$12$ reuse events against $2$ retrains on 5G Campus QoS, $133$ against $11$ on
UGR'16, $16$ against $2$ on NordicDat, and $6$ against $3$ on 9B 5G latency QoS.
The reuse ratio is therefore high on every stream, and it is the mechanism
behind the adaptation-CPU saving reported in Section~V-B.

The reuse count alone does not indicate whether reuse was beneficial. On UGR'16
RAPT reuses $133$ times yet records the lowest Macro-F1 of the five models
($0.8360$), while RAPT-Enhanced, which reuses and retrains the same number of
times ($133$ reuses, $11$ retrains), records $0.9276$. The two models differ in
the reuse *decision*, not in the reuse *frequency*. This is consistent with the
interpretation that, on a stream with strong recurrence and concept change,
reusing a stored policy without checking whether the regime's label relationship
is unchanged can be harmful. That interpretation is examined directly in the
controlled recurring-concept-drift experiment in Section~V-H.

On 5G Campus QoS the opposite pattern appears: reuse is associated with a
negative Macro-F1 difference relative to Full Retraining
($0.9381$ versus $0.9836$) with no change from RAPT-Enhanced
($0.9381$), indicating that on this stream the limiting factor is not the reuse
decision but the reused policy itself (Section~V-D).

---

## D. RAPT Efficiency Ablation

Table~\ref{tab:results_rapt_ablation} reports the efficiency ablation on the
5G Campus QoS stream. Each variant adds one mechanism to the base ladder, so the
table can be read as a sequence of incremental changes rather than as a set of
independently tuned configurations.

**Cost drivers.** Before the mechanism comparison, the refit cost was measured
directly. On this configuration the dominant refit cost driver is the number of
estimators: a $20$-tree refit cost approximately $22$~ms and a $100$-tree refit
approximately $104$~ms. Buffer size was a secondary factor: a $200$-sample buffer
cost approximately $80$~ms and a $1000$-sample buffer approximately $104$~ms.
These are measurements of the evaluated ensemble configuration and are not
claimed to be a general property of ensemble refitting.

**Base and trigger variants.** RAPT-Full recorded $0.9587 \pm 0.0000$ Macro-F1 at
$0.4283$ s adaptation CPU. Replacing the absolute refit trigger with a relative
degradation trigger (RAPT-Evidence) left the result unchanged at
$0.9587 \pm 0.0000$ and fired **zero** refreshes. A relative trigger compares
current performance against a decaying reference of that same policy's recent
performance; when a policy is *persistently* poor rather than *deteriorating*,
the reference converges toward the poor level and the trigger condition is never
met. This is examined further in Section~V-H.

**Absolute floor.** RAPT-Floor replaced the relative trigger with an absolute
accuracy floor and fired $7.6$ refreshes on average, recording
$0.9797 \pm 0.0050$ Macro-F1 at $0.5865$ s adaptation CPU. It therefore detected
the persistent low-performance condition that the relative trigger missed, at an
accuracy level below RAPT-Cheap and with higher seed variance
($\pm 0.0050$ versus $\pm 0.0029$). The floor value used was $0.97$; this is an
experimentally selected configuration for the evaluated stream and would require
calibration for another stream.

**Cheap refit.** RAPT-Cheap retained the periodic refresh of RAPT-Refresh-W5 but
used the cheaper refit (fewer trees, smaller buffer) identified by the cost-driver
measurement. It recorded $0.9851 \pm 0.0029$ Macro-F1 at $0.8477$ s adaptation
CPU and $2.330$ s runtime. Relative to the full-refit periodic refresh
(RAPT-Refresh-W5, $0.9847 \pm 0.0028$ Macro-F1 at $2.7168$ s adaptation CPU),
the cheap refit retained the accuracy while reducing adaptation CPU by
approximately $69\%$. Relative to Full Retraining it retained the accuracy while
reducing adaptation CPU by approximately $40\%$.

**Incremental variant.** RAPT-Incremental replaced the batch refresh with an
online river Hoeffding learner blended into the reused policy. It recorded
$0.9549 \pm 0.0020$ Macro-F1 at $0.5813$ s adaptation CPU — lower Macro-F1 than
the cheaper RAPT-Full ($0.9587$ at $0.4283$ s). It is therefore dominated on both
axes on this stream and is reported as a negative result in Section~V-H.

Fig.~\ref{fig:final_ladder} shows the ablation ladder and
Fig.~\ref{fig:final_regime_f1} the per-regime Macro-F1 for the ablations.
Fig.~\ref{fig:final_cost_tradeoff} shows the accuracy/cost scatter for the
ablation variants, in which RAPT-Cheap and RAPT-Floor occupy the high-accuracy /
low-cost region and RAPT-Incremental is displaced from it.

---

## E. RAPT-Cheap Cost-Efficiency Result

The RAPT-Cheap configuration is the only setting in the ablation in which
Macro-F1 is at least as high as Full Retraining *and* adaptation CPU is lower.
The measured values are:

| Configuration | Macro-F1 | Adaptation CPU (s) |
| :--- | ---: | ---: |
| Full Retraining | $0.9829 \pm 0.0023$ | $1.4063$ |
| RAPT-Full | $0.9587$ | $0.4283$ |
| **RAPT-Cheap** | $\mathbf{0.9851} \pm 0.0029$ | $\mathbf{0.8477}$ |
| RAPT-Floor | $0.9797 \pm 0.0050$ | $0.5829$ |
| RAPT-Incremental | $0.9549 \pm 0.0020$ | $0.5813$ |

The Macro-F1 difference is

$$\Delta \text{Macro-F1} = 0.9851 - 0.9829 = +0.0022.$$

The paired comparison against Full Retraining across seeds reports
$p = 0.3125$ with Cohen's $d = 0.47$ and a $95\%$ confidence interval for the
difference of $[-0.0036, +0.0080]$ (Table~\ref{tab:results_stats}). The
confidence interval contains zero and the test does not establish a difference at
$\alpha = 0.05$.

The supported statement is therefore: **RAPT-Cheap matched the predictive
performance of Full Retraining while reducing adaptation CPU by approximately
$40\%$.** The numerical Macro-F1 difference of $+0.0022$ is not statistically
significant, and no claim of statistical superiority is made. The relevant
evidence for RAPT-Cheap is the cost reduction, not the Macro-F1 difference.

A related observation concerns RAPT-Floor, which achieves a larger adaptation-CPU
reduction ($0.5829$ s, $-58.6\%$ versus Full Retraining) at a Macro-F1 of
$0.9797 \pm 0.0050$, also not significantly different from Full Retraining
($p = 0.375$). RAPT-Floor is thus the cheapest configuration that is not
statistically distinguishable from Full Retraining, while RAPT-Cheap is the
configuration with the smallest numerical Macro-F1 deficit.

---

## F. Historical Drift Detector Comparison

Table~\ref{tab:results_detectors} compares RAPT with four established drift
detectors (ADWIN, Page-Hinkley, EDD, EDMA). The purpose of this comparison is to
establish whether regime-aware policy reuse provides a different
computational/performance trade-off from drift-detection-driven adaptation; the
detectors are treated as historical baselines rather than as competing RAPT
architectures.

A first measured observation concerns detector sensitivity. On all three 9A
streams, ADWIN, Page-Hinkley and EDMA recorded identical Macro-F1 and identical
adaptation events. On 5G Campus QoS and UGR'16 they recorded zero adaptation
events, which is the same result as Frozen ($0.9364$ and $0.9690$ respectively).
On NordicDat they likewise recorded zero events and $0.2779$ Macro-F1, equal to
Frozen. Under the evaluated configuration these three detectors did not trigger
adaptation on these streams, and their results are therefore indistinguishable
from the frozen baseline.

EDD triggered adaptation on every stream ($38$--$39$ events) and recorded the
highest Macro-F1 of any method in this comparison on 5G Campus QoS
($0.9847 \pm 0.0019$) and on NordicDat ($0.4184 \pm 0.0165$). On UGR'16 EDD
recorded $0.9655 \pm 0.0016$ against $0.9690 \pm 0.0029$ for the untriggered
baseline. The adaptation cost of EDD was the highest in the comparison on every
stream ($3.98$, $6.10$ and $4.69$ s).

RAPT recorded $0.9381$ (5G Campus QoS), $0.8360$ (UGR'16) and $0.3818$
(NordicDat), with $2$, $11$ and $2$ adaptation events and adaptation CPU of
$0.2292$, $1.9711$ and $0.2594$ s. On UGR'16 and NordicDat, RAPT's adaptation
CPU was lower than EDD's ($1.9711$ versus $6.1012$ s, and $0.2594$ versus
$4.6875$ s), while its Macro-F1 was lower ($0.8360$ versus $0.9655$, and $0.3818$
versus $0.4184$). On 5G Campus QoS, RAPT recorded a higher Macro-F1 than the
three non-triggering detectors ($0.9381$ versus $0.9364$) at an adaptation cost
of $0.2292$ s.

The measured result is a trade-off rather than a uniform ordering: EDD attains
higher Macro-F1 on two of the three 9A streams at substantially higher adaptation
cost, RAPT attains lower adaptation cost on all three, and the non-triggering
detectors reproduce the frozen baseline. Fig.~\ref{fig:detectors_f1},
Fig.~\ref{fig:detectors_cpu}, Fig.~\ref{fig:detectors_events},
Fig.~\ref{fig:detectors_runtime} and
Fig.~\ref{fig:detectors_tradeoff} show the Macro-F1, adaptation-CPU,
adaptation-event, runtime and accuracy/cost-trade-off comparisons respectively.

---

## G. Cross-Dataset Analysis

The head-to-head cross-dataset comparison uses UGR'16 (9A) and the 9B 5G latency
QoS stream. Both are independent recurring-regime telecom streams; UGR'16 is a
binary anomaly-detection task and 9B is a 3-class QoS task.

**Findings that repeat across datasets.** RAPT reduced adaptation CPU relative to
Full Retraining on both streams ($23.6318 \rightarrow 1.9711$ s on UGR'16;
$0.7636 \rightarrow 0.3955$ s on 9B), and reuse events exceeded retrain events on
both ($133$ versus $11$; $6$ versus $3$). The direction of the adaptation-cost
result is consistent across all four evaluated streams (Section~V-B).

**Dataset-specific findings.** The predictive outcome differs by stream. On
UGR'16, base RAPT recorded the lowest Macro-F1 of the five models ($0.8360$)
while RAPT-Enhanced recorded $0.9276$, a recovery of $0.0917$. On 9B 5G latency
QoS the same two models differ by only $0.0021$ ($0.8894$ versus $0.8915$) and the
whole model set spans $0.0133$ Macro-F1. The enhancement therefore has a large
measured effect on one stream and a small one on the other; this difference is
associated with the strength of natural regime recurrence (UGR'16 reports 168
recurrence events, 9B reports the recurring sequence $ABCDACBDAB$).

**Findings that cannot yet be generalised.** The NordicDat results
(Macro-F1 in the range $0.25$--$0.42$) indicate that on a stream where all models
perform poorly in absolute terms, the relative ordering of accuracy and Macro-F1
can invert: RAPT recorded the highest accuracy ($0.6172$) and Full Retraining the
highest Macro-F1 ($0.4227$). Whether this inversion is a property of streams with
weak baselines or of the NordicDat target construction cannot be determined from
the present results, and the two are not merged into a single claim here.

Fig.~\ref{fig:cross_dataset_f1} shows the Macro-F1 cross-dataset comparison and
Fig.~\ref{fig:cross_dataset_adapt_cpu} the adaptation-CPU comparison.

---

## H. Statistical Significance and Effect Sizes

\label{sec:stats}

Table~\ref{tab:results_stats} reports the paired comparisons used in this
section. All tests are paired Wilcoxon signed-rank tests over seeds or windows as
indicated by the source file, with Cohen's $d$ as the effect size. The tests were
computed on the per-window paired differences and are not re-derived here.

**RAPT vs Full Retraining (9A, per-window).** On 5G Campus QoS the difference was
$-0.0455$ Macro-F1 ($p = 2.6 \times 10^{-4}$, $d = -0.32$), on UGR'16 $-0.1236$
($p = 2.6 \times 10^{-11}$, $d = -0.67$) and on NordicDat $-0.0409$
($p = 0.255$, not significant). The 5G Campus QoS and UGR'16 differences are
statistically significant and negative; the NordicDat difference is not
statistically significant.

**RAPT-Enhanced vs Full Retraining (9A, per-window).** On UGR'16 the difference
was $-0.0319$ ($p = 0.0031$, $d = -0.25$), significant but much smaller in
magnitude than the base-RAPT difference ($-0.1236$). On NordicDat it was
$-0.0444$ ($p = 0.239$, not significant).

**RAPT-Enhanced vs RAPT (9A, per-window).** On UGR'16 the difference was
$+0.0917$ ($p = 5.4 \times 10^{-9}$, $d = 0.56$), statistically significant and
positive. On 5G Campus QoS the two models were numerically identical
(difference $0.0000$, no test statistic reported). On NordicDat the difference
was $-0.0035$ ($p = 0.643$, not significant).

**RAPT vs Event-Driven (9A, per-window).** Significant and negative on 5G Campus
QoS ($-0.0255$, $p = 0.0057$) and UGR'16 ($-0.0612$, $p = 2.6 \times 10^{-4}$);
significant and positive on NordicDat ($+0.1271$, $p = 1.8 \times 10^{-4}$).

**9B natural drift (per-window).** RAPT vs Frozen $-0.005$ accuracy
($p = 0.0075$) and RAPT vs Full Retraining $-0.014$ accuracy ($p = 0.0043$) are
significant and negative; RAPT vs Event-Driven $-0.003$ ($p = 0.564$) is not
significant. These tests use accuracy on the 9B stream.

**RAPT-Cheap vs Full Retraining.** $\Delta = +0.0022$ Macro-F1,
$p = 0.3125$, $d = 0.47$, $95\%$ CI $[-0.0036, +0.0080]$. Not significant.

**RAPT-Floor vs Full Retraining.** $\Delta = -0.0032$ Macro-F1, $p = 0.375$,
$d = -0.53$. Not significant.

**RAPT-Incremental vs Full Retraining.** $\Delta = -0.0279$ Macro-F1,
$p = 0.0625$, $d = -13.51$. The effect size is large but the test does not reach
$\alpha = 0.05$ at five seeds.

Two caveats apply throughout. First, the 9A per-window tests use windows as the
unit of analysis and therefore have high power relative to the number of
independent streams; a significant per-window difference should not be read as
evidence of generalisation beyond these streams. Second, the five-seed tests
cannot attain $p < 0.0625$ in the two-sided Wilcoxon test, so for those
comparisons a large effect size with $p = 0.0625$ reflects limited statistical
power rather than absence of an effect. A non-significant result is reported as
"the experiment did not establish a statistically significant difference", not as
evidence that the models are identical.

Fig.~\ref{fig:detectors_tradeoff} and Fig.~\ref{fig:fig9b_accuracy_cost_tradeoff}
show the accuracy/cost trade-off underlying these comparisons.

---

## I. Failure Cases and Negative Findings

### Persistent deficiency is not detected by a relative trigger

RAPT-Evidence fired zero refreshes and reproduced RAPT-Full exactly
($0.9587 \pm 0.0000$). The regime in question had a reused policy that was
uniformly poor rather than deteriorating, so the relative trigger's decaying
reference converged toward the poor level and the condition was never satisfied.
The general statement supported by this result is: a relative degradation trigger
can identify deterioration relative to an established reference, but it may fail
when performance is persistently low and the reference itself adapts toward that
low level. The absolute floor (RAPT-Floor, $7.6$ refreshes, $0.9797$) detected
the same condition, at the cost of a threshold that requires calibration per
stream.

### Incremental update did not improve the cost/accuracy trade-off

RAPT-Incremental recorded $0.9549 \pm 0.0020$ Macro-F1 at $0.5813$ s adaptation
CPU, compared with $0.9587$ at $0.4283$ s for RAPT-Full: lower accuracy at higher
cost. Two diagnostic experiments were run to identify the cause.

*Standalone incremental learner.* A river Hoeffding ensemble evaluated alone
reached approximately $0.945$ Macro-F1, compared with approximately $0.98$ for
the batch ensemble on the same stream. Increasing the number of incremental trees
did not close the gap: $1$ tree $\approx 0.946$, $3$ trees $\approx 0.945$,
$10$ trees $\approx 0.941$, while cost grew linearly with tree count.

*Blended incremental learner.* Blending the incremental learner into the reused
batch policy at weight $0.5$ reduced Macro-F1 to approximately $0.963$; at weight
$0.3$ performance was approximately $0.983$, effectively identical to the
batch-only model.

The incremental learner could therefore neither replace the batch policy nor
usefully supplement it on this stream. This is reported as a stream-specific
negative finding: the batch ensemble is already close to ceiling on a 19-feature,
3-class problem, so there is little headroom for an incremental learner to
contribute. It is not evidence that incremental learning is universally
ineffective.

### Dataset dependence of the cheap refit

The $20$-tree cheap refit was sufficient to correct the reused policy on the
5G Campus QoS stream, where the refresh only needed to adjust a policy that was
already close to the target. A refit of this size reduces the refreshed policy's
capacity, and on a stream with a larger pre-/post-drift difference it may
underfit. The present results do not establish that the cheap refit is sufficient
on harder streams.

### Recurring concept drift

The controlled recurring-concept-drift experiment ($A \rightarrow$ drift
$\rightarrow B \rightarrow A'$, where $A'$ has similar feature characteristics to
$A$ but a changed $P(Y|X)$) is the direct test of blind reuse. In the per-phase
results, RAPT and Frozen recorded identical Macro-F1 in every phase and at every
severity; for example at severity $0.10$, phase $B$ both recorded $0.2453$, and
at severity $1.00$, phase $B$ both recorded the corresponding lowest values of
the model set. RAPT recorded one reuse event per scenario while Full Retraining
recorded two retrains. The measured outcome is that RAPT's reuse of the
historical $A$ policy did not yield a performance advantage over the frozen
baseline in the recurring scenario at any evaluated severity. This is reported as
an observed finding; the RAPT algorithm was not modified to address it.

### Statistical limitations

The five-seed experiments cannot attain $p < 0.0625$ in the two-sided Wilcoxon
test, which limits the strength of any claim based on them. Several comparisons
report large effect sizes with $p = 0.0625$; these are reported as not significant
at $\alpha = 0.05$. The per-window 9A tests have higher power but treat windows
from the same stream as independent, which overstates the effective sample size
for claims about other streams.

---

## J. Drift-Severity Results (9B)

The 9B stream was additionally evaluated under controlled covariate drift and
controlled concept drift at severities $\{0.10, 0.20, 0.30, 0.50, 1.00\}$. In the
covariate-drift construction the input distribution changes while the
feature-to-label relationship is held as consistent as possible; in the
concept-drift construction the feature-to-label relationship is changed
deterministically after the drift point.

Under **covariate drift**, Macro-F1 for the adaptive models is higher at
severity $1.00$ than at severity $0.10$, but it is not monotonic across
severities. RAPT recorded $0.6854$, $0.7944$, $0.7285$, $0.7767$ and
$0.8122$ at severities $0.10$--$1.00$; Full Retraining recorded $0.6850$,
$0.7894$, $0.7189$, $0.7671$ and $0.8080$; Frozen recorded $0.5265$, $0.5201$,
$0.5192$, $0.5102$ and $0.4893$. At severities $0.10$, $0.20$, $0.30$, $0.50$ and
$1.00$, RAPT exceeded Full Retraining by $+0.0004$, $+0.0050$, $+0.0096$,
$+0.0096$ and $+0.0042$ Macro-F1 respectively. RAPT-Enhanced reproduced RAPT
exactly at every severity, which is consistent with the enhancement being
inactive under pure covariate drift.

Under **concept drift**, the pattern differs. Frozen, Full Retraining and RAPT
recorded identical Macro-F1 at every severity ($0.8777$, $0.8683$, $0.8683$,
$0.8117$, $0.7916$), because none of the three adapted after the drift point
(Full Retraining recorded zero retrains and RAPT zero retrains and zero reuse at
every severity). Event-Driven, which did adapt, recorded $0.9241$, $0.9379$,
$0.9379$, $0.8274$ and $0.7916$. At severities $0.10$, $0.20$ and $0.30$,
Event-Driven exceeded the three non-adapting models by $+0.0464$, $+0.0696$ and
$+0.0696$ Macro-F1. The recovery analysis reports a pre-drift Macro-F1 of $1.0$
for every model and recovery windows of $40$ (the maximum observable in the
evaluation window) at every severity, indicating that none of the models
recovered to $95\%$ of pre-drift Macro-F1 within the window under this
construction.

Fig.~\ref{fig:fig9b_natural_drift_f1} shows the natural-drift Macro-F1 timeline
with regime transitions, Fig.~\ref{fig:fig9b_covariate_severity_f1} and
Fig.~\ref{fig:fig9b_covariate_severity_cpu} the covariate-drift severity response
in Macro-F1 and adaptation CPU, Fig.~\ref{fig:fig9b_covariate_severity_recovery}
the covariate-drift recovery behaviour, Fig.~\ref{fig:fig9b_concept_severity_f1},
Fig.~\ref{fig:fig9b_concept_severity_accuracy} and
Fig.~\ref{fig:fig9b_concept_severity_recovery} the concept-drift response, and
Fig.~\ref{fig:fig9b_drift_type_comparison} the covariate-versus-concept
comparison. Fig.~\ref{fig:fig9b_regime_f1} shows the per-regime Macro-F1 and
Fig.~\ref{fig:fig9b_rapt_adaptation_timeline} the RAPT adaptation timeline with
reuse and retraining events annotated. The two heatmaps,
Fig.~\ref{fig:fig9b_covariate_f1_heatmap} and
Fig.~\ref{fig:fig9b_concept_f1_heatmap}, give the model-by-severity Macro-F1 grid
for each drift type, and Fig.~\ref{fig:fig9b_reuse_vs_retraining} the reuse
versus retraining comparison across drift scenarios.

---

## Figure and Table Inventory

All figures and tables referenced in this section are listed below with their
source. Every item below is referenced at least once above.

| Label | File | Content |
| :--- | :--- | :--- |
| Fig.~\ref{fig:final_f1_bars} | `fig_final_f1_bars.png` | RAPT ablation Macro-F1 bars |
| Fig.~\ref{fig:final_ladder} | `fig_final_ladder.png` | RAPT ablation ladder |
| Fig.~\ref{fig:final_regime_f1} | `fig_final_regime_f1.png` | Per-regime Macro-F1, ablations |
| Fig.~\ref{fig:final_cost_tradeoff} | `fig_final_cost_tradeoff.png` | Accuracy/cost scatter, ablations |
| Fig.~\ref{fig:three_dataset_macro_f1} | `fig9a_three_dataset_macro_f1.png` | Macro-F1, three 9A streams |
| Fig.~\ref{fig:three_dataset_accuracy} | `fig9a_three_dataset_accuracy.png` | Accuracy, three 9A streams |
| Fig.~\ref{fig:three_dataset_precision} | `fig9a_three_dataset_precision.png` | Precision, three 9A streams |
| Fig.~\ref{fig:three_dataset_recall} | `fig9a_three_dataset_recall.png` | Recall, three 9A streams |
| Fig.~\ref{fig:three_dataset_adaptation_cpu} | `fig9a_three_dataset_adaptation_cpu.png` | Adaptation CPU, three 9A streams |
| Fig.~\ref{fig:three_dataset_runtime} | `fig9a_three_dataset_runtime.png` | Runtime, three 9A streams |
| Fig.~\ref{fig:three_dataset_reuse} | `fig9a_three_dataset_reuse.png` | Reuse events, three 9A streams |
| Fig.~\ref{fig:three_dataset_retraining} | `fig9a_three_dataset_retraining.png` | Retrain events, three 9A streams |
| Fig.~\ref{fig:three_dataset_new_trees} | `fig9a_three_dataset_new_trees.png` | Newly trained trees, three 9A streams |
| Fig.~\ref{fig:three_dataset_streaming_f1} | `fig9a_three_dataset_streaming_f1.png` | Per-window streaming Macro-F1, 9A |
| Fig.~\ref{fig:pareto} | `fig9a_pareto.png` | Accuracy/cost Pareto, main models |
| Fig.~\ref{fig:cross_dataset_f1} | `fig_cross_dataset_f1.png` | Cross-dataset Macro-F1 (UGR'16 vs 9B) |
| Fig.~\ref{fig:cross_dataset_adapt_cpu} | `fig_cross_dataset_adapt_cpu.png` | Cross-dataset adaptation CPU |
| Fig.~\ref{fig:cross_dataset_reuse} | `fig_cross_dataset_reuse.png` | Cross-dataset reuse counts |
| Fig.~\ref{fig:detectors_f1} | `fig9a_rapt_vs_drift_detectors_f1.png` | Macro-F1 vs drift detectors |
| Fig.~\ref{fig:detectors_cpu} | `fig9a_rapt_vs_drift_detectors_cpu.png` | Adapt CPU vs drift detectors |
| Fig.~\ref{fig:detectors_events} | `fig9a_rapt_vs_drift_detectors_events.png` | Adapt events vs drift detectors |
| Fig.~\ref{fig:detectors_runtime} | `fig9a_rapt_vs_drift_detectors_runtime.png` | Runtime vs drift detectors |
| Fig.~\ref{fig:detectors_tradeoff} | `fig9a_rapt_vs_drift_detectors_tradeoff.png` | Accuracy/cost trade-off vs detectors |
| Fig.~\ref{fig:fig9b_natural_drift_f1} | `fig9b_natural_drift_f1.png` | 9B natural-drift F1 timeline |
| Fig.~\ref{fig:fig9b_covariate_severity_f1} | `fig9b_covariate_severity_f1.png` | Covariate severity vs Macro-F1 |
| Fig.~\ref{fig:fig9b_covariate_severity_cpu} | `fig9b_covariate_severity_cpu.png` | Covariate severity vs adapt CPU |
| Fig.~\ref{fig:fig9b_covariate_severity_recovery} | `fig9b_covariate_severity_recovery.png` | Covariate severity vs recovery |
| Fig.~\ref{fig:fig9b_concept_severity_f1} | `fig9b_concept_severity_f1.png` | Concept severity vs Macro-F1 |
| Fig.~\ref{fig:fig9b_concept_severity_accuracy} | `fig9b_concept_severity_accuracy.png` | Concept severity vs accuracy |
| Fig.~\ref{fig:fig9b_concept_severity_recovery} | `fig9b_concept_severity_recovery.png` | Concept severity vs recovery |
| Fig.~\ref{fig:fig9b_drift_type_comparison} | `fig9b_drift_type_comparison.png` | Covariate vs concept comparison |
| Fig.~\ref{fig:fig9b_regime_f1} | `fig9b_regime_f1.png` | Per-regime Macro-F1, 9B |
| Fig.~\ref{fig:fig9b_rapt_adaptation_timeline} | `fig9b_rapt_adaptation_timeline.png` | RAPT adaptation timeline |
| Fig.~\ref{fig:fig9b_reuse_vs_retraining} | `fig9b_reuse_vs_retraining.png` | Reuse vs retraining, drift scenarios |
| Fig.~\ref{fig:fig9b_accuracy_cost_tradeoff} | `fig9b_accuracy_cost_tradeoff.png` | Accuracy/cost scatter, 9B |
| Fig.~\ref{fig:fig9b_covariate_f1_heatmap} | `fig9b_covariate_f1_heatmap.png` | Model×severity F1, covariate |
| Fig.~\ref{fig:fig9b_concept_f1_heatmap} | `fig9b_concept_f1_heatmap.png` | Model×severity F1, concept |
| Table~\ref{tab:results_main_performance} | `final_results_tables.tex` | Overall predictive performance |
| Table~\ref{tab:results_cost} | `final_results_tables.tex` | Computational cost |
| Table~\ref{tab:results_rapt_ablation} | `final_results_tables.tex` | RAPT efficiency ablation |
| Table~\ref{tab:results_stats} | `final_results_tables.tex` | Statistical comparisons |
| Table~\ref{tab:results_detectors} | `final_results_tables.tex` | Historical drift detectors |

**Note on figure labels.** No LaTeX source containing `\label{fig:...}`
definitions for these figures exists in the repository; the figures are produced
as PNG files by the experiment scripts. The `fig:` labels used above are defined
here for the manuscript and must be added to the figure environments when the
figures are embedded. The `tab:` labels, by contrast, are taken from the existing
generated LaTeX table files.
