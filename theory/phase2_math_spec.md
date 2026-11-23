# M-TTT Phase 2 | Frozen hypotheses and decision rules

Phase 1 established reversible replay on the tested paths. Every poison alarm arrived before the first update. The Fisher brake lost 2.125 points of Task 1 backward transfer to momentum SGD and 3.000 points to isotropic damping. Its difference against AdamW was +0.292 points, below the required superiority margin. Phase 2 tests delayed feedback, equal energy removal, anisotropic inertia and an explicit restoring force. [Phase 1 audit](evidence/phase1_replication_audit.json)

## One loss gradient with a synchronized endpoint

For an immutable batch \(b\), let \(H_b=U_b(q)+p^\top Vp/2\), where \(V=M^{-1}\) is constant and symmetric positive definite. The Phase 2 conservative update is position Verlet, in drift-kick-drift order.

\[
 x=q+\tfrac h2 Vp,\qquad p'=p-h\nabla U_b(x),\qquad q'=x+\tfrac h2 Vp'.
\]

Both outputs refer to the end of the step. Reflect \(p'\), use the same positive \(h\) and batch, and the first backward drift returns to \(x\). The backward kick gives \(-p\) and the final drift returns to \(q\). Reverse the order of batches and reflect momentum at the two ends of the replay interval. Each drift Jacobian has symmetric upper-right block \(hV/2\). Each kick has symmetric lower-left block \(-hU_b''\). Multiplication gives \(D\Phi^\top J D\Phi=J\) in every dimension. This proves exact real-arithmetic symplecticity and reversibility for constant Fisher mass as well as unit mass. The finite-precision gate tests the submitted map. [Symbolic products and inverse](phase2_verify_symbols.py) · [Störmer-Verlet with a positive definite mass matrix](https://na.uni-tuebingen.de/~lubich/icm.pdf)

Velocity Verlet can reuse the last endpoint gradient as the next first gradient when the entire force law at that point agrees, including batch, normalization, RNG and model state. A run of \(N\) identical-batch steps then costs \(N+1\) gradients with full endpoint momenta. At a batch change the two half-kicks use different forces. Their sum costs two evaluations at the shared position. A stale endpoint force changes the map. DKD uses exactly one fresh gradient per forward or reverse step on arbitrary ordered batches. Adjacent position drifts can be fused if endpoint diagnostics are preserved. The frozen interface materializes each endpoint. [Different-batch negative control](../eval_v2/physics.py)

## C2 | Three mechanisms

Let \(q_*\) be the Task 1 pretrained weights and \(f_i\) the empirical Fisher diagonal, obtained by averaging squared per-example scores on 96 separate Task 1 samples. Freeze \(W=\operatorname{diag}(f/\max(\bar f,10^{-12}))\). Split at the lower median of \(f\), putting ties in \(P_L\), and let \(P_H=I-P_L\).

**Diagnosis.** For a Phase 1 half-brake \(p'=e^{-\gamma hP_L/2}p\), the kinetic energy removed at a common incoming \(p\) was \((1-e^{-\gamma h})\|P_Lp\|^2/2\). The isotropic brake removed the same fraction of the entire kinetic energy. \(P_Hp\) passed untouched and could carry the Task 1 solution away from its old position. This explains why equal nominal strength confounded orientation and dissipation. It does not identify their causal contributions to the observed accuracy difference once trajectories diverge.

**C2-EnergyMatched.** Run three unit-mass DKD trajectories on identical batches and apply one endpoint brake to each. At its own post-DKD momentum, the isotropic arm requests \(D_0=(1-e^{-2h\gamma})K_I\). Measure capacities \(K_L=\|P_Lp_L\|^2/2,\ K_H=\|P_Hp_H\|^2/2\) on the *other arms' actual momenta*. Use the same scalar \(D=\min(D_0,K_L,K_H)\) on all three. For each projector \(P=I,P_L,P_H\) and the corresponding \(K_P\), set

\[
p_{\rm out}=(I-P)p+\sqrt{1-D/K_P}\,Pp.
\]

Use the identity for \(K_P=D=0\). Orthogonality gives \(K(p)-K(p_{\rm out})=D\). Pure damping on a subspace cannot remove more than \(K_P\). The coordinator records each requested value, both capacities, the common actual target and all three realized removals. A cap makes that trajectory unsuitable for a **fixed-strength isotropic** attribution. The three capped arms still answer the distinct question about equal *realized* dissipation. Any cap on the held-out schedule fails the primary fixed-strength gate. Coordinating the three live states is part of the experiment.

**C2-FisherMass.** Set \(M=I+\alpha W\) and use the synchronized DKD equations without friction. At the same \(p\), larger \(f_i\) reduces displacement per drift. The old solution is never supplied as a restoring anchor to this arm. The selected step size also defines a unit-mass glide control. The symplectic and reflection identities above hold for every frozen \(\alpha\ge0\). They imply no retention bound.

**C2-TrueMetriplectic.** Use the selected Fisher mass and a gated split that draws \(q\) back towards \(q_*\). Let \(A(q)=(q-q_*)^\top W(q-q_*)/2\). The prior entropy is \(-A(q)\). A nonconstant function of \(q\) cannot be a Casimir of the ordinary nondegenerate canonical bracket. Introduce an energy reservoir \(e\) and an entropy coordinate \(\sigma\), with total energy \(E=H_b+e\) and entropy \(S=\sigma-A(q)\). For \(z=(q,p,e,\sigma)\), define

\[
 B=\begin{pmatrix}I&0\\0&I\\0&0\\\nabla A^\top&0\end{pmatrix},
\quad L=BJB^\top,\qquad
 C=\begin{pmatrix}I\\0\\-\nabla U_b^\top\\0\end{pmatrix},
\quad K=\lambda g\, CC^\top,\qquad
 \dot z=L\nabla E+K\nabla S .
\]

Here \(g\) is the nonnegative external gate and \(\lambda\) is the anchor strength. \(L\) is canonical Poisson with two Casimirs in coordinates \((q,p,e,\sigma-A(q))\), so the Jacobi identity follows by change of coordinates. \(B^\top\nabla S=0,\ C^\top\nabla E=0\), and \(v^\top Kv=\lambda g\|C^\top v\|^2\ge0\). The dissipative equations give \(\dot q=-\lambda g\nabla A,\ \dot p=0,\ \dot\sigma=0,\ \dot S=\lambda g\|\nabla A\|^2\). For duration \(\tau\),

\[
 q'=q_*+e^{-\tau\lambda g W}(q-q_*),\qquad e'=e+U_b(q)-U_b(q').
\]

Bracket one Fisher-mass DKD step with two exact half anchor flows. On the conservative substep, \(e\) stays fixed and \(\sigma\) gains \(A(q_{\rm end})-A(q_{\rm start})\). The reservoirs do not feed back into \(q,p\). Production steps use one loss gradient and analytic anchor updates. The unit probe reconstructs heat from scalar potential values. The anchor subflow conserves total energy exactly in the algebra. DKD has its discretization error, and changing \(b\) supplies external work. Reversal by reflection is claimed only at zero gate. [Exact GENERIC identities](evidence/phase2/symbols.json) · [Shang and Öttinger on reversible and irreversible splitting](https://arxiv.org/abs/1804.05114)

## B2 | Delayed feedback, rollback policies and memory

Both regimes use the shared 179-parameter MLP and the same Task 1 pretrain as Phase 1, with fresh seeds. The online stream is from Task 2, so refusing all clean adaptation has a measurable opportunity cost. Task 1 measures retention and Task 2 measures clean adaptation. Labels, features and stochastic choices are fixed per event. The driver presents immutable otherwise opaque event IDs in a seeded permutation. Only the scorer receives the clean/dirty assignment and the true origin for numerical comparisons.

| Regime | Feedback accessible to the policy | Authorized replay interval |
|---|---|---|
| B2a, ramped onset | 48 batches of 32. The first 16 have clean labels. In dirty batch \(j=1,\ldots,32\) after onset, cycle the labels of \(\min(j,24)\) randomly chosen examples. The matched control has clean training labels throughout. A verifier audits 8 randomly chosen outcomes from each batch and releases **one aggregate disagreement rate after each block of 8 committed batches**. One verified label in every fourth batch has a simulated annotation error. Trigger when the released rate exceeds the validation-selected threshold. | Undo just the most recent 8 committed steps and stop updating until the transaction ends. The estimated origin is the beginning of the failed block. Compare it with the withheld state after 16 clean steps. A late alert can produce a locally correct inverse and still miss the true origin. |
| B2b, terminal verifier | 32 batches. The learner sees geometric teacher pseudo-labels from features. The dirty source cyclically shifts \(\min(2j,32)\) pseudo-labels at step \(j\). Only after step 32 does a separate 300-example labeled Task 1 canary return a single pass/fail bit. Failure requires at least nine fewer correct than its pre-window count. The scorer's 300 test examples are independent of the verifier. | On failure, undo the full 32-step window. On pass, keep it. A terminal skip cannot change the steps already taken. No trusted adaptation labels or intermediate verifier scores go to B2b policies. |

In B2a each pre-onset receipt contains at most two erroneous annotations among 64 outcomes. All candidate thresholds exceed \(2/64\). Even the first possible corruption-driven alarm arrives after 24 commits, including eight corrupted steps. Every compared method receives identical B2a receipts. B2b uses the same verifier rule on each method's own outcome. Results use all twelve paired seeds, including missed detections. The regular 8-step audits align with the injected onset in B2a. Arbitrary sub-block onset localization is outside this design.

A separate **passive early-warning control** records the first pre-update AdamW cross-entropy exceeding the mean plus four population standard deviations from 32 independent clean Task 2 calibration batches. It cannot act in the audited policy. More than one early warning before the eighth corrupted update falsifies the B2a stealth criterion. Scheduled feedback alone cannot establish that raw observations hid the corruption.

Compare DKD-rewind, DKD-skip-after-alarm, DKD-continue, AdamW-skip-after-alarm, AdamW-continue and always-skip. Skip preserves parameters **and optimizer state at the alarm**. Continue processes the rest of the window. Evaluate each policy at the common stream endpoint. Then spend all remaining credits from a common \(2N\)-gradient budget on a prefix of the same new clean Task 1 continuation. Every replay gradient is charged. Report the forward, replay and clean-repair counts separately. The first comparison matches incoming data. The second matches actual useful gradient work and gives cheaper policies more clean repair data. Neither charges scalar audit forwards as gradients.

Only the rewind policy retains an `array('q')` of 8-byte IDs. Report peak payload, host array allocation, dense weight checkpoint bytes, full optimizer-state checkpoint bytes and immutable external event payload separately. The frozen f64 model costs \(179\cdot8=1432\) bytes for weights, \(2864\) for both phase components and \(4304\) for AdamW weights, moments and an 8-byte step count under dense serialization. A 32-ID rewind log costs 256 payload bytes. Preserving an otherwise ephemeral minibatch requires its actual data or an equivalent deterministic reconstruction recipe. The evaluator's held-back phase state is scoring instrumentation. `anchor=None` and `fisher=None` in B2.

**Finite-precision horizon.** On the first three held-out seeds, run native f32 DKD at \(h=0.12\), with nonzero initial momentum and changing neural batches. At \(T=16,32,64,128,256,500\), replay from a fresh optimizer. Require both absolute L2 errors below \(10^{-5}\) and a change of at most one correct Task 1 prediction. \(T_{\max}\) is the longest consecutive prefix of the *tested grid* passing on every seed. Return the first failed grid point and all later observations. If every point passes, report a right-censored horizon of at least 500 on this grid. Drift alone does not diagnose deterministic chaos.

The integer control uses an authoritative pair of int64 arrays \(Q,P\), grid spacing \(2^{-44}\) and round-to-nearest with ties to even. The network and its force remain f32. With unit mass, make the three shear translations

\[
 X=Q+\operatorname{round}(hP/2),\quad
 P'=P-\operatorname{round}\!\left(2^{44}h\nabla U_b(\operatorname{f32}(2^{-44}X))\right),\quad
 Q'=X+\operatorname{round}(hP'/2).
\]

The scale and rounding intermediates use binary64. Integer additions are exact in the checked range. Reflected drifts use the oddness of rounding, and the backward kick reevaluates the identical float force at the identical integer midpoint. Lossless initial encoding, finite increments below \(2^{51}\) in magnitude, states below \(2^{52}\), and deterministic replay on one device are required. The hypothesis fails on any unequal recovered integer coordinate. Report forward distances from the ordinary f32 endpoints and both forward accuracies. This changes the forward discretization. Carrying two int64 vectors costs 2864 bytes at \(d=179\), twice the native f32 phase-state payload, with no per-step parameter history. [Rein and Tamayo's integer-shear construction](https://arxiv.org/abs/1704.07715)

## Locked selection and falsification

Validation seeds are 10301, 10303 and 10313. The fresh test seeds are 20903, 20921, 20929, 20939, 20947, 20959, 20963, 20981, 21001, 21011, 21013 and 21017. CPU validation is authoritative. Before either device scores the test seeds, retain the validation artifact SHA-256 independently. Both test runs verify that receipt, rederive its choices from validation rows, and require matching evaluator and adapter source digests. The seeds are public for reproduction. Independent custody of the lock and a complete return of all rows are part of the audit.

For C2, every arm consumes exactly 192 fresh minibatch gradients on the same four streams of 48. The phase initialization, pretrain and frozen empirical Fisher are common preparation. Choose the candidate with highest mean Task 1 backward transfer among candidates with mean live accuracy at least 92%. If none qualify, maximize mean live accuracy. Remaining ties go to higher live accuracy, then grid order. The matched triplet first favors feasibility at the nominal isotropic strength on every validation step, then applies this rule using the worst arm's mean live accuracy and worst arm's mean BWT. Each family gets four candidate settings.

| Family | Validation grid |
|---|---|
| Momentum SGD and AdamW | SGD rates 0.0036, 0.0072, 0.0144, 0.0288. AdamW rates 0.0025, 0.005, 0.01, 0.02. Baseline recurrences and weight decay follow Phase 1. |
| EWC | Momentum SGD with rates 0.0072 or 0.0144, adding \(sW(q-q_*)\) to the single data gradient, \(s=0.05\) or 0.5. |
| Fisher mass | \(h=0.06\) or 0.12, \(\alpha=1\) or 8. Unit-mass ablation uses the selected \(h\). |
| GENERIC | Hold the validation-selected mass and \(h\) fixed, then choose \(\lambda\) from 0.01, 0.05, 0.2, 1.0. Gate 1 for new tasks. Gate 0 is the exact mass-only ablation. |
| Energy matching | \(h=0.06\) or 0.12, \(\gamma=0.03\) or 0.3. All three arms use the same selected pair. |

B2 chooses glide \(h\in\{0.06,0.12,0.18\}\) and AdamW \(h\in\{0.0025,0.005,0.01\}\) by highest mean terminal clean Task 2 accuracy over both regimes on validation, with the first rate winning ties. Select the B2a threshold from 0.04, 0.08 and 0.12 by fewest validation clean alarms, most first-corrupted-block detections, then highest threshold. The B2b verifier threshold is fixed.

| Claim | Required result on held-out seeds |
|---|---|
| Integrity | Check both evaluator pins before and after each run. Reject source or lock mismatch, nonfinite endpoints, incorrect force count, bad midpoint, energy mismatch, or failed mechanics probe. A valid negative scientific result must still be reported. |
| B2a and B2b separately | Require at least 11/12 dirty windows with delayed detection, at least eight changed post-onset updates, both f64 true-origin errors below \(10^{-11}\) and endpoint accuracy within 1 point of the target. Allow at most 1/12 clean alarms. Require at least 11/12 dirty windows with independently scored damage of at least 2 points before rewind. Require mean immediate advantage of at least 2 points over **each** AdamW-skip and glide-skip, mean equal-budget advantage of at least 2 points over AdamW-skip, and clean live advantage of at least 1 point over always-skip. B2a additionally requires the passive stealth check above. Any failed component rejects that regime's full claim. |
| FisherMass and TrueMetriplectic separately | Require mean live accuracy at least 90% and within 2 points of the best tuned baseline or better. For BWT, require paired superiority by more than 2 points versus *each* tuned baseline and the matched ablation (unit mass for FisherMass, selected mass for GENERIC). On 300 test cases, drop ties at exactly six excess correct, then compute the upper tail of the fair-binomial sign test across seeds. Each contrast requires \(p\le 0.05/9\). Failure of any condition rejects that mechanism's retention-with-adaptation claim. |
| EnergyMatched | Require positive common removal on every held-out step, no cap for the fixed-strength claim, and absolute realized-removal residual at most \(10^{-12}+5\cdot10^{-11}D\) for every arm. Test high versus low at the same +2-point margin and \(0.05/9\). Require high's live mean within 2 points of low or better. This is the ninth contrast. Return all three arms against the tuned baselines descriptively. |
| Compute parity and speed | One loss force per DKD step is mandatory, including reversal. Report the actual wall-clock ratio separately. Five interleaved, synchronized 48-step runs per method on the first test seed, each warmed for eight calls, compare against momentum SGD. Median paired ratio \(\le1.05\) is the hardware-specific speed target for each method. The three jointly coordinated energy arms also report joint elapsed time. |

The numerical probe thresholds are endpoint error below \(10^{-13}\) on analytic one-step maps, canonical form residual below \(10^{-8}\) by centered differences, and 500-step f64 quadratic reversal error below \(10^{-11}\). Numerical failures require diagnosis before scientific attribution. Finite experimental failures stay in the record. [Physics assertions](../eval_v2/physics.py) · [Outcome calculator](../eval_v2/recovery.py) · [Retention calculator](../eval_v2/retention.py)
