# Phase 3 | Reversible Transformer weight-space search

This is a numerical and systems experiment on speculative adaptation. Phase 2's native replay tolerance miss on the small MLP motivates a test for exponential perturbation growth. A tolerance crossing by itself cannot establish chaos. Phase 3 has no continual-retention or poisoned-label arm.

## Model, data and units of replication

The model is a dense, bias-free, four-block pre-norm causal Transformer of width 512, with eight attention heads, GELU feed-forward width 1,536, learned input and position embeddings, and an untied 16-logit output projection. All **10,517,504** parameters participate in autograd and are passed to the optimizer. Attention uses explicit masked matrix products and softmax. Layer normalization has no affine parameters. No dropout or data download is involved. The initial model is randomly initialized on a shared \(2^{-20}\) grid with canonical positive zero. This makes starting-state fidelity testable across number formats. It does not assume an arbitrary pretrained binary32 checkpoint can be encoded losslessly in int32.

The task is to adapt a causal continuation rule from clean, hypothesis-conditioned examples. Sequences have six tokens, including a public address and irrelevant context. Each address admits two continuation symbols. At depth \(D\) there are \(D\) addresses, a public default bit per address, and one latent rule chosen uniformly among the default and the \(D\) possible single-address exceptions. The model must generate the entire continuation code on fresh queries. The allowed search tree is a trie with \(D+1\) length-\(D\) programs, \(D\) genuine forks on its default spine, and \(D+D(D+1)/2\) distinct edges. Its branching factor is constrained by the task. The D=8 test explores every one of its nine admissible leaves.

At each edge, take two steps on separate four-example support batches labeled with the proposed continuation. Both choices get the same feature distribution. Score loss, entropy and top-1 disagreement on a third fixed probe batch before, between and after the steps. The terminal challenge uses eight separate sequences per address. A verifier accepts only when **at least seven of eight predictions at every address** match its private rule. The verifier receives model predictions and a secret, with no access to the proposed path ID. Returning the same weights at every leaf cannot change its answers.

Public validation seeds are 43103 and 43117. Public test seeds are 57037, 57119 and 57331. Each seed defines a new model and procedural data. The private decisions are evaluated under the exact uniform prior and under 128 seeded draws per public problem. Repeated draws on one model are not independent Transformer replications. Secret generation is reproducible, not a security boundary. Candidate code may consume only phase state, the current force closure and frozen numerical coefficients.

## P3-A | How reverse error grows

With unit mass, a DKD step for batch \(b\) takes the form

\[
 x=q+hp/2,\qquad p'=p-hg_b(x),\qquad q'=x+hp'/2.
\]

There is one fresh gradient and full endpoint momentum. Flip \(p'\), use the same batch and positive step, and the reverse drift gives \(x\), the kick gives \(-p\), and the last drift gives \(q\). Reverse the order of batches and flip the final momentum. The force may change between batches. Reusing a previous batch's gradient is inadmissible.

For \(z=(q,p)\), write an implemented substep as \(F_i(z)+r_i\). Over the forward and reflected reverse composition of \(m\) substeps, local Lipschitz constants \(L_j\) on a neighborhood containing both paths give the bound

\[
 \|e_m\|\ \le\ \|e_0\|\prod_{j=1}^m L_j+
             \sum_{i=1}^m\|r_i\|\prod_{j=i+1}^m L_j .
\]

This follows by iterating \(\|e_{i+1}\|\le L_{i+1}\|e_i\|+\|r_{i+1}\|\). In the linearized regime, each injected error is transported by the subsequent Jacobian product. Even a reversible exact map can amplify arithmetic noise. Neither exponential amplification nor the direction or size of the roundoff injections is assumed.

Along the native forward trajectory propagate a seeded unit phase perturbation \((u,v)\). For each batch,

\[
 a=u+hv/2,\qquad v'=v-h\nabla^2 U_b(x)a,\qquad u'=a+hv'/2 .
\]

Compute the Hessian-vector product by double backward, measure \(s_i=\|(u',v')\|_2\), and renormalize. Store each \(\log s_i\), cumulative \(G_T=\sum_i\log s_i\), and \(G_T/(Th)\). The HVP is diagnostic work and is excluded from optimizer timing. A centered finite difference of the actual one-step map must agree within 1% in relative phase L2. This directional finite-time estimate is neither a rigorous bound on the operator norm nor an asymptotic Lyapunov exponent. [Panichi and Turchetti on finite-time Lyapunov and reversibility errors](https://arxiv.org/abs/2505.03921)

On the first two test seeds run the same changing, conflicting sequence tape with \(h=h_*/2,h_*,2h_*\) and \(T=8,16,32,64,128,256,500\). Use nonzero initial \(p=\operatorname{sign}(q_0)2^{-14}\). Rewind from each actual endpoint in a fresh engine. Return separate absolute L2 errors and the relative phase error \(\rho_T=\|(\Delta q,\Delta p)\|_2/\max(1,\|(q_0,p_0)\|_2)\). The cliff threshold is \(10^{-5}\). A first miss is located on the sampled grid; absence of a miss through 500 is right censoring.

**Growth criterion.** At \(2h_*\) on each of the two seeds require an initial pass, a later threshold crossing earlier than at \(h_*/2\) (a censored smaller step is allowed), and at least four samples with \(0<\rho_T<0.1\). Fit both \(\log\rho_T\sim T\) and \(\log\rho_T\sim\log T\) on the same samples. Require positive exponential slope, \(R^2_{\rm exp}\ge0.85\), \(R^2_{\rm exp}-R^2_{\rm power}\ge0.05\), correlation of \(G_T\) with \(\log\rho_T\) at least 0.8, and observed \(G_T\ge\log4\) on at least one fitted sample. Other rates and both fitted curves are returned regardless of this decision.

## P3-A | Checked int32 and int64 translations

Represent \(q=2^{-s_q}Q,\ p=2^{-s_p}P\). Keep both arrays authoritative integers of the selected width. Let \(r\) be round-to-nearest with ties to even.

\[
\begin{aligned}
 X &= Q+r\!\left((h/2)\,2^{s_q-s_p}P\right),\\
 P'&=P+r\!\left(-h\,2^{s_p}g_b(\operatorname{f32}(2^{-s_q}X))\right),\\
 Q'&=X+r\!\left((h/2)\,2^{s_q-s_p}P'\right).
\end{aligned}
\]

Shifts are global per state component and constant for an episode. On validation, measure the largest position and momentum magnitudes over both seeds and 64 native steps at \(h_*,2h_*\), including initial states. For either component with observed maximum \(a\), choose

\[
s=\min\bigl(s_{\max},\lfloor\log_2[L/(4\max(a,2^{-20}))]\rfloor\bigr).
\]

For int32 use \(L=2^{31}-1,\ s_{\max}=30\); for int64 use the conservative exact-intermediate limit \(L=2^{50}-1,\ s_{\max}=44\). Reserve headroom for reflection and addition. Require bit-identical decoded f32 initialization. Compute scaled increments in binary64, with dyadic step sizes, and verify that both rounded increment and prospective sum are finite and within \([-L,L]\) **before** storing any integer result. An int64 sum of two such operands is representable exactly in binary64. No saturation, wrap, adaptive rescaling, or roundoff-history buffer is allowed. The reference uses blocks of \(2^{21}\) coordinates to bound workspace. Reject deliberately out-of-range steps in a negative control.

Because \(r(-x)=-r(x)\), reflected drifts undo the original integer translations, and the reverse kick sees the identical integer midpoint. Bit-identical f32 loss gradients therefore give bit-exact recovery. Native, int32 and int64 runs use the same deterministic f32 model and no training-mode stochastic operations. Use `torch.use_deterministic_algorithms(True)`, one CPU thread, TF32 off, cuDNN benchmarking off and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization. A fresh repeated gradient at an identical cloned midpoint must have identical bytes and a nonzero gradient in every parameter tensor. Exactness is checked per backend. [Rein and Tamayo on integer-shear reversibility](https://arxiv.org/abs/1704.07715)

The dedicated correctness traversal retains independent auditor snapshots, unwinds **every distinct edge**, including the last path back to the root, and compares every position and momentum byte to its original parent. None of these snapshots exists in the systems traversal. On the first held-out seed, both integer formats must also recover the full Transformer at \(T=500\). Report the first nonzero sub-LSB increment rounded to zero with shear, location, coordinate and value, the number of such components, and any completely lost nonzero shear. Individual sub-LSB events are quantization, not evidence by themselves of failure. An overflow or a completely lost nonzero shear fails that arm. Return the observed failure location.

**Exactness and forward fidelity.** For *each* format require bit identity and zero L2 error on every audited parent and the long replay. At every held-out depth and seed, intended-rule token accuracy must be at least 90%, and neither it nor exact-prior solve rate may trail native f32 by more than one percentage point. Integer replay preserves the lattice trajectory, which can differ from the native forward trajectory. P3-A requires these tests, the repeat-gradient check, and the native growth criterion. Mechanical passes remain reportable when the growth criterion fails.

## P3-B | Does branching help without an early loss giveaway?

DFS visits the default code first, then the exceptions from deepest to shallowest. Reversible engines physically rewind the suffix to the fork. Checkpoint engines restore that fork's complete optimizer state. Single-path DKD performs exactly the first path and receives one terminal evaluation. Its predictions must match the first leaf of multi-path native DKD. Different rules cannot both obtain seven of eight hits at an address where they disagree. Thus *any* one submitted model, even with more optimization steps before submission, can pass at most \(1/(D+1)\) of the uniform private rules. The claim concerns search with a model-only verifier and these permitted update rules. A symbolic rule table with direct access to an output-submission interface would define a different systems baseline.

The driver executes the identical all-fail traversal once. For any private rule, a fail-only history identifies exactly the prefix ending at its first passing leaf. The evaluator can reconstruct success, synchronized elapsed time to stopping and cost without giving secret labels or decoded outcomes to the optimizer. This marginalization is valid for the locked policy whose only terminal action is stopping at the first pass. Record full-exhaustion time as well. At each depth require all three DKD formats to improve exact-prior solve rate by at least **20 points** over both measured single-path DKD and the one-submission upper bound, and to improve solve rate on sampled private draws.

For each leaf use only pre-verifier signals. At the first edge, halfway through a path, and at the end, record nine features: mean initial and final edge probe losses, last edge's final loss, maximum initial edge loss, maximum and mean per-step loss change, mean terminal entropy, mean initial disagreement, and last-edge final disagreement. A ridge detector with unit penalty is fitted on standardized features from validation only. Test each scalar in both orientations and the fitted score. Compute tied-rank ROC-AUC against **actual terminal model success**, including leaves where the intended rule failed to be learned. Report the maximum of AUC and one minus AUC among the bank. No test-time threshold fitting is used.

Require the maximum to be at most **0.60** on sampled private draws, under the exact uniform prior, and for pre-first-verifier signals on the first visited leaf. Also require the same threshold for pair-weighted within-visit-rank AUC on histories still alive after earlier failures. Report visit rank's standalone AUC to expose the information that the stopping history supplies. Undefined primary AUC fails. Among both successful and unsuccessful leaf exposures, at least 90% of visited edge curves must have strictly lower final than initial CE with each increment \(\le10^{-4}\). All requirements apply at every tested depth. Passing bounds this detector bank on this symmetric task. It cannot prove the absence of every conceivable shortcut.

## P3-C | Space and time

The measured arms are native/32-bit/64-bit reversible DKD, momentum SGD (0.9) and AdamW (betas 0.9/0.999, epsilon \(10^{-8}\), decay 0.01) with either GPU or pinned-host checkpoints, and single-path DKD. Save at unresolved binary forks only. All \(D\) forks are simultaneously live on the default spine. GPU checkpoints retain immutable parent buffers without a gratuitous memcpy; CPU checkpoints copy tensors out and back over the device link and keep the step count. Restore all moments. The no-backtrack arm holds no replay tape.

For \(N\) parameters, active native and int32 phase state each cost \(8N\) bytes; int64 costs \(16N\). AdamW needs \(12N\) bytes plus its step counter. A weights-only checkpoint costs \(4N\). Full SGD or AdamW rollback requires its complete state. Reversible working parameters have depth-independent size, plus up to \(8KD\) bytes of event-ID payload and small control metadata. Report actual ID allocation and the immutable support/probe corpus separately. Arithmetic conversions, gradients, activations, allocator reservations, and process memory are **measured** rather than inferred from these formulas.

Use a fresh CUDA process for each method/depth cell, in fixed depth order with the same three seeds in each cell. Warm a gradient using validation data. Reset peak allocator stats after cache release. Report CUDA peak allocated and reserved bytes, sampled device-wide usage (includes other processes), process RSS at start and sampled peak at 3 ms polling, sample count, and Windows lifetime high-water working set when available. Track peak live checkpoint bytes, actual host transfer payload and synchronized allocation/copy times. The OS high-water mark includes process setup and earlier seeds in the cell. Sampled RSS can miss a short peak. Report primary wall time through exhaustion from initialized state, time to terminal success including failures, and setup-plus-search wall time. Timing includes clean probes and terminal decisions. Instrumented correctness traversals with extra parent copies are excluded.

On the first held-out seed time five rotated, synchronized 12-step blocks per method with two untimed warm-up calls. Native DKD's median paired wall-time ratio to momentum SGD must be \(\le1.05\); integer times are reported separately. Record pinned D2H and H2D times for complete SGD and AdamW states, one warm-up plus five repeats with preallocated buffers. Independently time three real reversed suffixes for every \(K'\in\{1,2,4,8\}\), excluding forward setup. For each DKD format and peer report the first measured \(K'\) whose median rewind exceeds median H2D restore alone and the D2H+H2D round trip. Report all depths where peak allocated VRAM beats each GPU-checkpoint baseline and time ratios versus both offload schemes. A crossing at the first grid point is left-censored; a missing crossing is unobserved on the grid.

**Systems pass.** At D=4 and 8 all three reversible formats must have lower measured peak allocated VRAM than both GPU checkpoint baselines. Int32 exact-prior solve rate must be within one point of each checkpoint comparator or better at every depth. Require native forward speed parity and complete cells. Slower end-to-end execution or expensive integer reversals must be disclosed even when memory savings pass.

## Freeze and provenance

Only CUDA validation chooses rates: DKD \(1/32,1/16\); momentum SGD \(1/1024,1/256\); AdamW \(1/16384,1/4096\). Train and score the depth-four tree on validation. First favor the 90% smooth-loss criterion in both outcome classes, then higher mean intended-rule accuracy, then lower mean final edge-probe loss, then grid order. Lock the selected rates, both pairs of shifts, and the fitted detector **before** any held-out evaluation. CUDA tests require the recorded validation file digest, matching evaluator and factory-package Python source hashes, and recompute selections from the raw validation records. There are no external datasets or executable dependencies on earlier evaluators.

Return all raw cells, even failures. A reported range or memory failure is an experimental outcome. Hash mismatch, wrong force count, a non-deterministic repeat, or a failed control invalidates the corresponding attribution. The CPU smoke uses a reduced two-block transformer for traversal and physics, plus a full-size repeated gradient and tangent probe. CPU smoke is outside the CUDA claim. [Evaluator entry point](../eval_v3/run.py) · [Decision calculator](../eval_v3/metrics.py)
