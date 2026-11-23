# Metriplectic test-time training | Phase 1

The experiment asks three separate questions. Can we reconstruct a parameter trajectory? Can an online detector decide when reconstruction improves accuracy? Does projecting the brake improve retention at a useful level of adaptation? Each question has its own decision rule below.

## 1. Discrete mechanics

Flatten all trainable weights and biases into \(q\in\mathbb R^d\). Momentum \(p\) has the same coordinates and precision. Phase 1 uses the identity mass, Euclidean norms, fixed \(h>0\), and \(H_b(q,p)=U_b(q)+p^\top p/2\). In the learning experiment \(U_b\) is mean supervised cross-entropy on an ordered, immutable batch \(b\). A changing batch changes the Hamiltonian.

For one Störmer-Verlet step \(S_b\), evaluate both gradients on the same batch, at the two positions shown.

\[
u=p_n-\tfrac h2\nabla U_b(q_n),\qquad
q_{n+1}=q_n+hu,\qquad
p_{n+1}=u-\tfrac h2\nabla U_b(q_{n+1}).
\]

Each kick is a canonical shear because the potential Hessian is symmetric. The drift is a canonical shear because the mass is symmetric. Their composition satisfies \(S_b'{}^\top J S_b'=J\), with \(J=\left(\begin{smallmatrix}0&I\\-I&0\end{smallmatrix}\right)\). The symbolic checker multiplies these Jacobians with independent endpoint Hessians. The physics gate also differentiates the actual adapter by centered differences. [Exact identities](verify_symbols.py)

Let \(R(q,p)=(q,-p)\). From \((q_{n+1},-p_{n+1})\), the first kick yields \(-u\), the drift yields \(q_n\), and the second kick yields \(-p_n\). Hence \(S_b^{-1}=R S_b R\). For batches \(b_0,\ldots,b_{T-1}\), including their original labels and sample order, use this protocol.

1. Keep the endpoint \((q_T,p_T)\) and flip \(p_T\).
2. Run ordinary positive-\(h\) steps in execution order \(b_{T-1},\ldots,b_0\), with \(\gamma=0\) throughout.
3. Flip momentum once more. Compare **both** components to the evaluator's withheld initial state.

This is an exact identity in real arithmetic. Floating-point kicks and drifts discard low bits. Phase 1 measures absolute error for specified trajectories and dtypes. A bitwise inverse and a uniform long-time roundoff bound would be stronger claims. [Symbolic witness](evidence/symbolic.json) · [Float32 and float64 drift curves](evidence/analytic_cuda.json)

The optimizer can retain the current phase state and immutable configuration. The driver supplies exact batch replay. Batch IDs and a frozen data source suffice when the batches are reproducible. Otherwise the inputs, labels, augmentations and relevant RNG decisions must be retained. The evaluator keeps \(q_0,p_0\) for scoring in its own scope. Rollback receives only endpoint state, configuration and the batch stream. A fresh optimizer performs each replay. A single corrupted batch whose original force cannot be regenerated breaks the protocol.

## 2. Dissipative splits

Write \(D_{\tau,G}(q,p)=(q,\exp(-\tau\gamma G)p)\), where \(G=G^\top\succeq0\). Hold \(\gamma\geq0\) fixed within a step. Apply \(D_{h/2,G}\), then \(S_b\), then \(D_{h/2,G}\). The two half brakes use their respective endpoint positions if a future model makes \(G\) position dependent. The implemented \(G\) is frozen. Set \(\gamma=0\) to recover the glide map exactly in the update algebra.

**Variant A, isotropic brake.** Set \(\Gamma=G=I\). A half brake with \(r=\exp(-h\gamma/2)\) pulls the canonical two-form back to \(rJ\). The full split pulls it back to \(\exp(-h\gamma)J\). An anisotropic symmetric PSD \(\Gamma\) still defines a valid damping exponential, but a single conformal factor exists for this half brake only when the exponential is scalar. Momentum reflection commutes with a brake. Inverting positive damping would require an expanding exponential. The checkpoint-free glide test admits only zero gate steps. [Conformal-factor identity](verify_symbols.py) · [Actual split-map probes](evidence/physics_cuda.json)

**Variant B, projected brake.** At the Task 1 pretrained weights compute the observed-label empirical Fisher diagonal

\[
 f_i=\frac1N\sum_{j=1}^N
       \left[\partial_{q_i}\log P_{q_{\rm pre}}(y_j\mid x_j)\right]^2.
\]

Use 96 separate Task 1 examples and square individual scores before averaging. Freeze \(F_{\rm pre}=\operatorname{diag}(f)\). Let \(m\) be its lower median, \(P_L=\operatorname{diag}(\mathbf1[f_i\le m])\) and \(P_H=I-P_L\). Include all ties at \(m\). All-zero Fisher therefore gives \(G=I\). In Variant B set \(G=P_L\), which is its own positive semidefinite square root. \(P_HG=0\) and \(P_H F_{\rm pre}P_L=0\). This defines orthogonality in the eigenspaces of the fitted approximation. Its relation to the full Hessian is empirical.

For a K-FAC extension, an affine layer's column-vectorized weight block would use \(A\otimes C\), with \(A=\mathbb E[aa^\top]\) for activations and \(C=\mathbb E[\delta\delta^\top]\) for per-example output scores. Product eigenvectors have eigenvalues \(\lambda_i(A)\lambda_j(C)\). Apply the same threshold to the pooled approximate spectrum, with coordinates transformed into the corresponding eigenbases. The frozen experiment selects the diagonal estimator only.

**What is metriplectic here?** On \((q,p)\) the brake is mechanical friction and \(\dot H_b=-\gamma p^\top Gp\) during braking. To give a strict energy-degenerate metric bracket, adjoin a scalar heat reservoir \(s\) in unit-temperature coordinates. Set \(E=H_b+s,\ S=s\), extend \(J\) by a zero reservoir row and column, and evolve \(\dot z=J\nabla E+M(z)\nabla S\) with

\[
 M(z)=\gamma
 \begin{pmatrix}
 0&0&0\\
 0&G&-Gp\\
 0&-p^\top G&p^\top Gp
 \end{pmatrix}.
\]

For a covector \((a,v,c)\), its metric quadratic form is \(\gamma(v-cp)^\top G(v-cp)\ge0\). Also \(M\nabla E=0,\ J\nabla S=0\) and \(\dot S=\gamma p^\top Gp\ge0\). During a brake, \(q\) is fixed and \(s\) increases by the kinetic energy lost. The reservoir does not feed back, so the optimizer need only carry \((q,p)\). These identities describe the fixed-batch continuous brackets and exact brake subflows. The discrete Verlet part carries its usual energy error. Switching batches can change \(E\) externally. [Symbolic PSD and degeneracy checks](evidence/symbolic.json)

**A limit relevant to retention.** The irreversible momentum increment lies in the low subspace. The high component of the *position* can still move. For a flat potential and \(p=P_Hp\ne0\), a full split gives \(q'=q+hp\) and \(p'=p\) for any gate. More generally \(P_H(q'-q)=hP_H(p-\tfrac h2\nabla U_b(q))\) for Variant B. Protecting old accuracy does not follow from PSD or Fisher orthogonality. Matching or beating the isotropic brake is an empirical ablation. [Counterexample check](verify_symbols.py)

## 3. Analytic benchmark

Use \(U(x,y)=((x^2-1)^2+(y^2-1)^2)/4+0.06(x-y)^2\), \(q_0=(0.85,-0.72)\), \(p_0=(0.22,0.31)\). For rollback add \(a_i(x+y)\) to \(U\) on step \(i\), with \(a_i=((7i\bmod19)-9)/180\). Use \(h=0.025\). Record \(\|q_{\rm back}-q_0\|_2\) and \(\|p_{\rm back}-p_0\|_2\) at 1, 10, 50, 100, 250 and 500 steps. Measure f32 error with an f64 norm of the difference between stored f32 values. A separate f64 neural probe uses 500 steps, nonzero initial momentum, seed 1729, \(h=0.04\), and a repeated 48-batch sequence from Task 2.

For the energy test, remove the batch tilt and integrate to physical time 8 at \(h=0.08,0.04,0.02\). Measure the peak-to-peak span of \(H\) and of the truncated modified Hamiltonian

\[
 H_{[2]}=H+h^2\left(\frac{p^\top U''(q)p}{12}-\frac{\|\nabla U(q)\|^2}{24}\right).
\]

For kick-drift-kick the leading correction comes from the nested Poisson brackets of kinetic and potential energy. The physical-energy oscillation is predicted to have order two, and adding the correction cancels its leading term. Compute each observed order as \(\log_2({\rm span}_h/{\rm span}_{h/2})\). The test requires both adjacent physical orders in \([1.8,2.2]\), both shadow orders in \([3.5,4.5]\), and at least two changes in the sign of the physical-energy increments on every grid. The traces cover this fixed time interval. [Hairer, McLachlan and Skeel, equation 10](https://www.unige.ch/~hairer/preprints/taka-imada.pdf) · [Measured traces](evidence/analytic_cuda.json)

## 4. Learning protocol and metrics

The frozen synthetic benchmark needs no downloads. All five tasks share a 7-input, 16-hidden-unit tanh MLP and three-way output. The three cluster centers rotate by 25 degrees per task. Five noisy context coordinates make task information partially observable. No task-specific output heads are supplied. Each seed redraws the samples and initial weights. Task 1 uses a common 120-step SGD pretrain. Each task has 384 training and 300 held-out examples. Calibration, Fisher and guard data are drawn separately. Each method starts from the same pretrained parameters and zero optimizer buffers. The poison experiment and the clean five-task stream branch independently from this start.

**Poisoned adaptation.** A policy accepts a 24-batch adaptation window, accumulates an alarm online, then commits or rewinds the entire window at its boundary. The harness runs matched clean and corrupted windows on the same features. Corruption maps every label to \((y+1)\bmod3\). No corruption marker or clean counterpart is supplied to the update or detector. Use the pre-update mean supervised loss and flag the first batch strictly above the pretrained mean plus four population standard deviations from 32 clean calibration batches. Record predictive entropy as a diagnostic. Before any update, changing only the labels leaves predictive entropy fixed for the same features and weights.

On an alarm, glide and zero-gate split variants rewind via \(R\). Standard AdamW and momentum SGD keep their states and continue. Every method sees the same subsequent 32 clean batches. Report first alarm, false alarms on clean windows, uncorrected displacement and accuracy, rollback L2 for eligible actions, immediate held-out accuracy, and clean-follow-up accuracy. A null rollback metric records an unavailable or untriggered action. This benchmark assumes known transaction boundaries and access to supervised adaptation labels. It does not localize an arbitrary attack onset.

**Continual learning.** Train on Tasks 2 through 5 in order, carrying optimizer state, with 48 batches of 32 per task and no old-task replay. Set \(\gamma=1\) throughout new-task training for A and B, zero for glide. Learning rates are 0.12 for all phase methods, 0.0144 for momentum SGD and 0.01 for AdamW. The shared fixed budget is in incoming batches. Phase methods use two gradient evaluations per step, and the baselines use one. These fixed settings test specific optimizers. They do not establish an optimal learning-rate frontier.

Let \(a_{i,j}\) be held-out accuracy on task \(j\) after training through task \(i\), using 1-based task numbering. Report the full \(5\times5\) matrix, \(a_{5,1}-a_{1,1}\) for Task 1 backward transfer, and mean \(a_{j,j}\) over \(j=2,\ldots,5\) for live accuracy. Report arrival-to-live gains \(a_{j,j}-a_{j-1,j}\) and accuracy every 12 updates. Synchronize the device around update blocks and record wall time including force evaluations, excluding model preparation, test scoring and data generation. Report milliseconds per step, ratio to that seed's momentum SGD, and separately the cost of a triggered rewind. Timing is descriptive and hardware dependent.

## 5. Decision rules

The eight paired seeds are 1729, 2718, 3141, 4211, 5501, 6673, 7919 and 8861. The seed is the unit of replication. All test accuracies are fractions. A percentage point is 0.01 in these units. CPU and CUDA runs on these same seeds are reproducibility checks, not extra replicates.

| Question | Decision on this protocol |
|---|---|
| Integrity / implementation | Any evaluator digest mismatch invalidates the run. A wrong force site, split, effective brake, PSD check, native baseline trajectory, or broken order-sensitive replay probe is an implementation failure. Repair in Sol's code and rerun. If the auditor oracle also fails a gate, investigate the harness or numerical environment before attributing a scientific result. |
| A. Reversibility | Require strict absolute error below \(10^{-11}\) in both components at every f64 analytic horizon and at the 500-step neural endpoint. Report the entire f32 curve. Energy-order failure rejects the numerical-integrator validation independently of endpoint recovery. For a verified implementation, a replicated f64 threshold miss on the frozen inputs falsifies the specified finite-precision performance target. |
| B. Finite-horizon poisoned recovery | Require 8/8 glide detections and at most 1/8 clean alarms. Require both rewind errors below \(10^{-11}\) on each poison and immediate accuracy within 1 point of its pre-window value on every seed. Require mean uncorrected glide damage and immediate AdamW damage each at least 5 points relative to pre-window. After 32 clean steps require at least a 2-point mean matched clean-minus-poison accuracy gap for AdamW. Any failed condition rejects this operational B. Report which one failed. Permanent degradation has no finite-horizon acceptance test. |
| C. Retention with adaptation | For each baseline, run a paired one-sided exact sign test that the median Fisher-minus-baseline BWT advantage exceeds 2 points. Exclude ties on integer correct-count differences. Require \(p\le0.025\) for each of the two comparisons. Also require Fisher mean live accuracy at least 80%, mean arrival gain at least 2 points and mean live accuracy within 3 points below each baseline or better. Failing any gate rejects the Phase 1 superiority criterion. A nonsignificant contrast alone does not establish equivalence. |
| Projection attribution | Report the same paired BWT contrast and live difference against isotropic Variant A. A retention claim about the projection itself requires an advantage over that ablation. The two-baseline C decision remains separately visible. |

The test fixes one Fisher projector, brake strength, corruption and budget. A scientific failure stays in the record. Changing the task, mask, thresholds, rates or seeds creates a new experiment. A green process exit on the continual benchmark certifies numerical integrity. Its JSON fields `B_finite_horizon_pass` and `C_pass` carry the scientific decisions.

## 6. Reference observations

The independent implementation from Sol is still required. On the frozen evaluator's RTX 5070 reference run, the 500-step f64 L2 position error was \(5.55\times10^{-16}\) for the double well and \(2.73\times10^{-15}\) for the neural trajectory. The corresponding analytic f32 error was \(9.08\times10^{-7}\). The two measured physical-energy orders were 2.00086 and 2.00074. [Analytic artifact](evidence/analytic_cuda.json) · [Neural artifact](evidence/continual_cuda.json)

Glide detected all eight poisoned windows, raised one clean-window alarm, and restored pre-window test accuracy in all eight poison cases. AdamW retained a mean 4.83-point clean-minus-poison gap after the 32-step follow-up. Fisher reached 97.99% mean live accuracy. Its mean Task 1 BWT advantage was −2.125 points against momentum SGD, +0.292 against AdamW and −3.00 against the isotropic brake. Both primary one-sided sign-test \(p\)-values were 1. B met its finite-horizon criteria. C failed its retention criterion. [Per-seed results and verdicts](evidence/continual_cuda.json) · [CPU replication](evidence/continual_cpu.json)
