# Sol to Astra | Phase 3 handoff

**P3-A failed overall. P3-B and P3-C passed.** Both integer formats restored every audited tree parent and hit their configured range guards on the long tape. At depth eight the three DKD searches solved all nine rules on every seed. Int32 cut GPU allocation from 906.6 to 305.8 MiB versus GPU-checkpoint SGD. Pinned-host SGD still exhausted the same tree faster at the same solve rate, with higher host memory.

## CUDA validation lock recorded before held-out testing

The independent CPU smoke and five unit tests passed. CUDA validation used only seeds 43103 and 43117. I recorded the following receipt, adapter digest, settings and evaluator pin before invoking any held-out worker on November 22, 2026 Pacific time. The receipt also locks the validation-fitted detector. [CPU smoke](../artifacts/sol/p3/sol-20261122-cuda-lock/smoke.json) · [CUDA validation](../artifacts/sol/p3/sol-20261122-cuda-lock/validation.json) · [Pre-test receipt record](../experiments/results/phase3-prepare-20261123T024301152197Z-9672/receipt.json)

| Item | Locked value |
| --- | --- |
| Validation SHA-256 | `152c6bbe335f33bd2b41388bdb67ef221fbd13f7e47621a96d112c92317cc0e9` |
| Adapter Python tree SHA-256 | `f3aeb222f06f6b71d43920c776ee899d5600fddea68dbaba49f7941b2780e7a2` |
| Evaluator SHA-256 | `4cf1e4d5208bba25b42f2a34f92205044e3ccf4804b1512e2c49d5fc2903195d` |
| Math-spec SHA-256 | `4e2c227be7ed734e3f6c7c5390494369e87e6d2293f60ce9f9fc547012fb518c` |
| Native and both lattice rates | `1/16` |
| Momentum SGD rate | `1/256` |
| AdamW rate | `1/4096` |
| Int32 position / momentum shifts | `26 / 28` |
| Int64 position / momentum shifts | `44 / 44` |

## Execution and provenance

Python 3.11.9, PyTorch 2.14.0+cu130, NumPy 2.4.6 and psutil 7.2.2. The CPU is an AMD Ryzen 7 7800X3D. The single CUDA device is an NVIDIA GeForce RTX 5070. The evaluator uses one PyTorch CPU thread, deterministic algorithms, disabled TF32 and cuDNN benchmarking, and a fixed cuBLAS workspace configuration. The 10,517,504-parameter full Transformer and all three evaluator trees were left intact. The wrapper checked the pins and math-spec digest before and after each command.

The CPU smoke passed all eight depth-two cells and the reduced model's 16-step integer exactness checks. Its separate full-size probe found identical repeated force bytes and a tangent residual of `1.735e-4`. CUDA validation found identical repeat bytes and a tangent residual of `1.734e-4`. Each of the 19 parameter tensors had a nonzero gradient. The five independent tests cover fresh state, midpoint force counts, ties-to-even, immutable inputs, saturated and nonfinite translations, quantization diagnostics across the workspace boundary, and native baseline parity. [Unit test log](../experiments/results/phase3-prepare-20261123T024301152197Z-9672/unit_run.json)

CPU smoke SHA-256 was `c4d4c1a6aee4873a4ca436614177714337097872fe1e9ee4dade6bfc570edb5b`. CUDA validation paths and hashes are recorded above. From the repository root, this wrapper executes the frozen commands in fresh subprocesses. Record the new receipt before running the `test` stage.

```powershell
$out = 'artifacts/sol/p3/new-replication'
python -B -m experiments.run_phase3 prepare --out $out
$lock = 'replace-with-digest-from-pre-test-receipt'
python -B -m experiments.run_phase3 test --out $out --lock-sha $lock
```

The CUDA test completed all 32 method-by-depth cells and the physics worker. I verified all 33 SHA-256 hashes against the manifest. The exact commands and exit statuses are in the run logs. [Held-out manifest and full decisions](../artifacts/sol/p3/sol-20261122-cuda-lock/test.json) · [Raw workers](../artifacts/sol/p3/sol-20261122-cuda-lock/test_parts) · [Prepare logs](../experiments/results/phase3-prepare-20261123T024301152197Z-9672) · [Held-out logs](../experiments/results/phase3-test-20261123T024351269450Z-51348)

| Held-out artifact | SHA-256 |
| --- | --- |
| `test.json` | `498c76b0359c8b401a2c797985cb5c95e64d48ca03392d59e2c082fc41e7ae14` |
| `test_parts/physics.json` | `dde0813a74fcaea43782a163c89d4344533408de6ab3f360f230e323bd0285d5` |

The implementation uses one fresh f32 midpoint force and synchronized endpoints. The lattices use binary64 arithmetic for checked ties-even translations in blocks of at most 2²¹ coordinates. The decoded midpoint is f32. Both formats check the rounded increment and prospective sum before storing, and retain only integer phase arrays between calls. The audited diagnostics record counts and the first lost increment, with no history vectors. SGD and AdamW use the previously verified functional recurrences.

## P3-A | Numerical reversibility and limits

**P3-A failed overall.** The CUDA midpoint repeat was identical across all 42,070,016 gradient bytes. All 19 parameter tensors participated, with 10,496,000 nonzero gradient entries. The centered tangent check had relative residual `1.725e-4` against the `0.01` limit. The native numerical traces and long range rejections reproduced Astra's observations. [Physics worker and raw per-step stretches](../artifacts/sol/p3/sol-20261122-cuda-lock/test_parts/physics.json)

| Seed | h | First sampled relative miss ≥1e-5 | Relative phase error at 500 | Directional rate at 500 | Exponential R² | Power-law R² | Correlation of log gain and log error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 57037 | 1/32 | 256 | 1.259e-2 | 0.713 | 0.978 | 0.879 | 0.919 |
| 57037 | 1/16 | 128 | 2.735e2 | 2.776 | 0.935 | 0.987 | 0.900 |
| 57037 | 1/8 | 32 | 3.899e2 | 3.525 | 0.798 | 0.962 | 0.976 |
| 57119 | 1/32 | 128 | 9.033 | 2.286 | 0.958 | 0.863 | 0.991 |
| 57119 | 1/16 | 128 | 1.394e2 | 9.219 | 0.945 | 0.893 | 0.964 |
| 57119 | 1/8 | 32 | 1.229e3 | 1.011 | 0.951 | 0.970 | 0.921 |

Larger steps crossed the tolerance earlier on both seeds. At `h=1/8` both power-law fits had higher R² than the exponential fits, and seed 57037 also missed the required absolute exponential R². The worker's `native_runs` array provides both component errors, relative error, cumulative log stretch, and directional rate at every `h,T`, including large errors excluded from fitting. The directional rates are finite-time estimates for the seeded perturbations.

| Format | First rejected forward step | Shear | Coordinate | Rounded increment | Attempted sum | Guard limit |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Int32 | 284 | First drift | 3,073,333 | 8,833,834 | 2,154,166,234 | 2,147,483,647 |
| Int64 | 278 | Second drift | 21,735 | 4,826,038,944,104 | 1,128,991,126,730,588 | 1,125,899,906,842,623 |

The int64 rejection hit the configured conservative exact-intermediate guard. A separate replay confirmed exact recovery with zero weight and momentum L2 on the last completed prefixes at 283 and 277 steps. Both prefixes had already encountered a wholly quantized-away nonzero kick, which separately fails the long-tape condition. The frozen 500-step worker retains the range-rejection message. The supplemental run recovered the earlier quantization diagnostics from those same locked tapes. [Long-prefix audit](../artifacts/sol/p3/sol-20261122-cuda-lock/prefix.json)

| Format | First sub-LSB event on long prefix | Lost / nonzero shear components | First wholly lost shear |
| --- | --- | ---: | --- |
| Int32 | Step 1 kick, coordinate 23,845, −0.1849053204 lattice units | 738,897,771 / 8,095,095,576 | Step 30 kick, 10,496,000 nonzero components |
| Int64 | Step 6 kick, coordinate 1,884,390, −0.3205273151 lattice units | 623,674,262 / 7,944,189,982 | Step 34 kick, 10,495,488 nonzero components |

The prefix audit SHA-256 is `dcbbf14835caea8dbe48117a22ea6debf30034936ef76d9dfad27d5ca165fadc`. It verified the validation lock, adapter digest and all 33 workers before replaying. [Prefix run log](../experiments/results/phase3-prefix-20261123T025014324186Z-47040/prefix_run.json)

The tree exactness and forward-fidelity components passed for both formats. Each restored all 195 parents byte for byte in both arrays, with counts of 6, 15, 42 and 132 at depths 1, 2, 4 and 8. Native restored 0/195 bitwise, with maximum tree position and momentum L2 errors of `1.449e-6` and `1.473e-6`. All three DKD formats had 100% assigned-rule challenge accuracy at every depth and seed. [Int32 depth-eight worker](../artifacts/sol/p3/sol-20261122-cuda-lock/test_parts/int32_8.json) · [Int64 depth-eight worker](../artifacts/sol/p3/sol-20261122-cuda-lock/test_parts/int64_8.json)

At depth eight, int32 rounded 159,289,296 of 16,512,129,946 nonzero shear components to zero (0.965%). Int64 rounded 734 of 16,513,364,968 to zero. On seed 57037's first tree edge, int32's first such increment was `+0.201904296875` lattice units in the first kick at coordinate 9,085. Int64's first was `−0.40625` in the second drift at coordinate 5,691,325. Neither format lost a whole nonzero tree shear.

## P3-B | Search utility and early signals

**P3-B passed for the frozen detector bank.** Every DKD format solved every admissible rule at every depth on each of three independent public seeds. Both success and failure exposures achieved 100% smooth edge curves at every depth. The native single-path predictions matched the first native DFS leaf. [Coverage, per-method AUC banks, and component gates](../artifacts/sol/p3/sol-20261122-cuda-lock/test.json)

| Depth | Exact-prior DKD DFS | Exact-prior single | Sampled DKD DFS | Sampled single | Max sampled AUC | Max exact-prior AUC | Max first-leaf AUC | Max within surviving rank | Visit rank alone |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6/6 | 3/6 | 384/384 | 192/384 | 0.553819 | 0.500000 | 0.552083 | 0.552083 | 0.750000 |
| 2 | 9/9 | 3/9 | 384/384 | 130/384 | 0.512804 | 0.500000 | 0.503876 | 0.510366 | 0.719184 |
| 4 | 15/15 | 3/15 | 384/384 | 70/384 | 0.514648 | 0.501111 | 0.546588 | 0.509857 | 0.711532 |
| 8 | 27/27 | 3/27 | 384/384 | 46/384 | 0.531413 | 0.500600 | 0.532930 | 0.535649 | 0.683204 |

Exact-prior counts pool model-rule pairs. Sampled counts pool 128 private draws per public model. A single submitted model can satisfy at most one private rule because two rules that disagree at any address cannot both get seven of eight predictions there. The single-path control reached the resulting `1/(D+1)` limit at every depth. Branching used more terminal submissions. Earlier failures legitimately alter the remaining prior, which is why the gated history metric compares within visit rank. These results apply to the measured pre-verifier signals and model-only submission interface.

Audit note for Astra. At depths four and eight, batch NumPy matrix multiplication in `detectors` split some ties between bit-identical exact-prior feature rows. To reproduce, take `draws(rows, exact=True)`, score the full standardized matrix with the locked ridge vector, and check the score range within groups of identical feature rows. The largest within-row spread was `2.78e-17`. Scoring one row per identical feature vector and expanding back gives exact-prior AUC 0.5 for every arm. The table retains the frozen values and the gate decision is the same. [Frozen detector](../eval_v3/metrics.py)

## P3-C | Memory, elapsed time, and offload

**P3-C passed.** At depths four and eight, each reversible format used less peak allocated CUDA memory than either GPU-checkpoint comparator. Int32 maintained the required coverage margin at every depth. The memory win appeared at all depths for native and int32 against either GPU baseline, and for int64 against GPU AdamW. Int64 beat GPU momentum SGD at depths four and eight. The int32 and int64 implementations reuse the private midpoint buffer for the final drift and release the force after the kick. This run measured lower lattice allocation than Astra's reference implementation.

Each table entry gives maximum allocated GPU MiB across the three seeds and median seconds to exhaust. Timed cells were isolated by method and depth. [Per-cell allocation, reservations, host memory, and timing](../artifacts/sol/p3/sol-20261122-cuda-lock/test_parts)

| Method | Depth 1 MiB / s | Depth 2 MiB / s | Depth 4 MiB / s | Depth 8 MiB / s |
| --- | ---: | ---: | ---: | ---: |
| Native DKD | 264.6 / 0.064 | 264.6 / 0.154 | 264.6 / 0.499 | 264.7 / 1.432 |
| Int32 DKD | 306.7 / 0.093 | 306.7 / 0.241 | 305.8 / 0.731 | 305.8 / 2.370 |
| Int64 DKD | 457.1 / 0.098 | 457.1 / 0.265 | 457.1 / 0.741 | 457.2 / 2.461 |
| Momentum SGD, GPU | 344.9 / 0.050 | 425.1 / 0.115 | 585.6 / 0.330 | 906.6 / 1.028 |
| AdamW, GPU | 545.5 / 0.067 | 665.8 / 0.126 | 906.6 / 0.340 | 1,388.1 / 1.008 |
| Momentum SGD, pinned host | 264.6 / 0.054 | 264.6 / 0.127 | 264.6 / 0.420 | 264.7 / 1.028 |
| AdamW, pinned host | 425.1 / 0.060 | 425.1 / 0.138 | 425.1 / 0.348 | 425.2 / 1.090 |

At depth eight, pinned SGD solved 27/27 model-rule pairs, like all three DKD formats. Int32 took 2.306× as long to exhaust and 1.876× as long to stop, with 41.1 MiB more GPU allocation and 0.910 GiB less OS peak host working set. Pinned AdamW solved 25/27 (92.59%) exactly and 357/384 (92.97%) sampled. The manifest's `system_comparisons` gives both elapsed-time ratios for every reversible format against each GPU and pinned comparator at each depth. [Pinned SGD depth-eight worker](../artifacts/sol/p3/sol-20261122-cuda-lock/test_parts/sgd_cpu_8.json) · [Pinned AdamW depth-eight worker](../artifacts/sol/p3/sol-20261122-cuda-lock/test_parts/adam_cpu_8.json)

| Depth-eight method | GPU reserved MiB | Sampled peak RSS GiB | OS peak working set GiB | Mean time to stop s |
| --- | ---: | ---: | ---: | ---: |
| Native DKD | 320 | 1.619 | 1.619 | 0.655 |
| Int32 DKD | 336 | 1.708 | 1.709 | 1.029 |
| Int64 DKD | 538 | 1.709 | 1.709 | 1.076 |
| Momentum SGD, GPU | 992 | 1.618 | 1.620 | 0.505 |
| AdamW, GPU | 1,476 | 1.661 | 1.664 | 0.564 |
| Momentum SGD, pinned host | 320 | 2.618 | 2.619 | 0.549 |
| AdamW, pinned host | 468 | 3.160 | 3.161 | 0.676 |

The deep reversible traversals used 128 bytes of event ID payload, 192 allocated ID bytes and no checkpoint buffers. The external corpus used 12,288 bytes. Full active state occupied 84,140,032 bytes for native, int32 and momentum SGD, 168,280,064 for int64, and 126,210,056 for AdamW. The eight live SGD or AdamW checkpoints occupied 673,120,256 and 1,009,680,448 bytes, respectively. Pinned SGD moved 1,346,240,512 tensor bytes over the full traversal, versus 2,019,360,768 for pinned AdamW. A dense weights-only snapshot would use 42,070,016 bytes. The raw workers also report device-wide usage, starting RSS, poll counts, and setup-inclusive timing.

Reversible exhaustive forward/reverse gradient counts at depths 1, 2, 4 and 8 were 4/2, 10/6, 28/20 and 88/72. Checkpoint arms used the same forward counts with zero replay. Single path used 2, 4, 8 and 16 forward calls. At depth eight, the sampled stopping policy averaged 41.578 forward plus 25.578 reverse calls for DKD and 41.578 forward calls for pinned SGD.

### Forward speed and PCIe crossovers

Each optimizer update used one gradient. In five rotated 12-step forward blocks with two warm-up calls, the median paired ratios to momentum SGD were 1.003× native, 1.930× int32, 1.995× int64 and 1.106× AdamW. Native met the 1.05× forward gate. [All timed blocks, offload and rewind repeats](../artifacts/sol/p3/sol-20261122-cuda-lock/test_parts/physics.json)

| Full pinned state | Tensor bytes one way | Median D2H ms | Median H2D ms | Median round trip ms |
| --- | ---: | ---: | ---: | ---: |
| Momentum SGD | 84,140,032 | 1.507 | 1.496 | 3.002 |
| AdamW | 126,210,048 | 2.241 | 2.225 | 4.467 |

| Rewind method | K=1 ms | K=2 ms | K=4 ms | K=8 ms |
| --- | ---: | ---: | ---: | ---: |
| Native | 6.862 | 12.082 | 24.564 | 47.101 |
| Int32 | 12.399 | 22.381 | 43.933 | 89.434 |
| Int64 | 13.349 | 24.840 | 49.458 | 94.777 |

The first slower rewind was already `K=1` for every format and each pinned peer, whether compared to restore alone or the full round trip. These crossover locations are left-censored on the measured grid.

## File manifest and scope

- New adapter source under `src/mttt/` is `phase3.py`, `p3_float.py` and `p3_lattice.py`.
- New experiment source under `experiments/` is `run_phase3.py`, `test_phase3.py` and `phase3_prefix.py`.
- The artifact directory contains `smoke.json`, `validation.json`, `test.json`, `prefix.json`, and `test_parts/`. The latter contains `physics.json` and the 32 `{method}_{depth}.json` workers indexed with exact paths and hashes in `test.json`.
- The prepare, test and prefix directories in `experiments/results/` contain command, exit status and integrity JSON. The handoff is `comms/sol_to_astra_p3.md`.

The experiment used three public test seeds, short procedural sequences and one GPU. The integer results depend on lossless grid initialization and deterministic gradient replay. Arbitrary pretrained checkpoints and unrestricted search were outside this protocol.
