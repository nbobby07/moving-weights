# Sol to Astra | Phase 2 handoff

**B2a failed. B2b passed. FisherMass, GENERIC, and the EnergyMatched claim all failed.** Both backends reproduced Astra's frozen scientific decisions. Native f32 passed through 128 on the tested grid and failed at 256. The int64 control returned both state vectors bit for bit through 500. Glide, Fisher mass, and GENERIC met the 1.05× momentum SGD timing gate on this run.

## Pre-test CPU validation lock

Recorded on November 22, 2026 before either held-out invocation. The CPU validation used seeds 10301, 10303 and 10313. The same receipt and adapter source tree are required on both test devices. [Raw validation](../artifacts/sol/p2/sol-20261122-cpu-lock/validation.json) · [Pre-test receipt](../experiments/results/phase2-prepare-20261123T012711349182Z-19816/receipt.json)

| Pinned input | SHA-256 |
| --- | --- |
| CPU validation receipt | `aaa0c8d55729e2be05d9eaf4e9c863a6b55ef261d9104f555ba1dd9d46d8d3e3` |
| Adapter Python source tree | `5c5c9fc0a68bd4fd68d53214773146c29e09e2bd4ee6df855d87a827a3c41992` |
| Phase 2 evaluator | `19aef02de8768e14dec3f1557130b838f217e0684c6acd8f926e6a87f20270f0` |
| Shared Phase 1 evaluator | `a2c2f60d9fe7508f0f5c62701016d488b365d8d68710e72bac41cec7ebdcc89d` |
| Phase 2 math spec | `f0d22f7eff029c9aeb14d97c752c9d908534b545630d82bda877382dd5442721` |

| Protocol | Setting selected from validation |
| --- | --- |
| B2 Glide | `h=0.06` |
| B2 AdamW | `h=0.005` |
| B2a receipt threshold | `0.12` |
| C2 momentum SGD | `h=0.0036` |
| C2 AdamW | `h=0.0025` |
| C2 EWC | `h=0.0144`, `strength=0.5` |
| C2 Fisher mass | `h=0.06`, `alpha=8` |
| C2 unit-mass control | `h=0.06` inherited from Fisher mass |
| C2 GENERIC | `h=0.06`, `alpha=8`, `strength=0.2` |
| C2 energy triplet | `h=0.06`, `gamma=0.03` |

The full grids and validation scores are in the raw validation JSON and the validation summary CSV. C2 uses four candidates per family. It ranks BWT among candidates with mean live accuracy at least 92%, with a live-accuracy fallback. Energy matching first ranks nominal feasibility. B2 rates rank clean Task 2 accuracy across both regimes, and the receipt threshold ranks clean alarms, first-corrupted-block detection, then threshold. All held-out continual arms used 192 gradients. [Validation grid and scores](../artifacts/sol/p2/sol-20261122-cpu-lock/validation_summary.csv)

## Execution and integrity

Python 3.11.9 and PyTorch 2.14.0+cu130 on an AMD Ryzen 7 7800X3D and one NVIDIA GeForce RTX 5070. The runner used deterministic algorithms, one PyTorch CPU thread, and disabled TF32. The wrapper verified both frozen evaluator hashes and the math-spec hash before and after each subprocess, including the failed unit attempt. The frozen held-out commands also rederived the choices from the CPU receipt and checked the adapter tree before scoring. Phase 1 files and both evaluator trees match the checked-in versions.

| Raw artifact | SHA-256 |
| --- | --- |
| [CPU physics](../artifacts/sol/p2/sol-20261122-cpu-lock/physics_cpu.json) | `ebb101f186dd2cf683ad256c062d3caf725265c38249394e3f0cc8f7dac0c1a5` |
| [CPU validation](../artifacts/sol/p2/sol-20261122-cpu-lock/validation.json) | `aaa0c8d55729e2be05d9eaf4e9c863a6b55ef261d9104f555ba1dd9d46d8d3e3` |
| [CPU held-out](../artifacts/sol/p2/sol-20261122-cpu-lock/test_cpu.json) | `ac8f6352a54d718fbd745b8de50dc08b40d4356614566f1e3779c3cc95ab6126` |
| [CUDA held-out](../artifacts/sol/p2/sol-20261122-cpu-lock/test_cuda.json) | `b946d42816f0ca600fe0f6bcbff9db20354a3cfc7a6f97b366d4dacd86c89a7d` |

From the repository root, `prepare` runs the independent tests, frozen CPU physics and CPU validation. After recording the new lock in the handoff, `test` runs frozen CPU and CUDA held-out stages in separate processes. Use a fresh output folder for each replication. [Prepare logs](../experiments/results/phase2-prepare-20261123T012711349182Z-19816) · [Held-out logs](../experiments/results/phase2-test-20261123T012756643729Z-36592)

```powershell
$out = 'artifacts/sol/p2/new-replication'
python -B -m experiments.run_phase2 prepare --out $out
$lock = 'replace-with-digest-from-pre-test-receipt'
python -B -m experiments.run_phase2 test --out $out --lock-sha $lock
python -B -m experiments.phase2_report $out
```

The seven independent test methods passed on both backends. The first attempt failed a CUDA assertion at full brake saturation because the fixture's indexed reduction differed by roundoff from the coordinator's full-width masked sum. The corrected fixture uses the coordinator reduction for the exact saturation assertion and keeps an independent tolerance check for removed energy. No adapter change was needed for that failure. [Failed attempt](../experiments/results/phase2-prepare-20261123T012620816786Z-47908/unit_run.json) · [Passing retry](../experiments/results/phase2-prepare-20261123T012711349182Z-19816/unit_run.json)

## Mechanics and compute

The adapter evaluates one fresh force at the drift-kick-drift midpoint and returns synchronized weights and momentum. The tests check equivalence to fused staggered drifts and the fixed-force correspondence with FSAL velocity Verlet. Across two different force laws, the frozen stale-force control gave a gradient gap of 0.8487 L2 at the same position. Source inspection found only fixed coefficients and the authorized EWC and GENERIC anchors in optimizer instances. Reversal takes the endpoint and ordered replay tape.

All frozen probes passed. The largest 500-step f64 quadratic rollback errors were `2.702e-15` on CPU and `2.891e-15` on CUDA. The maximum canonical-form residuals were `1.569e-11` and `1.559e-11`. The GENERIC half-flow probe balanced potential energy with its reconstructed reservoir, and entropy increased. Projected brakes enforce the available capacity on each arm and the frozen coordinator supplies the shared target.

All continual arms used 192 fresh loss gradients per seed. The separate timing experiment measures five synchronized 48-step blocks per method after eight warm-up calls. The energy triplet uses 576 gradients jointly and averaged 144.74 ms on CPU and 688.71 ms on CUDA across four tasks. Its time includes coordination and per-arm scalar capacity validation. [All timing repeats and task blocks](../artifacts/sol/p2/sol-20261122-cpu-lock/timing.csv)

| Method | CPU median paired ratio | CPU ≤1.05× | CUDA median paired ratio | CUDA ≤1.05× |
| --- | ---: | :---: | ---: | :---: |
| Momentum SGD | 1.0000 | Yes | 1.0000 | Yes |
| Glide | 0.9688 | Yes | 0.9068 | Yes |
| Fisher mass | 0.9952 | Yes | 0.9723 | Yes |
| GENERIC | 1.0474 | Yes | 0.9591 | Yes |
| AdamW | 1.1235 | No | 1.0558 | No |

The speed gate passed for the three phase methods on these blocks. One CUDA momentum-SGD block took 37.76 ms. The per-repeat variation and Astra's different CUDA timing outcome make this a hardware- and run-specific measurement.

## B2 | Retroactive recovery

CPU and CUDA gave the same decisions and measured correct counts. Both regimes compare policies at the common endpoint and after spending the full common gradient budget. A skip policy freezes weights and optimizer state when it alarms. [All policies, return errors, alarms, budgets, and memory](../artifacts/sol/p2/sol-20261122-cpu-lock/b2_summary.csv) · [Paired effects per seed](../artifacts/sol/p2/sol-20261122-cpu-lock/b2_paired.csv) · [Recovery trajectories](../artifacts/sol/p2/sol-20261122-cpu-lock/poison_recovery.png)

| Frozen condition | B2a ramped | B2b terminal |
| --- | :---: | :---: |
| At least 11 delayed true-origin recoveries | Fail, 9/12 | Pass, 12/12 |
| At most one clean rewind alarm | Pass, 0/12 | Pass, 0/12 |
| At least 11 dirty windows with independently scored damage ≥2 pp | Fail, 0/12 | Pass, 12/12 |
| Immediate advantage over AdamW-skip ≥2 pp | Fail, +0.78 pp | Pass, +5.61 pp |
| Immediate advantage over glide-skip ≥2 pp | Fail, +0.08 pp | Pass, +65.58 pp |
| Equal-budget advantage over AdamW-skip ≥2 pp | Fail, −0.94 pp | Pass, +5.44 pp |
| Clean live advantage over always-skip ≥1 pp | Pass, +3.42 pp | Pass, +2.83 pp |
| At most one early passive AdamW loss warning | Fail, 12/12 | |
| Overall | **Fail** | **Pass** |

In B2a the first alarm came after 24 commits on nine seeds and after 32 on three, with 8 and 16 changed post-onset updates respectively. The true origin was step 16. Estimated origins were 16 and 24. For the nine on-time rewinds, the maximum CUDA weight and momentum errors to the true origin were `6.96e-17` and `1.67e-17`. Across the three late alarms, the CPU maxima to the true origin were 0.1135 and 0.1244. All twelve passive AdamW warnings arrived at zero-based pre-update indices 16–19.

In B2b every dirty rewind alarm came after 32 commits, with 32 changed updates. All estimated origins matched the true start at step zero. The largest CUDA true-origin errors were `6.57e-16` for weights and `3.03e-16` for momentum. AdamW-skip and AdamW-continue saw four clean terminal alarms. The terminal canary uses each policy's own adapted weights.

| Dirty rewind accuracy, mean over 12 seeds | B2a | B2b |
| --- | ---: | ---: |
| Task 1 at true origin | 99.000% | 99.028% |
| Task 1 before rewind at alarm | 98.861% | 33.444% |
| Task 1 at common window end | 98.944% | 99.028% |
| Task 1 at equal gradient budget | 97.917% | 99.028% |

| Dirty policy | B2a forward / replay / clean repair | B2b forward / replay / clean repair |
| --- | --- | --- |
| DKD rewind | 24 / 8 / 64 on nine, 32 / 8 / 56 on three | 32 / 32 / 0 |
| DKD and AdamW skip | 24 / 0 / 72 on nine, 32 / 0 / 64 on three | 32 / 0 / 32 |
| DKD and AdamW continue | 48 / 0 / 48 | 32 / 0 / 32 |
| Always skip | 0 / 0 / 96 | 0 / 0 / 64 |

Clean rewind used 48 / 0 / 48 in B2a and 32 / 0 / 32 in B2b. Under f64, a weights-only checkpoint requires 1,432 bytes. Full checkpoints require 2,864 bytes for phase state and 4,304 for AdamW. B2a dirty rewind held 192 or 256 ID payload bytes, with 280 or 352 allocated host bytes. B2a clean rewind held 384 payload and 512 allocated bytes. The B2a event bank held 98,304 bytes. B2b rewind used 256 payload and 352 allocated ID bytes, plus a 65,536-byte immutable external event bank. The optimizer kept zero trajectory-history bytes.

## C2 | Retention and adaptation

The two backends agreed to reported precision. Scores below are CUDA means ± sample SD across the twelve seeds. BWT and arrival gain are percentage points. Retention matrices include every Task 1–5 evaluation at every checkpoint. [All per-method metrics](../artifacts/sol/p2/sol-20261122-cpu-lock/c2_summary.csv) · [Retention matrices](../artifacts/sol/p2/sol-20261122-cpu-lock/c2_matrices.csv) · [Retention and compute figure](../artifacts/sol/p2/sol-20261122-cpu-lock/retention_pareto.png)

| Method | Task 1 BWT, pp | Live, % | Arrival gain, pp |
| --- | ---: | ---: | ---: |
| Momentum SGD | −79.722 ± 2.518 | 96.910 ± 0.520 | 15.278 ± 1.430 |
| AdamW | −89.500 ± 1.642 | 98.562 ± 0.320 | 8.007 ± 0.780 |
| EWC | −58.750 ± 8.910 | 92.715 ± 1.954 | 18.792 ± 2.743 |
| Fisher mass | −94.389 ± 3.068 | 96.632 ± 1.080 | 17.701 ± 2.993 |
| GENERIC | −88.528 ± 7.191 | 94.792 ± 2.005 | 21.062 ± 3.998 |
| Unit mass glide | −96.750 ± 0.669 | 98.861 ± 0.228 | 2.340 ± 0.673 |
| Energy isotropic | −96.639 ± 0.611 | 98.847 ± 0.224 | 2.493 ± 0.730 |
| Energy low | −95.917 ± 1.120 | 98.847 ± 0.246 | 2.549 ± 0.810 |
| Energy high | −96.667 ± 0.682 | 98.840 ± 0.252 | 2.528 ± 0.752 |

The one-sided tests drop ties at six excess correct predictions and require `p ≤ 0.05/9`. The paired CSV gives each seed's correct-count BWT difference and live-accuracy difference for both backends. [All primary and descriptive paired contrasts](../artifacts/sol/p2/sol-20261122-cpu-lock/c2_paired.csv)

| Comparison | Mean paired BWT, pp | Mean paired live, pp | Wins / non-ties | p |
| --- | ---: | ---: | ---: | ---: |
| Fisher mass vs momentum SGD | −14.667 | −0.278 | 0/12 | 1 |
| Fisher mass vs AdamW | −4.889 | −1.931 | 0/11 | 1 |
| Fisher mass vs EWC | −35.639 | +3.917 | 0/12 | 1 |
| Fisher mass vs unit mass | +2.361 | −2.229 | 5/12 | 0.806152 |
| GENERIC vs momentum SGD | −8.806 | −2.118 | 1/12 | 0.999756 |
| GENERIC vs AdamW | +0.972 | −3.771 | 4/12 | 0.927002 |
| GENERIC vs EWC | −29.778 | +2.076 | 0/12 | 1 |
| GENERIC vs Fisher mass | +5.861 | −1.840 | 9/11 | 0.032715 |
| Energy high vs low | −0.750 | −0.007 | 0/12 | 1 |

Fisher mass passed the live gate and failed all four retention contrasts. GENERIC exceeded 90% live and fell 3.77 points below tuned AdamW, so it failed the live gate and all four retention contrasts. The energy triplet had positive common removal on all 2,304 steps per backend and passed the realized-removal tolerance each time. The nominal isotropic target was capped on 899 steps. High passed the live comparison and failed the retention comparison against low. The largest removal residual was `1.738e-16` J on CPU and `1.152e-16` J on CUDA. Across the 12 seeds, the nominal request summed to 1.507457 J and the common target and each realized removal summed to 0.721259 J. [Every request, capacity, target and realized removal](../artifacts/sol/p2/sol-20261122-cpu-lock/energy_ledger.csv)

| Energy arm | BWT vs momentum SGD, pp | BWT vs AdamW, pp | BWT vs EWC, pp |
| --- | ---: | ---: | ---: |
| Isotropic | −16.917 | −7.139 | −37.889 |
| Low | −16.194 | −6.417 | −37.167 |
| High | −16.944 | −7.167 | −37.917 |

## Native f32 horizon and integer control

The table reports the largest absolute L2 errors across three seeds per backend. The forward columns compare the integer trajectory, decoded to f32, with the native f32 forward endpoint. Every recovered integer position and momentum matched exactly. Integer rollback L2 was zero at every horizon on both backends. No frozen horizon row changed the recovered Task 1 correct count. Forward correct counts agreed across the two discretizations at every tested horizon. [Per-seed rows and forward accuracies](../artifacts/sol/p2/sol-20261122-cpu-lock/horizon_rows.csv) · [Horizon summaries](../artifacts/sol/p2/sol-20261122-cpu-lock/horizon_summary.csv) · [Rollback figure](../artifacts/sol/p2/sol-20261122-cpu-lock/rollback_horizon.png)

| T | Native CPU q / p | Native CUDA q / p | Integer vs native forward CPU q / p | Integer vs native forward CUDA q / p | Native seeds passing per backend |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1.36e-7 / 2.29e-8 | 1.97e-7 / 2.88e-8 | 8.75e-7 / 7.92e-8 | 8.65e-7 / 7.88e-8 | 3/3 |
| 32 | 2.21e-7 / 4.29e-8 | 2.67e-7 / 4.03e-8 | 1.28e-6 / 1.12e-7 | 1.31e-6 / 1.05e-7 | 3/3 |
| 64 | 4.69e-7 / 7.85e-8 | 4.97e-7 / 7.63e-8 | 2.17e-6 / 2.06e-7 | 2.16e-6 / 2.19e-7 | 3/3 |
| 128 | 8.54e-7 / 1.51e-7 | 1.10e-6 / 1.81e-7 | 4.73e-6 / 5.15e-7 | 4.59e-6 / 4.89e-7 | 3/3 |
| 256 | 1.14e-4 / 1.06e-5 | 1.54e-4 / 1.56e-5 | 4.18e-5 / 1.73e-5 | 4.42e-5 / 1.81e-5 | 0/3 |
| 500 | 1.79e-1 / 1.82e-2 | 5.53e-2 / 7.72e-3 | 7.09e-3 / 1.51e-3 | 6.86e-3 / 2.09e-3 | 0/3 |

The last passing point before the first failure on the native grid was 128 on both backends. The integer result is right-censored at 500, with 2,864 bytes of authoritative phase state versus 1,432 bytes for native f32. The optimized native DKD uses scaled `add` operations, which can round differently from Astra's separate multiplication and addition. The numeric errors are specific to the adapter, the tape, and the backend. These measurements establish a finite-precision tolerance miss on the tested paths. The integer control changes the forward discretization and relies on lossless encoding, checked bounds and deterministic force replay.

### Supplemental dense native scan

With the frozen lock and the same three tapes, I replayed every prefix from 1 through the first miss on any seed. All replayed grid points agreed exactly with the corresponding frozen native results. Each first miss crossed the weight tolerance with the held-out correct count unchanged. The scan stopped at the first miss per backend, so the other seeds' first misses were not observed.

| Backend | First failed T | Seed | Weight L2 | Momentum L2 | Forward / replay calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 210 | 20903 | 1.07751e-5 | 1.03412e-6 | 630 / 66,465 |
| CUDA | 203 | 20921 | 1.03449e-5 | 1.21181e-6 | 609 / 62,118 |

| Raw scan | SHA-256 |
| --- | --- |
| [CPU dense prefixes](../artifacts/sol/p2/sol-20261122-cpu-lock/dense_cpu.json) | `a6b3f364f5e8aa02701a6dadf3949d21d98f0edbb85e4b643db884bfbe1da5d2` |
| [CUDA dense prefixes](../artifacts/sol/p2/sol-20261122-cpu-lock/dense_cuda.json) | `618da1d26e8dac838d935b8c385a64b5b462746669b0d4152976b0aca164c9d8` |

For a completed locked run, use the same recorded `$lock` and output directory `$out`:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -B -m experiments.dense_horizon_phase2 --artifact-dir $out --lock-sha $lock --device cpu --output "$out/dense_cpu.json"
python -B -m experiments.dense_horizon_phase2 --artifact-dir $out --lock-sha $lock --device cuda --output "$out/dense_cuda.json"
```

## File manifest

- New adapter files under `src/mttt/` are `phase2.py`, `p2_dkd.py`, `p2_lattice.py`, `p2_dissipation.py` and `p2_baselines.py`.
- New experiment files under `experiments/` are `run_phase2.py`, `test_phase2.py`, `dense_horizon_phase2.py` and `phase2_report.py`.
- The output directory contains `physics_cpu.json`, `validation.json`, `test_cpu.json`, `test_cuda.json`, `dense_cpu.json`, `dense_cuda.json`, `b2_summary.csv`, `b2_paired.csv`, `c2_summary.csv`, `c2_matrices.csv`, `c2_paired.csv`, `energy_ledger.csv`, `timing.csv`, `horizon_rows.csv`, `horizon_summary.csv`, `validation_summary.csv`, `poison_recovery.png`, `rollback_horizon.png` and `retention_pareto.png`.
- The run logs contain the exact commands, exit statuses, stdout, stderr, pre/post pin checks and raw output hashes. [First unit attempt](../experiments/results/phase2-prepare-20261123T012620816786Z-47908) · [Successful preparation](../experiments/results/phase2-prepare-20261123T012711349182Z-19816) · [Held-out runs](../experiments/results/phase2-test-20261123T012756643729Z-36592) · [Report and supplemental logs](../experiments/results/phase2-supplement-20261123T012912235520Z-33936)

The B2a onset was aligned to the audit schedule. Arbitrary sub-block onset and longer post-budget recovery remain open. The energy cap rules out the nominal fixed-strength attribution on this schedule, and timing is specific to the recorded blocks.
