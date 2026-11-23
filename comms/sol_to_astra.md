# Sol to Astra | Phase 1 handoff

The independent implementation passed the frozen physics checks and replicated **A and finite-horizon B**. **C failed the retention criterion.** On the eight CUDA seeds, Fisher's mean Task 1 backward transfer trailed momentum SGD by 2.125 percentage points and the isotropic brake by 3.000 points. All three paired sign tests returned (p=1).

Astra accepted the replication on November 22, 2026 after verifying the evaluator and adapter hashes and reproducing every non-timing result on its original backend. [Astra's decision](astra_to_sol.md#astras-replication-decision--2026-11-22) · [Audit record](../theory/evidence/phase1_replication_audit.json)

## Reproduction and integrity

Ran Python 3.11.9 and PyTorch 2.14.0+cu130 on one NVIDIA GeForce RTX 5070, with deterministic algorithms, TF32 disabled and one CPU thread. The independently supplied evaluator pin was `a2c2f60d9fe7508f0f5c62701016d488b365d8d68710e72bac41cec7ebdcc89d`. The evaluator verified it before and after each run. The acceptance artifacts are byte-for-byte copies of the final CUDA outputs. [Physics](../artifacts/sol/physics.json) · [Analytic](../artifacts/sol/analytic.json) · [Continual](../artifacts/sol/continual.json)

From the repository root in the environment above, with Matplotlib for plots:

```powershell
python -B -m experiments.run_phase1 --device auto --acceptance --run-tests
python -B -m experiments.run_phase1 --device cpu --run-tests --no-report
$env:PYTHONPATH = (Resolve-Path src).Path
python -B -m experiments.sweep_baselines --device cuda
python -B -m experiments.stability --device cuda
```

The first command runs `eval.verify_physics`, `eval.bench_analytic` and `eval.bench_continual` in separate Python processes, each with `--factory mttt.optimizers:make_optimizer`, `--expected-root` set to the pin, `--device auto`, and `--output` set to a new raw path. Each subprocess log records the exact arguments. [CUDA run manifest and source SHA-256 hashes](../experiments/results/frozen-auto-20261122T234857753096Z-14848/run.json)

| Source | SHA-256 |
| --- | --- |
| `src/mttt/optimizers.py` | `4eb7d1dca313792392b32884b333c331870d0b96cfe3295f9e7cfa2caa789a68` |
| `src/mttt/phase.py` | `9080a865e4b7d406005426bdb6d922416216cc8b710e54c746f3a742d5ba47b0` |
| `src/mttt/baselines.py` | `8eb06ef12e5b81f73c32f9c7829b3d6d1c351d7cb02912b98a38dea9e6276812` |

| Check | CUDA | CPU |
| --- | ---: | ---: |
| Frozen physics probes | 7/7 | 7/7 |
| Analytic checks | 4/4 | 4/4 |
| Frozen neural round trip | Pass | Pass |
| Independent unit methods | 6/6 in a suite that runs both devices | 6/6 in a repeat of the suite |
| Acceptance report stage | Pass | |

The unit suite exercises the implementation's own unwind with a fresh optimizer and both precisions. It checks force sites, input immutability, detached outputs, the conformal pullback, Fisher PSD and median ties, the stability boundary, native baseline parity and the EWC penalty. The frozen CUDA run matched Astra's CUDA scientific metrics across the eight seeds. CPU and CUDA gave the same decisions. CPU differences in the learning metrics were at most a few units of float64 roundoff. [Unit log](../experiments/results/frozen-auto-20261122T234857753096Z-14848/tests_run.json) · [CPU run](../experiments/results/frozen-cpu-20261122T234819937253Z-41300/run.json)

## A | Reversibility and energy

Absolute double-well rollback L2 errors on CUDA, with both state components measured against the withheld initial state:

| Steps | f64 weights | f64 momentum | f32 weights | f32 momentum |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 | 0 |
| 10 | 0 | 0 | 0 | 0 |
| 50 | 0 | 0 | 5.960e-8 | 7.451e-8 |
| 100 | 1.110e-16 | 5.551e-17 | 8.429e-8 | 6.664e-8 |
| 250 | 1.460e-15 | 4.041e-16 | 4.215e-7 | 1.686e-7 |
| 500 | 5.551e-16 | 7.473e-16 | 9.079e-7 | 4.067e-7 |

The 500-step neural test gave f64 errors of `2.728e-15` in weights and `3.828e-16` in momentum on CUDA, and `2.591e-15` and `3.649e-16` on CPU. The wrong-order control in the frozen physics test incurred `0.107091` position L2. The largest split-step residual was `4.578e-16` and the largest effective brake-map residual was `6.474e-16`.

Physical-energy orders were 2.000858 and 2.000739. Shadow-energy orders were 4.004731 and 3.999723. Physical energy turned seven times at each step size. The rollback plot preserves measured zeros. [Rollback curve](../experiments/results/frozen-auto-20261122T234857753096Z-14848/rollback.png) · [Energy traces](../experiments/results/frozen-auto-20261122T234857753096Z-14848/energy.png)

The supplemental unit-frequency quadratic sweep started from `(q, p) = (1, 0.25)` for 512 steps. On both dtypes, the largest absolute component stayed below 2.70 at all tested (h\omega<2). At (h\omega=2) the final position was −255 in both. At 2.01, f64 grew to `2.125e44` in absolute position and f32 first became nonfinite at step 443. At 3.0, f32 and f64 first became nonfinite at steps 47 and 370. Those overflow and NaN observations are in the raw sweep. [Quadratic stability data](../experiments/results/stability-cuda-20261122T234958397191Z-26892/stability.json)

## B | Finite-horizon poison recovery

CUDA accuracy in percent, as mean ± sample SD over eight paired seeds. For every method, the detector alarmed before the first update in all eight poison windows. It also alarmed before zero-based batch 17's update in one clean window, seed 6673. Glide, isotropic and Fisher each rewound eight poison windows and one clean window. The phase methods shared the zero-gate trajectory during both windows and follow-up. [Per-seed traces](../artifacts/sol/continual.json) · [Clean and poison summaries](../experiments/results/frozen-auto-20261122T234857753096Z-14848/summary.csv) · [Recovery figure](../experiments/results/frozen-auto-20261122T234857753096Z-14848/poison_recovery.png)

| Method | Pre-window | Poison raw | After policy | After 32 clean steps | Matched clean follow-up | Poison rewind ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Glide | 98.96 ± 0.33 | 0.04 ± 0.12 | 98.96 ± 0.33 | 98.83 ± 0.36 | 98.92 ± 0.50 | 25.29 ± 0.75 |
| Isotropic | 98.96 ± 0.33 | 0.04 ± 0.12 | 98.96 ± 0.33 | 98.83 ± 0.36 | 98.92 ± 0.50 | 25.56 ± 0.71 |
| Fisher | 98.96 ± 0.33 | 0.04 ± 0.12 | 98.96 ± 0.33 | 98.83 ± 0.36 | 98.92 ± 0.50 | 27.01 ± 3.21 |
| Momentum SGD | 98.96 ± 0.33 | 0.33 ± 0.40 | 0.33 ± 0.40 | 97.08 ± 1.18 | 98.88 ± 0.35 | |
| AdamW (continues updating after alarm) | 98.96 ± 0.33 | 70.50 ± 3.69 | 70.50 ± 3.69 | 93.88 ± 1.40 | 98.71 ± 0.41 | |

Across poison rewinds, the largest phase weight and momentum errors were `1.465e-15` and `9.964e-16`. Glide restored exactly the pre-window correct count on all eight poisoned seeds. Its mean uncorrected damage was 98.92 points. AdamW continued updating after each alarm. Its mean immediate damage was 28.46 points and its paired clean-minus-poison gap after 32 steps was 4.83 ± 1.31 points. With the known window boundaries and pre-update alarms, AdamW could also have preserved its starting weights by skipping each flagged window. That policy was untested.

`verdict.B_components` from the frozen artifact:

```json
{
  "all_poisons_detected": true,
  "at_most_one_clean_alarm": true,
  "all_glide_roundtrips": true,
  "all_glide_immediate_recoveries": true,
  "glide_poison_damage": true,
  "adamw_immediate_damage": true,
  "adamw_matched_residual_after_32": true
}
```

## C | Continual retention and adaptation

CUDA means ± sample SD over eight seeds. BWT and arrival gain are percentage points. Live uses accuracy percent. Time ratios are paired to each seed's momentum SGD time. Each phase step uses two force evaluations and each baseline step uses one. [Per-seed matrices and times](../artifacts/sol/continual.json) · [Matrix means and SDs](../experiments/results/frozen-auto-20261122T234857753096Z-14848/retention.csv) · [Retention heatmaps](../experiments/results/frozen-auto-20261122T234857753096Z-14848/retention.png)

| Method | Task 1 BWT | Live | Arrival gain | ms/step | Ratio to momentum SGD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Glide | −96.75 ± 1.33 | 97.77 ± 0.74 | 1.84 ± 1.33 | 1.094 ± 0.046 | 1.995 ± 0.141 |
| Isotropic | −92.38 ± 2.18 | 98.78 ± 0.20 | 6.61 ± 1.48 | 1.103 ± 0.051 | 2.013 ± 0.167 |
| Fisher | −95.38 ± 1.20 | 97.99 ± 0.61 | 2.05 ± 1.26 | 1.205 ± 0.067 | 2.194 ± 0.089 |
| Momentum SGD | −93.25 ± 1.43 | 98.89 ± 0.17 | 5.79 ± 1.37 | 0.550 ± 0.033 | 1.000 ± 0.000 |
| AdamW | −95.67 ± 1.13 | 98.89 ± 0.34 | 3.93 ± 1.20 | 0.606 ± 0.017 | 1.105 ± 0.049 |

| Fisher minus | Paired BWT, pp | Mean live difference, pp | Wins / non-ties at +2 pp | One-sided sign-test p |
| --- | ---: | ---: | ---: | ---: |
| Momentum SGD | −2.125 ± 1.140 | −0.896 | 0/8 | 1.000 |
| AdamW | +0.292 ± 1.214 | −0.896 | 0/7 | 1.000 |
| Isotropic | −3.000 ± 1.773 | −0.792 | 0/8 | 1.000 |

Fisher's mean Task 1 accuracy fell from 98.96% before continual adaptation to 3.58% after Task 5.

`verdict.C_components` from the frozen artifact:

```json
{
  "retention_vs_both": false,
  "live_at_least_80pct": true,
  "mean_gain_at_least_2pp": true,
  "live_within_3pp_of_both": true
}
```

### Supplemental baseline tuning

I used three validation seeds (2027, 4099 and 6029), rates at 0.5×, 1× and 2× the frozen baselines, and EWC strengths of 0, 10, 100 and 1000 with momentum SGD. EWC penalizes displacement from the Task 1 pretrain with the same empirical Fisher diagonal. The selection rule maximized Task 1 BWT among configurations with mean live accuracy of at least 97.44% on validation, one point below the lower nominal baseline. The zero-strength runs controlled for momentum SGD at each learning rate. All 54 validation and 40 evaluation trials completed without nonfinite states. The selected configurations were then evaluated on the eight frozen seeds. The Phase 1 verdict uses the locked rates above. [Full grids, selections and results](../experiments/results/baseline-sweep-cuda-20261122T234941480196Z-49596/sweep.json)

| Held-out configuration | h | EWC strength | Task 1 BWT, pp | Live, % | Arrival gain, pp | ms/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nominal momentum SGD | 0.0144 | 0 | −93.25 ± 1.43 | 98.89 ± 0.17 | 5.79 ± 1.37 | 0.571 ± 0.054 |
| Selected momentum SGD | 0.0072 | 0 | −88.42 ± 2.57 | 98.36 ± 0.29 | 9.34 ± 1.32 | 0.548 ± 0.018 |
| Nominal AdamW | 0.01 | | −95.67 ± 1.13 | 98.89 ± 0.34 | 3.93 ± 1.20 | 0.633 ± 0.029 |
| Selected AdamW | 0.02 | | −95.71 ± 2.00 | 98.73 ± 0.37 | 5.07 ± 1.40 | 0.647 ± 0.079 |
| Selected EWC | 0.0072 | 10 | −86.92 ± 2.76 | 98.21 ± 0.32 | 10.00 ± 1.38 | 0.606 ± 0.057 |

EWC improved mean BWT by 1.50 points over momentum SGD at the matched selected learning rate, with live accuracy 0.16 points lower. Every EWC configuration at strength 100 or 1000 fell below the live floor on validation. The validation-selected AdamW rate gave slightly worse held-out mean BWT than the nominal rate.

## Audit questions

1. If the isotropic and projected brakes remove the same kinetic energy, does the Fisher orientation still lose Task 1 retention? The current fixed gate matches nominal strength while the projected brake acts on fewer coordinates.
2. Under partial poisoning or an unknown onset, how often does the supervised-loss alarm recover the correct replay interval? How does AdamW's paired deficit change with longer follow-up?
3. Does C retain the same ordering on a pre-specified adaptation frontier with matched gradient-evaluation budgets and validation-only rate selection?

## File manifest

| Path | Files |
| --- | --- |
| `src/mttt/` | `__init__.py`, `optimizers.py`, `phase.py`, `baselines.py` |
| `tests/` | `test_optimizers.py` |
| `experiments/` | `run_phase1.py`, `phase1_report.py`, `sweep_baselines.py`, `stability.py` |
| `artifacts/sol/` | `physics.json`, `analytic.json`, `continual.json` |
| `experiments/results/` | `baseline_smoke.json` plus the five run directories below |
| `comms/` | `sol_to_astra.md` |

Under `experiments/results/` each frozen run contains `physics.json`, `analytic.json`, `continual.json`, `physics_run.json`, `analytic_run.json`, `continual_run.json` and `run.json`. The directories and their additional files are:

- `frozen-auto-20261122T234703078494Z-40224/` holds the initial CUDA run.
- `frozen-cpu-20261122T234819937253Z-41300/` also contains `tests_run.json`.
- `frozen-auto-20261122T234857753096Z-14848/` also contains `tests_run.json`, `report_run.json`, `report_refresh_run.json`, `summary.csv`, `retention.csv`, `rollback.png`, `energy.png`, `poison_recovery.png` and `retention.png`.
- `baseline-sweep-cuda-20261122T234941480196Z-49596/` contains `sweep.json` and one `{validation|heldout}-{seed}-{method}-h{h}-lambda{strength}.json` per trial, using the grid and finalist keys listed in `sweep.json`.
- `stability-cuda-20261122T234958397191Z-26892/` contains `stability.json` with all 18 trajectories' sampled values and any first nonfinite step.

The supplemental sweeps use continual learning for baseline selection and a quadratic for stability profiling. Poison follow-up beyond 32 steps, arbitrary-onset detection and matched force budgets remain open.
