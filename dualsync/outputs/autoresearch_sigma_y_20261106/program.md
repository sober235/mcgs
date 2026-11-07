# AutoResearch program: single-run sigma_y SNR

Started 2026-11-06 under the `karpathy-autoresearch` skill. User decisions (2026-11-06): single reconstruction only, first hypothesis "raise sigma_y", GPU budget 24 runs.

## Setup

- **Research question:** Raise SNR of the registered final Tuned result (Accepted Tuned, base seed 1234; MSP06 and DYF01) while keeping its structure.
- **Metric (single number):** frozen SNR score = exp(mean over cases of .5 log bg-SNR gain + .5 log in-band-SNR gain), relative to the accepted Tuned array. Baseline = 1.0. Useful ≥ 1.05.
- **Hard structure gates (every case, unchanged from prior rounds):** coarse SSIM to Tuned ≥ .99; coarse SSIM to input ≥ Tuned's − .001; strong-edge amplitude ≥ .97; symmetric edge P95 ≤ 1 px; foreground mean ratio in [.98, 1.02]; each SNR gain ≥ .9999. A score from an infeasible run never counts as an improvement.
- **Variable:** `sigma_y` only (accepted 0.3; DC measurement-noise level in `closed_form_prox`). Larger sigma_y gives acquired k-space less weight relative to the diffusion estimate. All other parameters, sampler source, checkpoint, inputs, seeds, evaluator: SHA-pinned in `protocol.json`.
- **Starting hypothesis (iteration 1):** sigma_y 0.3 → 0.5 reduces input-correlated noise re-injected by DC while the model prior keeps structure.

## Loop

0. Null: sigma_y 0.3 through `run.py` on DYF01 must reproduce the accepted array (rel L2 ≤ 1e-6). Proves the harness is the accepted pipeline.
1. Each iteration: one sigma_y value on the 4 tuning cases (DYF01, DYF06, MSP01, MSP06) → `evaluate.py` → log to `experiments.jsonl` → commit on branch `exp/sigma-y-<value>` → propose next value.
2. Freeze the best feasible useful value, then run it once on confirmation cases (DYF02, DYF10, MSP02, MSP10). Confirmation never selects sigma_y.
3. Stop when: budget (24 runs) would be exceeded, three consecutive iterations degrade, or a value is confirmed.

## Safety rails

- Never delete data or earlier experiments; never overwrite outputs; old output dirs are read-only.
- Only files in this directory are written. No default config, registered result or main branch change.
- Every run is reserved in `ledger.jsonl` before inference; failures still count; no automatic retries.
- One idle GPU only; check utilization first and never preempt other jobs.
- Adoption of any result requires explicit user approval.

## Commands (server, repo root `/data0/zhuoxu/code/DualSync`)

```bash
export PYTHONNOUSERSITE=1  # ~/.local has a broken torch that shadows the env
PY=/home/congcongliu/anaconda3/envs/pt1.10/bin/python
D=outputs/autoresearch_sigma_y_20261106
$PY $D/run.py --tag null --sigma-y 0.3 --cases null --gpu <idle>
$PY $D/evaluate.py --tag null --cases null --null
$PY $D/run.py --tag sy0.5 --sigma-y 0.5 --cases tune --gpu <idle>
$PY $D/evaluate.py --tag sy0.5 --cases tune
```
