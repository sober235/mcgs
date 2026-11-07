# AutoResearch summary: single-run SNR on the registered final Tuned

Status 2026-11-06: **paused by the stop rule** (three consecutive infeasible iterations). GPU runs used: 13 of 24, each 74 s at 10.3 GB peak on GPU 4. Nothing was adopted. The registered final image, defaults and old outputs are unchanged.

## Experiment log (4 tuning slices: DYF01, DYF06, MSP01, MSP06; base seed 1234)

| id | change | score | feasible | bg-SNR gain | in-band gain | min SSIM to final | max input-SSIM drop (limit .001) |
|---|---|---:|---|---|---|---:|---:|
| 0 | null (sigma_y 0.3, DYF01) | 1.000 | yes | bit-exact | bit-exact | 1 | 0 |
| 1 | sigma_y 0.5 | 1.319 | no | 1.21–1.33 | 1.26–1.44 | .977 | .059 |
| 2 | sigma_y 0.35 | 1.093 | no | 1.06–1.10 | 1.08–1.12 | .998 | .013 |
| 3 | eta 0.5 | 1.008 | no | 0.99–1.01 | 1.00–1.04 | .995 | .004 |

All four iteration-2 slices pass every gate except the input-SSIM drop: SSIM to final ≥ .998, strong edges ≥ .995, edge P95 = 0 px, foreground mean .999–1.000.

## Control: the accepted method at other seeds (CPU, existing pool, no inference)

Eight existing reseeds of the accepted Tuned (`seed_gate_check.json`) keep input-SSIM drop ≤ .0008 but fail SSIM-to-final (.974–.987) and show no SNR gain (scores .95–1.04). So:

- The input-SSIM failures above are systematic drift away from the input's coarse content, not seed noise.
- sigma_y 0.35 is closer to the final image (SSIM .998) than the method's own reseeds are (≤ .987).

## What we learned

1. sigma_y trades SNR against coarse agreement with the noisy input roughly in proportion. A value that passes the .001 input-SSIM limit extrapolates to about 0.303, worth about +0.7% SNR.
2. Damping transition noise (eta 0.5) barely changes SNR. The grain in the final is not mainly from the late stochastic steps.
3. Under the frozen gates, the only gate-passing SNR gain in any round is still seed averaging (K2/alpha .5), which this round excluded (single run only).

## Options (user decision)

- Loosen only the input-SSIM drop limit (for example to .015) and confirm sigma_y 0.35 on DYF02, DYF10, MSP02 and MSP10 (4 runs, 11 left).
- Keep the gates and stop here: the final stays as is, and this round is recorded as negative evidence.
- Allow seed averaging again (K2/alpha .5 already passed all gates on all 38 slices).

Confidence: moderate. 4 tuning slices from the same two scans; proxy SNR only; no clean reference.
