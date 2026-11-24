# Formal matrix rows (Task 11), state on 2026-11-24 05:45 PST (local clock)

Generated from `../experiments.jsonl` (untagged rows, seed 42, AF12). PSNR here is the TRAINER GAUGE
(head PSNR printed at the end of training), not the formal score of Task 12 (`mevaluate.py`, "evaluation"
PSNR in the study README). T1 = plain, T2 = contrast; single-contrast arms (I-T2, I-T2+w) have no T1.

| # | queue | row (order:cond:arm) | state | T1 head PSNR | T2 head PSNR | card |
|---|---|---|---|---|---|---|
| 1 | feasibility / zero-motion gate | il:C0:B0 | done | 35.834 | 36.306 | 3 |
| 2 | feasibility / zero-motion gate | il:C0:B1 | done | 35.822 | 36.323 | 6 |
| 3 | feasibility / zero-motion gate | il:C0:B2 | done | 35.833 | 36.305 | 2 |
| 4 | feasibility / zero-motion gate | il:C0:B12 | done | 35.821 | 36.322 | 1 |
| 5 | feasibility / zero-motion gate | il:C0:B123 | done | 35.821 | 36.322 | 2 |
| 6 | feasibility / zero-motion gate | il:C0:I-T2+w | done | 36.326 | — | 2 |
| 7 | feasibility / zero-motion gate | kz:C0:B2 | done | 35.834 | 36.306 | 2 |
| 8 | feasibility / zero-motion gate | kz:C0:B123 | done | 35.821 | 36.321 | 1 |
| 9 | upper | il:Ta:B0 | done | 35.823 | 34.408 | 2 |
| 10 | upper | il:Ta:Ow | done | 35.834 | 35.864 | 2 |
| 11 | upper | il:Tb:B0 | done | 35.824 | 34.455 | 2 |
| 12 | upper | il:Tb:Ow | done | 35.834 | 35.624 | 2 |
| 13 | upper | il:Tc:B0 | done | 35.783 | 27.714 | 2 |
| 14 | upper | il:Tc:Ow | done | 35.840 | 32.240 | 2 |
| 15 | upper | il:Tc:Op | done | 35.804 | 33.622 | 2 |
| 16 | upper | il:E2Tb:B0 | done | 35.777 | 35.031 | 2 |
| 17 | upper | il:E2Tb:Ow | done | 35.785 | 36.251 | 2 |
| 18 | upper | kz:Ta:B0 | done | 35.819 | 32.245 | 2 |
| 19 | upper | kz:Tc:B0 | done | 35.816 | 29.533 | 3 |
| 20 | upper | kz:Tc:Ow | done | 35.848 | 30.541 | 7 |
| 21 | upper | kz:Tc:Op | done | 35.818 | 34.679 | 7 |
| 22 | rest | il:Ta:B123 | done | 35.827 | 35.875 | 2 |
| 23 | rest | il:Tb:B123 | done | 35.819 | 35.684 | 2 |
| 24 | rest | il:Tc:B123 | done | 35.799 | 33.611 | 3 |
| 25 | rest | il:Ta:B2 | done | 35.833 | 35.859 | 2 |
| 26 | rest | il:Tb:B2 | done | 35.836 | 35.621 | 3 |
| 27 | rest | il:Tc:B2 | done | 35.783 | 27.715 | 2 |
| 28 | rest | il:E2:B0 | done | 35.782 | 36.950 | 3 |
| 29 | rest | il:E2:B1 | done | 35.817 | 36.987 | 2 |
| 30 | rest | il:E2:B123 | done | 35.817 | 36.985 | 2 |
| 31 | rest | il:E2Tb:B123 | done | 35.812 | 36.349 | 2 |
| 32 | rest | il:E2Tb:B12 | done | 35.817 | 36.290 | 2 |
| 33 | rest | il:TbL:B0 | done | 35.818 | 34.451 | 2 |
| 34 | rest | il:TbL:B123 | done | 35.818 | 35.682 | 2 |
| 35 | rest | il:C0:I-T2 | running | | | 2 |
| 36 | rest | il:Ta:I-T2 | queued | | |  |
| 37 | rest | il:Tb:I-T2 | queued | | |  |
| 38 | rest | il:Tc:I-T2 | queued | | |  |
| 39 | rest | il:E2:I-T2 | queued | | |  |
| 40 | rest | il:TbL:I-T2 | queued | | |  |
| 41 | rest | il:Tb:I-T2+w | queued | | |  |
| 42 | rest | kz:Ta:B123 | queued | | |  |
| 43 | rest | kz:Tc:B123 | queued | | |  |

Done 34 of 43, running 1, queued 8.
