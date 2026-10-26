# Full-amplitude representation outcome, October22

Neither predeclared architecture met the warm-start fidelity rule. Both runs
completed normally and are discarded as this initialization scheme. No new MRI
training or scoring was performed, and no MRI improvement is claimed. The
campaign remains at28 formal MRI training experiments,65 historical rescorings
and one separate discarded proximal output correction.

## Actual measurements

| Field | Plain raw relative L2 | Enhanced raw relative L2 | Decision |
| --- | ---: | ---: | --- |
| MLP, initial | 0.5317671111822541 | 0.5345927523644031 | initialization |
| MLP, after coefficient fitting | 0.36932492522163146 | 0.39568174217207236 | intermediate endpoint |
| MLP, final | 0.3247297186805714 | 0.3593065686077883 | discard |
| Grid, initial | 0.5317671111822541 | 0.5345927523644031 | initialization |
| Grid, after coefficient fitting | 0.3237226365877122 | 0.34593995210809386 | intermediate endpoint |
| Grid, final | 0.273462019725601 | 0.2974255657616767 | discard |

Admission required both final errors<=0.01, declared before either run. The
threshold was not changed. These are complex raw-render errors against a fresh
G2 teacher, not PSNR/SSIM/NMSE or errors against a scoring reference. The grid
fits better at this budget, but neither is a sufficiently faithful G2 warm start.
This does not establish a general neural-function capacity limit or rule out a
different initialization design.

| Completed stage | Amplitude parameters | Coefficient/raw updates | Gaussian forwards/VJPs | Numerical seconds | Supervisor seconds | Peak allocated GiB |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| Genuine G2 raw control | 0 | 0/0 | 2/0 | 5.5955201322212815 | 37.80201029777527 | 3.1972551345825195 |
| MLP | 7810 | 2000/128 | 264/256 | 120.69672320177779 | 136.1872751712799 | 3.4934444427490234 |
| Grid | 604290 | 2000/128 | 264/256 | 141.32650907803327 | 156.77273774147034 | 3.5049543380737305 |

Both fits kept all3.6M source Gaussians, shared geometry, source unit phases and
7.2M zero frozen phase parameters unchanged. They used only shared positive
amplitude parameters, two global scales and geometry features; no original
per-point modulus residual enters the forward function. All gradients were
finite. Actual counters, final checkpoints and complete traces passed the
separate parent validator; no intermediate checkpoint was selected.

## Control, code and evidence

Scientific source revision131bb0d passed304 CPU tests in33.10s and a fresh
independent whole-branch review with0 actionable findings. The real review gate
binds31 code files,29 G2/source/raw input files and the exact selected recipes.
The original Python renderer and existing CUDA binary were preserved. Both
genuine raw reporting-gauge arrays equal their frozen G2 anchors elementwise,
relative complex L2 zero. All stages used an idle GPU2 with observed child ACK,
float32, autocast disabled, CUDA matmul TF32 false and cuDNN TF32 true.

| Gate | Completed SHA256 |
| --- | --- |
| outputs/g2_fullamp_raw_control_v1/completed.json | 303b96d63383b54534dc79861aa7c41f7e834f02ece13199264aa65763fff48a |
| outputs/g2_fullamp_mlp_rep_v1/completed.json | 17471d4f8d10612b285beccd084b3e1e22ed8f29b375d81eaa8612b87be8e1cd |
| outputs/g2_fullamp_grid_rep_v1/completed.json | 49217c504a17ce64ea2fd0882f67b63e106c6878fe8ec163dc91f3c8deb42006 |

Final field checkpoints are retained in each fit directory. MLP SHA256:
1d5798e06c65f0d9f020964976c588d2ca5f367a8a83deafb96e9e080afbd07a.
Grid SHA256:99843b569cf734ca0d06d0b95ee4e498a54ba3e21bb81cf2b5accca71db71469.
Every completed gate was revalidated against current actual source/input and
artifact bytes before recording its decision in results.tsv/autoresearch.jsonl.
Setup plans and old inherited campaign fields are historical snapshots; final
completion records and representation_state.json determine current status.

The whole-head keeper remains lowreg_f075_solve_v1, score
-0.11118128532897047; all historical paired-plane wins remain0/59. The next
already researched direction is a separate fixed-count explicit Gaussian
relocation experiment. It will retain the fixed MRI benchmark and requires its
own implementation/review/controls; it is not an admitted full-field MRI trial.
