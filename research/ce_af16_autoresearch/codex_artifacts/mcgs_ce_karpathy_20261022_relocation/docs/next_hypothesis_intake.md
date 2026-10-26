# Next-hypothesis intake while relocation is running

Status: source/history investigation only. The event500 outcome is pending;
this document neither selects a new MRI recipe nor authorizes changing the
running experiment. The existing best observed whole-head result remains
lowreg_f075_solve_v1. Native G2 continuation500 did not improve that keeper.

One remaining candidate is a denoiser learned solely from this case, then used
as a prior while the final image remains the original shared-geometry Gaussian
reconstruction. This would test a different prior from fixed BM4D or local
patch low-rank. Historical execution versus proposal is being independently
checked before selecting it.

Primary-source check on 2026-10-22: ReSiDe-S alternates reconstruction with a
network trained on current intermediate patches plus synthetic complex noise;
its training targets are the intermediate patches. The paper studies brain
MRI at R4, so it supplies no measured R16 benefit for this case. Its discrepancy
update assumes an appropriate measurement-noise model. We must not treat an
undersampled, correlated image residual as white noise or promise a clinical
performance gain. Source: [paper, sections2.2 and2.4](https://arxiv.org/html/2304.12890v4).

The [official single-scan driver](https://github.com/OSU-MR/reside/blob/dc487b0c2600de40110ee193f1e21338febe94ae/ReSiDe-S.py)
initializes a new CNN inside each outer iteration and uses patch self-training.
Its reference-based NMSE reporting and dataset loader cannot enter this
campaign's training branch. The [official network definition](https://github.com/OSU-MR/reside/blob/dc487b0c2600de40110ee193f1e21338febe94ae/models.py)
is a residual CNN with spectral normalization. We inspected source only, without
executing it, installing packages or loading external data/weights. Official
repository HEAD was dc487b0c2600de40110ee193f1e21338febe94ae when checked.

Any adaptation must preserve the frozen mask/CSM, acquired measurements, raw
scales, rendering binary and evaluator. It must declare the network's actual
training budget, noise model, independent complex outputs, Gaussian fitting
objective and all additional rendering/FFT work. A current-case learned prior
is not an external pretrained prior, an independent validation set or evidence
that joint modeling outperforms an equally improved GS-single. A successful
MRI result would still need repeat and matched-control attribution.

A choice between this candidate and another relocation/optimization follow-up
will use the completed event500 evidence, existing failure history and a
bounded falsification test. No reference/ROI-fitted parameter can be transferred
into reconstruction training.
