# Widen-loop ledger snapshot, 2026-09-28

Output of `env PYTHONNOUSERSITE=1 PYTHONPATH= /home/congcongliu/anaconda3/envs/GSTMI/bin/python scripts/wd_ledger.py`, verbatim.
Every wd_ run of this branch plus the coupling-attribution single arms it reads (`ca_s_*_edge0`, `ca_s4_*`, worktree mcgs_hparam_wt).
Margin = MC-GS minus the best GS-single run at that AF (lambda_edge 0, no coupling knob; user choice A). Head-only metrics.
Probe runs (oracle partners / guides, asymmetric probe masks) are listed apart, without margins.
The score jsons live in the gitignored results_arc/ of worktree mcgs_widen_wt (server only).

| exp | AF | arm | knobs | T1 head PSNR / SSIM | FLAIR head PSNR / SSIM | margin T1 | margin FLAIR |
|---|---|---|---|---|---|---|---|
| wd_af12_p14_j | 12 | joint | le=0.001 --lambda_ptv 0.01 gaussian.rotation_lr=0.008 | 37.015 / 0.9516 | 36.359 / 0.9415 | +0.073 | +0.017 |
| wd_v_af12_j | 12 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 | 37.040 / 0.9530 | 36.364 / 0.9416 | +0.098 | +0.021 |
| wd_af12_p14_s_fl_e0b | 12 | FLAIR | le=0 --lambda_ptv 0.01 gaussian.rotation_lr=0.008 |  | 36.343 / 0.9414 |  |  |
| wd_af12_p14_s_t1_e0 | 12 | T1 | le=0 --lambda_ptv 0.01 gaussian.rotation_lr=0.008 | 36.906 / 0.9504 |  |  |  |
| wd_v_af12_s_fl | 12 | FLAIR | le=0 --lambda_ptv 0.01 |  | 36.318 / 0.9411 |  |  |
| wd_v_af12_s_t1 | 12 | T1 | le=0 --lambda_ptv 0.01 lambda_wav_plain=0.18 | 36.942 / 0.9516 |  |  |  |
| wd_af16_j_dg | 16 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 | 35.732 / 0.9401 | 34.979 / 0.9270 | -0.012 | +0.034 |
| wd_af16_j_dg_e002 | 16 | joint | le=0.002 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 | 35.736 / 0.9397 | 34.977 / 0.9266 | -0.008 | +0.032 |
| wd_af16_j_dg_p001 | 16 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 | 35.814 / 0.9399 | 35.002 / 0.9260 | +0.070 | +0.056 |
| wd_af16_j_dgpw | 16 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 | 35.841 / 0.9412 | 35.007 / 0.9261 | +0.097 | +0.062 |
| wd_af16_j_dgpw_dtv003 | 16 | joint | le=0.001 --lambda_dtv 0.03 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 | 35.843 / 0.9412 | 35.000 / 0.9253 | +0.100 | +0.054 |
| wd_af16_j_dgpw_e002 | 16 | joint | le=0.002 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 | 35.832 / 0.9408 | 35.002 / 0.9256 | +0.089 | +0.057 |
| wd_af16_j_dgpw_gd010 | 16 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.1 --lambda_ptv 0.01 lambda_wav_plain=0.18 | 35.840 / 0.9411 | 35.005 / 0.9260 | +0.096 | +0.060 |
| wd_af16_j_dgpw_pg025 | 16 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 | 35.850 / 0.9412 | 35.009 / 0.9260 | +0.106 | +0.064 |
| wd_af16_j_dgpw_pg025_gd050 | 16 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 edge_tv_guide_alpha=0.5 | 35.846 / 0.9413 | 35.007 / 0.9261 | +0.102 | +0.062 |
| wd_af16_j_dgpw_pg100 | 16 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=1.0 | 35.854 / 0.9412 | 35.006 / 0.9258 | +0.110 | +0.061 |
| wd_af16_j_dgpw_pg100_p002 | 16 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=1.0 --lambda_ptv 0.02 | 35.832 / 0.9413 | 35.012 / 0.9264 | +0.089 | +0.067 |
| wd_af16_j_dtv001 | 16 | joint | le=0.001 --lambda_dtv 0.01 | 35.707 / 0.9396 | 34.965 / 0.9265 | -0.036 | +0.020 |
| wd_af16_j_dtv003 | 16 | joint | le=0.001 --lambda_dtv 0.03 | 35.707 / 0.9396 | 34.963 / 0.9258 | -0.037 | +0.018 |
| wd_af16_j_e002_r | 16 | joint | le=0.002 | 35.715 / 0.9394 | 34.936 / 0.9259 | -0.029 | -0.010 |
| wd_af16_j_gd010 | 16 | joint | le=0.001 edge_tv_guide_alpha=0.1 | 35.733 / 0.9400 | 34.940 / 0.9264 | -0.011 | -0.005 |
| wd_af16_j_gd025 | 16 | joint | le=0.001 edge_tv_guide_alpha=0.25 | 35.730 / 0.9401 | 34.938 / 0.9264 | -0.014 | -0.008 |
| wd_af16_j_gd050 | 16 | joint | le=0.001 edge_tv_guide_alpha=0.5 | 35.718 / 0.9399 | 34.929 / 0.9263 | -0.026 | -0.016 |
| wd_af16_j_gd100 | 16 | joint | le=0.001 edge_tv_guide_alpha=1.0 | 35.695 / 0.9395 | 34.911 / 0.9260 | -0.049 | -0.034 |
| wd_af16_j_p0003 | 16 | joint | le=0.001 --lambda_ptv 0.003 | 35.792 / 0.9388 | 34.948 / 0.9244 | +0.049 | +0.003 |
| wd_af16_j_p001 | 16 | joint | le=0.001 --lambda_ptv 0.01 | 35.785 / 0.9393 | 34.959 / 0.9253 | +0.041 | +0.014 |
| wd_af16_j_p001w18 | 16 | joint | le=0.001 --lambda_ptv 0.01 lambda_wav_plain=0.18 | 35.817 / 0.9407 | 34.961 / 0.9253 | +0.074 | +0.016 |
| wd_v_af16_j | 16 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 | 35.850 / 0.9412 | 35.009 / 0.9260 | +0.106 | +0.064 |
| ca_s4_fl_geo007 | 16 | FLAIR | le=0 --lambda_geo 0.07 |  | 34.847 / 0.9245 |  |  |
| ca_s4_fl_geo014 | 16 | FLAIR | le=0 --lambda_geo 0.14 |  | 34.909 / 0.9251 |  |  |
| ca_s4_fl_kq075 | 16 | FLAIR | le=0 --edge_tv_kappa_q 0.75 |  | 34.836 / 0.9240 |  |  |
| ca_s4_fl_kq092 | 16 | FLAIR | le=0 --edge_tv_kappa_q 0.92 |  | 34.918 / 0.9251 |  |  |
| ca_s4_fl_ptv001 | 16 | FLAIR | le=0 --lambda_ptv 0.01 |  | 34.945 / 0.9249 |  |  |
| ca_s4_fl_ptv006 | 16 | FLAIR | le=0 --lambda_ptv 0.06 |  | 34.793 / 0.9249 |  |  |
| ca_s4_fl_wav0072 | 16 | FLAIR | le=0 lambda_wav_plain=0.072 |  | 34.864 / 0.9242 |  |  |
| ca_s4_fl_wav010 | 16 | FLAIR | le=0 lambda_wav_plain=0.10 |  | 34.904 / 0.9253 |  |  |
| ca_s4_fl_wav018 | 16 | FLAIR | le=0 lambda_wav_plain=0.18 |  | 34.825 / 0.9248 |  |  |
| ca_s4_fl_wf005 | 16 | FLAIR | le=0 --edge_tv_wfloor 0.05 |  | 34.925 / 0.9259 |  |  |
| ca_s4_fl_wf020 | 16 | FLAIR | le=0 --edge_tv_wfloor 0.20 |  | 34.847 / 0.9243 |  |  |
| ca_s4_t1_geo007 | 16 | T1 | le=0 --lambda_geo 0.07 | 35.510 / 0.9367 |  |  |  |
| ca_s4_t1_geo014 | 16 | T1 | le=0 --lambda_geo 0.14 | 35.613 / 0.9381 |  |  |  |
| ca_s4_t1_kq075 | 16 | T1 | le=0 --edge_tv_kappa_q 0.75 | 35.546 / 0.9369 |  |  |  |
| ca_s4_t1_kq092 | 16 | T1 | le=0 --edge_tv_kappa_q 0.92 | 35.608 / 0.9382 |  |  |  |
| ca_s4_t1_ptv001 | 16 | T1 | le=0 --lambda_ptv 0.01 | 35.681 / 0.9378 |  |  |  |
| ca_s4_t1_ptv006 | 16 | T1 | le=0 --lambda_ptv 0.06 | 35.407 / 0.9370 |  |  |  |
| ca_s4_t1_wav0072 | 16 | T1 | le=0 lambda_wav_plain=0.072 | 35.380 / 0.9348 |  |  |  |
| ca_s4_t1_wav018 | 16 | T1 | le=0 lambda_wav_plain=0.18 | 35.659 / 0.9394 |  |  |  |
| ca_s4_t1_wf005 | 16 | T1 | le=0 --edge_tv_wfloor 0.05 | 35.581 / 0.9382 |  |  |  |
| ca_s4_t1_wf020 | 16 | T1 | le=0 --edge_tv_wfloor 0.20 | 35.599 / 0.9376 |  |  |  |
| ca_s_fl_edge0 | 16 | FLAIR | le=0 |  | 34.904 / 0.9255 |  |  |
| ca_s_t1_edge0 | 16 | T1 | le=0 | 35.599 / 0.9381 |  |  |  |
| wd_af16_s_fl_p0003 | 16 | FLAIR | le=0 --lambda_ptv 0.003 |  | 34.943 / 0.9242 |  |  |
| wd_af16_s_t1_p0003 | 16 | T1 | le=0 --lambda_ptv 0.003 | 35.684 / 0.9371 |  |  |  |
| wd_af16_s_t1_p001w18 | 16 | T1 | le=0 --lambda_ptv 0.01 lambda_wav_plain=0.18 | 35.744 / 0.9394 |  |  |  |
| wd_v_af16_s_fl | 16 | FLAIR | le=0 --lambda_ptv 0.01 |  | 34.945 / 0.9249 |  |  |
| wd_v_af16_s_t1 | 16 | T1 | le=0 --lambda_ptv 0.01 lambda_wav_plain=0.18 | 35.744 / 0.9394 |  |  |  |
| wd_af20_j_dg_e002 | 20 | joint | le=0.002 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 | 34.861 / 0.9290 | 34.236 / 0.9147 | -0.002 | +0.057 |
| wd_af20_j_dgpw | 20 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 | 34.966 / 0.9306 | 34.246 / 0.9138 | +0.103 | +0.068 |
| wd_af20_j_dgpw_e002 | 20 | joint | le=0.002 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 | 34.958 / 0.9300 | 34.245 / 0.9132 | +0.095 | +0.067 |
| wd_af20_j_dgpw_pg025 | 20 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 | 34.975 / 0.9306 | 34.247 / 0.9135 | +0.112 | +0.068 |
| wd_af20_j_dgpw_pg100 | 20 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=1.0 | 34.979 / 0.9306 | 34.243 / 0.9133 | +0.116 | +0.065 |
| wd_af20_j_e002 | 20 | joint | le=0.002 | 34.843 / 0.9286 | 34.191 / 0.9140 | -0.020 | +0.013 |
| wd_af20_j_e003 | 20 | joint | le=0.003 | 34.815 / 0.9276 | 34.165 / 0.9129 | -0.048 | -0.014 |
| wd_v_af20_j | 20 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 | 34.975 / 0.9306 | 34.247 / 0.9135 | +0.112 | +0.068 |
| wd_af20_s_fl_e0_r | 20 | FLAIR | le=0 |  | 34.150 / 0.9135 |  |  |
| wd_af20_s_fl_p001 | 20 | FLAIR | le=0 --lambda_ptv 0.01 |  | 34.178 / 0.9124 |  |  |
| wd_af20_s_t1_e0 | 20 | T1 | le=0 | 34.715 / 0.9271 |  |  |  |
| wd_af20_s_t1_p001w18 | 20 | T1 | le=0 --lambda_ptv 0.01 lambda_wav_plain=0.18 | 34.863 / 0.9285 |  |  |  |
| wd_v_af20_s_fl | 20 | FLAIR | le=0 --lambda_ptv 0.01 |  | 34.178 / 0.9124 |  |  |
| wd_v_af20_s_t1 | 20 | T1 | le=0 --lambda_ptv 0.01 lambda_wav_plain=0.18 | 34.863 / 0.9285 |  |  |  |
| wd_af24_j_dg_e002 | 24 | joint | le=0.002 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 | 33.798 / 0.9131 | 33.250 / 0.8973 | +0.018 | +0.100 |
| wd_af24_j_dgpw | 24 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 | 33.890 / 0.9148 | 33.240 / 0.8954 | +0.109 | +0.091 |
| wd_af24_j_dgpw_e002 | 24 | joint | le=0.002 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 | 33.882 / 0.9140 | 33.244 / 0.8948 | +0.101 | +0.095 |
| wd_af24_j_dgpw_pg025 | 24 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 | 33.897 / 0.9148 | 33.239 / 0.8950 | +0.117 | +0.089 |
| wd_af24_j_dgpw_pg100 | 24 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=1.0 | 33.899 / 0.9148 | 33.234 / 0.8947 | +0.119 | +0.084 |
| wd_af24_j_e001_r | 24 | joint | le=0.001 | 33.766 / 0.9131 | 33.182 / 0.8966 | -0.014 | +0.032 |
| wd_af24_j_e002 | 24 | joint | le=0.002 | 33.781 / 0.9128 | 33.202 / 0.8964 | +0.000 | +0.052 |
| wd_af24_j_e003 | 24 | joint | le=0.003 | 33.758 / 0.9115 | 33.182 / 0.8951 | -0.022 | +0.033 |
| wd_v_af24_j | 24 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 | 33.897 / 0.9148 | 33.239 / 0.8950 | +0.117 | +0.090 |
| wd_af24_s_fl_e0_r | 24 | FLAIR | le=0 |  | 33.137 / 0.8954 |  |  |
| wd_af24_s_fl_p001 | 24 | FLAIR | le=0 --lambda_ptv 0.01 |  | 33.150 / 0.8935 |  |  |
| wd_af24_s_t1_e0_r | 24 | T1 | le=0 | 33.641 / 0.9107 |  |  |  |
| wd_af24_s_t1_p001w18 | 24 | T1 | le=0 --lambda_ptv 0.01 lambda_wav_plain=0.18 | 33.781 / 0.9122 |  |  |  |
| wd_v_af24_s_fl | 24 | FLAIR | le=0 --lambda_ptv 0.01 |  | 33.149 / 0.8935 |  |  |
| wd_v_af24_s_t1 | 24 | T1 | le=0 --lambda_ptv 0.01 lambda_wav_plain=0.18 | 33.781 / 0.9122 |  |  |  |

probe runs (oracle partners / guides, asymmetric probe masks): raw head metrics, no margin -- see the ceiling record
| exp | arm | knobs | T1 head PSNR / SSIM | FLAIR head PSNR / SSIM |
|---|---|---|---|---|
| wd_a_j_t16_fl32 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 configs/t1flair_v3_af16_jyz_asym_t16_fl32.yaml | 35.828 / 0.9408 | 32.766 / 0.8856 |
| wd_a_j_t16_fl64 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 configs/t1flair_v3_af16_jyz_asym_t16_fl64.yaml | 35.808 / 0.9404 | 31.039 / 0.8393 |
| wd_a_j_t8_fl32 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 configs/t1flair_v3_af16_jyz_asym_t8_fl32.yaml | 38.736 / 0.9660 | 32.784 / 0.8864 |
| wd_a_j_tfull_fl32_s | joint | le=0.003 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 --lambda_dtv 0.03 edge_tv_guide_alpha=1.0 ptv_guide_alpha=1.0 configs/t1flair_v3_af16_jyz_asym_tfull_fl32.yaml | 154.205 / 1.0000 | 32.880 / 0.8857 |
| wd_a_j_tfull_fl32 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 configs/t1flair_v3_af16_jyz_asym_tfull_fl32.yaml | 154.205 / 1.0000 | 32.802 / 0.8860 |
| wd_a_j_tfull_fl64 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 configs/t1flair_v3_af16_jyz_asym_tfull_fl64.yaml | 154.205 / 1.0000 | 31.068 / 0.8398 |
| wd_a_s_fl32 | FLAIR | le=0 --lambda_ptv 0.01 configs/t1flair_v3_af16_jyz_asym_singleFLAIR32.yaml |  | 32.634 / 0.8825 |
| wd_a_s_fl64 | FLAIR | le=0 --lambda_ptv 0.01 configs/t1flair_v3_af16_jyz_asym_singleFLAIR64.yaml |  | 30.806 / 0.8307 |
| wd_o_ffull_j | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 configs/t1flair_v3_af16_jyz_oracle_flairfull.yaml | 35.831 / 0.9409 | 154.867 / 1.0000 |
| wd_o_ffull_js | joint | le=0.003 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 --lambda_dtv 0.03 edge_tv_guide_alpha=1.0 ptv_guide_alpha=1.0 configs/t1flair_v3_af16_jyz_oracle_flairfull.yaml | 35.816 / 0.9398 | 154.867 / 1.0000 |
| wd_o_s_fl_dtvp010_gt1 | FLAIR | le=0 --lambda_ptv 0.01 guide_path_plain=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/recon/gt_T1_complex.npy guide_file_alpha=0 lambda_dtv_proj=0.10 |  | 34.284 / 0.9118 |
| wd_o_s_fl_gself | FLAIR | le=0 --lambda_ptv 0.01 guide_path_plain=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/recon/gt_FLAIR_complex.npy |  | 35.048 / 0.9267 |
| wd_o_s_fl_gt1 | FLAIR | le=0 --lambda_ptv 0.01 guide_path_plain=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/recon/gt_T1_complex.npy |  | 34.966 / 0.9253 |
| wd_o_s_t1_dtvp005_gfl | T1 | le=0 --lambda_ptv 0.01 lambda_wav_plain=0.18 guide_path_plain=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/recon/gt_FLAIR_complex.npy guide_file_alpha=0 lambda_dtv_proj=0.05 | 35.413 / 0.9342 |  |
| wd_o_s_t1_dtvp015_gfl | T1 | le=0 --lambda_ptv 0.01 lambda_wav_plain=0.18 guide_path_plain=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/recon/gt_FLAIR_complex.npy guide_file_alpha=0 lambda_dtv_proj=0.15 | 34.109 / 0.9156 |  |
| wd_o_s_t1_gfl | T1 | le=0 --lambda_ptv 0.01 lambda_wav_plain=0.18 guide_path_plain=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/recon/gt_FLAIR_complex.npy | 35.780 / 0.9401 |  |
| wd_o_s_t1_gself_geo03 | T1 | le=0 --lambda_ptv 0.01 lambda_wav_plain=0.18 guide_path_plain=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/recon/gt_T1_complex.npy --lambda_geo 0.3 | 35.598 / 0.9345 |  |
| wd_o_s_t1_gself | T1 | le=0 --lambda_ptv 0.01 lambda_wav_plain=0.18 guide_path_plain=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/recon/gt_T1_complex.npy | 35.872 / 0.9412 |  |
| wd_o_t1full_j | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 configs/t1flair_v3_af16_jyz_oracle_t1full.yaml | 154.205 / 1.0000 | 35.006 / 0.9253 |
| wd_o_t1full_js | joint | le=0.003 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 --lambda_dtv 0.03 edge_tv_guide_alpha=1.0 ptv_guide_alpha=1.0 configs/t1flair_v3_af16_jyz_oracle_t1full.yaml | 154.205 / 1.0000 | 35.010 / 0.9240 |
| wd_va_j_t16_fl32 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 configs/t1flair_v3_af16_jyz_asym_t16_fl32.yaml | 35.828 / 0.9408 | 32.766 / 0.8856 |
| wd_va_j_t16_fl64 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 configs/t1flair_v3_af16_jyz_asym_t16_fl64.yaml | 35.807 / 0.9404 | 31.039 / 0.8393 |
| wd_va_j_tfull_fl32 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 configs/t1flair_v3_af16_jyz_asym_tfull_fl32.yaml | 154.205 / 1.0000 | 32.802 / 0.8860 |
| wd_va_j_tfull_fl64 | joint | le=0.001 --lambda_dtv 0.01 edge_tv_guide_alpha=0.25 --lambda_ptv 0.01 lambda_wav_plain=0.18 ptv_guide_alpha=0.25 configs/t1flair_v3_af16_jyz_asym_tfull_fl64.yaml | 154.205 / 1.0000 | 31.068 / 0.8398 |
| wd_va_s_fl32 | FLAIR | le=0 --lambda_ptv 0.01 configs/t1flair_v3_af16_jyz_asym_singleFLAIR32.yaml |  | 32.634 / 0.8825 |
| wd_va_s_fl64 | FLAIR | le=0 --lambda_ptv 0.01 configs/t1flair_v3_af16_jyz_asym_singleFLAIR64.yaml |  | 30.806 / 0.8307 |

GS-single = best single run: {'AF12 FLAIR': (36.343, 'wd_af12_p14_s_fl_e0b'), 'AF12 T1': (36.942, 'wd_v_af12_s_t1'), 'AF16 FLAIR': (34.945, 'ca_s4_fl_ptv001'), 'AF16 T1': (35.744, 'wd_af16_s_t1_p001w18'), 'AF20 FLAIR': (34.178, 'wd_v_af20_s_fl'), 'AF20 T1': (34.863, 'wd_af20_s_t1_p001w18'), 'AF24 FLAIR': (33.15, 'wd_af24_s_fl_p001'), 'AF24 T1': (33.781, 'wd_v_af24_s_t1')}
