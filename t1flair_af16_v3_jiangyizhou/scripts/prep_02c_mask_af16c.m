% prep_02c_mask_af16c.m -- phase-3 acquisition probe, step 1 (worktree mcgs_af16_wt,
% docs/superpowers/specs/2026-09-12-af16-phase3-nonlocal-design.md, section 5): a pool of fresh
% vd-Poisson draws on the (ky,kz) = (256,176) PE plane. Nothing is chosen here:
% scripts/prep_04c_export_af16c.py picks the draw whose count inside FLAIR's acquired support
% (elliptical vendor shutter, 34139 of 45056 points) is closest to 2966 from above and trims it
% to exactly 2966 -> data/mask_vdpois_256x176_af16c_flair.mat, data/mask_af16c_flair_3d.h5.
%
% Same house generator and convention as prep_02_mask.m: accely = 4, calib = 24, ellipse = 0,
% pp = 1 (/data0/congcong/code/ESPIRiT/utils/gen_vdpois_masks.m). accelx is swept from the
% AF12-like value 3.0 named in the spec up to 3.9: at accelx 3.0 a draw puts about 3500 points
% inside the shutter (the AF12 mask: 3505), 540 above the budget, so a draw at 3.0 would lose
% its whole outer ring to the trim; the higher values land just above 2966 and the trim stays small.
% vdPoisMex reseeds from time(0) at ONE-SECOND resolution: two calls inside the same second return
% the same pattern (verified 2026-09-12), hence the pause before every draw.
addpath('/data0/congcong/code/ESPIRiT/utils');
outdir = '/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou/data/probe_c_draws';
if ~exist(outdir, 'dir'), mkdir(outdir); end
accely = 4; calib = 24; ellipse = 0; pp = 1;
accelx_list = 3.0:0.1:3.9;
ndraw = 5;                                % draws per accelx value
k = 0;
for accelx = accelx_list
    for d = 1:ndraw
        k = k + 1;
        pause(1.1);                       % fresh time(0) seed for every call
        mask = vdPoisMex(256, 176, 256, 176, accelx, accely, calib, ellipse, pp);
        mask = single(mask ~= 0);
        n = nnz(mask); af_eff = numel(mask)/n;
        fn = fullfile(outdir, sprintf('draw_%02d.mat', k));
        save(fn, 'mask', 'accelx', 'accely', 'calib', 'ellipse', 'pp', '-v7');
        fprintf('draw %02d accelx %.2f accely %d calib %d nnz %d AF_eff %.4f samp %.4f -> %s\n', ...
                k, accelx, accely, calib, n, af_eff, mean(mask(:)), fn);
    end
end
exit
