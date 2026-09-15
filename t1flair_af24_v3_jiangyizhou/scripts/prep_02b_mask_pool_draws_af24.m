% prep_02b_mask_pool_draws_af24.m -- a second pool of house-convention vd-Poisson draws, for three more AF24 masks
% (b, c, d) with the registered mask's rule: exactly 1877 of 45056 points (T1 at 24x), untrimmed (2026-09-15).
% Mask-draw variance of the MC-GS margin, as at AF20. Same call as prep_02_mask_pool_af24.m; accelx is centred on the
% accelx* = 6.8456 that the first pool's stage-A fit gave (logs/prep_02_mask_pool_af24.log), cycling through
% accelx* + (-0.030:0.005:0.030). Draws until three have exactly 1877 points, at most 480; the stopping rule looks at
% the point count only. vdPoisMex reseeds from time(0) at one-second resolution, hence pause(1.1) before every call.
% prep_03b_pick_export_draws_af24.py chooses. An existing pool is never added to or overwritten.
% Run: matlab -batch "run('<this file>')"
addpath('/data0/congcong/code/ESPIRiT/utils');
R = '/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF24_v3_jiangyizhou';
outdir = fullfile(R, 'data', 'mask_pool_af24_draws');
if exist(outdir, 'dir') && ~isempty(dir(fullfile(outdir, 'draw_*.mat')))
    error('a pool already exists in %s; refusing to add to or overwrite it', outdir);
end
if ~exist(outdir, 'dir'), mkdir(outdir); end
TARGET = 1877; NEED = 3; MAXDRAW = 480;
accely = 4; calib = 24; ellipse = 0; pp = 1; stage = 'D';
accel_star = 6.8456;
offs = -0.030:0.005:0.030;
hits = 0; k = 0;
while hits < NEED && k < MAXDRAW
    k = k + 1;
    accelx = accel_star + offs(mod(k - 1, numel(offs)) + 1);
    pause(1.1);                                   % fresh time(0) seed for every call
    mask = single(vdPoisMex(256, 176, 256, 176, accelx, accely, calib, ellipse, pp) ~= 0);
    n = nnz(mask);
    save(fullfile(outdir, sprintf('draw_%03d.mat', k)), 'mask', 'accelx', 'accely', 'calib', 'ellipse', 'pp', 'stage', '-v7');
    hit = '';
    if n == TARGET, hits = hits + 1; hit = sprintf('  <- exactly the target (%d)', hits); end
    fprintf('draw %03d accelx %.4f nnz %d AF_T1 %.3f%s\n', k, accelx, n, numel(mask) / n, hit);
end
fprintf('POOL_DRAWS_DONE %d draws, %d exact\n', k, hits);
